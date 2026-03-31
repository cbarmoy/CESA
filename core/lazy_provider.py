"""Lazy signal provider that computes envelopes on demand from raw PSG data."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

try:  # pragma: no cover - optional dependency for runtime
    from mne.io import BaseRaw  # type: ignore
    import mne  # type: ignore
except Exception as exc:  # pragma: no cover - defer failure until use
    BaseRaw = object  # type: ignore
    mne = None  # type: ignore

from .providers import EnvelopeBatch, SignalProvider
from .raw_loader import open_raw_file
from .telemetry import telemetry

EnvelopeArray = npt.NDArray[np.float32]


def _round_power_of_two(value: float) -> int:
    value = max(float(value), 1.0)
    exp = math.ceil(math.log2(value))
    return 1 << int(exp)


def _sanitize(signal: np.ndarray) -> np.ndarray:
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


class _LRUCache:
    """Byte-capped LRU cache for numpy arrays."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(1, int(max_bytes))
        self._entries: "OrderedDict[Tuple[int, int, int, int], np.ndarray]" = OrderedDict()
        self._total_bytes = 0

    def get(self, key: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        array = self._entries.get(key)
        if array is not None:
            self._entries.move_to_end(key)
            return array
        return None

    def put(self, key: Tuple[int, int, int, int], array: np.ndarray) -> None:
        array = np.asarray(array, dtype=np.float32)
        if key in self._entries:
            previous = self._entries.pop(key)
            self._total_bytes -= previous.nbytes
        stored = array.copy()
        self._entries[key] = stored
        self._total_bytes += stored.nbytes
        self._evict()

    def _evict(self) -> None:
        while self._total_bytes > self._max_bytes and self._entries:
            _, oldest = self._entries.popitem(last=False)
            self._total_bytes -= oldest.nbytes


@dataclass(frozen=True)
class _ChannelMeta:
    name: str
    index: int


class LazyProvider(SignalProvider):
    """Compute min/max envelopes lazily from an MNE Raw object."""

    def __init__(
        self,
        source: str | Path | BaseRaw,
        *,
        cache_bytes: int = 384 * 1024 * 1024,
        max_bin_size: int = 65536,
    ) -> None:
        if mne is None:  # pragma: no cover - defensive
            raise RuntimeError("mne is required for LazyProvider")

        self._raw: BaseRaw = self._resolve_raw(source)
        self._fs = float(self._raw.info.get("sfreq", 0.0))
        if self._fs <= 0.0:
            raise RuntimeError("Sampling frequency must be positive for LazyProvider")

        self._channel_names: Tuple[str, ...] = tuple(self._raw.ch_names)
        self._channel_map: Dict[str, _ChannelMeta] = {
            name: _ChannelMeta(name=name, index=idx)
            for idx, name in enumerate(self._channel_names)
        }
        self._total_samples = int(getattr(self._raw, "n_times", 0))
        self._cache = _LRUCache(cache_bytes)
        self._max_bin_size = max(1, int(max_bin_size))
        self._chunk_len = 65536
        self._scale_uv = 1e6

    # ------------------------------------------------------------------
    # SignalProvider API
    # ------------------------------------------------------------------
    def get_channel_names(self) -> Sequence[str]:
        return self._channel_names

    def get_sampling_frequency(self) -> float:
        return self._fs

    def get_total_duration_seconds(self) -> float:
        return self._total_samples / self._fs if self._fs > 0 else 0.0

    def get_envelopes(
        self,
        t0: float,
        t1: float,
        width_px: int,
        channels: Sequence[str],
        *,
        gains: Optional[Mapping[str, float]] = None,
        offsets: Optional[Mapping[str, float]] = None,
    ) -> EnvelopeBatch:
        del gains, offsets  # Not applied here; handled by DataBridge.

        width_px = max(1, int(width_px))
        start_sample = max(0, int(math.floor(t0 * self._fs)))
        stop_sample = min(self._total_samples, int(math.ceil(t1 * self._fs)))
        if stop_sample <= start_sample:
            stop_sample = min(self._total_samples, start_sample + 1)
        n_samples = max(1, stop_sample - start_sample)

        samples_per_pixel = n_samples / width_px
        bin_size = min(self._max_bin_size, _round_power_of_two(samples_per_pixel))

        telemetry.mark_mode("lazy", source="LazyProvider.get_envelopes")
        base_sample = telemetry.new_sample(
            {
                "start_s": float(t0),
                "duration_s": float(max(0.0, t1 - t0)),
                "viewport_px": int(width_px),
                "spp_screen": float(samples_per_pixel),
                "level_k": int(bin_size),
            }
        )

        signals: Dict[str, EnvelopeArray] = {}
        total_bytes = 0
        chunk_count = math.ceil(n_samples / self._chunk_len)

        for channel in channels:
            meta = self._resolve_channel(channel)
            cache_key = (meta.index, start_sample, stop_sample, bin_size)
            cached = self._cache.get(cache_key)

            sample_row = telemetry.new_sample(base_sample)
            sample_row["channel"] = meta.name
            sample_row["chunks_read"] = int(chunk_count)

            if cached is not None:
                envelope = cached.copy()
                sample_row["cache_hit"] = True
                sample_row["bytes_read"] = 0
            else:
                sample_row["cache_hit"] = False
                with telemetry.measure(sample_row, "io_ms"):
                    data = self._raw.get_data(
                        picks=[meta.index],
                        start=start_sample,
                        stop=stop_sample,
                        reject_by_annotation=False,
                    )[0]

                data = _sanitize(np.asarray(data, dtype=np.float32)) * self._scale_uv
                total_bytes += int(data.nbytes)
                envelope = self._compute_envelope(data, bin_size)
                self._cache.put(cache_key, envelope)
                sample_row["bytes_read"] = int(data.nbytes)

            telemetry.commit(sample_row)
            signals[meta.name] = envelope

        return EnvelopeBatch(
            signals=signals,
            level_bin_size=bin_size,
            bytes_read=total_bytes,
        )

    def get_raw_segment(self, channel: str, start_idx: int, stop_idx: int) -> EnvelopeArray:
        meta = self._resolve_channel(channel)
        data = self._raw.get_data(
            picks=[meta.index],
            start=max(0, int(start_idx)),
            stop=max(int(start_idx), int(stop_idx)),
            reject_by_annotation=False,
        )[0]
        data = _sanitize(np.asarray(data, dtype=np.float32)) * self._scale_uv
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_channel(self, channel: str | int) -> _ChannelMeta:
        if isinstance(channel, int):
            if channel < 0 or channel >= len(self._channel_names):
                raise KeyError(f"Channel index out of range: {channel}")
            name = self._channel_names[channel]
            return self._channel_map[name]

        key = str(channel)
        meta = self._channel_map.get(key)
        if meta is None:
            raise KeyError(f"Unknown channel: {channel}")
        return meta

    @staticmethod
    def _resolve_raw(source: str | Path | BaseRaw) -> BaseRaw:
        if isinstance(source, BaseRaw):  # type: ignore[isinstance-error]
            return source
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        return open_raw_file(path, preload=False, verbose=False)

    @staticmethod
    def _compute_envelope(data: np.ndarray, bin_size: int) -> np.ndarray:
        if bin_size <= 1:
            stacked = np.empty((data.size, 2), dtype=np.float32)
            stacked[:, 0] = data
            stacked[:, 1] = data
            return stacked.reshape(-1)

        remainder = data.size % bin_size
        if remainder:
            pad_value = data[-1]
            pad = np.full(bin_size - remainder, pad_value, dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        reshaped = data.reshape(-1, bin_size)
        mins = reshaped.min(axis=1)
        maxs = reshaped.max(axis=1)
        stacked = np.empty((mins.size, 2), dtype=np.float32)
        stacked[:, 0] = mins
        stacked[:, 1] = maxs
        return stacked.reshape(-1)


__all__ = ["LazyProvider"]


