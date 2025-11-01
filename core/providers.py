"""Signal provider interfaces and implementations for multiscale PSG data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt

from .store import MultiscaleStore, open_multiscale


EnvelopeArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class EnvelopeBatch:
    """Container returned by signal providers for a window request."""

    signals: Dict[str, EnvelopeArray]
    level_bin_size: int
    bytes_read: int


@runtime_checkable
class SignalProvider(Protocol):
    """Abstract port used by the UI bridge to fetch PSG envelopes."""

    def get_channel_names(self) -> Sequence[str]:
        """Return the ordered list of channel identifiers."""

    def get_sampling_frequency(self) -> float:
        """Return the native sampling frequency in Hz."""

    def get_total_duration_seconds(self) -> float:
        """Return the total duration of the recording in seconds."""

    def get_envelopes(
        self,
        t0: float,
        t1: float,
        width_px: int,
        channels: Sequence[str],
        *,
        gains: Mapping[str, float] | None = None,
        offsets: Mapping[str, float] | None = None,
    ) -> EnvelopeBatch:
        """Fetch min/max envelopes for the given time window.

        Implementations may ignore the ``gains``/``offsets`` parameters if they only
        expose raw envelopes. The bridge guarantees those adjustments are applied once
        before the UI receives the buffers.
        """

    def get_raw_segment(self, channel: str, start_idx: int, stop_idx: int) -> EnvelopeArray:
        """Optional fallback to access raw samples when needed."""


class PrecomputedProvider(SignalProvider):
    """Adapter that reads multiscale Zarr min/max pyramids."""

    def __init__(self, root_path: str | Path) -> None:
        path = Path(root_path)
        if not path.exists():
            raise FileNotFoundError(f"Multiscale store not found: {path}")

        self._store: MultiscaleStore = open_multiscale(path)
        meta = self._store.metadata

        self._fs = meta.sampling_frequency
        self._channel_names = list(meta.channel_names)
        self._duration_seconds = meta.duration_seconds
        self._total_samples = meta.total_samples

        self._channel_indices: Dict[str, int] = {
            name: idx for idx, name in enumerate(self._channel_names)
        }
        self._normalized_lookup: Dict[str, str] = {}
        for name in self._channel_names:
            key = self._normalize_channel_key(name)
            if key and key not in self._normalized_lookup:
                self._normalized_lookup[key] = name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_channel_names(self) -> Sequence[str]:
        return self._channel_names

    def get_sampling_frequency(self) -> float:
        return self._fs

    def get_total_duration_seconds(self) -> float:
        return self._duration_seconds

    # ------------------------------------------------------------------
    def get_envelopes(
        self,
        t0: float,
        t1: float,
        width_px: int,
        channels: Sequence[str],
        *,
        gains: Mapping[str, float] | None = None,
        offsets: Mapping[str, float] | None = None,
    ) -> EnvelopeBatch:
        del gains, offsets  # handled by bridge layer; keep signature compatibility
        width_px = max(1, int(width_px))

        start_sample, stop_sample = self._store.window_to_samples(t0, t1)
        samples_per_pixel = (stop_sample - start_sample) / width_px
        level = self._store.select_level(samples_per_pixel)
        start_bin, stop_bin = self._store.samples_to_bins(level, start_sample, stop_sample)

        signals: Dict[str, EnvelopeArray] = {}
        bytes_read = 0
        arr = level.dataset

        for channel_key in channels:
            chan_name, chan_index = self._resolve_channel(channel_key)
            window = arr.oindex[chan_index, start_bin:stop_bin, :]
            window = np.asarray(window, dtype=np.float32, order="C")
            bytes_read += int(window.nbytes)
            signals[chan_name] = window.reshape(-1)

        return EnvelopeBatch(
            signals=signals,
            level_bin_size=level.bin_size,
            bytes_read=bytes_read,
        )

    def get_raw_segment(self, channel: str, start_idx: int, stop_idx: int) -> EnvelopeArray:
        raise NotImplementedError("Raw samples are not stored in the multiscale pyramid")

    def _resolve_channel(self, channel: str | int) -> tuple[str, int]:
        if isinstance(channel, int):
            if channel < 0 or channel >= len(self._channel_names):
                raise IndexError(f"Channel index out of range: {channel}")
            return self._channel_names[channel], channel

        channel = str(channel)
        try:
            return channel, self._channel_indices[channel]
        except KeyError as exc:
            normalized = self._normalize_channel_key(channel)
            if normalized and normalized in self._normalized_lookup:
                resolved = self._normalized_lookup[normalized]
                return resolved, self._channel_indices[resolved]

            if channel.isdigit():
                idx = int(channel)
                if 0 <= idx < len(self._channel_names):
                    resolved = self._channel_names[idx]
                    return resolved, idx

            raise KeyError(f"Unknown channel: {channel}") from exc

    @staticmethod
    def _normalize_channel_key(name: str) -> str:
        name = (name or "").upper()
        if not name:
            return ""
        return "".join(ch for ch in name if ch.isalnum())


