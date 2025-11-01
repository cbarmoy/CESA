"""Compatibility bridge between the UI and signal providers."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .providers import EnvelopeBatch, SignalProvider


@dataclass(frozen=True)
class WindowResult:
    """Structured response for a window request."""

    signals: Dict[str, tuple[np.ndarray, float]]
    level_bin_size: int
    bytes_read: int
    n_bins: int


class DataBridge:
    """Adapts a :class:`SignalProvider` to the legacy PSG UI expectations."""

    def __init__(self, provider: SignalProvider, executor: Optional[Executor] = None) -> None:
        self._provider = provider
        self._executor = executor
        self._fs = float(provider.get_sampling_frequency())

    # ------------------------------------------------------------------
    # Metadata passthrough
    # ------------------------------------------------------------------
    def get_channel_names(self) -> Sequence[str]:
        return self._provider.get_channel_names()

    def get_sampling_frequency(self) -> float:
        return self._fs

    def get_total_duration_seconds(self) -> float:
        return self._provider.get_total_duration_seconds()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def submit_window_request(
        self,
        t0: float,
        duration: float,
        width_px: int,
        *,
        channels: Sequence[str],
        gains: Mapping[str, float] | None = None,
        offsets: Mapping[str, float] | None = None,
    ) -> Future[WindowResult]:
        if self._executor is None:
            raise RuntimeError("No executor configured for asynchronous window requests")
        return self._executor.submit(
            self.get_signals_for_window,
            t0,
            duration,
            width_px,
            channels=channels,
            gains=gains,
            offsets=offsets,
        )

    def get_signals_for_window(
        self,
        t0: float,
        duration: float,
        width_px: int,
        *,
        channels: Sequence[str],
        gains: Mapping[str, float] | None = None,
        offsets: Mapping[str, float] | None = None,
    ) -> WindowResult:
        if duration <= 0.0:
            duration = max(1.0 / max(self._fs, 1.0), 1e-3)
        t1 = t0 + duration
        batch = self._provider.get_envelopes(
            t0,
            t1,
            width_px,
            channels,
            gains=gains,
            offsets=offsets,
        )

        signals: Dict[str, tuple[np.ndarray, float]] = {}
        window_duration = max(t1 - t0, 1e-3)

        for name, envelope in batch.signals.items():
            wave = _envelope_to_wave(envelope)
            wave = _apply_gain_offset(wave, gains, offsets, name)
            effective_fs = _effective_sampling_rate(wave.size, window_duration, self._fs)
            signals[name] = (wave, effective_fs)

        return WindowResult(
            signals=signals,
            level_bin_size=batch.level_bin_size,
            bytes_read=batch.bytes_read,
            n_bins=_count_bins(batch),
        )

    # Optional fallback to raw samples
    def get_raw_segment(self, channel: str, start_idx: int, stop_idx: int) -> np.ndarray:
        return self._provider.get_raw_segment(channel, start_idx, stop_idx)


def _envelope_to_wave(envelope: np.ndarray) -> np.ndarray:
    data = np.asarray(envelope, dtype=np.float32).reshape(-1)
    return data.copy()


def _apply_gain_offset(
    data: np.ndarray,
    gains: Mapping[str, float] | None,
    offsets: Mapping[str, float] | None,
    channel: str,
) -> np.ndarray:
    if gains is not None:
        gain = float(gains.get(channel, 1.0))
    else:
        gain = 1.0
    if offsets is not None:
        offset = float(offsets.get(channel, 0.0))
    else:
        offset = 0.0
    if gain != 1.0 or offset != 0.0:
        data = data * gain + offset
    return data


def _effective_sampling_rate(n_points: int, duration: float, fallback_fs: float) -> float:
    if n_points <= 1 or duration <= 0.0:
        return max(fallback_fs, 1.0)
    return max((n_points - 1) / duration, 1.0)


def _count_bins(batch: EnvelopeBatch) -> int:
    signals = batch.signals
    if not signals:
        return 0
    first = next(iter(signals.values()))
    return max(1, int(len(first) / 2))




