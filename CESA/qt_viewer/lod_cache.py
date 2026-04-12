"""Level-of-Detail multi-resolution cache for long recordings.

Pre-computes coarse representations (min-max envelopes) at multiple
zoom levels so that zoomed-out views can be rendered instantly from
cache instead of re-processing gigabytes of raw samples.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .downsampler import downsample_minmax

logger = logging.getLogger(__name__)

# Resolution tiers (target points per screen width at each LOD)
LOD_TIERS: List[int] = [500, 2000, 8000, 32000]


class _ChannelLOD:
    """Multi-resolution cache for a single channel."""

    def __init__(self, data: np.ndarray, sfreq: float) -> None:
        self._raw = data
        self._sfreq = sfreq
        self._tiers: Dict[int, np.ndarray] = {}
        self._build()

    def _build(self) -> None:
        n = len(self._raw)
        for tier in LOD_TIERS:
            if tier >= n:
                self._tiers[tier] = self._raw.copy()
            else:
                _idx, vals = downsample_minmax(self._raw, tier)
                self._tiers[tier] = vals

    def get(self, target_points: int) -> np.ndarray:
        """Return the best LOD tier for *target_points*."""
        best_tier = LOD_TIERS[-1]
        for tier in LOD_TIERS:
            if tier >= target_points:
                best_tier = tier
                break
        return self._tiers.get(best_tier, self._raw)

    @property
    def raw(self) -> np.ndarray:
        return self._raw

    @property
    def sfreq(self) -> float:
        return self._sfreq


class LODCache:
    """Multi-channel Level-of-Detail cache.

    Usage::

        cache = LODCache()
        cache.set_channel("EEG Fp1", data, 256.0)
        # When rendering:
        lod_data = cache.get_lod("EEG Fp1", target_points=4000)
    """

    def __init__(self) -> None:
        self._channels: Dict[str, _ChannelLOD] = {}

    def set_channel(self, name: str, data: np.ndarray, sfreq: float) -> None:
        self._channels[name] = _ChannelLOD(data, sfreq)

    def get_lod(self, name: str, target_points: int = 4000) -> Optional[np.ndarray]:
        ch = self._channels.get(name)
        if ch is None:
            return None
        return ch.get(target_points)

    def get_raw(self, name: str) -> Optional[np.ndarray]:
        ch = self._channels.get(name)
        return ch.raw if ch else None

    def get_sfreq(self, name: str) -> Optional[float]:
        ch = self._channels.get(name)
        return ch.sfreq if ch else None

    def get_segment(
        self,
        name: str,
        start_s: float,
        duration_s: float,
        target_points: int = 4000,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (time_array, data_array) for a segment at the best LOD."""
        ch = self._channels.get(name)
        if ch is None:
            return None

        total_samples = len(ch.raw)
        total_dur = total_samples / ch.sfreq if ch.sfreq > 0 else 0
        if total_dur <= 0:
            return None

        # Determine which LOD tier to use based on zoom level
        full_points = int(total_samples * (duration_s / total_dur))
        if full_points <= target_points:
            # Raw data is fine
            i0 = max(0, int(start_s * ch.sfreq))
            i1 = min(total_samples, int((start_s + duration_s) * ch.sfreq))
            segment = ch.raw[i0:i1]
            times = np.linspace(start_s, start_s + duration_s, len(segment))
            return times, segment

        # Use LOD
        lod = ch.get(target_points)
        lod_ratio = len(lod) / total_samples
        li0 = max(0, int(start_s * ch.sfreq * lod_ratio))
        li1 = min(len(lod), int((start_s + duration_s) * ch.sfreq * lod_ratio))
        segment = lod[li0:li1]
        times = np.linspace(start_s, start_s + duration_s, len(segment))
        return times, segment

    def has_channel(self, name: str) -> bool:
        return name in self._channels

    def channel_names(self) -> List[str]:
        return list(self._channels.keys())

    def clear(self) -> None:
        self._channels.clear()

    @property
    def memory_mb(self) -> float:
        total = 0
        for ch in self._channels.values():
            total += ch.raw.nbytes
            for arr in ch._tiers.values():
                total += arr.nbytes
        return total / (1024 * 1024)
