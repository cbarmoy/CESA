"""Helpers to open and navigate multiscale PSG Zarr stores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import zarr

# Magasins multiscale CESA : métadonnées Zarr v2 (compatibilité dossiers existants). zarr-python 3 gère ce format via zarr_format=2.
MULTISCALE_ZARR_FORMAT = 2


@dataclass(frozen=True)
class MultiscaleMetadata:
    """Metadata describing a multiscale PSG store."""

    sampling_frequency: float
    channel_names: Sequence[str]
    level_bin_sizes: Sequence[int]
    total_samples: int
    dtype: np.dtype

    @property
    def duration_seconds(self) -> float:
        if not self.sampling_frequency:
            return 0.0
        return self.total_samples / self.sampling_frequency


@dataclass(frozen=True)
class LevelDescriptor:
    """Descriptor for a single min/max pyramid level."""

    bin_size: int
    dataset: zarr.Array
    n_bins: int
    chunk_bins: int

    @property
    def name(self) -> str:
        return f"lvl{self.bin_size}"


class MultiscaleStore:
    """Convenience accessor around a multiscale PSG Zarr store."""

    def __init__(
        self,
        root_path: Path,
        group: zarr.Group,
        metadata: MultiscaleMetadata,
        levels: Iterable[LevelDescriptor],
    ) -> None:
        self.root_path = root_path
        self.group = group
        self.metadata = metadata
        self._levels: Dict[int, LevelDescriptor] = {
            level.bin_size: level for level in levels
        }
        if not self._levels:
            raise RuntimeError("Multiscale store contains no pyramid levels")

        self._sorted_bins: List[int] = sorted(self._levels)

    # ------------------------------------------------------------------
    # Level helpers
    # ------------------------------------------------------------------
    def get_level(self, bin_size: int) -> LevelDescriptor:
        return self._levels[bin_size]

    def available_levels(self) -> Sequence[int]:
        return self._sorted_bins

    def select_level(self, samples_per_pixel: float) -> LevelDescriptor:
        spp = max(samples_per_pixel, 1.0)
        threshold = 2 * spp
        candidates = [b for b in self._sorted_bins if b <= threshold]
        if candidates:
            bin_size = candidates[-1]
        else:
            bin_size = self._sorted_bins[0]
        return self._levels[bin_size]

    # ------------------------------------------------------------------
    # Time/sample/bin helpers
    # ------------------------------------------------------------------
    def clamp_samples(self, start_sample: int, stop_sample: int) -> tuple[int, int]:
        total = self.metadata.total_samples
        start = max(0, min(total, start_sample))
        stop = max(start, min(total, stop_sample))
        if stop == start:
            stop = min(total, start + 1)
        return start, stop

    def window_to_samples(self, t0: float, t1: float) -> tuple[int, int]:
        fs = self.metadata.sampling_frequency
        if t1 <= t0:
            t1 = t0 + max(1.0 / fs if fs else 1.0, 1e-3)
        start_sample = int(math.floor(t0 * fs))
        stop_sample = int(math.ceil(t1 * fs))
        return self.clamp_samples(start_sample, stop_sample)

    def samples_to_bins(self, level: LevelDescriptor, start_sample: int, stop_sample: int) -> tuple[int, int]:
        bin_size = level.bin_size
        start_bin = start_sample // bin_size
        stop_bin = min(level.n_bins, math.ceil(stop_sample / bin_size))
        if stop_bin <= start_bin:
            stop_bin = min(level.n_bins, start_bin + 1)
        return start_bin, stop_bin

    def window_to_bin_range(
        self, t0: float, t1: float, level: LevelDescriptor
    ) -> tuple[int, int, int, int]:
        start_sample, stop_sample = self.window_to_samples(t0, t1)
        start_bin, stop_bin = self.samples_to_bins(level, start_sample, stop_sample)
        return start_sample, stop_sample, start_bin, stop_bin


def open_multiscale(path: str | Path) -> MultiscaleStore:
    """Open a multiscale PSG store from disk."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Multiscale path does not exist: {root}")

    group = zarr.open_group(str(root), mode="r", zarr_format=MULTISCALE_ZARR_FORMAT)
    attrs = dict(group.attrs)

    try:
        fs = float(attrs["fs"])
        channel_names = list(attrs["channel_names"])
        levels = [int(v) for v in attrs["levels"]]
        total_samples = int(attrs["n_samples"])
        dtype = np.dtype(attrs.get("dtype", "float32"))
    except KeyError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Missing attribute in multiscale store: {exc}") from exc

    metadata = MultiscaleMetadata(
        sampling_frequency=fs,
        channel_names=channel_names,
        level_bin_sizes=levels,
        total_samples=total_samples,
        dtype=dtype,
    )

    level_descriptors: List[LevelDescriptor] = []
    for bin_size in levels:
        dataset = group.get(f"levels/lvl{bin_size}")
        if dataset is None:
            raise RuntimeError(f"Missing dataset for level lvl{bin_size}")
        if dataset.ndim != 3 or dataset.shape[2] != 2:
            raise ValueError(
                f"Level lvl{bin_size} must have shape (channels, bins, 2); got {dataset.shape}"
            )
        n_channels, n_bins, _ = dataset.shape
        if n_channels != len(channel_names):
            raise ValueError(
                "Channel dimension mismatch between metadata and level " f"lvl{bin_size}"
            )
        chunk_bins = dataset.chunks[1] if dataset.chunks else n_bins
        level_descriptors.append(
            LevelDescriptor(
                bin_size=bin_size,
                dataset=dataset,
                n_bins=n_bins,
                chunk_bins=chunk_bins,
            )
        )

    return MultiscaleStore(root, group, metadata, level_descriptors)




