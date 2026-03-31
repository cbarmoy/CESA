"""
Validation script to sanity-check the MSE pipeline on synthetic data.

Two stage segments are generated:
    - Wake (W): high irregularity (white noise) -> high entropy expected
    - N3: regular sine wave -> low entropy expected

We emulate the group-analysis pipeline:
    1. Build a dummy Raw-like object
    2. Create a scoring DataFrame with epoch start times
    3. Use `_extract_stage_data` to slice the signal per stage
    4. Run `_compute_mse_profile` to obtain per-scale entropies

Run:
    python scripts/validate_mse_pipeline.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CESA.group_analysis import (  # type: ignore
    GroupAnalysisConfig,
    _compute_mse_profile,
    _extract_stage_data,
)
from CESA.entropy import MultiscaleEntropyConfig  # type: ignore


class DummyRaw:
    """Minimal Raw-like object implementing the methods used by `_extract_stage_data`."""

    def __init__(self, data: np.ndarray, sfreq: float, ch_names: list[str]) -> None:
        self._data = data
        self.info = {"sfreq": sfreq}
        self.n_times = data.shape[1]
        self.ch_names = ch_names
        self.filenames = ["synthetic.edf"]

    def get_data(self, *, picks=None, start=None, stop=None, return_times=False):
        arr = self._data
        if picks is not None:
            arr = arr[picks]
        start = 0 if start is None else max(0, start)
        stop = self.n_times if stop is None else min(self.n_times, stop)
        sliced = arr[:, start:stop]
        if return_times:
            times = np.arange(start, stop, dtype=float) / float(self.info["sfreq"])
            return sliced, times
        return sliced

    def close(self) -> None:  # parity with real mne.Raw objects
        return


def _build_synthetic_raw(sfreq: float, epoch_s: float, stages: list[str]) -> DummyRaw:
    samples_per_epoch = int(round(epoch_s * sfreq))
    segments = []
    rng = np.random.default_rng(42)
    for stage in stages:
        if stage == "W":
            segments.append(rng.normal(loc=0.0, scale=1.0, size=samples_per_epoch))
        else:  # N3
            t = np.arange(samples_per_epoch) / sfreq
            segments.append(0.3 * np.sin(2 * math.pi * 1.0 * t))
    signal = np.concatenate(segments, axis=0)[None, :]
    return DummyRaw(signal.astype(np.float32), sfreq, ["C3"])


def main() -> None:
    sfreq = 128.0
    epoch_len = 30.0
    stage_sequence = ["W", "W", "N3", "N3", "W"]
    raw = _build_synthetic_raw(sfreq, epoch_len, stage_sequence)

    # Build scoring DataFrame replicating the provided sequence
    times = np.arange(len(stage_sequence), dtype=float) * epoch_len
    scoring = pd.DataFrame({"time": times, "stage": stage_sequence})

    mse_cfg = MultiscaleEntropyConfig(scales=range(1, 6), max_samples=None)
    ga_cfg = GroupAnalysisConfig(mse_config=mse_cfg, epoch_seconds=epoch_len)

    results = {}
    for stage in ["W", "N3"]:
        stage_data = _extract_stage_data(raw, scoring, stage, epoch_len, picks=None, channel_names=None)
        profile = _compute_mse_profile(stage_data, sfreq, ga_cfg, progress_label=f"synthetic-{stage}")
        mean_entropy = float(np.nanmean(list(profile.values())))
        results[stage] = {"profile": profile, "mean": mean_entropy}

    print("Synthetic MSE validation (wake should be > N3):")
    for stage, payload in results.items():
        print(f"- Stage {stage}: mean={payload['mean']:.4f}, scales={payload['profile']}")

    if results["W"]["mean"] > results["N3"]["mean"]:
        print("✔ Pipeline preserves expected ordering (Wake > N3).")
    else:
        print("✖ Unexpected ordering; investigate extraction/aggregation.")


if __name__ == "__main__":
    main()

