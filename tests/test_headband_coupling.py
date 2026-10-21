"""Unit tests for the headband path of the SO-spindle coupling pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from CESA.sleep_pipeline import headband_coupling as hb
from CESA.sleep_pipeline import so_spindle_coupling as sc
from tests.test_so_spindle_coupling import (
    DURATION_S,
    SFREQ,
    SO_PERIOD_S,
    synthetic_recording,
)

BLINK_AMPLITUDE_UV = 200.0
BLINK_DURATION_S = 0.2
#: Blinks sit between two SO cycles, far enough not to mask a real event.
BLINK_OFFSET_S = 5.0

#: A clinical PSG montage as exported by the MBSR-HTD recorder.
PSG_MONTAGE = [
    "C3", "C3 Impedance", "C3-M2", "C4-M1", "F3", "F3 Impedance", "F3-M2",
    "F4-M1", "O1-M2", "E1-M2", "E2-M1", "ECG", "Heart Rate", "SpO2 B-B",
    "M1", "M2", "Left Leg",
]


def with_blinks(
    signal: np.ndarray, starts: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Copy of *signal* with one blink between each pair of SO cycles.

    Returns the contaminated signal and the sample window of each blink.
    """
    out = np.asarray(signal, dtype=float).copy()
    width = int(BLINK_DURATION_S * SFREQ)
    shape = BLINK_AMPLITUDE_UV * np.hanning(width)
    windows: list[tuple[int, int]] = []
    for start in starts:
        i0 = int((start + BLINK_OFFSET_S) * SFREQ)
        if i0 + width >= out.size:
            continue
        out[i0:i0 + width] += shape
        windows.append((i0, i0 + width))
    return out, windows


class TestBlinkDetection:
    def test_detects_every_injected_blink(self):
        signal, starts = synthetic_recording(seed=1)
        contaminated, windows = with_blinks(signal, starts)
        blinks = hb.detect_blinks(contaminated, SFREQ)
        assert len(blinks) == len(windows)
        for blink in blinks:
            duration = (blink.end_idx - blink.start_idx) / SFREQ
            assert hb.BLINK_MIN_DURATION_S <= duration <= hb.BLINK_MAX_DURATION_S
            assert blink.peak_amplitude_uv >= hb.BLINK_MIN_AMPLITUDE_UV
            assert blink.start_idx <= blink.peak_idx < blink.end_idx

    def test_slow_oscillations_alone_are_not_blinks(self):
        signal, _ = synthetic_recording(seed=1)
        assert hb.detect_blinks(signal, SFREQ) == []

    def test_flat_signal_yields_nothing(self):
        assert hb.detect_blinks(np.zeros(int(60 * SFREQ)), SFREQ) == []

    def test_keep_mask_pads_both_sides(self):
        blink = hb.Blink(start_idx=500, end_idx=520, peak_idx=510, peak_amplitude_uv=200.0)
        keep = hb.blink_keep_mask(1000, SFREQ, [blink], padding_s=0.2)
        pad = int(0.2 * SFREQ)
        assert not keep[500 - pad:520 + pad].any()
        assert keep[: 500 - pad].all()
        assert keep[520 + pad:].all()

    def test_keep_mask_without_blinks_keeps_everything(self):
        assert hb.blink_keep_mask(100, SFREQ, []).all()


class TestBlinkRejectionInAnalysis:
    def test_blinks_do_not_create_slow_oscillations(self):
        signal, starts = synthetic_recording(seed=1)
        contaminated, windows = with_blinks(signal, starts)

        _, clean = sc.analyze_channel(signal, SFREQ, "F3-M2")
        _, naive = sc.analyze_channel(contaminated, SFREQ, "F3-M2")
        _, rejected = sc.analyze_channel(
            contaminated, SFREQ, "F3-M2", reject_blinks=True,
        )

        # Left alone, blinks are detected as spurious slow oscillations.
        assert naive["n_so"] > clean["n_so"]
        assert rejected["n_so"] < naive["n_so"]
        assert rejected["n_blinks"] == len(windows)
        assert rejected["blink_rejected_s"] > 0.0

    def test_no_detected_event_overlaps_a_blink(self):
        signal, starts = synthetic_recording(seed=1)
        contaminated, windows = with_blinks(signal, starts)
        events, _ = sc.analyze_channel(
            contaminated, SFREQ, "F3-M2", reject_blinks=True,
        )
        for event in events:
            onset = int(round(event.onset_s * SFREQ))
            offset = onset + int(round(event.duration_s * SFREQ))
            assert all(
                offset <= i0 or onset >= i1 for i0, i1 in windows
            ), f"{event.event_type} at {event.onset_s:.2f}s overlaps a blink"

    def test_preferred_phase_survives_blink_rejection(self):
        signal, starts = synthetic_recording(seed=1)
        contaminated, _ = with_blinks(signal, starts)
        _, row = sc.analyze_channel(
            contaminated, SFREQ, "F3-M2", reject_blinks=True,
        )
        assert row["n_coupled"] >= 0.8 * len(starts)
        # Bursts still sit on the SO up-state: preferred phase near 0 rad.
        assert abs(row["preferred_phase_deg"]) < 30.0
        assert row["resultant_length"] > 0.6
        assert row["rayleigh_p"] < 0.001

    def test_signal_is_not_modified(self):
        signal, starts = synthetic_recording(seed=1)
        contaminated, _ = with_blinks(signal, starts)
        before = contaminated.copy()
        sc.analyze_channel(contaminated, SFREQ, "F3-M2", reject_blinks=True)
        assert np.array_equal(contaminated, before)

    def test_blink_mask_combines_with_the_hypnogram(self):
        signal, starts = synthetic_recording(seed=1)
        contaminated, _ = with_blinks(signal, starts)
        hypno = np.full(contaminated.size, 2)
        half = contaminated.size // 2
        hypno[:half] = 0  # Wake: nothing may be detected there.
        mask = sc.stage_mask(hypno, sc.SPINDLE_STAGES, contaminated.size)

        events, row = sc.analyze_channel(
            contaminated, SFREQ, "F3-M2",
            so_mask=sc.stage_mask(hypno, sc.SO_STAGES, contaminated.size),
            spindle_mask=mask,
            reject_blinks=True,
        )
        assert row["n_coupled"] > 0
        assert all(e.onset_s * SFREQ >= half for e in events)

    def test_columns_are_absent_without_rejection(self):
        signal, _ = synthetic_recording(seed=1)
        _, row = sc.analyze_channel(signal, SFREQ, "F3-M2")
        assert "n_blinks" not in row
        assert "blink_rejected_s" not in row


class TestRecordingLevelRun:
    def test_stats_carry_the_blink_columns(self):
        signal, starts = synthetic_recording(seed=1)
        contaminated, windows = with_blinks(signal, starts)
        result = sc.detect_so_spindle_coupling(
            contaminated[None, :], SFREQ, ["F3-M2"], reject_blinks=True,
        )
        assert list(result.stats.columns) == sc.HEADBAND_STATS_COLUMNS
        assert int(result.stats.iloc[0]["n_blinks"]) == len(windows)

    def test_psg_run_keeps_the_original_columns(self):
        signal, _ = synthetic_recording(seed=1)
        result = sc.detect_so_spindle_coupling(signal[None, :], SFREQ, ["C3-M2"])
        assert list(result.stats.columns) == sc.STATS_COLUMNS


class TestChannelSelection:
    def test_clinical_psg_keeps_only_frontal_derivations(self):
        picked, skipped = hb.select_headband_channels(PSG_MONTAGE)
        assert picked == ["F3-M2", "F4-M1"]
        for name in ("C3-M2", "O1-M2", "E1-M2", "ECG", "SpO2 B-B", "F3 Impedance"):
            assert name in skipped

    def test_headband_without_eog_keeps_its_electrodes(self):
        picked, skipped = hb.select_headband_channels(
            ["AF7", "AF8", "Fpz", "Accelerometer X", "Battery"]
        )
        assert picked == ["AF7", "AF8", "Fpz"]
        assert skipped == ["Accelerometer X", "Battery"]

    def test_monopolar_is_dropped_when_referenced_exists(self):
        picked, _ = hb.select_headband_channels(["F3", "F3-M2", "F4"])
        assert picked == ["F3-M2", "F4"]

    def test_requested_channels_are_filtered(self):
        picked, skipped = hb.select_headband_channels(
            PSG_MONTAGE, ["F3-M2", "C3-M2"],
        )
        assert picked == ["F3-M2"]
        assert skipped == ["C3-M2"]

    def test_requested_channel_absent_raises(self):
        with pytest.raises(ValueError, match="aucun canal demande"):
            hb.select_headband_channels(PSG_MONTAGE, ["Fp1-M2"])

    def test_montage_without_frontal_raises(self):
        with pytest.raises(ValueError, match="aucun canal frontal"):
            hb.select_headband_channels(["C3-M2", "O1-M2", "ECG"])

    def test_frontal_predicate(self):
        assert hb.is_frontal_channel("F7-O1")
        assert hb.is_frontal_channel("EEG Fp1-A2")
        assert not hb.is_frontal_channel("F3 Impedance")
        assert not hb.is_frontal_channel("E1-M2")
        assert not hb.is_frontal_channel("C4-M1")
