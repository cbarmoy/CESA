"""Unit tests for the classical SO-spindle coupling pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from CESA.sleep_pipeline.contracts import EventType
from CESA.sleep_pipeline import so_spindle_coupling as sc


SFREQ = 100.0
DURATION_S = 400.0
SO_PERIOD_S = 10.0
# The injected SO is -sin over 1 s: trough at 0.25 s, positive peak at 0.75 s,
# where the analytic SO phase is ~0 rad.
SO_POSITIVE_PEAK_OFFSET_S = 0.75


def synthetic_recording(
    *,
    seed: int = 1,
    spindle_offsets: Optional[np.ndarray] = None,
    noise_uv: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadband noise plus SO cycles, each carrying one sigma burst.

    Returns ``(signal_uv, so_start_times_s)``.  When *spindle_offsets* is
    omitted every burst sits on the SO positive peak, so the expected
    preferred phase is 0 rad.
    """
    t = np.arange(0, DURATION_S, 1.0 / SFREQ)
    rng = np.random.default_rng(seed)
    signal = sc.bandpass(rng.normal(0.0, noise_uv, t.size), SFREQ, 0.3, 40.0)

    starts = np.arange(20.0, DURATION_S - 20.0, SO_PERIOD_S)
    offsets = (
        np.full(starts.size, SO_POSITIVE_PEAK_OFFSET_S)
        if spindle_offsets is None else spindle_offsets
    )
    for start, offset in zip(starts, offsets):
        i0 = int(start * SFREQ)
        i1 = i0 + int(SFREQ)
        cycle_t = np.arange(i1 - i0) / SFREQ
        signal[i0:i1] += -70.0 * np.sin(2.0 * np.pi * cycle_t)

        center = start + offset
        burst = (t > center - 0.6) & (t < center + 0.6)
        signal[burst] += 45.0 * np.exp(
            -0.5 * ((t[burst] - center) / 0.30) ** 2
        ) * np.sin(2.0 * np.pi * 13.0 * (t[burst] - center))
    return signal, starts


class TestCircularPhaseStats:
    def test_concentrated_phases(self):
        angles = np.full(20, 0.3) + np.linspace(-0.05, 0.05, 20)
        stats = sc.circular_phase_stats(angles)
        assert stats["n"] == 20
        assert stats["preferred_phase_rad"] == pytest.approx(0.3, abs=0.01)
        assert stats["preferred_phase_deg"] == pytest.approx(np.degrees(0.3), abs=1.0)
        assert stats["resultant_length"] > 0.99
        assert stats["rayleigh_p"] < 0.001

    def test_uniform_phases_are_not_significant(self):
        angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
        stats = sc.circular_phase_stats(angles)
        assert stats["resultant_length"] < 0.05
        assert stats["rayleigh_p"] > 0.5

    def test_wraps_around_pi(self):
        # Angles straddling the +/-pi discontinuity: the circular mean must be
        # near pi, not near 0 as an arithmetic mean would give.
        angles = np.array([3.0, 3.1, -3.1, -3.0])
        stats = sc.circular_phase_stats(angles)
        assert abs(abs(stats["preferred_phase_rad"]) - np.pi) < 0.2
        assert stats["resultant_length"] > 0.9

    def test_empty_returns_nan(self):
        stats = sc.circular_phase_stats([])
        assert stats["n"] == 0
        assert np.isnan(stats["preferred_phase_rad"])
        assert stats["backend"] == "none"

    def test_numpy_fallback_matches_pycircstat2(self, monkeypatch):
        angles = np.random.default_rng(3).vonmises(0.6, 3.0, 40)
        reference = sc.circular_phase_stats(angles)
        if reference["backend"] != "pycircstat2":
            pytest.skip("pycircstat2 not installed")

        monkeypatch.setattr(sc, "_circular_backend", lambda: None)
        fallback = sc.circular_phase_stats(angles)
        assert fallback["backend"] == "numpy"
        for key in (
            "preferred_phase_rad", "resultant_length", "rayleigh_z", "rayleigh_p",
        ):
            assert fallback[key] == pytest.approx(reference[key], rel=1e-6, abs=1e-9)


class TestHelpers:
    def test_wrap_to_pi(self):
        assert sc.wrap_to_pi(0.0) == pytest.approx(0.0)
        assert sc.wrap_to_pi(-3.0 * np.pi / 2.0) == pytest.approx(np.pi / 2.0)
        assert sc.wrap_to_pi(2.5 * np.pi) == pytest.approx(np.pi / 2.0)
        # The trough maps to the -pi end of the half-open range.
        assert sc.wrap_to_pi(3.0 * np.pi) == pytest.approx(-np.pi, abs=1e-9)

    def test_stage_mask(self):
        hypno = np.array([0, 0, 2, 2, 3, 3, 4, 4])
        mask = sc.stage_mask(hypno, (2, 3), 8)
        assert mask is not None
        assert mask.tolist() == [False, False, True, True, True, True, False, False]

    def test_stage_mask_without_hypno(self):
        assert sc.stage_mask(None, (2,), 100) is None

    def test_stage_mask_pads_short_hypno(self):
        mask = sc.stage_mask(np.array([2, 2]), (2,), 5)
        assert mask.tolist() == [True, True, False, False, False]

    def test_bandpass_attenuates_out_of_band(self):
        t = np.arange(0, 10.0, 1.0 / SFREQ)
        signal = np.sin(2.0 * np.pi * 1.0 * t) + np.sin(2.0 * np.pi * 30.0 * t)
        filtered = sc.bandpass(signal, SFREQ, 0.5, 4.0)
        # Only the 1 Hz component should survive (amplitude ~1).
        assert 0.8 < np.max(np.abs(filtered[100:-100])) < 1.2


class TestDetectors:
    def test_detects_injected_slow_oscillations(self):
        signal, starts = synthetic_recording()
        events = sc.detect_slow_oscillations(signal, SFREQ)
        assert len(events) >= len(starts)
        for event in events:
            duration = (event.end_idx - event.start_idx) / SFREQ
            assert sc.SO_MIN_DURATION_S <= duration <= sc.SO_MAX_DURATION_S
            assert event.trough_uv < 0.0
            assert event.start_idx <= event.trough_idx < event.end_idx

    def test_detects_injected_spindles(self):
        signal, starts = synthetic_recording()
        events = sc.detect_spindles_hilbert(signal, SFREQ)
        assert 0.8 * len(starts) <= len(events) <= 1.2 * len(starts)
        for event in events:
            duration = (event.end_idx - event.start_idx) / SFREQ
            assert sc.SPINDLE_MIN_DURATION_S <= duration <= sc.SPINDLE_MAX_DURATION_S
            assert 11.0 < event.frequency_hz < 17.0

    def test_mask_restricts_detection(self):
        signal, _ = synthetic_recording()
        half = signal.size // 2
        mask = np.zeros(signal.size, dtype=bool)
        mask[half:] = True
        events = sc.detect_spindles_hilbert(signal, SFREQ, mask=mask)
        assert events
        assert all(event.start_idx >= half for event in events)

    def test_flat_signal_yields_nothing(self):
        flat = np.zeros(int(60 * SFREQ))
        assert sc.detect_slow_oscillations(flat, SFREQ) == []
        assert sc.detect_spindles_hilbert(flat, SFREQ) == []


class TestCoupling:
    def test_recovers_known_preferred_phase(self):
        signal, starts = synthetic_recording(seed=1)
        result = sc.detect_so_spindle_coupling(signal[None, :], SFREQ, ["Cz"])

        assert len(result.stats) == 1
        row = result.stats.iloc[0]
        assert row["channel"] == "Cz"
        assert row["n_coupled"] >= 0.8 * len(starts)
        # Bursts sit on the SO up-state: preferred phase near 0 rad.
        assert abs(row["preferred_phase_rad"]) < 0.5
        assert abs(row["preferred_phase_deg"]) < 30.0
        assert row["resultant_length"] > 0.6
        assert row["rayleigh_p"] < 0.001

    def test_uniform_offsets_give_weak_coupling(self):
        rng = np.random.default_rng(11)
        n_so = int((DURATION_S - 40.0) / SO_PERIOD_S) + 1
        signal, _ = synthetic_recording(
            seed=1, spindle_offsets=rng.uniform(0.0, 1.0, n_so),
        )
        result = sc.detect_so_spindle_coupling(signal[None, :], SFREQ, ["Cz"])
        row = result.stats.iloc[0]
        assert row["n_coupled"] > sc.MIN_COUPLED_EVENTS
        assert row["resultant_length"] < 0.4

    def test_coupled_spindles_carry_phase_metadata(self):
        signal, _ = synthetic_recording(seed=2)
        result = sc.detect_so_spindle_coupling(signal[None, :], SFREQ, ["Cz"])

        spindles = [e for e in result.events if e.event_type == EventType.SPINDLE]
        coupled = [e for e in spindles if "so_phase_rad" in e.metadata]
        assert coupled
        event = coupled[0]
        assert event.metadata["detector"] == "cesa_classic"
        assert -np.pi <= event.metadata["so_phase_rad"] <= np.pi
        assert event.metadata["so_phase_deg"] == pytest.approx(
            np.degrees(event.metadata["so_phase_rad"]), abs=0.01
        )
        assert event.metadata["coupled_so_onset_s"] <= event.metadata["peak_s"]
        assert event.to_dict()["channel"] == "Cz"

        slow_waves = [e for e in result.events if e.event_type == EventType.SLOW_WAVE]
        assert any(e.metadata["CooccurringSpindle"] for e in slow_waves)

    def test_too_few_coupled_events_hides_preferred_phase(self):
        signal, _ = synthetic_recording(seed=1)
        result = sc.detect_so_spindle_coupling(
            signal[None, :], SFREQ, ["Cz"], min_coupled=10_000,
        )
        row = result.stats.iloc[0]
        assert row["n_coupled"] > 0
        assert np.isnan(row["preferred_phase_rad"])
        assert np.isnan(row["resultant_length"])
        assert any("phase preferee non publiee" in w for w in result.warnings)

    def test_hypnogram_restricts_analysis(self):
        signal, _ = synthetic_recording(seed=1)
        hypno = np.full(signal.size, 0, dtype=np.int16)  # W everywhere
        hypno[signal.size // 2:] = 2  # N2 on the second half only
        result = sc.detect_so_spindle_coupling(
            signal[None, :], SFREQ, ["Cz"], hypno=hypno,
        )
        onsets = [
            e.onset_s for e in result.events if e.event_type == EventType.SPINDLE
        ]
        assert onsets
        assert min(onsets) >= (signal.size / 2) / SFREQ

    def test_stats_columns_are_stable(self):
        signal, _ = synthetic_recording(seed=1)
        result = sc.detect_so_spindle_coupling(signal[None, :], SFREQ, ["Cz"])
        assert list(result.stats.columns) == sc.STATS_COLUMNS

    def test_short_recording_is_rejected(self):
        short = np.zeros((1, int(5 * SFREQ)))
        result = sc.detect_so_spindle_coupling(short, SFREQ, ["Cz"])
        assert result.events == []
        assert result.stats.empty
        assert any("trop court" in w.lower() for w in result.warnings)

    def test_channel_count_mismatch_raises(self):
        data = np.zeros((2, int(60 * SFREQ)))
        with pytest.raises(ValueError):
            sc.detect_so_spindle_coupling(data, SFREQ, ["Cz"])


class TestControllerCoupling:
    def _controller(self, derivations):
        from CESA.app_controller import AppController

        ctrl = AppController.__new__(AppController)
        ctrl.derivations = derivations
        ctrl.raw = None
        ctrl.sfreq = SFREQ
        ctrl.scoring_epoch_duration = 30.0
        ctrl.yasa_eeg_candidates = []
        ctrl.yasa_eog_candidates = []
        ctrl.get_active_scoring_df = lambda: None  # type: ignore[method-assign]
        return ctrl

    def test_no_recording_returns_warning(self):
        from CESA.app_controller import AppController

        ctrl = self._controller({})
        out = AppController.detect_so_spindle_coupling(ctrl)
        assert out["events"] == []
        assert out["coupling"] == []
        assert out["warnings"]

    def test_runs_on_loaded_channel(self):
        from CESA.app_controller import AppController

        signal, _ = synthetic_recording(seed=1)
        ctrl = self._controller({"C3": signal})
        out = AppController.detect_so_spindle_coupling(ctrl)

        assert out["eeg_channels"] == ["C3"]
        assert out["used_hypnogram"] is False
        assert out["coupling"]
        assert out["coupling"][0]["channel"] == "C3"
        assert abs(out["coupling"][0]["preferred_phase_rad"]) < 0.5
        assert any(ev["type"] == "spindle" for ev in out["events"])

    def test_uses_hypnogram_when_scoring_available(self):
        from CESA.app_controller import AppController

        signal, _ = synthetic_recording(seed=1)
        ctrl = self._controller({"C3": signal})
        epochs = int(DURATION_S // 30.0)
        scoring = pd.DataFrame({
            "time": [30.0 * i for i in range(epochs)],
            "stage": ["N2"] * epochs,
        })
        ctrl.get_active_scoring_df = lambda: scoring  # type: ignore[method-assign]

        out = AppController.detect_so_spindle_coupling(ctrl)
        assert out["used_hypnogram"] is True
        assert out["coupling"]
