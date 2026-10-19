"""Tests for the EEG signal sanity check behind the N3 diagnosis."""

from __future__ import annotations

import numpy as np
import pytest

from CESA.scoring_signal_check import (
    SLOW_WAVE_UV,
    amplitude_diagnosis,
    eeg_signal_stats,
    n3_capability_note,
    n3_epoch_mask,
    promote_n3,
)

SFREQ = 100.0
EPOCH = 30.0


def _signal(amplitude_uv, freq=1.0, minutes=10.0, noise_uv=5.0, seed=0):
    """Sine wave of a given peak-to-peak amplitude, expressed in volts."""
    rng = np.random.default_rng(seed)
    n = int(SFREQ * 60 * minutes)
    t = np.arange(n) / SFREQ
    wave = (amplitude_uv / 2.0) * np.sin(2 * np.pi * freq * t)
    wave = wave + rng.normal(0.0, noise_uv, size=n)
    return wave * 1e-6


class TestSignalStats:
    def test_amplitude_is_measured_in_microvolts(self):
        stats = eeg_signal_stats(_signal(150.0), SFREQ, EPOCH)
        assert stats is not None
        assert stats["amplitude_p2p_uv_mediane"] == pytest.approx(150.0, rel=0.25)

    def test_slow_waves_above_threshold_are_detected(self):
        stats = eeg_signal_stats(_signal(200.0, freq=1.0), SFREQ, EPOCH)
        assert stats["onde_lente_p2p_uv_p95"] > SLOW_WAVE_UV
        assert stats["pct_epoques_criteres_n3"] > 90.0

    def test_low_amplitude_signal_meets_no_n3_criterion(self):
        stats = eeg_signal_stats(_signal(20.0, freq=1.0, noise_uv=1.0), SFREQ, EPOCH)
        assert stats["pct_epoques_criteres_n3"] == pytest.approx(0.0)

    def test_fast_activity_is_not_counted_as_slow_wave(self):
        # A large 10 Hz alpha rhythm must not satisfy the 0.5-2 Hz criterion.
        stats = eeg_signal_stats(_signal(200.0, freq=10.0), SFREQ, EPOCH)
        assert stats["pct_epoques_criteres_n3"] == pytest.approx(0.0)

    def test_epoch_count_follows_epoch_length(self):
        stats = eeg_signal_stats(_signal(100.0, minutes=5.0), SFREQ, EPOCH)
        assert stats["n_epoques_analysees"] == 10

    def test_signal_shorter_than_one_epoch_returns_none(self):
        short = np.zeros(int(SFREQ * 10))
        assert eeg_signal_stats(short, SFREQ, EPOCH) is None

    def test_non_finite_samples_are_ignored(self):
        data = _signal(150.0)
        data[100:150] = np.nan
        stats = eeg_signal_stats(data, SFREQ, EPOCH)
        assert stats is not None
        assert np.isfinite(stats["amplitude_p2p_uv_mediane"])

    def test_unit_scale_can_be_overridden_for_data_already_in_uv(self):
        data_uv = _signal(150.0) * 1e6
        stats = eeg_signal_stats(data_uv, SFREQ, EPOCH, unit_scale=1.0)
        assert stats["amplitude_p2p_uv_mediane"] == pytest.approx(150.0, rel=0.25)


class TestAmplitudeDiagnosis:
    def test_plausible_amplitude_is_silent(self):
        assert amplitude_diagnosis({"amplitude_p2p_uv_mediane": 120.0}) is None

    def test_amplitude_far_too_low_suggests_unit_scaling(self):
        message = amplitude_diagnosis({"amplitude_p2p_uv_mediane": 0.15})
        assert message is not None
        assert "facteur 1000" in message

    def test_amplitude_slightly_too_low_is_flagged_without_factor_claim(self):
        message = amplitude_diagnosis({"amplitude_p2p_uv_mediane": 8.0})
        assert message is not None
        assert "facteur 1000" not in message

    def test_amplitude_far_too_high_is_flagged(self):
        message = amplitude_diagnosis({"amplitude_p2p_uv_mediane": 5000.0})
        assert message is not None
        assert "au-dela" in message

    def test_missing_stats_are_tolerated(self):
        assert amplitude_diagnosis(None) is None
        assert amplitude_diagnosis({}) is None


class TestPromoteN3:
    def test_only_n2_is_rewritten(self):
        stages, n = promote_n3(
            ["W", "N2", "N2", "R", "N3"],
            [True, True, False, True, True],
        )
        assert stages == ["W", "N3", "N2", "R", "N3"]
        assert n == 1

    def test_mask_shorter_than_hypnogram_is_safe(self):
        stages, n = promote_n3(["N2", "N2", "N2"], [True])
        assert stages == ["N3", "N2", "N2"]
        assert n == 1

    def test_n3_mask_marks_high_amplitude_slow_waves(self):
        mask = n3_epoch_mask(_signal(200.0, freq=1.0, minutes=1.0), SFREQ, EPOCH)
        assert mask is not None
        assert bool(mask[0])

    def test_n3_mask_rejects_low_amplitude(self):
        mask = n3_epoch_mask(
            _signal(20.0, freq=1.0, minutes=1.0, noise_uv=1.0), SFREQ, EPOCH,
        )
        assert mask is not None
        assert not bool(mask[0])


class TestN3CapabilityNote:
    def test_signal_without_slow_waves_blames_the_signal(self):
        note = n3_capability_note(
            {"pct_epoques_criteres_n3": 0.0, "onde_lente_p2p_uv_p95": 20.0},
            predicted_n3_pct=0.0,
            manual_n3_pct=20.1,
        )
        assert note is not None
        assert "vient donc du signal" in note
        assert "20.1%" in note

    def test_slow_waves_present_but_unpredicted_blames_the_model(self):
        note = n3_capability_note(
            {"pct_epoques_criteres_n3": 25.0, "onde_lente_p2p_uv_p95": 150.0},
            predicted_n3_pct=0.1,
            manual_n3_pct=20.0,
        )
        assert note is not None
        assert "limite vient du modele" in note

    def test_consistent_prediction_is_silent(self):
        assert n3_capability_note(
            {"pct_epoques_criteres_n3": 18.0, "onde_lente_p2p_uv_p95": 150.0},
            predicted_n3_pct=17.0,
        ) is None

    def test_missing_measurement_is_tolerated(self):
        assert n3_capability_note(None) is None
        assert n3_capability_note({"amplitude_p2p_uv_mediane": 100.0}) is None
