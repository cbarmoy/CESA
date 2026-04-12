"""Tests for advanced sleep-specific features in features.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.features import (
    _delta_amplitude_fraction,
    _detect_kcomplexes,
    _detect_rem_bursts,
    _detect_spindles,
    _emg_phasic_bursts,
    _emg_tonic_level,
    _slow_eye_movement_index,
    _spectral_entropy,
    epoch_features_eeg,
    epoch_features_emg,
    epoch_features_eog,
)


@pytest.fixture
def sfreq():
    return 100.0


@pytest.fixture
def alpha_epoch(sfreq):
    """30 s of pure 10 Hz alpha."""
    t = np.arange(int(30 * sfreq)) / sfreq
    return np.sin(2 * np.pi * 10 * t) * 50.0


@pytest.fixture
def delta_epoch(sfreq):
    """30 s of pure 2 Hz delta with 100 uV amplitude."""
    t = np.arange(int(30 * sfreq)) / sfreq
    return np.sin(2 * np.pi * 2 * t) * 100.0


@pytest.fixture
def sigma_epoch(sfreq):
    """30 s containing a spindle-like 14 Hz burst."""
    t = np.arange(int(30 * sfreq)) / sfreq
    sig = np.random.RandomState(42).randn(len(t)) * 5.0
    # Add a 1-second spindle at t=10s
    spindle_start = int(10.0 * sfreq)
    spindle_dur = int(1.0 * sfreq)
    sig[spindle_start: spindle_start + spindle_dur] += np.sin(
        2 * np.pi * 14 * t[:spindle_dur]
    ) * 40.0
    return sig


class TestSpectralEntropy:
    def test_narrowband_low_entropy(self, sfreq):
        from scipy.signal import welch
        t = np.arange(int(30 * sfreq)) / sfreq
        pure_sine = np.sin(2 * np.pi * 2 * t) * 100.0
        _, psd = welch(pure_sine, fs=sfreq, nperseg=int(4 * sfreq))
        se = _spectral_entropy(psd)
        assert se < 0.5

    def test_broadband_high_entropy(self, sfreq):
        from scipy.signal import welch
        noise = np.random.RandomState(42).randn(int(30 * sfreq))
        _, psd = welch(noise, fs=sfreq, nperseg=int(4 * sfreq))
        se = _spectral_entropy(psd)
        assert se > 0.7

    def test_in_eeg_features(self, alpha_epoch, sfreq):
        feats = epoch_features_eeg(alpha_epoch, sfreq)
        assert "spectral_entropy" in feats
        assert 0.0 <= feats["spectral_entropy"] <= 1.0


class TestSpindleDetection:
    def test_detects_spindle(self, sigma_epoch, sfreq):
        result = _detect_spindles(sigma_epoch, sfreq)
        assert result["spindle_count"] >= 1

    def test_no_spindle_in_delta(self, delta_epoch, sfreq):
        result = _detect_spindles(delta_epoch, sfreq)
        assert result["spindle_count"] == 0

    def test_in_eeg_features(self, sigma_epoch, sfreq):
        feats = epoch_features_eeg(sigma_epoch, sfreq)
        assert "spindle_count" in feats
        assert "spindle_density" in feats
        assert "spindle_mean_amplitude" in feats


class TestKComplexDetection:
    def test_no_kcomplex_in_alpha(self, alpha_epoch, sfreq):
        count = _detect_kcomplexes(alpha_epoch, sfreq)
        assert count == 0

    def test_in_eeg_features(self, alpha_epoch, sfreq):
        feats = epoch_features_eeg(alpha_epoch, sfreq)
        assert "kcomplex_count" in feats


class TestDeltaAmplitudeFraction:
    def test_high_amplitude_delta(self, sfreq):
        # Use a higher-amplitude low-frequency signal to survive bandpass filtering
        t = np.arange(int(30 * sfreq)) / sfreq
        strong_delta = np.sin(2 * np.pi * 1.0 * t) * 200.0  # 200 uV at 1 Hz
        frac = _delta_amplitude_fraction(strong_delta, sfreq, threshold_uv=75.0)
        assert frac > 0.1

    def test_low_amplitude_no_fraction(self, sfreq):
        t = np.arange(int(30 * sfreq)) / sfreq
        small_delta = np.sin(2 * np.pi * 2 * t) * 10.0  # 10 uV only
        frac = _delta_amplitude_fraction(small_delta, sfreq, threshold_uv=75.0)
        assert frac < 0.05

    def test_in_eeg_features(self, delta_epoch, sfreq):
        feats = epoch_features_eeg(delta_epoch, sfreq)
        assert "delta_fraction_above_75uv" in feats


class TestEOGAdvanced:
    def test_rem_burst_detection(self, sfreq):
        t = np.arange(int(30 * sfreq)) / sfreq
        # Simulate rapid eye movements as sharp deflections
        eog = np.random.RandomState(42).randn(len(t)) * 5.0
        for pos in [500, 1000, 1500, 2000]:
            eog[pos: pos + 5] += 200.0
            eog[pos + 5: pos + 10] -= 200.0
        count, density = _detect_rem_bursts(eog, sfreq)
        assert count >= 2

    def test_sem_index_low_freq(self, sfreq):
        t = np.arange(int(30 * sfreq)) / sfreq
        # Very low frequency signal (<0.5 Hz)
        slow = np.sin(2 * np.pi * 0.3 * t) * 50.0
        sem = _slow_eye_movement_index(slow, sfreq)
        assert sem > 10.0

    def test_eog_features_keys(self, sfreq):
        epoch = np.random.RandomState(42).randn(int(30 * sfreq))
        feats = epoch_features_eog(epoch, sfreq)
        assert "eog_rem_count" in feats
        assert "eog_rem_density" in feats
        assert "eog_sem_index" in feats


class TestEMGAdvanced:
    def test_tonic_level(self, sfreq):
        epoch = np.random.RandomState(42).randn(int(30 * sfreq)) * 10.0
        level = _emg_tonic_level(epoch, sfreq)
        assert level > 0

    def test_phasic_bursts(self, sfreq):
        epoch = np.random.RandomState(42).randn(int(30 * sfreq)) * 2.0
        # Add clear phasic bursts well above 3x baseline
        for pos in [500, 1000, 1500]:
            dur = int(0.2 * sfreq)
            epoch[pos: pos + dur] = 50.0  # Very large absolute value
        count = _emg_phasic_bursts(epoch, sfreq)
        assert count >= 1

    def test_emg_features_keys(self, sfreq):
        epoch = np.random.RandomState(42).randn(int(30 * sfreq))
        feats = epoch_features_emg(epoch, sfreq)
        assert "emg_tonic_level" in feats
        assert "emg_phasic_count" in feats
