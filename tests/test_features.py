"""Tests for CESA.sleep_pipeline.features."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.features import (
    BANDS,
    _band_power,
    _hjorth,
    _zero_crossing_rate,
    epoch_features_eeg,
    epoch_features_emg,
    epoch_features_eog,
)


@pytest.fixture
def sfreq():
    return 100.0


@pytest.fixture
def epoch_30s(sfreq):
    """30 s of synthetic EEG at 100 Hz (sine at 10 Hz = alpha)."""
    t = np.arange(int(30 * sfreq)) / sfreq
    return np.sin(2 * np.pi * 10 * t) * 50.0  # 50 uV alpha


@pytest.fixture
def delta_epoch(sfreq):
    """30 s of dominant delta (2 Hz)."""
    t = np.arange(int(30 * sfreq)) / sfreq
    return np.sin(2 * np.pi * 2 * t) * 100.0


class TestBandPower:
    def test_alpha_dominates(self, epoch_30s, sfreq):
        from scipy.signal import welch
        freqs, psd = welch(epoch_30s, fs=sfreq, nperseg=int(4 * sfreq))
        alpha_pow = _band_power(psd, freqs, 8.0, 12.0)
        delta_pow = _band_power(psd, freqs, 0.5, 4.0)
        assert alpha_pow > delta_pow * 5

    def test_delta_dominates(self, delta_epoch, sfreq):
        from scipy.signal import welch
        freqs, psd = welch(delta_epoch, fs=sfreq, nperseg=int(4 * sfreq))
        delta_pow = _band_power(psd, freqs, 0.5, 4.0)
        alpha_pow = _band_power(psd, freqs, 8.0, 12.0)
        assert delta_pow > alpha_pow * 10


class TestEpochFeatures:
    def test_eeg_features_keys(self, epoch_30s, sfreq):
        feats = epoch_features_eeg(epoch_30s, sfreq)
        expected_bands = list(BANDS.keys())
        for band in expected_bands:
            assert f"power_{band}" in feats
            assert f"relpow_{band}" in feats
        assert "ratio_delta_beta" in feats
        assert "variance" in feats
        assert "rms" in feats
        assert "zcr" in feats
        assert "hjorth_activity" in feats

    def test_eeg_alpha_dominant_features(self, epoch_30s, sfreq):
        feats = epoch_features_eeg(epoch_30s, sfreq)
        assert feats["relpow_alpha"] > 0.3  # alpha-dominant signal

    def test_eog_features_keys(self, epoch_30s, sfreq):
        feats = epoch_features_eog(epoch_30s, sfreq)
        assert "eog_variance" in feats
        assert "eog_rem_activity" in feats

    def test_emg_features_keys(self, epoch_30s, sfreq):
        feats = epoch_features_emg(epoch_30s, sfreq)
        assert "emg_rms" in feats
        assert "emg_mean_rect" in feats


class TestHjorth:
    def test_constant_signal(self):
        x = np.ones(1000)
        act, mob, comp = _hjorth(x)
        assert act < 1e-10

    def test_sine_signal(self):
        t = np.arange(1000) / 100.0
        x = np.sin(2 * np.pi * 5 * t)
        act, mob, comp = _hjorth(x)
        assert act > 0
        assert mob > 0


class TestZeroCrossing:
    def test_pure_sine(self):
        t = np.arange(1000) / 100.0
        x = np.sin(2 * np.pi * 5 * t)
        zcr = _zero_crossing_rate(x)
        # 5 Hz sine -> ~10 crossings/s -> ZCR ~ 10/100 = 0.1
        assert 0.05 < zcr < 0.2

    def test_constant(self):
        x = np.ones(100)
        assert _zero_crossing_rate(x) == 0.0
