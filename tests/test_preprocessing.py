"""Tests for CESA.sleep_pipeline.preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.preprocessing import _bandpass, PreprocessingConfig


class TestBandpass:
    def test_passband(self):
        sfreq = 256.0
        t = np.arange(int(10 * sfreq)) / sfreq
        # 10 Hz signal (should pass alpha band 8-12)
        sig = np.sin(2 * np.pi * 10 * t)
        filtered = _bandpass(sig, sfreq, 8.0, 12.0, order=4)
        assert np.std(filtered) > 0.3  # signal preserved

    def test_stopband(self):
        sfreq = 256.0
        t = np.arange(int(10 * sfreq)) / sfreq
        # 30 Hz signal (should be removed by alpha band 8-12)
        sig = np.sin(2 * np.pi * 30 * t)
        filtered = _bandpass(sig, sfreq, 8.0, 12.0, order=4)
        assert np.std(filtered) < 0.1  # signal attenuated

    def test_invalid_range(self):
        sig = np.ones(100)
        result = _bandpass(sig, 100.0, 50.0, 10.0)
        np.testing.assert_array_equal(result, sig)


class TestPreprocessingConfig:
    def test_defaults(self):
        config = PreprocessingConfig()
        assert config.target_sfreq == 100.0
        assert config.epoch_duration_s == 30.0
        assert config.eeg_bandpass == (0.3, 35.0)
        assert config.artifact_uv_threshold == 500.0

    def test_custom(self):
        config = PreprocessingConfig(target_sfreq=128.0, epoch_duration_s=20.0)
        assert config.target_sfreq == 128.0
        assert config.epoch_duration_s == 20.0
