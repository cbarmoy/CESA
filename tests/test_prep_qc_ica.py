"""Tests for night QC heuristics and ICA cleaning helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from CESA.prep_qc import compute_night_qc, should_exclude_channel_name
from CESA import ica_cleaning as ica


class TestPrepQc:
    def test_detects_flat_and_ok_channels(self):
        fs = 100.0
        t = np.arange(0, 30.0, 1 / fs)
        good = 20.0 * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t)) * 2
        flat = np.ones_like(t) * 0.01
        report = compute_night_qc(
            {"C3": good, "Flat": flat},
            fs,
            {"C3": "eeg", "Flat": "eeg"},
            flat_std_uv=0.5,
            snr_warn_db=-100,  # disable SNR warnings for this unit test
            line_noise_warn=100,
        )
        assert "Flat" in report.bad_channels
        flat_score = next(c for c in report.per_channel if c.name == "Flat")
        assert flat_score.is_flat

    def test_empty_derivations(self):
        report = compute_night_qc({}, 100.0)
        assert report.n_channels == 0
        assert report.bad_channels == []
        assert report.warnings

    def test_single_adc_spike_not_extreme(self):
        """One clipped sample at ~3276 µV must not flag extreme amp."""
        fs = 100.0
        rng = np.random.default_rng(0)
        t = np.arange(0, 60.0, 1 / fs)
        good = 25.0 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 3, size=t.shape)
        good[100] = 3276.0
        report = compute_night_qc(
            {"C3": good},
            fs,
            {"C3": "eeg"},
            extreme_amp_uv=1500.0,
            extreme_frac_min=0.001,
            snr_warn_db=-100,
            line_noise_warn=100,
        )
        score = report.per_channel[0]
        assert not score.extreme_amp
        assert "C3" not in report.bad_channels

    def test_sustained_saturation_is_extreme(self):
        """~5 % of samples above threshold should flag extreme amp."""
        fs = 100.0
        n = 10_000
        x = np.random.default_rng(1).normal(0, 20, size=n)
        n_hi = int(0.05 * n)
        x[:n_hi] = 2000.0
        report = compute_night_qc(
            {"C4": x},
            fs,
            {"C4": "eeg"},
            extreme_amp_uv=1500.0,
            extreme_frac_min=0.001,
            snr_warn_db=-100,
            line_noise_warn=100,
            flat_std_uv=0.0,
        )
        score = report.per_channel[0]
        assert score.extreme_amp
        assert any("extreme amp" in r for r in score.reasons)

    def test_impedance_and_spo2_excluded(self):
        fs = 100.0
        t = np.arange(0, 10.0, 1 / fs)
        eeg = 20.0 * np.sin(2 * np.pi * 10 * t)
        imp = np.full_like(t, 50_000.0)
        spo2 = np.full_like(t, 98.0)
        report = compute_night_qc(
            {
                "C3": eeg,
                "C3 Impédance": imp,
                "SpO2": spo2,
            },
            fs,
            {
                "C3": "eeg",
                "C3 Impédance": "eeg",  # mis-typed like real PSG files
                "SpO2": "eeg",
            },
            snr_warn_db=-100,
            line_noise_warn=100,
        )
        names = {c.name for c in report.per_channel}
        assert names == {"C3"}
        assert report.n_channels == 1

    def test_should_exclude_channel_name(self):
        assert should_exclude_channel_name("C3 Impédance")
        assert should_exclude_channel_name("SpO2")
        assert should_exclude_channel_name("SAO2 Finger")
        assert not should_exclude_channel_name("C3")
        assert not should_exclude_channel_name("Fz-Cz")

    def test_no_fallback_when_only_aux(self):
        report = compute_night_qc(
            {"C3 Impédance": np.ones(1000) * 1e5},
            100.0,
            {"C3 Impédance": "eeg"},
        )
        assert report.n_channels == 0
        assert report.per_channel == []


class TestIcaCleaning:
    def test_preview_exclude_mocked(self):
        fake_raw = MagicMock()
        fake_raw.copy.return_value = fake_raw
        cleaned = MagicMock()
        cleaned.get_data.return_value = np.ones((2, 100)) * 1e-6  # Volts
        fake_ica = MagicMock()
        fake_ica.apply.return_value = cleaned

        fit = ica.IcaFitResult(
            ica=fake_ica,
            raw=fake_raw,
            ch_names=["C3", "C4"],
            sfreq=100.0,
            explained_var=[40.0, 30.0],
            n_components=2,
        )
        out = ica.preview_exclude(fit, exclude=[0])
        assert set(out.keys()) == {"C3", "C4"}
        assert np.allclose(out["C3"], 1.0)  # back to µV
        assert fake_ica.exclude == [0]

    def test_component_labels(self):
        fit = ica.IcaFitResult(
            ica=None, raw=None, ch_names=["a"], sfreq=100.0,
            explained_var=[12.5], n_components=1,
        )
        assert ica.component_labels(fit) == ["IC0 (12.5%)"]

    def test_component_labels_enriched(self):
        fit = ica.IcaFitResult(
            ica=None, raw=None, ch_names=["a"], sfreq=100.0,
            explained_var=[12.5], n_components=1,
        )
        stats = [{
            "index": 0,
            "kurtosis": 8.2,
            "var_pct": 12.5,
            "corr_eog": 0.71,
            "corr_ecg": 0.1,
            "label_hint": "eog",
        }]
        labels = ica.component_labels(fit, stats)
        assert "IC0" in labels[0]
        assert "kurt=8.2" in labels[0]
        assert "~eog" in labels[0]

    def test_component_stats_synthetic(self):
        """Kurtosis / corr hints on synthetic sources without MNE fit."""
        fs = 100.0
        n = 5000
        t = np.arange(n) / fs
        # IC0 ~ EOG-like (high corr with eog, leptokurtic blinks)
        eog = np.sin(2 * np.pi * 0.3 * t)
        eog[::500] += 20.0
        # IC1 ~ noise
        noise = np.random.default_rng(2).normal(0, 1, size=n)
        sources = np.vstack([eog, noise])

        fake_ica = MagicMock()
        fake_raw = MagicMock()
        fit = ica.IcaFitResult(
            ica=fake_ica,
            raw=fake_raw,
            ch_names=["C3", "C4"],
            sfreq=fs,
            explained_var=[60.0, 40.0],
            n_components=2,
            sources=sources,
        )
        stats = ica.component_stats(fit, eog_uv=eog, ecg_uv=noise * 0.01)
        assert len(stats) == 2
        assert stats[0]["corr_eog"] > 0.9
        assert stats[0]["label_hint"] == "eog"
        assert stats[0]["var_pct"] == 60.0
        # IC1 should not be tagged eog
        assert stats[1]["label_hint"] in ("other", "ecg")

    def test_component_spectrum(self):
        fs = 100.0
        t = np.arange(0, 10.0, 1 / fs)
        x = np.sin(2 * np.pi * 10 * t)
        freqs, psd = ica.component_spectrum(x, fs)
        assert freqs.size == psd.size
        peak = freqs[int(np.argmax(psd))]
        assert 8.0 <= peak <= 12.0

    def test_per_component_var_pct_via_fit(self):
        """Explained variance must be > 0 after fit (MNE modern API)."""
        mne = pytest.importorskip("mne")
        from mne.preprocessing import ICA  # noqa: F401

        fs = 100.0
        n = 5000
        rng = np.random.default_rng(0)
        t = np.arange(n) / fs
        data = {
            "C3": 20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 2, n),
            "C4": 18 * np.sin(2 * np.pi * 10 * t + 0.3) + rng.normal(0, 2, n),
            "O1": 15 * np.sin(2 * np.pi * 8 * t) + rng.normal(0, 2, n),
        }
        fit = ica.fit_ica(data, fs, n_components=2)
        assert fit.explained_var
        assert all(v > 0 for v in fit.explained_var)
        assert abs(sum(fit.explained_var) - 0) > 1.0  # not all zeros

    def test_get_sources_window(self):
        mne = pytest.importorskip("mne")
        fs = 100.0
        n = 2000
        rng = np.random.default_rng(1)
        data = {
            "C3": rng.normal(0, 10, n),
            "C4": rng.normal(0, 10, n),
            "O1": rng.normal(0, 10, n),
        }
        fit = ica.fit_ica(data, fs, n_components=2)
        src = ica.get_sources(fit, tmin=2.0, tmax=5.0)
        assert src.shape[0] == 2
        assert abs(src.shape[1] - int(3.0 * fs)) <= 2

    def test_controller_ica_backup(self):
        from CESA.app_controller import AppController

        ctrl = AppController.__new__(AppController)
        ctrl.derivations = {"C3": np.array([1.0, 2.0, 3.0])}
        ctrl._ica_backup = None
        n = AppController.apply_ica_cleaning(
            ctrl, {"C3": np.array([9.0, 9.0, 9.0])}
        )
        assert n == 1
        assert ctrl.has_ica_cleaning()
        assert np.allclose(ctrl.derivations["C3"], [9, 9, 9])
        restored = AppController.restore_ica_backup(ctrl)
        assert np.allclose(restored["C3"], [1, 2, 3])
        assert not ctrl.has_ica_cleaning()
