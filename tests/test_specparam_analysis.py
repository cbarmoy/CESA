"""Tests for SpecParam analysis helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from CESA import specparam_analysis as spa


def _synthetic_psd(n: int = 100):
    freqs = np.linspace(0.5, 40.0, n)
    # 1/f + alpha peak (linear power)
    psd = 10 ** (-1.5 - 1.2 * np.log10(freqs)) + 0.4 * np.exp(
        -0.5 * ((freqs - 10.0) / 1.2) ** 2
    )
    return freqs, psd


class TestFitSpecParam:
    def test_fit_real_specparam(self):
        pytest.importorskip("specparam")
        freqs, psd = _synthetic_psd()
        result = spa.fit_specparam(freqs, psd, (0.5, 40.0))
        assert result["ok"]
        assert np.isfinite(result["aperiodic_exponent"])
        assert result["aperiodic_exponent"] > 0.5

    def test_short_psd_returns_warning(self):
        result = spa.fit_specparam(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert result["ok"] is False
        assert result["warning"]

    def test_results_to_rows(self):
        rows = spa.results_to_rows({
            "N2": {
                "ok": True,
                "aperiodic_offset": -1.0,
                "aperiodic_exponent": 1.5,
                "r_squared": 0.9,
                "error": 0.1,
                "peaks": [{"cf": 12.0, "pw": 1.0, "bw": 2.0}],
                "warning": "",
                "n_epochs": 10,
            }
        }, channel="C3")
        assert len(rows) == 1
        assert rows[0]["channel"] == "C3"
        assert rows[0]["peak_cf"] == 12.0

    def test_by_stage_with_mock_welch(self):
        freqs, psd = _synthetic_psd()
        fake_stage = {
            "N2": (freqs, psd, np.zeros_like(psd), 5),
            "W": (freqs, psd * 0.5, np.zeros_like(psd), 3),
        }
        signal = np.random.randn(1000)
        with patch.object(spa, "compute_stage_psd_welch_for_array", return_value=fake_stage):
            with patch.object(spa, "fit_specparam", side_effect=lambda *a, **k: {
                "ok": True,
                "aperiodic_offset": 0.0,
                "aperiodic_exponent": 1.0,
                "peaks": [],
                "r_squared": 0.8,
                "error": 0.1,
                "freqs": freqs,
                "psd": psd,
                "modeled_spectrum": psd,
                "warning": "",
            }):
                out = spa.fit_specparam_by_stage(
                    signal, 100.0, pd.DataFrame({"time": [0], "stage": ["N2"]}),
                    stages=["N2", "W"],
                )
        assert "N2" in out and out["N2"]["ok"]
        assert out["N2"]["n_epochs"] == 5
