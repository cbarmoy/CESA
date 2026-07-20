"""Tests for shared PSG summary metrics."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_metrics import compute_psg_summary, export_psg_summary_csv


def _sample_df() -> pd.DataFrame:
    stages = ["W", "W", "N1", "N2", "N2", "N3", "R", "R", "W"]
    return pd.DataFrame({
        "time": [i * 30.0 for i in range(len(stages))],
        "stage": stages,
    })


class TestPsgSummary:
    def test_basic_metrics(self):
        summary = compute_psg_summary(_sample_df(), epoch_seconds=30.0)
        cm = summary.clinical
        assert cm.n_epochs == 9
        assert cm.total_sleep_time_min > 0
        assert len(summary.episodes) >= 4

    def test_night_window_filter(self):
        summary = compute_psg_summary(
            _sample_df(),
            epoch_seconds=30.0,
            night_start_s=60.0,
            night_end_s=210.0,
        )
        assert summary.night_start_s == 60.0
        assert summary.clinical.n_epochs < 9

    def test_tib_uses_night_window_duration(self):
        df = pd.DataFrame({
            "time": [0.0, 30.0, 60.0, 90.0],
            "stage": ["W", "W", "N2", "N2"],
        })
        summary = compute_psg_summary(
            df,
            epoch_seconds=30.0,
            night_start_s=0.0,
            night_end_s=600.0,
        )
        assert summary.clinical.time_in_bed_min == pytest.approx(10.0)
        assert summary.clinical.total_sleep_time_min == pytest.approx(1.0)
        assert summary.clinical.time_in_bed_min > summary.clinical.total_sleep_time_min
        assert summary.clinical.sleep_efficiency_pct == pytest.approx(10.0)

    def test_tib_span_without_night_window(self):
        df = pd.DataFrame({
            "time": [0.0, 30.0, 60.0, 90.0],
            "stage": ["W", "N2", "N2", "N2"],
        })
        summary = compute_psg_summary(df, epoch_seconds=30.0)
        assert summary.clinical.time_in_bed_min == pytest.approx(2.0)
        assert summary.clinical.total_sleep_time_min == pytest.approx(1.5)

    def test_export_csv(self, tmp_path):
        summary = compute_psg_summary(_sample_df(), epoch_seconds=30.0)
        out = tmp_path / "psg.csv"
        export_psg_summary_csv(summary, str(out))
        text = out.read_text(encoding="utf-8")
        assert "TIB_min" in text
        assert "start_min" in text

    def test_export_csv_with_annotations(self, tmp_path):
        summary = compute_psg_summary(_sample_df(), epoch_seconds=30.0)
        annotations = [{
            "id": "abc",
            "onset": 120.0,
            "duration": 15.0,
            "type": "arousal",
            "label": "Arousal test",
            "channel": "C3",
            "source": "manual",
        }]
        out = tmp_path / "psg_ann.csv"
        export_psg_summary_csv(summary, str(out), annotations=annotations)
        text = out.read_text(encoding="utf-8")
        assert "arousal" in text
        assert "Arousal test" in text
        assert "2.00" in text  # onset_min
