"""Round-trip tests for hypnogram import/export (CSV, JSON, clinical Excel)."""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.manual_scoring_service import ManualScoringService
from CESA.scoring_io import (
    export_scoring_csv,
    export_scoring_json,
    import_scoring_json,
    import_scoring_table,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "time": [0.0, 30.0, 60.0, 90.0, 120.0],
        "stage": ["W", "N1", "N2", "N2", "R"],
    })


class TestCesaNativeRoundtrip:
    def test_csv_roundtrip(self):
        df = _sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoring.csv"
            export_scoring_csv(df, path, epoch_seconds=30.0)
            result = ManualScoringService.import_file_path(
                str(path),
                absolute_start_datetime=None,
                recording_duration_s=150.0,
                default_epoch_seconds=30.0,
            )
        assert list(result.df["stage"]) == ["W", "N1", "N2", "N2", "R"]
        assert list(result.df["time"]) == [0.0, 30.0, 60.0, 90.0, 120.0]

    def test_csv_roundtrip_with_night_window(self):
        df = _sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoring.csv"
            export_scoring_csv(
                df,
                path,
                epoch_seconds=30.0,
                night_start_s=30.0,
                night_end_s=120.0,
            )
            result = ManualScoringService.import_file_path(
                str(path),
                absolute_start_datetime=None,
                recording_duration_s=150.0,
                default_epoch_seconds=30.0,
            )
        assert result.night_start_s == 30.0
        assert result.night_end_s == 120.0

    def test_json_roundtrip(self):
        df = _sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoring.json"
            export_scoring_json(df, path, epoch_seconds=30.0)
            loaded = import_scoring_json(path, epoch_seconds=30.0)
            result = ManualScoringService.validate(loaded)
        assert list(result["stage"]) == ["W", "N1", "N2", "N2", "R"]
        assert list(result["time"]) == [0.0, 30.0, 60.0, 90.0, 120.0]

    def test_json_roundtrip_with_night_window(self):
        df = _sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoring.json"
            export_scoring_json(
                df,
                path,
                epoch_seconds=30.0,
                night_start_s=0.0,
                night_end_s=90.0,
            )
            result = ManualScoringService.import_file_path(
                str(path),
                absolute_start_datetime=None,
                recording_duration_s=150.0,
                default_epoch_seconds=30.0,
            )
        assert result.night_start_s == 0.0
        assert result.night_end_s == 90.0

    def test_numeric_seconds_not_treated_as_datetime(self):
        """Regression: 0/30/60 must not be parsed as 1970 calendar dates."""
        df = pd.DataFrame({"time": [0, 30, 60, 90], "stage": ["W", "N1", "N2", "N3"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoring.csv"
            export_scoring_csv(df, path)
            result = ManualScoringService.import_file_path(
                str(path),
                absolute_start_datetime=pd.Timestamp("2024-02-01 23:00:00"),
                recording_duration_s=120.0,
                default_epoch_seconds=30.0,
            )
        assert list(result.df["time"]) == [0.0, 30.0, 60.0, 90.0]


class TestClinicalFormat:
    def test_stage_datetime_import(self):
        df = pd.DataFrame({
            "Stade": ["W", "N1", "N2"],
            "Heure": [
                "2024-02-01 23:58:30",
                "2024-02-01 23:59:00",
                "2024-02-01 23:59:30",
            ],
        })
        base = pd.Timestamp("2024-02-01 23:58:30")
        out = import_scoring_table(
            df,
            absolute_start_datetime=base,
            epoch_seconds=30.0,
            recording_duration_s=120.0,
        )
        assert len(out) >= 3
        assert out["stage"].iloc[0] == "W"
        assert float(out["time"].iloc[0]) == pytest.approx(0.0, abs=1.0)

    def test_import_scoring_table_cesa_native(self):
        df = _sample_df()
        out = import_scoring_table(df, epoch_seconds=30.0)
        assert list(out["stage"]) == ["W", "N1", "N2", "N2", "R"]
