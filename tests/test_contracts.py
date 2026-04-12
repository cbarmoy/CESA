"""Tests for CESA.sleep_pipeline.contracts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.contracts import (
    CLINICAL_STAGES,
    Epoch,
    ScoringResult,
    StageLabel,
)


class TestStageLabel:
    def test_from_string_standard(self):
        assert StageLabel.from_string("W") == StageLabel.W
        assert StageLabel.from_string("N1") == StageLabel.N1
        assert StageLabel.from_string("N2") == StageLabel.N2
        assert StageLabel.from_string("N3") == StageLabel.N3
        assert StageLabel.from_string("R") == StageLabel.R

    def test_from_string_aliases(self):
        assert StageLabel.from_string("Wake") == StageLabel.W
        assert StageLabel.from_string("EVEIL") == StageLabel.W
        assert StageLabel.from_string("éveil") == StageLabel.W
        assert StageLabel.from_string("REM") == StageLabel.R
        assert StageLabel.from_string("paradoxal") == StageLabel.R
        assert StageLabel.from_string("SWS") == StageLabel.N3
        assert StageLabel.from_string("N4") == StageLabel.N3

    def test_from_string_artifacts(self):
        assert StageLabel.from_string("artefact") == StageLabel.U
        assert StageLabel.from_string("movement") == StageLabel.U
        assert StageLabel.from_string("MT") == StageLabel.U

    def test_from_string_unknown(self):
        assert StageLabel.from_string("xyz") == StageLabel.U
        assert StageLabel.from_string("") == StageLabel.U

    def test_is_sleep(self):
        assert StageLabel.N1.is_sleep
        assert StageLabel.N2.is_sleep
        assert StageLabel.N3.is_sleep
        assert StageLabel.R.is_sleep
        assert not StageLabel.W.is_sleep
        assert not StageLabel.U.is_sleep

    def test_clinical_stages_count(self):
        assert len(CLINICAL_STAGES) == 5


class TestScoringResult:
    def _make_result(self, n: int = 10) -> ScoringResult:
        stages = [StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.N3, StageLabel.R]
        epochs = [
            Epoch(index=i, start_s=i * 30.0, stage=stages[i % 5], confidence=0.8)
            for i in range(n)
        ]
        return ScoringResult(epochs=epochs, epoch_duration_s=30.0, backend="test")

    def test_to_dataframe(self):
        result = self._make_result(5)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["time", "stage", "confidence", "reason"]
        assert len(df) == 5
        assert df["stage"].iloc[0] == "W"
        assert df["time"].iloc[2] == 60.0

    def test_from_dataframe_roundtrip(self):
        original = self._make_result(10)
        df = original.to_dataframe()
        restored = ScoringResult.from_dataframe(df)
        assert len(restored.epochs) == 10
        assert restored.epochs[0].stage == StageLabel.W
        assert restored.epochs[1].stage == StageLabel.N1

    def test_stages_property(self):
        result = self._make_result(5)
        assert result.stages == ["W", "N1", "N2", "N3", "R"]

    def test_stage_array(self):
        result = self._make_result(5)
        arr = result.stage_array()
        np.testing.assert_array_equal(arr, [0, 1, 2, 3, 4])

    def test_empty_result(self):
        result = ScoringResult()
        df = result.to_dataframe()
        assert len(df) == 0
