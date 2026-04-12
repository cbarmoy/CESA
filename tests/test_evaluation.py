"""Tests for CESA.sleep_pipeline.evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.contracts import Epoch, ScoringResult, StageLabel
from CESA.sleep_pipeline.evaluation import ComparisonReport, _cohen_kappa, compare


def _make_scoring(stages_str: list[str], epoch_dur: float = 30.0) -> ScoringResult:
    """Helper: create a ScoringResult from a list of stage strings."""
    epochs = [
        Epoch(index=i, start_s=i * epoch_dur, stage=StageLabel.from_string(s), confidence=0.9)
        for i, s in enumerate(stages_str)
    ]
    return ScoringResult(epochs=epochs, epoch_duration_s=epoch_dur)


class TestCohenKappa:
    def test_perfect_agreement(self):
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        k = _cohen_kappa(y, y, list(range(5)))
        assert abs(k - 1.0) < 1e-10

    def test_no_agreement(self):
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        k = _cohen_kappa(y_true, y_pred, list(range(5)))
        assert k < 0.01

    def test_partial_agreement(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 2, 4])
        k = _cohen_kappa(y_true, y_pred, list(range(5)))
        assert 0.3 < k < 1.0

    def test_empty(self):
        assert _cohen_kappa(np.array([]), np.array([]), []) == 0.0


class TestCompare:
    def test_identical_scoring(self):
        stages = ["W", "N1", "N2", "N3", "R", "W", "N2", "N3", "R", "W"]
        ref = _make_scoring(stages)
        pred = _make_scoring(stages)
        report = compare(ref, pred)
        assert report.accuracy == 1.0
        assert abs(report.cohen_kappa - 1.0) < 1e-10
        assert report.n_epochs == 10

    def test_completely_wrong(self):
        ref = _make_scoring(["W", "W", "W", "W", "W"])
        pred = _make_scoring(["N1", "N1", "N1", "N1", "N1"])
        report = compare(ref, pred)
        assert report.accuracy == 0.0
        assert report.cohen_kappa <= 0.0

    def test_exclude_u(self):
        ref = _make_scoring(["U", "W", "N1", "U", "N2"])
        pred = _make_scoring(["W", "W", "N1", "N3", "N2"])
        report = compare(ref, pred, exclude_u=True)
        # Only epochs 1, 2, 4 should be compared (U excluded from ref)
        assert report.n_epochs == 3
        assert report.accuracy == 1.0

    def test_no_overlap(self):
        ref = ScoringResult(
            epochs=[Epoch(index=0, start_s=0.0, stage=StageLabel.W)],
            epoch_duration_s=30.0,
        )
        pred = ScoringResult(
            epochs=[Epoch(index=0, start_s=1000.0, stage=StageLabel.N1)],
            epoch_duration_s=30.0,
        )
        report = compare(ref, pred)
        assert report.n_epochs == 0

    def test_per_stage_metrics(self):
        ref = _make_scoring(["W", "N2", "N2", "N3", "R"])
        pred = _make_scoring(["W", "N2", "N3", "N3", "R"])
        report = compare(ref, pred)
        assert report.per_stage["W"].recall == 1.0
        assert report.per_stage["N3"].recall == 1.0
        # N2: 1 correct, 1 misclassified as N3 -> recall = 0.5
        assert abs(report.per_stage["N2"].recall - 0.5) < 1e-10

    def test_summary_text(self):
        ref = _make_scoring(["W", "N1", "N2"])
        pred = _make_scoring(["W", "N1", "N2"])
        report = compare(ref, pred)
        text = report.summary_text()
        assert "Accuracy" in text
        assert "Cohen" in text

    def test_to_dict(self):
        ref = _make_scoring(["W", "N1"])
        pred = _make_scoring(["W", "N1"])
        report = compare(ref, pred)
        d = report.to_dict()
        assert d["accuracy"] == 1.0
        assert "per_stage" in d
