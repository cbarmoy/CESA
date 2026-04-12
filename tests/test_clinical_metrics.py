"""Tests for clinical metrics and error analysis in evaluation.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.contracts import Epoch, ScoringResult, StageLabel
from CESA.sleep_pipeline.evaluation import (
    ClinicalMetrics,
    ErrorAnalysis,
    compute_clinical_metrics,
    error_analysis,
)


def _make_result(stages_str: list[str], epoch_dur: float = 30.0) -> ScoringResult:
    epochs = [
        Epoch(index=i, start_s=i * epoch_dur, stage=StageLabel.from_string(s), confidence=0.9)
        for i, s in enumerate(stages_str)
    ]
    return ScoringResult(epochs=epochs, epoch_duration_s=epoch_dur)


class TestClinicalMetrics:
    def test_basic_night(self):
        # W W N1 N2 N2 N3 N3 N2 R R W
        stages = ["W", "W", "N1", "N2", "N2", "N3", "N3", "N2", "R", "R", "W"]
        result = _make_result(stages)
        cm = compute_clinical_metrics(result)

        # TIB = 11 * 30 / 60 = 5.5 min
        assert abs(cm.time_in_bed_min - 5.5) < 0.01
        # TST = 8 sleep epochs (N1,N2,N2,N3,N3,N2,R,R) * 30 / 60 = 4.0 min
        assert abs(cm.total_sleep_time_min - 4.0) < 0.01
        # SE = 4.0/5.5 * 100
        assert abs(cm.sleep_efficiency_pct - (4.0 / 5.5 * 100)) < 0.1
        # SOL = 2 epochs * 30s / 60 = 1.0 min (first sleep at index 2)
        assert abs(cm.sleep_onset_latency_min - 1.0) < 0.01

    def test_all_wake(self):
        stages = ["W", "W", "W"]
        cm = compute_clinical_metrics(_make_result(stages))
        assert cm.total_sleep_time_min == 0.0
        assert cm.sleep_efficiency_pct == 0.0
        assert cm.sleep_onset_latency_min == cm.time_in_bed_min

    def test_all_sleep(self):
        stages = ["N2", "N2", "N3", "R"]
        cm = compute_clinical_metrics(_make_result(stages))
        assert cm.sleep_efficiency_pct == 100.0
        assert cm.sleep_onset_latency_min == 0.0
        assert cm.waso_min == 0.0

    def test_waso(self):
        # N2 W W N2 (WASO = 2 epochs)
        stages = ["N2", "W", "W", "N2"]
        cm = compute_clinical_metrics(_make_result(stages))
        assert abs(cm.waso_min - 1.0) < 0.01  # 2 * 30 / 60

    def test_rem_latency(self):
        # W N1 N2 N2 R (sleep onset at idx 1, REM at idx 4)
        stages = ["W", "N1", "N2", "N2", "R"]
        cm = compute_clinical_metrics(_make_result(stages))
        # REM latency = (4 - 1) * 30 / 60 = 1.5 min
        assert abs(cm.rem_latency_min - 1.5) < 0.01

    def test_stage_percentages(self):
        # 2 N2, 2 N3, 1 R = 5 sleep epochs
        stages = ["N2", "N2", "N3", "N3", "R"]
        cm = compute_clinical_metrics(_make_result(stages))
        assert abs(cm.n2_pct - 40.0) < 0.1
        assert abs(cm.n3_pct - 40.0) < 0.1
        assert abs(cm.rem_pct - 20.0) < 0.1

    def test_empty(self):
        cm = compute_clinical_metrics(ScoringResult())
        assert cm.total_sleep_time_min == 0.0

    def test_to_dict(self):
        stages = ["W", "N2", "R"]
        cm = compute_clinical_metrics(_make_result(stages))
        d = cm.to_dict()
        assert "total_sleep_time_min" in d
        assert "sleep_efficiency_pct" in d


class TestErrorAnalysis:
    def test_perfect_agreement(self):
        stages = [StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.N3, StageLabel.R]
        ea = error_analysis(stages, stages)
        assert ea.transition_accuracy == 1.0
        for cp in ea.confusion_pairs:
            assert cp.count == 0

    def test_n1_wake_confusion(self):
        true = [StageLabel.N1, StageLabel.N1, StageLabel.N1]
        pred = [StageLabel.W, StageLabel.W, StageLabel.N1]
        ea = error_analysis(true, pred)
        n1_w = next(cp for cp in ea.confusion_pairs if cp.true_stage == "N1" and cp.pred_stage == "W")
        assert n1_w.count == 2
        assert abs(n1_w.rate - 2 / 3) < 0.01

    def test_transition_detection(self):
        # Reference: W W N2 N2 (one transition at index 2)
        true = [StageLabel.W, StageLabel.W, StageLabel.N2, StageLabel.N2]
        # Prediction: W N1 N2 N2 (transitions at 1 and 2 -- detects the reference one)
        pred = [StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.N2]
        ea = error_analysis(true, pred)
        assert ea.n_transitions_ref == 1
        assert ea.transition_accuracy == 1.0

    def test_boundary_accuracy(self):
        true = [StageLabel.W, StageLabel.W, StageLabel.N2, StageLabel.N2]
        pred = [StageLabel.W, StageLabel.W, StageLabel.N2, StageLabel.N2]
        ea = error_analysis(true, pred)
        assert ea.boundary_accuracy == 1.0

    def test_empty(self):
        ea = error_analysis([], [])
        assert ea.transition_accuracy == 0.0

    def test_to_dict(self):
        true = [StageLabel.W, StageLabel.N2]
        pred = [StageLabel.W, StageLabel.N2]
        d = error_analysis(true, pred).to_dict()
        assert "confusion_pairs" in d
        assert "transition_accuracy" in d
