"""Tests for CESA.sleep_pipeline.explainability."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.contracts import Epoch, ScoringResult, StageLabel
from CESA.sleep_pipeline.explainability import (
    EpochExplanation,
    explain_epoch,
    rule_coverage_report,
)


class TestEpochExplanation:
    def test_to_dict(self):
        exp = EpochExplanation(
            epoch_index=0,
            predicted_stage="W",
            confidence=0.85,
            top_features=["relpow_alpha", "emg_rms"],
            rule_fired="alpha=0.30>=thr,emg=15.0",
            rule_margin=0.15,
            agreement_ml_rules=True,
        )
        d = exp.to_dict()
        assert d["epoch_index"] == 0
        assert d["predicted_stage"] == "W"
        assert len(d["top_features"]) == 2


class TestExplainEpoch:
    def test_with_rules_only(self):
        rule_epoch = Epoch(
            index=0, start_s=0.0, stage=StageLabel.W,
            confidence=0.85, decision_reason="alpha=0.30>=thr,emg=15.0",
        )
        exp = explain_epoch(
            epoch_features={"relpow_alpha": 0.3, "emg_rms": 15.0},
            rules_result_epoch=rule_epoch,
        )
        assert exp.predicted_stage == "W"
        assert exp.confidence == 0.85
        assert "alpha" in exp.rule_fired

    def test_without_model(self):
        exp = explain_epoch(
            epoch_features={"relpow_alpha": 0.3},
        )
        assert exp.predicted_stage == "U"
        assert exp.ml_shap_values == {}


class TestRuleCoverageReport:
    def test_counts_rules(self):
        epochs = [
            Epoch(index=0, start_s=0.0, stage=StageLabel.W, decision_reason="alpha_rule"),
            Epoch(index=1, start_s=30.0, stage=StageLabel.W, decision_reason="alpha_rule"),
            Epoch(index=2, start_s=60.0, stage=StageLabel.N3, decision_reason="delta_rule"),
            Epoch(index=3, start_s=90.0, stage=StageLabel.N2, decision_reason="sigma_rule|smoothed_from_N1"),
        ]
        result = ScoringResult(epochs=epochs)
        report = rule_coverage_report(result)
        assert report["alpha_rule"] == 2
        assert report["delta_rule"] == 1
        assert report["sigma_rule"] == 1

    def test_empty(self):
        report = rule_coverage_report(ScoringResult())
        assert report == {}

    def test_hmm_suffix_stripped(self):
        epochs = [
            Epoch(index=0, start_s=0.0, stage=StageLabel.R,
                  decision_reason="rem_detection|hmm_from_N1"),
        ]
        result = ScoringResult(epochs=epochs)
        report = rule_coverage_report(result)
        assert "rem_detection" in report


class TestTemporalContext:
    def test_add_temporal_context(self):
        from CESA.sleep_pipeline.ml_scorer import add_temporal_context
        features = [
            {"relpow_delta": 0.1 * i, "relpow_alpha": 0.3, "spectral_entropy": 0.5,
             "relpow_theta": 0.15, "relpow_sigma": 0.05, "relpow_beta": 0.1,
             "spindle_count": 0.0, "eog_rem_activity": 10.0, "emg_rms": 5.0}
            for i in range(10)
        ]
        augmented = add_temporal_context(features)
        assert len(augmented) == 10
        # Check that temporal features exist
        assert "relpow_delta_t-1" in augmented[5]
        assert "relpow_delta_t+1" in augmented[5]
        assert "night_fraction" in augmented[0]
        assert "delta_power_trend" in augmented[5]
        # Night fraction should be 0 at start, 1 at end
        assert augmented[0]["night_fraction"] == 0.0
        assert augmented[-1]["night_fraction"] == 1.0
        # Boundary padding
        assert augmented[0]["relpow_delta_t-2"] == 0.0
