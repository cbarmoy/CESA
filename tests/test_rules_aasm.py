"""Tests for CESA.sleep_pipeline.rules_aasm."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.contracts import StageLabel
from CESA.sleep_pipeline.rules_aasm import (
    DEFAULT_THRESHOLDS,
    _classify_epoch,
    score_rule_based,
    smooth_stages,
)


class TestClassifyEpoch:
    def test_wake_alpha_high_emg(self):
        feats = {
            "relpow_alpha": 0.30,
            "relpow_delta": 0.05,
            "relpow_theta": 0.10,
            "relpow_sigma": 0.02,
            "emg_rms": 15.0,
            "eog_rem_activity": 5.0,
            "ratio_delta_beta": 0.5,
            "is_artifact": 0.0,
        }
        stage, conf, reason = _classify_epoch(feats, DEFAULT_THRESHOLDS)
        assert stage == StageLabel.W
        assert conf > 0.5
        assert "alpha" in reason

    def test_n3_high_delta(self):
        feats = {
            "relpow_alpha": 0.02,
            "relpow_delta": 0.45,
            "relpow_theta": 0.15,
            "relpow_sigma": 0.01,
            "emg_rms": 3.0,
            "eog_rem_activity": 2.0,
            "ratio_delta_beta": 8.0,
            "is_artifact": 0.0,
        }
        stage, conf, reason = _classify_epoch(feats, DEFAULT_THRESHOLDS)
        assert stage == StageLabel.N3
        assert "delta" in reason

    def test_rem_eog_atonia(self):
        feats = {
            "relpow_alpha": 0.05,
            "relpow_delta": 0.10,
            "relpow_theta": 0.20,
            "relpow_sigma": 0.02,
            "emg_rms": 2.0,
            "eog_rem_activity": 100.0,
            "ratio_delta_beta": 1.0,
            "is_artifact": 0.0,
        }
        stage, conf, reason = _classify_epoch(feats, DEFAULT_THRESHOLDS)
        assert stage == StageLabel.R
        assert "rem_eog" in reason

    def test_n2_sigma(self):
        feats = {
            "relpow_alpha": 0.05,
            "relpow_delta": 0.12,
            "relpow_theta": 0.15,
            "relpow_sigma": 0.08,
            "emg_rms": 4.0,
            "eog_rem_activity": 5.0,
            "ratio_delta_beta": 1.5,
            "is_artifact": 0.0,
        }
        stage, conf, reason = _classify_epoch(feats, DEFAULT_THRESHOLDS)
        assert stage == StageLabel.N2
        assert "sigma" in reason

    def test_artifact_rejected(self):
        feats = {"is_artifact": 1.0}
        stage, conf, reason = _classify_epoch(feats, DEFAULT_THRESHOLDS)
        assert stage == StageLabel.U
        assert "artifact" in reason


class TestSmoothing:
    def test_impossible_w_to_n3_corrected(self):
        stages = [StageLabel.W, StageLabel.N3]
        confs = [0.8, 0.8]
        smoothed = smooth_stages(stages, confs)
        assert smoothed[1] != StageLabel.N3
        assert smoothed[1] in (StageLabel.N1, StageLabel.N2)

    def test_allowed_transition_unchanged(self):
        stages = [StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.N3]
        confs = [0.8] * 4
        smoothed = smooth_stages(stages, confs)
        assert smoothed == stages

    def test_single_epoch(self):
        stages = [StageLabel.N2]
        smoothed = smooth_stages(stages, [0.8])
        assert smoothed == [StageLabel.N2]


class TestScoreRuleBased:
    def test_basic_scoring(self):
        # Simulate 3 epochs with different dominant features
        features = [
            {
                "relpow_alpha": 0.30, "relpow_delta": 0.05, "relpow_theta": 0.10,
                "relpow_sigma": 0.02, "emg_rms": 15.0, "eog_rem_activity": 5.0,
                "ratio_delta_beta": 0.5, "is_artifact": 0.0,
            },
            {
                "relpow_alpha": 0.03, "relpow_delta": 0.50, "relpow_theta": 0.10,
                "relpow_sigma": 0.01, "emg_rms": 2.0, "eog_rem_activity": 2.0,
                "ratio_delta_beta": 10.0, "is_artifact": 0.0,
            },
            {
                "relpow_alpha": 0.04, "relpow_delta": 0.08, "relpow_theta": 0.15,
                "relpow_sigma": 0.02, "emg_rms": 1.5, "eog_rem_activity": 120.0,
                "ratio_delta_beta": 0.8, "is_artifact": 0.0,
            },
        ]
        result = score_rule_based(features, apply_smoothing=False)
        assert len(result.epochs) == 3
        assert result.epochs[0].stage == StageLabel.W
        assert result.epochs[1].stage == StageLabel.N3
        assert result.epochs[2].stage == StageLabel.R
        assert result.backend == "aasm_rules"

    def test_empty_features(self):
        result = score_rule_based([])
        assert len(result.epochs) == 0
