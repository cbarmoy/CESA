"""Tests for CESA.sleep_pipeline.rules_aasm."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.contracts import StageLabel
from CESA.sleep_pipeline.rules_aasm import (
    DEFAULT_THRESHOLDS,
    _classify_epoch,
    add_relative_eog_activity,
    eog_activity_baseline,
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


class TestMontageWithoutEmg:
    """An absent channel must not veto a rule.

    On a montage without a chin EMG the features report ``emg_rms = 0``; read as
    a measurement, that made Wake unreachable and atonia always true.
    """

    def _feats(self, **over):
        feats = {
            "relpow_alpha": 0.05, "relpow_delta": 0.10, "relpow_theta": 0.15,
            "relpow_sigma": 0.02, "emg_rms": 0.0, "eog_rem_activity": 0.0,
            "ratio_delta_beta": 1.0, "is_artifact": 0.0,
            "has_eog": 1.0, "has_emg": 0.0,
        }
        feats.update(over)
        return feats

    def test_alpha_plus_eye_movements_is_rem_without_an_emg(self):
        # Without chin tone, alpha is ambiguous: the EOG decides.
        stage, _conf, reason = _classify_epoch(
            self._feats(relpow_alpha=0.22, eog_rem_activity=100.0),
            DEFAULT_THRESHOLDS,
        )
        assert stage == StageLabel.R
        assert "sans_emg" in reason

    def test_alpha_with_quiet_eyes_is_still_wake_without_an_emg(self):
        stage, _conf, reason = _classify_epoch(
            self._feats(relpow_alpha=0.30), DEFAULT_THRESHOLDS,
        )
        assert stage == StageLabel.W
        assert "sans_emg" in reason

    def test_alpha_with_slow_waves_is_not_called_wake(self):
        stage, _conf, _reason = _classify_epoch(
            self._feats(relpow_alpha=0.30, relpow_delta=0.40, ratio_delta_beta=8.0),
            DEFAULT_THRESHOLDS,
        )
        assert stage == StageLabel.N3

    def test_rem_still_detected_from_the_eog_alone(self):
        stage, conf, _reason = _classify_epoch(
            self._feats(eog_rem_activity=100.0), DEFAULT_THRESHOLDS,
        )
        assert stage == StageLabel.R
        assert conf < 0.75

    def test_no_rem_without_an_eog_either(self):
        stage, _conf, _reason = _classify_epoch(
            self._feats(has_eog=0.0, eog_rem_activity=0.0, relpow_theta=0.30),
            DEFAULT_THRESHOLDS,
        )
        assert stage != StageLabel.R

    def test_emg_rules_still_apply_when_an_emg_exists(self):
        stage, conf, _reason = _classify_epoch(
            self._feats(has_emg=1.0, relpow_alpha=0.30, emg_rms=15.0),
            DEFAULT_THRESHOLDS,
        )
        assert stage == StageLabel.W
        assert conf == 0.85

    def test_a_quiet_emg_does_not_give_a_confident_wake(self):
        # Alpha alone still lands on Wake through the dominant-band fallback,
        # but must not claim the confidence of a rule that checked the tone.
        stage, conf, reason = _classify_epoch(
            self._feats(has_emg=1.0, relpow_alpha=0.30, emg_rms=1.0),
            DEFAULT_THRESHOLDS,
        )
        assert stage == StageLabel.W
        assert "fallback" in reason
        assert conf < 0.6

    def test_features_record_which_modalities_were_measured(self):
        import numpy as np

        from CESA.sleep_pipeline.features import extract_all_features
        from CESA.sleep_pipeline.preprocessing import EpochedSignals

        rng = np.random.default_rng(0)
        eeg = rng.normal(0, 20, size=(2, 3000))
        epoched = EpochedSignals(
            eeg_epochs=eeg, eog_epochs=None, emg_epochs=None,
            sfreq=100.0, epoch_duration_s=30.0, n_epochs=2,
            eeg_channel="C4-M1", eog_channel=None, emg_channel=None,
        )
        feats = extract_all_features(epoched)
        assert feats[0]["has_eog"] == 0.0
        assert feats[0]["has_emg"] == 0.0


class TestNightRelativeEogActivity:
    """REM eye movements judged against the recording's own quiet baseline.

    The absolute variance of the EOG derivative scales with the amplifier range
    and shrinks as the sampling rate rises, so a fixed threshold cannot transfer
    between recordings.
    """

    def _night(self, activities, **over):
        feats = []
        for value in activities:
            f = {
                "relpow_alpha": 0.05, "relpow_delta": 0.10, "relpow_theta": 0.15,
                "relpow_sigma": 0.02, "emg_rms": 0.0, "ratio_delta_beta": 1.0,
                "is_artifact": 0.0, "has_eog": 1.0, "has_emg": 0.0,
                "eog_rem_activity": float(value),
            }
            f.update(over)
            feats.append(f)
        return feats

    def test_baseline_is_the_median_of_the_night(self):
        assert eog_activity_baseline(self._night([2.0] * 30)) == 2.0

    def test_no_baseline_from_too_few_epochs(self):
        assert eog_activity_baseline(self._night([2.0] * 5)) is None

    def test_no_baseline_from_a_flat_eog(self):
        assert eog_activity_baseline(self._night([0.0] * 30)) is None

    def test_no_baseline_without_an_eog(self):
        assert eog_activity_baseline(self._night([2.0] * 30, has_eog=0.0)) is None

    def test_activity_is_expressed_in_baseline_units(self):
        feats = add_relative_eog_activity(self._night([2.0] * 29 + [10.0]))
        assert feats[0]["eog_rem_relative"] == 1.0
        assert feats[-1]["eog_rem_relative"] == 5.0

    def test_rem_detected_far_below_the_absolute_threshold(self):
        # Activity of 6 uV^2 would never reach the absolute threshold of 50,
        # yet it is three times this recording's baseline.
        result = score_rule_based(
            self._night([2.0] * 29 + [6.0]), apply_smoothing=False,
        )
        assert result.epochs[-1].stage == StageLabel.R
        assert result.epochs[0].stage != StageLabel.R
        assert "baseline" in result.epochs[-1].decision_reason

    def test_criterion_is_invariant_to_the_recording_scale(self):
        quiet = [2.0] * 29 + [6.0]
        stages = [
            [ep.stage for ep in score_rule_based(
                self._night([v * scale for v in quiet]), apply_smoothing=False,
            ).epochs]
            for scale in (1.0, 1000.0, 0.001)
        ]
        assert stages[0] == stages[1] == stages[2]

    def test_a_uniformly_quiet_night_yields_no_rem(self):
        result = score_rule_based(self._night([2.0] * 30), apply_smoothing=False)
        assert all(ep.stage != StageLabel.R for ep in result.epochs)

    def test_absolute_threshold_still_used_when_no_baseline_exists(self):
        result = score_rule_based(self._night([100.0] * 3), apply_smoothing=False)
        assert all(ep.stage == StageLabel.R for ep in result.epochs)

    def test_sparse_rems_are_detected_by_count_not_variance(self):
        # A few REMs barely move the 30 s derivative variance, which is why
        # F1 REM stayed at 0 on a real night after the variance-only criterion.
        feats = self._night([2.0] * 29 + [2.4])
        for f in feats[:-1]:
            f["eog_rem_count"] = 1.0
        feats[-1]["eog_rem_count"] = 8.0
        result = score_rule_based(feats, apply_smoothing=False)
        assert result.epochs[-1].stage == StageLabel.R
        assert "rem_count" in result.epochs[-1].decision_reason
        assert all(ep.stage != StageLabel.R for ep in result.epochs[:-1])

    def test_high_alpha_does_not_steal_rem_when_the_eog_stands_out(self):
        feats = self._night([2.0] * 29 + [8.0], relpow_alpha=0.22)
        result = score_rule_based(feats, apply_smoothing=False)
        assert result.epochs[-1].stage == StageLabel.R
        assert result.epochs[0].stage == StageLabel.W


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
