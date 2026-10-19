"""Tests for the auto-vs-manual scoring comparison.

Regression guard for the bug where importing a manual scoring overwrote the
automatic one with a copy of itself, yielding a meaningless 100 % agreement.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.app_controller import AppController
from CESA.scoring_comparison import (
    cohens_kappa,
    confusion_matrix,
    kappa_label,
    macro_f1,
    present_stages,
    stage_metrics,
)
from CESA.scoring_io import (
    export_scoring_csv,
    normalize_stage_label,
    normalize_stage_series,
)


def _df(stages, epoch_seconds=30.0, start=0.0):
    times = [start + i * epoch_seconds for i in range(len(stages))]
    return pd.DataFrame({"time": times, "stage": list(stages)})


def _controller(auto=None, manual=None, epoch_seconds=30.0):
    ctrl = AppController()
    if auto is not None:
        ctrl.auto_scoring_data = auto
        ctrl.auto_scoring_epoch_seconds = epoch_seconds
        ctrl.auto_scoring_method = "yasa"
    if manual is not None:
        ctrl.manual_scoring_data = manual
        ctrl.manual_scoring_epoch_seconds = epoch_seconds
        ctrl.manual_scoring_source = "test"
    ctrl.scoring_epoch_duration = epoch_seconds
    return ctrl


class TestStageNormalization:
    def test_common_aliases(self):
        assert normalize_stage_label("REM") == "R"
        assert normalize_stage_label("Wake") == "W"
        assert normalize_stage_label("Éveil") == "W"
        assert normalize_stage_label("Stade 2") == "N2"
        assert normalize_stage_label("artefact") == "U"
        assert normalize_stage_label(None) == "U"

    def test_numeric_codes(self):
        assert normalize_stage_label(0) == "W"
        assert normalize_stage_label(1) == "N1"
        assert normalize_stage_label(2) == "N2"
        assert normalize_stage_label(3) == "N3"
        assert normalize_stage_label("4") == "N3"
        assert normalize_stage_label(5) == "R"

    def test_series_helper(self):
        out = normalize_stage_series(pd.Series(["REM", "wake", 2]))
        assert list(out) == ["R", "W", "N2"]


class TestCompareScoring:
    def test_basic_agreement_and_metrics(self):
        ctrl = _controller(
            auto=_df(["W", "N1", "N2", "N2", "R"]),
            manual=_df(["W", "W", "N2", "N3", "R"]),
        )
        merged = ctrl.compare_scoring()
        assert merged is not None
        assert len(merged) == 5
        assert int(merged["match"].sum()) == 3
        assert merged["match"].mean() == pytest.approx(0.6)

    def test_labels_are_normalized_before_comparison(self):
        """A manual 'REM' must not count as a disagreement with an auto 'R'."""
        ctrl = _controller(
            auto=_df(["W", "R", "N3"]),
            manual=_df(["Eveil", "REM", "Stade 3"]),
        )
        merged = ctrl.compare_scoring()
        assert list(merged["manual"]) == ["W", "R", "N3"]
        assert bool(merged["match"].all())

    def test_identical_hypnograms_are_flagged(self):
        stages = ["W", "N1", "N2", "N3", "R"]
        ctrl = _controller(auto=_df(stages), manual=_df(stages))
        merged = ctrl.compare_scoring()
        diag = merged.attrs["diagnostics"]
        assert merged["match"].mean() == 1.0
        assert diag["identical"] is True
        assert diag["same_object"] is False

    def test_same_object_is_flagged(self):
        shared = _df(["W", "N2", "N3"])
        ctrl = _controller(auto=shared, manual=shared)
        diag = ctrl.compare_scoring().attrs["diagnostics"]
        assert diag["same_object"] is True
        assert diag["identical"] is True

    def test_distinct_scorings_are_not_flagged(self):
        ctrl = _controller(
            auto=_df(["W", "N1", "N2"]),
            manual=_df(["W", "N2", "N2"]),
        )
        diag = ctrl.compare_scoring().attrs["diagnostics"]
        assert diag["identical"] is False

    def test_unscored_epochs_excluded_by_default(self):
        ctrl = _controller(
            auto=_df(["W", "U", "N2", "N2"]),
            manual=_df(["W", "N1", "U", "N2"]),
        )
        merged = ctrl.compare_scoring()
        diag = merged.attrs["diagnostics"]
        assert len(merged) == 2
        assert diag["n_excluded_u"] == 2
        assert merged["match"].mean() == 1.0

    def test_unscored_epochs_can_be_kept(self):
        ctrl = _controller(
            auto=_df(["W", "U", "N2"]),
            manual=_df(["W", "N1", "N2"]),
        )
        merged = ctrl.compare_scoring(exclude_unscored=False)
        assert len(merged) == 3
        assert merged.attrs["diagnostics"]["n_excluded_u"] == 0

    def test_partial_overlap_reports_coverage(self):
        ctrl = _controller(
            auto=_df(["W", "N1", "N2", "N3", "R"]),
            manual=_df(["N2", "N3"], start=60.0),
        )
        merged = ctrl.compare_scoring()
        diag = merged.attrs["diagnostics"]
        assert diag["n_auto"] == 5
        assert diag["n_manual"] == 2
        assert diag["n_common"] == 2
        assert merged["match"].mean() == 1.0

    def test_different_epoch_lengths_use_majority_vote(self):
        """A 30 s auto scoring compared with a 60 s manual one."""
        ctrl = AppController()
        ctrl.auto_scoring_data = _df(["N2", "N2", "N3", "R"], epoch_seconds=30.0)
        ctrl.auto_scoring_epoch_seconds = 30.0
        ctrl.manual_scoring_data = _df(["N2", "N3"], epoch_seconds=60.0)
        ctrl.manual_scoring_epoch_seconds = 60.0
        merged = ctrl.compare_scoring()
        diag = merged.attrs["diagnostics"]
        assert diag["comparison_epoch_seconds"] == 60.0
        assert len(merged) == 2
        # First 60 s window is N2/N2 -> N2 ; second is N3/R -> majority keeps one.
        assert merged.loc[0, "auto"] == "N2"
        assert bool(merged.loc[0, "match"])

    def test_duplicate_epochs_are_collapsed(self):
        auto = pd.DataFrame({
            "time": [0.0, 0.0, 30.0],
            "stage": ["N2", "N2", "N3"],
        })
        ctrl = _controller(auto=auto, manual=_df(["N2", "N3"]))
        merged = ctrl.compare_scoring()
        assert len(merged) == 2
        assert bool(merged["match"].all())

    def test_returns_none_without_manual_scoring(self):
        ctrl = _controller(auto=_df(["W", "N2"]))
        assert ctrl.compare_scoring() is None

    def test_returns_none_without_auto_scoring(self):
        ctrl = _controller(manual=_df(["W", "N2"]))
        ctrl.active_scoring_source = "manual"
        assert ctrl.compare_scoring() is None

    def test_night_window_restricts_comparison(self):
        ctrl = _controller(
            auto=_df(["W", "N1", "N2", "N3", "R"]),
            manual=_df(["W", "W", "N2", "N3", "R"]),
        )
        ctrl.set_night_window(60.0, 120.0)
        merged = ctrl.compare_scoring()
        assert len(merged) == 3
        assert bool(merged["match"].all())


class TestStoreSeparation:
    def test_import_does_not_destroy_auto_scoring(self):
        """The 100 % bug: importing a manual scoring used to overwrite the auto one."""
        ctrl = AppController()
        auto = _df(["W", "N1", "N2", "N2", "R"])
        ctrl.auto_scoring_data = auto.copy()
        ctrl.auto_scoring_epoch_seconds = 30.0
        ctrl.auto_scoring_method = "yasa"
        ctrl.sleep_scoring_data = auto.copy()
        ctrl.active_scoring_source = "auto"
        ctrl.total_duration_s = 150.0

        manual = _df(["W", "W", "N2", "N3", "R"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.csv"
            export_scoring_csv(manual, path, epoch_seconds=30.0)
            assert ctrl.import_scoring_file(str(path)) is True

        assert ctrl.auto_scoring_data is not None
        assert list(ctrl.auto_scoring_data["stage"]) == ["W", "N1", "N2", "N2", "R"]
        assert ctrl.active_scoring_source == "manual"

        merged = ctrl.compare_scoring()
        assert merged.attrs["diagnostics"]["identical"] is False
        assert merged["match"].mean() < 1.0

    def test_set_active_scoring_targets_manual_store(self):
        ctrl = AppController()
        ctrl.auto_scoring_data = _df(["W", "N1", "N2"])
        ctrl.auto_scoring_epoch_seconds = 30.0
        edited = _df(["W", "W", "N2"])
        ctrl.set_active_scoring(edited, "manual", epoch_seconds=30.0)
        assert list(ctrl.manual_scoring_data["stage"]) == ["W", "W", "N2"]
        assert list(ctrl.auto_scoring_data["stage"]) == ["W", "N1", "N2"]
        assert ctrl.active_scoring_source == "manual"
        assert list(ctrl.sleep_scoring_data["stage"]) == ["W", "W", "N2"]

    def test_set_active_scoring_targets_auto_store(self):
        ctrl = AppController()
        ctrl.manual_scoring_data = _df(["W", "N1", "N2"])
        corrected = _df(["W", "N2", "N2"])
        ctrl.set_active_scoring(corrected, "auto", epoch_seconds=30.0)
        assert list(ctrl.auto_scoring_data["stage"]) == ["W", "N2", "N2"]
        assert list(ctrl.manual_scoring_data["stage"]) == ["W", "N1", "N2"]

    def test_active_only_target_leaves_both_stores_untouched(self):
        ctrl = AppController()
        ctrl.auto_scoring_data = _df(["W", "N1"])
        ctrl.manual_scoring_data = _df(["W", "N2"])
        ctrl.set_active_scoring(_df(["N3", "N3"]), "active", epoch_seconds=30.0)
        assert list(ctrl.auto_scoring_data["stage"]) == ["W", "N1"]
        assert list(ctrl.manual_scoring_data["stage"]) == ["W", "N2"]
        assert list(ctrl.sleep_scoring_data["stage"]) == ["N3", "N3"]

    def test_seed_manual_from_auto_is_explicit(self):
        ctrl = AppController()
        assert ctrl.copy_auto_scoring_to_manual() is False
        ctrl.auto_scoring_data = _df(["W", "N1", "N2"])
        ctrl.auto_scoring_epoch_seconds = 30.0
        assert ctrl.copy_auto_scoring_to_manual() is True
        # Seeding on purpose yields 100 %: the dialog warns about it.
        assert ctrl.compare_scoring().attrs["diagnostics"]["identical"] is True


class _FakeRaw:
    """Minimal stand-in exposing only what channel resolution needs."""

    def __init__(self, ch_names):
        self.ch_names = list(ch_names)
        self.info = {"sfreq": 256.0}


class TestAutoScoringWarnings:
    def _ctrl(self, ch_names, emg_candidates):
        ctrl = AppController()
        ctrl.raw = _FakeRaw(ch_names)
        ctrl.yasa_emg_candidates = list(emg_candidates)
        return ctrl

    def test_leg_emg_configured_as_staging_emg_is_flagged(self):
        ctrl = self._ctrl(
            ["C4-M1", "E1-M2", "Jambe droite", "Jambe gauche"],
            ["Jambe droite"],
        )
        warnings = ctrl.auto_scoring_warnings("yasa")
        assert warnings
        assert "EMG de membre" in warnings[0]

    def test_chin_alternative_is_pointed_out_when_available(self):
        ctrl = self._ctrl(
            ["C4-M1", "Chin1-Chin2", "Jambe droite"], ["Jambe droite"],
        )
        warnings = " ".join(ctrl.auto_scoring_warnings("yasa"))
        assert "Chin1-Chin2" in warnings

    def test_mastoid_reference_configured_as_emg_is_flagged(self):
        ctrl = self._ctrl(["C3-M2", "E1-M2", "M1", "M2"], ["M1"])
        warnings = ctrl.auto_scoring_warnings("yasa")
        assert warnings
        assert "n'est pas un EMG" in warnings[0]

    def test_unused_12f_chin_is_flagged(self):
        ctrl = self._ctrl(
            ["C4-M1", "1", "1-2", "1-2 (Imp)", "1-F", "2", "2-F"],
            [],
        )
        warnings = " ".join(ctrl.auto_scoring_warnings("yasa"))
        assert "1-2" in warnings
        assert "n'est pas utilise" in warnings

    def test_default_candidates_resolve_bipolar_12_first(self):
        ctrl = AppController()
        ctrl.raw = _FakeRaw(["C4-M1", "1", "1 Impédance", "1-2", "1-F", "2"])
        assert ctrl.resolved_scoring_channel(ctrl.yasa_emg_candidates) == "1-2"

    def test_no_emg_warning_for_a_backend_that_ignores_the_emg(self):
        # U-Sleep never reads the EMG, so a questionable EMG is harmless there.
        ctrl = self._ctrl(["C4-M1", "E1-M2", "Jambe droite"], ["Jambe droite"])
        assert ctrl.auto_scoring_warnings("usleep") == []


    def test_no_chin_channel_means_no_suggested_alternative(self):
        # Montage without any submental EMG: the only advice is an empty list.
        ctrl = self._ctrl(["C3-M2", "E1-M2", "M1", "Left Leg"], ["M1"])
        warnings = ctrl.auto_scoring_warnings("yasa")
        assert len(warnings) == 1
        assert "Videz la liste EMG" in warnings[0]

    def test_no_warning_for_a_chin_emg(self):
        ctrl = self._ctrl(["C4-M1", "Chin1-Chin2"], ["Chin1-Chin2"])
        assert ctrl.auto_scoring_warnings("yasa") == []

    def test_no_warning_when_no_emg_resolves(self):
        ctrl = self._ctrl(["C4-M1", "E1-M2"], ["Chin1-Chin2", "Menton"])
        assert ctrl.auto_scoring_warnings("yasa") == []

    def test_backends_ignoring_emg_are_not_warned(self):
        ctrl = self._ctrl(["C4-M1", "Jambe droite"], ["Jambe droite"])
        assert ctrl.auto_scoring_warnings("aasm_rules") == []

    def test_no_recording_yields_no_warning(self):
        ctrl = AppController()
        assert ctrl.auto_scoring_warnings("yasa") == []

    def test_channel_resolution_follows_candidate_priority(self):
        ctrl = self._ctrl(["C4-M1", "Menton", "Chin1-Chin2"], [])
        assert ctrl.resolved_scoring_channel(["Chin1-Chin2", "Menton"]) == "Chin1-Chin2"
        assert ctrl.resolved_scoring_channel(["Absent", "Menton"]) == "Menton"
        assert ctrl.resolved_scoring_channel(["Absent"]) is None

    def test_channel_resolution_is_case_insensitive(self):
        ctrl = self._ctrl(["C4-M1", "Chin1-Chin2"], [])
        assert ctrl.resolved_scoring_channel(["chin1-chin2"]) == "Chin1-Chin2"


class TestPerBackendChannels:
    def _ctrl(self):
        ctrl = AppController()
        ctrl.raw = _FakeRaw(["C4-M1", "E1-M2", "Chin1-Chin2"])
        return ctrl

    def test_defaults_differ_between_backends(self):
        ctrl = self._ctrl()
        assert ctrl.channel_candidates_for("yasa")["emg"]
        assert ctrl.channel_candidates_for("usleep")["emg"] == []

    def test_customisation_is_remembered_per_backend(self):
        ctrl = self._ctrl()
        ctrl.set_channel_candidates_for("yasa", eeg=["C4-M1"])
        ctrl.set_channel_candidates_for("aasm_rules", eeg=["F4-M1"])
        assert ctrl.channel_candidates_for("yasa")["eeg"] == ["C4-M1"]
        assert ctrl.channel_candidates_for("aasm_rules")["eeg"] == ["F4-M1"]

    def test_applying_a_profile_updates_the_active_lists(self):
        ctrl = self._ctrl()
        ctrl.set_channel_candidates_for("yasa", eeg=["C3-M2"])
        ctrl.apply_channel_profile("usleep")
        assert ctrl.yasa_emg_candidates == []
        ctrl.apply_channel_profile("yasa")
        assert ctrl.yasa_eeg_candidates == ["C3-M2"]

    def test_directly_assigned_lists_still_win(self):
        # Long-standing behaviour: setting the list drives the next run.
        ctrl = self._ctrl()
        ctrl.yasa_eeg_candidates = ["F4-M1"]
        assert ctrl.effective_channel_candidates("yasa")["eeg"] == ["F4-M1"]

    def test_an_unused_modality_is_never_passed_to_the_backend(self):
        ctrl = self._ctrl()
        ctrl.yasa_emg_candidates = ["Chin1-Chin2"]
        assert ctrl.effective_channel_candidates("usleep")["emg"] == []

    def test_an_empty_list_means_no_channel_not_a_default(self):
        # Substituting a default here would make "score without EOG"
        # impossible, and would silently run the sweep's no-EOG configuration
        # with an EOG.
        ctrl = self._ctrl()
        ctrl.yasa_eog_candidates = []
        assert ctrl.effective_channel_candidates("yasa")["eog"] == []


def _merged_frame():
    return pd.DataFrame({
        "epoch": range(5),
        "auto": ["W", "N1", "N2", "N2", "R"],
        "manual": ["W", "W", "N2", "N3", "R"],
    })


class TestStageMetrics:
    def test_per_stage_metrics(self):
        merged = _merged_frame()
        metrics = stage_metrics(merged, ["W", "N1", "N2", "N3", "R"])
        assert metrics["W"]["support"] == 2
        assert metrics["W"]["tp"] == 1
        assert metrics["W"]["recall"] == pytest.approx(0.5)
        assert metrics["W"]["precision"] == pytest.approx(1.0)
        assert metrics["N1"]["support"] == 0
        assert metrics["N1"]["fp"] == 1
        assert metrics["N2"]["precision"] == pytest.approx(0.5)
        assert metrics["N2"]["recall"] == pytest.approx(1.0)
        assert metrics["R"]["f1"] == pytest.approx(1.0)

    def test_macro_f1_ignores_absent_stages(self):
        metrics = stage_metrics(_merged_frame(), ["W", "N1", "N2", "N3", "R"])
        value = macro_f1(metrics)
        assert 0.0 < value < 1.0

    def test_present_stages_follow_canonical_order(self):
        assert present_stages(_merged_frame()) == ["W", "N1", "N2", "N3", "R"]

    def test_confusion_matrix_rows_are_auto(self):
        stages = ["W", "N1", "N2", "N3", "R"]
        cm = confusion_matrix(_merged_frame(), stages)
        assert cm.sum() == 5
        # auto N1 vs manual W
        assert cm[stages.index("N1"), stages.index("W")] == 1
        # auto N2 vs manual N3
        assert cm[stages.index("N2"), stages.index("N3")] == 1

    def test_kappa_perfect_and_labels(self):
        assert cohens_kappa(["W", "N2", "R"], ["W", "N2", "R"]) == pytest.approx(1.0)
        assert kappa_label(0.85) == "excellent"
        assert kappa_label(0.5) == "modere"
        assert kappa_label(float("nan")) == "indetermine"
