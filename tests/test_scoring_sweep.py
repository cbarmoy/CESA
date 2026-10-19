"""Tests for the scoring-configuration sweep engine."""

from __future__ import annotations

import pandas as pd
import pytest

from CESA.app_controller import AppController
from CESA.scoring_sweep import (
    SweepConfig,
    SweepResult,
    apply_config,
    available_eeg,
    available_emg,
    available_eog,
    best_result,
    build_backend_comparison,
    build_default_configs,
    estimate_duration_s,
    rank_results,
    run_sweep,
    sweep_summary,
    usable_backends,
)

MONTAGE = [
    "C4-M1", "C3-M2", "F4-M1", "O2-M1", "E1-M2", "E2-M1",
    "Chin1-Chin2", "Left Leg", "M1", "ECG", "SpO2",
]


class _FakeRaw:
    def __init__(self, ch_names):
        self.ch_names = list(ch_names)
        self.info = {"sfreq": 200.0}


def _df(stages, epoch_seconds=30.0):
    return pd.DataFrame({
        "time": [i * epoch_seconds for i in range(len(stages))],
        "stage": list(stages),
    })


class _StubController(AppController):
    """Controller whose scoring is scripted instead of computed.

    ``outcomes`` maps the EEG channel of a configuration to the stages the
    backend should return, or to an Exception to simulate a failure.
    """

    def __init__(self, outcomes, manual):
        super().__init__()
        self.raw = _FakeRaw(MONTAGE)
        self._outcomes = outcomes
        self.runs: list = []
        self.manual_scoring_data = _df(manual)
        self.manual_scoring_epoch_seconds = 30.0
        self.manual_scoring_source = "test"
        self.scoring_epoch_duration = 30.0

    def run_auto_scoring(self, method=None, progress_callback=None):
        # Resolved the way the real controller does, so a configuration that
        # silently gains a channel it should not have shows up here.
        used = self.effective_channel_candidates(method)
        eeg = used["eeg"][0] if used["eeg"] else None
        self.runs.append({
            "eeg": eeg,
            "eog": list(used["eog"]),
            "emg": list(used["emg"]),
            "age": self.yasa_age,
            "male": self.yasa_male,
            "method": method,
        })
        outcome = self._outcomes.get(eeg)
        if outcome is None:
            self.last_auto_scoring_error = f"Pas de sortie simulee pour {eeg}"
            return False
        if isinstance(outcome, Exception):
            raise outcome
        self.auto_scoring_data = _df(outcome)
        self.auto_scoring_epoch_seconds = 30.0
        self.auto_scoring_method = method or "yasa"
        self.auto_scoring_params = {"backend": method, "duree_calcul_s": 12.0}
        self.sleep_scoring_data = self.auto_scoring_data.copy()
        self.active_scoring_source = "auto"
        return True


class TestChannelDiscovery:
    def test_eeg_derivations_follow_standard_priority(self):
        assert available_eeg(MONTAGE)[:2] == ["C4-M1", "C3-M2"]

    def test_eog_channels_are_found(self):
        assert available_eog(MONTAGE) == ["E1-M2", "E2-M1"]

    def test_only_submental_emg_is_offered(self):
        # The leg EMG and the mastoid must never be proposed as an EMG input.
        assert available_emg(MONTAGE) == ["Chin1-Chin2"]

    def test_unknown_channel_names_are_ignored(self):
        assert available_eeg(["Foo", "Bar"]) == []

    def test_matching_is_case_insensitive_and_keeps_original_case(self):
        assert available_eeg(["c4-m1"]) == ["c4-m1"]


class TestConfigGeneration:
    def test_every_eeg_derivation_is_paired_with_the_best_eog(self):
        configs = build_default_configs(MONTAGE)
        first = configs[0]
        assert first.eeg == "C4-M1"
        assert first.eog == "E1-M2"

    def test_dropping_the_eog_is_tested(self):
        configs = build_default_configs(MONTAGE)
        assert any(c.eeg == "C4-M1" and c.eog is None for c in configs)

    def test_valid_emg_is_tested_on_the_reference_derivation(self):
        configs = build_default_configs(MONTAGE)
        assert any(c.emg == "Chin1-Chin2" for c in configs)

    def test_metadata_effect_is_isolated_when_age_is_known(self):
        configs = build_default_configs(MONTAGE, age=40, male=True)
        assert any(c.age is None and c.male is None for c in configs)
        assert any(c.age == 40 for c in configs)

    def test_no_metadata_config_added_when_none_is_known(self):
        configs = build_default_configs(MONTAGE)
        assert all(c.age is None for c in configs)

    def test_number_of_eeg_derivations_is_capped(self):
        configs = build_default_configs(MONTAGE, max_eeg=2)
        # Two priority EEG plus the dedicated frontal N3 probe.
        assert {c.eeg for c in configs} == {"C4-M1", "C3-M2", "F4-M1"}

    def test_a_frontal_derivation_is_always_probed_for_n3(self):
        configs = build_default_configs(MONTAGE, max_eeg=2)
        assert any(c.eeg == "F4-M1" for c in configs)

    def test_no_configs_without_a_known_eeg_derivation(self):
        assert build_default_configs(["Foo", "SpO2"]) == []

    def test_configs_are_unique(self):
        configs = build_default_configs(MONTAGE)
        assert len(configs) == len(set(configs))

    def test_label_is_readable(self):
        label = SweepConfig(eeg="C4-M1", eog="E1-M2").label
        assert label == "yasa, EEG C4-M1, EOG E1-M2, sans EMG"

    def test_label_omits_the_emg_for_a_backend_that_ignores_it(self):
        label = SweepConfig(eeg="C4-M1", eog="E1-M2", method="usleep").label
        assert "EMG" not in label


class TestPerBackendConfigs:
    def test_usleep_configs_never_carry_an_emg(self):
        configs = build_default_configs(MONTAGE, method="usleep")
        assert configs
        assert all(c.emg is None for c in configs)

    def test_yasa_configs_do_test_the_emg(self):
        configs = build_default_configs(MONTAGE, method="yasa")
        assert any(c.emg == "Chin1-Chin2" for c in configs)

    def test_metadata_is_only_swept_for_backends_that_read_it(self):
        configs = build_default_configs(MONTAGE, method="aasm_rules", age=40, male=True)
        assert all(c.age == 40 for c in configs)

    def test_pftsleep_only_offers_its_own_derivations(self):
        montage = ["C4-M1", "F4-M1", "O2-M1", "E1-M2", "Chin1-Chin2"]
        configs = build_default_configs(montage, method="pftsleep")
        assert {c.eeg for c in configs} == {"C4-M1"}

    def test_backend_comparison_gives_one_config_per_method(self):
        configs = build_backend_comparison(MONTAGE, ["yasa", "usleep", "aasm_rules"])
        assert [c.method for c in configs] == ["yasa", "usleep", "aasm_rules"]
        by_method = {c.method: c for c in configs}
        assert by_method["yasa"].emg == "Chin1-Chin2"
        assert by_method["usleep"].emg is None

    def test_backend_comparison_skips_a_method_without_usable_eeg(self):
        configs = build_backend_comparison(["Foo"], ["yasa"])
        assert configs == []


class TestConfigsAreRunAsDescribed:
    """A configuration must use exactly the channels its label announces."""

    def _ctrl(self):
        return _StubController(
            outcomes={"C4-M1": ["W", "N2", "N2", "R"]}, manual=["W", "N2", "N2", "R"],
        )

    def test_a_no_eog_config_really_runs_without_an_eog(self):
        ctrl = self._ctrl()
        run_sweep(ctrl, [SweepConfig(eeg="C4-M1", eog=None)])
        assert ctrl.runs[0]["eog"] == []

    def test_a_no_emg_config_really_runs_without_an_emg(self):
        ctrl = self._ctrl()
        run_sweep(ctrl, [SweepConfig(eeg="C4-M1", eog="E1-M2", emg=None)])
        assert ctrl.runs[0]["emg"] == []

    def test_the_requested_channels_are_the_ones_used(self):
        ctrl = self._ctrl()
        run_sweep(ctrl, [
            SweepConfig(eeg="C4-M1", eog="E2-M1", emg="Chin1-Chin2"),
        ])
        assert ctrl.runs[0] == {
            "eeg": "C4-M1", "eog": ["E2-M1"], "emg": ["Chin1-Chin2"],
            "age": None, "male": None, "method": "yasa",
        }

    def test_dropping_the_eog_gives_a_different_run_than_keeping_it(self):
        ctrl = self._ctrl()
        run_sweep(ctrl, [
            SweepConfig(eeg="C4-M1", eog="E1-M2"),
            SweepConfig(eeg="C4-M1", eog=None),
        ])
        assert ctrl.runs[0]["eog"] != ctrl.runs[1]["eog"]


class TestBlockedBackends:
    class _Ctrl(_StubController):
        def auto_scoring_preflight(self, method=None):
            return "Paquet absent." if method == "pftsleep" else None

    def test_a_backend_failing_preflight_is_not_run(self):
        ctrl = self._Ctrl(outcomes={"C4-M1": ["W", "N2"]}, manual=["W", "N2"])
        results = run_sweep(
            ctrl, [SweepConfig(eeg="C4-M1", method="pftsleep")],
        )
        assert not results[0].ok
        assert "Paquet absent." in (results[0].error or "")
        assert ctrl.runs == []

    def test_usable_backends_reports_the_blocking_reason(self):
        ctrl = self._Ctrl(outcomes={}, manual=["W"])
        status = usable_backends(ctrl, ["yasa", "pftsleep"])
        assert status["yasa"] is None
        assert "Paquet absent." in str(status["pftsleep"])


class TestRunSweep:
    def _ctrl(self):
        return _StubController(
            outcomes={
                "C4-M1": ["W", "N2", "N2", "R"],      # perfect
                "C3-M2": ["W", "N2", "N1", "R"],      # one error
                "F4-M1": ["W", "W", "W", "W"],        # poor
            },
            manual=["W", "N2", "N2", "R"],
        )

    def test_each_configuration_is_scored_and_measured(self):
        ctrl = self._ctrl()
        configs = [SweepConfig(eeg="C4-M1"), SweepConfig(eeg="C3-M2")]
        results = run_sweep(ctrl, configs)
        assert [r.config.eeg for r in results] == ["C4-M1", "C3-M2"]
        assert results[0].accuracy == pytest.approx(100.0)
        assert results[1].accuracy == pytest.approx(75.0)
        assert all(r.ok for r in results)

    def test_configuration_is_pushed_to_the_controller_before_each_run(self):
        ctrl = self._ctrl()
        run_sweep(ctrl, [SweepConfig(eeg="C3-M2", eog="E1-M2", emg="Chin1-Chin2", age=44)])
        assert ctrl.runs[0]["eeg"] == "C3-M2"
        assert ctrl.runs[0]["eog"] == ["E1-M2"]
        assert ctrl.runs[0]["emg"] == ["Chin1-Chin2"]
        assert ctrl.runs[0]["age"] == 44

    def test_controller_state_is_restored_afterwards(self):
        ctrl = self._ctrl()
        ctrl.yasa_eeg_candidates = ["C4-M1", "C3-M2"]
        ctrl.yasa_age = 30
        original_auto = _df(["N3", "N3", "N3", "N3"])
        ctrl.auto_scoring_data = original_auto
        ctrl.sleep_scoring_data = original_auto.copy()

        run_sweep(ctrl, [SweepConfig(eeg="C3-M2"), SweepConfig(eeg="F4-M1")])

        assert ctrl.yasa_eeg_candidates == ["C4-M1", "C3-M2"]
        assert ctrl.yasa_age == 30
        assert list(ctrl.auto_scoring_data["stage"]) == ["N3"] * 4

    def test_state_is_restored_even_when_a_run_raises(self):
        ctrl = _StubController(
            outcomes={"C4-M1": RuntimeError("backend casse")},
            manual=["W", "N2"],
        )
        ctrl.yasa_age = 22
        results = run_sweep(ctrl, [SweepConfig(eeg="C4-M1")])
        assert not results[0].ok
        assert ctrl.yasa_age == 22

    def test_failed_run_is_reported_with_its_reason(self):
        ctrl = self._ctrl()
        results = run_sweep(ctrl, [SweepConfig(eeg="O2-M1")])
        assert not results[0].ok
        assert "O2-M1" in (results[0].error or "")

    def test_cancellation_stops_before_the_next_configuration(self):
        ctrl = self._ctrl()
        configs = [SweepConfig(eeg="C4-M1"), SweepConfig(eeg="C3-M2")]
        results = run_sweep(ctrl, configs, should_cancel=lambda: len(ctrl.runs) >= 1)
        assert len(results) == 1

    def test_progress_callback_receives_each_configuration(self):
        ctrl = self._ctrl()
        seen = []
        run_sweep(
            ctrl, [SweepConfig(eeg="C4-M1"), SweepConfig(eeg="C3-M2")],
            progress_cb=lambda i, n, c: seen.append((i, n, c.eeg)),
        )
        assert seen == [(0, 2, "C4-M1"), (1, 2, "C3-M2")]

    def test_missing_manual_reference_is_refused(self):
        ctrl = self._ctrl()
        ctrl.manual_scoring_data = None
        with pytest.raises(ValueError, match="scoring manuel"):
            run_sweep(ctrl, [SweepConfig(eeg="C4-M1")])

    def test_missing_recording_is_refused(self):
        ctrl = self._ctrl()
        ctrl.raw = None
        with pytest.raises(ValueError, match="enregistrement"):
            run_sweep(ctrl, [SweepConfig(eeg="C4-M1")])


class TestRanking:
    def _results(self):
        return [
            SweepResult(config=SweepConfig(eeg="A"), accuracy=70.0, kappa=0.4,
                        n_compared=100, stage_f1={"N3": 0.05}),
            SweepResult(config=SweepConfig(eeg="B"), accuracy=80.0, kappa=0.6,
                        n_compared=100, stage_f1={"N3": 0.5}),
            SweepResult(config=SweepConfig(eeg="C"), error="echec"),
        ]

    def test_best_is_ranked_on_kappa_by_default(self):
        assert best_result(self._results()).config.eeg == "B"

    def test_failed_results_are_excluded_from_the_ranking(self):
        assert [r.config.eeg for r in rank_results(self._results())] == ["B", "A"]

    def test_ranking_key_can_be_changed(self):
        results = [
            SweepResult(config=SweepConfig(eeg="A"), accuracy=90.0, kappa=0.3,
                        n_compared=10),
            SweepResult(config=SweepConfig(eeg="B"), accuracy=60.0, kappa=0.7,
                        n_compared=10),
        ]
        assert best_result(results, "accuracy").config.eeg == "A"
        assert best_result(results, "kappa").config.eeg == "B"

    def test_best_of_nothing_is_none(self):
        assert best_result([]) is None
        assert best_result([SweepResult(config=SweepConfig(eeg="A"), error="x")]) is None


class TestSummary:
    def test_summary_names_the_winner(self):
        results = [
            SweepResult(config=SweepConfig(eeg="C4-M1"), accuracy=85.0, kappa=0.75,
                        n_compared=900),
            SweepResult(config=SweepConfig(eeg="O2-M1"), accuracy=60.0, kappa=0.35,
                        n_compared=900),
        ]
        text = " ".join(sweep_summary(results))
        assert "C4-M1" in text
        assert "0.750" in text

    def test_summary_flags_equivalent_configurations(self):
        results = [
            SweepResult(config=SweepConfig(eeg="C4-M1"), accuracy=80.0, kappa=0.700,
                        n_compared=900),
            SweepResult(config=SweepConfig(eeg="C3-M2"), accuracy=79.9, kappa=0.695,
                        n_compared=900),
        ]
        assert any("equivalentes" in note for note in sweep_summary(results))

    def test_summary_points_to_a_better_n3_configuration(self):
        results = [
            SweepResult(config=SweepConfig(eeg="C4-M1"), accuracy=80.0, kappa=0.70,
                        n_compared=900, stage_f1={"N3": 0.05}),
            SweepResult(config=SweepConfig(eeg="C3-M2"), accuracy=75.0, kappa=0.60,
                        n_compared=900, stage_f1={"N3": 0.55}),
        ]
        assert any("sommeil lent profond" in note for note in sweep_summary(results))

    def test_summary_reports_failures(self):
        results = [
            SweepResult(config=SweepConfig(eeg="C4-M1"), accuracy=80.0, kappa=0.7,
                        n_compared=900),
            SweepResult(config=SweepConfig(eeg="X"), error="modele absent"),
        ]
        assert any("modele absent" in note for note in sweep_summary(results))

    def test_summary_without_any_usable_result(self):
        results = [SweepResult(config=SweepConfig(eeg="X"), error="modele absent")]
        notes = sweep_summary(results)
        assert "Aucune configuration" in notes[0]
        assert "modele absent" in notes[0]


def test_apply_config_writes_the_controller_fields():
    ctrl = AppController()
    apply_config(ctrl, SweepConfig(
        eeg="C4-M1", eog="E1-M2", emg=None, age=51, male=False, method="yasa",
    ))
    assert ctrl.yasa_eeg_candidates == ["C4-M1"]
    assert ctrl.yasa_eog_candidates == ["E1-M2"]
    assert ctrl.yasa_emg_candidates == []
    assert ctrl.yasa_age == 51
    assert ctrl.yasa_male is False


def test_duration_estimate_scales_with_configuration_count():
    assert estimate_duration_s(3, 40.0) == pytest.approx(120.0)
    assert estimate_duration_s(0) == 0.0
