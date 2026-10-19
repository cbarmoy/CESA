"""Unit tests for micro-architecture detection (Vague 1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from CESA.sleep_pipeline.contracts import EventType, ScoredEvent
from CESA.sleep_pipeline import micro_events as me


class TestScoredEventToDict:
    def test_ui_keys(self):
        ev = ScoredEvent(
            event_type=EventType.SPINDLE,
            onset_s=12.5,
            duration_s=1.2,
            confidence=0.9,
            metadata={"channel": "C3-M2", "Frequency": 13.5},
        )
        d = ev.to_dict()
        assert d["onset"] == 12.5
        assert d["duration"] == 1.2
        assert d["type"] == "spindle"
        assert d["description"] == "spindle"
        assert d["channel"] == "C3-M2"
        assert d["confidence"] == 0.9
        assert d["metadata"]["Frequency"] == 13.5
        assert "channel" not in d["metadata"]
        assert d["onset_s"] == 12.5
        assert d["event_type"] == "spindle"

    def test_new_event_types_exist(self):
        assert EventType.SPINDLE.value == "spindle"
        assert EventType.SLOW_WAVE.value == "slow_wave"
        assert EventType.REM_BURST.value == "rem_burst"
        assert EventType.K_COMPLEX.value == "k_complex"


class TestHypnoBuilder:
    def test_scoring_df_to_hypno(self):
        df = pd.DataFrame({
            "time": [0.0, 30.0, 60.0],
            "stage": ["W", "N2", "N3"],
        })
        sfreq = 10.0
        n = 900  # 90 s
        hypno = me.scoring_df_to_hypno(df, n, sfreq, epoch_seconds=30.0)
        assert hypno is not None
        assert hypno.shape == (n,)
        assert hypno[0] == 0  # W
        assert hypno[300] == 2  # N2 at 30 s
        assert hypno[600] == 3  # N3 at 60 s

    def test_empty_scoring_returns_none(self):
        assert me.scoring_df_to_hypno(None, 100, 100.0) is None
        assert me.scoring_df_to_hypno(pd.DataFrame(), 100, 100.0) is None


class TestYasaConversion:
    def test_row_to_scored_event_spindle(self):
        row = pd.Series({
            "Start": 100.0,
            "Duration": 0.8,
            "Channel": "C4-M1",
            "Frequency": 13.2,
            "AbsPower": 1.5,
        })
        ev = me._row_to_scored_event(row, EventType.SPINDLE)
        assert ev.event_type == EventType.SPINDLE
        assert ev.onset_s == 100.0
        assert ev.duration_s == 0.8
        assert ev.metadata["channel"] == "C4-M1"
        assert ev.metadata["Frequency"] == 13.2

    def test_summary_to_events_empty(self):
        assert me._summary_to_events(None, EventType.SPINDLE) == []
        assert me._summary_to_events(pd.DataFrame(), EventType.SPINDLE) == []

    def test_detect_spindles_mocked(self):
        summary = pd.DataFrame([
            {"Start": 10.0, "Duration": 1.0, "Channel": "Cz", "Frequency": 12.5},
            {"Start": 20.0, "Duration": 0.7, "Channel": "Cz", "Frequency": 13.0},
        ])
        fake_res = MagicMock()
        fake_res.summary.return_value = summary

        data = np.random.randn(1, 1000)
        with patch.object(me, "_require_yasa") as req:
            yasa = MagicMock()
            yasa.spindles_detect.return_value = fake_res
            req.return_value = yasa
            events = me.detect_spindles(data, 100.0, ["Cz"])
        assert len(events) == 2
        assert events[0].event_type == EventType.SPINDLE
        assert events[0].onset_s == 10.0
        yasa.spindles_detect.assert_called_once()

    def test_detect_slow_waves_with_coupling_mocked(self):
        summary = pd.DataFrame([
            {
                "Start": 5.0,
                "Duration": 1.5,
                "Channel": "Fz",
                "ndPAC": 0.4,
                "PhaseAtSigmaPeak": 1.2,
                "CooccurringSpindle": True,
            },
        ])
        fake_res = MagicMock()
        fake_res.summary.return_value = summary

        data = np.random.randn(1, 500)
        with patch.object(me, "_require_yasa") as req:
            yasa = MagicMock()
            yasa.sw_detect.return_value = fake_res
            req.return_value = yasa
            events = me.detect_slow_waves(
                data, 100.0, ["Fz"], coupling=True,
                spindle_summary=pd.DataFrame([
                    {"Start": 5.2, "Duration": 0.8, "Channel": "Fz"},
                ]),
            )
        assert len(events) == 1
        assert events[0].event_type == EventType.SLOW_WAVE
        assert events[0].metadata["ndPAC"] == 0.4
        fake_res.find_cooccurring_spindles.assert_called_once()


class TestSummaries:
    def test_summarize_densities(self):
        events = [
            ScoredEvent(
                EventType.SPINDLE, 5.0, 1.0, metadata={"channel": "C3", "Frequency": 13.0},
            ),
            ScoredEvent(
                EventType.SPINDLE, 35.0, 0.8, metadata={"channel": "C3", "Frequency": 12.5},
            ),
            ScoredEvent(
                EventType.SLOW_WAVE, 40.0, 1.2, metadata={"channel": "C3"},
            ),
        ]
        scoring = pd.DataFrame({
            "time": [0.0, 30.0],
            "stage": ["N2", "N2"],
        })
        df = me.summarize_densities(events, scoring, epoch_seconds=30.0)
        assert not df.empty
        assert set(df["type"]) >= {"spindle", "slow_wave"}
        spin = df[(df["type"] == "spindle") & (df["channel"] == "C3")]
        assert len(spin) >= 1
        assert int(spin.iloc[0]["count"]) == 2

    def test_summarize_densities_empty(self):
        df = me.summarize_densities([])
        assert df.empty

    def test_cooccurrence_summary(self):
        events = [
            ScoredEvent(
                EventType.SLOW_WAVE, 1.0, 1.0,
                metadata={
                    "channel": "C4",
                    "ndPAC": 0.5,
                    "PhaseAtSigmaPeak": 0.1,
                    "CooccurringSpindle": True,
                },
            ),
            ScoredEvent(
                EventType.SLOW_WAVE, 2.0, 1.0,
                metadata={
                    "channel": "C4",
                    "ndPAC": 0.3,
                    "PhaseAtSigmaPeak": -0.2,
                    "CooccurringSpindle": False,
                },
            ),
        ]
        df = me.cooccurrence_summary(events)
        assert len(df) == 1
        assert df.iloc[0]["n_sw"] == 2
        assert df.iloc[0]["n_cooccurring"] == 1
        assert df.iloc[0]["pct_cooccurring"] == 50.0
        assert abs(df.iloc[0]["mean_ndPAC"] - 0.4) < 1e-6


class TestControllerDetectMicroEvents:
    def test_no_derivations_returns_empty(self):
        from CESA.app_controller import AppController

        ctrl = AppController.__new__(AppController)
        ctrl.derivations = {}
        ctrl.raw = None
        ctrl.sfreq = 100.0
        ctrl.scoring_epoch_duration = 30.0
        ctrl.sleep_scoring_data = None
        ctrl.yasa_eeg_candidates = []
        ctrl.yasa_eog_candidates = []
        ctrl.night_start_s = None
        ctrl.night_end_s = None

        # Minimal stubs used by get_active_scoring_df path
        ctrl.get_active_scoring_df = lambda: None  # type: ignore[method-assign]

        out = AppController.detect_micro_events(ctrl, ["spindles"])
        assert out["events"] == []
        assert out["warnings"]

    def test_unknown_kinds_returns_warning(self):
        from CESA.app_controller import AppController

        ctrl = AppController.__new__(AppController)
        ctrl.derivations = {"C3": np.zeros(100)}
        ctrl.sfreq = 100.0
        ctrl.scoring_epoch_duration = 30.0
        ctrl.yasa_eeg_candidates = ["C3"]
        ctrl.yasa_eog_candidates = []
        ctrl.get_active_scoring_df = lambda: None  # type: ignore[method-assign]

        out = AppController.detect_micro_events(ctrl, ["not_a_kind"])
        assert out["events"] == []
        assert any("valide" in w.lower() for w in out["warnings"])
