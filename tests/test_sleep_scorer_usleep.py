"""U-Sleep scoring: official API path, label normalisation, preflight.

No call ever leaves the machine: the ``usleep_api`` package is replaced by a
stub, and the temporary-EDF export is short-circuited.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import mne
import numpy as np
import pytest

from CESA.app_controller import AppController
from CESA.sleep_scorer import SleepScorer


@pytest.fixture(autouse=True)
def _no_env_token(monkeypatch):
    monkeypatch.delenv("USLEEP_API_TOKEN", raising=False)


def _raw(ch_names=("F4-M1", "E1-M2"), sfreq=200.0, seconds=120.0):
    n = int(sfreq * seconds)
    data = np.zeros((len(ch_names), n), dtype=np.float64)
    info = mne.create_info(list(ch_names), sfreq, ch_types=["eeg"] * len(ch_names))
    return mne.io.RawArray(data, info, verbose="ERROR")


class _FakeRaw:
    def __init__(self, names):
        self.ch_names = list(names)
        self.info = {"sfreq": 200.0}


def _install_fake_api(monkeypatch, hypnogram=("W", "N2", "N3", "R"), succeed=True):
    sessions: list = []

    class Session:
        def __init__(self):
            self.model = None
            self.uploaded = None
            self.predict_kwargs = None
            self.deleted = False
            self.confidence_requested = False

        def set_model(self, name):
            self.model = name

        def upload_file(self, path, anonymize_before_upload=False):
            self.uploaded = (path, anonymize_before_upload)

        def predict(self, **kwargs):
            self.predict_kwargs = kwargs

        def wait_for_completion(self):
            return succeed

        def get_hypnogram(self, **kwargs):
            self.confidence_requested = bool(kwargs.get("with_confidence_scores"))
            return {"hypnogram": list(hypnogram), "confidence": [0.9] * len(hypnogram)}

        def delete_session(self):
            self.deleted = True

    class API:
        def __init__(self, api_token=None):
            self.api_token = api_token

        def new_session(self, session_name=None):
            session = Session()
            sessions.append(session)
            return session

    mod = types.ModuleType("usleep_api")
    mod.USleepAPI = API
    monkeypatch.setitem(sys.modules, "usleep_api", mod)
    return sessions


def _scorer_api(**kwargs):
    defaults = dict(
        method="usleep",
        eeg_candidates=("F4-M1",),
        eog_candidates=("E1-M2",),
        usleep_api_token="session-token",
        epoch_length=30.0,
        usleep_sfreq=128.0,
    )
    defaults.update(kwargs)
    scorer = SleepScorer(**defaults)
    scorer._prepare_usleep_upload = lambda raw, ch: ("C:/tmp/cesa_usleep.edf", lambda: None)
    return scorer


class TestNormaliseApiStage:
    def test_numeric_and_named_labels(self):
        assert SleepScorer._normalize_api_stage(0) == "W"
        assert SleepScorer._normalize_api_stage(4) == "R"
        assert SleepScorer._normalize_api_stage("REM") == "R"
        assert SleepScorer._normalize_api_stage("wake") == "W"
        assert SleepScorer._normalize_api_stage("3") == "N3"
        assert SleepScorer._normalize_api_stage("inconnu") == "U"

    def test_list_becomes_a_hypnogram(self):
        df = SleepScorer()._usleep_api_dataframe(["W", "N2", "N3", "R"])
        assert list(df["stage"]) == ["W", "N2", "N3", "R"]
        assert list(df["time"]) == [0.0, 30.0, 60.0, 90.0]

    def test_dict_and_confidence_are_unpacked(self):
        df = SleepScorer()._usleep_api_dataframe({
            "hypnogram": [0, 2, 3, 4],
            "confidence": [0.8, 0.7, 0.9, 0.6],
        })
        assert list(df["stage"]) == ["W", "N2", "N3", "R"]
        assert list(df["confidence"]) == pytest.approx([0.8, 0.7, 0.9, 0.6])

    def test_interval_rows_are_expanded(self):
        df = SleepScorer()._usleep_api_dataframe([(0, 60, "N2"), (60, 30, "N3")])
        assert list(df["stage"]) == ["N2", "N2", "N3"]

    def test_probabilities_become_confidence(self):
        proba = [
            [0.9, 0.05, 0.03, 0.01, 0.01],
            [0.1, 0.1, 0.7, 0.05, 0.05],
        ]
        df = SleepScorer()._usleep_api_dataframe({
            "hypnogram": ["W", "N2"],
            "probabilities": proba,
        })
        assert df["confidence"].tolist() == pytest.approx([0.9, 0.7])
        assert df["proba_N2"].tolist() == pytest.approx([0.03, 0.7])


class TestPrefersApi:
    def test_a_token_wins_over_a_checkpoint(self, tmp_path):
        ckpt = tmp_path / "weights.ckpt"
        ckpt.write_bytes(b"x")
        scorer = SleepScorer(
            usleep_api_token="tok",
            usleep_checkpoint_path=str(ckpt),
        )
        assert scorer._usleep_prefers_api()

    def test_checkpoint_file_without_token_stays_local(self, tmp_path):
        ckpt = tmp_path / "weights.ckpt"
        ckpt.write_bytes(b"x")
        scorer = SleepScorer(usleep_checkpoint_path=str(ckpt))
        assert not scorer._usleep_prefers_api()

    def test_env_token_is_read(self, monkeypatch):
        monkeypatch.setenv("USLEEP_API_TOKEN", "from-env")
        scorer = SleepScorer()
        assert scorer._resolve_usleep_token() == "from-env"
        assert scorer._usleep_prefers_api()


class TestApiScoring:
    def test_selected_channels_are_sent_and_the_session_is_deleted(
        self, monkeypatch,
    ):
        sessions = _install_fake_api(monkeypatch)
        df = _scorer_api().score(_raw())
        assert list(df["stage"]) == ["W", "N2", "N3", "R"]
        assert "confidence" in df.columns
        session = sessions[0]
        assert session.model == "U-Sleep v1.0"
        assert session.uploaded == ("C:/tmp/cesa_usleep.edf", True)
        assert session.predict_kwargs["channel_groups"] == [["F4-M1", "E1-M2"]]
        assert session.predict_kwargs["data_per_prediction"] == 128 * 30
        assert session.deleted is True

    def test_without_eog_only_the_eeg_is_sent(self, monkeypatch):
        sessions = _install_fake_api(monkeypatch, hypnogram=["N2", "N2"])
        _scorer_api(usleep_use_eog=False).score(_raw())
        assert sessions[0].predict_kwargs["channel_groups"] == [["F4-M1"]]

    def test_a_rejected_job_still_deletes_the_session(self, monkeypatch):
        sessions = _install_fake_api(monkeypatch, succeed=False)
        with pytest.raises(RuntimeError, match="rejete"):
            _scorer_api().score(_raw())
        assert sessions[0].deleted is True

    def test_missing_package_is_explained(self, monkeypatch):
        mod = types.ModuleType("usleep_api")
        monkeypatch.setitem(sys.modules, "usleep_api", mod)
        with pytest.raises(RuntimeError, match="usleep-api"):
            _scorer_api().score(_raw())


class TestUploadFallback:
    def test_without_edfio_the_original_edf_is_used(self, monkeypatch, tmp_path):
        edf = tmp_path / "night.edf"
        edf.write_bytes(b"edf")
        real_find = importlib.util.find_spec

        def _find(name, *args, **kwargs):
            if name == "edfio":
                return None
            return real_find(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _find)
        scorer = SleepScorer(edf_path=str(edf))
        path, cleanup = scorer._prepare_usleep_upload(_raw(), ["F4-M1", "E1-M2"])
        assert path == str(edf)
        cleanup()

    def test_without_edfio_or_file_the_error_explains_both_options(
        self, monkeypatch,
    ):
        real_find = importlib.util.find_spec

        def _find(name, *args, **kwargs):
            if name == "edfio":
                return None
            return real_find(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _find)
        with pytest.raises(RuntimeError, match="edfio"):
            SleepScorer()._prepare_usleep_upload(_raw(), ["F4-M1"])


class TestPreflight:
    def _ctrl(self, names=("F4-M1", "E1-M2")):
        ctrl = AppController()
        ctrl.raw = _FakeRaw(names)
        return ctrl

    def test_without_token_or_checkpoint_the_api_is_required(self):
        problems = " ".join(self._ctrl().auto_scoring_preflight("usleep"))
        assert "token" in problems.lower()
        assert "Checkpoint U-Sleep introuvable" not in problems

    def test_a_token_does_not_require_torch(self, monkeypatch):
        ctrl = self._ctrl()
        ctrl.usleep_api_token = "session-token"
        real_find = importlib.util.find_spec

        def _find(name, *args, **kwargs):
            if name in ("usleep_api", "edfio"):
                return types.SimpleNamespace()
            if name in ("torch", "braindecode"):
                return None
            return real_find(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _find)
        problems = ctrl.auto_scoring_preflight("usleep")
        assert problems == []

    def test_a_token_without_the_package_is_blocked(self, monkeypatch):
        ctrl = self._ctrl()
        ctrl.usleep_api_token = "session-token"
        real_find = importlib.util.find_spec

        def _find(name, *args, **kwargs):
            if name == "usleep_api":
                return None
            if name == "edfio":
                return types.SimpleNamespace()
            return real_find(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _find)
        problems = " ".join(ctrl.auto_scoring_preflight("usleep"))
        assert "usleep-api" in problems

    def test_local_checkpoint_does_not_require_the_api_package(
        self, monkeypatch, tmp_path,
    ):
        ckpt = tmp_path / "weights.pt"
        ckpt.write_bytes(b"x")
        ctrl = self._ctrl()
        ctrl.usleep_checkpoint_path = str(ckpt)
        real_find = importlib.util.find_spec

        def _find(name, *args, **kwargs):
            if name in ("torch", "braindecode"):
                return types.SimpleNamespace()
            if name == "usleep_api":
                return None
            return real_find(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _find)
        assert ctrl.usleep_uses_api() is False
        assert ctrl.auto_scoring_preflight("usleep") == []

    def test_env_token_is_enough(self, monkeypatch):
        monkeypatch.setenv("USLEEP_API_TOKEN", "from-env")
        ctrl = self._ctrl()
        assert ctrl.usleep_token() == "from-env"
        assert ctrl.usleep_uses_api() is True
