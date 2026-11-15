"""Unit tests for the SO-spindle coupling batch runner."""

from __future__ import annotations

import csv
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from CESA import batch_engine as be
from tests.test_so_spindle_coupling import SFREQ, DURATION_S, synthetic_recording


class DummyRaw:
    """Minimal stand-in for an MNE Raw object, with data in volts."""

    def __init__(self, signals_uv: dict, sfreq: float = SFREQ):
        self.ch_names = list(signals_uv)
        self._data = np.vstack([signals_uv[c] for c in self.ch_names]) * 1e-6
        self.info = {"sfreq": float(sfreq)}
        self.n_times = self._data.shape[1]

    def get_data(self, picks=None):
        if picks is None:
            return self._data
        return self._data[list(picks)]


@pytest.fixture
def coupled_raw():
    signal, _ = synthetic_recording(seed=1)
    return DummyRaw({"C3": signal, "EMG1": signal * 0.1})


def _install_raw(monkeypatch, mapping, *, scoring: Optional[pd.DataFrame] = None):
    """Make the batch engine read *mapping* (path -> DummyRaw) instead of disk."""
    def fake_open(path, **_kwargs):
        raw = mapping.get(path)
        if raw is None:
            raise FileNotFoundError(path)
        return raw

    monkeypatch.setattr(be, "open_raw_file", fake_open)
    monkeypatch.setattr(be, "resolve_scoring_for_file", lambda *a, **k: scoring)


class TestChannelResolution:
    def test_requested_channels_are_kept(self, coupled_raw):
        assert be._coupling_channels(coupled_raw, ["C3"]) == ["C3"]

    def test_missing_requested_channels_raise(self, coupled_raw):
        with pytest.raises(ValueError, match="aucun canal demande"):
            be._coupling_channels(coupled_raw, ["Fz", "Pz"])

    def test_empty_request_falls_back_to_eeg(self, coupled_raw):
        # EMG1 is typed as EMG by detect_signal_type and must be dropped.
        assert be._coupling_channels(coupled_raw, []) == ["C3"]

    def test_no_eeg_channel_raises(self):
        raw = DummyRaw({"EMG1": np.zeros(1000), "ECG": np.zeros(1000)})
        with pytest.raises(ValueError, match="aucun canal EEG"):
            be._coupling_channels(raw, [])


class TestBatchRunner:
    def test_writes_one_row_per_channel(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"], include_scoring=False,
        )
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)

        assert result.processed == 1
        assert result.failed == 0
        assert len(result.outputs) == 1

        df = pd.read_csv(result.outputs[0])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["subject"] == "nap1"
        assert row["file"] == "nap1.edf"
        assert row["channel"] == "C3"
        assert row["scoring_used"] == False  # noqa: E712 - numpy bool
        assert row["n_coupled"] > 0
        # Bursts are locked to the SO up-state in the synthetic recording.
        assert abs(row["preferred_phase_rad"]) < 0.5
        assert row["rayleigh_p"] < 0.01

    def test_csv_columns_match_module_contract(self, tmp_path, monkeypatch, coupled_raw):
        from CESA.sleep_pipeline.so_spindle_coupling import STATS_COLUMNS

        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"], include_scoring=False,
        )
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)

        df = pd.read_csv(result.outputs[0])
        assert list(df.columns) == [
            "subject", "file", "group", "scoring_used", "duration_min",
        ] + list(STATS_COLUMNS)

    def test_group_labels_are_written(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(monkeypatch, {"a.edf": coupled_raw, "b.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"], include_scoring=False,
        )
        result = be.run_batch_so_spindle_coupling(
            ["a.edf", "b.edf"], config,
            groups={"a.edf": "AVANT", "b.edf": "APRES"},
        )

        df = pd.read_csv(result.outputs[0]).set_index("file")
        assert df.loc["a.edf", "group"] == "AVANT"
        assert df.loc["b.edf", "group"] == "APRES"

    def test_scoring_restricts_and_is_reported(self, tmp_path, monkeypatch, coupled_raw):
        epochs = int(DURATION_S // 30.0)
        scoring = pd.DataFrame({
            "time": [30.0 * i for i in range(epochs)],
            "stage": ["N2"] * epochs,
        })
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw}, scoring=scoring)
        config = be.BatchCouplingConfig(output_dir=str(tmp_path), channels=["C3"])
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)

        df = pd.read_csv(result.outputs[0])
        assert bool(df.iloc[0]["scoring_used"]) is True
        assert df.iloc[0]["n_coupled"] > 0

    def test_missing_scoring_is_a_warning_not_an_error(
        self, tmp_path, monkeypatch, coupled_raw,
    ):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw}, scoring=None)
        config = be.BatchCouplingConfig(output_dir=str(tmp_path), channels=["C3"])
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)

        assert result.processed == 1
        assert result.failed == 0
        assert any("aucun scoring resolu" in w for w in result.warnings)

    def test_low_event_count_warns_and_hides_phase(
        self, tmp_path, monkeypatch, coupled_raw,
    ):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"],
            include_scoring=False, min_coupled=10_000,
        )
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)

        df = pd.read_csv(result.outputs[0])
        assert df.iloc[0]["n_coupled"] > 0
        assert pd.isna(df.iloc[0]["preferred_phase_rad"])
        assert any("phase preferee non publiee" in w for w in result.warnings)

    def test_unreadable_file_is_counted_as_failure(
        self, tmp_path, monkeypatch, coupled_raw,
    ):
        _install_raw(monkeypatch, {"ok.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"], include_scoring=False,
        )
        result = be.run_batch_so_spindle_coupling(["ok.edf", "ghost.edf"], config)

        assert result.processed == 1
        assert result.failed == 1
        assert any("ghost.edf" in e for e in result.errors)
        # The readable file still produced its CSV.
        assert len(pd.read_csv(result.outputs[0])) == 1

    def test_progress_and_stop(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(monkeypatch, {"a.edf": coupled_raw, "b.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"], include_scoring=False,
        )
        seen: List[float] = []
        # Stop once the first file has been announced.
        result = be.run_batch_so_spindle_coupling(
            ["a.edf", "b.edf"], config,
            progress_cb=lambda pct, msg: seen.append(pct),
            should_stop=lambda: len(seen) >= 1,
        )

        assert result.stopped is True
        assert result.processed == 1
        assert seen[-1] == 100.0

    def test_emits_file_scoring_and_channel_checkpoints(
        self, tmp_path, monkeypatch, coupled_raw,
    ):
        epochs = int(DURATION_S // 30.0)
        scoring = pd.DataFrame({
            "time": [30.0 * i for i in range(epochs)],
            "stage": ["N2"] * epochs,
        })
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw}, scoring=scoring)
        config = be.BatchCouplingConfig(output_dir=str(tmp_path), channels=["C3"])
        messages: List[str] = []
        be.run_batch_so_spindle_coupling(
            ["nap1.edf"], config,
            progress_cb=lambda _pct, msg: messages.append(msg),
        )
        text = "\n".join(messages)
        assert "Chargement: nap1.edf" in text
        assert "Scoring:" in text
        assert "Couplage SO-fuseau: C3" in text
        assert "fuseaux" in text
        assert "couples" in text

    def test_no_file_writes_nothing(self, tmp_path, monkeypatch):
        _install_raw(monkeypatch, {})
        config = be.BatchCouplingConfig(output_dir=str(tmp_path))
        result = be.run_batch_so_spindle_coupling([], config)

        assert result.outputs == []
        assert result.processed == 0


class TestEventsExport:
    def test_per_file_events_csv(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"],
            include_scoring=False, export_events=True,
        )
        be.run_batch_so_spindle_coupling(["nap1.edf"], config)

        events_path = tmp_path / "nap1_coupling_events.csv"
        assert events_path.exists()
        rows = list(csv.DictReader(events_path.open(encoding="utf-8")))
        assert rows
        assert list(rows[0]) == list(be.COUPLING_EVENT_COLUMNS)
        assert all(r["file"] == "nap1.edf" for r in rows)
        coupled = [r for r in rows if r["coupled"] == "True"]
        assert coupled
        assert -np.pi <= float(coupled[0]["so_phase_rad"]) <= np.pi

    def test_events_not_written_by_default(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"], include_scoring=False,
        )
        be.run_batch_so_spindle_coupling(["nap1.edf"], config)

        assert not (tmp_path / "nap1_coupling_events.csv").exists()


class TestOptionalOutputs:
    def test_figures_off_by_default(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"], include_scoring=False,
        )
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)
        names = [p.replace("\\", "/").split("/")[-1] for p in result.outputs]
        assert names == ["so_spindle_coupling_batch.csv"]
        df = pd.read_csv(result.outputs[0])
        assert "r_z" not in df.columns

    def test_permutation_adds_columns(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"],
            include_scoring=False, compute_permutation=True,
        )
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)
        df = pd.read_csv(result.outputs[0])
        assert "r_z" in df.columns
        assert "r_z_p" in df.columns
        assert df.iloc[0]["r_z"] == df.iloc[0]["r_z"]  # not NaN

    def test_circular_figures_when_requested(self, tmp_path, monkeypatch, coupled_raw):
        _install_raw(
            monkeypatch,
            {"S001_Av.edf": coupled_raw, "S001_Ap.edf": coupled_raw},
        )
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"],
            include_scoring=False, export_figures=True,
        )
        result = be.run_batch_so_spindle_coupling(
            ["S001_Av.edf", "S001_Ap.edf"], config,
            groups={"S001_Av.edf": "AVANT", "S001_Ap.edf": "APRES"},
        )
        names = [p.replace("\\", "/").split("/")[-1] for p in result.outputs]
        assert "coupling_phases.png" in names

    def test_failing_figure_does_not_fail_the_batch(
        self, tmp_path, monkeypatch, coupled_raw,
    ):
        _install_raw(monkeypatch, {"nap1.edf": coupled_raw})

        def boom(*_a, **_k):
            raise RuntimeError("nope")

        monkeypatch.setattr(
            "CESA.coupling_plots.save_coupling_figures", boom, raising=True,
        )
        config = be.BatchCouplingConfig(
            output_dir=str(tmp_path), channels=["C3"],
            include_scoring=False, export_figures=True,
        )
        result = be.run_batch_so_spindle_coupling(["nap1.edf"], config)
        assert result.processed == 1
        assert result.failed == 0
        assert any("figures circulaires" in w for w in result.warnings)
        assert (tmp_path / "so_spindle_coupling_batch.csv").exists()


class TestFFTCheckpoints:
    def test_emits_load_scoring_channel_and_export(
        self, tmp_path, monkeypatch,
    ):
        raw = DummyRaw({"C3": np.ones(2000)})
        monkeypatch.setattr(be, "open_raw_file", lambda _path, **_k: raw)
        monkeypatch.setattr(be, "find_scoring_sidecar", lambda _p: None)
        monkeypatch.setattr(be, "resolve_scoring_for_file", lambda *_a, **_k: None)
        monkeypatch.setattr(
            be,
            "compute_channel_psd_by_stage",
            lambda *_a, **_k: {
                "N2": (
                    np.array([1.0]), np.array([0.1]), np.array([0.01]), 12,
                ),
            },
        )
        messages: List[str] = []
        config = be.BatchFFTConfig(output_dir=str(tmp_path), include_scoring=True)
        be.run_batch_fft(
            ["nap1.edf"], ["C3"], config,
            progress_cb=lambda _pct, msg: messages.append(msg),
        )
        text = "\n".join(messages)
        assert "Chargement: nap1.edf" in text
        assert "Scoring: aucun sidecar" in text
        assert "FFT: nap1.edf - C3" in text
        assert "Stades: N2:12" in text
        assert "Export: nap1_C3_FFT.csv" in text
