"""Cheap scoring-sidecar lookup used when scanning a batch folder."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from CESA.batch_engine import find_scoring_sidecar
from CESA.qt_viewer.dialogs.batch_analysis_dialog import (
    GROUP_BEFORE,
    GROUP_NONE,
    _ROLE_SCORING,
    BatchAnalysisDialog,
)


@pytest.fixture
def app():
    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing
    return QtWidgets.QApplication([])


def test_finds_same_stem_csv(tmp_path):
    rec = tmp_path / "nap1.edf"
    rec.write_bytes(b"0")
    sidecar = tmp_path / "nap1.csv"
    sidecar.write_text("time,stage\n0,N2\n", encoding="utf-8")
    (tmp_path / "Other_Hypnogram.edf").write_bytes(b"0")
    assert find_scoring_sidecar(str(rec)) == str(sidecar)


def test_finds_folder_hypnogram_when_no_stem_sidecar(tmp_path):
    rec = tmp_path / "nap1.edf"
    rec.write_bytes(b"0")
    hyp = tmp_path / "S030_Hypnogram.edf"
    hyp.write_bytes(b"0")
    assert find_scoring_sidecar(str(rec)) == str(hyp)


def test_returns_none_when_nothing_matches(tmp_path):
    rec = tmp_path / "nap1.edf"
    rec.write_bytes(b"0")
    (tmp_path / "notes.txt").write_text("no scoring", encoding="utf-8")
    assert find_scoring_sidecar(str(rec)) is None


def test_format_item_always_shows_scoring_tag():
    text = BatchAnalysisDialog._format_item(r"C:\data\nap1.edf", GROUP_NONE, False)
    assert text == "nap1.edf  [sans scoring]"
    tagged = BatchAnalysisDialog._format_item(
        r"C:\data\S030_Av.edf", GROUP_BEFORE, True,
    )
    assert tagged == "S030_Av.edf  [AVANT]  [scoring]"


def test_add_path_marks_missing_scoring_immediately(app, tmp_path):
    rec = tmp_path / "nap1.edf"
    rec.write_bytes(b"0")
    dlg = BatchAnalysisDialog(None)
    dlg._add_path(str(rec))
    dlg._refresh_files_status()
    item = dlg._files_list.item(0)
    assert "[sans scoring]" in item.text()
    assert "Aucun scoring" in item.toolTip()
    assert item.data(_ROLE_SCORING) in ("", None)
    assert "1 fichier" in dlg._files_status.text()
    assert "0 avec scoring" in dlg._files_status.text()
    assert "1 sans scoring" in dlg._files_status.text()
    dlg.close()


def test_add_path_marks_detected_scoring_and_keeps_it_when_grouped(app, tmp_path):
    rec = tmp_path / "S030_Av.edf"
    rec.write_bytes(b"0")
    sidecar = tmp_path / "S030_Av.csv"
    sidecar.write_text("time,stage\n0,N2\n", encoding="utf-8")
    dlg = BatchAnalysisDialog(None)
    dlg._add_path(str(rec), group=GROUP_BEFORE)
    dlg._refresh_files_status()
    item = dlg._files_list.item(0)
    assert "[AVANT]" in item.text()
    assert "[scoring]" in item.text()
    assert "sans scoring" not in item.text()
    assert sidecar.name in item.toolTip()
    assert "1 avec scoring" in dlg._files_status.text()

    item.setSelected(True)
    dlg._mark_selected(GROUP_NONE)
    assert "[scoring]" in item.text()
    assert "[AVANT]" not in item.text()
    dlg.close()
