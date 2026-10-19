"""The sweep dialog must actually switch algorithm when the user picks one."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

from CESA.app_controller import AppController
from CESA.qt_viewer.dialogs.auto_scoring_config_dialog import METHOD_LABELS
from CESA.qt_viewer.dialogs.scoring_sweep_dialog import ScoringSweepDialog


MONTAGE = [
    "C4-M1", "C3-M2", "F4-M1", "O2-M1", "E1-M2", "E2-M1",
    "Chin1-Chin2", "Left Leg", "M1", "ECG", "SpO2",
]


class _FakeRaw:
    def __init__(self, names):
        self.ch_names = list(names)
        self.info = {"sfreq": 200.0}


@pytest.fixture
def app():
    from PySide6 import QtWidgets

    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing
    return QtWidgets.QApplication([])


def _controller():
    ctrl = AppController()
    ctrl.raw = _FakeRaw(MONTAGE)
    ctrl.manual_scoring_data = pd.DataFrame({
        "time": [0.0, 30.0, 60.0],
        "stage": ["W", "N2", "R"],
    })
    ctrl.manual_scoring_epoch_seconds = 30.0
    ctrl.sleep_scoring_method = "yasa"
    return ctrl


def _index_of(dlg: ScoringSweepDialog, method: str) -> int:
    index = dlg._method_combo.findData(method)
    assert index >= 0, f"{method} missing from the combo"
    return index


def test_every_backend_in_the_combo_can_be_selected(app):
    dlg = ScoringSweepDialog(None, _controller())
    try:
        for index in range(dlg._method_combo.count()):
            item = dlg._method_combo.model().item(index)
            assert item is None or item.isEnabled(), (
                f"{dlg._method_combo.itemData(index)} is greyed out and "
                "cannot be chosen"
            )
    finally:
        dlg.deleteLater()


def test_choosing_aasm_rules_rebuilds_the_table_for_that_backend(app):
    dlg = ScoringSweepDialog(None, _controller())
    try:
        dlg._method_combo.setCurrentIndex(_index_of(dlg, "aasm_rules"))
        assert dlg._current_method() == "aasm_rules"
        assert dlg._configs
        assert all(c.method == "aasm_rules" for c in dlg._configs)
        algorithms = [
            dlg._config_table.item(row, 1).text()
            for row in range(dlg._config_table.rowCount())
        ]
        assert algorithms
        assert all("YASA" not in text for text in algorithms)
        assert all("AASM" in text for text in algorithms)
    finally:
        dlg.deleteLater()


def test_a_blocked_backend_is_selectable_and_explains_why(app):
    dlg = ScoringSweepDialog(None, _controller())
    try:
        # U-Sleep is typically blocked (no token / no package) but must still
        # be reachable: greying it out left the combo stuck on YASA.
        dlg._method_combo.setCurrentIndex(_index_of(dlg, "usleep"))
        assert dlg._current_method() == "usleep"
        assert dlg._configs
        assert all(c.method == "usleep" for c in dlg._configs)
        note = dlg._method_note.text()
        if dlg._blocked.get("usleep"):
            assert "ne peut pas etre utilise" in note
            assert not dlg._run_btn.isEnabled()
    finally:
        dlg.deleteLater()


def test_combo_user_data_is_the_method_id(app):
    dlg = ScoringSweepDialog(None, _controller())
    try:
        listed = [
            dlg._method_combo.itemData(i) for i in range(dlg._method_combo.count())
        ]
        assert listed == list(METHOD_LABELS)
    finally:
        dlg.deleteLater()
