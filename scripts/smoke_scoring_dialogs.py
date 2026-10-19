"""Offscreen smoke check: both scoring dialogs build and react to a backend change.

Run with: python scripts/smoke_scoring_dialogs.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from PySide6 import QtWidgets

from CESA.app_controller import AppController

MONTAGE = [
    "C4-M1", "C3-M2", "F4-M1", "O2-M1", "E1-M2", "E2-M1",
    "Chin1-Chin2", "Left Leg", "M1", "ECG", "SpO2",
]


class _FakeRaw:
    def __init__(self, names):
        self.ch_names = list(names)
        self.info = {"sfreq": 200.0}


def _controller() -> AppController:
    ctrl = AppController()
    ctrl.raw = _FakeRaw(MONTAGE)
    ctrl.manual_scoring_data = pd.DataFrame({
        "time": [0.0, 30.0, 60.0, 90.0],
        "stage": ["W", "N2", "N3", "R"],
    })
    ctrl.manual_scoring_epoch_seconds = 30.0
    ctrl.scoring_epoch_duration = 30.0
    # Reproduces the reported problem: the controller's default backend is one
    # whose package is missing.
    ctrl.sleep_scoring_method = "pftsleep"
    return ctrl


def check_config_dialog(ctrl: AppController) -> None:
    from CESA.qt_viewer.dialogs.auto_scoring_config_dialog import (
        AutoScoringConfigDialog,
    )

    dlg = AutoScoringConfigDialog(None, ctrl, MONTAGE)
    for method in ("yasa", "usleep", "aasm_rules", "pftsleep"):
        index = dlg._methods.index(method)
        dlg._method_combo.setCurrentIndex(index)
        emg_enabled = dlg._emg_list.isEnabled()
        selected = [
            dlg._eeg_list.item(i).text()
            for i in range(dlg._eeg_list.count())
            if dlg._eeg_list.item(i).isSelected()
        ]
        emg_selected = [
            dlg._emg_list.item(i).text()
            for i in range(dlg._emg_list.count())
            if dlg._emg_list.item(i).isSelected()
        ]
        print(
            f"  {method:11s} EEG={selected} EMG={emg_selected} "
            f"emg_actif={emg_enabled} canaux_affiches="
            f"{dlg._groups['channels'].isVisibleTo(dlg)}"
        )
    dlg.deleteLater()


def check_sweep_dialog(ctrl: AppController) -> None:
    from CESA.qt_viewer.dialogs.scoring_sweep_dialog import ScoringSweepDialog

    dlg = ScoringSweepDialog(None, ctrl)
    print(f"  backend par defaut : {dlg._current_method()}")
    print(f"  indisponibles : "
          f"{[m for m, r in dlg._blocked.items() if r]}")
    print(f"  configurations : {len(dlg._configs)}")
    print(f"  bouton lancer actif : {dlg._run_btn.isEnabled()}")
    dlg._compare_backends.setChecked(True)
    methods = sorted({c.method for c in dlg._configs})
    print(f"  avec comparaison inter-algorithmes : {methods}")
    dlg.deleteLater()


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ctrl = _controller()
    print("Dialogue de configuration :")
    check_config_dialog(ctrl)
    print("Dialogue de balayage :")
    check_sweep_dialog(ctrl)
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
