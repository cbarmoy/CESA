"""Qt dialog for selecting channels for the multiscale pyramid."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets


class ChannelSelectorDialog(QtWidgets.QDialog):
    """Dialog to choose which channels to include in the multiscale pyramid."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        available_channels: List[str],
        preselected: Optional[List[str]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Selection des canaux")
        self.resize(700, 600)
        self.available = sorted(available_channels)
        self.preselected = set(preselected or [])
        self.selected_channels: List[str] = []
        self._checks: Dict[str, QtWidgets.QCheckBox] = {}
        self._build()

    def _detect_types(self) -> Dict[str, List[str]]:
        types: Dict[str, List[str]] = {}
        for ch in self.available:
            cl = ch.lower()
            if any(t in cl for t in ('eog', 'e1', 'e2', 'loc', 'roc', 'eye')):
                cat = 'EOG'
            elif any(t in cl for t in ('emg', 'chin', 'menton', 'leg', 'jambe')):
                cat = 'EMG'
            elif any(t in cl for t in ('ecg', 'ekg', 'heart')):
                cat = 'ECG'
            elif any(t in cl for t in ('flow', 'nasal', 'abdomen', 'resp', 'thorax')):
                cat = 'Respiration'
            elif any(t in cl for t in ('spo2', 'sat', 'pulse')):
                cat = 'SpO2'
            elif any(t in cl for t in ('f3', 'f4', 'c3', 'c4', 'o1', 'o2', '-m1', '-m2')):
                cat = 'EEG'
            else:
                cat = 'Autres'
            types.setdefault(cat, []).append(ch)
        return types

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel(
            "Choisissez les canaux a inclure dans le fichier de navigation rapide."
        ))

        # Quick actions
        quick = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Tout selectionner")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QtWidgets.QPushButton("Tout deselectionner")
        btn_none.clicked.connect(lambda: self._set_all(False))
        quick.addWidget(btn_all)
        quick.addWidget(btn_none)
        quick.addStretch()
        layout.addLayout(quick)

        # Scrollable channel list grouped by type
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_lay = QtWidgets.QVBoxLayout(content)

        for cat, channels in self._detect_types().items():
            grp = QtWidgets.QGroupBox(f"{cat} ({len(channels)})")
            grp_lay = QtWidgets.QVBoxLayout(grp)
            for ch in channels:
                cb = QtWidgets.QCheckBox(ch)
                cb.setChecked(ch in self.preselected)
                self._checks[ch] = cb
                grp_lay.addWidget(cb)
            content_lay.addWidget(grp)

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        # Buttons
        btn_lay = QtWidgets.QHBoxLayout()
        btn_lay.addStretch()
        ok = QtWidgets.QPushButton("Confirmer")
        ok.clicked.connect(self._confirm)
        cancel = QtWidgets.QPushButton("Annuler")
        cancel.clicked.connect(self.reject)
        btn_lay.addWidget(ok)
        btn_lay.addWidget(cancel)
        layout.addLayout(btn_lay)

    def _set_all(self, state: bool) -> None:
        for cb in self._checks.values():
            cb.setChecked(state)

    def _confirm(self) -> None:
        self.selected_channels = [ch for ch, cb in self._checks.items() if cb.isChecked()]
        if not self.selected_channels:
            QtWidgets.QMessageBox.warning(
                self, "Aucun canal", "Selectionnez au moins un canal.",
            )
            return
        self.accept()

    def get_selection(self) -> Optional[List[str]]:
        if self.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return self.selected_channels
        return None
