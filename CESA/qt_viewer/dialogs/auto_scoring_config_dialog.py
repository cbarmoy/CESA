"""Qt dialog: advanced automatic-scoring configuration.

Lets the user pick the scoring backend and tune the parameters consumed by
``AppController.run_auto_scoring`` (epoch length, candidate EEG/EOG/EMG
channels, resampling target, YASA confidence threshold) before launching.

Replaces the Tk ``_open_sleep_scoring_settings`` window.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6 import QtWidgets

logger = logging.getLogger(__name__)

_METHOD_LABELS = {
    "pftsleep": "PFTSleep (deep learning)",
    "yasa": "YASA (feature-based)",
    "usleep": "U-Sleep (deep learning)",
    "aasm_rules": "Regles AASM (rule-based)",
    "ml": "ML (gradient boosting)",
    "ml_hmm": "ML + HMM (sequence model)",
    "rules_hmm": "Regles AASM + HMM",
}


class AutoScoringConfigDialog(QtWidgets.QDialog):
    """Configure backend + parameters for automatic scoring."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        controller,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configuration du scoring automatique")
        self.resize(560, 640)
        self._ctrl = controller
        self._channels = list(channel_names or [])
        self._selected_method: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self._method_combo = QtWidgets.QComboBox()
        self._methods = list(self._ctrl.SUPPORTED_SCORING_METHODS)
        self._method_combo.addItems([_METHOD_LABELS.get(m, m) for m in self._methods])
        cur = getattr(self._ctrl, "sleep_scoring_method", "pftsleep")
        if cur in self._methods:
            self._method_combo.setCurrentIndex(self._methods.index(cur))
        form.addRow("Backend :", self._method_combo)

        self._epoch = QtWidgets.QDoubleSpinBox()
        self._epoch.setRange(1.0, 120.0)
        self._epoch.setSingleStep(1.0)
        self._epoch.setSuffix(" s")
        self._epoch.setValue(float(getattr(self._ctrl, "auto_scoring_epoch_length", 30.0)))
        form.addRow("Duree d'epoque :", self._epoch)

        self._target_sfreq = QtWidgets.QDoubleSpinBox()
        self._target_sfreq.setRange(0.0, 1000.0)
        self._target_sfreq.setSingleStep(10.0)
        self._target_sfreq.setSuffix(" Hz")
        self._target_sfreq.setValue(float(getattr(self._ctrl, "yasa_target_sfreq", 100.0)))
        form.addRow("Reechantillonnage cible :", self._target_sfreq)

        self._conf = QtWidgets.QDoubleSpinBox()
        self._conf.setRange(0.0, 1.0)
        self._conf.setSingleStep(0.05)
        self._conf.setDecimals(2)
        self._conf.setValue(float(getattr(self._ctrl, "yasa_confidence_threshold", 0.80)))
        form.addRow("Seuil de confiance (YASA) :", self._conf)

        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel(
            "Canaux candidats (selection multiple ; vide = liste par defaut) :"
        ))
        chan_box = QtWidgets.QHBoxLayout()
        self._eeg_list = self._make_channel_list(getattr(self._ctrl, "yasa_eeg_candidates", []))
        self._eog_list = self._make_channel_list(getattr(self._ctrl, "yasa_eog_candidates", []))
        self._emg_list = self._make_channel_list(getattr(self._ctrl, "yasa_emg_candidates", []))
        chan_box.addLayout(self._labeled_list("EEG", self._eeg_list))
        chan_box.addLayout(self._labeled_list("EOG", self._eog_list))
        chan_box.addLayout(self._labeled_list("EMG", self._emg_list))
        layout.addLayout(chan_box, stretch=1)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Lancer le scoring")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _labeled_list(self, label: str, widget: QtWidgets.QListWidget) -> QtWidgets.QVBoxLayout:
        v = QtWidgets.QVBoxLayout()
        v.addWidget(QtWidgets.QLabel(label))
        v.addWidget(widget)
        return v

    def _make_channel_list(self, preselected: List[str]) -> QtWidgets.QListWidget:
        lw = QtWidgets.QListWidget()
        lw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        # Show available channels first; fall back to known candidates.
        items = self._channels or list(preselected)
        pre = {p.upper() for p in preselected}
        for name in items:
            it = QtWidgets.QListWidgetItem(name)
            lw.addItem(it)
            if name.upper() in pre:
                it.setSelected(True)
        return lw

    @staticmethod
    def _selected(lw: QtWidgets.QListWidget) -> List[str]:
        return [it.text() for it in lw.selectedItems()]

    def _on_accept(self) -> None:
        method = self._methods[self._method_combo.currentIndex()]
        self._ctrl.sleep_scoring_method = method
        self._ctrl.auto_scoring_epoch_length = float(self._epoch.value())
        self._ctrl.yasa_target_sfreq = float(self._target_sfreq.value())
        self._ctrl.yasa_confidence_threshold = float(self._conf.value())

        eeg = self._selected(self._eeg_list)
        eog = self._selected(self._eog_list)
        emg = self._selected(self._emg_list)
        if eeg:
            self._ctrl.yasa_eeg_candidates = eeg
        if eog:
            self._ctrl.yasa_eog_candidates = eog
        if emg:
            self._ctrl.yasa_emg_candidates = emg

        self._selected_method = method
        self.accept()

    @property
    def selected_method(self) -> Optional[str]:
        return self._selected_method
