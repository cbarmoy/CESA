"""Qt dialog for mandatory manual channel mapping to profile sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from PySide6 import QtCore, QtWidgets


@dataclass
class MappingResult:
    accepted: bool
    channel_mapping: Dict[str, str]


class ChannelMappingDialog(QtWidgets.QDialog):
    """Blocking dialog that asks user to map channels to sections."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        channels: Iterable[str],
        section_labels: Dict[str, str],
        prefill: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assignation manuelle des canaux")
        self.resize(760, 520)
        self.channels = list(channels)
        self.section_labels = dict(section_labels)
        self.prefill = dict(prefill or {})
        self.mapping_result = MappingResult(accepted=False, channel_mapping={})
        self._combos: Dict[str, QtWidgets.QComboBox] = {}
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel("Assignez chaque canal a une section d'affichage (ou Ignorer).")
        )

        # Preset button
        preset_btn = QtWidgets.QPushButton("Preset: derivations uniquement")
        preset_btn.clicked.connect(self._apply_derivations_preset)
        layout.addWidget(preset_btn)

        # Scrollable channel list
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QtWidgets.QWidget()
        form_lay = QtWidgets.QFormLayout(form_widget)

        choices = ["Ignorer"] + [self.section_labels[k] for k in self.section_labels]
        self._choice_keys = ["__ignore__"] + list(self.section_labels.keys())

        for ch in self.channels:
            combo = QtWidgets.QComboBox()
            combo.addItems(choices)
            initial = self.prefill.get(ch, "__ignore__")
            idx = self._choice_keys.index(initial) if initial in self._choice_keys else 0
            combo.setCurrentIndex(idx)
            form_lay.addRow(ch, combo)
            self._combos[ch] = combo

        scroll.setWidget(form_widget)
        layout.addWidget(scroll, stretch=1)

        # Buttons
        btn_lay = QtWidgets.QHBoxLayout()
        btn_lay.addStretch()
        ok_btn = QtWidgets.QPushButton("Valider")
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QtWidgets.QPushButton("Annuler")
        cancel_btn.clicked.connect(self.reject)
        btn_lay.addWidget(ok_btn)
        btn_lay.addWidget(cancel_btn)
        layout.addLayout(btn_lay)

    def _accept(self) -> None:
        mapping: Dict[str, str] = {}
        for ch, combo in self._combos.items():
            idx = combo.currentIndex()
            mapping[ch] = self._choice_keys[idx]
        self.mapping_result = MappingResult(accepted=True, channel_mapping=mapping)
        self.accept()

    def _suggest_family(self, name: str) -> str:
        s = name.upper()
        compact = "".join(c for c in s if c not in " -_/:.")
        if "IMP" in s:
            return "__ignore__"
        if any(t in compact for t in ("E1M2", "E2M1", "EOG")):
            return "eog"
        if any(t in compact for t in ("ECG", "EKG")):
            return "ecg"
        if any(t in compact for t in ("EMG", "CHIN", "MENTON")):
            return "emg"
        eeg_tokens = ("F3M2", "F4M1", "C3M2", "C4M1", "O1M2", "O2M1")
        if any(t in compact for t in eeg_tokens):
            return "eeg"
        if "M1" in compact or "M2" in compact:
            return "eeg"
        return "__ignore__"

    def _apply_derivations_preset(self) -> None:
        for ch, combo in self._combos.items():
            family = self._suggest_family(ch)
            if family in self._choice_keys:
                combo.setCurrentIndex(self._choice_keys.index(family))
            else:
                combo.setCurrentIndex(0)

    def get_result(self) -> MappingResult:
        if self.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return self.mapping_result
        return MappingResult(accepted=False, channel_mapping={})
