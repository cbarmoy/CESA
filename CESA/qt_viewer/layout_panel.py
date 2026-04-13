"""Dockable layout / visual guides control panel.

Contains per-type spacing multipliers, presets, auto-layout, centering,
and visual guide toggles (baselines, amplitude scale, fine grid,
artifact highlights).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_PRESET_KEY_MAP = {
    "Standard": "standard",
    "Clinical EEG": "clinical_eeg",
    "Compact": "compact",
    "Auto": "auto",
    "Custom": "custom",
}
_PRESET_LABEL_MAP = {v: k for k, v in _PRESET_KEY_MAP.items()}


class LayoutPanel(QtWidgets.QDockWidget):
    """Dockable panel for layout / baseline / visual guide controls."""

    layout_toggled = QtCore.Signal(bool)
    layout_preset_changed = QtCore.Signal(str)
    layout_type_multiplier_changed = QtCore.Signal(str, float)
    layout_center_changed = QtCore.Signal(bool)
    layout_auto_requested = QtCore.Signal()
    guides_baselines_changed = QtCore.Signal(bool)
    guides_amplitude_changed = QtCore.Signal(bool)
    guides_grid_fine_changed = QtCore.Signal(bool)
    guides_artifact_changed = QtCore.Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Layout", parent)
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        )

        container = QtWidgets.QWidget()
        self.setWidget(container)
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Enable / disable
        self._chk_enabled = QtWidgets.QCheckBox("Activer le layout")
        self._chk_enabled.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Bold))
        root.addWidget(self._chk_enabled)

        # -- Preset --
        preset_group = QtWidgets.QGroupBox("Preset")
        pl = QtWidgets.QVBoxLayout(preset_group)
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItems(list(_PRESET_KEY_MAP.keys()))
        pl.addWidget(self._preset_combo)
        self._btn_auto = QtWidgets.QPushButton("Auto-layout intelligent")
        self._btn_auto.setToolTip(
            "Analyse les amplitudes et types de canaux pour calculer\n"
            "le meilleur espacement automatiquement."
        )
        pl.addWidget(self._btn_auto)
        root.addWidget(preset_group)

        # -- Per-type multipliers --
        type_group = QtWidgets.QGroupBox("Multiplicateurs par type")
        tl = QtWidgets.QFormLayout(type_group)
        self._type_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for label, key in [("EEG", "eeg"), ("EOG", "eog"), ("EMG", "emg"), ("ECG", "ecg")]:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.1, 3.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setValue(1.0)
            spin.setProperty("type_key", key)
            tl.addRow(f"{label} :", spin)
            self._type_spins[key] = spin
        root.addWidget(type_group)

        # -- Centering --
        self._chk_center = QtWidgets.QCheckBox("Centrer les signaux (mediane)")
        root.addWidget(self._chk_center)

        # -- Visual guides --
        guides_group = QtWidgets.QGroupBox("Guides visuels")
        gl = QtWidgets.QVBoxLayout(guides_group)
        self._chk_baselines = QtWidgets.QCheckBox("Lignes de baseline")
        self._chk_amp_scale = QtWidgets.QCheckBox("Echelle amplitude (\u00b5V)")
        self._chk_grid_fine = QtWidgets.QCheckBox("Grille temporelle fine (1s / 5s)")
        self._chk_artifact = QtWidgets.QCheckBox("Surligner artefacts")
        for chk in (self._chk_baselines, self._chk_amp_scale,
                    self._chk_grid_fine, self._chk_artifact):
            gl.addWidget(chk)
        root.addWidget(guides_group)

        root.addStretch()

        self._connect()

    def _connect(self) -> None:
        self._chk_enabled.toggled.connect(self.layout_toggled.emit)
        self._preset_combo.currentTextChanged.connect(self._on_preset)
        for key, spin in self._type_spins.items():
            spin.valueChanged.connect(
                lambda v, k=key: self._on_type_spin(k, v))
        self._chk_center.toggled.connect(self.layout_center_changed.emit)
        self._btn_auto.clicked.connect(self.layout_auto_requested.emit)
        self._chk_baselines.toggled.connect(self.guides_baselines_changed.emit)
        self._chk_amp_scale.toggled.connect(self.guides_amplitude_changed.emit)
        self._chk_grid_fine.toggled.connect(self.guides_grid_fine_changed.emit)
        self._chk_artifact.toggled.connect(self.guides_artifact_changed.emit)

    def _on_preset(self, text: str) -> None:
        key = _PRESET_KEY_MAP.get(text, "custom")
        self.layout_preset_changed.emit(key)

    def _on_type_spin(self, type_key: str, value: float) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentText("Custom")
        self._preset_combo.blockSignals(False)
        self.layout_type_multiplier_changed.emit(type_key, value)

    # -- public restore ---------------------------------------------------

    def sync_multipliers(self, multipliers: dict) -> None:
        """Update spinbox values without emitting signals."""
        for key, spin in self._type_spins.items():
            spin.blockSignals(True)
            spin.setValue(multipliers.get(key, 1.0))
            spin.blockSignals(False)

    def set_state(
        self,
        enabled: bool,
        mode: str,
        multipliers: dict,
        center: bool,
        baselines: bool = False,
        amp_scale: bool = False,
        grid_fine: bool = False,
        artifact: bool = False,
    ) -> None:
        """Restore full UI state without emitting signals."""
        widgets = [
            self._chk_enabled, self._preset_combo, self._chk_center,
            self._chk_baselines, self._chk_amp_scale,
            self._chk_grid_fine, self._chk_artifact,
        ]
        widgets.extend(self._type_spins.values())
        for w in widgets:
            w.blockSignals(True)

        self._chk_enabled.setChecked(enabled)

        label = _PRESET_LABEL_MAP.get(mode, "Custom")
        self._preset_combo.setCurrentText(label)

        for key, spin in self._type_spins.items():
            spin.setValue(multipliers.get(key, 1.0))

        self._chk_center.setChecked(center)
        self._chk_baselines.setChecked(baselines)
        self._chk_amp_scale.setChecked(amp_scale)
        self._chk_grid_fine.setChecked(grid_fine)
        self._chk_artifact.setChecked(artifact)

        for w in widgets:
            w.blockSignals(False)
