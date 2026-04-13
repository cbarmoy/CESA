"""Dockable scaling control panel.

Contains gain, spacing, auto-scale, and clipping controls that were
previously inlined in the navigation bar.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class ScalingPanel(QtWidgets.QDockWidget):
    """Dockable panel for EEG scaling / gain controls."""

    scaling_toggled = QtCore.Signal(bool)
    scaling_gain_changed = QtCore.Signal(float)
    scaling_spacing_changed = QtCore.Signal(int)
    scaling_auto_requested = QtCore.Signal()
    scaling_clip_changed = QtCore.Signal(bool, float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Scaling", parent)
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
        self._chk_enabled = QtWidgets.QCheckBox("Activer le scaling")
        self._chk_enabled.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Bold))
        root.addWidget(self._chk_enabled)

        # -- Gain --
        gain_group = QtWidgets.QGroupBox("Gain global")
        gl = QtWidgets.QFormLayout(gain_group)
        self._gain_spin = QtWidgets.QDoubleSpinBox()
        self._gain_spin.setRange(0.1, 100.0)
        self._gain_spin.setSingleStep(0.1)
        self._gain_spin.setDecimals(1)
        self._gain_spin.setValue(1.0)
        gl.addRow("Gain :", self._gain_spin)
        root.addWidget(gain_group)

        # -- Spacing --
        spacing_group = QtWidgets.QGroupBox("Espacement vertical")
        sl = QtWidgets.QFormLayout(spacing_group)
        self._spacing_spin = QtWidgets.QSpinBox()
        self._spacing_spin.setRange(10, 2000)
        self._spacing_spin.setSingleStep(10)
        self._spacing_spin.setValue(150)
        self._spacing_spin.setSuffix(" \u00b5V")
        sl.addRow("Espacement :", self._spacing_spin)
        root.addWidget(spacing_group)

        # -- Auto-scale --
        self._btn_auto = QtWidgets.QPushButton("Auto-scale (percentile)")
        root.addWidget(self._btn_auto)

        # -- Clipping --
        clip_group = QtWidgets.QGroupBox("Clipping")
        cl = QtWidgets.QFormLayout(clip_group)
        self._chk_clip = QtWidgets.QCheckBox("Activer")
        cl.addRow(self._chk_clip)
        self._clip_spin = QtWidgets.QSpinBox()
        self._clip_spin.setRange(50, 5000)
        self._clip_spin.setSingleStep(50)
        self._clip_spin.setValue(500)
        self._clip_spin.setSuffix(" \u00b5V")
        self._clip_spin.setEnabled(False)
        cl.addRow("Seuil :", self._clip_spin)
        root.addWidget(clip_group)

        root.addStretch()

        self._connect()

    def _connect(self) -> None:
        self._chk_enabled.toggled.connect(self._on_enabled)
        self._gain_spin.valueChanged.connect(
            lambda v: self.scaling_gain_changed.emit(v))
        self._spacing_spin.valueChanged.connect(
            lambda v: self.scaling_spacing_changed.emit(v))
        self._btn_auto.clicked.connect(self.scaling_auto_requested.emit)
        self._chk_clip.toggled.connect(self._on_clip_toggled)
        self._clip_spin.valueChanged.connect(self._on_clip_value)

    def _on_enabled(self, checked: bool) -> None:
        self._gain_spin.setEnabled(checked)
        self._spacing_spin.setEnabled(checked)
        self._btn_auto.setEnabled(checked)
        self._chk_clip.setEnabled(checked)
        self._clip_spin.setEnabled(checked and self._chk_clip.isChecked())
        self.scaling_toggled.emit(checked)

    def _on_clip_toggled(self, checked: bool) -> None:
        self._clip_spin.setEnabled(checked)
        self.scaling_clip_changed.emit(checked, float(self._clip_spin.value()))

    def _on_clip_value(self, value: int) -> None:
        if self._chk_clip.isChecked():
            self.scaling_clip_changed.emit(True, float(value))

    # -- public restore ---------------------------------------------------

    def set_state(
        self,
        enabled: bool,
        gain: float,
        spacing: int,
        clip_on: bool,
        clip_val: float,
    ) -> None:
        """Restore UI state without emitting signals."""
        for w in (self._chk_enabled, self._gain_spin, self._spacing_spin,
                  self._chk_clip, self._clip_spin):
            w.blockSignals(True)

        self._chk_enabled.setChecked(enabled)
        self._gain_spin.setValue(gain)
        self._gain_spin.setEnabled(enabled)
        self._spacing_spin.setValue(spacing)
        self._spacing_spin.setEnabled(enabled)
        self._btn_auto.setEnabled(enabled)
        self._chk_clip.setChecked(clip_on)
        self._chk_clip.setEnabled(enabled)
        self._clip_spin.setValue(int(clip_val))
        self._clip_spin.setEnabled(enabled and clip_on)

        for w in (self._chk_enabled, self._gain_spin, self._spacing_spin,
                  self._chk_clip, self._clip_spin):
            w.blockSignals(False)
