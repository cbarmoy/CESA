"""Dockable control panel for the wavelet spectrogram bars.

Exposes every :class:`CESA.profile_schema.SpectrogramConfig` option: which
band(s) are visible, their height, the analysed channel, frequency band
(with presets), colormap, scale, 1/f correction, raw vs filtered signal,
frequency resolution / axis, opacity, hypnogram overlay and image export.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_FREQ_PRESETS = [
    ("Large bande (0.5-30 Hz)", 0.5, 30.0),
    ("Delta (0.5-4 Hz)", 0.5, 4.0),
    ("Theta (4-8 Hz)", 4.0, 8.0),
    ("Alpha (8-12 Hz)", 8.0, 12.0),
    ("Sigma / fuseaux (11-16 Hz)", 11.0, 16.0),
    ("Beta (16-30 Hz)", 16.0, 30.0),
    ("Large bande (0.5-45 Hz)", 0.5, 45.0),
    ("Personnalise", None, None),
]

_COLORMAPS = ["viridis", "magma", "turbo", "inferno", "plasma", "grey"]

_SCALES = [("dB", "db"), ("Lineaire", "linear"), ("Z-score / frequence", "zscore")]


class SpectrogramPanel(QtWidgets.QDockWidget):
    """Dockable panel for the night / window wavelet spectrogram bars."""

    night_toggled = QtCore.Signal(bool)
    window_toggled = QtCore.Signal(bool)
    night_height_changed = QtCore.Signal(float)
    window_height_changed = QtCore.Signal(int)
    channel_changed = QtCore.Signal(str)
    fmin_changed = QtCore.Signal(float)
    fmax_changed = QtCore.Signal(float)
    n_freqs_changed = QtCore.Signal(int)
    freq_log_toggled = QtCore.Signal(bool)
    colormap_changed = QtCore.Signal(str)
    scale_changed = QtCore.Signal(str)
    use_filtered_toggled = QtCore.Signal(bool)
    onef_correction_toggled = QtCore.Signal(bool)
    opacity_changed = QtCore.Signal(float)
    overlay_hypnogram_toggled = QtCore.Signal(bool)
    export_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Spectrogramme", parent)
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        )

        container = QtWidgets.QWidget()
        self.setWidget(container)
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # -- Visibility --
        vis_group = QtWidgets.QGroupBox("Affichage")
        vl = QtWidgets.QVBoxLayout(vis_group)
        self._chk_night = QtWidgets.QCheckBox("Bandeau nuit complete (sous l'hypnogramme)")
        self._chk_window = QtWidgets.QCheckBox("Bandeau fenetre visible (sous l'EEG)")
        vl.addWidget(self._chk_night)
        vl.addWidget(self._chk_window)

        height_form = QtWidgets.QFormLayout()
        self._night_height_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._night_height_slider.setRange(10, 90)
        self._night_height_slider.setValue(40)
        height_form.addRow("Hauteur bandeau nuit :", self._night_height_slider)
        self._window_height_spin = QtWidgets.QSpinBox()
        self._window_height_spin.setRange(30, 300)
        self._window_height_spin.setSingleStep(10)
        self._window_height_spin.setValue(90)
        self._window_height_spin.setSuffix(" px")
        height_form.addRow("Hauteur bandeau fenetre :", self._window_height_spin)
        vl.addLayout(height_form)
        root.addWidget(vis_group)

        # -- Channel --
        ch_group = QtWidgets.QGroupBox("Canal analyse")
        cl = QtWidgets.QFormLayout(ch_group)
        self._channel_combo = QtWidgets.QComboBox()
        self._channel_combo.setEditable(False)
        cl.addRow("Canal :", self._channel_combo)
        self._channel_hint = QtWidgets.QLabel(
            "Defaut auto : D11 > C3-M1/C3-M2 > premier EEG"
        )
        self._channel_hint.setStyleSheet("color: gray; font-size: 10px;")
        self._channel_hint.setWordWrap(True)
        cl.addRow(self._channel_hint)
        root.addWidget(ch_group)

        # -- Frequency band --
        freq_group = QtWidgets.QGroupBox("Bande de frequence")
        fl = QtWidgets.QFormLayout(freq_group)
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItems([p[0] for p in _FREQ_PRESETS])
        fl.addRow("Preset :", self._preset_combo)
        self._fmin_spin = QtWidgets.QDoubleSpinBox()
        self._fmin_spin.setRange(0.1, 100.0)
        self._fmin_spin.setDecimals(1)
        self._fmin_spin.setSingleStep(0.5)
        self._fmin_spin.setValue(0.5)
        fl.addRow("Fmin (Hz) :", self._fmin_spin)
        self._fmax_spin = QtWidgets.QDoubleSpinBox()
        self._fmax_spin.setRange(0.5, 200.0)
        self._fmax_spin.setDecimals(1)
        self._fmax_spin.setSingleStep(0.5)
        self._fmax_spin.setValue(30.0)
        fl.addRow("Fmax (Hz) :", self._fmax_spin)
        self._n_freqs_spin = QtWidgets.QSpinBox()
        self._n_freqs_spin.setRange(8, 200)
        self._n_freqs_spin.setValue(40)
        fl.addRow("Resolution (nb freq.) :", self._n_freqs_spin)
        self._chk_freq_log = QtWidgets.QCheckBox("Axe frequence logarithmique")
        fl.addRow(self._chk_freq_log)
        root.addWidget(freq_group)

        # -- Rendering --
        render_group = QtWidgets.QGroupBox("Rendu")
        rl = QtWidgets.QFormLayout(render_group)
        self._colormap_combo = QtWidgets.QComboBox()
        self._colormap_combo.addItems(_COLORMAPS)
        rl.addRow("Colormap :", self._colormap_combo)
        self._scale_combo = QtWidgets.QComboBox()
        self._scale_combo.addItems([s[0] for s in _SCALES])
        rl.addRow("Echelle :", self._scale_combo)
        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(10, 100)
        self._opacity_slider.setValue(100)
        rl.addRow("Opacite :", self._opacity_slider)
        root.addWidget(render_group)

        # -- Signal / options --
        opt_group = QtWidgets.QGroupBox("Options")
        ol = QtWidgets.QVBoxLayout(opt_group)
        self._chk_filtered = QtWidgets.QCheckBox("Utiliser le signal filtre (plutot que brut)")
        self._chk_onef = QtWidgets.QCheckBox("Correction 1/f (met en valeur fuseaux/rapide)")
        self._chk_overlay = QtWidgets.QCheckBox("Superposer le trace d'hypnogramme")
        ol.addWidget(self._chk_filtered)
        ol.addWidget(self._chk_onef)
        ol.addWidget(self._chk_overlay)
        root.addWidget(opt_group)

        # -- Export --
        self._btn_export = QtWidgets.QPushButton("Exporter l'image...")
        root.addWidget(self._btn_export)

        root.addStretch()

        self._connect()
        self._update_enabled_state()

    # ------------------------------------------------------------------

    def _connect(self) -> None:
        self._chk_night.toggled.connect(self._on_night_toggled)
        self._chk_window.toggled.connect(self._on_window_toggled)
        self._night_height_slider.valueChanged.connect(
            lambda v: self.night_height_changed.emit(v / 100.0))
        self._window_height_spin.valueChanged.connect(self.window_height_changed.emit)
        self._channel_combo.currentTextChanged.connect(self._on_channel_changed)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self._fmin_spin.valueChanged.connect(self._on_fmin_changed)
        self._fmax_spin.valueChanged.connect(self._on_fmax_changed)
        self._n_freqs_spin.valueChanged.connect(self.n_freqs_changed.emit)
        self._chk_freq_log.toggled.connect(self.freq_log_toggled.emit)
        self._colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        self._scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        self._opacity_slider.valueChanged.connect(
            lambda v: self.opacity_changed.emit(v / 100.0))
        self._chk_filtered.toggled.connect(self.use_filtered_toggled.emit)
        self._chk_onef.toggled.connect(self.onef_correction_toggled.emit)
        self._chk_overlay.toggled.connect(self.overlay_hypnogram_toggled.emit)
        self._btn_export.clicked.connect(self.export_requested.emit)

    def _on_night_toggled(self, checked: bool) -> None:
        self._update_enabled_state()
        self.night_toggled.emit(checked)

    def _on_window_toggled(self, checked: bool) -> None:
        self._update_enabled_state()
        self.window_toggled.emit(checked)

    def _update_enabled_state(self) -> None:
        any_on = self._chk_night.isChecked() or self._chk_window.isChecked()
        self._night_height_slider.setEnabled(self._chk_night.isChecked())
        self._window_height_spin.setEnabled(self._chk_window.isChecked())
        for w in (
            self._channel_combo, self._preset_combo, self._fmin_spin, self._fmax_spin,
            self._n_freqs_spin, self._chk_freq_log, self._colormap_combo, self._scale_combo,
            self._opacity_slider, self._chk_filtered, self._chk_onef, self._chk_overlay,
            self._btn_export,
        ):
            w.setEnabled(any_on)

    def _on_channel_changed(self, text: str) -> None:
        if text:
            self.channel_changed.emit(text)

    def _on_preset_changed(self, index: int) -> None:
        if index < 0 or index >= len(_FREQ_PRESETS):
            return
        _, fmin, fmax = _FREQ_PRESETS[index]
        if fmin is None:
            return
        self._fmin_spin.blockSignals(True)
        self._fmax_spin.blockSignals(True)
        self._fmin_spin.setValue(fmin)
        self._fmax_spin.setValue(fmax)
        self._fmin_spin.blockSignals(False)
        self._fmax_spin.blockSignals(False)
        self.fmin_changed.emit(fmin)
        self.fmax_changed.emit(fmax)

    def _on_fmin_changed(self, value: float) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(len(_FREQ_PRESETS) - 1)
        self._preset_combo.blockSignals(False)
        self.fmin_changed.emit(value)

    def _on_fmax_changed(self, value: float) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(len(_FREQ_PRESETS) - 1)
        self._preset_combo.blockSignals(False)
        self.fmax_changed.emit(value)

    def _on_scale_changed(self, index: int) -> None:
        if 0 <= index < len(_SCALES):
            self.scale_changed.emit(_SCALES[index][1])

    # ------------------------------------------------------------------
    # Public restore helpers
    # ------------------------------------------------------------------

    def set_channels(self, names: List[str], current: str = "") -> None:
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        self._channel_combo.addItems(list(names))
        if current and current in names:
            self._channel_combo.setCurrentText(current)
        self._channel_combo.blockSignals(False)

    def set_state(
        self,
        *,
        enabled_night: bool,
        enabled_window: bool,
        night_height_ratio: float,
        window_height_px: int,
        fmin: float,
        fmax: float,
        n_freqs: int,
        freq_log: bool,
        colormap: str,
        scale: str,
        use_filtered: bool,
        onef_correction: bool,
        opacity: float,
        overlay_hypnogram: bool,
    ) -> None:
        widgets = [
            self._chk_night, self._chk_window, self._night_height_slider,
            self._window_height_spin, self._preset_combo, self._fmin_spin,
            self._fmax_spin, self._n_freqs_spin, self._chk_freq_log,
            self._colormap_combo, self._scale_combo, self._opacity_slider,
            self._chk_filtered, self._chk_onef, self._chk_overlay,
        ]
        for w in widgets:
            w.blockSignals(True)

        self._chk_night.setChecked(enabled_night)
        self._chk_window.setChecked(enabled_window)
        self._night_height_slider.setValue(int(round(night_height_ratio * 100)))
        self._window_height_spin.setValue(int(window_height_px))
        self._fmin_spin.setValue(fmin)
        self._fmax_spin.setValue(fmax)
        preset_idx = len(_FREQ_PRESETS) - 1
        for i, (_, pf_min, pf_max) in enumerate(_FREQ_PRESETS):
            if pf_min is not None and abs(pf_min - fmin) < 1e-6 and abs(pf_max - fmax) < 1e-6:
                preset_idx = i
                break
        self._preset_combo.setCurrentIndex(preset_idx)
        self._n_freqs_spin.setValue(n_freqs)
        self._chk_freq_log.setChecked(freq_log)
        if colormap in _COLORMAPS:
            self._colormap_combo.setCurrentText(colormap)
        for i, (_, key) in enumerate(_SCALES):
            if key == scale:
                self._scale_combo.setCurrentIndex(i)
                break
        self._opacity_slider.setValue(int(round(opacity * 100)))
        self._chk_filtered.setChecked(use_filtered)
        self._chk_onef.setChecked(onef_correction)
        self._chk_overlay.setChecked(overlay_hypnogram)

        for w in widgets:
            w.blockSignals(False)
        self._update_enabled_state()
