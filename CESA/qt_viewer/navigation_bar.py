"""Bottom navigation bar: time slider, duration, epoch buttons, shortcuts.

The bar drives the viewer's visible window via Qt signals without any
knowledge of the rendering internals.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class NavigationBar(QtWidgets.QWidget):
    """Horizontal transport / navigation bar.

    Signals (Qt)
    -------------
    time_changed(float)
        The desired start time changed (from slider or buttons).
    duration_changed(float)
        The window duration changed (zoom).
    play_toggled(bool)
        Play/pause toggled (True = playing).
    filter_toggled()
        Global filter toggle requested.
    """

    time_changed = QtCore.Signal(float)
    duration_changed = QtCore.Signal(float)
    play_toggled = QtCore.Signal(bool)
    filter_toggled = QtCore.Signal()
    normalize_toggled = QtCore.Signal(bool)
    time_slider_released = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._total_duration_s: float = 0.0
        self._epoch_len: float = 30.0
        self._current_time: float = 0.0
        self._duration: float = 30.0
        self._playing: bool = False

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(1000)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()
        self._connect_signals()

    # ----- public setters -----------------------------------------------

    def set_total_duration(self, total_s: float) -> None:
        self._total_duration_s = max(0.0, float(total_s))
        self._slider.setMaximum(int(self._total_duration_s * 10))
        self._update_time_label()

    def set_current_time(self, t: float) -> None:
        self._current_time = max(0.0, float(t))
        self._slider.blockSignals(True)
        self._slider.setValue(int(self._current_time * 10))
        self._slider.blockSignals(False)
        self._update_time_label()

    def set_duration(self, dur: float) -> None:
        self._duration = max(1.0, float(dur))
        self._dur_spin.blockSignals(True)
        self._dur_spin.setValue(int(self._duration))
        self._dur_spin.blockSignals(False)

    def set_epoch_length(self, epoch_len: float) -> None:
        self._epoch_len = max(1.0, float(epoch_len))

    # ----- UI construction ----------------------------------------------

    def _build_ui(self) -> None:
        main = QtWidgets.QHBoxLayout(self)
        main.setContentsMargins(6, 2, 6, 2)
        main.setSpacing(4)

        # -- Transport buttons --
        btn_style = "QPushButton { padding: 2px 6px; }"

        self._btn_home = QtWidgets.QPushButton("\u23EE")
        self._btn_home.setToolTip("Debut (Home)")
        self._btn_home.setStyleSheet(btn_style)

        self._btn_prev_epoch = QtWidgets.QPushButton("\u23EA")
        self._btn_prev_epoch.setToolTip("Epoch precedente (PgUp pas lineaire, Q grille scoring)")
        self._btn_prev_epoch.setStyleSheet(btn_style)

        self._btn_prev = QtWidgets.QPushButton("\u25C0")
        self._btn_prev.setToolTip("Reculer 1s (Left)")
        self._btn_prev.setStyleSheet(btn_style)

        self._btn_play = QtWidgets.QPushButton("\u25B6")
        self._btn_play.setToolTip("Lecture/Pause (Space)")
        self._btn_play.setStyleSheet(btn_style)
        self._btn_play.setCheckable(True)

        self._btn_next = QtWidgets.QPushButton("\u25B6")
        self._btn_next.setToolTip("Avancer 1s (Right)")
        self._btn_next.setStyleSheet(btn_style)

        self._btn_next_epoch = QtWidgets.QPushButton("\u23E9")
        self._btn_next_epoch.setToolTip("Epoch suivante (PgDn pas lineaire, D grille scoring)")
        self._btn_next_epoch.setStyleSheet(btn_style)

        self._btn_end = QtWidgets.QPushButton("\u23ED")
        self._btn_end.setToolTip("Fin (End)")
        self._btn_end.setStyleSheet(btn_style)

        for btn in (
            self._btn_home, self._btn_prev_epoch, self._btn_prev,
            self._btn_play, self._btn_next, self._btn_next_epoch,
            self._btn_end,
        ):
            btn.setFixedWidth(36)
            main.addWidget(btn)

        main.addSpacing(8)

        # -- Time slider --
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(10000)
        self._slider.setSingleStep(10)
        self._slider.setPageStep(300)
        # Ne pas prendre le focus clavier : sinon les flèches déplacent le slider au lieu du viewer.
        self._slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        main.addWidget(self._slider, stretch=1)

        # -- Time label --
        self._time_label = QtWidgets.QLabel("00:00:00")
        self._time_label.setFixedWidth(70)
        self._time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._time_label.setFont(QtGui.QFont("Consolas", 9))
        main.addWidget(self._time_label)

        main.addSpacing(8)

        # -- Duration control --
        main.addWidget(QtWidgets.QLabel("Duree:"))
        self._dur_spin = QtWidgets.QSpinBox()
        self._dur_spin.setRange(1, 300)
        self._dur_spin.setValue(30)
        self._dur_spin.setSuffix(" s")
        self._dur_spin.setFixedWidth(75)
        main.addWidget(self._dur_spin)

        # -- Zoom buttons --
        self._btn_zoom_in = QtWidgets.QPushButton("+")
        self._btn_zoom_in.setToolTip("Zoom in (+)")
        self._btn_zoom_in.setFixedWidth(28)
        self._btn_zoom_out = QtWidgets.QPushButton("-")
        self._btn_zoom_out.setToolTip("Zoom out (-)")
        self._btn_zoom_out.setFixedWidth(28)
        main.addWidget(self._btn_zoom_in)
        main.addWidget(self._btn_zoom_out)

        main.addSpacing(8)

        # -- Filter toggle --
        self._btn_filter = QtWidgets.QPushButton("F")
        self._btn_filter.setToolTip("Toggle filtres (Ctrl+F)")
        self._btn_filter.setCheckable(True)
        self._btn_filter.setChecked(True)
        self._btn_filter.setFixedWidth(28)
        main.addWidget(self._btn_filter)

        # -- Normalize toggle --
        self._btn_normalize = QtWidgets.QPushButton("Normaliser")
        self._btn_normalize.setToolTip("Normaliser les amplitudes (Ctrl+N)")
        self._btn_normalize.setCheckable(True)
        self._btn_normalize.setChecked(False)
        main.addWidget(self._btn_normalize)

    def _connect_signals(self) -> None:
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderReleased.connect(self.time_slider_released.emit)
        self._dur_spin.valueChanged.connect(self._on_duration_spin)
        self._btn_home.clicked.connect(self._go_home)
        self._btn_end.clicked.connect(self._go_end)
        self._btn_prev.clicked.connect(self._step_back)
        self._btn_next.clicked.connect(self._step_fwd)
        self._btn_prev_epoch.clicked.connect(self._prev_epoch)
        self._btn_next_epoch.clicked.connect(self._next_epoch)
        self._btn_play.toggled.connect(self._toggle_play)
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        self._btn_filter.clicked.connect(lambda: self.filter_toggled.emit())
        self._btn_normalize.toggled.connect(self.normalize_toggled.emit)

    # ----- slots --------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        t = value / 10.0
        self._current_time = t
        self._update_time_label()
        self.time_changed.emit(t)

    def _on_duration_spin(self, value: int) -> None:
        self._duration = float(value)
        self.duration_changed.emit(self._duration)

    def _go_home(self) -> None:
        self._emit_time(0.0)

    def _go_end(self) -> None:
        self._emit_time(max(0.0, self._total_duration_s - self._duration))

    def _step_back(self) -> None:
        self._emit_time(max(0.0, self._current_time - 1.0))

    def _step_fwd(self) -> None:
        self._emit_time(self._current_time + 1.0)

    def _prev_epoch(self) -> None:
        self._emit_time(max(0.0, self._current_time - self._epoch_len))

    def _next_epoch(self) -> None:
        self._emit_time(self._current_time + self._epoch_len)

    def _jump_prev_scored_epoch(self) -> None:
        """Début de l'époque de scoring précédente (grille t=0, L=_epoch_len)."""
        L = max(1.0, float(self._epoch_len))
        idx = int(math.floor(self._current_time / L))
        if idx <= 0:
            self._emit_time(0.0)
        else:
            self._emit_time(float((idx - 1) * L))

    def _jump_next_scored_epoch(self) -> None:
        """Début de l'époque de scoring suivante (grille t=0, L=_epoch_len)."""
        L = max(1.0, float(self._epoch_len))
        idx = int(math.floor(self._current_time / L))
        self._emit_time(float((idx + 1) * L))

    def _step_back_window(self) -> None:
        """Recule d'une fenêtre visible (aligne avec +/- durée affichée)."""
        step = max(0.5, float(self._duration))
        self._emit_time(max(0.0, self._current_time - step))

    def _step_fwd_window(self) -> None:
        """Avance d'une fenêtre visible."""
        step = max(0.5, float(self._duration))
        self._emit_time(self._current_time + step)

    def _toggle_play(self, checked: bool) -> None:
        self._playing = checked
        if checked:
            self._btn_play.setText("\u23F8")
            self._play_timer.start()
        else:
            self._btn_play.setText("\u25B6")
            self._play_timer.stop()
        self.play_toggled.emit(checked)

    def _on_play_tick(self) -> None:
        new_t = self._current_time + 1.0
        if new_t >= self._total_duration_s - self._duration:
            self._btn_play.setChecked(False)
            return
        logger.info(
            "[VIEWER-CHK-31] navigation play tick new_t=%.4f total=%.4f dur=%.4f",
            new_t,
            self._total_duration_s,
            self._duration,
        )
        self._emit_time(new_t)

    def _zoom_in(self) -> None:
        new_dur = max(1, int(self._duration * 0.75))
        self._dur_spin.setValue(new_dur)

    def _zoom_out(self) -> None:
        new_dur = min(300, int(self._duration * 1.33))
        self._dur_spin.setValue(new_dur)

    def _emit_time(self, t: float) -> None:
        t = max(0.0, min(t, self._total_duration_s - self._duration))
        self._current_time = t
        self._slider.blockSignals(True)
        self._slider.setValue(int(t * 10))
        self._slider.blockSignals(False)
        self._update_time_label()
        self.time_changed.emit(t)

    def _update_time_label(self) -> None:
        s = self._current_time
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        self._time_label.setText(f"{h:02d}:{m:02d}:{sec:02d}")

    # ----- keyboard shortcuts (forwarded from main window) ---------------

    def handle_key_ints(
        self, key: int, mod: QtCore.Qt.KeyboardModifiers,
    ) -> bool:
        """Traite une touche (sans QKeyEvent) — sûr après QTimer.singleShot depuis le plot."""
        if key == QtCore.Qt.Key.Key_Left:
            self._step_back()
            return True
        if key == QtCore.Qt.Key.Key_Right:
            self._step_fwd()
            return True
        if key == QtCore.Qt.Key.Key_PageUp:
            self._prev_epoch()
            return True
        if key == QtCore.Qt.Key.Key_PageDown:
            self._next_epoch()
            return True
        if key == QtCore.Qt.Key.Key_Home:
            self._go_home()
            return True
        if key == QtCore.Qt.Key.Key_End:
            self._go_end()
            return True
        if key == QtCore.Qt.Key.Key_Space:
            self._btn_play.toggle()
            return True
        if key == QtCore.Qt.Key.Key_Plus or key == QtCore.Qt.Key.Key_Equal:
            self._zoom_in()
            return True
        if key == QtCore.Qt.Key.Key_Minus:
            self._zoom_out()
            return True
        if key == QtCore.Qt.Key.Key_F and mod & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.filter_toggled.emit()
            return True
        if key == QtCore.Qt.Key.Key_N and mod & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._btn_normalize.toggle()
            return True

        # AZERTY: Q/D = grille des époques scorer; Z/S = pas = durée fenêtre (ex. ±30s si fenêtre 30s)
        _combo = (
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.AltModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
            | QtCore.Qt.KeyboardModifier.ShiftModifier
        )
        if not (mod & _combo):
            if key == QtCore.Qt.Key.Key_Q:
                self._jump_prev_scored_epoch()
                return True
            if key == QtCore.Qt.Key.Key_D:
                self._jump_next_scored_epoch()
                return True
            if key == QtCore.Qt.Key.Key_Z:
                self._step_back_window()
                return True
            if key == QtCore.Qt.Key.Key_S:
                self._step_fwd_window()
                return True

        return False

    def handle_key(self, event: QtGui.QKeyEvent) -> bool:
        """Process a key event; return True if handled."""
        return self.handle_key_ints(event.key(), event.modifiers())
