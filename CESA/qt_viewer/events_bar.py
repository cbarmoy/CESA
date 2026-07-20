"""Scoring / events strip under the EEG viewer.

Primary role (Tk parity): show the sleep-stage band for the *visible*
time window — one coloured block per epoch, with short labels (E, N1…).
If the window spans several epochs (> 30 s or not epoch-aligned), all
visible stages appear side by side above the HH:MM:SS time axis.

Clinical annotations (arousals, apneas, …) are drawn as a thin overlay.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .themes import DARK, ThemePalette, event_color, normalize_stage, stage_color
from .time_axis_item import TimeAxisItem

logger = logging.getLogger(__name__)

_SHORT_LABELS = {
    "W": "E",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "R": "R",
    "U": "U",
}

_SLEEP_STAGES = {"W", "N1", "N2", "N3", "R", "U"}


class EventsBar(QtWidgets.QWidget):
    """Compact scoring band synchronised with the EEG viewer.

    Layout (top → bottom): coloured stage blocks + optional clinical
    events, then the shared ``HH:MM:SS`` time axis (horodatage).
    """

    event_clicked = QtCore.Signal(dict)
    stage_clicked = QtCore.Signal(float)  # epoch onset seconds

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._theme: ThemePalette = dict(DARK)
        self._events: List[Dict[str, Any]] = []
        self._hypnogram: List[str] = []
        self._epoch_len: float = 30.0
        self._start_s: float = 0.0
        self._duration_s: float = 30.0

        self.setFixedHeight(64)
        self.setToolTip("Bande de scoring — stades visibles dans la fenetre")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        time_axis = TimeAxisItem(orientation="bottom")
        self._pw = pg.PlotWidget(axisItems={"bottom": time_axis})
        self._plot: pg.PlotItem = self._pw.getPlotItem()
        layout.addWidget(self._pw)

        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideAxis("left")
        self._plot.showAxis("bottom")
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setYRange(0, 1, padding=0)
        self._plot.getAxis("bottom").setHeight(22)

        self._rect_items: List[Any] = []
        self._text_items: List[Any] = []

        self._pw.scene().sigMouseClicked.connect(self._on_click)
        self._apply_theme()

    # ----- public API ---------------------------------------------------

    def set_hypnogram(
        self,
        labels: List[str],
        epoch_len: float = 30.0,
    ) -> None:
        """Drive the stage band from the full-night hypnogram."""
        self._hypnogram = list(labels or [])
        self._epoch_len = max(1.0, float(epoch_len))
        self._redraw()

    def set_events(self, events: List[Dict[str, Any]]) -> None:
        """Set clinical annotation events (arousal, apnea, …).

        Sleep-stage scoring is owned by :meth:`set_hypnogram` and is
        *not* replaced by this call.
        """
        self._events = list(events or [])
        self._redraw()

    def set_time_window(self, start_s: float, duration_s: float) -> None:
        self._start_s = float(start_s)
        self._duration_s = max(0.1, float(duration_s))
        self._plot.setXRange(self._start_s, self._start_s + self._duration_s, padding=0)
        self._redraw()

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)
        self._apply_theme()
        self._redraw()

    # ----- internal -----------------------------------------------------

    def _clear_items(self) -> None:
        for item in self._rect_items:
            self._plot.removeItem(item)
        for item in self._text_items:
            self._plot.removeItem(item)
        self._rect_items.clear()
        self._text_items.clear()

    def _redraw(self) -> None:
        self._clear_items()
        start = self._start_s
        end = start + self._duration_s

        if self._hypnogram:
            self._draw_stage_band(start, end)
        elif self._events:
            # Fallback: treat event descriptions as stages (legacy path)
            self._draw_events_as_stages(start, end)

        self._draw_clinical_events(start, end)

        if not self._hypnogram and not self._events:
            txt = pg.TextItem(
                text="Aucun scoring dans la fenetre",
                color=self._theme.get("text_dim", "#888888"),
                anchor=(0.5, 0.5),
            )
            txt.setFont(QtGui.QFont("Segoe UI", 8))
            txt.setPos(start + self._duration_s * 0.5, 0.5)
            self._plot.addItem(txt)
            self._text_items.append(txt)

    def _draw_stage_band(self, start: float, end: float) -> None:
        ep = self._epoch_len
        n = len(self._hypnogram)
        if n == 0 or ep <= 0:
            return

        first_i = max(0, int(start / ep) - 1)
        last_i = min(n - 1, int(end / ep) + 1)
        fg = self._theme.get("foreground", "#1f2335")
        surface = self._theme.get("surface", "#ffffff")

        for i in range(first_i, last_i + 1):
            epoch_start = i * ep
            epoch_end = epoch_start + ep
            if epoch_end <= start or epoch_start >= end:
                continue
            d_start = max(epoch_start, start)
            d_end = min(epoch_end, end)
            width = d_end - d_start
            if width <= 0:
                continue

            code = normalize_stage(self._hypnogram[i])
            color = stage_color(self._theme, code)
            brush = pg.mkBrush(color + "D9")
            pen = pg.mkPen(self._theme.get("border", "#222222"), width=0.5)

            rect = pg.QtWidgets.QGraphicsRectItem(d_start, 0.08, width, 0.84)
            rect.setBrush(brush)
            rect.setPen(pen)
            self._plot.addItem(rect)
            self._rect_items.append(rect)

            # Label when the visible slice is wide enough
            if width >= max(2.0, self._duration_s * 0.015):
                short = _SHORT_LABELS.get(code, code)
                txt = pg.TextItem(text=short, color=fg, anchor=(0.5, 0.5), fill=pg.mkBrush(surface + "99"))
                txt.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.Bold))
                txt.setPos(d_start + width * 0.5, 0.5)
                self._plot.addItem(txt)
                self._text_items.append(txt)

    def _draw_events_as_stages(self, start: float, end: float) -> None:
        """Legacy: colour blocks from scoring-shaped event dicts."""
        for ev in self._events:
            onset = float(ev.get("onset", 0.0))
            dur = float(ev.get("duration", 1.0))
            if onset + dur < start or onset > end:
                continue
            etype = str(
                ev.get("type") or ev.get("description") or ev.get("label") or "U"
            )
            code = normalize_stage(etype)
            if code not in _SLEEP_STAGES:
                continue
            d_start = max(onset, start)
            d_end = min(onset + dur, end)
            width = d_end - d_start
            if width <= 0:
                continue
            color = stage_color(self._theme, code)
            rect = pg.QtWidgets.QGraphicsRectItem(d_start, 0.08, width, 0.84)
            rect.setBrush(pg.mkBrush(color + "D9"))
            rect.setPen(pg.mkPen(self._theme.get("border", "#222222"), width=0.5))
            self._plot.addItem(rect)
            self._rect_items.append(rect)
            if width >= max(2.0, self._duration_s * 0.015):
                short = _SHORT_LABELS.get(code, code)
                txt = pg.TextItem(
                    text=short,
                    color=self._theme.get("foreground", "#111"),
                    anchor=(0.5, 0.5),
                )
                txt.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.Bold))
                txt.setPos(d_start + width * 0.5, 0.5)
                self._plot.addItem(txt)
                self._text_items.append(txt)

    def _draw_clinical_events(self, start: float, end: float) -> None:
        """Thin overlay lane for non-stage clinical events."""
        for ev in self._events:
            onset = float(ev.get("onset", 0.0))
            dur = max(0.5, float(ev.get("duration", 1.0)))
            if onset + dur < start or onset > end:
                continue
            etype = str(
                ev.get("type") or ev.get("description") or ev.get("label") or "event"
            )
            # Skip pure sleep-stage payloads when hypnogram already drives the band
            if self._hypnogram and normalize_stage(etype) in _SLEEP_STAGES:
                raw = etype.upper().strip()
                if raw in _SLEEP_STAGES or normalize_stage(etype) == raw:
                    continue

            color = event_color(self._theme, etype)
            d_start = max(onset, start)
            d_end = min(onset + dur, end)
            width = d_end - d_start
            if width <= 0:
                continue
            rect = pg.QtWidgets.QGraphicsRectItem(d_start, 0.78, width, 0.18)
            rect.setBrush(pg.mkBrush(color + "EE"))
            rect.setPen(pg.mkPen(color, width=1))
            self._plot.addItem(rect)
            self._rect_items.append(rect)

            label_text = str(ev.get("label") or etype)[:14]
            if width > self._duration_s * 0.04:
                txt = pg.TextItem(text=label_text, color="#FFFFFF", anchor=(0, 0.5))
                txt.setFont(QtGui.QFont("Segoe UI", 7))
                txt.setPos(d_start + width * 0.02, 0.87)
                self._plot.addItem(txt)
                self._text_items.append(txt)

    def _apply_theme(self) -> None:
        self._pw.setBackground(self._theme.get("surface_alt", "#2A2A3C"))
        axis = self._plot.getAxis("bottom")
        fg = self._theme.get("foreground", "#CDD6F4")
        axis.setPen(pg.mkPen(fg))
        axis.setTextPen(pg.mkPen(fg))

    def _on_click(self, event) -> None:
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._plot.vb.mapSceneToView(pos)
        t = float(mouse_point.x())

        for ev in self._events:
            onset = float(ev.get("onset", 0.0))
            dur = float(ev.get("duration", 1.0))
            etype = str(ev.get("type") or ev.get("description") or "")
            if normalize_stage(etype) in _SLEEP_STAGES and etype.upper().strip() in _SLEEP_STAGES:
                continue
            if onset <= t <= onset + dur:
                self.event_clicked.emit(ev)
                return

        if self._hypnogram and self._epoch_len > 0:
            ep_onset = int(t / self._epoch_len) * self._epoch_len
            self.stage_clicked.emit(float(ep_onset))
