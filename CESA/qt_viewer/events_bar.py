"""Events display bar for arousals, apneas, hypopneas, REM bursts.

Renders clinical events as coloured rectangles with short text labels
inside a compact ``PlotWidget``.  The bar follows the same time axis
as the main EEG viewer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .themes import DARK, ThemePalette, event_color
from .time_axis_item import TimeAxisItem

logger = logging.getLogger(__name__)


class EventsBar(QtWidgets.QWidget):
    """Compact event strip synchronised with the EEG viewer.

    Parameters
    ----------
    parent : QWidget, optional
    """

    event_clicked = QtCore.Signal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._theme: ThemePalette = dict(DARK)
        self._events: List[Dict[str, Any]] = []
        self._start_s: float = 0.0
        self._duration_s: float = 30.0

        self.setFixedHeight(45)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        time_axis = TimeAxisItem(orientation="bottom")
        self._pw = pg.PlotWidget(axisItems={"bottom": time_axis})
        self._plot: pg.PlotItem = self._pw.getPlotItem()
        layout.addWidget(self._pw)

        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setYRange(0, 1, padding=0)

        self._rect_items: List[pg.QtWidgets.QGraphicsRectItem] = []
        self._text_items: List[pg.TextItem] = []

        self._pw.scene().sigMouseClicked.connect(self._on_click)
        self._apply_theme()

    # ----- public API ---------------------------------------------------

    def set_events(self, events: List[Dict[str, Any]]) -> None:
        """Set the list of events.

        Each event dict should have at minimum:
        ``{"onset": float, "duration": float, "type": str}``
        Optional keys: ``"label"``, ``"description"``.
        """
        self._events = list(events)
        self._redraw()

    def set_time_window(self, start_s: float, duration_s: float) -> None:
        self._start_s = start_s
        self._duration_s = duration_s
        self._plot.setXRange(start_s, start_s + duration_s, padding=0)
        self._redraw()

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)
        self._apply_theme()
        self._redraw()

    # ----- internal -----------------------------------------------------

    def _redraw(self) -> None:
        for item in self._rect_items:
            self._plot.removeItem(item)
        for item in self._text_items:
            self._plot.removeItem(item)
        self._rect_items.clear()
        self._text_items.clear()

        start = self._start_s
        end = start + self._duration_s

        for ev in self._events:
            onset = float(ev.get("onset", 0.0))
            dur = float(ev.get("duration", 1.0))
            ev_end = onset + dur
            if ev_end < start or onset > end:
                continue

            etype = str(ev.get("type", "event"))
            color = event_color(self._theme, etype)
            brush = pg.mkBrush(color + "88")  # semi-transparent
            pen = pg.mkPen(color, width=0.5)

            rect = pg.QtWidgets.QGraphicsRectItem(onset, 0.05, dur, 0.9)
            rect.setBrush(brush)
            rect.setPen(pen)
            self._plot.addItem(rect)
            self._rect_items.append(rect)

            label_text = str(ev.get("label", etype))[:12]
            if dur > (self._duration_s * 0.03):
                txt = pg.TextItem(text=label_text, color=self._theme["foreground"],
                                  anchor=(0, 0.5))
                txt.setFont(QtGui.QFont("Segoe UI", 7))
                txt.setPos(onset + dur * 0.02, 0.5)
                self._plot.addItem(txt)
                self._text_items.append(txt)

    def _apply_theme(self) -> None:
        self._pw.setBackground(self._theme["surface_alt"])

    def _on_click(self, event) -> None:
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._plot.vb.mapSceneToView(pos)
        t = mouse_point.x()
        for ev in self._events:
            onset = float(ev.get("onset", 0.0))
            dur = float(ev.get("duration", 1.0))
            if onset <= t <= onset + dur:
                self.event_clicked.emit(ev)
                return
