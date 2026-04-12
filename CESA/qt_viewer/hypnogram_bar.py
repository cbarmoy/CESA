"""Compact hypnogram display bar for the EEG viewer.

Renders the full-night hypnogram as coloured stage rectangles and
highlights the current visible window.  Click to jump to an epoch.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .themes import DARK, ThemePalette, stage_color

logger = logging.getLogger(__name__)

# Canonical Y position for each stage (higher = lighter sleep)
_STAGE_Y: Dict[str, float] = {
    "W": 5.0,
    "WAKE": 5.0,
    "R": 4.0,
    "REM": 4.0,
    "N1": 3.0,
    "N2": 2.0,
    "N3": 1.0,
    "U": 0.0,
}


class HypnogramBar(QtWidgets.QWidget):
    """Compact hypnogram that emits ``epoch_clicked(float)`` on click.

    Parameters
    ----------
    parent : QWidget, optional
    """

    epoch_clicked = QtCore.Signal(float)
    epoch_context_stage = QtCore.Signal(float, str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._theme: ThemePalette = dict(DARK)
        self._labels: List[str] = []
        self._epoch_len: float = 30.0
        self._total_duration_s: float = 0.0
        self._window_start: float = 0.0
        self._window_dur: float = 30.0

        self.setFixedHeight(70)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._pw = pg.PlotWidget()
        self._plot: pg.PlotItem = self._pw.getPlotItem()
        layout.addWidget(self._pw)

        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()

        # Stage rectangles drawn as BarGraphItem
        self._bars: Optional[pg.BarGraphItem] = None

        # Window highlight region
        self._highlight = pg.LinearRegionItem(
            values=[0, 30], movable=False,
            brush=pg.mkBrush(255, 255, 255, 40),
        )
        self._highlight.setZValue(10)
        self._plot.addItem(self._highlight)

        # Click handler
        self._pw.scene().sigMouseClicked.connect(self._on_click)

        self._apply_theme()

    # ----- public API ---------------------------------------------------

    def set_hypnogram(
        self,
        labels: List[str],
        epoch_len: float,
        total_duration_s: float = 0.0,
    ) -> None:
        """Set hypnogram data.

        Parameters
        ----------
        labels : list of str
            One stage label per epoch (``"W"``, ``"N1"``...).
        epoch_len : float
            Duration of one epoch in seconds.
        total_duration_s : float
            Total recording duration (for X range).
        """
        self._labels = list(labels)
        self._epoch_len = max(1.0, float(epoch_len))
        n = len(labels)
        self._total_duration_s = max(
            total_duration_s, n * self._epoch_len,
        )
        self._redraw_bars()

    def set_window(self, start_s: float, duration_s: float) -> None:
        self._window_start = start_s
        self._window_dur = duration_s
        self._highlight.setRegion([start_s, start_s + duration_s])

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)
        self._apply_theme()
        self._redraw_bars()

    # ----- internal -----------------------------------------------------

    def _redraw_bars(self) -> None:
        if self._bars is not None:
            self._plot.removeItem(self._bars)
            self._bars = None

        if not self._labels:
            return

        n = len(self._labels)
        x = np.arange(n) * self._epoch_len
        heights = np.array(
            [_STAGE_Y.get(s.upper().strip(), 0.0) for s in self._labels],
            dtype=np.float64,
        )
        colors = [
            pg.mkBrush(stage_color(self._theme, s))
            for s in self._labels
        ]

        self._bars = pg.BarGraphItem(
            x0=x,
            width=self._epoch_len,
            height=heights,
            brushes=colors,
            pens=[pg.mkPen(None)] * n,
        )
        self._plot.addItem(self._bars)

        self._plot.setXRange(0, self._total_duration_s, padding=0)
        self._plot.setYRange(-0.5, 6, padding=0)

    def _apply_theme(self) -> None:
        t = self._theme
        self._pw.setBackground(t["surface"])
        self._highlight.setBrush(pg.mkBrush(255, 255, 255, 40))

    def _on_click(self, event) -> None:
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._plot.vb.mapSceneToView(pos)
        t = mouse_point.x()
        epoch_start = max(0.0, int(t / self._epoch_len) * self._epoch_len)
        try:
            btn = event.button()
        except Exception:
            btn = QtCore.Qt.MouseButton.LeftButton
        if btn == QtCore.Qt.MouseButton.RightButton:
            menu = QtWidgets.QMenu(self)
            for st in ("W", "N1", "N2", "N3", "R", "U"):
                act = menu.addAction(f"Scorer {st}")
                act.triggered.connect(
                    lambda _checked=False, es=epoch_start, sts=st: self.epoch_context_stage.emit(float(es), sts),
                )
            menu.exec(QtGui.QCursor.pos())
            return
        if btn == QtCore.Qt.MouseButton.LeftButton:
            self.epoch_clicked.emit(epoch_start)
