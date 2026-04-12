"""Global synchronized cursor across all viewer panels.

The ``SyncCursorManager`` propagates a time position to vertical
cursor lines in the EEG viewer, hypnogram bar, and events bar.  It
also displays hover information (time + signal values) in a tooltip
label.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .themes import ThemePalette, DARK


class SyncCursorManager(QtCore.QObject):
    """Manages a synchronised vertical cursor across multiple PlotWidgets.

    Signals
    -------
    cursor_moved(float)
        Emitted when the cursor position changes (seconds).
    """

    cursor_moved = QtCore.Signal(float)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._panels: List[Tuple[pg.PlotItem, pg.InfiniteLine]] = []
        self._time_s: float = 0.0
        self._theme: ThemePalette = dict(DARK)
        self._hover_label: Optional[QtWidgets.QLabel] = None
        self._signals: Dict[str, Tuple[np.ndarray, float]] = {}
        self._epoch_len: float = 30.0

    def register_panel(self, plot_item: pg.PlotItem) -> pg.InfiniteLine:
        """Add a plot panel and return the cursor line added to it."""
        line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(self._theme.get("cursor", "#CBA6F7"),
                         width=1, style=QtCore.Qt.PenStyle.DashLine),
        )
        line.setZValue(100)
        plot_item.addItem(line, ignoreBounds=True)
        line.setVisible(False)
        self._panels.append((plot_item, line))
        return line

    def set_hover_label(self, label: QtWidgets.QLabel) -> None:
        self._hover_label = label

    def set_signals(self, signals: Dict[str, Tuple[np.ndarray, float]]) -> None:
        self._signals = signals

    def set_epoch_length(self, epoch_len: float) -> None:
        self._epoch_len = max(1.0, epoch_len)

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)
        pen = pg.mkPen(theme.get("cursor", "#CBA6F7"),
                       width=1, style=QtCore.Qt.PenStyle.DashLine)
        for _pi, line in self._panels:
            line.setPen(pen)

    def update_position(self, time_s: float) -> None:
        """Move the cursor to *time_s* on all panels."""
        self._time_s = time_s
        for _pi, line in self._panels:
            line.setValue(time_s)
            line.setVisible(True)
        self.cursor_moved.emit(time_s)
        self._update_hover_text(time_s)

    def hide(self) -> None:
        for _pi, line in self._panels:
            line.setVisible(False)
        if self._hover_label:
            self._hover_label.setText("")

    @property
    def time_s(self) -> float:
        return self._time_s

    def epoch_index(self) -> int:
        return int(self._time_s / self._epoch_len) if self._epoch_len > 0 else 0

    def _update_hover_text(self, t: float) -> None:
        if self._hover_label is None:
            return
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        parts = [f"{h:02d}:{m:02d}:{s:05.2f}"]
        parts.append(f"Ep {self.epoch_index()}")

        for name, (data, fs) in list(self._signals.items())[:4]:
            idx = int(t * fs)
            if 0 <= idx < len(data):
                val = data[idx]
                parts.append(f"{name}: {val:.1f} uV")

        self._hover_label.setText("  |  ".join(parts))
