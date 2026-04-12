"""ML/DL prediction overlay for the EEG viewer.

Renders per-epoch stage predictions as coloured background rectangles
on the main EEG plot, with optional confidence shading and multi-backend
comparison.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .themes import ThemePalette, DARK, stage_color

logger = logging.getLogger(__name__)


class _PredictionLayer:
    """One set of per-epoch prediction rectangles for a single backend."""

    def __init__(self, plot_item: pg.PlotItem, y_band: Tuple[float, float]) -> None:
        self._plot = plot_item
        self._y_lo, self._y_hi = y_band
        self._rects: List[pg.QtWidgets.QGraphicsRectItem] = []
        self._visible = True

    def update(
        self,
        stages: List[str],
        confidences: List[float],
        epoch_len: float,
        start_s: float,
        end_s: float,
        theme: ThemePalette,
    ) -> None:
        self.clear()
        if not stages:
            return
        for i, stage in enumerate(stages):
            t0 = i * epoch_len
            t1 = t0 + epoch_len
            if t1 < start_s or t0 > end_s:
                continue
            color = stage_color(theme, stage)
            conf = confidences[i] if i < len(confidences) else 1.0
            alpha = max(20, min(80, int(conf * 80)))
            brush = pg.mkBrush(color + f"{alpha:02X}")
            rect = pg.QtWidgets.QGraphicsRectItem(
                t0, self._y_lo, epoch_len, self._y_hi - self._y_lo,
            )
            rect.setBrush(brush)
            rect.setPen(pg.mkPen(None))
            rect.setZValue(-10)
            self._plot.addItem(rect)
            self._rects.append(rect)

    def set_visible(self, vis: bool) -> None:
        self._visible = vis
        for r in self._rects:
            r.setVisible(vis)

    def clear(self) -> None:
        for r in self._rects:
            self._plot.removeItem(r)
        self._rects.clear()


class MLOverlayManager:
    """Manages multiple prediction layers on the EEG viewer plot.

    Supports adding multiple backends (e.g., "rules", "ml", "ml_hmm")
    each rendered as a thin coloured band at the top of the plot.
    """

    def __init__(self, plot_item: pg.PlotItem) -> None:
        self._plot = plot_item
        self._layers: Dict[str, _PredictionLayer] = {}
        self._data: Dict[str, Dict[str, Any]] = {}
        self._theme: ThemePalette = dict(DARK)
        self._epoch_len: float = 30.0
        self._band_height: float = 15.0

    def set_backend_predictions(
        self,
        backend_name: str,
        stages: List[str],
        confidences: Optional[List[float]] = None,
    ) -> None:
        """Set stage predictions for a given backend."""
        if confidences is None:
            confidences = [1.0] * len(stages)
        self._data[backend_name] = {
            "stages": stages,
            "confidences": confidences,
        }

    def set_epoch_length(self, epoch_len: float) -> None:
        self._epoch_len = max(1.0, epoch_len)

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)

    def toggle_backend(self, backend_name: str, visible: bool) -> None:
        layer = self._layers.get(backend_name)
        if layer:
            layer.set_visible(visible)

    def update_visible(self, start_s: float, end_s: float, y_top: float) -> None:
        """Refresh overlay rectangles for the visible window."""
        for i, (name, info) in enumerate(self._data.items()):
            if name not in self._layers:
                y_hi = y_top - i * self._band_height
                y_lo = y_hi - self._band_height * 0.8
                self._layers[name] = _PredictionLayer(self._plot, (y_lo, y_hi))
            self._layers[name].update(
                info["stages"],
                info["confidences"],
                self._epoch_len,
                start_s,
                end_s,
                self._theme,
            )

    def clear(self) -> None:
        for layer in self._layers.values():
            layer.clear()
        self._layers.clear()
        self._data.clear()

    @property
    def backend_names(self) -> List[str]:
        return list(self._data.keys())
