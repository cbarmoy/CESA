"""Wavelet spectrogram display bar, aligned with the hypnogram time axis.

Two instances are used by the main window:

- **Night** band (below the hypnogram): fixed X range over the whole
  recording, with a movable highlight showing the currently visible window
  (mirrors :class:`CESA.qt_viewer.hypnogram_bar.HypnogramBar`).
- **Window** band (below the EEG trace): X range follows the visible
  window directly (recomputed at high resolution on navigation).

:class:`NightSpectrogramWorker` runs the (potentially slow) whole-night
Morlet transform on a background thread so the UI never freezes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .spectrogram_compute import SpectrogramResult, compute_tfr
from .themes import ThemePalette, normalize_stage, stage_color, DARK

logger = logging.getLogger(__name__)


class NightSpectrogramWorker(QtCore.QThread):
    """Background thread computing the whole-night Morlet spectrogram."""

    result_ready = QtCore.Signal(object)  # SpectrogramResult
    failed = QtCore.Signal(str)

    def __init__(
        self,
        signal: np.ndarray,
        fs: float,
        params: Dict[str, Any],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._signal = signal
        self._fs = fs
        self._params = dict(params)

    def run(self) -> None:  # noqa: D102 (Qt override)
        try:
            result = compute_tfr(self._signal, self._fs, **self._params)
            self.result_ready.emit(result)
        except Exception as exc:  # pragma: no cover - background safety net
            logger.exception("Spectrogram (nuit) : echec du calcul")
            self.failed.emit(str(exc))


class SpectrogramBar(QtWidgets.QWidget):
    """Time-frequency (Morlet) power display, X-aligned with the hypnogram.

    Signals
    -------
    navigate_to(float)
        Click on the band: jump the viewer to this time (seconds).
    """

    navigate_to = QtCore.Signal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._theme: ThemePalette = dict(DARK)
        self._follow_window: bool = True  # True: window mode: bind xrange to (start,dur)
        self._total_duration_s: float = 0.0
        self._window_start: float = 0.0
        self._window_dur: float = 30.0
        self._freqs: np.ndarray = np.array([])
        self._times: np.ndarray = np.array([])
        self._opacity: float = 1.0
        self._colormap_name: str = "viridis"
        self._hypno_labels: List[str] = []
        self._hypno_epoch_len: float = 30.0
        self._overlay_enabled: bool = False

        self.setMinimumHeight(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._pw = pg.PlotWidget()
        self._plot: pg.PlotItem = self._pw.getPlotItem()
        layout.addWidget(self._pw)

        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideAxis("bottom")
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.getAxis("left").setWidth(40)
        self._plot.disableAutoRange(axis="x")
        self._plot.setLimits(xMin=0)

        self._image_item = pg.ImageItem()
        self._image_item.setZValue(0)
        self._plot.addItem(self._image_item)

        self._highlight = pg.LinearRegionItem(
            values=[0, 30], movable=False,
            brush=pg.mkBrush(255, 255, 255, 40),
        )
        self._highlight.setZValue(10)
        self._highlight.setVisible(False)
        self._plot.addItem(self._highlight)

        self._overlay_curve = pg.PlotCurveItem(pen=pg.mkPen("#FFFFFF", width=1.5))
        self._overlay_curve.setZValue(5)
        self._overlay_curve.setVisible(False)
        self._plot.addItem(self._overlay_curve)

        self._placeholder = pg.TextItem(
            text="Spectrogramme desactive", color="#888888", anchor=(0.5, 0.5),
        )
        self._placeholder.setZValue(20)
        self._plot.addItem(self._placeholder)

        self._apply_theme()
        self.set_placeholder_text("Spectrogramme desactive")

    # ================================================================
    # Mode
    # ================================================================

    def set_follow_window(self, follow: bool) -> None:
        """True: X range tracks the visible viewer window (fenetre mode).

        False: X range stays fixed to the full recording, and a movable
        highlight shows the current window instead (nuit mode).
        """
        self._follow_window = bool(follow)
        self._highlight.setVisible(not self._follow_window and bool(self._freqs.size))
        # Night band: keep a 40 px left gutter (freq ticks) matching the
        # hypnogram. Window band: hide the left axis so the plot area
        # lines up with the EEG viewer (which also has no left axis).
        left = self._plot.getAxis("left")
        if self._follow_window:
            left.setWidth(0)
            left.setStyle(showValues=False)
            left.setVisible(False)
        else:
            left.setVisible(True)
            left.setWidth(40)
            left.setStyle(showValues=True)

    def _set_night_xrange(self) -> None:
        """Exact [0, total] — same convention as HypnogramBar."""
        total = max(self._total_duration_s, 1.0)
        self._plot.setXRange(0.0, total, padding=0)

    # ================================================================
    # Data
    # ================================================================

    def set_data(self, result: SpectrogramResult, time_offset_s: float = 0.0) -> None:
        self._freqs = np.asarray(result.freqs, dtype=float)
        self._times = np.asarray(result.times, dtype=float) + float(time_offset_s)
        power = np.asarray(result.power, dtype=float)

        n_freqs, n_times = power.shape
        if n_times < 1 or n_freqs < 1:
            self.clear_data()
            return

        self._image_item.setImage(power.T, autoLevels=False)
        self._image_item.setLevels((result.vmin, result.vmax))
        self._apply_colormap()
        self._image_item.setOpacity(self._opacity)

        t0 = float(self._times[0]) if n_times else 0.0
        t1 = float(self._times[-1]) + (self._times[1] - self._times[0] if n_times > 1 else 1.0)
        if not self._follow_window and self._total_duration_s > 0:
            # Stretch to exact [0, total] so start/end match the hypnogram
            # even if Morlet decimation stopped a few samples early.
            t0, t1 = 0.0, self._total_duration_s
        self._image_item.setRect(QtCore.QRectF(t0, 0.0, max(t1 - t0, 1e-6), n_freqs))

        self._update_freq_ticks()
        self._placeholder.setVisible(False)
        self._highlight.setVisible(not self._follow_window)

        if not self._follow_window:
            self._set_night_xrange()
        self._update_overlay()

    def clear_data(self) -> None:
        self._freqs = np.array([])
        self._times = np.array([])
        self._image_item.clear()
        self._highlight.setVisible(False)
        self._overlay_curve.setVisible(False)

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder.setText(text, color=self._theme.get("text_dim", "#888888"))
        self._placeholder.setVisible(not self._freqs.size)
        cx = (
            self._window_start + self._window_dur / 2.0
            if self._follow_window else self._total_duration_s / 2.0
        )
        self._placeholder.setPos(cx, 0.5)

    def _update_freq_ticks(self) -> None:
        n = len(self._freqs)
        if n == 0:
            return
        n_ticks = min(5, n)
        idxs = np.linspace(0, n - 1, n_ticks).astype(int)
        ticks = [(float(i) + 0.5, f"{self._freqs[i]:.1f}") for i in idxs]
        self._plot.getAxis("left").setTicks([ticks])
        self._plot.setYRange(0, n, padding=0)

    def _apply_colormap(self) -> None:
        try:
            cmap = pg.colormap.get(self._colormap_name)
            self._image_item.setColorMap(cmap)
        except Exception:
            logger.debug("Colormap '%s' indisponible", self._colormap_name, exc_info=True)

    # ================================================================
    # Hypnogram overlay
    # ================================================================

    def set_hypnogram_overlay(self, labels: List[str], epoch_len: float, enabled: bool) -> None:
        self._hypno_labels = list(labels or [])
        self._hypno_epoch_len = max(1.0, float(epoch_len))
        self._overlay_enabled = bool(enabled)
        self._update_overlay()

    _OVERLAY_DEPTH = {"W": 0.92, "R": 0.72, "N1": 0.55, "N2": 0.35, "N3": 0.1, "U": 0.5}

    def _update_overlay(self) -> None:
        if not self._overlay_enabled or not self._hypno_labels or not self._freqs.size:
            self._overlay_curve.setVisible(False)
            return
        n = len(self._freqs)
        xs: List[float] = []
        ys: List[float] = []
        ep = self._hypno_epoch_len
        for i, lbl in enumerate(self._hypno_labels):
            code = normalize_stage(lbl)
            depth = self._OVERLAY_DEPTH.get(code, 0.5)
            y = depth * n
            x0, x1 = i * ep, (i + 1) * ep
            xs.extend([x0, x1])
            ys.extend([y, y])
        self._overlay_curve.setData(xs, ys)
        self._overlay_curve.setVisible(True)

    # ================================================================
    # Window / navigation sync
    # ================================================================

    def set_total_duration(self, total_s: float) -> None:
        self._total_duration_s = max(0.0, float(total_s))
        if not self._follow_window:
            if self._freqs.size:
                # Re-stretch the already-computed image to the (possibly
                # updated) full duration, keeping it pixel-aligned.
                self._image_item.setRect(
                    QtCore.QRectF(0.0, 0.0, max(self._total_duration_s, 1e-6), len(self._freqs))
                )
            self._set_night_xrange()

    def set_window(self, start_s: float, duration_s: float) -> None:
        self._window_start = float(start_s)
        self._window_dur = float(duration_s)
        if self._follow_window:
            self._plot.setXRange(start_s, start_s + duration_s, padding=0)
        else:
            self._highlight.setRegion([start_s, start_s + duration_s])
        if not self._freqs.size:
            self._placeholder.setPos(start_s + duration_s / 2.0, 0.5)

    # ================================================================
    # Appearance
    # ================================================================

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        t = self._theme
        bg = t.get("hypno_bg", t.get("surface", "#121826"))
        self._pw.setBackground(bg)
        self._plot.getAxis("left").setTextPen(t.get("text_dim", "#6C7086"))
        win_brush = t.get("hypno_window", "#FFFFFF28")
        self._highlight.setBrush(pg.mkBrush(win_brush))
        border_col = t.get("hypno_window_border", "#FFFFFF50")
        for line in self._highlight.lines:
            line.setPen(pg.mkPen(border_col, width=1.5))
        self._placeholder.setColor(t.get("text_dim", "#888888"))

    def set_colormap(self, name: str) -> None:
        self._colormap_name = name or "viridis"
        if self._freqs.size:
            self._apply_colormap()

    def set_opacity(self, value: float) -> None:
        self._opacity = max(0.0, min(1.0, float(value)))
        self._image_item.setOpacity(self._opacity)

    # ================================================================
    # Export
    # ================================================================

    def export_image(self, path: str) -> bool:
        try:
            from pyqtgraph.exporters import ImageExporter

            exporter = ImageExporter(self._plot)
            exporter.export(path)
            return True
        except Exception:
            logger.exception("Echec de l'export du spectrogramme")
            return False

    # ================================================================
    # Interaction (click to navigate)
    # ================================================================

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._freqs.size:
            scene_pos = self._pw.mapToScene(event.position().toPoint())
            vb = self._plot.vb
            if vb is not None and self._plot.sceneBoundingRect().contains(scene_pos):
                t = vb.mapSceneToView(scene_pos).x()
                self.navigate_to.emit(max(0.0, float(t)))
                return
        super().mousePressEvent(event)
