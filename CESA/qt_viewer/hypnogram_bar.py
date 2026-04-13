"""Clinical hypnogram bar — interactive sleep-stage navigation strip.

Renders the full-night hypnogram as colour-coded stage blocks with:
  - click to jump, drag to scroll, wheel to zoom
  - hover tooltip (time + stage + epoch)
  - cursor line synchronised with the EEG viewer
  - sleep-cycle boundary markers
  - right-click context menu for manual scoring
  - rounded epoch rectangles with smooth transitions
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .themes import DARK, ThemePalette, stage_color, normalize_stage

logger = logging.getLogger(__name__)

_STAGE_Y: Dict[str, float] = {
    "W": 5.0,
    "R": 4.0,
    "N1": 3.0,
    "N2": 2.0,
    "N3": 1.0,
    "U": 0.0,
}

_STAGE_LABELS: Dict[str, str] = {
    "W": "Eveil",
    "R": "REM",
    "N1": "N1", "N2": "N2", "N3": "N3",
    "U": "Inconnu",
}

_Y_LABELS: List[Tuple[float, str]] = [
    (5.0, "W"),
    (4.0, "R"),
    (3.0, "N1"),
    (2.0, "N2"),
    (1.0, "N3"),
]


class _RoundedRectItem(QtWidgets.QGraphicsRectItem):
    """Rectangle with rounded corners and configurable opacity."""

    def __init__(
        self, x: float, y: float, w: float, h: float,
        brush: QtGui.QBrush, pen: QtGui.QPen,
        radius: float = 2.0, opacity: float = 1.0,
    ) -> None:
        super().__init__(x, y, w, h)
        self._brush = brush
        self._pen = pen
        self._radius = radius
        self.setOpacity(opacity)

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(self._brush)
        painter.setPen(self._pen)
        painter.drawRoundedRect(self.rect(), self._radius, self._radius)


class HypnogramBar(QtWidgets.QWidget):
    """Interactive clinical hypnogram with drag, zoom and hover.

    Signals
    -------
    epoch_clicked(float)
        Left-click: jump to epoch onset in seconds.
    epoch_context_stage(float, str)
        Right-click context menu: assign *stage* at *onset*.
    navigate_to(float)
        Drag/scroll navigation: move viewer to *time_s*.
    zoom_requested(float)
        Wheel zoom: requested new window duration.
    """

    epoch_clicked = QtCore.Signal(float)
    epoch_context_stage = QtCore.Signal(float, str)
    navigate_to = QtCore.Signal(float)
    zoom_requested = QtCore.Signal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._theme: ThemePalette = dict(DARK)
        self._labels: List[str] = []
        self._confidences: List[float] = []
        self._epoch_len: float = 30.0
        self._total_duration_s: float = 0.0
        self._window_start: float = 0.0
        self._window_dur: float = 30.0

        self._drag_active: bool = False
        self._drag_origin_x: float = 0.0
        self._drag_origin_start: float = 0.0

        self.setFixedHeight(90)
        self.setMouseTracking(True)

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

        self._pw.viewport().setMouseTracking(True)
        self._pw.viewport().installEventFilter(self)

        self._stage_items: list = []
        self._cycle_lines: list = []
        self._y_labels: list = []

        # Window highlight
        self._highlight = pg.LinearRegionItem(
            values=[0, 30], movable=False,
            brush=pg.mkBrush(255, 255, 255, 40),
        )
        self._highlight.setZValue(10)
        self._plot.addItem(self._highlight)

        # Hover tooltip
        self._tooltip_label = QtWidgets.QLabel(self)
        self._tooltip_label.setStyleSheet(
            "background: rgba(30,30,46,0.92); color: #CDD6F4; "
            "padding: 4px 8px; border-radius: 4px; font-size: 11px;"
        )
        self._tooltip_label.setVisible(False)
        self._tooltip_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._pw.scene().sigMouseClicked.connect(self._on_click)

        self._apply_theme()

    # ================================================================
    # Public API (backwards-compatible)
    # ================================================================

    def set_hypnogram(
        self,
        labels: List[str],
        epoch_len: float,
        total_duration_s: float = 0.0,
    ) -> None:
        self._labels = list(labels)
        self._epoch_len = max(1.0, float(epoch_len))
        n = len(labels)
        self._total_duration_s = max(total_duration_s, n * self._epoch_len)
        self._confidences = []
        self._redraw()

    def set_hypnogram_with_confidence(
        self,
        labels: List[str],
        confidences: List[float],
        epoch_len: float,
        total_duration_s: float = 0.0,
    ) -> None:
        self._labels = list(labels)
        self._confidences = list(confidences)
        self._epoch_len = max(1.0, float(epoch_len))
        n = len(labels)
        self._total_duration_s = max(total_duration_s, n * self._epoch_len)
        self._redraw()

    def set_window(self, start_s: float, duration_s: float) -> None:
        self._window_start = start_s
        self._window_dur = duration_s
        self._highlight.setRegion([start_s, start_s + duration_s])

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)
        self._apply_theme()
        self._redraw()

    # ================================================================
    # Drawing
    # ================================================================

    def _redraw(self) -> None:
        for item in self._stage_items:
            self._plot.removeItem(item)
        for item in self._cycle_lines:
            self._plot.removeItem(item)
        for item in self._y_labels:
            self._plot.removeItem(item)
        self._stage_items.clear()
        self._cycle_lines.clear()
        self._y_labels.clear()

        if not self._labels:
            return

        n = len(self._labels)
        ep = self._epoch_len
        theme = self._theme
        has_conf = len(self._confidences) == n

        # --- Y-axis stage labels (left side) ---
        for y_pos, stage_name in _Y_LABELS:
            txt = pg.TextItem(
                text=stage_name,
                color=theme.get("text_dim", "#6C7086"),
                anchor=(1.0, 0.5),
            )
            txt.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.Bold))
            txt.setPos(-self._total_duration_s * 0.005, y_pos + 0.5)
            txt.setZValue(20)
            self._plot.addItem(txt)
            self._y_labels.append(txt)

        # --- Stage rectangles ---
        for i in range(n):
            s = normalize_stage(self._labels[i])
            x0 = i * ep
            h = _STAGE_Y.get(s, 0.0) + 0.8
            col_hex = stage_color(theme, s)

            opacity = 1.0
            if has_conf:
                opacity = max(0.3, min(1.0, self._confidences[i]))

            brush = pg.mkBrush(col_hex)
            pen = pg.mkPen(col_hex, width=0.5)

            rect = _RoundedRectItem(
                x0, 0.0, ep, h,
                brush=brush, pen=pen,
                radius=1.5, opacity=opacity,
            )
            rect.setZValue(1)
            self._plot.addItem(rect)
            self._stage_items.append(rect)

        # --- Transition highlights (stage changes) ---
        for i in range(1, n):
            prev = normalize_stage(self._labels[i - 1])
            curr = normalize_stage(self._labels[i])
            if prev != curr:
                x_t = i * ep
                pen_col = theme.get("hypno_grid", "#2A2F3A")
                line = pg.InfiniteLine(
                    pos=x_t, angle=90,
                    pen=pg.mkPen(pen_col, width=0.5,
                                 style=QtCore.Qt.PenStyle.DotLine),
                )
                line.setZValue(2)
                self._plot.addItem(line)
                self._stage_items.append(line)

        # --- Sleep cycle boundaries ---
        self._draw_cycle_markers()

        # --- Horizontal grid lines for each stage level ---
        for y_pos, _ in _Y_LABELS:
            grid_line = pg.InfiniteLine(
                pos=y_pos, angle=0,
                pen=pg.mkPen(
                    theme.get("hypno_grid", "#2A2F3A"),
                    width=0.5,
                    style=QtCore.Qt.PenStyle.DotLine,
                ),
            )
            grid_line.setZValue(0)
            self._plot.addItem(grid_line)
            self._stage_items.append(grid_line)

        # --- Time grid (hourly markers) ---
        total = self._total_duration_s
        hour_s = 3600.0
        t = hour_s
        while t < total:
            vline = pg.InfiniteLine(
                pos=t, angle=90,
                pen=pg.mkPen(
                    theme.get("hypno_grid", "#2A2F3A"),
                    width=0.5,
                ),
            )
            vline.setZValue(0)
            self._plot.addItem(vline)
            self._stage_items.append(vline)

            h_val = int(t / 3600)
            lbl = pg.TextItem(
                text=f"{h_val}h",
                color=theme.get("text_dim", "#6C7086"),
                anchor=(0.5, 1.0),
            )
            lbl.setFont(QtGui.QFont("Segoe UI", 7))
            lbl.setPos(t, -0.3)
            lbl.setZValue(5)
            self._plot.addItem(lbl)
            self._stage_items.append(lbl)
            t += hour_s

        self._plot.setXRange(
            -self._total_duration_s * 0.02,
            self._total_duration_s * 1.01,
            padding=0,
        )
        self._plot.setYRange(-0.8, 6.5, padding=0)

    def _draw_cycle_markers(self) -> None:
        """Detect and mark approximate sleep cycle boundaries.

        A new cycle starts when the hypnogram transitions from REM (or W
        after a REM block) back down into NREM.
        """
        if len(self._labels) < 10:
            return

        ep = self._epoch_len
        theme = self._theme
        in_rem = False
        cycle_num = 0

        for i, lbl in enumerate(self._labels):
            s = normalize_stage(lbl)
            if s == "R":
                in_rem = True
            elif in_rem and s in ("N1", "N2", "N3"):
                in_rem = False
                cycle_num += 1
                x_c = i * ep
                line = pg.InfiniteLine(
                    pos=x_c, angle=90,
                    pen=pg.mkPen(
                        theme.get("hypno_cycle_line", "#FFFFFF30"),
                        width=1.5,
                        style=QtCore.Qt.PenStyle.DashLine,
                    ),
                )
                line.setZValue(3)
                self._plot.addItem(line)
                self._cycle_lines.append(line)

                lbl_item = pg.TextItem(
                    text=f"C{cycle_num + 1}",
                    color=theme.get("text_dim", "#6C7086"),
                    anchor=(0.5, 0.0),
                )
                lbl_item.setFont(QtGui.QFont("Segoe UI", 7, QtGui.QFont.Weight.Bold))
                lbl_item.setPos(x_c, 6.0)
                lbl_item.setZValue(5)
                self._plot.addItem(lbl_item)
                self._cycle_lines.append(lbl_item)

    # ================================================================
    # Theme
    # ================================================================

    def _apply_theme(self) -> None:
        t = self._theme
        bg = t.get("hypno_bg", t.get("surface", "#121826"))
        self._pw.setBackground(bg)

        win_brush = t.get("hypno_window", "#FFFFFF28")
        self._highlight.setBrush(pg.mkBrush(win_brush))
        border_col = t.get("hypno_window_border", "#FFFFFF50")
        for line in self._highlight.lines:
            line.setPen(pg.mkPen(border_col, width=1.5))

        is_dark = t.get("background", "#1E1E2E").startswith("#1")
        if is_dark:
            self._tooltip_label.setStyleSheet(
                "background: rgba(30,30,46,0.92); color: #CDD6F4; "
                "padding: 4px 8px; border-radius: 4px; font-size: 11px;"
            )
        else:
            self._tooltip_label.setStyleSheet(
                "background: rgba(255,255,255,0.95); color: #1E293B; "
                "padding: 4px 8px; border-radius: 4px; font-size: 11px; "
                "border: 1px solid #CBD5E1;"
            )

    # ================================================================
    # Interactions
    # ================================================================

    def eventFilter(self, obj, event: QtCore.QEvent) -> bool:
        if obj is self._pw.viewport():
            etype = event.type()

            if etype == QtCore.QEvent.Type.MouseMove:
                self._handle_mouse_move(event)
                return False

            if etype == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._drag_active = True
                    self._drag_origin_x = event.position().x()
                    self._drag_origin_start = self._window_start
                    return False

            if etype == QtCore.QEvent.Type.MouseButtonRelease:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    if self._drag_active:
                        dx = abs(event.position().x() - self._drag_origin_x)
                        self._drag_active = False
                        if dx < 4:
                            return False
                        return True
                    return False

            if etype == QtCore.QEvent.Type.Wheel:
                self._handle_wheel(event)
                return True

            if etype == QtCore.QEvent.Type.Leave:
                self._tooltip_label.setVisible(False)
                return False

        return super().eventFilter(obj, event)

    def _handle_mouse_move(self, event) -> None:
        pos = event.position()
        scene_pos = self._pw.mapToScene(pos.toPoint())
        vb = self._plot.vb
        if vb is None:
            return
        mouse_point = vb.mapSceneToView(scene_pos)
        t = mouse_point.x()

        if self._drag_active:
            pixels_per_sec = self._pw.width() / max(1.0, self._total_duration_s)
            dx_pixels = pos.x() - self._drag_origin_x
            dt = dx_pixels / max(0.01, pixels_per_sec)
            new_start = self._drag_origin_start - dt
            new_start = max(0.0, min(new_start, self._total_duration_s - self._window_dur))
            self.navigate_to.emit(new_start)
            return

        # Hover tooltip
        if 0 <= t <= self._total_duration_s and self._labels:
            ep_idx = int(t / self._epoch_len)
            ep_idx = max(0, min(ep_idx, len(self._labels) - 1))
            stage = normalize_stage(self._labels[ep_idx])
            stage_disp = _STAGE_LABELS.get(stage, stage)

            hours = int(t // 3600)
            mins = int((t % 3600) // 60)
            secs = int(t % 60)
            time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

            parts = [f"<b>{stage_disp}</b>", f"Ep {ep_idx}", time_str]

            if self._confidences and ep_idx < len(self._confidences):
                conf = self._confidences[ep_idx]
                parts.append(f"Conf: {conf:.0%}")

            self._tooltip_label.setText("  &middot;  ".join(parts))
            self._tooltip_label.adjustSize()

            lx = int(pos.x()) + 12
            ly = int(pos.y()) - self._tooltip_label.height() - 4
            if ly < 0:
                ly = int(pos.y()) + 12
            if lx + self._tooltip_label.width() > self.width():
                lx = int(pos.x()) - self._tooltip_label.width() - 12

            pw_pos = self._pw.mapTo(self, pos.toPoint())
            self._tooltip_label.move(pw_pos.x() + 12, max(2, pw_pos.y() - self._tooltip_label.height() - 4))
            self._tooltip_label.setVisible(True)
        else:
            self._tooltip_label.setVisible(False)

    def _handle_wheel(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 0.8 if delta > 0 else 1.25
        new_dur = self._window_dur * factor
        new_dur = max(10.0, min(new_dur, self._total_duration_s))
        self.zoom_requested.emit(new_dur)

    def _on_click(self, event) -> None:
        if self._drag_active:
            return

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
            self._show_context_menu(epoch_start)
            return

        if btn == QtCore.Qt.MouseButton.LeftButton:
            self.epoch_clicked.emit(epoch_start)

    def _show_context_menu(self, epoch_start: float) -> None:
        ep_idx = int(epoch_start / self._epoch_len)
        current = ""
        if 0 <= ep_idx < len(self._labels):
            current = normalize_stage(self._labels[ep_idx])

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #313244; color: #CDD6F4; border: 1px solid #585B70; }"
            "QMenu::item:selected { background: #45475A; }"
            "QMenu::separator { background: #585B70; height: 1px; }"
        )

        hours = int(epoch_start // 3600)
        mins = int((epoch_start % 3600) // 60)
        header = menu.addAction(f"Ep {ep_idx}  ({hours:02d}:{mins:02d})")
        header.setEnabled(False)
        menu.addSeparator()

        for st, label in [("W", "Wake"), ("N1", "N1"), ("N2", "N2"),
                          ("N3", "N3"), ("R", "REM"), ("U", "Inconnu")]:
            act = menu.addAction(f"  {label}")
            if st == current:
                f = act.font()
                f.setBold(True)
                act.setFont(f)
            act.triggered.connect(
                lambda _c=False, es=epoch_start, sts=st:
                    self.epoch_context_stage.emit(float(es), sts)
            )

        menu.exec(QtGui.QCursor.pos())

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Double-click: zoom to fit ~one sleep cycle (~90 min)."""
        scene_pos = self._pw.mapToScene(event.position().toPoint())
        mouse_point = self._plot.vb.mapSceneToView(scene_pos)
        center_t = mouse_point.x()
        cycle_dur = 90 * 60.0
        half = cycle_dur / 2.0
        start = max(0.0, center_t - half)
        self.zoom_requested.emit(cycle_dur)
        self.navigate_to.emit(start)
