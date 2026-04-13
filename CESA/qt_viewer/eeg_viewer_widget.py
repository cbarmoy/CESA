"""Core multi-channel EEG viewer widget based on PyQtGraph.

``EEGViewerWidget`` is a ``QWidget`` that stacks N channel traces in a
single ``PlotItem`` with a shared time axis, epoch grid, and crosshair
cursor.  It delegates per-channel rendering to :class:`ChannelStrip`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .channel_strip import ChannelStrip
from .downsampler import compute_target_points
from .themes import CHANNEL_TYPE_COLORS, DARK, ThemePalette
from .time_axis_item import TimeAxisItem

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]  # (data_uv, sfreq)

try:
    from CESA.filters import detect_signal_type as cesa_detect_signal_type
except Exception:
    def cesa_detect_signal_type(name: str) -> str:
        n = name.upper()
        if any(k in n for k in ("EOG", "EYE")):
            return "eog"
        if any(k in n for k in ("EMG", "CHIN")):
            return "emg"
        if any(k in n for k in ("ECG", "EKG", "HR")):
            return "ecg"
        return "eeg"


class EEGViewerWidget(QtWidgets.QWidget):
    """High-performance multi-channel EEG viewer.

    Signals (Qt)
    -------------
    time_window_changed(float, float)
        Emitted when the visible window changes (start_s, duration_s).
    channel_clicked(str)
        Emitted when a channel label / trace is clicked.
    cursor_time_changed(float)
        Emitted when the cursor moves (time in seconds).
    annotation_requested(float, float)
        Emitted when the user selects a region for annotation (start_s, end_s).
    """

    time_window_changed = QtCore.Signal(float, float)
    channel_clicked = QtCore.Signal(str)
    cursor_time_changed = QtCore.Signal(float)
    annotation_requested = QtCore.Signal(float, float)

    SPACING_UV = 150.0  # default micro-volt spacing between channels

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._theme: ThemePalette = dict(DARK)
        self._signals: Dict[str, Signal] = {}
        self._channel_order: List[str] = []
        self._strips: Dict[str, ChannelStrip] = {}
        self._channel_types: Dict[str, str] = {}
        self._pipelines: Dict[str, Any] = {}
        self._global_filter: bool = True
        self._active_layout: Optional[Any] = None  # LayoutConfig or None

        self._start_s: float = 0.0
        self._duration_s: float = 30.0
        self._total_duration_s: float = 0.0
        self._epoch_len: float = 30.0
        self._spacing: float = self.SPACING_UV

        # Annotation overlay (lazy)
        self._annotation_overlay: Optional[Any] = None
        # ML overlay (lazy)
        self._ml_overlay: Optional[Any] = None

        # Visual guide state
        self._show_baselines: bool = False
        self._show_amplitude_scale: bool = False
        self._show_grid_fine: bool = False
        self._show_artifact_highlight: bool = False
        self._artifact_threshold: float = 500.0
        self._fine_grid_lines: List[pg.InfiniteLine] = []
        self._artifact_regions: List[pg.LinearRegionItem] = []

        # Selection region for annotation creation
        self._selection_region: Optional[pg.LinearRegionItem] = None
        self._selecting: bool = False

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # PyQtGraph plot widget
        time_axis = TimeAxisItem(orientation="bottom")
        self._pw = pg.PlotWidget(axisItems={"bottom": time_axis})
        self._plot: pg.PlotItem = self._pw.getPlotItem()
        layout.addWidget(self._pw)

        # Configure plot
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.hideAxis("left")
        self._plot.setClipToView(True)
        self._plot.setDownsampling(mode="peak")

        # Crosshair cursor
        self._cursor_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(self._theme["cursor"], width=1,
                         style=QtCore.Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._cursor_line, ignoreBounds=True)
        self._cursor_line.setVisible(False)

        # Epoch grid lines (managed pool)
        self._epoch_lines: List[pg.InfiniteLine] = []

        # Epoch highlight (30s background rect for current epoch)
        self._epoch_highlight = pg.LinearRegionItem(
            values=[0, 30], movable=False,
            brush=pg.mkBrush(255, 255, 255, 12),
            pen=pg.mkPen(None),
        )
        self._epoch_highlight.setZValue(-15)
        self._plot.addItem(self._epoch_highlight, ignoreBounds=True)

        # Mouse tracking for crosshair
        self._proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        # Right-click for annotation creation
        self._pw.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Scroll / zoom via view range change
        self._plot.sigXRangeChanged.connect(self._on_xrange_changed)

        self._apply_theme()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_signals(self, signals: Dict[str, Signal]) -> None:
        """Replace all channel data.  Rebuilds strips."""
        self._signals = dict(signals)
        total_dur = 0.0
        for _name, (data, fs) in signals.items():
            dur = len(data) / fs if fs > 0 else 0.0
            total_dur = max(total_dur, dur)
        self._total_duration_s = total_dur

        if not self._channel_order:
            self._channel_order = list(signals.keys())

        self._rebuild_strips()
        self._refresh_view()

    def set_channel_order(self, names: List[str]) -> None:
        self._channel_order = list(names)
        self._rebuild_strips()
        self._refresh_view()

    def set_time_window(self, start_s: float, duration_s: float) -> None:
        """Programmatically set the visible time window."""
        self._start_s = max(0.0, float(start_s))
        self._duration_s = max(0.5, float(duration_s))
        end = self._start_s + self._duration_s
        self._plot.blockSignals(True)
        self._plot.setXRange(self._start_s, end, padding=0)
        self._plot.blockSignals(False)
        self._refresh_view()

    def set_filter_pipelines(self, pipelines: Dict[str, Any]) -> None:
        self._pipelines = dict(pipelines)
        for name, strip in self._strips.items():
            strip.set_pipeline(pipelines.get(name))
        self._refresh_view()

    def set_global_filter_enabled(self, enabled: bool) -> None:
        self._global_filter = enabled
        for strip in self._strips.values():
            strip.set_filter_enabled(enabled)
        self._refresh_view()

    def set_visible_channels(self, names: List[str]) -> None:
        visible_set = set(names)
        for name, strip in self._strips.items():
            strip.set_visible(name in visible_set)
        self._recompute_offsets()
        self._refresh_view()

    def set_channel_gain(self, name: str, gain: float) -> None:
        strip = self._strips.get(name)
        if strip:
            strip.set_gain(gain)
            self._refresh_view()

    def set_all_gains(self, gain: float) -> None:
        for strip in self._strips.values():
            strip.set_gain(gain)
        self._refresh_view()

    def normalize_gains(self) -> None:
        """Auto-scale each channel so they all fill roughly the same vertical space.

        Uses the 98th percentile of absolute amplitude over the full recording
        for robust estimation (immune to transient artefacts).  Gains remain
        stable when navigating.
        """
        target_half = self._spacing * 0.4
        for name, strip in self._strips.items():
            if not strip.visible:
                continue
            data, fs = self._signals.get(name, (np.array([]), 1.0))
            if data.size < 2:
                continue
            # Subsample for speed on very long recordings (every Nth sample)
            n = data.size
            step = max(1, n // 200_000)
            sub = data[::step]
            p98 = float(np.percentile(np.abs(sub), 98))
            if p98 < 1e-9:
                continue
            gain = target_half / p98
            gain = max(0.01, min(gain, 1e6))
            strip.set_gain(gain)
        self._refresh_view()

    def reset_gains(self) -> None:
        """Reset all channel gains to 1.0."""
        for strip in self._strips.values():
            strip.set_gain(1.0)
        self._refresh_view()

    def apply_scaling(
        self,
        config,
        channel_types: Dict[str, str],
    ) -> None:
        """Apply a full ScalingConfig to all strips.

        Effective gain per channel =
            global_gain * per_type_gain[type] * per_channel_gain.get(ch, 1.0)
        """
        if not config.enabled:
            for strip in self._strips.values():
                strip.set_gain(1.0)
                strip.set_clip(False)
            if abs(self._spacing - config.spacing_uv) > 0.1:
                self.set_spacing(config.spacing_uv)
            else:
                self._refresh_view()
            return

        if abs(self._spacing - config.spacing_uv) > 0.1:
            self._spacing = max(10.0, config.spacing_uv)
            self._recompute_offsets()

        per_type = config.per_type_gains or {}
        per_ch = config.per_channel_gains or {}
        g_global = config.global_gain

        for name, strip in self._strips.items():
            ch_type = channel_types.get(name, "eeg")
            type_gain = per_type.get(ch_type, 1.0)
            ch_gain = per_ch.get(name, 1.0)
            effective = g_global * type_gain * ch_gain
            strip.set_gain(max(0.01, effective))
            strip.set_clip(config.clipping_enabled, config.clip_value_uv)

        self._refresh_view()

    def apply_layout(
        self,
        layout,
        channel_types: Dict[str, str],
        signal_medians: Optional[Dict[str, float]] = None,
    ) -> None:
        """Apply a LayoutConfig: type-aware spacing + optional DC centering."""
        self._active_layout = layout
        self._channel_types.update(channel_types)

        if layout and layout.enabled and layout.center_signal and signal_medians:
            for name, strip in self._strips.items():
                strip.set_center_offset(signal_medians.get(name, 0.0))
        else:
            for strip in self._strips.values():
                strip.set_center_offset(0.0)

        self._recompute_offsets()
        self._refresh_view()

    # ------------------------------------------------------------------
    # Visual guides
    # ------------------------------------------------------------------

    def set_show_baselines(self, show: bool) -> None:
        self._show_baselines = show
        for strip in self._strips.values():
            strip.set_show_baseline(show)
        self._refresh_view()

    def set_show_amplitude_scale(self, show: bool) -> None:
        self._show_amplitude_scale = show
        for strip in self._strips.values():
            strip.set_show_amplitude_scale(show)
        self._refresh_view()

    def set_show_grid_fine(self, show: bool) -> None:
        self._show_grid_fine = show
        self._update_fine_grid()
        self._refresh_view()

    def set_show_artifact_highlight(self, show: bool, threshold: float = 500.0) -> None:
        self._show_artifact_highlight = show
        self._artifact_threshold = threshold
        if not show:
            self._clear_artifact_regions()
        self._refresh_view()

    def _update_fine_grid(self) -> None:
        """Draw fine (1s) and medium (5s) vertical grid lines."""
        for line in self._fine_grid_lines:
            self._plot.removeItem(line)
        self._fine_grid_lines.clear()

        if not self._show_grid_fine:
            return

        start = self._start_s
        end = start + self._duration_s
        t = int(start)
        while t <= end + 1:
            if t >= start - 1:
                is_5s = (t % 5 == 0)
                pen = pg.mkPen(
                    self._theme.get("grid", "#45475A"),
                    width=0.6 if is_5s else 0.3,
                    style=(QtCore.Qt.PenStyle.DashLine if is_5s
                           else QtCore.Qt.PenStyle.DotLine),
                )
                line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
                line.setValue(float(t))
                line.setZValue(-20)
                self._plot.addItem(line, ignoreBounds=True)
                self._fine_grid_lines.append(line)
            t += 1

    def _clear_artifact_regions(self) -> None:
        for r in self._artifact_regions:
            self._plot.removeItem(r)
        self._artifact_regions.clear()

    def _update_artifact_highlights(self) -> None:
        """Detect and highlight epochs where any channel exceeds the threshold."""
        self._clear_artifact_regions()
        if not self._show_artifact_highlight:
            return

        start = self._start_s
        end = start + self._duration_s
        threshold = self._artifact_threshold

        epoch_start = int(start / self._epoch_len) * self._epoch_len
        t = epoch_start
        while t < end + self._epoch_len:
            if t + self._epoch_len <= start:
                t += self._epoch_len
                continue
            has_artifact = False
            for name, strip in self._strips.items():
                if not strip.visible:
                    continue
                sig = self._signals.get(name)
                if sig is None:
                    continue
                data, fs = sig
                s0 = max(0, int(t * fs))
                s1 = min(len(data), int((t + self._epoch_len) * fs))
                if s1 <= s0:
                    continue
                step = max(1, (s1 - s0) // 5000)
                chunk = data[s0:s1:step]
                if float(np.max(np.abs(chunk))) > threshold:
                    has_artifact = True
                    break
            if has_artifact:
                region = pg.LinearRegionItem(
                    values=[t, t + self._epoch_len],
                    movable=False,
                    brush=pg.mkBrush(255, 60, 60, 18),
                    pen=pg.mkPen(None),
                )
                region.setZValue(-18)
                self._plot.addItem(region, ignoreBounds=True)
                self._artifact_regions.append(region)
            t += self._epoch_len

    def set_spacing(self, spacing_uv: float) -> None:
        self._spacing = max(10.0, float(spacing_uv))
        self._recompute_offsets()
        self._refresh_view()

    def set_epoch_length(self, epoch_len: float) -> None:
        self._epoch_len = max(1.0, float(epoch_len))
        self._update_epoch_grid()

    def set_theme(self, theme: ThemePalette) -> None:
        self._theme = dict(theme)
        self._apply_theme()
        self._rebuild_strips()
        self._refresh_view()

    def set_show_raw_underlay(self, show: bool) -> None:
        for strip in self._strips.values():
            strip.set_show_raw(show)
        self._refresh_view()

    def set_total_duration(self, total_s: float) -> None:
        self._total_duration_s = max(0.0, float(total_s))

    @property
    def start_s(self) -> float:
        return self._start_s

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def channel_names(self) -> List[str]:
        return list(self._channel_order)

    @property
    def plot_item(self) -> pg.PlotItem:
        return self._plot

    def set_annotation_overlay(self, overlay: Any) -> None:
        """Attach an AnnotationOverlay for rendering annotations on the plot."""
        self._annotation_overlay = overlay

    def set_ml_overlay(self, overlay: Any) -> None:
        """Attach an MLOverlayManager for rendering ML predictions."""
        self._ml_overlay = overlay

    def start_selection(self, x: float) -> None:
        """Begin a region selection for creating an annotation."""
        if self._selection_region is not None:
            self._plot.removeItem(self._selection_region)
        self._selection_region = pg.LinearRegionItem(
            values=[x, x + 1],
            movable=True,
            brush=pg.mkBrush("#89B4FA20"),
            pen=pg.mkPen("#89B4FA", width=1),
        )
        self._selection_region.setZValue(50)
        self._plot.addItem(self._selection_region)
        self._selecting = True

    def finish_selection(self) -> Optional[Tuple[float, float]]:
        """Finish selection and return (start_s, end_s), or None."""
        if self._selection_region is None:
            return None
        region = self._selection_region.getRegion()
        self._plot.removeItem(self._selection_region)
        self._selection_region = None
        self._selecting = False
        return (min(region), max(region))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_strips(self) -> None:
        """Destroy existing strips and create new ones."""
        for strip in self._strips.values():
            strip.cleanup()
        self._strips.clear()

        visible = [
            n for n in self._channel_order if n in self._signals
        ]
        n_ch = len(visible)
        if n_ch == 0:
            return

        for i, name in enumerate(visible):
            data, fs = self._signals[name]
            ch_type = self._channel_types.get(name) or cesa_detect_signal_type(name)
            color_key = CHANNEL_TYPE_COLORS.get(ch_type, "eeg_color")
            color = self._theme.get(color_key, self._theme["eeg_color"])
            y_off = -i * self._spacing

            strip = ChannelStrip(
                name=name,
                data=data,
                sfreq=fs,
                color=color,
                plot_item=self._plot,
                y_offset=y_off,
            )
            pipeline = self._pipelines.get(name)
            if pipeline is not None:
                strip.set_pipeline(pipeline)
            strip.set_filter_enabled(self._global_filter)
            strip.set_show_baseline(self._show_baselines)
            strip.set_show_amplitude_scale(self._show_amplitude_scale)

            self._strips[name] = strip

        # Y range to show all channels
        y_top = self._spacing
        y_bot = -(n_ch) * self._spacing
        self._plot.setYRange(y_bot, y_top, padding=0.02)

    def _recompute_offsets(self) -> None:
        """Update Y offsets after visibility or order change.

        When a LayoutConfig is active, per-type spacing multipliers are
        applied so that e.g. EEG channels get more vertical room while
        EMG/ECG are compressed.
        """
        layout = self._active_layout
        cumulative = 0.0
        n_visible = 0
        for name in self._channel_order:
            strip = self._strips.get(name)
            if strip is None or not strip.visible:
                continue
            ch_type = self._channel_types.get(name, "eeg")
            if layout and layout.enabled:
                mult = layout.per_type_spacing_multiplier.get(ch_type, 1.0)
                custom = layout.per_channel_offset.get(name, 0.0)
            else:
                mult = 1.0
                custom = 0.0
            strip.set_y_offset(-(cumulative + custom))
            cumulative += self._spacing * mult
            n_visible += 1
        if n_visible > 0:
            self._plot.setYRange(
                -(cumulative + self._spacing * 0.2),
                self._spacing * 0.5,
                padding=0.02,
            )

    def _refresh_view(self) -> None:
        """Push current window data to all strips."""
        w = self._pw.width() or 2000
        for strip in self._strips.values():
            strip.update_view(self._start_s, self._duration_s, w)
        self._update_epoch_grid()
        if self._show_grid_fine:
            self._update_fine_grid()
        if self._show_artifact_highlight:
            self._update_artifact_highlights()

        # Update annotation overlay
        if self._annotation_overlay is not None:
            try:
                self._annotation_overlay.update_visible(
                    self._start_s, self._start_s + self._duration_s
                )
            except Exception:
                pass

        # Update ML overlay
        if self._ml_overlay is not None:
            try:
                y_top = self._spacing
                self._ml_overlay.update_visible(
                    self._start_s, self._start_s + self._duration_s, y_top,
                )
            except Exception:
                pass

    def _update_epoch_grid(self) -> None:
        """Draw vertical epoch boundary lines in the visible window."""
        if self._epoch_len <= 0:
            return

        start = self._start_s
        end = start + self._duration_s
        first_epoch = int(start / self._epoch_len) * self._epoch_len
        positions = []
        t = first_epoch
        while t <= end + self._epoch_len:
            if t >= start - self._epoch_len:
                positions.append(t)
            t += self._epoch_len

        # Reuse / grow pool
        while len(self._epoch_lines) < len(positions):
            line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen(self._theme["epoch_line"], width=0.5,
                             style=QtCore.Qt.PenStyle.DotLine),
            )
            self._plot.addItem(line, ignoreBounds=True)
            self._epoch_lines.append(line)

        for i, line in enumerate(self._epoch_lines):
            if i < len(positions):
                line.setValue(positions[i])
                line.setVisible(True)
            else:
                line.setVisible(False)

    def _apply_theme(self) -> None:
        t = self._theme
        self._pw.setBackground(t["background"])
        self._plot.getAxis("bottom").setPen(pg.mkPen(t["foreground"]))
        self._plot.getAxis("bottom").setTextPen(pg.mkPen(t["foreground"]))
        self._cursor_line.setPen(
            pg.mkPen(t["cursor"], width=1,
                     style=QtCore.Qt.PenStyle.DashLine)
        )
        for line in self._epoch_lines:
            line.setPen(
                pg.mkPen(t["epoch_line"], width=0.5,
                         style=QtCore.Qt.PenStyle.DotLine)
            )

    # ------------------------------------------------------------------
    # Mouse / interaction
    # ------------------------------------------------------------------

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if self._plot.sceneBoundingRect().contains(pos):
            mouse_point = self._plot.vb.mapSceneToView(pos)
            t = mouse_point.x()
            self._cursor_line.setValue(t)
            self._cursor_line.setVisible(True)
            # Update epoch highlight
            epoch_start = int(t / self._epoch_len) * self._epoch_len
            self._epoch_highlight.setRegion(
                [epoch_start, epoch_start + self._epoch_len]
            )
            self.cursor_time_changed.emit(t)
        else:
            self._cursor_line.setVisible(False)

    def _on_mouse_clicked(self, event) -> None:
        """Handle right-click for annotation creation."""
        if event.button() != QtCore.Qt.MouseButton.RightButton:
            return
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._plot.vb.mapSceneToView(pos)
        t = mouse_point.x()
        # Emit annotation request at current epoch
        epoch_start = int(t / self._epoch_len) * self._epoch_len
        self.annotation_requested.emit(epoch_start, epoch_start + self._epoch_len)

    def _on_xrange_changed(self, _vb, x_range) -> None:
        """Called when the user scrolls/zooms interactively."""
        if len(x_range) >= 2:
            new_start = max(0.0, x_range[0])
            new_end = x_range[1]
            dur = max(0.5, new_end - new_start)
            if abs(new_start - self._start_s) > 0.01 or abs(dur - self._duration_s) > 0.01:
                self._start_s = new_start
                self._duration_s = dur
                self._refresh_view()
                self.time_window_changed.emit(self._start_s, self._duration_s)

    # ------------------------------------------------------------------
    # Wheel zoom override: zoom X axis only, centred on cursor
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 0.85 if delta > 0 else 1.0 / 0.85

        # Centre zoom on the mouse x position
        pos = event.position()
        scene_pos = self._pw.mapToScene(pos.toPoint())
        vb = self._plot.vb
        mouse_x = vb.mapSceneToView(scene_pos).x()

        new_dur = max(1.0, min(300.0, self._duration_s * factor))
        ratio = (mouse_x - self._start_s) / self._duration_s if self._duration_s > 0 else 0.5
        new_start = mouse_x - ratio * new_dur
        new_start = max(0.0, min(new_start, self._total_duration_s - new_dur))

        self._start_s = new_start
        self._duration_s = new_dur
        self._plot.setXRange(new_start, new_start + new_dur, padding=0)
        event.accept()
