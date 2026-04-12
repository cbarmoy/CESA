"""Single-channel rendering unit for the EEG viewer.

A ``ChannelStrip`` manages one ``PlotDataItem`` that displays a signal
channel inside a shared ``PlotItem``.  It handles:

* lazy filtering via ``FilterPipeline.apply()`` on the visible window,
* adaptive min-max downsampling,
* per-channel gain (amplitude) scaling,
* optional raw-signal underlay.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from .downsampler import compute_target_points, downsample_minmax

logger = logging.getLogger(__name__)

try:
    from CESA.filter_engine import FilterPipeline
except Exception:
    FilterPipeline = None  # type: ignore[assignment,misc]


class ChannelStrip:
    """Manages the visual representation of a single channel.

    Parameters
    ----------
    name : str
        Human-readable channel name (e.g. ``"C3-M2"``).
    data : np.ndarray
        Full-recording signal in micro-volts.
    sfreq : float
        Sampling frequency in Hz.
    color : str
        Hex colour for the filtered trace.
    plot_item : pg.PlotItem
        Parent plot item where the data item is added.
    y_offset : float
        Vertical offset for stacked display.
    """

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        sfreq: float,
        color: str,
        plot_item: pg.PlotItem,
        y_offset: float = 0.0,
    ) -> None:
        self.name = name
        self._full_data = np.asarray(data, dtype=np.float64)
        self._sfreq = sfreq
        self._color = color
        self._plot_item = plot_item
        self._y_offset = y_offset

        self._gain: float = 1.0
        self._pipeline: Optional[Any] = None  # FilterPipeline
        self._filter_enabled: bool = True
        self._visible: bool = True

        # Filtered-data cache: (start_sample, end_sample, array)
        self._cache: Optional[Tuple[int, int, np.ndarray]] = None

        # PyQtGraph items
        pen_raw = pg.mkPen(color="#6C7086", width=0.6, style=pg.QtCore.Qt.PenStyle.DotLine)
        self._raw_curve = pg.PlotDataItem(pen=pen_raw, skipFiniteCheck=True)
        self._raw_curve.setVisible(False)
        self._filtered_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=color, width=1.0),
            skipFiniteCheck=True,
        )

        plot_item.addItem(self._raw_curve)
        plot_item.addItem(self._filtered_curve)

        # Channel label
        self._label = pg.TextItem(
            text=name,
            color=color,
            anchor=(0, 0.5),
        )
        self._label.setFont(pg.QtGui.QFont("Segoe UI", 8))
        plot_item.addItem(self._label)

    # ----- public setters -----------------------------------------------

    def set_data(self, data: np.ndarray, sfreq: float) -> None:
        self._full_data = np.asarray(data, dtype=np.float64)
        self._sfreq = sfreq
        self._cache = None

    def set_gain(self, gain: float) -> None:
        self._gain = max(0.01, float(gain))

    def set_pipeline(self, pipeline: Optional[Any]) -> None:
        self._pipeline = pipeline
        self._cache = None

    def set_filter_enabled(self, enabled: bool) -> None:
        self._filter_enabled = enabled
        self._cache = None

    def set_color(self, color: str) -> None:
        self._color = color
        self._filtered_curve.setPen(pg.mkPen(color=color, width=1.0))
        self._label.setColor(color)

    def set_y_offset(self, offset: float) -> None:
        self._y_offset = offset

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self._filtered_curve.setVisible(visible)
        self._raw_curve.setVisible(visible and self._raw_curve.isVisible())
        self._label.setVisible(visible)

    def set_show_raw(self, show: bool) -> None:
        self._raw_curve.setVisible(show and self._visible)

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def visible(self) -> bool:
        return self._visible

    # ----- rendering ----------------------------------------------------

    def update_view(
        self,
        start_s: float,
        duration_s: float,
        widget_width_px: int = 2000,
    ) -> None:
        """Recompute and push the visible data segment to the curves."""
        if not self._visible or self._full_data.size == 0:
            self._filtered_curve.setData([], [])
            self._raw_curve.setData([], [])
            return

        fs = self._sfreq
        total_samples = len(self._full_data)
        s0 = max(0, int(start_s * fs))
        s1 = min(total_samples, int((start_s + duration_s) * fs))
        if s1 <= s0:
            self._filtered_curve.setData([], [])
            self._raw_curve.setData([], [])
            return

        raw_slice = self._full_data[s0:s1]

        # Filtered signal (with cache)
        filtered = self._get_filtered(s0, s1, raw_slice)

        # Apply gain
        display = filtered * self._gain + self._y_offset
        raw_display = raw_slice * self._gain + self._y_offset

        # Downsample for rendering
        target_pts = compute_target_points(widget_width_px)
        time_base = np.arange(s1 - s0) / fs + start_s

        idx_f, val_f = downsample_minmax(display, target_pts)
        self._filtered_curve.setData(time_base[idx_f], val_f)

        if self._raw_curve.isVisible():
            idx_r, val_r = downsample_minmax(raw_display, target_pts)
            self._raw_curve.setData(time_base[idx_r], val_r)

        # Position label at the left edge
        self._label.setPos(start_s, self._y_offset)

    # ----- internal -----------------------------------------------------

    def _get_filtered(
        self, s0: int, s1: int, raw_slice: np.ndarray,
    ) -> np.ndarray:
        """Return filtered data for the window, using cache when valid."""
        if not self._filter_enabled or self._pipeline is None:
            return raw_slice

        if self._cache is not None:
            cs0, cs1, cached = self._cache
            if cs0 == s0 and cs1 == s1:
                return cached

        try:
            filtered = self._pipeline.apply(raw_slice, self._sfreq)
        except Exception:
            logger.debug("Filter apply failed for %s, returning raw", self.name)
            filtered = raw_slice

        self._cache = (s0, s1, filtered)
        return filtered

    def cleanup(self) -> None:
        """Remove all items from the plot."""
        try:
            self._plot_item.removeItem(self._filtered_curve)
            self._plot_item.removeItem(self._raw_curve)
            self._plot_item.removeItem(self._label)
        except Exception:
            pass
