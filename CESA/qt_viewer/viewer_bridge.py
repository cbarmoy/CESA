"""Compatibility adapter bridging ``PSGPlotter`` API to the Qt viewer.

``ViewerBridge`` exposes the same public method surface that
``eeg_studio_fixed.py`` used to drive the matplotlib ``PSGPlotter``,
but delegates to :class:`EEGViewerMainWindow` instead.

``create_and_launch`` is the entry point called by
``CESA.qt_viewer.launch_viewer()``.  It creates a ``QApplication``
(if needed) and opens the window.

.. note::

   The Tk ``processEvents`` pump has been removed.  The application
   now runs entirely within ``QApplication.exec()``.  Legacy
   ``tk_root`` / ``start_tk_pump`` parameters are accepted but
   ignored for backward compatibility.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]


class ViewerBridge:
    """PSGPlotter-compatible facade for the Qt viewer.

    Calling code that currently does::

        plotter = PSGPlotter(signals=..., ...)
        plotter.set_time_window(t, d)

    can instead do::

        bridge = launch_viewer(signals=..., ...)
        bridge.set_time_window(t, d)

    All calls are forwarded to the underlying
    :class:`EEGViewerMainWindow`.
    """

    def __init__(self, window: Any) -> None:
        self._win = window

    # -- PSGPlotter-compatible public API --------------------------------

    def set_time_window(self, start_time_s: float, duration_s: float) -> None:
        self._win.set_time_window(start_time_s, duration_s)

    def update_signals(self, signals: Dict[str, Signal]) -> None:
        self._win.update_signals(signals)

    def update_preprocessed_signals(self, signals: Dict[str, Signal]) -> None:
        self._win.update_signals(signals)

    def set_hypnogram(self, hypnogram: Optional[Tuple[List[str], float]]) -> None:
        self._win.set_hypnogram(hypnogram)

    def set_global_filter_enabled(self, enabled: bool) -> None:
        self._win.set_global_filter_enabled(enabled)

    def set_theme(self, theme_name: str) -> None:
        self._win.set_theme(theme_name)

    def set_total_duration(self, total_duration_s: float) -> None:
        self._win.set_total_duration(total_duration_s)

    def set_nav_callback(self, callback: Callable[[float], None]) -> None:
        self._win._on_navigate = callback

    def set_baseline_enabled(self, enabled: bool) -> None:
        pass

    def set_autoscale_enabled(self, enabled: bool) -> None:
        pass

    def set_scoring_annotations(self, events: List[Dict[str, Any]]) -> None:
        self._win.set_scoring_annotations(events)

    # -- Advanced features --

    @property
    def annotation_store(self):
        return self._win.annotation_store

    def add_annotation(self, ann) -> str:
        return self._win.add_annotation(ann)

    def remove_annotation(self, ann_id: str) -> None:
        self._win.remove_annotation(ann_id)

    def set_ml_predictions(
        self, backend_name: str, stages: list, confidences=None,
    ) -> None:
        self._win.set_ml_predictions(backend_name, stages, confidences)

    def toggle_ml_backend(self, backend_name: str, visible: bool) -> None:
        self._win.toggle_ml_backend(backend_name, visible)

    def set_epoch_features(self, features: list) -> None:
        self._win.set_epoch_features(features)

    def set_ml_probabilities(self, probs: list) -> None:
        self._win.set_ml_probabilities(probs)

    def set_epoch_decisions(self, decisions: list) -> None:
        self._win.set_epoch_decisions(decisions)

    def update_dashboard(self) -> None:
        self._win.update_dashboard()

    def export_report_html(self, path: str) -> None:
        self._win.export_report_html(path)

    def save_png(self, filepath: str, dpi: int = 150) -> None:
        try:
            pixmap = self._win.grab()
            pixmap.save(filepath, "PNG")
        except Exception as exc:
            logger.warning("PNG export failed: %s", exc)

    def save_pdf(self, filepath: str) -> None:
        try:
            from PySide6 import QtGui
            from PySide6.QtPrintSupport import QPrinter
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(filepath)
            painter = QtGui.QPainter(printer)
            self._win.render(painter)
            painter.end()
        except Exception as exc:
            logger.warning("PDF export failed: %s", exc)

    def export_scoring_csv(self, filepath: str) -> None:
        pass

    @property
    def filter_pipelines_by_channel(self) -> Dict[str, Any]:
        return self._win._viewer._pipelines

    @filter_pipelines_by_channel.setter
    def filter_pipelines_by_channel(self, value: Dict[str, Any]) -> None:
        self._win.set_filter_pipelines(value)

    @property
    def global_filter_enabled(self) -> bool:
        return self._win._viewer._global_filter

    @global_filter_enabled.setter
    def global_filter_enabled(self, value: bool) -> None:
        self._win.set_global_filter_enabled(value)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        try:
            self._win.close()
        except Exception:
            pass

    def is_alive(self) -> bool:
        try:
            return self._win.isVisible()
        except Exception:
            return False

    # -- Legacy Tk pump stubs (no-ops) -----------------------------------

    def start_tk_pump(self, tk_root: Any = None, interval_ms: int = 16,
                      after_pump: Any = None) -> None:
        """No-op: Tk pump removed. Qt runs its own event loop."""
        pass

    def pause_qt_pump(self) -> None:
        """No-op: Tk pump removed."""
        pass

    def resume_qt_pump(self) -> None:
        """No-op: Tk pump removed."""
        pass


# ======================================================================
# Factory
# ======================================================================

def create_and_launch(
    signals: Optional[Dict[str, Signal]] = None,
    *,
    hypnogram: Optional[Tuple[List[str], float]] = None,
    scoring_annotations: Optional[List[Dict[str, Any]]] = None,
    filter_pipelines: Optional[Dict[str, Any]] = None,
    channel_types: Optional[Dict[str, str]] = None,
    global_filter_enabled: bool = True,
    start_time_s: float = 0.0,
    duration_s: float = 30.0,
    total_duration_s: Optional[float] = None,
    theme_name: str = "dark",
    on_navigate: Optional[Callable[[float], None]] = None,
    tk_root: Optional[Any] = None,
    after_qt_pump: Optional[Callable[[], None]] = None,
    on_request_auto_scoring: Optional[Callable[[], None]] = None,
    on_open_filter_config: Optional[Callable[[], None]] = None,
    on_open_manual_scoring_editor: Optional[Callable[[], None]] = None,
    on_request_stage_for_current_epoch: Optional[Callable[[str], None]] = None,
    on_request_stage_at_epoch_time: Optional[Callable[[float, str], None]] = None,
    on_global_filter_toggled: Optional[Callable[[bool], None]] = None,
) -> ViewerBridge:
    """Create and show the Qt EEG viewer, returning a :class:`ViewerBridge`.

    The *tk_root* and *after_qt_pump* parameters are accepted for
    backward compatibility but are ignored -- Qt now runs its own
    event loop.
    """
    from PySide6.QtWidgets import QApplication
    from .main_window import EEGViewerMainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = EEGViewerMainWindow(
        signals=signals,
        hypnogram=hypnogram,
        scoring_annotations=scoring_annotations,
        filter_pipelines=filter_pipelines,
        channel_types=channel_types,
        global_filter_enabled=global_filter_enabled,
        start_time_s=start_time_s,
        duration_s=duration_s,
        total_duration_s=total_duration_s,
        theme_name=theme_name,
        on_navigate=on_navigate,
        on_request_auto_scoring=on_request_auto_scoring,
        on_open_filter_config=on_open_filter_config,
        on_open_manual_scoring_editor=on_open_manual_scoring_editor,
        on_request_stage_for_current_epoch=on_request_stage_for_current_epoch,
        on_request_stage_at_epoch_time=on_request_stage_at_epoch_time,
        on_global_filter_toggled=on_global_filter_toggled,
    )
    window.show()

    bridge = ViewerBridge(window)
    logger.info("Qt viewer launched (standalone Qt event loop)")

    return bridge
