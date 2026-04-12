"""QMainWindow shell that composes all viewer sub-widgets.

Layout
------
+---------------------------------------------+
|  Menu bar                                    |
+--------+------------------------------------+
| Channel|  HypnogramBar                      |
| list   |  EEGViewerWidget                   |
|        |  EventsBar                         |
| Gain   |  NavigationBar                     |
+--------+------------------------------------+
|  Status bar                                  |
+---------------------------------------------+
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .annotations import (
    Annotation, AnnotationDialog, AnnotationOverlay, AnnotationStore,
)
from .dashboard_panel import DashboardPanel
from .eeg_viewer_widget import EEGViewerWidget
from .events_bar import EventsBar
from .filter_metrics import count_effective_filter_channels
from .hypnogram_bar import HypnogramBar
from .inspection_panel import InspectionPanel
from .ml_overlay import MLOverlayManager
from .navigation_bar import NavigationBar
from .report_export import ReportBuilder
from .smart_navigation import SmartNavWidget
from .sync_cursor import SyncCursorManager
from .themes import CHANNEL_TYPE_COLORS, DARK, LIGHT, THEMES, ThemePalette

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

# Modificateurs qui doivent bloquer N/P/T et Q/D/Z/S (NumLock / keypad ne comptent pas).
_NAV_COMBO_MODIFIERS = (
    QtCore.Qt.KeyboardModifier.ControlModifier
    | QtCore.Qt.KeyboardModifier.AltModifier
    | QtCore.Qt.KeyboardModifier.MetaModifier
    | QtCore.Qt.KeyboardModifier.ShiftModifier
)


def _mods_block_combo_nav(mod: QtCore.Qt.KeyboardModifiers) -> bool:
    return bool(mod & _NAV_COMBO_MODIFIERS)


def _keyboard_modifiers_int(*parts: QtCore.Qt.KeyboardModifier) -> int:
    """Masque entier pour _dispatch_key_shortcuts_ints (Py3.14 : int(enum) interdit)."""
    total = 0
    for p in parts:
        v = getattr(p, "value", None)
        if isinstance(v, int):
            total |= v
        elif v is not None:
            total |= int(v)
        else:
            raise TypeError(f"Cannot convert Qt keyboard modifier to int: {p!r}")
    return total


def _qt_key_int(key: QtCore.Qt.Key) -> int:
    """Code touche Qt en int (Py3.14 : int(Qt.Key) interdit sur l'enum Shiboken)."""
    v = getattr(key, "value", None)
    if isinstance(v, int):
        return v
    if v is not None:
        return int(v)
    raise TypeError(f"Cannot convert Qt key to int: {key!r}")

try:
    from CESA.filters import detect_signal_type as cesa_detect_signal_type
except Exception:
    def cesa_detect_signal_type(name: str) -> str:  # type: ignore[misc]
        n = name.upper()
        if any(k in n for k in ("EOG", "EYE")):
            return "eog"
        if any(k in n for k in ("EMG", "CHIN")):
            return "emg"
        if any(k in n for k in ("ECG", "EKG")):
            return "ecg"
        return "eeg"


class _ChannelListWidget(QtWidgets.QWidget):
    """Left sidebar: channel visibility checkboxes, reorder, gain slider."""

    channel_visibility_changed = QtCore.Signal()
    gain_changed = QtCore.Signal(str, float)
    all_gain_changed = QtCore.Signal(float)
    order_changed = QtCore.Signal(list)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(200)

        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)

        lbl = QtWidgets.QLabel("Canaux")
        lbl.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.Bold))
        main.addWidget(lbl)

        self._list = QtWidgets.QListWidget()
        self._list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        # Pas de focus clavier par défaut : évite ↑/↓ et lettres qui changent de canal.
        self._list.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        main.addWidget(self._list, stretch=1)

        # Global gain slider
        gain_frame = QtWidgets.QGroupBox("Gain global")
        gain_lay = QtWidgets.QHBoxLayout(gain_frame)
        gain_lay.setContentsMargins(4, 4, 4, 4)
        self._gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._gain_slider.setRange(1, 200)
        self._gain_slider.setValue(100)
        self._gain_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._gain_slider.setTickInterval(25)
        self._gain_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        gain_lay.addWidget(self._gain_slider)
        self._gain_label = QtWidgets.QLabel("1.0x")
        self._gain_label.setFixedWidth(40)
        gain_lay.addWidget(self._gain_label)
        main.addWidget(gain_frame)

        self._gain_slider.valueChanged.connect(self._on_gain_slider)

        # Spacing slider
        space_frame = QtWidgets.QGroupBox("Espacement")
        space_lay = QtWidgets.QHBoxLayout(space_frame)
        space_lay.setContentsMargins(4, 4, 4, 4)
        self._space_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._space_slider.setRange(20, 1000)
        self._space_slider.setValue(150)
        self._space_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        space_lay.addWidget(self._space_slider)
        self._space_label = QtWidgets.QLabel("150")
        self._space_label.setFixedWidth(35)
        space_lay.addWidget(self._space_label)
        main.addWidget(space_frame)

        self._channels: List[str] = []

    def set_channels(self, names: List[str], types: Dict[str, str]) -> None:
        self._list.clear()
        self._channels = list(names)
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            ch_type = types.get(name, "eeg")
            color_key = CHANNEL_TYPE_COLORS.get(ch_type, "eeg_color")
            color = DARK.get(color_key, "#89B4FA")
            item.setForeground(QtGui.QColor(color))
            self._list.addItem(item)
        self._list.itemChanged.connect(self._on_item_changed)

    def visible_channels(self) -> List[str]:
        out: List[str] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def channel_order(self) -> List[str]:
        return [self._list.item(i).text() for i in range(self._list.count())]

    @property
    def spacing_slider(self) -> QtWidgets.QSlider:
        return self._space_slider

    @property
    def spacing_label(self) -> QtWidgets.QLabel:
        return self._space_label

    def _on_item_changed(self, _item) -> None:
        self.channel_visibility_changed.emit()

    def _on_gain_slider(self, value: int) -> None:
        gain = value / 100.0
        self._gain_label.setText(f"{gain:.1f}x")
        self.all_gain_changed.emit(gain)

    def _on_rows_moved(self, *_args) -> None:
        self.order_changed.emit(self.channel_order())


class EEGViewerMainWindow(QtWidgets.QMainWindow):
    """Top-level Qt window for the CESA EEG viewer.

    Composes :class:`EEGViewerWidget`, :class:`HypnogramBar`,
    :class:`EventsBar`, :class:`NavigationBar` and a channel sidebar.

    When used as the sole application window (no Tk), call
    :meth:`set_app_controller` after construction to wire the full
    menu bar (File / Analysis / Sleep / Tools / Help).
    """

    def __init__(
        self,
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
        on_request_auto_scoring: Optional[Callable[[], None]] = None,
        on_open_filter_config: Optional[Callable[[], None]] = None,
        on_open_manual_scoring_editor: Optional[Callable[[], None]] = None,
        on_request_stage_for_current_epoch: Optional[Callable[[str], None]] = None,
        on_request_stage_at_epoch_time: Optional[Callable[[float, str], None]] = None,
        on_global_filter_toggled: Optional[Callable[[bool], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CESA (Complex EEG Studio Analysis) v0.0beta1.0")
        self.resize(1500, 900)
        self.setMinimumSize(900, 500)

        self._app_controller = None  # set later via set_app_controller

        self._theme_name = theme_name
        self._theme: ThemePalette = dict(THEMES.get(theme_name, DARK))
        self._on_navigate = on_navigate
        self._on_request_auto_scoring = on_request_auto_scoring
        self._on_open_filter_config = on_open_filter_config
        self._on_open_manual_scoring_editor = on_open_manual_scoring_editor
        self._on_request_stage_for_current_epoch = on_request_stage_for_current_epoch
        self._on_request_stage_at_epoch_time = on_request_stage_at_epoch_time
        self._on_global_filter_toggled = on_global_filter_toggled
        self._pending_tk_nav_t: Optional[float] = None
        self._tk_nav_timer = QtCore.QTimer(self)
        self._tk_nav_timer.setSingleShot(True)
        self._tk_nav_timer.setInterval(100)
        self._tk_nav_timer.timeout.connect(self._flush_pending_tk_navigation)
        self._signals_data: Dict[str, Signal] = {}
        self._hypnogram_labels: List[str] = []
        self._epoch_len: float = 30.0

        # -- Core widgets --
        self._hypno = HypnogramBar()
        self._viewer = EEGViewerWidget()
        self._events = EventsBar()
        self._nav = NavigationBar()
        self._channel_list = _ChannelListWidget()

        # -- Annotation system --
        self._annotation_store = AnnotationStore()
        self._annotation_overlay = AnnotationOverlay(
            self._viewer.plot_item, self._annotation_store,
        )
        self._viewer.set_annotation_overlay(self._annotation_overlay)

        # -- ML overlay --
        self._ml_overlay = MLOverlayManager(self._viewer.plot_item)
        self._viewer.set_ml_overlay(self._ml_overlay)

        # -- Synchronized cursor --
        self._sync_cursor = SyncCursorManager(self)
        self._sync_cursor.register_panel(self._hypno._plot)
        self._sync_cursor.register_panel(self._events._plot)

        # -- Smart navigation --
        self._smart_nav = SmartNavWidget()

        # -- Inspection panel (dock) --
        self._inspection = InspectionPanel(self)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._inspection,
        )
        self._inspection.setVisible(False)

        # -- Dashboard panel (dock) --
        self._dashboard = DashboardPanel(self)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._dashboard,
        )
        self._dashboard.setVisible(False)

        # Central widget layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Left sidebar
        outer.addWidget(self._channel_list)

        # Right content area
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self._hypno)
        self._scoring_toolbar = self._build_scoring_toolbar()
        right.addWidget(self._scoring_toolbar)
        right.addWidget(self._viewer, stretch=1)
        right.addWidget(self._events)
        right.addWidget(self._smart_nav)
        right.addWidget(self._nav)
        outer.addLayout(right, stretch=1)

        # Status bar with hover info
        self._status = self.statusBar()
        self._hover_label = QtWidgets.QLabel("")
        self._hover_label.setFont(QtGui.QFont("Consolas", 8))
        self._status.addWidget(self._hover_label, stretch=1)
        self._fps_label = QtWidgets.QLabel("FPS: --")
        self._status.addPermanentWidget(self._fps_label)

        self._sync_cursor.set_hover_label(self._hover_label)

        # Menu bar
        self._build_menus()

        self._install_scoring_shortcuts()

        # Connect signals
        self._connect_signals()

        # Apply theme
        self._apply_theme()

        # Drag-and-drop support for EDF files
        self.setAcceptDrops(True)

        # Set initial data
        self._channel_types = channel_types or {}
        if signals:
            self._set_initial_data(
                signals=signals,
                hypnogram=hypnogram,
                scoring_annotations=scoring_annotations or [],
                filter_pipelines=filter_pipelines or {},
                global_filter_enabled=global_filter_enabled,
                start_time_s=start_time_s,
                duration_s=duration_s,
                total_duration_s=total_duration_s,
            )

    # ------------------------------------------------------------------
    # Public API (called by ViewerBridge)
    # ------------------------------------------------------------------

    def update_signals(self, signals: Dict[str, Signal]) -> None:
        self._viewer.set_signals(signals)

    def set_time_window(self, start_s: float, duration_s: float) -> None:
        # #region agent log
        try:
            from CESA.agent_debug_f8b011 import log as _agent_log

            _agent_log(
                "E",
                "main_window.set_time_window:enter",
                "Qt set_time_window",
                {"start_s": float(start_s), "duration_s": float(duration_s)},
            )
        except Exception:
            pass
        # #endregion
        self._viewer.set_time_window(start_s, duration_s)
        self._hypno.set_window(start_s, duration_s)
        self._events.set_time_window(start_s, duration_s)
        self._nav.set_current_time(start_s)
        self._nav.set_duration(duration_s)

    def set_hypnogram(self, hypnogram: Optional[Tuple[List[str], float]]) -> None:
        if hypnogram is None:
            return
        labels, epoch_len = hypnogram
        self._hypnogram_labels = list(labels)
        self._epoch_len = epoch_len
        total_dur = self._viewer._total_duration_s or len(labels) * epoch_len
        self._hypno.set_hypnogram(labels, epoch_len, total_dur)
        self._viewer.set_epoch_length(epoch_len)
        self._nav.set_epoch_length(epoch_len)
        self._sync_cursor.set_epoch_length(epoch_len)
        self._ml_overlay.set_epoch_length(epoch_len)
        self._smart_nav.navigator.set_hypnogram(labels, epoch_len)

    def set_global_filter_enabled(self, enabled: bool) -> None:
        self._viewer.set_global_filter_enabled(enabled)
        self._nav._btn_filter.setChecked(enabled)
        self._refresh_filter_dashboard()

    def set_filter_pipelines(self, pipelines: Dict[str, Any]) -> None:
        self._viewer.set_filter_pipelines(pipelines)
        self._refresh_filter_dashboard()

    def _refresh_filter_dashboard(self) -> None:
        pipelines = getattr(self._viewer, "_pipelines", None) or {}
        g = bool(getattr(self._viewer, "_global_filter", True))
        n = count_effective_filter_channels(pipelines)
        self._dashboard.set_filter_info(n, g)

    def set_total_duration(self, total_s: float) -> None:
        self._viewer.set_total_duration(total_s)
        self._nav.set_total_duration(total_s)

    def set_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name
        self._theme = dict(THEMES.get(theme_name, DARK))
        self._apply_theme()
        self._viewer.set_theme(self._theme)
        self._hypno.set_theme(self._theme)
        self._events.set_theme(self._theme)
        self._sync_cursor.set_theme(self._theme)
        self._ml_overlay.set_theme(self._theme)

    def set_scoring_annotations(self, events: List[Dict[str, Any]]) -> None:
        self._events.set_events(events)
        self._smart_nav.navigator.set_events(events)

    # -- Annotation API --

    @property
    def annotation_store(self) -> AnnotationStore:
        return self._annotation_store

    def add_annotation(self, ann: Annotation) -> str:
        return self._annotation_store.add(ann)

    def remove_annotation(self, ann_id: str) -> None:
        self._annotation_store.remove(ann_id)

    # -- ML overlay API --

    def set_ml_predictions(
        self,
        backend_name: str,
        stages: List[str],
        confidences: Optional[List[float]] = None,
    ) -> None:
        self._ml_overlay.set_backend_predictions(backend_name, stages, confidences)

    def toggle_ml_backend(self, backend_name: str, visible: bool) -> None:
        self._ml_overlay.toggle_backend(backend_name, visible)

    # -- Inspection API --

    def set_epoch_features(self, features: List[Dict[str, float]]) -> None:
        self._inspection.set_features(features)

    def set_ml_probabilities(self, probs: List[Dict[str, float]]) -> None:
        self._inspection.set_ml_probabilities(probs)

    def set_epoch_decisions(self, decisions: List[str]) -> None:
        self._inspection.set_epoch_decisions(decisions)

    # -- Dashboard API --

    def update_dashboard(self) -> None:
        """Recompute and refresh dashboard metrics."""
        if self._signals_data:
            try:
                snr = self._dashboard.compute_snr(self._signals_data)
                self._dashboard.set_snr(snr)
            except Exception:
                pass
        if self._hypnogram_labels:
            dist = self._dashboard.compute_stage_distribution(self._hypnogram_labels)
            self._dashboard.set_stage_distribution(dist)
        self._dashboard.set_event_counts(
            annotations=len(self._annotation_store),
        )

    # -- Report export --

    def export_report_html(self, path: str) -> None:
        builder = ReportBuilder()
        builder.set_title("CESA - Rapport PSG")
        if self._hypnogram_labels:
            builder.set_hypnogram(self._hypnogram_labels, self._epoch_len)
            builder.add_hypnogram_figure()
            dist = self._dashboard.compute_stage_distribution(self._hypnogram_labels)
            builder.set_stage_distribution(dist)
        builder.set_annotations(self._annotation_store.to_event_list())
        builder.save_html(path)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _set_initial_data(
        self,
        signals: Dict[str, Signal],
        hypnogram: Optional[Tuple[List[str], float]],
        scoring_annotations: List[Dict[str, Any]],
        filter_pipelines: Dict[str, Any],
        global_filter_enabled: bool,
        start_time_s: float,
        duration_s: float,
        total_duration_s: Optional[float],
    ) -> None:
        self._signals_data = dict(signals)

        # Detect channel types
        for name in signals:
            if name not in self._channel_types:
                self._channel_types[name] = cesa_detect_signal_type(name)
        self._viewer._channel_types = dict(self._channel_types)

        # Total duration
        total = total_duration_s or 0.0
        for _n, (data, fs) in signals.items():
            total = max(total, len(data) / fs if fs > 0 else 0.0)
        self._viewer.set_total_duration(total)
        self._nav.set_total_duration(total)

        # Channel list sidebar
        self._channel_list.set_channels(list(signals.keys()), self._channel_types)

        # Signals
        self._viewer.set_signals(signals)

        # Feed signals to sync cursor for hover values
        self._sync_cursor.set_signals(signals)

        # Filters
        self._viewer.set_filter_pipelines(filter_pipelines)
        self._viewer.set_global_filter_enabled(global_filter_enabled)
        self._nav._btn_filter.setChecked(global_filter_enabled)
        self._refresh_filter_dashboard()

        # Hypnogram
        if hypnogram:
            self.set_hypnogram(hypnogram)

        # Events
        self._events.set_events(scoring_annotations)
        self._smart_nav.navigator.set_events(scoring_annotations)

        # Initial window
        self.set_time_window(start_time_s, duration_s)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Navigation bar -> viewer
        self._nav.time_changed.connect(self._on_nav_time)
        self._nav.time_slider_released.connect(self._flush_pending_tk_navigation)
        self._nav.duration_changed.connect(self._on_nav_duration)
        self._nav.filter_toggled.connect(self._on_filter_toggle)
        self._nav.normalize_toggled.connect(self._on_normalize_toggle)

        # Hypnogram click -> navigate
        self._hypno.epoch_clicked.connect(self._on_epoch_clicked)
        self._hypno.epoch_context_stage.connect(self._on_hypno_context_stage)

        # Events bar click -> navigate
        self._events.event_clicked.connect(self._on_event_clicked)

        # Viewer interactive scroll -> sync nav
        self._viewer.time_window_changed.connect(self._on_viewer_range)

        # Viewer cursor -> sync cursor across all panels
        self._viewer.cursor_time_changed.connect(self._on_cursor_moved)

        # Viewer annotation request (right-click)
        self._viewer.annotation_requested.connect(self._on_annotation_requested)

        # Smart navigation -> viewer
        self._smart_nav.navigate_to.connect(self._on_smart_navigate)

        # Channel list -> viewer
        self._channel_list.channel_visibility_changed.connect(self._on_visibility)
        self._channel_list.all_gain_changed.connect(self._on_global_gain)
        self._channel_list.order_changed.connect(self._on_order)
        self._channel_list.spacing_slider.valueChanged.connect(self._on_spacing)

        self._install_navigation_shortcuts()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        ctrl = self._app_controller
        if ctrl is not None:
            if ctrl.scoring_dirty:
                ret = QtWidgets.QMessageBox.question(
                    self,
                    "Modifications non enregistrees",
                    "Le scoring a ete modifie. Quitter sans sauvegarder ?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                )
                if ret == QtWidgets.QMessageBox.StandardButton.No:
                    event.ignore()
                    return
            ctrl.shutdown()
        logger.info("[VIEWER-CHK-29] EEGViewerMainWindow closeEvent (fenêtre Qt se ferme)")
        super().closeEvent(event)

    # -- Drag-and-drop recording files -------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    from core.raw_loader import SUPPORTED_RECORDING_EXTENSIONS
                    from pathlib import Path as _P
                    ext = _P(path).suffix.lower()
                    if ext in SUPPORTED_RECORDING_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self._open_recording_wizard(initial_path=path)
                return
        super().dropEvent(event)

    def _schedule_tk_navigation_callback(self, t: float) -> None:
        """Regroupe les appels Tk (évite saturation / crash pendant le drag du slider)."""
        if not self._on_navigate:
            return
        self._pending_tk_nav_t = float(t)
        self._tk_nav_timer.stop()
        self._tk_nav_timer.start()

    def _flush_pending_tk_navigation(self) -> None:
        self._tk_nav_timer.stop()
        if not self._on_navigate or self._pending_tk_nav_t is None:
            return
        logger.info(
            "[VIEWER-CHK-28] _flush_pending_tk_navigation invoke callback t=%.4f",
            float(self._pending_tk_nav_t),
        )
        try:
            self._on_navigate(self._pending_tk_nav_t)
        except Exception:
            logger.exception("[VIEWER-CHK-ERR-QT] _flush_pending_tk_navigation callback failed")

    def _install_navigation_shortcuts(self) -> None:
        """Raccourcis au niveau fenêtre : marchent même si le plot n'a pas le focus."""
        ctx = QtCore.Qt.ShortcutContext.WindowShortcut
        _K = QtCore.Qt.Key
        # Pas d'auto-répétition OS sur la nav : sinon rafale → update_plot/Tk et plantage.
        no_repeat_keys = (
            _K.Key_Left,
            _K.Key_Right,
            _K.Key_PageUp,
            _K.Key_PageDown,
            _K.Key_Home,
            _K.Key_End,
            _K.Key_Space,
            _K.Key_Plus,
            _K.Key_Minus,
            _K.Key_Equal,
            _K.Key_Q,
            _K.Key_D,
            _K.Key_Z,
            _K.Key_S,
            _K.Key_N,
            _K.Key_P,
            _K.Key_T,
        )
        for k in no_repeat_keys:
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.setContext(ctx)
            sc.setAutoRepeat(False)
            sc.activated.connect(
                functools.partial(self._dispatch_key_shortcuts_ints, _qt_key_int(k), 0),
            )
        mod_ctrl = _keyboard_modifiers_int(QtCore.Qt.KeyboardModifier.ControlModifier)
        for desc, key_enum, mod_int in (
            ("Ctrl+A", _K.Key_A, mod_ctrl),
            ("Ctrl+F", _K.Key_F, mod_ctrl),
        ):
            key_int = _qt_key_int(key_enum)
            sc = QtGui.QShortcut(QtGui.QKeySequence(desc), self)
            sc.setContext(ctx)
            sc.activated.connect(
                functools.partial(self._dispatch_key_shortcuts_ints, key_int, mod_int),
            )

    def _on_nav_time(self, t: float) -> None:
        epoch_idx = int(t / self._epoch_len) if self._epoch_len > 0 else 0
        insp_vis = self._inspection.isVisible()
        logger.info(
            "[VIEWER-CHK-27] _on_nav_time t=%.4f dur=%.4f epoch_idx=%s inspection_visible=%s",
            float(t),
            float(self._viewer.duration_s),
            epoch_idx,
            insp_vis,
        )
        self._viewer.set_time_window(t, self._viewer.duration_s)
        self._hypno.set_window(t, self._viewer.duration_s)
        self._events.set_time_window(t, self._viewer.duration_s)
        self._smart_nav.set_current_time(t)
        # Update inspection panel for current epoch
        if insp_vis:
            logger.info("[VIEWER-CHK-30] _on_nav_time inspection.set_epoch")
            self._inspection.set_epoch(epoch_idx, self._epoch_len)
        if self._on_navigate:
            self._schedule_tk_navigation_callback(t)

    def _on_nav_duration(self, dur: float) -> None:
        self._viewer.set_time_window(self._viewer.start_s, dur)
        self._hypno.set_window(self._viewer.start_s, dur)
        self._events.set_time_window(self._viewer.start_s, dur)

    def _on_epoch_clicked(self, t: float) -> None:
        dur = self._viewer.duration_s
        self._viewer.set_time_window(t, dur)
        self._hypno.set_window(t, dur)
        self._events.set_time_window(t, dur)
        self._nav.set_current_time(t)
        self._smart_nav.set_current_time(t)
        if self._on_navigate:
            self._schedule_tk_navigation_callback(t)

    def _on_event_clicked(self, event: dict) -> None:
        onset = float(event.get("onset", 0.0))
        dur = self._viewer.duration_s
        self._viewer.set_time_window(onset, dur)
        self._hypno.set_window(onset, dur)
        self._events.set_time_window(onset, dur)
        self._nav.set_current_time(onset)
        self._smart_nav.set_current_time(onset)
        if self._on_navigate:
            self._schedule_tk_navigation_callback(onset)

    def _on_viewer_range(self, start_s: float, dur_s: float) -> None:
        # Pas d'appel _on_navigate ici : time_window_changed part à chaque scroll/zoom (saturation CESA).
        self._nav.set_current_time(start_s)
        self._nav.set_duration(dur_s)
        self._hypno.set_window(start_s, dur_s)
        self._events.set_time_window(start_s, dur_s)
        self._smart_nav.set_current_time(start_s)

    def _on_cursor_moved(self, t: float) -> None:
        """Synchronize cursor across all panels."""
        self._sync_cursor.update_position(t)

    def _on_annotation_requested(self, start_s: float, end_s: float) -> None:
        """Open annotation dialog for the selected region."""
        dlg = AnnotationDialog(
            self, onset_s=start_s, duration_s=end_s - start_s,
        )
        result = dlg.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted and dlg.annotation:
            self._annotation_store.add(dlg.annotation)
            self._viewer._refresh_view()
            self._dashboard.set_event_counts(
                annotations=len(self._annotation_store),
            )
        elif result == 2 and dlg._ann_id:
            # Delete action
            self._annotation_store.remove(dlg._ann_id)
            self._viewer._refresh_view()

    def _on_smart_navigate(self, t: float) -> None:
        """Handle smart navigation jump."""
        dur = self._viewer.duration_s
        self._viewer.set_time_window(t, dur)
        self._hypno.set_window(t, dur)
        self._events.set_time_window(t, dur)
        self._nav.set_current_time(t)
        self._smart_nav.set_current_time(t)
        if self._on_navigate:
            self._schedule_tk_navigation_callback(t)

    def _on_filter_toggle(self) -> None:
        cur = self._nav._btn_filter.isChecked()
        self._viewer.set_global_filter_enabled(cur)
        self._refresh_filter_dashboard()
        if self._on_global_filter_toggled is not None:
            try:
                self._on_global_filter_toggled(cur)
            except Exception:
                logger.exception("[VIEWER-ERR] on_global_filter_toggled failed")

    def _on_normalize_toggle(self, checked: bool) -> None:
        if hasattr(self, '_normalize_action'):
            self._normalize_action.blockSignals(True)
            self._normalize_action.setChecked(checked)
            self._normalize_action.blockSignals(False)
        if checked:
            self._viewer.normalize_gains()
        else:
            self._viewer.reset_gains()

    def _on_normalize_menu(self, checked: bool) -> None:
        self._nav._btn_normalize.blockSignals(True)
        self._nav._btn_normalize.setChecked(checked)
        self._nav._btn_normalize.blockSignals(False)
        if checked:
            self._viewer.normalize_gains()
        else:
            self._viewer.reset_gains()

    def _on_visibility(self) -> None:
        self._viewer.set_visible_channels(self._channel_list.visible_channels())

    def _on_global_gain(self, gain: float) -> None:
        self._viewer.set_all_gains(gain)

    def _on_order(self, names: List[str]) -> None:
        self._viewer.set_channel_order(names)

    def _on_spacing(self, value: int) -> None:
        self._channel_list.spacing_label.setText(str(value))
        self._viewer.set_spacing(float(value))

    # ------------------------------------------------------------------
    # Scoring toolbar + studio callbacks (Tk)
    # ------------------------------------------------------------------

    def _build_scoring_toolbar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(4)
        title = QtWidgets.QLabel("Scoring époque courante")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        lay.addWidget(title)
        has_fn = self._on_request_stage_for_current_epoch is not None
        stages = [
            ("W", "Wake (touche 1)"),
            ("N1", "N1 (touche 2)"),
            ("N2", "N2 (touche 3)"),
            ("N3", "N3 (touche 4)"),
            ("R", "REM (touche 5)"),
            ("U", "Inconnu (touche 0)"),
        ]
        for st, tip in stages:
            btn = QtWidgets.QPushButton(st)
            btn.setToolTip(tip)
            btn.setEnabled(has_fn)
            btn.setFixedWidth(32)
            btn.clicked.connect(functools.partial(self._invoke_stage_current_epoch, st))
            lay.addWidget(btn)
        lay.addStretch(1)
        hint = QtWidgets.QLabel("Clic droit sur l’hypnogramme : autre époque")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(hint)
        return bar

    def _invoke_stage_current_epoch(self, stage: str) -> None:
        fn = self._on_request_stage_for_current_epoch
        if fn is None:
            return
        try:
            fn(stage)
        except Exception:
            logger.debug("on_request_stage_for_current_epoch failed", exc_info=True)

    def _invoke_auto_scoring_menu(self) -> None:
        self._menu_auto_scoring()

    def _invoke_manual_editor_menu(self) -> None:
        if self._on_open_manual_scoring_editor:
            try:
                self._on_open_manual_scoring_editor()
            except Exception:
                logger.debug("on_open_manual_scoring_editor failed", exc_info=True)

    def _invoke_filter_config_menu(self) -> None:
        ctrl = self._app_controller
        if ctrl is not None and ctrl.raw is not None:
            self._open_qt_filter_dialog()
            return
        if self._on_open_filter_config:
            try:
                self._on_open_filter_config()
            except Exception:
                logger.debug("on_open_filter_config failed", exc_info=True)

    def _open_qt_filter_dialog(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return

        from .dialogs.filter_config_dialog import FilterConfigDialog

        # Use the viewer's current signal names and types (which include
        # wizard renames and user-assigned types) rather than raw MNE names.
        signals = dict(self._signals_data) if self._signals_data else ctrl.build_viewer_signals()
        ch_names = list(signals.keys())
        ch_types = dict(self._channel_types) if self._channel_types else ctrl.channel_types_dict()

        def _on_apply(pipelines, global_enabled):
            ctrl.channel_filter_pipelines = {
                ch: pipe for ch, pipe in pipelines.items()
            }
            ctrl.filter_enabled = global_enabled
            self._viewer.set_filter_pipelines(pipelines)
            self._viewer.set_global_filter_enabled(global_enabled)
            self._nav._btn_filter.setChecked(global_enabled)
            self._refresh_filter_dashboard()
            ctrl.save_active_profile()
            logger.info("[FILTERS] Applied %d pipelines, global=%s",
                        len(pipelines), global_enabled)

        dlg = FilterConfigDialog(
            self,
            channel_names=ch_names,
            sfreq=ctrl.sfreq,
            channel_pipelines=ctrl.channel_filter_pipelines,
            channel_types=ch_types,
            signals=signals,
            global_enabled=ctrl.filter_enabled,
            on_apply=_on_apply,
        )
        dlg.exec()

    def _on_hypno_context_stage(self, onset_s: float, stage: str) -> None:
        fn = self._on_request_stage_at_epoch_time
        if fn is None:
            return
        try:
            fn(float(onset_s), str(stage))
        except Exception:
            logger.debug("on_request_stage_at_epoch_time failed", exc_info=True)

    def _install_scoring_shortcuts(self) -> None:
        if self._on_request_stage_for_current_epoch is None:
            return
        ctx = QtCore.Qt.ShortcutContext.WindowShortcut
        pairs = (
            (QtCore.Qt.Key.Key_1, "W"),
            (QtCore.Qt.Key.Key_2, "N1"),
            (QtCore.Qt.Key.Key_3, "N2"),
            (QtCore.Qt.Key.Key_4, "N3"),
            (QtCore.Qt.Key.Key_5, "R"),
            (QtCore.Qt.Key.Key_0, "U"),
        )
        for key_enum, st in pairs:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key_enum), self)
            sc.setContext(ctx)
            sc.setAutoRepeat(False)
            sc.activated.connect(functools.partial(self._invoke_stage_current_epoch, st))

    # ------------------------------------------------------------------
    # AppController integration
    # ------------------------------------------------------------------

    def set_app_controller(self, ctrl) -> None:
        """Wire the application controller and rebuild menus."""
        self._app_controller = ctrl
        self._on_request_stage_for_current_epoch = self._ctrl_stage_current_epoch
        self._on_request_stage_at_epoch_time = self._ctrl_stage_at_time
        self.menuBar().clear()
        self._build_menus()
        self._install_scoring_shortcuts()

    def _ctrl_stage_current_epoch(self, stage: str) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.sleep_scoring_data is None:
            return
        epoch_idx = int(self._viewer.start_s / self._epoch_len) if self._epoch_len > 0 else 0
        try:
            ctrl.sleep_scoring_data.loc[epoch_idx, 'stage'] = stage
            ctrl.scoring_dirty = True
            hyp = ctrl.build_hypnogram_tuple()
            if hyp:
                self.set_hypnogram(hyp)
            self.set_scoring_annotations(ctrl.build_scoring_annotations())
        except Exception:
            logger.debug("stage edit failed", exc_info=True)

    def _ctrl_stage_at_time(self, onset_s: float, stage: str) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.sleep_scoring_data is None:
            return
        epoch_idx = int(onset_s / self._epoch_len) if self._epoch_len > 0 else 0
        try:
            ctrl.sleep_scoring_data.loc[epoch_idx, 'stage'] = stage
            ctrl.scoring_dirty = True
            hyp = ctrl.build_hypnogram_tuple()
            if hyp:
                self.set_hypnogram(hyp)
            self.set_scoring_annotations(ctrl.build_scoring_annotations())
        except Exception:
            logger.debug("stage edit at time failed", exc_info=True)

    # ------------------------------------------------------------------
    # Menu actions wired to AppController
    # ------------------------------------------------------------------

    def _menu_open_recording(self) -> None:
        self._open_recording_wizard()

    def _open_recording_wizard(self, initial_path: Optional[str] = None) -> None:
        """Launch the EDF import wizard and feed the result into the viewer."""
        ctrl = self._app_controller
        if ctrl is None:
            return

        from .dialogs.edf_import_wizard import EDFImportWizard

        wizard = EDFImportWizard(
            theme_name=self._theme_name,
            controller=ctrl,
            parent=self,
        )
        if initial_path:
            wizard.set_initial_path(initial_path)

        result = wizard.run()
        if result is None:
            return

        from pathlib import Path as _Path

        session = result.session

        # Apply channel mapping from wizard selections
        if ctrl.raw is not None:
            mapping = session.channel_type_mapping
            ctrl.profile_channel_map_runtime = mapping
            selected = session.selected_channel_names
            ctrl.psg_channels_used = selected or list(ctrl.raw.ch_names)
            ctrl.selected_channels = ctrl.psg_channels_used[:8]
            logger.info(
                "[OPEN] wizard: %d channels selected", len(ctrl.psg_channels_used)
            )

        signals = ctrl.build_viewer_signals()
        logger.info("[OPEN] build_viewer_signals returned %d channels", len(signals))
        if not signals:
            logger.warning("[OPEN] No signals built -- using all derivations as fallback")
            signals = {ch: (arr, ctrl.sfreq) for ch, arr in ctrl.derivations.items()}

        # Apply channel renames from wizard (D1 -> F3-M2, etc.)
        renames = session.rename_mapping
        if renames:
            renamed: Dict[str, Signal] = {}
            for orig_name, sig in signals.items():
                display_name = renames.get(orig_name, orig_name)
                renamed[display_name] = sig
            signals = renamed
            logger.info("[OPEN] Renamed %d channels: %s", len(renames), renames)

        if signals:
            # Use wizard-assigned types as source of truth (not auto-detect)
            wizard_types = session.channel_type_mapping
            ch_types = ctrl.channel_types_dict()
            ch_types.update(wizard_types)
            if renames:
                renamed_types: Dict[str, str] = {}
                for orig, ctype in ch_types.items():
                    renamed_types[renames.get(orig, orig)] = ctype
                ch_types = renamed_types
            self._channel_types = ch_types

            # Restore per-channel filter pipelines saved in the import profile
            from CESA.filter_engine import FilterPipeline
            for ch in session.channels:
                if ch.filter_pipeline_dict and ch.selected:
                    try:
                        display = renames.get(ch.name, ch.name) if renames else ch.name
                        pipe = FilterPipeline.from_dict(ch.filter_pipeline_dict)
                        ctrl.channel_filter_pipelines[display] = pipe
                    except Exception:
                        logger.debug("Could not restore filter for %s", ch.name, exc_info=True)

            if session.global_filter_enabled is not None:
                ctrl.filter_enabled = session.global_filter_enabled

            self._set_initial_data(
                signals=signals,
                hypnogram=ctrl.build_hypnogram_tuple(),
                scoring_annotations=ctrl.build_scoring_annotations(),
                filter_pipelines=ctrl.build_viewer_filter_pipelines(),
                global_filter_enabled=ctrl.filter_enabled,
                start_time_s=0.0,
                duration_s=ctrl.duration,
                total_duration_s=ctrl.total_duration_s,
            )
            logger.info(
                "[OPEN] _set_initial_data called with %d signals, total_dur=%.1f",
                len(signals),
                ctrl.total_duration_s,
            )

        file_name = _Path(session.file_path).name
        self.setWindowTitle(f"CESA v0.0beta1.0 - {file_name}")
        self._status.showMessage(f"Fichier charge : {file_name}", 5000)

    def _menu_auto_scoring(self) -> None:
        if self._on_request_auto_scoring:
            try:
                self._on_request_auto_scoring()
            except Exception:
                logger.debug("on_request_auto_scoring failed", exc_info=True)
            return

        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return

        methods = list(ctrl.SUPPORTED_SCORING_METHODS)
        labels = {
            "pftsleep": "PFTSleep (deep learning)",
            "yasa": "YASA (feature-based)",
            "usleep": "U-Sleep (deep learning)",
            "aasm_rules": "Regles AASM (rule-based)",
            "ml": "ML (gradient boosting)",
            "ml_hmm": "ML + HMM (sequence model)",
            "rules_hmm": "Regles AASM + HMM",
        }
        display = [labels.get(m, m) for m in methods]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Scoring automatique", "Methode :", display, 0, False,
        )
        if ok:
            idx = display.index(choice)
            method = methods[idx]
        else:
            method = None
        if not ok or method is None:
            return

        progress = QtWidgets.QProgressDialog("Scoring en cours...", None, 0, 0, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        QtWidgets.QApplication.processEvents()

        success = ctrl.run_auto_scoring(
            method=method,
            progress_callback=lambda msg: (
                progress.setLabelText(msg),
                QtWidgets.QApplication.processEvents(),
            ),
        )
        progress.close()

        if success:
            hyp = ctrl.build_hypnogram_tuple()
            if hyp:
                self.set_hypnogram(hyp)
            self.set_scoring_annotations(ctrl.build_scoring_annotations())
            self._status.showMessage(f"Scoring {method} termine.", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Scoring", "Le scoring automatique a echoue.")

    def _menu_import_scoring_excel(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importer scoring Excel", "", "Excel (*.xlsx *.xls);;Tous (*.*)",
        )
        if not path:
            return
        if ctrl.import_scoring_excel(path):
            self._status.showMessage("Scoring Excel importe.", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Import du scoring echoue.")

    def _menu_import_scoring_edf(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importer scoring EDF+", "", "EDF (*.edf *.edf+);;Tous (*.*)",
        )
        if not path:
            return
        if ctrl.import_scoring_edf(path):
            self._status.showMessage("Scoring EDF+ importe.", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Import du scoring echoue.")

    def _menu_save_scoring_csv(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.sleep_scoring_data is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Aucun scoring disponible.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Sauvegarder scoring", "", "CSV (*.csv);;Tous (*.*)",
        )
        if not path:
            return
        if ctrl.save_scoring_csv(path):
            self._status.showMessage(f"Scoring exporte: {path}", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Export du scoring echoue.")

    def _menu_compare_scoring(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        if ctrl.sleep_scoring_data is None or ctrl.manual_scoring_data is None:
            QtWidgets.QMessageBox.warning(
                self, "Attention",
                "Il faut un scoring automatique ET un scoring manuel pour comparer.\n"
                "Lancez le scoring auto et importez un scoring manuel d'abord.")
            return
        merged = ctrl.compare_scoring()
        if merged is None or merged.empty:
            QtWidgets.QMessageBox.information(
                self, "Comparaison", "Aucune epoque commune trouvee.")
            return
        n_total = len(merged)
        n_match = int(merged['match'].sum())
        acc = n_match / n_total * 100 if n_total > 0 else 0.0

        stage_counts = merged.groupby('auto')['match'].agg(['sum', 'count'])
        details = []
        for stage, row in stage_counts.iterrows():
            s_acc = row['sum'] / row['count'] * 100 if row['count'] > 0 else 0
            details.append(f"  {stage}: {int(row['sum'])}/{int(row['count'])} ({s_acc:.1f}%)")

        msg = (
            f"Epoques comparees: {n_total}\n"
            f"Concordance globale: {n_match}/{n_total} ({acc:.1f}%)\n\n"
            f"Par stade (auto):\n" + "\n".join(details)
        )
        QtWidgets.QMessageBox.information(self, "Comparaison auto vs manuel", msg)

    def _menu_scoring_info(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        auto = ctrl.sleep_scoring_data
        manual = ctrl.manual_scoring_data

        lines = ["=== Informations Scoring ===\n"]
        if auto is not None and not auto.empty:
            lines.append(f"Scoring automatique: {len(auto)} epoques")
            lines.append(f"  Methode: {ctrl.sleep_scoring_method}")
            lines.append(f"  Duree epoque: {ctrl.scoring_epoch_duration:.0f}s")
            if 'stage' in auto.columns:
                counts = auto['stage'].value_counts()
                for stage, n in counts.items():
                    lines.append(f"  {stage}: {n} epoques")
        else:
            lines.append("Scoring automatique: non disponible")

        lines.append("")
        if manual is not None and not manual.empty:
            lines.append(f"Scoring manuel: {len(manual)} epoques")
            if 'stage' in manual.columns:
                counts = manual['stage'].value_counts()
                for stage, n in counts.items():
                    lines.append(f"  {stage}: {n} epoques")
        else:
            lines.append("Scoring manuel: non disponible")

        QtWidgets.QMessageBox.information(
            self, "Informations Scoring", "\n".join(lines))

    def _menu_detect_events(self, event_type: str) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return

        labels = {
            "arousals": "Arousals (micro-eveils)",
            "apneas": "Apnees / Hypopnees",
            "desaturations": "Desaturations SpO2",
        }
        progress = QtWidgets.QProgressDialog(
            f"Detection: {labels.get(event_type, event_type)}...",
            None, 0, 0, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        QtWidgets.QApplication.processEvents()

        events = ctrl.detect_events(
            event_type,
            progress_callback=lambda msg: (
                progress.setLabelText(msg),
                QtWidgets.QApplication.processEvents(),
            ),
        )
        progress.close()

        if not events:
            QtWidgets.QMessageBox.information(
                self, "Detection",
                f"Aucun evenement '{labels.get(event_type, event_type)}' detecte.\n"
                "Verifiez que les canaux necessaires sont presents dans l'enregistrement.")
            return

        lines = [f"{labels.get(event_type, event_type)}: {len(events)} evenements detectes\n"]
        for i, ev in enumerate(events[:20]):
            onset = ev.get('onset', ev.get('start', 0))
            dur = ev.get('duration', 0)
            desc = ev.get('type', ev.get('description', ''))
            lines.append(f"  {i+1}. t={onset:.1f}s  dur={dur:.1f}s  {desc}")
        if len(events) > 20:
            lines.append(f"  ... et {len(events) - 20} autres")

        QtWidgets.QMessageBox.information(
            self, f"Detection - {labels.get(event_type, event_type)}",
            "\n".join(lines))

        # Add detected events to the events bar
        new_events = []
        for ev in events:
            onset = ev.get('onset', ev.get('start', 0))
            dur = ev.get('duration', 0)
            desc = ev.get('type', ev.get('description', event_type))
            new_events.append({
                "onset": float(onset),
                "duration": float(dur),
                "description": str(desc),
            })
        existing = getattr(self._events, '_events', [])
        self._events.set_events(list(existing) + new_events)
        self._status.showMessage(
            f"{len(events)} {labels.get(event_type, event_type)} detecte(s)", 5000)

    def _menu_spectral_analysis(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return
        signals = ctrl.build_viewer_signals()
        channels = ctrl.psg_channels_used or list(signals.keys())
        from .dialogs.spectral_dialog import SpectralAnalysisDialog
        dlg = SpectralAnalysisDialog(
            self,
            signals=signals,
            channels=channels[:8],
            current_time=ctrl.current_time,
            duration=ctrl.duration,
        )
        dlg.exec()

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "A propos de CESA",
            "<h3>CESA (Complex EEG Studio Analysis)</h3>"
            "<p>Version 0.0beta1.0</p>"
            "<p>Auteur: Come Barmoy<br>"
            "Unite Neuropsychologie du Stress - IRBA</p>"
            "<p>Licence: MIT</p>",
        )

    def _open_docs(self) -> None:
        import webbrowser
        webbrowser.open("https://github.com/cbarmoy")

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        mb = self.menuBar()
        has_ctrl = self._app_controller is not None

        # ---- Fichier ----
        file_m = mb.addMenu("&Fichier")
        act_open = file_m.addAction("Ouvrir un enregistrement...")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._menu_open_recording)
        act_open.setEnabled(has_ctrl)
        file_m.addSeparator()

        exp_sub = file_m.addMenu("Exporter")
        exp_html = exp_sub.addAction("Rapport HTML...")
        exp_html.triggered.connect(self._export_html_report)
        exp_pdf = exp_sub.addAction("Rapport PDF...")
        exp_pdf.triggered.connect(self._export_pdf_report)
        exp_ann = exp_sub.addAction("Annotations JSON...")
        exp_ann.triggered.connect(self._save_annotations)

        file_m.addSeparator()
        act_quit = file_m.addAction("Quitter")
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

        # ---- Affichage ----
        view = mb.addMenu("&Affichage")
        theme_dark = view.addAction("Theme sombre")
        theme_dark.triggered.connect(lambda: self.set_theme("dark"))
        theme_light = view.addAction("Theme clair")
        theme_light.triggered.connect(lambda: self.set_theme("light"))
        view.addSeparator()
        raw_toggle = view.addAction("Afficher signal brut")
        raw_toggle.setCheckable(True)
        raw_toggle.toggled.connect(self._viewer.set_show_raw_underlay)
        view.addSeparator()
        self._normalize_action = view.addAction("Normaliser les amplitudes (Ctrl+N)")
        self._normalize_action.setCheckable(True)
        self._normalize_action.toggled.connect(self._on_normalize_menu)

        # ---- Panneaux ----
        panels = mb.addMenu("&Panneaux")
        insp_action = panels.addAction("Inspection (F8)")
        insp_action.setCheckable(True)
        insp_action.setShortcut("F8")
        insp_action.toggled.connect(self._inspection.setVisible)
        dash_action = panels.addAction("Dashboard (F9)")
        dash_action.setCheckable(True)
        dash_action.setShortcut("F9")
        dash_action.toggled.connect(lambda v: (
            self._dashboard.setVisible(v),
            self.update_dashboard() if v else None,
        ))

        # ---- Filtres ----
        filt_m = mb.addMenu("Filt&res")
        act_fc = filt_m.addAction("Configuration des filtres...")
        act_fc.setShortcut("Ctrl+Shift+F")
        act_fc.triggered.connect(self._invoke_filter_config_menu)
        act_fc.setEnabled(self._on_open_filter_config is not None or has_ctrl)

        # ---- Sommeil ----
        sleep_m = mb.addMenu("&Sommeil")
        act_auto = sleep_m.addAction("Scoring automatique...")
        act_auto.triggered.connect(self._menu_auto_scoring)
        act_auto.setEnabled(has_ctrl or self._on_request_auto_scoring is not None)

        import_sub = sleep_m.addMenu("Importer scoring")
        act_imp_excel = import_sub.addAction("Depuis fichier Excel...")
        act_imp_excel.triggered.connect(self._menu_import_scoring_excel)
        act_imp_excel.setEnabled(has_ctrl)
        act_imp_edf = import_sub.addAction("Depuis annotations EDF+...")
        act_imp_edf.triggered.connect(self._menu_import_scoring_edf)
        act_imp_edf.setEnabled(has_ctrl)

        sleep_m.addSeparator()
        act_save_csv = sleep_m.addAction("Sauvegarder scoring (CSV)...")
        act_save_csv.triggered.connect(self._menu_save_scoring_csv)
        act_save_csv.setEnabled(has_ctrl)

        act_compare = sleep_m.addAction("Comparer auto vs manuel...")
        act_compare.triggered.connect(self._menu_compare_scoring)
        act_compare.setEnabled(has_ctrl)

        sleep_m.addSeparator()
        act_info = sleep_m.addAction("Informations scoring")
        act_info.triggered.connect(self._menu_scoring_info)
        act_info.setEnabled(has_ctrl)

        sleep_m.addSeparator()
        tip = sleep_m.addAction("Clic droit sur l'hypnogramme : scorer une epoque")
        tip.setEnabled(False)

        # ---- Detection d'evenements ----
        events_m = mb.addMenu("&Evenements")
        act_arousals = events_m.addAction("Detecter les arousals...")
        act_arousals.triggered.connect(lambda: self._menu_detect_events("arousals"))
        act_arousals.setEnabled(has_ctrl)
        act_apneas = events_m.addAction("Detecter apnees / hypopnees...")
        act_apneas.triggered.connect(lambda: self._menu_detect_events("apneas"))
        act_apneas.setEnabled(has_ctrl)
        act_desats = events_m.addAction("Detecter desaturations SpO2...")
        act_desats.triggered.connect(lambda: self._menu_detect_events("desaturations"))
        act_desats.setEnabled(has_ctrl)

        # ---- Annotations ----
        ann = mb.addMenu("&Annotations")
        add_ann = ann.addAction("Nouvelle annotation")
        add_ann.setShortcut("Ctrl+A")
        add_ann.triggered.connect(self._create_annotation_at_cursor)
        ann.addSeparator()
        save_ann = ann.addAction("Sauvegarder annotations...")
        save_ann.triggered.connect(self._save_annotations)
        load_ann = ann.addAction("Charger annotations...")
        load_ann.triggered.connect(self._load_annotations)
        ann.addSeparator()
        clear_ann = ann.addAction("Effacer toutes les annotations")
        clear_ann.triggered.connect(self._clear_annotations)

        # ---- Analyse ----
        analyse = mb.addMenu("Anal&yse")
        act_spectral = analyse.addAction("Analyse spectrale (PSD Welch)...")
        act_spectral.triggered.connect(self._menu_spectral_analysis)
        act_spectral.setEnabled(has_ctrl)
        for label in (
            "PSD par stade de sommeil...",
            "Spectrogramme wavelet...",
            "Entropie renormalisee...",
            "Analyse temporelle...",
        ):
            act = analyse.addAction(label)
            act.setEnabled(False)
            act.setToolTip("Migration en cours (Phase 6)")

        # ---- ML/DL ----
        ml = mb.addMenu("&ML/DL")
        ml_info = ml.addAction("Aucun backend charge")
        ml_info.setEnabled(False)
        self._ml_menu = ml

        # ---- Aide ----
        aide = mb.addMenu("A&ide")
        about_act = aide.addAction("A propos de CESA...")
        about_act.triggered.connect(self._show_about)
        aide.addSeparator()
        docs_act = aide.addAction("Documentation en ligne...")
        docs_act.triggered.connect(self._open_docs)

    # ------------------------------------------------------------------
    # Annotation actions
    # ------------------------------------------------------------------

    def _create_annotation_at_cursor(self) -> None:
        t = self._sync_cursor.time_s or self._viewer.start_s
        dlg = AnnotationDialog(self, onset_s=t, duration_s=self._epoch_len)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.annotation:
            self._annotation_store.add(dlg.annotation)
            self._viewer._refresh_view()

    def _save_annotations(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Sauvegarder annotations", "", "JSON (*.json)",
        )
        if path:
            self._annotation_store.save_json(path)

    def _load_annotations(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Charger annotations", "", "JSON (*.json)",
        )
        if path:
            count = self._annotation_store.load_json(path)
            self._viewer._refresh_view()
            self._status.showMessage(f"{count} annotations chargees", 3000)

    def _clear_annotations(self) -> None:
        self._annotation_overlay.clear()
        self._annotation_store.clear()
        self._viewer._refresh_view()

    # ------------------------------------------------------------------
    # Export actions
    # ------------------------------------------------------------------

    def _export_html_report(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export rapport HTML", "", "HTML (*.html)",
        )
        if path:
            self.export_report_html(path)
            self._status.showMessage(f"Rapport exporte: {path}", 3000)

    def _export_pdf_report(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export rapport PDF", "", "PDF (*.pdf)",
        )
        if path:
            builder = ReportBuilder()
            builder.set_title("CESA - Rapport PSG")
            if self._hypnogram_labels:
                builder.set_hypnogram(self._hypnogram_labels, self._epoch_len)
                builder.add_hypnogram_figure()
            builder.set_annotations(self._annotation_store.to_event_list())
            builder.save_pdf(path)
            self._status.showMessage(f"Rapport exporte: {path}", 3000)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        t = self._theme
        self.setStyleSheet(f"""
            QMainWindow {{ background: {t['background']}; }}
            QWidget {{ background: {t['background']}; color: {t['foreground']}; }}
            QMenuBar {{ background: {t['surface']}; color: {t['foreground']}; }}
            QMenuBar::item:selected {{ background: {t['accent']}; }}
            QMenu {{ background: {t['surface']}; color: {t['foreground']}; }}
            QMenu::item:selected {{ background: {t['accent']}; }}
            QStatusBar {{ background: {t['surface']}; color: {t['foreground']}; }}
            QListWidget {{ background: {t['surface_alt']}; color: {t['foreground']};
                           border: 1px solid {t['border']}; }}
            QListWidget::item:selected {{ background: {t['accent']}; color: white; }}
            QGroupBox {{ border: 1px solid {t['border']}; border-radius: 4px;
                         margin-top: 8px; padding-top: 12px;
                         color: {t['foreground']}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px;
                                padding: 0 4px; color: {t['accent']}; }}
            QSlider::groove:horizontal {{ background: {t['surface_alt']};
                                          height: 6px; border-radius: 3px; }}
            QSlider::handle:horizontal {{ background: {t['accent']}; width: 14px;
                                          margin: -4px 0; border-radius: 7px; }}
            QPushButton {{ background: {t['surface']}; color: {t['foreground']};
                           border: 1px solid {t['border']}; border-radius: 3px;
                           padding: 3px 8px; }}
            QPushButton:hover {{ background: {t['accent']}; color: white; }}
            QPushButton:checked {{ background: {t['accent']}; color: white; }}
            QSpinBox {{ background: {t['surface_alt']}; color: {t['foreground']};
                        border: 1px solid {t['border']}; }}
            QLabel {{ background: transparent; }}
        """)

    # ------------------------------------------------------------------
    # Keyboard (QShortcut WindowShortcut + dispatch pour la barre de nav)
    # ------------------------------------------------------------------

    def _dispatch_key_shortcuts_ints(self, key: int, mod_int: int) -> bool:
        """Return True if handled (plot deferred path + keyPressEvent)."""
        mod = QtCore.Qt.KeyboardModifiers(mod_int)

        if key == QtCore.Qt.Key.Key_A and mod & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._create_annotation_at_cursor()
            return True

        if key == QtCore.Qt.Key.Key_N and mod & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._nav._btn_normalize.toggle()
            return True

        if key == QtCore.Qt.Key.Key_N and not _mods_block_combo_nav(mod):
            self._smart_nav.navigator.next_event()
            return True
        if key == QtCore.Qt.Key.Key_P and not _mods_block_combo_nav(mod):
            self._smart_nav.navigator.prev_event()
            return True

        if key == QtCore.Qt.Key.Key_T and not _mods_block_combo_nav(mod):
            self._smart_nav.navigator.next_stage_change()
            return True

        return bool(self._nav.handle_key_ints(key, mod))

