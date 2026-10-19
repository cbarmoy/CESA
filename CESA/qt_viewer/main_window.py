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
from pathlib import Path
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
from .layout_panel import LayoutPanel
from .ml_overlay import MLOverlayManager
from .navigation_bar import NavigationBar
from .scaling_panel import ScalingPanel
from .report_export import ReportBuilder
from .smart_navigation import SmartNavWidget
from .spectrogram_bar import NightSpectrogramWorker, SpectrogramBar
from .spectrogram_compute import SpectrogramResult, compute_tfr
from .spectrogram_panel import SpectrogramPanel
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
    menu bar (Fichier / Affichage / Filtres / Sommeil / Analyse / Aide).
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
        on_set_manual_auto_advance: Optional[Callable[[bool], None]] = None,
        on_get_manual_auto_advance: Optional[Callable[[], bool]] = None,
        on_open_stage_psd_fft: Optional[Callable[[], None]] = None,
        on_open_stage_psd_before_after: Optional[Callable[[], None]] = None,
        on_open_wavelet_spectrogram: Optional[Callable[[], None]] = None,
        on_open_batch_fft_automation: Optional[Callable[[], None]] = None,
        on_open_group_analysis: Optional[Callable[[], None]] = None,
        on_toggle_fft_scoring_view: Optional[Callable[[], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CESA (Complex EEG Studio Analysis) v0.0beta2.0")
        self.resize(1500, 900)
        self.setMinimumSize(900, 500)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

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
        self._on_set_manual_auto_advance = on_set_manual_auto_advance
        self._on_get_manual_auto_advance = on_get_manual_auto_advance
        self._on_open_stage_psd_fft = on_open_stage_psd_fft
        self._on_open_stage_psd_before_after = on_open_stage_psd_before_after
        self._on_open_wavelet_spectrogram = on_open_wavelet_spectrogram
        self._on_open_batch_fft_automation = on_open_batch_fft_automation
        self._on_open_group_analysis = on_open_group_analysis
        self._on_toggle_fft_scoring_view = on_toggle_fft_scoring_view
        self._fft_scoring_dialog: Optional[QtWidgets.QDialog] = None
        self._registered_nav_shortcuts: List[QtGui.QShortcut] = []
        self._registered_scoring_shortcuts: List[QtGui.QShortcut] = []
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

        # -- Wavelet spectrogram bars (night: full recording / window: visible EEG) --
        self._spectro_night = SpectrogramBar()
        self._spectro_night.set_follow_window(False)
        self._spectro_night.setVisible(False)
        self._spectro_window = SpectrogramBar()
        self._spectro_window.set_follow_window(True)
        self._spectro_window.setVisible(False)
        self._spectro_night_worker: Optional[NightSpectrogramWorker] = None
        self._spectro_window_worker: Optional[NightSpectrogramWorker] = None
        self._spectro_night_cache_key: Optional[Tuple[Any, ...]] = None
        self._spectro_night_result: Optional[SpectrogramResult] = None
        self._spectro_window_pending: Tuple[float, float] = (0.0, 30.0)
        self._spectro_window_timer = QtCore.QTimer(self)
        self._spectro_window_timer.setSingleShot(True)
        self._spectro_window_timer.setInterval(150)
        self._spectro_window_timer.timeout.connect(self._recompute_window_spectrogram)

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

        # -- Scaling panel (dock) --
        self._scaling_panel = ScalingPanel(self)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._scaling_panel,
        )
        self._scaling_panel.setVisible(False)

        # -- Layout panel (dock) --
        self._layout_panel = LayoutPanel(self)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._layout_panel,
        )
        self._layout_panel.setVisible(False)

        # -- Spectrogram panel (dock) --
        self._spectro_panel = SpectrogramPanel(self)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self._spectro_panel,
        )
        self._spectro_panel.setVisible(False)

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
        right.addWidget(self._spectro_night)
        self._scoring_toolbar = self._build_scoring_toolbar()
        right.addWidget(self._scoring_toolbar)
        right.addWidget(self._viewer, stretch=1)
        right.addWidget(self._spectro_window)
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
        self._signals_data = dict(signals)
        self._viewer.set_signals(signals)
        self._update_fft_scoring_menu_enabled()

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
        self._refresh_stage_indicator(start_s)
        self._spectro_night.set_window(start_s, duration_s)
        self._schedule_window_spectrogram_update(start_s, duration_s)

    def _refresh_stage_indicator(self, t: Optional[float] = None) -> None:
        """Update sleep-stage readouts in the nav bar and scoring toolbar."""
        from .themes import normalize_stage, stage_badge_stylesheet, stage_display_name

        if t is None:
            t = float(self._viewer.start_s)
        code = ""
        disp = "—"
        ep = 0
        if self._hypnogram_labels and self._epoch_len > 0:
            ep = int(t / self._epoch_len)
            ep = max(0, min(ep, len(self._hypnogram_labels) - 1))
            code = normalize_stage(self._hypnogram_labels[ep])
            disp = stage_display_name(code)

        readout = getattr(self, "_current_stage_readout", None)
        if readout is not None:
            if self._hypnogram_labels:
                readout.setText(f"Epoque {ep}: {disp} ({code})")
                readout.setStyleSheet(stage_badge_stylesheet(self._theme, code))
            else:
                readout.setText("Epoque: —")
                readout.setStyleSheet(
                    f"color: {self._theme.get('text_dim', '#888')};"
                )

        for btn in getattr(self, "_stage_buttons", []):
            active = bool(code) and btn.text() == code
            btn.setStyleSheet(
                stage_badge_stylesheet(self._theme, code)
                if active else ""
            )

    def set_hypnogram(self, hypnogram: Optional[Tuple[List[str], float]]) -> None:
        if hypnogram is None:
            self._hypnogram_labels = []
            self._nav.set_hypnogram_labels([])
            self._sync_cursor.set_hypnogram_labels([])
            self._events.set_hypnogram([], self._epoch_len)
            self._refresh_stage_indicator()
            self._refresh_spectro_overlay()
            return
        labels, epoch_len = hypnogram
        self._hypnogram_labels = list(labels)
        self._epoch_len = epoch_len
        # Shared X span for hypnogram + night spectrogram (pixel-aligned).
        total_dur = max(
            float(self._viewer._total_duration_s or 0.0),
            len(labels) * float(epoch_len),
        )
        self._hypno.set_hypnogram(labels, epoch_len, total_dur)
        self._events.set_hypnogram(labels, epoch_len)
        self._viewer.set_epoch_length(epoch_len)
        self._nav.set_epoch_length(epoch_len)
        self._nav.set_hypnogram_labels(labels)
        self._sync_cursor.set_epoch_length(epoch_len)
        self._sync_cursor.set_hypnogram_labels(labels)
        self._ml_overlay.set_epoch_length(epoch_len)
        self._smart_nav.navigator.set_hypnogram(labels, epoch_len)
        self._refresh_stage_indicator()
        self._spectro_night.set_total_duration(total_dur)
        # Keep hypno and night spectro X-axes locked together.
        try:
            self._spectro_night._plot.setXLink(self._hypno._plot)
        except Exception:
            pass
        self._refresh_spectro_overlay()
        self._maybe_recompute_night_spectrogram()

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
        self._spectro_night.set_total_duration(total_s)

    def set_theme(self, theme_name: str) -> None:
        self._theme_name = theme_name
        self._theme = dict(THEMES.get(theme_name, DARK))
        self._apply_theme()
        self._viewer.set_theme(self._theme)
        self._hypno.set_theme(self._theme)
        self._events.set_theme(self._theme)
        self._sync_cursor.set_theme(self._theme)
        self._ml_overlay.set_theme(self._theme)
        self._spectro_night.set_theme(self._theme)
        self._spectro_window.set_theme(self._theme)

    def set_scoring_annotations(self, events: List[Dict[str, Any]]) -> None:
        """Refresh clinical annotation overlays (does not replace the stage band).

        Sleep stages are driven by :meth:`set_hypnogram`.  The *events*
        argument is kept for API compatibility with older call sites.
        """
        del events  # hypnogram owns the stage band; annotations come from the store
        self._refresh_annotation_ui()

    def _refresh_annotation_ui(self) -> None:
        """Sync clinical annotations across hypnogram, events bar and smart nav."""
        events = self._annotation_store.to_event_list()
        self._hypno.set_annotations(events)
        # Overlay only — does not clear the hypnogram stage band
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
        ctrl = self._app_controller
        if ctrl is not None and ctrl.sleep_scoring_data is not None and not ctrl.sleep_scoring_data.empty:
            try:
                from CESA.sleep_metrics import compute_psg_summary
                df = ctrl.sleep_scoring_data
                if df is not None and not df.empty:
                    summary = compute_psg_summary(
                        df,
                        ctrl.scoring_epoch_duration,
                        night_start_s=ctrl.night_start_s,
                        night_end_s=ctrl.night_end_s,
                    )
                    cm = summary.clinical
                    self._dashboard.set_sleep_metrics(
                        tst_min=cm.total_sleep_time_min,
                        se_pct=cm.sleep_efficiency_pct,
                        sol_min=cm.sleep_onset_latency_min,
                        waso_min=cm.waso_min,
                        rem_lat_min=cm.rem_latency_min,
                    )
                    dist = {
                        "W": summary.stage_pct_of_tst.get("W", 0.0),
                        "N1": summary.stage_pct_of_tst.get("N1", 0.0),
                        "N2": summary.stage_pct_of_tst.get("N2", 0.0),
                        "N3": summary.stage_pct_of_tst.get("N3", 0.0),
                        "REM": summary.stage_pct_of_tst.get("R", 0.0),
                    }
                    self._dashboard.set_stage_distribution(dist)
            except Exception:
                logger.debug("PSG dashboard update failed", exc_info=True)
        elif self._hypnogram_labels:
            dist = self._dashboard.compute_stage_distribution(self._hypnogram_labels)
            self._dashboard.set_stage_distribution(dist)
        self._dashboard.set_event_counts(
            annotations=len(self._annotation_store),
        )
        # Night QC heuristics
        if ctrl is not None and ctrl.derivations:
            try:
                from CESA.prep_qc import compute_night_qc
                report = compute_night_qc(
                    ctrl.derivations,
                    ctrl.sfreq,
                    ctrl.channel_types_dict(),
                )
                n_flat = sum(1 for c in report.per_channel if c.is_flat)
                self._dashboard.set_night_qc(
                    n_channels=report.n_channels,
                    n_bad=len(report.bad_channels),
                    n_flat=n_flat,
                    warnings=report.warnings,
                )
            except Exception:
                logger.debug("Night QC dashboard update failed", exc_info=True)
        self._update_night_window_status()

    # -- Report export --

    def export_report_html(self, path: str) -> None:
        builder = ReportBuilder()
        builder.set_title("CESA - Rapport PSG")
        self._populate_report_builder(builder)
        builder.save_html(path)

    def _populate_report_builder(self, builder: ReportBuilder) -> None:
        ctrl = self._app_controller
        if self._hypnogram_labels:
            builder.set_hypnogram(self._hypnogram_labels, self._epoch_len)
            builder.add_hypnogram_figure()
        if ctrl is not None and ctrl.sleep_scoring_data is not None and not ctrl.sleep_scoring_data.empty:
            try:
                from CESA.sleep_metrics import compute_psg_summary
                df = ctrl.sleep_scoring_data
                if df is not None and not df.empty:
                    summary = compute_psg_summary(
                        df,
                        ctrl.scoring_epoch_duration,
                        night_start_s=ctrl.night_start_s,
                        night_end_s=ctrl.night_end_s,
                    )
                    cm = summary.clinical
                    builder.set_sleep_metrics({
                        "TIB (min)": cm.time_in_bed_min,
                        "TST (min)": cm.total_sleep_time_min,
                        "SOL (min)": cm.sleep_onset_latency_min,
                        "WASO (min)": cm.waso_min,
                        "SE (%)": cm.sleep_efficiency_pct,
                        "REM lat. (min)": cm.rem_latency_min,
                        "N1 (% TST)": cm.n1_pct,
                        "N2 (% TST)": cm.n2_pct,
                        "N3 (% TST)": cm.n3_pct,
                        "REM (% TST)": cm.rem_pct,
                        "Reveils": float(summary.awakenings),
                    })
                    builder.set_stage_distribution({
                        "W": summary.stage_pct_of_tst.get("W", 0.0),
                        "N1": summary.stage_pct_of_tst.get("N1", 0.0),
                        "N2": summary.stage_pct_of_tst.get("N2", 0.0),
                        "N3": summary.stage_pct_of_tst.get("N3", 0.0),
                        "REM": summary.stage_pct_of_tst.get("R", 0.0),
                    })
                    if ctrl.has_night_window:
                        builder.set_recording_info({
                            "Fenetre analyse debut (min)": f"{ctrl.night_start_s / 60.0:.1f}",
                            "Fenetre analyse fin (min)": f"{ctrl.night_end_s / 60.0:.1f}",
                        })
                    builder.set_episodes([
                        {
                            "start_min": ep.start_min,
                            "end_min": ep.end_min,
                            "stage": ep.stage,
                            "duration_min": ep.duration_min,
                        }
                        for ep in summary.episodes
                    ])
            except Exception:
                logger.debug("Report PSG metrics failed", exc_info=True)
        elif self._hypnogram_labels:
            dist = self._dashboard.compute_stage_distribution(self._hypnogram_labels)
            builder.set_stage_distribution(dist)
        builder.set_annotations(self._annotation_store.to_event_list())

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

        # Hypnogram (also feeds the scoring band under the EEG)
        if hypnogram:
            self.set_hypnogram(hypnogram)

        # Clinical annotations only (stages come from hypnogram)
        self._refresh_annotation_ui()

        # Initial window
        self.set_time_window(start_time_s, duration_s)
        self._update_fft_scoring_menu_enabled()

        # Wavelet spectrogram bars (channel list depends on the freshly loaded signals)
        self._refresh_spectrogram_panel_state()
        self._maybe_recompute_night_spectrogram()

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

        # Nav bar toggle buttons -> show/hide dock panels
        self._nav.scaling_panel_toggled.connect(self._on_scaling_panel_toggled)
        self._nav.layout_panel_toggled.connect(self._on_layout_panel_toggled)
        self._nav.spectrogram_panel_toggled.connect(self._on_spectrogram_panel_toggled)

        # Scaling dock panel signals
        self._scaling_panel.scaling_toggled.connect(self._on_scaling_toggle_nav)
        self._scaling_panel.scaling_gain_changed.connect(self._on_scaling_gain)
        self._scaling_panel.scaling_spacing_changed.connect(self._on_scaling_spacing)
        self._scaling_panel.scaling_auto_requested.connect(self._on_scaling_auto)
        self._scaling_panel.scaling_clip_changed.connect(self._on_scaling_clip)

        # Layout dock panel signals
        self._layout_panel.layout_toggled.connect(self._on_layout_toggle)
        self._layout_panel.layout_preset_changed.connect(self._on_layout_preset)
        self._layout_panel.layout_type_multiplier_changed.connect(self._on_layout_type_mult)
        self._layout_panel.layout_center_changed.connect(self._on_layout_center)
        self._layout_panel.layout_auto_requested.connect(self._on_layout_auto)
        self._layout_panel.guides_baselines_changed.connect(self._on_guide_baselines)
        self._layout_panel.guides_amplitude_changed.connect(self._on_guide_amplitude)
        self._layout_panel.guides_grid_fine_changed.connect(self._on_guide_grid_fine)
        self._layout_panel.guides_artifact_changed.connect(self._on_guide_artifact)

        # Keep dock panel visibility in sync when user closes the dock via its X button
        self._scaling_panel.visibilityChanged.connect(self._on_scaling_dock_visibility)
        self._layout_panel.visibilityChanged.connect(self._on_layout_dock_visibility)
        self._spectro_panel.visibilityChanged.connect(self._on_spectrogram_dock_visibility)

        # Spectrogram dock panel signals
        self._spectro_panel.night_toggled.connect(self._on_spectro_night_toggled)
        self._spectro_panel.window_toggled.connect(self._on_spectro_window_toggled)
        self._spectro_panel.night_height_changed.connect(self._on_spectro_night_height)
        self._spectro_panel.window_height_changed.connect(self._on_spectro_window_height)
        self._spectro_panel.channel_changed.connect(self._on_spectro_channel)
        self._spectro_panel.fmin_changed.connect(self._on_spectro_fmin)
        self._spectro_panel.fmax_changed.connect(self._on_spectro_fmax)
        self._spectro_panel.n_freqs_changed.connect(self._on_spectro_n_freqs)
        self._spectro_panel.freq_log_toggled.connect(self._on_spectro_freq_log)
        self._spectro_panel.colormap_changed.connect(self._on_spectro_colormap)
        self._spectro_panel.scale_changed.connect(self._on_spectro_scale)
        self._spectro_panel.use_filtered_toggled.connect(self._on_spectro_use_filtered)
        self._spectro_panel.onef_correction_toggled.connect(self._on_spectro_onef_correction)
        self._spectro_panel.opacity_changed.connect(self._on_spectro_opacity)
        self._spectro_panel.overlay_hypnogram_toggled.connect(self._on_spectro_overlay_toggled)
        self._spectro_panel.export_requested.connect(self._on_spectro_export)

        # Spectrogram bars click -> navigate (same behaviour as the hypnogram)
        self._spectro_night.navigate_to.connect(self._on_hypno_navigate)
        self._spectro_window.navigate_to.connect(self._on_hypno_navigate)

        # Hypnogram click -> navigate
        self._hypno.epoch_clicked.connect(self._on_epoch_clicked)
        self._hypno.epoch_context_stage.connect(self._on_hypno_context_stage)
        self._hypno.navigate_to.connect(self._on_hypno_navigate)
        self._hypno.zoom_requested.connect(self._on_hypno_zoom)
        self._hypno.night_start_requested.connect(self._on_night_start_requested)
        self._hypno.night_end_requested.connect(self._on_night_end_requested)
        self._pending_night_start_s: Optional[float] = None

        # Events bar click -> navigate
        self._events.event_clicked.connect(self._on_event_clicked)
        self._events.stage_clicked.connect(self._on_epoch_clicked)

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
        self._stop_spectrogram_workers()
        logger.info("[VIEWER-CHK-29] EEGViewerMainWindow closeEvent (fenêtre Qt se ferme)")
        super().closeEvent(event)

    def _stop_spectrogram_workers(self) -> None:
        """Wait for any in-flight spectrogram computation before teardown.

        QThread instances must not be destroyed while still running (crash
        risk), so we block briefly for a clean finish rather than closing
        the window out from under them.
        """
        for worker in (self._spectro_night_worker, self._spectro_window_worker):
            if worker is not None and worker.isRunning():
                worker.wait(5000)

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
        for sc in self._registered_nav_shortcuts:
            sc.deleteLater()
        self._registered_nav_shortcuts.clear()
        # ApplicationShortcut: robuste en mode hybride Tk+Qt
        # (les touches restent actives même si le focus est dans un sous-widget Qt).
        ctx = QtCore.Qt.ShortcutContext.ApplicationShortcut
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
            _K.Key_F,
        )
        for k in no_repeat_keys:
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.setContext(ctx)
            sc.setAutoRepeat(False)
            sc.activated.connect(
                functools.partial(self._dispatch_key_shortcuts_ints, _qt_key_int(k), 0),
            )
            self._registered_nav_shortcuts.append(sc)
        mod_ctrl = _keyboard_modifiers_int(QtCore.Qt.KeyboardModifier.ControlModifier)
        mod_ctrl_shift = _keyboard_modifiers_int(
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.ShiftModifier,
        )
        for desc, key_enum, mod_int in (
            ("Ctrl+F", _K.Key_F, mod_ctrl),
            ("Ctrl+Shift+N", _K.Key_N, mod_ctrl_shift),
        ):
            key_int = _qt_key_int(key_enum)
            sc = QtGui.QShortcut(QtGui.QKeySequence(desc), self)
            sc.setContext(ctx)
            sc.activated.connect(
                functools.partial(self._dispatch_key_shortcuts_ints, key_int, mod_int),
            )
            self._registered_nav_shortcuts.append(sc)

        for desc, key_enum in (("[", _K.Key_BracketLeft), ("]", _K.Key_BracketRight)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(desc), self)
            sc.setContext(ctx)
            sc.setAutoRepeat(False)
            sc.activated.connect(
                functools.partial(self._dispatch_key_shortcuts_ints, _qt_key_int(key_enum), 0),
            )
            self._registered_nav_shortcuts.append(sc)

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
        self._refresh_stage_indicator(t)
        self._spectro_night.set_window(t, self._viewer.duration_s)
        self._schedule_window_spectrogram_update(t, self._viewer.duration_s)
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
        self._spectro_night.set_window(self._viewer.start_s, dur)
        self._schedule_window_spectrogram_update(self._viewer.start_s, dur)

    def _on_epoch_clicked(self, t: float) -> None:
        dur = self._viewer.duration_s
        self._viewer.set_time_window(t, dur)
        self._hypno.set_window(t, dur)
        self._events.set_time_window(t, dur)
        self._nav.set_current_time(t)
        self._smart_nav.set_current_time(t)
        self._spectro_night.set_window(t, dur)
        self._schedule_window_spectrogram_update(t, dur)
        if self._on_navigate:
            self._schedule_tk_navigation_callback(t)

    def _on_hypno_navigate(self, t: float) -> None:
        dur = self._viewer.duration_s
        self._viewer.set_time_window(t, dur)
        self._hypno.set_window(t, dur)
        self._events.set_time_window(t, dur)
        self._nav.set_current_time(t)
        self._smart_nav.set_current_time(t)
        self._spectro_night.set_window(t, dur)
        self._schedule_window_spectrogram_update(t, dur)

    def _on_hypno_zoom(self, new_dur: float) -> None:
        start = self._viewer.start_s
        self._viewer.set_time_window(start, new_dur)
        self._hypno.set_window(start, new_dur)
        self._events.set_time_window(start, new_dur)
        self._nav.set_duration(new_dur)
        self._spectro_night.set_window(start, new_dur)
        self._schedule_window_spectrogram_update(start, new_dur)

    def _on_event_clicked(self, event: dict) -> None:
        onset = float(event.get("onset", 0.0))
        dur = self._viewer.duration_s
        self._viewer.set_time_window(onset, dur)
        self._hypno.set_window(onset, dur)
        self._events.set_time_window(onset, dur)
        self._nav.set_current_time(onset)
        self._smart_nav.set_current_time(onset)
        self._spectro_night.set_window(onset, dur)
        self._schedule_window_spectrogram_update(onset, dur)
        if self._on_navigate:
            self._schedule_tk_navigation_callback(onset)

    def _on_viewer_range(self, start_s: float, dur_s: float) -> None:
        # Pas d'appel _on_navigate ici : time_window_changed part à chaque scroll/zoom (saturation CESA).
        self._nav.set_current_time(start_s)
        self._nav.set_duration(dur_s)
        self._hypno.set_window(start_s, dur_s)
        self._events.set_time_window(start_s, dur_s)
        self._smart_nav.set_current_time(start_s)
        self._refresh_stage_indicator(start_s)
        self._spectro_night.set_window(start_s, dur_s)
        self._schedule_window_spectrogram_update(start_s, dur_s)

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
            self._refresh_annotation_ui()
            self._dashboard.set_event_counts(
                annotations=len(self._annotation_store),
            )
        elif result == 2 and dlg._ann_id:
            # Delete action
            self._annotation_store.remove(dlg._ann_id)
            self._viewer._refresh_view()
            self._refresh_annotation_ui()

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

    # ------------------------------------------------------------------
    # Panel toggle handlers (nav bar buttons -> dock visibility)
    # ------------------------------------------------------------------

    def _on_scaling_panel_toggled(self, checked: bool) -> None:
        self._scaling_panel.setVisible(checked)

    def _on_layout_panel_toggled(self, checked: bool) -> None:
        self._layout_panel.setVisible(checked)

    def _on_scaling_dock_visibility(self, visible: bool) -> None:
        self._nav.set_scaling_button(visible)

    def _on_layout_dock_visibility(self, visible: bool) -> None:
        self._nav.set_layout_button(visible)

    # ------------------------------------------------------------------
    # Scaling handlers
    # ------------------------------------------------------------------

    def _apply_scaling(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        self._viewer.apply_scaling(ctrl.scaling, getattr(self, '_channel_types', {}))

    def _on_scaling_toggle_nav(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.scaling.enabled = checked
        self._apply_scaling()
        ctrl.save_active_profile()

    def _on_scaling_gain(self, value: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.scaling.global_gain = value
        if ctrl.scaling.enabled:
            self._apply_scaling()

    def _on_scaling_spacing(self, value: int) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.scaling.spacing_uv = float(value)
        self._channel_list.spacing_slider.blockSignals(True)
        self._channel_list.spacing_slider.setValue(value)
        self._channel_list.spacing_slider.blockSignals(False)
        self._channel_list.spacing_label.setText(str(value))
        if ctrl.scaling.enabled:
            self._apply_scaling()

    def _on_scaling_auto(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ch_types = getattr(self, '_channel_types', {})
        auto_gains = ctrl.compute_auto_gains(ch_types, ctrl.scaling.spacing_uv)
        ctrl.scaling.per_channel_gains = auto_gains
        ctrl.scaling.mode = "auto"
        ctrl.scaling.enabled = True
        self._scaling_panel.set_state(
            enabled=True,
            gain=ctrl.scaling.global_gain,
            spacing=int(ctrl.scaling.spacing_uv),
            clip_on=ctrl.scaling.clipping_enabled,
            clip_val=ctrl.scaling.clip_value_uv,
        )
        self._apply_scaling()
        ctrl.save_active_profile()
        self._status.showMessage("Auto-scale applique", 3000)

    def _on_scaling_clip(self, enabled: bool, value: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.scaling.clipping_enabled = enabled
        ctrl.scaling.clip_value_uv = value
        if ctrl.scaling.enabled:
            self._apply_scaling()

    def _restore_scaling_state(self) -> None:
        """Sync scaling dock panel controls with controller state after file open."""
        ctrl = self._app_controller
        if ctrl is None:
            return
        sc = ctrl.scaling
        self._scaling_panel.set_state(
            enabled=sc.enabled,
            gain=sc.global_gain,
            spacing=int(sc.spacing_uv),
            clip_on=sc.clipping_enabled,
            clip_val=sc.clip_value_uv,
        )
        if sc.enabled:
            self._apply_scaling()

    # ------------------------------------------------------------------
    # Layout handlers
    # ------------------------------------------------------------------

    def _apply_layout(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ch_types = getattr(self, '_channel_types', {})
        medians = None
        if ctrl.layout.center_signal:
            medians = ctrl.compute_signal_medians()
        self._viewer.apply_layout(ctrl.layout, ch_types, medians)

    def _on_layout_toggle(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.layout.enabled = checked
        self._apply_layout()
        ctrl.save_active_profile()

    def _on_layout_preset(self, preset_key: str) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        from CESA.profile_schema import LAYOUT_PRESETS
        if preset_key == "custom":
            ctrl.layout.mode = "custom"
        else:
            ctrl.layout.apply_preset(preset_key)
            self._layout_panel.sync_multipliers(ctrl.layout.per_type_spacing_multiplier)
        if ctrl.layout.enabled:
            self._apply_layout()
        ctrl.save_active_profile()

    def _on_layout_type_mult(self, type_key: str, value: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.layout.per_type_spacing_multiplier[type_key] = value
        ctrl.layout.mode = "custom"
        if ctrl.layout.enabled:
            self._apply_layout()

    def _on_layout_center(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.layout.center_signal = checked
        if ctrl.layout.enabled:
            self._apply_layout()

    def _on_layout_auto(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ch_types = getattr(self, '_channel_types', {})
        result = ctrl.compute_auto_layout(ch_types)
        ctrl.layout.mode = "auto"
        ctrl.layout.per_type_spacing_multiplier = result["multipliers"]
        ctrl.layout.center_signal = result["center"]
        ctrl.layout.enabled = True
        self._layout_panel.set_state(
            enabled=True,
            mode="auto",
            multipliers=result["multipliers"],
            center=result["center"],
            baselines=ctrl.layout.show_baselines,
            amp_scale=ctrl.layout.show_amplitude_scale,
            grid_fine=ctrl.layout.show_grid_fine,
            artifact=ctrl.layout.show_artifact_highlight,
        )
        self._apply_layout()
        ctrl.save_active_profile()
        self._status.showMessage(
            f"Auto-layout applique (contexte: {result['context']})", 3000)

    def _on_guide_baselines(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl:
            ctrl.layout.show_baselines = checked
        self._viewer.set_show_baselines(checked)

    def _on_guide_amplitude(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl:
            ctrl.layout.show_amplitude_scale = checked
        self._viewer.set_show_amplitude_scale(checked)

    def _on_guide_grid_fine(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl:
            ctrl.layout.show_grid_fine = checked
        self._viewer.set_show_grid_fine(checked)

    def _on_guide_artifact(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl:
            ctrl.layout.show_artifact_highlight = checked
        self._viewer.set_show_artifact_highlight(checked)

    def _restore_layout_state(self) -> None:
        """Sync layout dock panel controls with controller state after file open."""
        ctrl = self._app_controller
        if ctrl is None:
            return
        lc = ctrl.layout
        self._layout_panel.set_state(
            enabled=lc.enabled,
            mode=lc.mode,
            multipliers=lc.per_type_spacing_multiplier,
            center=lc.center_signal,
            baselines=lc.show_baselines,
            amp_scale=lc.show_amplitude_scale,
            grid_fine=lc.show_grid_fine,
            artifact=lc.show_artifact_highlight,
        )
        if lc.show_baselines:
            self._viewer.set_show_baselines(True)
        if lc.show_amplitude_scale:
            self._viewer.set_show_amplitude_scale(True)
        if lc.show_grid_fine:
            self._viewer.set_show_grid_fine(True)
        if lc.show_artifact_highlight:
            self._viewer.set_show_artifact_highlight(True, lc.artifact_threshold_uv)
        if lc.enabled:
            self._apply_layout()

    # ------------------------------------------------------------------
    # Spectrogram (wavelet) handlers
    # ------------------------------------------------------------------

    def _on_spectrogram_panel_toggled(self, checked: bool) -> None:
        self._spectro_panel.setVisible(checked)

    def _on_spectrogram_dock_visibility(self, visible: bool) -> None:
        self._nav.set_spectrogram_button(visible)

    def _spectro_available_channels(self) -> List[str]:
        return list(self._signals_data.keys())

    def _spectro_resolve_channel(self) -> str:
        ctrl = self._app_controller
        names = self._spectro_available_channels()
        if not names:
            return ""
        cfg = ctrl.spectrogram if ctrl else None
        if cfg and cfg.channel and cfg.channel in names:
            return cfg.channel
        if ctrl is not None:
            return ctrl.resolve_default_spectrogram_channel(names)
        return names[0]

    def _refresh_spectro_overlay(self) -> None:
        ctrl = self._app_controller
        enabled = bool(ctrl.spectrogram.overlay_hypnogram) if ctrl else False
        self._spectro_night.set_hypnogram_overlay(self._hypnogram_labels, self._epoch_len, enabled)
        self._spectro_window.set_hypnogram_overlay(self._hypnogram_labels, self._epoch_len, enabled)

    def _apply_night_band_heights(self) -> None:
        """Hypnogram + night spectrogram share a fixed height budget.

        Spectro OFF -> hypnogram keeps its normal full height. Spectro ON
        -> the panel's height slider decides the split between the two.
        """
        ctrl = self._app_controller
        cfg = ctrl.spectrogram if ctrl else None
        enabled = bool(cfg.enabled_night) if cfg else False
        self._spectro_night.setVisible(enabled)
        if not enabled:
            self._hypno.setFixedHeight(100)
            return
        ratio = max(0.1, min(0.9, float(cfg.night_height_ratio)))
        total = 160
        spectro_h = max(20, min(140, int(round(total * ratio))))
        hypno_h = max(40, total - spectro_h)
        self._hypno.setFixedHeight(hypno_h)
        self._spectro_night.setFixedHeight(spectro_h)

    def _apply_window_band_height(self) -> None:
        ctrl = self._app_controller
        cfg = ctrl.spectrogram if ctrl else None
        enabled = bool(cfg.enabled_window) if cfg else False
        self._spectro_window.setVisible(enabled)
        if enabled and cfg:
            self._spectro_window.setFixedHeight(int(cfg.window_height_px))

    def _refresh_spectrogram_panel_state(self) -> None:
        """Sync the spectrogram dock panel + bars with controller state."""
        ctrl = self._app_controller
        if ctrl is None:
            return
        cfg = ctrl.spectrogram
        names = self._spectro_available_channels()
        channel = self._spectro_resolve_channel()
        self._spectro_panel.set_channels(names, channel)
        self._spectro_panel.set_state(
            enabled_night=cfg.enabled_night,
            enabled_window=cfg.enabled_window,
            night_height_ratio=cfg.night_height_ratio,
            window_height_px=cfg.window_height_px,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            n_freqs=cfg.n_freqs,
            freq_log=cfg.freq_log,
            colormap=cfg.colormap,
            scale=cfg.scale,
            use_filtered=cfg.use_filtered,
            onef_correction=cfg.onef_correction,
            opacity=cfg.opacity,
            overlay_hypnogram=cfg.overlay_hypnogram,
        )
        self._spectro_night.set_colormap(cfg.colormap)
        self._spectro_window.set_colormap(cfg.colormap)
        self._spectro_night.set_opacity(cfg.opacity)
        self._spectro_window.set_opacity(cfg.opacity)
        self._apply_night_band_heights()
        self._apply_window_band_height()
        self._refresh_spectro_overlay()
        if cfg.enabled_window:
            self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)

    def _on_spectro_night_toggled(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.enabled_night = checked
        self._apply_night_band_heights()
        if checked:
            self._maybe_recompute_night_spectrogram()
        ctrl.save_active_profile()

    def _on_spectro_window_toggled(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.enabled_window = checked
        self._apply_window_band_height()
        if checked:
            self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_night_height(self, ratio: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.night_height_ratio = ratio
        self._apply_night_band_heights()
        ctrl.save_active_profile()

    def _on_spectro_window_height(self, px: int) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.window_height_px = px
        self._apply_window_band_height()
        ctrl.save_active_profile()

    def _on_spectro_channel(self, channel: str) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.channel = channel
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_fmin(self, value: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.fmin = value
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_fmax(self, value: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.fmax = value
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_n_freqs(self, value: int) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.n_freqs = value
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_freq_log(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.freq_log = checked
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_colormap(self, name: str) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.colormap = name
        self._spectro_night.set_colormap(name)
        self._spectro_window.set_colormap(name)
        ctrl.save_active_profile()

    def _on_spectro_scale(self, scale: str) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.scale = scale
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_use_filtered(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.use_filtered = checked
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_onef_correction(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.onef_correction = checked
        self._maybe_recompute_night_spectrogram()
        self._schedule_window_spectrogram_update(self._viewer.start_s, self._viewer.duration_s)
        ctrl.save_active_profile()

    def _on_spectro_opacity(self, value: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.opacity = value
        self._spectro_night.set_opacity(value)
        self._spectro_window.set_opacity(value)
        ctrl.save_active_profile()

    def _on_spectro_overlay_toggled(self, checked: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.spectrogram.overlay_hypnogram = checked
        self._refresh_spectro_overlay()
        ctrl.save_active_profile()

    def _on_spectro_export(self) -> None:
        night_visible = self._spectro_night.isVisible()
        window_visible = self._spectro_window.isVisible()
        if not night_visible and not window_visible:
            QtWidgets.QMessageBox.information(
                self, "Exporter le spectrogramme",
                "Activez au moins un bandeau (nuit ou fenetre) avant d'exporter.",
            )
            return
        band = "nuit" if night_visible else "fenetre"
        target = self._spectro_night if night_visible else self._spectro_window
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter le spectrogramme", f"spectrogramme_{band}.png",
            "Images PNG (*.png)",
        )
        if not path:
            return
        if target.export_image(path):
            self._status.showMessage(f"Spectrogramme exporte : {path}", 4000)
        else:
            QtWidgets.QMessageBox.warning(self, "Export", "Echec de l'export de l'image.")

    # -- Night band: background thread, recomputed only when needed ------

    def _spectro_night_cache_signature(self, channel: str) -> Tuple[Any, ...]:
        ctrl = self._app_controller
        cfg = ctrl.spectrogram
        return (
            channel, round(float(cfg.fmin), 3), round(float(cfg.fmax), 3),
            int(cfg.n_freqs), bool(cfg.freq_log), str(cfg.scale),
            bool(cfg.use_filtered), bool(cfg.onef_correction),
        )

    def _maybe_recompute_night_spectrogram(self, force: bool = False) -> None:
        ctrl = self._app_controller
        if ctrl is None or not ctrl.spectrogram.enabled_night:
            return
        channel = self._spectro_resolve_channel()
        if not channel or channel not in self._signals_data:
            self._spectro_night.set_placeholder_text("Aucun canal disponible")
            return
        sig = self._spectro_night_cache_signature(channel)
        if (
            not force
            and sig == self._spectro_night_cache_key
            and self._spectro_night_result is not None
        ):
            self._spectro_night.set_data(self._spectro_night_result)
            return

        data, fs = self._signals_data[channel]
        if data is None or len(data) < 8 or fs <= 0:
            return

        cfg = ctrl.spectrogram
        total_dur = self._viewer._total_duration_s or (len(data) / fs)
        target_cols = int(max(500, min(3000, total_dur / max(1.0, self._epoch_len) * 6)))
        params = dict(
            fmin=cfg.fmin, fmax=cfg.fmax, n_freqs=cfg.n_freqs, freq_log=cfg.freq_log,
            scale=cfg.scale, use_filtered=cfg.use_filtered, onef_correction=cfg.onef_correction,
            target_fs=100.0, target_cols=target_cols,
        )
        self._spectro_night.set_placeholder_text("Calcul du spectrogramme (nuit) en cours...")

        worker = NightSpectrogramWorker(np.asarray(data, dtype=float), float(fs), params, self)
        self._spectro_night_worker = worker
        worker.result_ready.connect(
            functools.partial(self._on_night_spectrogram_ready, worker, sig))
        worker.failed.connect(functools.partial(self._on_night_spectrogram_failed, worker))
        worker.start()

    def _on_night_spectrogram_ready(
        self, worker: NightSpectrogramWorker, sig: Tuple[Any, ...], result: SpectrogramResult,
    ) -> None:
        if worker is not self._spectro_night_worker:
            return  # a newer computation superseded this one
        self._spectro_night_result = result
        self._spectro_night_cache_key = sig
        # Prefer the hypnogram's X span so both bars stay pixel-aligned.
        total_dur = float(getattr(self._hypno, "_total_duration_s", 0.0) or 0.0)
        if not total_dur:
            total_dur = float(self._viewer._total_duration_s or 0.0)
        if total_dur:
            self._spectro_night.set_total_duration(total_dur)
        self._spectro_night.set_data(result)
        self._spectro_night.set_window(self._viewer.start_s, self._viewer.duration_s)

    def _on_night_spectrogram_failed(self, worker: NightSpectrogramWorker, message: str) -> None:
        if worker is not self._spectro_night_worker:
            return
        logger.error("Spectrogramme (nuit) : %s", message)
        self._spectro_night.set_placeholder_text("Erreur de calcul du spectrogramme")

    # -- Window band: background thread, debounced on navigation ---------

    def _schedule_window_spectrogram_update(self, start_s: float, dur_s: float) -> None:
        ctrl = self._app_controller
        if ctrl is None or not ctrl.spectrogram.enabled_window:
            return
        self._spectro_window_pending = (float(start_s), float(dur_s))
        self._spectro_window_timer.start()

    def _recompute_window_spectrogram(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or not ctrl.spectrogram.enabled_window:
            return
        start_s, dur_s = self._spectro_window_pending
        channel = self._spectro_resolve_channel()
        if not channel or channel not in self._signals_data:
            self._spectro_window.set_placeholder_text("Aucun canal disponible")
            return
        data, fs = self._signals_data[channel]
        if data is None or fs <= 0:
            return
        i0 = max(0, int(start_s * fs))
        i1 = min(len(data), int((start_s + dur_s) * fs))
        if i1 - i0 < 8:
            return
        seg = np.asarray(data[i0:i1], dtype=float)
        cfg = ctrl.spectrogram
        params = dict(
            fmin=cfg.fmin, fmax=cfg.fmax, n_freqs=cfg.n_freqs, freq_log=cfg.freq_log,
            scale=cfg.scale, use_filtered=cfg.use_filtered, onef_correction=cfg.onef_correction,
            time_offset_s=start_s, target_cols=1000,
        )
        self._spectro_window.set_placeholder_text("Calcul en cours...")

        worker = NightSpectrogramWorker(seg, float(fs), params, self)
        self._spectro_window_worker = worker
        worker.result_ready.connect(
            functools.partial(self._on_window_spectrogram_ready, worker, start_s, dur_s))
        worker.failed.connect(functools.partial(self._on_window_spectrogram_failed, worker))
        worker.start()

    def _on_window_spectrogram_ready(
        self,
        worker: NightSpectrogramWorker,
        start_s: float,
        dur_s: float,
        result: SpectrogramResult,
    ) -> None:
        if worker is not self._spectro_window_worker:
            return  # stale: a newer request superseded this one
        self._spectro_window.set_window(start_s, dur_s)
        self._spectro_window.set_data(result)

    def _on_window_spectrogram_failed(self, worker: NightSpectrogramWorker, message: str) -> None:
        if worker is not self._spectro_window_worker:
            return
        logger.error("Spectrogramme (fenetre) : %s", message)
        self._spectro_window.set_placeholder_text("Erreur de calcul")

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
        self._current_stage_readout = QtWidgets.QLabel("Epoque: —")
        self._current_stage_readout.setMinimumWidth(130)
        self._current_stage_readout.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Bold))
        lay.addWidget(self._current_stage_readout)
        lay.addSpacing(8)
        self._stage_buttons: List[QtWidgets.QPushButton] = []
        stages = [
            ("W", "WK / Wake (touche 0)"),
            ("N1", "N1 (touche 1)"),
            ("N2", "N2 (touche 2)"),
            ("N3", "N3 (touche 3)"),
            ("R", "REM (touche 4)"),
            ("U", "Artefact / Inconnu (touche 5)"),
        ]
        for st, tip in stages:
            btn = QtWidgets.QPushButton(st)
            btn.setToolTip(tip)
            btn.setEnabled(False)
            btn.setFixedWidth(32)
            btn.clicked.connect(functools.partial(self._invoke_stage_current_epoch, st))
            lay.addWidget(btn)
            self._stage_buttons.append(btn)
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
                return
            except Exception:
                logger.debug("on_open_manual_scoring_editor failed", exc_info=True)
        ctrl = self._app_controller
        if ctrl is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return
        if ctrl.sleep_scoring_data is None or ctrl.sleep_scoring_data.empty:
            QtWidgets.QMessageBox.information(
                self, "Editeur de scoring",
                "Aucun scoring disponible. Lancez un scoring automatique ou importez-en un d'abord.",
            )
            return
        from .dialogs.scoring_editor_dialog import ScoringEditorDialog
        ScoringEditorDialog(self, ctrl).exec()

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
        for sc in self._registered_scoring_shortcuts:
            sc.deleteLater()
        self._registered_scoring_shortcuts.clear()

        # ApplicationShortcut: robuste en mode hybride Tk+Qt.
        ctx = QtCore.Qt.ShortcutContext.ApplicationShortcut
        if self._on_request_stage_for_current_epoch is not None:
            pairs = (
                (QtCore.Qt.Key.Key_0, "W"),
                (QtCore.Qt.Key.Key_1, "N1"),
                (QtCore.Qt.Key.Key_2, "N2"),
                (QtCore.Qt.Key.Key_3, "N3"),
                (QtCore.Qt.Key.Key_4, "R"),
                (QtCore.Qt.Key.Key_5, "U"),
            )
            for key_enum, st in pairs:
                sc = QtGui.QShortcut(QtGui.QKeySequence(key_enum), self)
                sc.setContext(ctx)
                sc.setAutoRepeat(False)
                sc.activated.connect(functools.partial(self._invoke_stage_current_epoch, st))
                self._registered_scoring_shortcuts.append(sc)
        if self._on_set_manual_auto_advance is not None:
            sc_toggle = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+A"), self)
            sc_toggle.setContext(ctx)
            sc_toggle.setAutoRepeat(False)
            sc_toggle.activated.connect(self._toggle_manual_auto_advance)
            self._registered_scoring_shortcuts.append(sc_toggle)
        # Toujours installer : run.py n'a pas de callback Tk ; le repli ouvre l'analyse spectrale Qt.
        self._sc_fft_scoring = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+F"), self)
        self._sc_fft_scoring.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._sc_fft_scoring.setAutoRepeat(False)
        self._sc_fft_scoring.activated.connect(self._toggle_fft_scoring_view)
        self._registered_scoring_shortcuts.append(self._sc_fft_scoring)
        self._sc_fft_scoring_alt = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Alt+X"), self)
        self._sc_fft_scoring_alt.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._sc_fft_scoring_alt.setAutoRepeat(False)
        self._sc_fft_scoring_alt.activated.connect(self._toggle_fft_scoring_view)
        self._registered_scoring_shortcuts.append(self._sc_fft_scoring_alt)
        self._sc_fft_scoring_f = QtGui.QShortcut(QtGui.QKeySequence("F"), self)
        self._sc_fft_scoring_f.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._sc_fft_scoring_f.setAutoRepeat(False)
        self._sc_fft_scoring_f.activated.connect(self._toggle_fft_scoring_view)
        self._registered_scoring_shortcuts.append(self._sc_fft_scoring_f)
        logger.info("[FFT-SHORTCUT] Qt ApplicationShortcuts installed: F, Ctrl+Shift+F and Ctrl+Alt+X")

    def _toggle_manual_auto_advance(self) -> None:
        setter = self._on_set_manual_auto_advance
        if setter is None:
            return
        current = False
        if self._on_get_manual_auto_advance is not None:
            try:
                current = bool(self._on_get_manual_auto_advance())
            except Exception:
                current = False
        try:
            setter(not current)
            if hasattr(self, "_act_auto_advance_manual") and self._act_auto_advance_manual is not None:
                self._act_auto_advance_manual.setChecked(not current)
        except Exception:
            logger.debug("toggle manual auto advance failed", exc_info=True)

    def _update_fft_scoring_menu_enabled(self) -> None:
        act = getattr(self, "_act_fft_scoring", None)
        if act is None:
            return
        has_ctrl = self._app_controller is not None
        raw_ok = has_ctrl and getattr(self._app_controller, "raw", None) is not None
        has_signals = bool(self._signals_data)
        fft_available = (
            self._on_toggle_fft_scoring_view is not None
            or (raw_ok and has_signals)
        )
        act.setEnabled(fft_available)

    def _toggle_fft_scoring_view(self) -> None:
        logger.info("[FFT-SHORTCUT] Qt _toggle_fft_scoring_view triggered")
        fw = QtWidgets.QApplication.focusWidget()
        if fw is not None and isinstance(
            fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
        ):
            return
        cb = self._on_toggle_fft_scoring_view
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.debug("toggle fft scoring view failed", exc_info=True)
            return
        if self._fft_scoring_dialog is not None and self._fft_scoring_dialog.isVisible():
            self._fft_scoring_dialog.close()
            self._fft_scoring_dialog = None
            return
        self._open_fft_scoring_standalone()

    def _open_fft_scoring_standalone(self) -> None:
        """Mode run.py : même rôle que la fenêtre Tk « FFT scoring », via PSD Welch Qt."""
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(
                self, "FFT scoring", "Chargez un enregistrement.",
            )
            return
        if not self._signals_data:
            QtWidgets.QMessageBox.warning(
                self, "FFT scoring", "Aucun signal affiché dans le viewer.",
            )
            return
        from CESA.qt_viewer.dialogs.spectral_dialog import SpectralAnalysisDialog

        el = float(self._epoch_len) if self._epoch_len > 0 else float(
            getattr(ctrl, "scoring_epoch_duration", 30.0),
        )
        t0 = float(self._viewer.start_s)
        epoch_start = float(int(t0 // el) * el)
        dlg = SpectralAnalysisDialog(
            self,
            dict(self._signals_data),
            list(self._signals_data.keys()),
            epoch_start,
            el,
        )
        dlg.setWindowTitle("FFT scoring (époque ciblée)")
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._fft_scoring_dialog = dlg
        dlg.finished.connect(lambda *_: setattr(self, "_fft_scoring_dialog", None))
        dlg.show()

    # ------------------------------------------------------------------
    # AppController integration
    # ------------------------------------------------------------------

    def set_app_controller(self, ctrl) -> None:
        """Wire the application controller and rebuild menus."""
        self._app_controller = ctrl
        self._on_request_stage_for_current_epoch = self._ctrl_stage_current_epoch
        self._on_request_stage_at_epoch_time = self._ctrl_stage_at_time
        self._on_set_manual_auto_advance = self._ctrl_set_manual_auto_advance
        self._on_get_manual_auto_advance = self._ctrl_get_manual_auto_advance
        self.menuBar().clear()
        self._build_menus()
        self._install_scoring_shortcuts()
        self._refresh_night_window_display()
        for btn in getattr(self, "_stage_buttons", []):
            btn.setEnabled(True)

    def _ctrl_set_manual_auto_advance(self, enabled: bool) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.manual_scoring_auto_advance = bool(enabled)
        logger.info(
            "[SCORING] Auto-avance %s",
            "activée" if enabled else "désactivée",
        )
        act = getattr(self, "_act_auto_advance_manual", None)
        if act is not None:
            act.blockSignals(True)
            act.setChecked(bool(enabled))
            act.blockSignals(False)

    def _ctrl_get_manual_auto_advance(self) -> bool:
        ctrl = self._app_controller
        if ctrl is None:
            return True
        return bool(getattr(ctrl, "manual_scoring_auto_advance", True))

    def _advance_viewer_after_stage(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        el = float(self._epoch_len) if self._epoch_len > 0 else float(
            getattr(ctrl, "scoring_epoch_duration", 30.0),
        )
        next_t = float(self._viewer.start_s) + el
        max_start = max(
            0.0,
            float(ctrl.total_duration_s) - float(self._viewer.duration_s),
        )
        next_t = min(next_t, max_start)
        self.set_time_window(next_t, float(self._viewer.duration_s))
        ctrl.current_time = next_t

    def _ctrl_stage_current_epoch(self, stage: str) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        if not ctrl.ensure_scoring_dataframe():
            return
        epoch_idx = int(self._viewer.start_s / self._epoch_len) if self._epoch_len > 0 else 0
        df = ctrl.sleep_scoring_data
        try:
            if df is not None and epoch_idx >= len(df.index):
                epoch_idx = len(df.index) - 1
            if df is not None and epoch_idx < 0:
                epoch_idx = 0
            ctrl.sleep_scoring_data.loc[epoch_idx, 'stage'] = stage
            ctrl.scoring_dirty = True
            hyp = ctrl.build_hypnogram_tuple()
            if hyp:
                self.set_hypnogram(hyp)
            self.set_scoring_annotations(ctrl.build_scoring_annotations())
        except Exception:
            logger.debug("stage edit failed", exc_info=True)
            return
        if getattr(ctrl, "manual_scoring_auto_advance", True):
            self._advance_viewer_after_stage()

    def _ctrl_stage_at_time(self, onset_s: float, stage: str) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        if not ctrl.ensure_scoring_dataframe():
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
        self.setWindowTitle(f"CESA v0.0beta2.0 - {file_name}")
        self._status.showMessage(f"Fichier charge : {file_name}", 5000)

        self._restore_scaling_state()
        self._restore_layout_state()
        self._offer_scoring_import(ctrl)

    def _offer_scoring_import(self, ctrl) -> None:
        """After opening a recording, propose to import a scoring file."""
        has_embedded = ctrl.has_embedded_scoring()

        if has_embedded:
            msg = (
                "Voulez-vous importer un scoring ?\n\n"
                "  - Oui (Excel) : importer depuis un fichier Excel\n"
                "  - Oui (EDF+) : utiliser les annotations du fichier charge\n"
                "  - Non : continuer sans scoring"
            )
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Importer un scoring ?")
            box.setText(msg)
            btn_excel = box.addButton("Oui (Excel)", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            btn_edf = box.addButton("Oui (EDF+)", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            btn_no = box.addButton("Non", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(btn_excel)
            box.exec()
            clicked = box.clickedButton()
        else:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Importer un scoring ?",
                "Voulez-vous importer un scoring Excel\n"
                "pour cet enregistrement ?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            clicked = None
            btn_excel = None
            btn_edf = None
            btn_no = None
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                clicked = btn_excel  # treat as Excel
            else:
                return

        if has_embedded and clicked == btn_no:
            return

        if has_embedded and clicked == btn_edf:
            if ctrl.import_embedded_scoring():
                hyp = ctrl.build_hypnogram_tuple()
                if hyp:
                    self.set_hypnogram(hyp)
                self.set_scoring_annotations(ctrl.build_scoring_annotations())
                n = len(ctrl.sleep_scoring_data)
                self._status.showMessage(f"Scoring EDF+ importe : {n} epoques", 5000)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Attention",
                    "Les annotations de scoring n'ont pas pu etre importees.",
                )
            return

        # File import path (default)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importer scoring", "",
            "Scoring (*.csv *.xlsx *.xls *.json);;Tous (*.*)",
        )
        if not path:
            return
        if ctrl.import_scoring_file(path):
            hyp = ctrl.build_hypnogram_tuple()
            if hyp:
                self.set_hypnogram(hyp)
            self.set_scoring_annotations(ctrl.build_scoring_annotations())
            self._refresh_night_window_display()
            self.update_dashboard()
            n = len(ctrl.sleep_scoring_data) if ctrl.sleep_scoring_data is not None else 0
            self._status.showMessage(f"Scoring importe : {n} epoques", 5000)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Erreur", "L'import du scoring a echoue.",
            )

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

        from .dialogs.auto_scoring_config_dialog import AutoScoringConfigDialog
        ch_names = list(ctrl.raw.ch_names) if ctrl.raw is not None else []
        cfg = AutoScoringConfigDialog(self, ctrl, channel_names=ch_names)
        if cfg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        method = cfg.selected_method
        if method is None:
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
            self.update_dashboard()
            self._status.showMessage(f"Scoring {method} termine.", 5000)
        else:
            reason = getattr(ctrl, "last_auto_scoring_error", None)
            detail = f"\n\nCause :\n{reason}" if reason else ""
            QtWidgets.QMessageBox.warning(
                self, "Scoring",
                f"Le scoring automatique ({method}) a echoue.{detail}")

    def _menu_import_scoring_excel(self) -> None:
        self._menu_import_scoring_file()

    def _menu_import_scoring_file(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importer scoring", "",
            "Scoring (*.csv *.xlsx *.xls *.json);;Tous (*.*)",
        )
        if not path:
            return
        if ctrl.import_scoring_file(path):
            hyp = ctrl.build_hypnogram_tuple()
            if hyp:
                self.set_hypnogram(hyp)
            self.set_scoring_annotations(ctrl.build_scoring_annotations())
            self._refresh_night_window_display()
            self.update_dashboard()
            self._status.showMessage(
                f"Scoring importe ({len(ctrl.sleep_scoring_data)} epoques).", 5000,
            )
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
            hyp = ctrl.build_hypnogram_tuple()
            if hyp:
                self.set_hypnogram(hyp)
            self.set_scoring_annotations(ctrl.build_scoring_annotations())
            n = len(ctrl.sleep_scoring_data) if ctrl.sleep_scoring_data is not None else 0
            self._status.showMessage(f"Scoring EDF+ importe ({n} epoques).", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Import du scoring echoue.")

    def _menu_save_scoring(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.sleep_scoring_data is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Aucun scoring disponible.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Sauvegarder scoring", "",
            "CSV (*.csv);;JSON (*.json);;Tous (*.*)",
        )
        if not path:
            return
        if ctrl.save_scoring_file(path):
            self._status.showMessage(f"Scoring exporte: {path}", 5000)
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Export du scoring echoue.")

    def _menu_save_scoring_csv(self) -> None:
        self._menu_save_scoring()

    def _parse_time_to_seconds(self, raw: str) -> float:
        txt = (raw or "").strip()
        if not txt:
            raise ValueError("empty")
        if ":" not in txt:
            return float(txt) * 60.0
        parts = txt.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("invalid time")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
        return hours * 3600.0 + minutes * 60.0 + seconds

    def _refresh_night_window_display(self) -> None:
        ctrl = self._app_controller
        if ctrl is not None and ctrl.has_night_window:
            self._hypno.set_night_window(ctrl.night_start_s, ctrl.night_end_s)
        else:
            self._hypno.set_night_window(None, None)
        self._update_night_window_status()

    def _update_night_window_status(self) -> None:
        ctrl = self._app_controller
        if ctrl is not None and ctrl.has_night_window:
            self._status.showMessage(
                f"Fenetre analyse: {ctrl.night_start_s / 60.0:.1f}–{ctrl.night_end_s / 60.0:.1f} min",
                0,
            )

    def _apply_night_window(self, start_s: float, end_s: float) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        if not ctrl.set_night_window(start_s, end_s):
            QtWidgets.QMessageBox.warning(
                self, "Fenetre analyse",
                "Fenetre invalide (debut doit etre avant la fin).",
            )
            return
        self._pending_night_start_s = None
        self._refresh_night_window_display()
        self.update_dashboard()

    def _on_night_start_requested(self, time_s: float) -> None:
        self._pending_night_start_s = float(time_s)
        self._status.showMessage(
            f"Debut nuit: {time_s / 60.0:.1f} min — definissez la fin (clic droit ou ])",
            5000,
        )

    def _on_night_end_requested(self, time_s: float) -> None:
        start_s = self._pending_night_start_s
        if start_s is None:
            ctrl = self._app_controller
            if ctrl is not None and ctrl.night_start_s is not None:
                start_s = float(ctrl.night_start_s)
            else:
                QtWidgets.QMessageBox.information(
                    self, "Fenetre analyse",
                    "Definissez d'abord le debut de nuit (clic droit ou [).",
                )
                return
        self._apply_night_window(float(start_s), float(time_s))

    def _menu_set_night_window(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        start_default = ""
        end_default = ""
        if ctrl.has_night_window:
            start_default = f"{ctrl.night_start_s / 60.0:.0f}"
            end_default = f"{ctrl.night_end_s / 60.0:.0f}"
        start_txt, ok1 = QtWidgets.QInputDialog.getText(
            self, "Fenetre d'analyse", "Debut nuit (HH:MM ou minutes):",
            text=start_default,
        )
        if not ok1:
            return
        end_txt, ok2 = QtWidgets.QInputDialog.getText(
            self, "Fenetre d'analyse", "Fin nuit (HH:MM ou minutes):",
            text=end_default,
        )
        if not ok2:
            return
        if not start_txt.strip() or not end_txt.strip():
            ctrl.clear_night_window()
            self._refresh_night_window_display()
            self.update_dashboard()
            self._status.showMessage("Fenetre d'analyse effacee.", 3000)
            return
        try:
            start_s = self._parse_time_to_seconds(start_txt)
            end_s = self._parse_time_to_seconds(end_txt)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Fenetre analyse",
                "Format invalide. Utilisez HH:MM ou des minutes.",
            )
            return
        self._apply_night_window(start_s, end_s)

    def _menu_clear_night_window(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return
        ctrl.clear_night_window()
        self._pending_night_start_s = None
        self._refresh_night_window_display()
        self.update_dashboard()
        self._status.showMessage("Fenetre d'analyse effacee.", 3000)

    def _menu_export_psg_csv(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.sleep_scoring_data is None or ctrl.sleep_scoring_data.empty:
            QtWidgets.QMessageBox.warning(self, "Attention", "Aucun scoring disponible.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter rapport PSG", "", "CSV (*.csv);;Tous (*.*)",
        )
        if not path:
            return
        try:
            from CESA.sleep_metrics import compute_psg_summary, export_psg_summary_csv
            df = ctrl.sleep_scoring_data
            if df is None or df.empty:
                QtWidgets.QMessageBox.warning(self, "Attention", "Aucune epoque dans la fenetre.")
                return
            summary = compute_psg_summary(
                df,
                ctrl.scoring_epoch_duration,
                night_start_s=ctrl.night_start_s,
                night_end_s=ctrl.night_end_s,
            )
            export_psg_summary_csv(
                summary, path, annotations=self._annotation_store.to_event_list(),
            )
            self._status.showMessage(f"Rapport PSG exporte: {path}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Export PSG echoue: {exc}")

    def _menu_compare_scoring(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            return

        def _empty(df) -> bool:
            return df is None or df.empty

        auto_missing = _empty(getattr(ctrl, "auto_scoring_data", None))
        manual_missing = _empty(getattr(ctrl, "manual_scoring_data", None))
        if auto_missing and manual_missing:
            QtWidgets.QMessageBox.warning(
                self, "Comparaison impossible",
                "Il faut un scoring automatique ET un scoring manuel pour comparer.\n"
                "Aucun des deux n'est en memoire.")
            return
        if auto_missing:
            QtWidgets.QMessageBox.warning(
                self, "Comparaison impossible",
                "Aucun scoring automatique en memoire.\n\n"
                "Lancez Sommeil > Scoring automatique. L'import d'un scoring manuel "
                "n'ecrase plus le resultat automatique : les deux peuvent coexister, "
                "dans l'ordre que vous voulez.")
            return
        if manual_missing:
            QtWidgets.QMessageBox.warning(
                self, "Comparaison impossible",
                "Aucun scoring manuel de reference.\n\n"
                "Importez un scoring (Sommeil > Importer scoring) ou scorez a la main "
                "dans l'editeur en choisissant la cible 'Scoring manuel'.")
            return

        merged = ctrl.compare_scoring()
        if merged is None or merged.empty:
            QtWidgets.QMessageBox.information(
                self, "Comparaison",
                "Aucune epoque comparable : les deux scorings ne se recouvrent pas "
                "dans le temps, ou toutes les epoques communes sont non scorees (U).")
            return
        from .dialogs.scoring_comparison_dialog import ScoringComparisonDialog
        ScoringComparisonDialog(
            self, merged, epoch_len=ctrl.scoring_epoch_duration, controller=ctrl,
        ).exec()

    def _menu_sweep_scoring(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return
        manual = getattr(ctrl, "manual_scoring_data", None)
        if manual is None or manual.empty:
            QtWidgets.QMessageBox.warning(
                self, "Comparaison impossible",
                "Cet outil mesure chaque configuration face a un scoring manuel "
                "de reference, qui n'est pas en memoire.\n\n"
                "Importez un scoring (Sommeil > Importer scoring) ou scorez a la "
                "main dans l'editeur en choisissant la cible 'Scoring manuel'.")
            return

        from .dialogs.scoring_sweep_dialog import ScoringSweepDialog
        dlg = ScoringSweepDialog(self, ctrl)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        config = dlg.applied_config
        if config is None:
            return
        self._status.showMessage(
            f"Configuration appliquee : {config.label}. Relancez le scoring "
            "automatique pour l'utiliser.", 8000,
        )

    def _menu_scoring_info(self) -> None:
        from CESA.qt_viewer.dialogs.psg_report_dialog import PsgReportDialog
        dlg = PsgReportDialog.open_from_controller(self, self._app_controller)
        if dlg is not None:
            self._psg_report_dialog = dlg

    def _menu_show_psg_report(self) -> None:
        self._menu_scoring_info()

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

    def _menu_micro_events(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(
                self, "Attention", "Chargez d'abord un enregistrement.",
            )
            return
        types = ctrl.channel_types_dict()
        all_chs = list(ctrl.derivations.keys()) if ctrl.derivations else list(types.keys())
        eeg_chs = [ch for ch, t in types.items() if t == "eeg"]
        eog_chs = [ch for ch, t in types.items() if t == "eog"]
        # Prefer derivations order if types missed some names
        if not eeg_chs and ctrl.derivations:
            eeg_chs = [
                ch for ch in ctrl.derivations
                if cesa_detect_signal_type(ch) == "eeg"
            ]
        if not eog_chs and ctrl.derivations:
            eog_chs = [
                ch for ch in ctrl.derivations
                if cesa_detect_signal_type(ch) == "eog"
            ]
        has_scoring = ctrl.get_active_scoring_df() is not None
        from .dialogs.micro_events_dialog import MicroEventsDialog
        dlg = MicroEventsDialog(
            self,
            ctrl,
            eeg_chs,
            eog_chs,
            all_channels=all_chs,
            channel_types=types,
            has_scoring=has_scoring,
        )
        dlg.applied.connect(self._apply_micro_event_annotations)
        dlg.exec()

    def _apply_micro_event_annotations(self, payload: list) -> None:
        """Push auto-detected micro-events into the annotation store."""
        if not payload:
            return
        # Remove previous auto micro-events of the same types to avoid duplicates
        micro_types = {"spindle", "slow_wave", "rem_burst", "k_complex"}
        to_remove = [
            a.id for a in self._annotation_store.all()
            if a.source == "auto" and a.ann_type in micro_types
        ]
        for aid in to_remove:
            self._annotation_store.remove(aid)
        for item in payload:
            try:
                self._annotation_store.add(Annotation(
                    onset_s=float(item.get("onset_s", 0.0)),
                    duration_s=float(item.get("duration_s", 0.0)),
                    ann_type=str(item.get("ann_type", "custom")),
                    channel=str(item.get("channel", "") or ""),
                    text=str(item.get("text", "") or ""),
                    confidence=float(item.get("confidence", 1.0)),
                    source="auto",
                ))
            except Exception as exc:
                logger.warning("Skip micro annotation: %s", exc)
        self._refresh_annotation_ui()
        self._status.showMessage(
            f"{len(payload)} micro-evenement(s) ajoutes aux annotations", 5000,
        )

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

    def _native_stage_signals(self):
        """Return ``(signals, channels, scoring_df, epoch_seconds)`` for the
        loaded recording, or ``None`` when no recording is available."""
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(
                self, "Attention", "Chargez d'abord un enregistrement."
            )
            return None
        signals = ctrl.build_viewer_signals()
        channels = ctrl.psg_channels_used or list(signals.keys())
        try:
            scoring_df = ctrl.get_active_scoring_df()
        except Exception:
            scoring_df = ctrl.sleep_scoring_data
        epoch = getattr(ctrl, "scoring_epoch_duration", 30.0)
        return signals, channels, scoring_df, float(epoch)

    def _open_stage_psd_native(self, method: str, before_after: bool) -> None:
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, scoring_df, epoch = data
        from .dialogs.stage_psd_dialog import StagePSDDialog
        StagePSDDialog(
            self,
            signals=signals,
            channels=channels,
            scoring_df=scoring_df,
            epoch_seconds=epoch,
            initial_method=method,
            initial_before_after=before_after,
        ).exec()

    def _menu_stage_psd_fft(self) -> None:
        cb = self._on_open_stage_psd_fft
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.debug("open_stage_psd_fft failed", exc_info=True)
            return
        self._open_stage_psd_native(method="fft", before_after=False)

    def _menu_stage_psd_before_after(self) -> None:
        cb = self._on_open_stage_psd_before_after
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.debug("open_stage_psd_before_after failed", exc_info=True)
            return
        self._open_stage_psd_native(method="welch", before_after=True)

    def _menu_specparam(self) -> None:
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, scoring_df, epoch = data
        from .dialogs.specparam_dialog import SpecParamDialog
        SpecParamDialog(
            self,
            signals=signals,
            channels=channels[:16],
            scoring_df=scoring_df,
            epoch_seconds=epoch,
        ).exec()

    def _menu_night_qc(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or not ctrl.derivations:
            QtWidgets.QMessageBox.warning(
                self, "Attention", "Chargez d'abord un enregistrement.",
            )
            return
        from .dialogs.night_qc_dialog import NightQcDialog
        NightQcDialog(self, ctrl).exec()
        self.update_dashboard()

    def _menu_wavelet_spectrogram(self) -> None:
        cb = self._on_open_wavelet_spectrogram
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.debug("open_wavelet_spectrogram failed", exc_info=True)
            return
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, scoring_df, _epoch = data
        ctrl = self._app_controller
        from .dialogs.wavelet_spectrogram_dialog import WaveletSpectrogramDialog
        WaveletSpectrogramDialog(
            self,
            signals=signals,
            channels=channels,
            scoring_df=scoring_df,
            current_time=getattr(ctrl, "current_time", 0.0),
            duration=getattr(ctrl, "duration", 30.0),
        ).exec()

    def _menu_batch_fft_automation(self) -> None:
        cb = self._on_open_batch_fft_automation
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.debug("open_batch_fft_automation failed", exc_info=True)
            return
        from .dialogs.batch_analysis_dialog import BatchAnalysisDialog
        BatchAnalysisDialog(self).exec()

    def _menu_group_analysis(self) -> None:
        cb = self._on_open_group_analysis
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.debug("open_group_analysis failed", exc_info=True)
            return
        from .dialogs.group_analysis_dialog import GroupAnalysisDialog
        GroupAnalysisDialog(self).exec()

    def _menu_entropy_analysis(self) -> None:
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, scoring_df, epoch = data
        from .dialogs.entropy_dialog import EntropyAnalysisDialog
        EntropyAnalysisDialog(self, signals, channels, scoring_df, epoch).exec()

    def _menu_temporal_analysis(self) -> None:
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, scoring_df, epoch = data
        from .dialogs.temporal_analysis_dialog import TemporalAnalysisDialog
        TemporalAnalysisDialog(self, signals, channels, scoring_df, epoch).exec()

    def _menu_connectivity_analysis(self) -> None:
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, scoring_df, epoch = data
        from .dialogs.connectivity_dialog import ConnectivityDialog
        ConnectivityDialog(self, signals, channels, scoring_df, epoch).exec()

    def _menu_artifact_detection(self) -> None:
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, _scoring_df, _epoch = data
        from .dialogs.artifact_detection_dialog import ArtifactDetectionDialog
        ArtifactDetectionDialog(self, signals, channels).exec()

    def _menu_ocular_cleaning(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return
        signals = dict(self._signals_data) if self._signals_data else ctrl.build_viewer_signals()
        if not signals:
            QtWidgets.QMessageBox.warning(self, "Attention", "Aucun signal charge.")
            return
        channels = list(signals.keys())
        try:
            t0 = float(getattr(self._viewer, "start_s", 0.0) or 0.0)
            dur = float(getattr(self._viewer, "duration_s", 30.0) or 30.0)
        except Exception:
            t0, dur = 0.0, 30.0

        def _apply(cleaned: Dict[str, np.ndarray]) -> None:
            originals = {
                name: np.asarray(self._signals_data[name][0], dtype=float)
                for name in cleaned
                if name in self._signals_data
            }
            n = ctrl.apply_ocular_cleaning(cleaned, originals=originals)
            for name, arr in cleaned.items():
                if name not in self._signals_data:
                    continue
                fs = self._signals_data[name][1]
                self._signals_data[name] = (np.asarray(arr, dtype=float), fs)
            self._viewer.set_signals(dict(self._signals_data))
            if hasattr(self, "_sync_cursor") and self._sync_cursor is not None:
                self._sync_cursor.set_signals(dict(self._signals_data))
            self._status.showMessage(
                f"Nettoyage oculaire applique ({n} canaux, memoire) — fichier non modifie.",
                8000,
            )

        def _restore() -> None:
            restored = ctrl.restore_ocular_backup()
            for name, arr in restored.items():
                if name in self._signals_data:
                    fs = self._signals_data[name][1]
                    self._signals_data[name] = (arr, fs)
            # Also pull any derivation keys that match
            for ch, sig in ctrl.build_viewer_signals().items():
                if ch in self._signals_data and ch not in restored:
                    self._signals_data[ch] = sig
            self._viewer.set_signals(dict(self._signals_data))
            if hasattr(self, "_sync_cursor") and self._sync_cursor is not None:
                self._sync_cursor.set_signals(dict(self._signals_data))
            self._status.showMessage(
                f"Signaux originaux restaures ({len(restored)} canaux).",
                5000,
            )

        from .dialogs.ocular_cleaning_dialog import OcularCleaningDialog
        OcularCleaningDialog(
            self,
            signals,
            channels,
            channel_types=dict(self._channel_types),
            preview_t0=t0,
            preview_duration=dur,
            has_active_cleaning=ctrl.has_ocular_cleaning(),
            on_apply=_apply,
            on_restore=_restore,
        ).exec()

    def _menu_ica_cleaning(self) -> None:
        ctrl = self._app_controller
        if ctrl is None:
            QtWidgets.QMessageBox.warning(
                self, "Attention", "Chargez d'abord un enregistrement.",
            )
            return
        signals = dict(self._signals_data) if self._signals_data else ctrl.build_viewer_signals()
        if not signals:
            QtWidgets.QMessageBox.warning(self, "Attention", "Aucun signal charge.")
            return
        channels = list(signals.keys())
        try:
            t0 = float(getattr(self._viewer, "start_s", 0.0) or 0.0)
            dur = float(getattr(self._viewer, "duration_s", 30.0) or 30.0)
        except Exception:
            t0, dur = 0.0, 30.0

        def _apply(cleaned: Dict[str, np.ndarray]) -> None:
            originals = {
                name: np.asarray(self._signals_data[name][0], dtype=float)
                for name in cleaned
                if name in self._signals_data
            }
            n = ctrl.apply_ica_cleaning(cleaned, originals=originals)
            for name, arr in cleaned.items():
                if name not in self._signals_data:
                    continue
                fs = self._signals_data[name][1]
                self._signals_data[name] = (np.asarray(arr, dtype=float), fs)
            self._viewer.set_signals(dict(self._signals_data))
            if hasattr(self, "_sync_cursor") and self._sync_cursor is not None:
                self._sync_cursor.set_signals(dict(self._signals_data))
            self._status.showMessage(
                f"Nettoyage ICA applique ({n} canaux, memoire) — fichier non modifie.",
                8000,
            )

        def _restore() -> None:
            restored = ctrl.restore_ica_backup()
            for name, arr in restored.items():
                if name in self._signals_data:
                    fs = self._signals_data[name][1]
                    self._signals_data[name] = (arr, fs)
            for ch, sig in ctrl.build_viewer_signals().items():
                if ch in self._signals_data and ch not in restored:
                    self._signals_data[ch] = sig
            self._viewer.set_signals(dict(self._signals_data))
            if hasattr(self, "_sync_cursor") and self._sync_cursor is not None:
                self._sync_cursor.set_signals(dict(self._signals_data))
            self._status.showMessage(
                f"Signaux restaures apres ICA ({len(restored)} canaux).",
                5000,
            )

        from .dialogs.ica_cleaning_dialog import IcaCleaningDialog
        filters_on = True
        try:
            filters_on = bool(getattr(self._viewer, "global_filter_enabled", True))
        except Exception:
            filters_on = bool(getattr(ctrl, "filter_enabled", True))
        IcaCleaningDialog(
            self,
            signals,
            channels,
            channel_types=dict(self._channel_types),
            sfreq=float(ctrl.sfreq),
            preview_t0=t0,
            preview_duration=dur,
            has_active_cleaning=ctrl.has_ica_cleaning(),
            filters_on=filters_on,
            ocular_cleaning_active=ctrl.has_ocular_cleaning(),
            on_apply=_apply,
            on_restore=_restore,
        ).exec()

    def _refresh_signals_from_controller(self) -> None:
        """Rebuild viewer signal cache from controller derivations."""
        ctrl = self._app_controller
        if ctrl is None:
            return
        signals = ctrl.build_viewer_signals()
        if not signals and ctrl.derivations:
            signals = {ch: (arr, ctrl.sfreq) for ch, arr in ctrl.derivations.items()}
        if not signals:
            return
        # Preserve display names already in the viewer when possible
        if self._signals_data and set(self._signals_data.keys()) == set(signals.keys()):
            pass
        elif self._signals_data:
            # Keep current key set if derivations use original names
            merged: Dict[str, Signal] = dict(self._signals_data)
            for ch, sig in signals.items():
                if ch in merged:
                    merged[ch] = sig
            signals = merged
        self._signals_data = dict(signals)
        self._viewer.set_signals(signals)
        if hasattr(self, "_sync_cursor") and self._sync_cursor is not None:
            self._sync_cursor.set_signals(signals)

    def _menu_microstates_analysis(self) -> None:
        data = self._native_stage_signals()
        if data is None:
            return
        signals, channels, _scoring_df, _epoch = data
        from .dialogs.microstates_dialog import MicrostatesDialog
        MicrostatesDialog(self, signals, channels).exec()

    def _menu_sleep_periods(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return
        from .dialogs.sleep_periods_dialog import SleepPeriodsDialog
        SleepPeriodsDialog(self, ctrl).exec()

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "A propos de CESA",
            "<h3>CESA (Complex EEG Studio Analysis)</h3>"
            "<p>Version 0.0beta2.0</p>"
            "<p>Auteur: Come Barmoy<br>"
            "Unite Neuropsychologie du Stress - IRBA</p>"
            "<p>Licence: MIT</p>",
        )

    def _open_docs(self) -> None:
        import webbrowser
        webbrowser.open("https://github.com/cbarmoy")

    def _show_feature_search(self) -> None:
        from .dialogs.feature_search_dialog import FeatureSearchDialog

        FeatureSearchDialog(self).exec()

    # ------------------------------------------------------------------
    # Navigation : Aller au temps
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_goto_time(raw: str) -> float:
        """Parse ``HH:MM:SS``, ``MM:SS`` or plain seconds into seconds."""
        txt = (raw or "").strip().replace(",", ".")
        if not txt:
            raise ValueError("empty")
        if ":" in txt:
            parts = txt.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("invalid time")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600.0 + int(m) * 60.0 + float(s)
            m, s = parts
            return int(m) * 60.0 + float(s)
        return float(txt)

    def _menu_goto_time(self) -> None:
        total = float(getattr(self._viewer, "_total_duration_s", 0.0) or 0.0)
        cur = float(self._viewer.start_s)
        hint = ""
        if total > 0:
            hint = (
                f"\nEnregistrement: 0 – {total:.0f} s "
                f"(soit {int(total // 3600):02d}:{int((total % 3600) // 60):02d}:"
                f"{int(total % 60):02d})"
            )
        raw, ok = QtWidgets.QInputDialog.getText(
            self,
            "Aller au temps",
            "Position (HH:MM:SS, MM:SS ou secondes) :" + hint,
            text=f"{cur:.0f}",
        )
        if not ok:
            return
        try:
            t = self._parse_goto_time(raw)
        except (ValueError, TypeError):
            QtWidgets.QMessageBox.warning(
                self, "Aller au temps",
                "Format invalide. Utilisez HH:MM:SS, MM:SS ou un nombre de secondes.",
            )
            return
        if t < 0:
            t = 0.0
        dur = float(self._viewer.duration_s) or 30.0
        if total > 0:
            t = min(t, max(0.0, total - dur))
        self.set_time_window(t, dur)
        if self._on_navigate is not None:
            try:
                self._on_navigate(t)
            except Exception:
                logger.debug("on_navigate failed", exc_info=True)
        self._status.showMessage(f"Position: {t:.1f} s", 3000)

    # ------------------------------------------------------------------
    # Aide : raccourcis clavier
    # ------------------------------------------------------------------

    def _show_shortcuts(self) -> None:
        html = """
        <h3>Raccourcis clavier</h3>
        <table cellpadding='4'>
        <tr><th align='left'>Navigation</th><th></th></tr>
        <tr><td><b>&larr; / &rarr;</b></td><td>Epoque precedente / suivante</td></tr>
        <tr><td><b>Q / D</b></td><td>Reculer / avancer d'une fenetre</td></tr>
        <tr><td><b>Z / S</b></td><td>Zoom temporel +/-</td></tr>
        <tr><td><b>Ctrl+G</b></td><td>Aller a un temps precis</td></tr>
        <tr><td><b>N / P</b></td><td>Evenement suivant / precedent</td></tr>
        <tr><td><b>T</b></td><td>Prochain changement de stade</td></tr>
        <tr><td><b>[ / ]</b></td><td>Debut / fin de la fenetre d'analyse</td></tr>
        <tr><th align='left'>Scoring manuel</th><th></th></tr>
        <tr><td><b>0</b></td><td>Eveil (W)</td></tr>
        <tr><td><b>1 / 2 / 3</b></td><td>N1 / N2 / N3</td></tr>
        <tr><td><b>4</b></td><td>REM (R)</td></tr>
        <tr><td><b>5</b></td><td>Inconnu (U)</td></tr>
        <tr><td><b>Ctrl+Shift+A</b></td><td>Basculer l'auto-avance</td></tr>
        <tr><th align='left'>Analyse</th><th></th></tr>
        <tr><td><b>F</b></td><td>FFT scoring (epoque ciblee)</td></tr>
        <tr><td><b>Ctrl+N</b></td><td>Normaliser les amplitudes</td></tr>
        <tr><td><b>Ctrl+B</b></td><td>Automatisation en lot</td></tr>
        <tr><td><b>Ctrl+Shift+N</b></td><td>Nouvelle annotation</td></tr>
        <tr><th align='left'>Panneaux</th><th></th></tr>
        <tr><td><b>F1</b></td><td>Cette aide</td></tr>
        <tr><td><b>Ctrl+K</b></td><td>Rechercher une fonctionnalite</td></tr>
        <tr><td><b>F8 / F9</b></td><td>Inspection / Dashboard</td></tr>
        <tr><td><b>Ctrl+Shift+P</b></td><td>Rapport PSG (graphiques)</td></tr>
        <tr><td><b>Ctrl+O / Ctrl+Q</b></td><td>Ouvrir / Quitter</td></tr>
        </table>
        """
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Raccourcis clavier")
        box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        box.setText(html)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        box.exec()

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        mb = self.menuBar()
        has_ctrl = self._app_controller is not None

        def _add(
            menu: QtWidgets.QMenu,
            text: str,
            slot,
            *,
            shortcut: Optional[str] = None,
            enabled: bool = True,
            checkable: bool = False,
        ) -> QtGui.QAction:
            action = menu.addAction(text)
            if shortcut:
                action.setShortcut(shortcut)
            action.setEnabled(enabled)
            if checkable:
                action.setCheckable(True)
                action.toggled.connect(slot)
            else:
                action.triggered.connect(slot)
            return action

        # ---- Fichier ----
        file_m = mb.addMenu("&Fichier")
        _add(file_m, "Ouvrir un enregistrement...", self._menu_open_recording,
             shortcut="Ctrl+O", enabled=has_ctrl)
        file_m.addSeparator()
        exp_sub = file_m.addMenu("Exporter")
        _add(exp_sub, "Rapport HTML...", self._export_html_report)
        _add(exp_sub, "Rapport PDF...", self._export_pdf_report)
        _add(exp_sub, "Annotations JSON...", self._save_annotations)
        exp_sub.addSeparator()
        _add(exp_sub, "Segment EDF (plage temporelle)...",
             self._menu_export_edf_segment, enabled=has_ctrl)
        file_m.addSeparator()
        _add(file_m, "Quitter", self.close, shortcut="Ctrl+Q")

        # ---- Affichage (theme, trace, panneaux) ----
        view = mb.addMenu("&Affichage")
        _add(view, "Theme sombre", lambda: self.set_theme("dark"))
        _add(view, "Theme clair", lambda: self.set_theme("light"))
        view.addSeparator()
        _add(view, "Afficher signal brut",
             self._viewer.set_show_raw_underlay, checkable=True)
        self._normalize_action = _add(
            view, "Normaliser les amplitudes (Ctrl+N)",
            self._on_normalize_menu, checkable=True,
        )
        view.addSeparator()
        _add(view, "Aller au temps...", self._menu_goto_time, shortcut="Ctrl+G")
        view.addSeparator()
        panels = view.addMenu("Panneaux")
        _add(panels, "Inspection", self._inspection.setVisible,
             shortcut="F8", checkable=True)
        dash_action = panels.addAction("Dashboard")
        dash_action.setCheckable(True)
        dash_action.setShortcut("F9")
        dash_action.toggled.connect(lambda v: (
            self._dashboard.setVisible(v),
            self.update_dashboard() if v else None,
        ))
        panels.addSeparator()
        _add(panels, "Scaling", self._scaling_panel.setVisible, checkable=True)
        _add(panels, "Layout", self._layout_panel.setVisible, checkable=True)
        _add(panels, "Spectrogramme", self._spectro_panel.setVisible, checkable=True)

        # ---- Filtres ----
        filt_m = mb.addMenu("Filt&res")
        _add(filt_m, "Configuration des filtres...", self._invoke_filter_config_menu,
             shortcut="Ctrl+Alt+F",
             enabled=self._on_open_filter_config is not None or has_ctrl)
        filt_m.addSeparator()
        _add(filt_m, "Nettoyage oculaire...", self._menu_ocular_cleaning,
             enabled=has_ctrl)
        _add(filt_m, "Nettoyage ICA...", self._menu_ica_cleaning, enabled=has_ctrl)

        # ---- Sommeil : scoring, fenetre, rapports, evenements, annotations ----
        sleep_m = mb.addMenu("&Sommeil")
        _add(sleep_m, "Scoring automatique...", self._menu_auto_scoring,
             enabled=has_ctrl or self._on_request_auto_scoring is not None)
        _add(sleep_m, "Editeur de scoring manuel...", self._invoke_manual_editor_menu,
             enabled=has_ctrl or self._on_open_manual_scoring_editor is not None)
        import_sub = sleep_m.addMenu("Importer scoring")
        _add(import_sub, "Depuis fichier (CSV/Excel/JSON)...",
             self._menu_import_scoring_file, enabled=has_ctrl)
        _add(import_sub, "Depuis annotations EDF+...",
             self._menu_import_scoring_edf, enabled=has_ctrl)
        _add(sleep_m, "Sauvegarder scoring (CSV/JSON)...",
             self._menu_save_scoring, enabled=has_ctrl)
        sleep_m.addSeparator()
        self._act_auto_advance_manual = sleep_m.addAction(
            "Auto-avancer après scoring manuel"
        )
        self._act_auto_advance_manual.setCheckable(True)
        self._act_auto_advance_manual.setEnabled(
            self._on_set_manual_auto_advance is not None
        )
        if self._on_get_manual_auto_advance is not None:
            try:
                self._act_auto_advance_manual.setChecked(
                    bool(self._on_get_manual_auto_advance())
                )
            except Exception:
                self._act_auto_advance_manual.setChecked(False)
        self._act_auto_advance_manual.toggled.connect(
            lambda checked: (
                self._on_set_manual_auto_advance(checked)
                if self._on_set_manual_auto_advance else None
            )
        )
        self._act_fft_scoring = _add(
            sleep_m, "FFT scoring (époque ciblée)",
            self._toggle_fft_scoring_view, shortcut="F",
        )
        self._act_fft_scoring.triggered.connect(
            lambda: logger.info("[FFT-SHORTCUT] Qt menu action FFT scoring triggered")
        )
        self._update_fft_scoring_menu_enabled()
        sleep_m.addSeparator()
        night_m = sleep_m.addMenu("Fenetre d'analyse")
        _add(night_m, "Definir...", self._menu_set_night_window, enabled=has_ctrl)
        _add(night_m, "Effacer", self._menu_clear_night_window, enabled=has_ctrl)
        reports_m = sleep_m.addMenu("Rapports PSG")
        _add(reports_m, "Exporter CSV...", self._menu_export_psg_csv, enabled=has_ctrl)
        _add(reports_m, "Graphiques...", self._menu_show_psg_report,
             shortcut="Ctrl+Shift+P", enabled=has_ctrl)
        _add(sleep_m, "Comparer auto vs manuel...",
             self._menu_compare_scoring, enabled=has_ctrl)
        _add(sleep_m, "Comparer des configurations de scoring...",
             self._menu_sweep_scoring, enabled=has_ctrl)
        _add(sleep_m, "Informations scoring", self._menu_scoring_info, enabled=has_ctrl)
        sleep_m.addSeparator()
        events_m = sleep_m.addMenu("Evenements")
        _add(events_m, "Detecter micro-architecture (spindles/SO/REM)...",
             self._menu_micro_events, enabled=has_ctrl)
        events_m.addSeparator()
        _add(events_m, "Detecter les arousals...",
             lambda: self._menu_detect_events("arousals"), enabled=has_ctrl)
        _add(events_m, "Detecter apnees / hypopnees...",
             lambda: self._menu_detect_events("apneas"), enabled=has_ctrl)
        _add(events_m, "Detecter desaturations SpO2...",
             lambda: self._menu_detect_events("desaturations"), enabled=has_ctrl)
        ann = sleep_m.addMenu("Annotations")
        _add(ann, "Nouvelle annotation...", self._create_annotation_at_cursor,
             shortcut="Ctrl+Shift+N")
        ann.addSeparator()
        _add(ann, "Sauvegarder annotations...", self._save_annotations)
        _add(ann, "Charger annotations...", self._load_annotations)
        ann.addSeparator()
        _add(ann, "Effacer toutes les annotations", self._clear_annotations)
        ann.addSeparator()
        _add(ann, "Marqueurs temporels...", self._menu_markers, enabled=has_ctrl)

        # ---- Analyse ----
        analyse = mb.addMenu("Anal&yse")
        spectral_m = analyse.addMenu("Spectrale")
        _add(spectral_m, "Analyse spectrale (PSD Welch)...",
             self._menu_spectral_analysis, enabled=has_ctrl)
        _add(spectral_m, "PSD par stade (FFT - Analyse_spectrale)...",
             self._menu_stage_psd_fft,
             enabled=has_ctrl or self._on_open_stage_psd_fft is not None)
        _add(spectral_m, "PSD par stade (Welch) avant/apres...",
             self._menu_stage_psd_before_after,
             enabled=has_ctrl or self._on_open_stage_psd_before_after is not None)
        _add(spectral_m, "SpecParam (FOOOF / 1/f)...",
             self._menu_specparam, enabled=has_ctrl)
        _add(spectral_m, "Spectrogramme ondelettes (avant/apres)...",
             self._menu_wavelet_spectrogram,
             enabled=has_ctrl or self._on_open_wavelet_spectrogram is not None)
        quality_m = analyse.addMenu("Qualite")
        _add(quality_m, "QC nuit...", self._menu_night_qc, enabled=has_ctrl)
        _add(quality_m, "Detection d'artefacts...",
             self._menu_artifact_detection, enabled=has_ctrl)
        signals_m = analyse.addMenu("Signaux")
        _add(signals_m, "Entropie (MSE / Renormalisee)...",
             self._menu_entropy_analysis, enabled=has_ctrl)
        _add(signals_m, "Analyse temporelle (RMS, kurtosis...)...",
             self._menu_temporal_analysis, enabled=has_ctrl)
        _add(signals_m, "Connectivite (coherence, correlation, PLV)...",
             self._menu_connectivity_analysis, enabled=has_ctrl)
        _add(signals_m, "Micro-etats (K-means)...",
             self._menu_microstates_analysis, enabled=has_ctrl)
        _add(signals_m, "Periodes de sommeil (SleepEEGpy)...",
             self._menu_sleep_periods, enabled=has_ctrl)
        analyse.addSeparator()
        batch_menu = analyse.addMenu("Batch / Multi-sujets")
        _add(batch_menu, "Automatisation en lot (FFT / Spaghetti / PSG)...",
             self._menu_batch_fft_automation, shortcut="Ctrl+B")
        _add(batch_menu, "Analyse Groupe (MSE/REN/HRV)...",
             self._menu_group_analysis)
        ml = analyse.addMenu("ML/DL")
        ml_info = ml.addAction("Aucun backend charge")
        ml_info.setEnabled(False)
        self._ml_menu = ml

        # ---- Aide ----
        aide = mb.addMenu("A&ide")
        act_search = _add(
            aide, "Rechercher une fonctionnalite...",
            self._show_feature_search, shortcut="Ctrl+K",
        )
        act_search.setObjectName("featureSearch")
        aide.addSeparator()
        _add(aide, "Raccourcis clavier", self._show_shortcuts, shortcut="F1")
        aide.addSeparator()
        _add(aide, "Documentation en ligne...", self._open_docs)
        aide.addSeparator()
        _add(aide, "A propos de CESA...", self._show_about)

    # ------------------------------------------------------------------
    # Annotation actions
    # ------------------------------------------------------------------

    def _create_annotation_at_cursor(self) -> None:
        t = self._sync_cursor.time_s or self._viewer.start_s
        dlg = AnnotationDialog(self, onset_s=t, duration_s=self._epoch_len)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.annotation:
            self._annotation_store.add(dlg.annotation)
            self._viewer._refresh_view()
            self._refresh_annotation_ui()

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
            self._refresh_annotation_ui()
            self._status.showMessage(f"{count} annotations chargees", 3000)

    def _clear_annotations(self) -> None:
        self._annotation_overlay.clear()
        self._annotation_store.clear()
        self._refresh_annotation_ui()
        self._viewer._refresh_view()

    def _menu_markers(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return
        from .dialogs.markers_dialog import MarkersDialog
        dlg = MarkersDialog(self, ctrl)
        dlg.exec()

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

    def _menu_export_edf_segment(self) -> None:
        ctrl = self._app_controller
        if ctrl is None or ctrl.raw is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Chargez d'abord un enregistrement.")
            return
        from .dialogs.edf_segment_export_dialog import EdfSegmentExportDialog
        EdfSegmentExportDialog(self, ctrl).exec()

    def _export_pdf_report(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export rapport PDF", "", "PDF (*.pdf)",
        )
        if path:
            builder = ReportBuilder()
            builder.set_title("CESA - Rapport PSG")
            self._populate_report_builder(builder)
            builder.save_pdf(path)
            self._status.showMessage(f"Rapport exporte: {path}", 3000)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        t = self._theme
        self._nav.set_stage_theme(t)
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
            QLabel#stageBadge {{
                border: none;
            }}
        """)
        self._refresh_stage_indicator()

    # ------------------------------------------------------------------
    # Keyboard (QShortcut WindowShortcut + dispatch pour la barre de nav)
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 (Qt API)
        """Fallback clavier robuste si QShortcut est intercepté par l'environnement."""
        try:
            key = int(event.key())
            mod = _keyboard_modifiers_int(event.modifiers())
            handled = self._dispatch_key_shortcuts_ints(key, mod)
            if handled:
                event.accept()
                return
        except Exception:
            logger.debug("keyPressEvent shortcut dispatch failed", exc_info=True)
        super().keyPressEvent(event)

    def _dispatch_key_shortcuts_ints(self, key: int, mod_int: int) -> bool:
        """Return True if handled (plot deferred path + keyPressEvent)."""
        mod = QtCore.Qt.KeyboardModifiers(mod_int)
        ctrl = bool(mod & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift = bool(mod & QtCore.Qt.KeyboardModifier.ShiftModifier)
        alt = bool(mod & QtCore.Qt.KeyboardModifier.AltModifier)
        meta = bool(mod & QtCore.Qt.KeyboardModifier.MetaModifier)

        if key == QtCore.Qt.Key.Key_F1:
            self._show_shortcuts()
            return True
        if ctrl and not shift and not alt and not meta and key == QtCore.Qt.Key.Key_K:
            self._show_feature_search()
            return True
        if ctrl and not shift and not alt and not meta and key == QtCore.Qt.Key.Key_G:
            self._menu_goto_time()
            return True
        if ctrl and shift and not alt and not meta and key == QtCore.Qt.Key.Key_A:
            self._toggle_manual_auto_advance()
            return True
        if ctrl and shift and not alt and not meta and key == QtCore.Qt.Key.Key_F:
            self._toggle_fft_scoring_view()
            return True
        if ctrl and alt and not meta and key == QtCore.Qt.Key.Key_X:
            self._toggle_fft_scoring_view()
            return True

        if not _mods_block_combo_nav(mod):
            stage_by_key = {
                QtCore.Qt.Key.Key_0: "W",
                QtCore.Qt.Key.Key_1: "N1",
                QtCore.Qt.Key.Key_2: "N2",
                QtCore.Qt.Key.Key_3: "N3",
                QtCore.Qt.Key.Key_4: "R",
                QtCore.Qt.Key.Key_5: "U",
            }
            st = stage_by_key.get(key)
            if st is not None:
                self._invoke_stage_current_epoch(st)
                return True

        if (
            key == QtCore.Qt.Key.Key_N
            and mod
            & (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.ShiftModifier
            )
            == (
                QtCore.Qt.KeyboardModifier.ControlModifier
                | QtCore.Qt.KeyboardModifier.ShiftModifier
            )
        ):
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

        if not _mods_block_combo_nav(mod) and key == QtCore.Qt.Key.Key_F:
            self._toggle_fft_scoring_view()
            return True

        if not _mods_block_combo_nav(mod):
            if key == QtCore.Qt.Key.Key_BracketLeft:
                t = float(self._viewer.start_s)
                ep = float(self._epoch_len) if self._epoch_len > 0 else 30.0
                self._on_night_start_requested(int(t / ep) * ep)
                return True
            if key == QtCore.Qt.Key.Key_BracketRight:
                t = float(self._viewer.start_s)
                ep = float(self._epoch_len) if self._epoch_len > 0 else 30.0
                self._on_night_end_requested(int(t / ep) * ep)
                return True

        return bool(self._nav.handle_key_ints(key, mod))

