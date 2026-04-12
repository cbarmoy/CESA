"""Tests for advanced CESA viewer features:
annotations, smart navigation, sync cursor, ML overlay, LOD cache,
inspection panel, dashboard, and report export.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# -----------------------------------------------------------------------
# Pure-Python tests (no Qt required)
# -----------------------------------------------------------------------


class TestAnnotationModel:

    def test_create_annotation(self):
        from CESA.qt_viewer.annotations import Annotation
        ann = Annotation(onset_s=10.0, duration_s=5.0, ann_type="artifact", text="EMG burst")
        assert ann.onset_s == 10.0
        assert ann.duration_s == 5.0
        assert ann.ann_type == "artifact"
        assert ann.text == "EMG burst"
        assert len(ann.id) == 12

    def test_effective_color(self):
        from CESA.qt_viewer.annotations import Annotation
        ann = Annotation(ann_type="artifact")
        assert ann.effective_color() == "#F38BA8"

    def test_effective_color_override(self):
        from CESA.qt_viewer.annotations import Annotation
        ann = Annotation(ann_type="artifact", color="#FFFFFF")
        assert ann.effective_color() == "#FFFFFF"

    def test_to_dict_from_dict_roundtrip(self):
        from CESA.qt_viewer.annotations import Annotation
        ann = Annotation(onset_s=5.0, duration_s=2.0, ann_type="spindle", text="test")
        d = ann.to_dict()
        restored = Annotation.from_dict(d)
        assert restored.onset_s == ann.onset_s
        assert restored.ann_type == ann.ann_type
        assert restored.text == ann.text


class TestAnnotationStore:

    def test_add_and_remove(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        ann = Annotation(onset_s=0, duration_s=1)
        aid = store.add(ann)
        assert len(store) == 1
        store.remove(aid)
        assert len(store) == 0

    def test_in_range(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        store.add(Annotation(onset_s=10, duration_s=5))
        store.add(Annotation(onset_s=50, duration_s=5))
        store.add(Annotation(onset_s=100, duration_s=5))
        visible = store.in_range(8, 20)
        assert len(visible) == 1
        assert visible[0].onset_s == 10

    def test_by_type(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        store.add(Annotation(onset_s=0, duration_s=1, ann_type="artifact"))
        store.add(Annotation(onset_s=5, duration_s=1, ann_type="spindle"))
        store.add(Annotation(onset_s=10, duration_s=1, ann_type="artifact"))
        artifacts = store.by_type("artifact")
        assert len(artifacts) == 2

    def test_next_prev_of_type(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        store.add(Annotation(onset_s=10, duration_s=1, ann_type="artifact"))
        store.add(Annotation(onset_s=50, duration_s=1, ann_type="artifact"))
        store.add(Annotation(onset_s=90, duration_s=1, ann_type="artifact"))
        nxt = store.next_of_type("artifact", 20)
        assert nxt is not None and nxt.onset_s == 50
        prev = store.prev_of_type("artifact", 80)
        assert prev is not None and prev.onset_s == 50

    def test_update(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        ann = Annotation(onset_s=0, duration_s=1, text="old")
        store.add(ann)
        updated = store.update(ann.id, text="new")
        assert updated is not None
        assert updated.text == "new"

    def test_save_load_json(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        store.add(Annotation(onset_s=5, duration_s=2, ann_type="arousal", text="test1"))
        store.add(Annotation(onset_s=15, duration_s=3, ann_type="artifact", text="test2"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            store.save_json(path)
            store2 = AnnotationStore()
            count = store2.load_json(path)
            assert count == 2
            assert len(store2) == 2
            all_ann = store2.all()
            assert all_ann[0].onset_s == 5
            assert all_ann[1].onset_s == 15
        finally:
            os.unlink(path)

    def test_to_event_list(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        store.add(Annotation(onset_s=1, duration_s=2, ann_type="note", text="hello"))
        events = store.to_event_list()
        assert len(events) == 1
        assert events[0]["onset"] == 1
        assert events[0]["type"] == "note"

    def test_listener_notification(self):
        from CESA.qt_viewer.annotations import Annotation, AnnotationStore
        store = AnnotationStore()
        calls = []
        store.add_listener(lambda: calls.append(1))
        store.add(Annotation(onset_s=0, duration_s=1))
        assert len(calls) == 1
        store.clear()
        assert len(calls) == 2


class TestSmartNavigator:

    def test_next_event(self):
        from CESA.qt_viewer.smart_navigation import SmartNavigator
        nav = SmartNavigator()
        nav.set_events([
            {"onset": 10, "type": "arousal"},
            {"onset": 50, "type": "apnea"},
            {"onset": 100, "type": "arousal"},
        ])
        nav.set_current_time(5)
        t = nav.next_event()
        assert t == 10

    def test_next_event_with_type(self):
        from CESA.qt_viewer.smart_navigation import SmartNavigator
        nav = SmartNavigator()
        nav.set_events([
            {"onset": 10, "type": "arousal"},
            {"onset": 50, "type": "apnea"},
            {"onset": 100, "type": "arousal"},
        ])
        nav.set_current_time(15)
        t = nav.next_event("apnea")
        assert t == 50

    def test_prev_event(self):
        from CESA.qt_viewer.smart_navigation import SmartNavigator
        nav = SmartNavigator()
        nav.set_events([
            {"onset": 10, "type": "arousal"},
            {"onset": 50, "type": "apnea"},
        ])
        nav.set_current_time(60)
        t = nav.prev_event()
        assert t == 50

    def test_next_stage_change(self):
        from CESA.qt_viewer.smart_navigation import SmartNavigator
        nav = SmartNavigator()
        nav.set_hypnogram(["W", "W", "N1", "N1", "N2"], 30.0)
        nav.set_current_time(10)
        t = nav.next_stage_change()
        assert t == 60.0  # index 2

    def test_prev_stage_change(self):
        from CESA.qt_viewer.smart_navigation import SmartNavigator
        nav = SmartNavigator()
        nav.set_hypnogram(["W", "W", "N1", "N1", "N2"], 30.0)
        nav.set_current_time(130)
        t = nav.prev_stage_change()
        assert t == 120.0  # index 4

    def test_jump_to_stage(self):
        from CESA.qt_viewer.smart_navigation import SmartNavigator
        nav = SmartNavigator()
        nav.set_hypnogram(["W", "N1", "N2", "N3", "R"], 30.0)
        nav.set_current_time(0)
        t = nav.jump_to_stage("N3")
        assert t == 90.0

    def test_search(self):
        from CESA.qt_viewer.smart_navigation import SmartNavigator
        nav = SmartNavigator()
        nav.set_events([
            {"onset": 10, "type": "arousal", "label": "arousal A"},
            {"onset": 50, "type": "rem_burst", "label": "REM B"},
        ])
        nav.set_hypnogram(["W", "N1", "REM"], 30.0)
        results = nav.search("REM")
        assert len(results) >= 2  # event + hypnogram stage


class TestLODCache:

    def test_set_and_get_channel(self):
        from CESA.qt_viewer.lod_cache import LODCache
        cache = LODCache()
        data = np.random.randn(100000)
        cache.set_channel("C3", data, 256.0)
        assert cache.has_channel("C3")
        assert not cache.has_channel("C4")

    def test_get_lod_reduces(self):
        from CESA.qt_viewer.lod_cache import LODCache
        cache = LODCache()
        data = np.random.randn(100000)
        cache.set_channel("C3", data, 256.0)
        lod = cache.get_lod("C3", target_points=2000)
        assert lod is not None
        assert len(lod) <= 4100  # 2000 * 2 interleaved min/max + some slack

    def test_get_segment(self):
        from CESA.qt_viewer.lod_cache import LODCache
        cache = LODCache()
        data = np.random.randn(256000)  # ~1000 seconds at 256Hz
        cache.set_channel("C3", data, 256.0)
        result = cache.get_segment("C3", 10.0, 30.0, target_points=2000)
        assert result is not None
        times, vals = result
        assert len(times) == len(vals)
        assert len(times) > 0

    def test_memory_mb(self):
        from CESA.qt_viewer.lod_cache import LODCache
        cache = LODCache()
        data = np.random.randn(100000)
        cache.set_channel("C3", data, 256.0)
        assert cache.memory_mb > 0

    def test_clear(self):
        from CESA.qt_viewer.lod_cache import LODCache
        cache = LODCache()
        cache.set_channel("C3", np.zeros(1000), 256.0)
        cache.clear()
        assert not cache.has_channel("C3")


class TestReportBuilder:

    def test_render_html(self):
        from CESA.qt_viewer.report_export import ReportBuilder
        builder = ReportBuilder()
        builder.set_title("Test Report")
        builder.set_recording_info({"Patient": "Test", "Date": "2026-01-01"})
        builder.set_sleep_metrics({"TST (min)": 420, "SE (%)": 85.3})
        builder.set_stage_distribution({"W": 10, "N1": 5, "N2": 50, "N3": 20, "REM": 15})
        builder.set_warnings(["Low SNR on EMG1"])
        html = builder.render_html()
        assert "Test Report" in html
        assert "TST (min)" in html
        assert "Low SNR" in html
        assert "<!DOCTYPE html>" in html

    def test_save_html(self):
        from CESA.qt_viewer.report_export import ReportBuilder
        builder = ReportBuilder()
        builder.set_title("Save Test")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            p = builder.save_html(path)
            assert p.exists()
            content = p.read_text(encoding="utf-8")
            assert "Save Test" in content
        finally:
            os.unlink(path)

    def test_set_hypnogram_figure(self):
        pytest.importorskip("matplotlib")
        from CESA.qt_viewer.report_export import ReportBuilder
        builder = ReportBuilder()
        builder.set_hypnogram(["W", "N1", "N2", "N3", "R"] * 20, 30.0)
        builder.add_hypnogram_figure()
        html = builder.render_html()
        assert "data:image/png;base64," in html

    def test_ml_results_in_report(self):
        from CESA.qt_viewer.report_export import ReportBuilder
        builder = ReportBuilder()
        builder.set_ml_results([
            {"backend": "rules", "accuracy": 0.78, "kappa": 0.65, "macro_f1": 0.72},
            {"backend": "ml_hmm", "accuracy": 0.85, "kappa": 0.80, "macro_f1": 0.82},
        ])
        html = builder.render_html()
        assert "rules" in html
        assert "ml_hmm" in html

    def test_annotations_in_report(self):
        from CESA.qt_viewer.report_export import ReportBuilder
        builder = ReportBuilder()
        builder.set_annotations([
            {"onset": 10, "duration": 5, "type": "artifact", "label": "EMG"},
            {"onset": 120, "duration": 3, "type": "arousal", "label": "Spontaneous"},
        ])
        html = builder.render_html()
        assert "EMG" in html
        assert "Spontaneous" in html


# -----------------------------------------------------------------------
# Qt-dependent tests
# -----------------------------------------------------------------------

try:
    from PySide6.QtWidgets import QApplication
    import pyqtgraph  # noqa: F401
    _app = QApplication.instance() or QApplication(sys.argv)
    HAS_QT = True
except ImportError:
    HAS_QT = False

qt_required = pytest.mark.skipif(not HAS_QT, reason="PySide6/pyqtgraph not installed")


@qt_required
class TestSyncCursor:

    def test_register_and_move(self):
        import pyqtgraph as pg
        from CESA.qt_viewer.sync_cursor import SyncCursorManager
        mgr = SyncCursorManager()
        pw = pg.PlotWidget()
        plot = pw.getPlotItem()
        line = mgr.register_panel(plot)
        assert line is not None
        mgr.update_position(15.0)
        assert mgr.time_s == pytest.approx(15.0)

    def test_epoch_index(self):
        from CESA.qt_viewer.sync_cursor import SyncCursorManager
        mgr = SyncCursorManager()
        mgr.set_epoch_length(30.0)
        mgr.update_position(65.0)
        assert mgr.epoch_index() == 2

    def test_hover_label_update(self):
        from PySide6.QtWidgets import QLabel
        from CESA.qt_viewer.sync_cursor import SyncCursorManager
        mgr = SyncCursorManager()
        label = QLabel()
        mgr.set_hover_label(label)
        mgr.set_signals({"C3": (np.arange(100, dtype=float), 10.0)})
        mgr.update_position(5.0)
        assert "C3" in label.text()

    def test_hide(self):
        from PySide6.QtWidgets import QLabel
        from CESA.qt_viewer.sync_cursor import SyncCursorManager
        mgr = SyncCursorManager()
        label = QLabel()
        mgr.set_hover_label(label)
        mgr.update_position(1.0)
        mgr.hide()
        assert label.text() == ""


@qt_required
class TestAnnotationOverlay:

    def test_overlay_update(self):
        import pyqtgraph as pg
        from CESA.qt_viewer.annotations import (
            Annotation, AnnotationOverlay, AnnotationStore,
        )
        store = AnnotationStore()
        store.add(Annotation(onset_s=10, duration_s=5, ann_type="artifact"))
        pw = pg.PlotWidget()
        plot = pw.getPlotItem()
        overlay = AnnotationOverlay(plot, store)
        overlay.update_visible(0, 30)
        assert len(overlay._items) == 1

    def test_overlay_clears(self):
        import pyqtgraph as pg
        from CESA.qt_viewer.annotations import (
            Annotation, AnnotationOverlay, AnnotationStore,
        )
        store = AnnotationStore()
        store.add(Annotation(onset_s=10, duration_s=5))
        pw = pg.PlotWidget()
        overlay = AnnotationOverlay(pw.getPlotItem(), store)
        overlay.update_visible(0, 30)
        overlay.clear()
        assert len(overlay._items) == 0


@qt_required
class TestMLOverlay:

    def test_set_predictions(self):
        import pyqtgraph as pg
        from CESA.qt_viewer.ml_overlay import MLOverlayManager
        pw = pg.PlotWidget()
        overlay = MLOverlayManager(pw.getPlotItem())
        overlay.set_epoch_length(30.0)
        overlay.set_backend_predictions("rules", ["W", "N1", "N2", "N3", "R"])
        assert "rules" in overlay.backend_names

    def test_toggle_backend(self):
        import pyqtgraph as pg
        from CESA.qt_viewer.ml_overlay import MLOverlayManager
        pw = pg.PlotWidget()
        overlay = MLOverlayManager(pw.getPlotItem())
        overlay.set_epoch_length(30.0)
        overlay.set_backend_predictions("ml", ["W"] * 10, [0.9] * 10)
        overlay.update_visible(0, 300, 150.0)
        overlay.toggle_backend("ml", False)
        # Check it's hidden
        layer = overlay._layers.get("ml")
        assert layer is not None
        assert layer._visible is False

    def test_clear(self):
        import pyqtgraph as pg
        from CESA.qt_viewer.ml_overlay import MLOverlayManager
        pw = pg.PlotWidget()
        overlay = MLOverlayManager(pw.getPlotItem())
        overlay.set_backend_predictions("test", ["W", "N1"])
        overlay.update_visible(0, 60, 150.0)
        overlay.clear()
        assert len(overlay.backend_names) == 0


@qt_required
class TestSmartNavWidget:

    def test_creation(self):
        from CESA.qt_viewer.smart_navigation import SmartNavWidget
        w = SmartNavWidget()
        assert w is not None

    def test_navigator_property(self):
        from CESA.qt_viewer.smart_navigation import SmartNavWidget
        w = SmartNavWidget()
        w.navigator.set_hypnogram(["W", "N1", "N2"], 30.0)
        w.navigator.set_current_time(0)
        t = w.navigator.next_stage_change()
        assert t == 30.0


@qt_required
class TestInspectionPanel:

    def test_creation(self):
        from CESA.qt_viewer.inspection_panel import InspectionPanel
        panel = InspectionPanel()
        assert panel is not None

    def test_set_features(self):
        from CESA.qt_viewer.inspection_panel import InspectionPanel
        panel = InspectionPanel()
        panel.set_features([
            {"delta_power": 0.5, "theta_power": 0.2},
            {"delta_power": 0.3, "theta_power": 0.4},
        ])
        panel.set_epoch(0)
        assert panel._features_table.rowCount() == 2

    def test_set_ml_probabilities(self):
        from CESA.qt_viewer.inspection_panel import InspectionPanel
        panel = InspectionPanel()
        panel.set_ml_probabilities([
            {"W": 0.1, "N1": 0.6, "N2": 0.2, "N3": 0.05, "R": 0.05},
        ])
        panel.set_epoch(0)
        assert panel._ml_table.rowCount() == 5


@qt_required
class TestDashboardPanel:

    def test_creation(self):
        from CESA.qt_viewer.dashboard_panel import DashboardPanel
        panel = DashboardPanel()
        assert panel is not None

    def test_set_sleep_metrics(self):
        from CESA.qt_viewer.dashboard_panel import DashboardPanel
        panel = DashboardPanel()
        panel.set_sleep_metrics(tst_min=420, se_pct=85.3, sol_min=12.5)
        assert "420.0" in panel._tst_label.text()
        assert "85.3" in panel._se_label.text()

    def test_set_stage_distribution(self):
        from CESA.qt_viewer.dashboard_panel import DashboardPanel
        panel = DashboardPanel()
        panel.set_stage_distribution({"W": 10, "N1": 5, "N2": 50, "N3": 20, "REM": 15})
        _, bar = panel._dist_bars["N2"]
        assert bar.value() == 50

    def test_compute_stage_distribution(self):
        from CESA.qt_viewer.dashboard_panel import DashboardPanel
        panel = DashboardPanel()
        dist = panel.compute_stage_distribution(["W"] * 10 + ["N2"] * 30 + ["R"] * 10)
        assert dist["W"] == pytest.approx(20.0)
        assert dist["N2"] == pytest.approx(60.0)

    def test_set_warnings(self):
        from CESA.qt_viewer.dashboard_panel import DashboardPanel
        panel = DashboardPanel()
        panel.set_warnings(["Low SNR on EMG", "Electrode artifact"])
        assert panel._warnings_list.count() == 2


@qt_required
class TestMainWindowAdvanced:

    def _make_window(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        sfreq = 256.0
        n = int(sfreq * 300)  # 5 minutes
        signals = {
            "C3-M2": (np.random.randn(n), sfreq),
            "EOG-L": (np.random.randn(n), sfreq),
        }
        hypnogram = (["W", "W", "N1", "N2", "N2", "N3", "N3", "R", "R", "W"], 30.0)
        return EEGViewerMainWindow(
            signals=signals,
            hypnogram=hypnogram,
            scoring_annotations=[
                {"onset": 30.0, "duration": 3.0, "type": "arousal"},
                {"onset": 120.0, "duration": 10.0, "type": "apnea"},
            ],
            total_duration_s=300.0,
        )

    def test_annotation_store_exists(self):
        win = self._make_window()
        assert win.annotation_store is not None
        assert len(win.annotation_store) == 0

    def test_add_annotation(self):
        from CESA.qt_viewer.annotations import Annotation
        win = self._make_window()
        aid = win.add_annotation(Annotation(onset_s=10, duration_s=5, ann_type="artifact"))
        assert len(win.annotation_store) == 1
        win.remove_annotation(aid)
        assert len(win.annotation_store) == 0

    def test_ml_predictions(self):
        win = self._make_window()
        win.set_ml_predictions("rules", ["W"] * 10, [0.8] * 10)
        assert "rules" in win._ml_overlay.backend_names

    def test_epoch_features(self):
        win = self._make_window()
        features = [{"delta": 0.5, "theta": 0.3}] * 10
        win.set_epoch_features(features)
        win._inspection.set_epoch(0)
        assert win._inspection._features_table.rowCount() == 2

    def test_update_dashboard(self):
        win = self._make_window()
        win._dashboard.setVisible(True)
        win.update_dashboard()
        # Stage distribution should be set
        _, bar = win._dashboard._dist_bars["W"]
        assert bar.value() > 0

    def test_export_html_report(self):
        win = self._make_window()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            win.export_report_html(path)
            content = open(path, encoding="utf-8").read()
            assert "CESA" in content
            assert "Hypnogramme" in content
        finally:
            os.unlink(path)

    def test_smart_navigation_wired(self):
        win = self._make_window()
        # Smart navigator should have hypnogram
        t = win._smart_nav.navigator.next_stage_change()
        assert t is not None
        assert t == 60.0  # W->N1 transition

    def test_sync_cursor(self):
        win = self._make_window()
        win._sync_cursor.update_position(45.0)
        assert win._sync_cursor.time_s == pytest.approx(45.0)
        assert win._sync_cursor.epoch_index() == 1

    def test_keyboard_navigation_shortcuts(self):
        """Test that navigation keyboard shortcuts are wired."""
        from PySide6 import QtCore, QtGui
        win = self._make_window()
        # N key -> next event
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_N,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        win.keyPressEvent(event)
        # T key -> next stage transition
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_T,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        win.keyPressEvent(event)
        # No crash is the assertion


@qt_required
class TestViewerBridgeAdvanced:

    def test_bridge_annotation_api(self):
        from CESA.qt_viewer.annotations import Annotation
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        from CESA.qt_viewer.viewer_bridge import ViewerBridge
        win = EEGViewerMainWindow()
        bridge = ViewerBridge(win)
        aid = bridge.add_annotation(Annotation(onset_s=0, duration_s=1))
        assert len(bridge.annotation_store) == 1
        bridge.remove_annotation(aid)
        assert len(bridge.annotation_store) == 0

    def test_bridge_ml_api(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        from CESA.qt_viewer.viewer_bridge import ViewerBridge
        win = EEGViewerMainWindow()
        bridge = ViewerBridge(win)
        bridge.set_ml_predictions("test", ["W", "N1"], [0.9, 0.8])

    def test_bridge_dashboard(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        from CESA.qt_viewer.viewer_bridge import ViewerBridge
        win = EEGViewerMainWindow()
        bridge = ViewerBridge(win)
        bridge.update_dashboard()
