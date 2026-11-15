"""Smoke tests for the CESA PyQtGraph viewer package."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Force offscreen rendering for headless CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure project root is on the path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# -----------------------------------------------------------------------
# Pure-Python / NumPy tests (no Qt required)
# -----------------------------------------------------------------------

class TestDownsampler:

    def test_passthrough_short(self):
        from CESA.qt_viewer.downsampler import downsample_minmax
        data = np.array([1.0, 2.0, 3.0])
        idx, vals = downsample_minmax(data, 100)
        assert len(vals) == 3
        np.testing.assert_array_equal(vals, data)

    def test_reduces_long_signal(self):
        from CESA.qt_viewer.downsampler import downsample_minmax
        data = np.sin(np.linspace(0, 20 * np.pi, 100_000))
        idx, vals = downsample_minmax(data, 4000)
        assert len(vals) < len(data)
        assert len(vals) >= 2000

    def test_preserves_envelope(self):
        from CESA.qt_viewer.downsampler import downsample_minmax
        data = np.zeros(10_000)
        data[5000] = 100.0  # spike
        _, vals = downsample_minmax(data, 500)
        assert np.max(vals) == pytest.approx(100.0)

    def test_compute_target_points(self):
        from CESA.qt_viewer.downsampler import compute_target_points
        assert compute_target_points(1000) == 2000
        assert compute_target_points(100) == 500  # clamped to min


class TestThemes:

    def test_dark_and_light_exist(self):
        from CESA.qt_viewer.themes import DARK, LIGHT, THEMES
        assert "dark" in THEMES
        assert "light" in THEMES
        assert "background" in DARK
        assert "foreground" in LIGHT

    def test_stage_color(self):
        from CESA.qt_viewer.themes import DARK, stage_color
        assert stage_color(DARK, "W") == DARK["stage_W"]
        assert stage_color(DARK, "N3") == DARK["stage_N3"]
        # Unknown stage
        assert stage_color(DARK, "XYZ") == DARK["stage_U"]

    def test_event_color(self):
        from CESA.qt_viewer.themes import DARK, event_color
        assert event_color(DARK, "arousal") == DARK["event_arousal"]
        assert event_color(DARK, "central_apnea") == DARK["event_apnea"]
        assert event_color(DARK, "unknown_event") == DARK["event_default"]


class TestFilterMetrics:
    """Compteur dashboard filtres (sans import PySide6)."""

    def test_count_effective_filter_channels(self):
        from CESA.filter_engine import BandpassFilter, FilterPipeline
        from CESA.qt_viewer.filter_metrics import count_effective_filter_channels

        bp_on = BandpassFilter(enabled=True)
        bp_off = BandpassFilter(enabled=False)
        assert count_effective_filter_channels({}) == 0
        assert count_effective_filter_channels({"A": None}) == 0
        assert count_effective_filter_channels(
            {"A": FilterPipeline(filters=[bp_on], enabled=True)},
        ) == 1
        assert count_effective_filter_channels(
            {"A": FilterPipeline(filters=[bp_off], enabled=True)},
        ) == 0
        assert count_effective_filter_channels(
            {"A": FilterPipeline(filters=[], enabled=True)},
        ) == 0
        assert count_effective_filter_channels(
            {"A": FilterPipeline(filters=[bp_on], enabled=False)},
        ) == 0


# -----------------------------------------------------------------------
# Qt-dependent tests (skip if PySide6 / pyqtgraph unavailable)
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
class TestTimeAxisItem:

    def test_fmt_hhmmss(self):
        from CESA.qt_viewer.time_axis_item import TimeAxisItem
        axis = TimeAxisItem()
        assert axis._fmt(0) == "00:00"
        assert axis._fmt(90) == "01:30"
        assert axis._fmt(3661) == "01:01:01"

    def test_tick_strings(self):
        from CESA.qt_viewer.time_axis_item import TimeAxisItem
        axis = TimeAxisItem()
        result = axis.tickStrings([0, 60, 3600], 1.0, 60)
        assert len(result) == 3
        assert result[0] == "00:00"


@qt_required
class TestEEGViewerWidget:

    def test_creation(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        assert w is not None
        assert w.channel_names == []

    def test_set_signals(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        signals = {
            "C3": (np.random.randn(2560), 256.0),
            "C4": (np.random.randn(2560), 256.0),
        }
        w.set_signals(signals)
        assert set(w.channel_names) == {"C3", "C4"}

    def test_set_time_window(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        w.set_time_window(10.0, 30.0)
        assert w.start_s == pytest.approx(10.0)
        assert w.duration_s == pytest.approx(30.0)


@qt_required
class TestHypnogramBar:

    def test_set_hypnogram(self):
        from CESA.qt_viewer.hypnogram_bar import HypnogramBar
        bar = HypnogramBar()
        bar.set_hypnogram(["W", "N1", "N2", "N3", "R"], 30.0, 150.0)
        assert bar._labels == ["W", "N1", "N2", "N3", "R"]

    def test_set_window(self):
        from CESA.qt_viewer.hypnogram_bar import HypnogramBar
        bar = HypnogramBar()
        bar.set_hypnogram(["W"] * 10, 30.0, 300.0)
        bar.set_window(60.0, 30.0)
        assert bar._window_start == 60.0


@qt_required
class TestEventsBar:

    def test_set_events(self):
        from CESA.qt_viewer.events_bar import EventsBar
        bar = EventsBar()
        bar.set_events([
            {"onset": 10.0, "duration": 3.0, "type": "arousal"},
            {"onset": 60.0, "duration": 10.0, "type": "apnea"},
        ])
        assert len(bar._events) == 2


@qt_required
class TestNavigationBar:

    def test_creation(self):
        from CESA.qt_viewer.navigation_bar import NavigationBar
        nav = NavigationBar()
        nav.set_total_duration(28800.0)
        nav.set_current_time(120.0)
        assert nav._current_time == pytest.approx(120.0)

    def test_duration(self):
        from CESA.qt_viewer.navigation_bar import NavigationBar
        nav = NavigationBar()
        nav.set_duration(60.0)
        assert nav._duration == pytest.approx(60.0)


@qt_required
class TestMainWindow:

    def test_creation_with_signals(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        sfreq = 256.0
        n = int(sfreq * 60)
        signals = {
            "C3-M2": (np.random.randn(n), sfreq),
            "EOG-L": (np.random.randn(n), sfreq),
        }
        win = EEGViewerMainWindow(
            signals=signals,
            hypnogram=(["W", "N1"], 30.0),
            total_duration_s=60.0,
        )
        assert win is not None
        assert win.isVisible() is False  # not shown in test

    def test_set_time_window(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        win = EEGViewerMainWindow()
        win.set_time_window(30.0, 30.0)
        # No crash is the assertion


@qt_required
class TestMainWindowStudioCallbacks:
    """Smoke : callbacks scoring / filtres / auto (pont Tk) sont invoqués."""

    def test_invoke_stage_current_epoch_calls_callback(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow

        called: list[str] = []
        win = EEGViewerMainWindow(
            on_request_stage_for_current_epoch=lambda s: called.append(s),
        )
        win._invoke_stage_current_epoch("N2")
        assert called == ["N2"]

    def test_hypnogram_context_stage_reaches_callback(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow

        got: list[tuple[float, str]] = []
        win = EEGViewerMainWindow(
            on_request_stage_at_epoch_time=lambda t, s: got.append((float(t), str(s))),
        )
        win._hypno.epoch_context_stage.emit(90.0, "R")
        assert len(got) == 1
        assert got[0][0] == pytest.approx(90.0)
        assert got[0][1] == "R"

    def test_create_and_launch_forwards_auto_scoring_callback(self):
        from CESA.qt_viewer.viewer_bridge import create_and_launch

        flags: list[str] = []
        bridge = create_and_launch(
            tk_root=None,
            on_request_auto_scoring=lambda: flags.append("auto"),
        )
        bridge._win._invoke_auto_scoring_menu()
        assert flags == ["auto"]

    def test_create_and_launch_forwards_global_filter_callback(self):
        from CESA.qt_viewer.viewer_bridge import create_and_launch

        got: list[bool] = []
        bridge = create_and_launch(
            tk_root=None,
            on_global_filter_toggled=lambda b: got.append(bool(b)),
        )
        win = bridge._win
        win._nav._btn_filter.setChecked(False)
        win._on_filter_toggle()
        assert got == [False]

    def test_filter_toggle_invokes_callback(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow

        got: list[bool] = []
        win = EEGViewerMainWindow(on_global_filter_toggled=lambda b: got.append(bool(b)))
        win._nav._btn_filter.setChecked(False)
        win._on_filter_toggle()
        assert got == [False]

    def test_set_filter_pipelines_refreshes_dashboard(self):
        from CESA.filter_engine import BandpassFilter, FilterPipeline
        from CESA.qt_viewer.main_window import EEGViewerMainWindow

        sfreq = 256.0
        n = int(sfreq * 10)
        signals = {"C3": (np.random.randn(n), sfreq)}
        win = EEGViewerMainWindow(signals=signals, total_duration_s=10.0)
        assert win._dashboard._n_filters_label.text() == "0"
        pipe = FilterPipeline(filters=[BandpassFilter(enabled=True)], enabled=True)
        win.set_filter_pipelines({"C3": pipe})
        assert win._dashboard._n_filters_label.text() == "1"
        win.set_global_filter_enabled(False)
        assert win._dashboard._filter_status_label.text() == "OFF"


@qt_required
class TestViewerBridge:

    def test_bridge_api(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        from CESA.qt_viewer.viewer_bridge import ViewerBridge
        win = EEGViewerMainWindow()
        bridge = ViewerBridge(win)
        bridge.set_time_window(10.0, 30.0)
        bridge.set_global_filter_enabled(True)
        assert bridge.is_alive() is False  # window not shown

    def test_bridge_signals(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow
        from CESA.qt_viewer.viewer_bridge import ViewerBridge
        sfreq = 256.0
        n = int(sfreq * 30)
        signals = {
            "C3": (np.random.randn(n), sfreq),
        }
        win = EEGViewerMainWindow(signals=signals, total_duration_s=30.0)
        bridge = ViewerBridge(win)
        bridge.update_signals(signals)
        bridge.set_hypnogram((["W"], 30.0))
        # No crash is the assertion


@qt_required
class TestFeatureSearch:

    def test_fold_and_alias_match(self):
        from CESA.qt_viewer.dialogs.feature_search_dialog import (
            action_search_haystack, query_matches,
        )
        hay = action_search_haystack(
            "Detecter micro-architecture (spindles/SO/REM)...",
            "Sommeil > Evenements",
            "",
        )
        assert query_matches("spindle", hay)
        assert query_matches("fuseaux", hay)
        assert not query_matches("fooof", hay)

    def test_indexes_reorganized_menus(self):
        from CESA.qt_viewer.dialogs.feature_search_dialog import (
            collect_menu_features, query_matches, action_search_haystack,
        )
        from CESA.qt_viewer.main_window import EEGViewerMainWindow

        win = EEGViewerMainWindow()
        rows = collect_menu_features(
            win.menuBar(), skip_object_names=("featureSearch",),
        )
        labels = {label.casefold() for _, label, _, _ in rows}
        assert any("scoring automatique" in lab for lab in labels)
        assert any("nettoyage ica" in lab for lab in labels)
        paths = " | ".join(path for _, _, path, _ in rows)
        assert "Evenements" in paths
        assert "Spectrale" in paths

        matches = [
            (label, path)
            for _, label, path, shortcut in rows
            if query_matches("filtre", action_search_haystack(label, path, shortcut))
        ]
        assert matches


# -----------------------------------------------------------------------
# Display traces (channel clones) + annotation density
# -----------------------------------------------------------------------

class TestDisplayTraceModel:

    def test_next_clone_id_and_label(self):
        from CESA.qt_viewer.display_trace import next_clone_id, next_clone_label
        assert next_clone_id("C3", ["C3"]) == "C3::2"
        assert next_clone_id("C3", ["C3", "C3::2"]) == "C3::3"
        assert next_clone_label("C3", ["C3"]) == "C3 (2)"

    def test_profile_roundtrip_display_traces(self):
        from CESA.profile_schema import DisplayProcessingProfile, build_default_profile
        p = build_default_profile()
        p.display_traces = [{
            "trace_id": "C3::2",
            "source_name": "C3",
            "label": "C3 (2)",
            "color": "#FF0000",
            "visible": True,
        }]
        restored = DisplayProcessingProfile.from_dict(p.to_dict())
        assert len(restored.display_traces) == 1
        assert restored.display_traces[0]["trace_id"] == "C3::2"


@qt_required
class TestDisplayTraceViewer:

    def test_duplicate_shares_source_array(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        data = np.random.randn(2560)
        w.set_signals({"C3": (data, 256.0)})
        new_id = w.duplicate_trace("C3")
        assert new_id == "C3::2"
        assert set(w.channel_names) == {"C3", "C3::2"}
        assert w.source_name_for("C3::2") == "C3"
        # Same numpy buffer (no RAM duplication)
        assert w._strips["C3"]._full_data is w._strips["C3::2"]._full_data
        assert len(w._signals) == 1

    def test_independent_color_and_pipeline_keys(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        w.set_signals({"C3": (np.random.randn(512), 256.0)})
        w.duplicate_trace("C3")
        w.set_trace_color("C3::2", "#AABBCC")
        assert w._traces["C3::2"].color == "#AABBCC"
        # Pipelines keyed by trace id independently
        w.set_filter_pipelines({"C3": object(), "C3::2": object()})
        assert "C3" in w._pipelines and "C3::2" in w._pipelines

    def test_cannot_remove_primary(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        w.set_signals({"C3": (np.random.randn(100), 100.0)})
        assert w.remove_trace("C3") is False
        w.duplicate_trace("C3")
        assert w.remove_trace("C3::2") is True
        assert "C3::2" not in w.channel_names

    def test_clone_keeps_pipeline_when_primaries_synced(self):
        """Primary-only sync must not wipe a clone's independent filter."""
        from CESA.filter_engine import BandpassFilter, FilterPipeline
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget

        w = EEGViewerWidget()
        data = np.random.randn(2560).astype(np.float64)
        w.set_signals({"C3": (data, 256.0)})
        w.duplicate_trace("C3")
        pipe_bp = FilterPipeline(
            filters=[BandpassFilter(low_hz=11.0, high_hz=16.0)], enabled=True,
        )
        pipe_off = FilterPipeline(filters=[], enabled=False)
        w.set_filter_pipelines(
            {"C3": pipe_off, "C3::2": pipe_bp}, replace=True,
        )
        # Simulate Tk / profile sync that only knows primary channels
        w.set_filter_pipelines({"C3": pipe_off})
        assert "C3::2" in w._pipelines
        assert w._pipelines["C3::2"] is pipe_bp
        assert w._strips["C3::2"]._pipeline is pipe_bp

        w.set_global_filter_enabled(True)
        w.set_time_window(0.0, 10.0)
        _x1, y1 = w._strips["C3"]._filtered_curve.getData()
        _x2, y2 = w._strips["C3::2"]._filtered_curve.getData()
        d1 = np.asarray(y1) - w._strips["C3"]._y_offset
        d2 = np.asarray(y2) - w._strips["C3::2"]._y_offset
        n = min(len(d1), len(d2))
        assert np.max(np.abs(d1[:n] - d2[:n])) > 0.5

    def test_multi_view_recipe_n2(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        from CESA.qt_viewer.multi_view_recipes import get_recipe

        w = EEGViewerWidget()
        w.set_signals({"C3-M2": (np.random.randn(2560), 256.0)})
        recipe = get_recipe("n2_scoring")
        assert recipe is not None
        ids = w.apply_multi_view_recipe("C3-M2", recipe)
        assert len(ids) == 3
        assert ids[0] == "C3-M2"
        labels = {t.trace_id: t.label for t in w.get_traces()}
        assert labels["C3-M2"] == "C3-M2 · Standard"
        assert any("Spindles" in lab for lab in labels.values())
        assert any("K-complexe" in lab for lab in labels.values())
        colors = {t.trace_id: t.color for t in w.get_traces() if t.source_name == "C3-M2"}
        assert len(set(colors.values())) == 3
        # One physical source only
        assert list(w._signals.keys()) == ["C3-M2"]


@qt_required
class TestHypnogramAnnotationDensity:

    def test_many_annotations_bounded_items_and_yrange(self):
        from CESA.qt_viewer.hypnogram_bar import HypnogramBar
        bar = HypnogramBar()
        bar.set_hypnogram(["N2"] * 100, 30.0, 3000.0)
        events = [
            {
                "onset": float(i) * 0.5,
                "duration": 0.4,
                "type": "spindle",
                "color": "#89B4FA",
                "label": "spindle",
            }
            for i in range(3000)
        ]
        bar.set_annotations(events)
        # Density mode: far fewer items than events
        assert bar.annotation_item_count < 50
        assert bar.annotation_item_count < len(events)
        y_range = bar._plot.viewRange()[1]
        assert y_range[0] == pytest.approx(-0.8, abs=0.15)
        assert y_range[1] == pytest.approx(6.8, abs=0.15)

    def test_sparse_annotations_no_vertical_lines(self):
        from CESA.qt_viewer.hypnogram_bar import HypnogramBar
        import pyqtgraph as pg
        bar = HypnogramBar()
        bar.set_hypnogram(["W"] * 10, 30.0, 300.0)
        bar.set_annotations([
            {"onset": 10.0, "duration": 2.0, "type": "arousal", "color": "#F9E2AF"},
            {"onset": 40.0, "duration": 1.0, "type": "spindle", "color": "#89B4FA"},
        ])
        assert bar.annotation_item_count == 2
        for item in bar._annotation_items:
            assert not isinstance(item, pg.InfiniteLine)


@qt_required
class TestAnnotationOverlayDensity:

    def test_dense_window_uses_batch_not_per_event(self):
        from CESA.qt_viewer.annotations import (
            Annotation, AnnotationOverlay, AnnotationStore,
        )
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        w.set_signals({"C3": (np.random.randn(256 * 60), 256.0)})
        store = AnnotationStore()
        for i in range(200):
            store.add(Annotation(
                onset_s=float(i) * 0.2,
                duration_s=0.15,
                ann_type="spindle",
                text=f"sp{i}",
            ))
        overlay = AnnotationOverlay(w.plot_item, store)
        overlay.update_visible(0.0, 60.0)
        assert overlay._last_mode == "batch"
        # Batched: a handful of curves, not 200 regions
        assert overlay.graphics_item_count < 30
        assert len(overlay._items) == 0

    def test_sparse_uses_ignore_bounds_regions(self):
        from CESA.qt_viewer.annotations import (
            Annotation, AnnotationOverlay, AnnotationStore,
        )
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget
        w = EEGViewerWidget()
        w.set_signals({"C3": (np.random.randn(2560), 256.0)})
        y_before = w._plot.viewRange()[1]
        store = AnnotationStore()
        store.add(Annotation(onset_s=1.0, duration_s=1.0, ann_type="note", text="x"))
        overlay = AnnotationOverlay(w.plot_item, store)
        overlay.update_visible(0.0, 30.0)
        assert overlay._last_mode == "sparse"
        assert len(overlay._items) == 1
        y_after = w._plot.viewRange()[1]
        # ignoreBounds should keep channel Y range stable
        assert y_after[0] == pytest.approx(y_before[0], abs=1.0)
        assert y_after[1] == pytest.approx(y_before[1], abs=1.0)

