"""Tests de stress : avancement temporel (hypothèse plantage ~30 s / rafale).

Exécuter : pytest tests/test_qt_viewer_time_stress.py -v --tb=short
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from PySide6 import QtCore, QtWidgets
    import pyqtgraph  # noqa: F401

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    HAS_QT = True
except ImportError:
    HAS_QT = False

qt_required = pytest.mark.skipif(not HAS_QT, reason="PySide6/pyqtgraph not installed")


def _process(n: int = 3) -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for _ in range(n):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 5)


@qt_required
class TestNavigationBarTimeStress:
    """Beaucoup d'appels _emit_time / pas — sans fenêtre complète."""

    def test_many_step_forward_crossing_30s_epochs(self):
        from CESA.qt_viewer.main_window import _qt_key_int
        from CESA.qt_viewer.navigation_bar import NavigationBar

        nav = NavigationBar()
        nav.set_total_duration(3600.0)
        nav.set_duration(30.0)
        nav.set_epoch_length(30.0)
        nav.set_current_time(0.0)
        last = 0.0
        key_r = _qt_key_int(QtCore.Qt.Key.Key_Right)
        for i in range(500):
            nav.handle_key_ints(key_r, QtCore.Qt.KeyboardModifiers())
            _process(1)
            assert nav._current_time >= last - 1e-6
            last = nav._current_time
        assert nav._current_time <= 3600.0 - 30.0 + 1e-3

    def test_emit_at_upper_bound_idempotent(self):
        from CESA.qt_viewer.navigation_bar import NavigationBar

        nav = NavigationBar()
        nav.set_total_duration(120.0)
        nav.set_duration(30.0)
        max_t = 120.0 - 30.0
        for _ in range(200):
            nav._emit_time(max_t + 50.0)  # clamp
            _process(1)
        assert nav._current_time == pytest.approx(max_t)

    def test_jump_scored_epochs_many_times(self):
        from CESA.qt_viewer.navigation_bar import NavigationBar

        nav = NavigationBar()
        nav.set_total_duration(600.0)
        nav.set_duration(30.0)
        nav.set_epoch_length(30.0)
        nav.set_current_time(15.0)
        for _ in range(300):
            nav._jump_next_scored_epoch()
            _process(1)
            assert 0 <= nav._current_time <= 570.0 + 1e-3


@qt_required
class TestEEGViewerWidgetTimeStress:
    def test_set_time_window_sweep_with_refresh(self):
        from CESA.qt_viewer.eeg_viewer_widget import EEGViewerWidget

        w = EEGViewerWidget()
        sfreq = 256.0
        n = int(sfreq * 600)
        rng = np.random.default_rng(42)
        w.set_signals({"C3": (rng.standard_normal(n).astype(np.float64), sfreq)})
        w.set_total_duration(600.0)
        for t in range(0, 450, 1):
            w.set_time_window(float(t), 30.0)
            w._refresh_view()
            if t % 20 == 0:
                _process(2)
        assert w.start_s == pytest.approx(449.0)


@qt_required
class TestMainWindowTimeStress:
    def test_nav_key_burst_crossing_30s(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow, _qt_key_int

        sfreq = 256.0
        n = int(sfreq * 1200)
        rng = np.random.default_rng(0)
        signals = {
            "C3-M2": (rng.standard_normal(n).astype(np.float64), sfreq),
            "EOG-L": (rng.standard_normal(n).astype(np.float64), sfreq),
        }
        win = EEGViewerMainWindow(
            signals=signals,
            hypnogram=(["W"] * 40, 30.0),
            total_duration_s=float(n / sfreq),
            duration_s=30.0,
            start_time_s=0.0,
        )
        key_r = _qt_key_int(QtCore.Qt.Key.Key_Right)
        for i in range(400):
            win._dispatch_key_shortcuts_ints(key_r, 0)
            if i % 10 == 0:
                _process(3)
        assert win._viewer.start_s > 25.0
        assert win._viewer.start_s <= float(n / sfreq) - 30.0 + 0.01

    def test_set_time_window_repeated_main_window(self):
        from CESA.qt_viewer.main_window import EEGViewerMainWindow

        sfreq = 128.0
        n = int(sfreq * 300)
        rng = np.random.default_rng(1)
        signals = {"A": (rng.standard_normal(n).astype(np.float64), sfreq)}
        win = EEGViewerMainWindow(
            signals=signals,
            total_duration_s=300.0,
            duration_s=30.0,
        )
        for t in np.linspace(0, 250, 800):
            win.set_time_window(float(t), 30.0)
            if int(t) % 15 == 0:
                _process(2)
        assert win._viewer.start_s == pytest.approx(250.0, abs=0.5)

    def test_play_timer_ticks_stress(self):
        """Simule lecture : +1 s par tick comme _on_play_tick."""
        from CESA.qt_viewer.navigation_bar import NavigationBar

        nav = NavigationBar()
        nav.set_total_duration(400.0)
        nav.set_duration(30.0)
        times = []

        def rec(t: float) -> None:
            times.append(t)

        nav.time_changed.connect(rec)
        for _ in range(320):
            nav._on_play_tick()
            _process(1)
        assert nav._current_time <= 370.0 + 0.1


@qt_required
class TestTimeAxisEdgeCases:
    def test_large_time_values(self):
        from CESA.qt_viewer.time_axis_item import TimeAxisItem

        axis = TimeAxisItem()
        s = axis._fmt(86400.0 + 90.0)
        assert "01:30" in s or "25:" in s or len(s) >= 4
