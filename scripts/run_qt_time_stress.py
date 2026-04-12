#!/usr/bin/env python3
"""Stress manuel Qt : avancement temporel (repro hypothèse crash ~30 s).

Usage (depuis la racine du dépôt) :
  set QT_QPA_PLATFORM=offscreen   # optionnel, CI / sans écran
  python scripts/run_qt_time_stress.py
  python scripts/run_qt_time_stress.py --gui
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress navigation temporelle viewer Qt")
    parser.add_argument("--gui", action="store_true", help="Afficher la fenêtre (défaut: offscreen)")
    parser.add_argument("--iterations", type=int, default=500, help="Pas de 1 s (Right)")
    parser.add_argument("--total-sec", type=float, default=3600.0, help="Durée totale enregistrement")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Fenêtre visible")
    args = parser.parse_args()

    if not args.gui:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import numpy as np
    from PySide6 import QtCore, QtWidgets

    from CESA.qt_viewer.main_window import EEGViewerMainWindow, _qt_key_int

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    sfreq = 256.0
    n = int(args.total_sec * sfreq)
    rng = np.random.default_rng(0)
    signals = {
        "C3-M2": (rng.standard_normal(n).astype(np.float64), sfreq),
        "EOG-L": (rng.standard_normal(n).astype(np.float64), sfreq),
    }
    win = EEGViewerMainWindow(
        signals=signals,
        hypnogram=(["W"] * max(1, int(args.total_sec / 30)), 30.0),
        total_duration_s=args.total_sec,
        duration_s=args.window_sec,
    )
    if args.gui:
        win.show()

    key_r = _qt_key_int(QtCore.Qt.Key.Key_Right)
    t0 = time.perf_counter()
    for i in range(args.iterations):
        win._dispatch_key_shortcuts_ints(key_r, 0)
        if i % 25 == 0:
            for _ in range(4):
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 5)
    elapsed = time.perf_counter() - t0
    for _ in range(20):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 10)

    print(
        f"OK: {args.iterations} pas, start_s={win._viewer.start_s:.2f}, "
        f"durée {elapsed:.2f}s wall"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
