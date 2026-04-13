# PyInstaller runtime hook — s'exécute avant le script utilisateur (run.py).
# Sans cela, Windows peut charger une Qt6Core.dll du PATH (Anaconda, PyMOL, etc.)
# incompatible avec les extensions PySide6 du bundle.

import os
import sys


def _bootstrap_qt_paths() -> None:
    if not getattr(sys, "frozen", False):
        return
    if sys.platform != "win32":
        return
    base = getattr(sys, "_MEIPASS", "") or ""
    if not base or not os.path.isdir(base):
        return

    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    os.environ.setdefault("QT_API", "pyside6")

    add = getattr(os, "add_dll_directory", None)
    if not callable(add):
        return

    exe_dir = os.path.dirname(sys.executable)
    candidates = (
        exe_dir,
        base,
        os.path.join(base, "PySide6"),
        os.path.join(base, "PySide6", "openssl"),
        os.path.join(base, "PySide6", "plugins"),
        os.path.join(base, "PySide6", "plugins", "platforms"),
        os.path.join(base, "shiboken6"),
    )
    seen = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        try:
            if os.path.isdir(d):
                add(d)
        except (OSError, ValueError, FileNotFoundError):
            pass

    plugins = os.path.join(base, "PySide6", "plugins")
    if os.path.isdir(plugins):
        os.environ.setdefault("QT_PLUGIN_PATH", plugins)

    prepend = exe_dir + os.pathsep + base + os.pathsep + os.path.join(base, "PySide6")
    os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")

    # Ne pas precharger les DLL Qt avec ctypes.WinDLL : il faut que PySide6/__init__.py
    # charge shiboken6 et enregistre les repertoires AVANT Qt6*.dll, sinon erreur
    # "procedure introuvable" sur QtCore.


_bootstrap_qt_paths()
del _bootstrap_qt_paths
