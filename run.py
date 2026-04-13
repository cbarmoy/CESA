#!/usr/bin/env python3
"""
CESA (Complex EEG Studio Analysis) 0.0beta1.1 - Main Launcher
===============================================================

Script de lancement principal pour l'application CESA.
Lance l'interface Qt unique avec verification des dependances.

Auteur: Come Barmoy (Unite Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.1
Date: 2026-04-05
"""

import sys
import os
import importlib.util
import logging
import io
import traceback

# PyInstaller sets sys._MEIPASS when running from a bundled .exe
FROZEN = getattr(sys, 'frozen', False)
BASE_DIR = os.path.dirname(sys.executable) if FROZEN else os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = getattr(sys, '_MEIPASS', BASE_DIR)

# pyqtgraph choisit le binding Qt au premier import ; sans cela il peut tenter PyQt5
# (exclu du bundle) et echouer alors que PySide6 est bien present.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
# matplotlib.backends.qt_compat lit QT_API pour choisir PySide6 vs PyQt5
os.environ.setdefault("QT_API", "pyside6")


def _register_win_bundle_dll_dirs() -> None:
    """Sous Windows + PyInstaller : forcer la resolution des DLL Qt depuis _internal.

    Sinon une autre Qt6Core.dll (Anaconda, PyMOL, PATH) est chargee en premier et
    provoque : DLL load failed ... procedure introuvable.
    """
    if not FROZEN or sys.platform != "win32":
        return
    root = BUNDLE_DIR
    if not root or not os.path.isdir(root):
        return
    add = getattr(os, "add_dll_directory", None)
    if not callable(add):
        return
    seen = set()
    candidates = [
        BASE_DIR,
        root,
        os.path.join(root, "PySide6"),
        os.path.join(root, "PySide6", "openssl"),
        os.path.join(root, "PySide6", "plugins"),
        os.path.join(root, "PySide6", "plugins", "platforms"),
        os.path.join(root, "shiboken6"),
    ]
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        try:
            if os.path.isdir(d):
                add(d)
        except (OSError, FileNotFoundError, ValueError):
            pass
    plugins = os.path.join(root, "PySide6", "plugins")
    if os.path.isdir(plugins):
        os.environ.setdefault("QT_PLUGIN_PATH", plugins)
    # Secours : certaines chaines de chargement consultent encore PATH en premier
    prepend = BASE_DIR + os.pathsep + root + os.pathsep + os.path.join(root, "PySide6")
    os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")


_register_win_bundle_dll_dirs()


def _frozen_log_path(name: str = "cesa_launch.log") -> str:
    d = os.path.join(BASE_DIR, "logs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _setup_early_logging() -> None:
    """File logging so windowed (console=False) builds still leave a trace."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if FROZEN:
        fh = logging.FileHandler(_frozen_log_path(), encoding="utf-8", delay=False)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if not FROZEN or sys.stdout:
        sh = logging.StreamHandler(sys.stdout if sys.stdout else sys.__stdout__)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _win_message_box(title: str, text: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text[:1024], title[:128], 0x10)
    except Exception:
        pass


_setup_early_logging()

# Forcer PySide6 du bundle AVANT matplotlib : laisser PySide6/__init__.py executer
# _setupQtDirectories() (shiboken6 + add_dll_directory) AVANT tout chargement Qt6*.dll.
# Ne pas precharger les DLL avec ctypes.WinDLL (ordre incorrect -> "procedure introuvable").
if FROZEN:
    try:
        import PySide6  # noqa: F401, E402 — enregistre shiboken + chemins DLL
        import PySide6.QtCore  # noqa: F401, E402
        import PySide6.QtGui  # noqa: F401, E402
        import PySide6.QtWidgets  # noqa: F401, E402
    except Exception as exc:
        logging.critical("Impossible de charger PySide6 (bundle Qt): %s", exc)
        try:
            with open(_frozen_log_path("cesa_crash.log"), "w", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
        except Exception:
            pass
        _win_message_box(
            "CESA",
            "Qt6 du programme ne se charge pas. Installez les Visual C++ Redistributable "
            "2015-2022 (x64) depuis Microsoft, ou reconstruisez l'exe avec Python officiel "
            "dans un venv (pas PyMOL/Anaconda).\n\n" + str(exc)[:400],
        )
        raise SystemExit(1) from exc

# Force matplotlib to use the Qt backend before any other import touches it.
import matplotlib
matplotlib.use("QtAgg")

# pyqtgraph AVANT tout import MNE dans check_dependencies (MNE peut declencher Qt via mpl).
if FROZEN:
    try:
        import pyqtgraph  # noqa: F401, E402
    except Exception as exc:
        logging.critical("pyqtgraph / Qt apres matplotlib: %s", exc, exc_info=True)
        try:
            with open(_frozen_log_path("cesa_crash.log"), "w", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
        except Exception:
            pass
        _win_message_box(
            "CESA",
            "pyqtgraph ne se charge pas (conflit DLL Qt ou build PyInstaller).\n"
            "Reconstruisez avec Python 3.10/3.11 officiel + venv, pas PyMOL/Anaconda.\n\n"
            + str(exc)[:400],
        )
        raise SystemExit(1) from exc

if sys.platform == 'win32':
    try:
        out = sys.stdout
        err = sys.stderr
        if out is not None and hasattr(out, "buffer"):
            sys.stdout = io.TextIOWrapper(out.buffer, encoding='utf-8', errors='replace')
        if err is not None and hasattr(err, "buffer"):
            sys.stderr = io.TextIOWrapper(err.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )


def _resource_path(*parts):
    """Resolve a resource path that works both in dev and in a PyInstaller bundle."""
    return os.path.join(BUNDLE_DIR, *parts)


def _find_logo_path():
    """Trouve le chemin du logo IRBA."""
    logo_dir = _resource_path('CESA', 'logo')
    for name in ('logo_IRBA.jpg', 'logo_IRBA.png', 'logo.png', 'logo.jpg'):
        p = os.path.join(logo_dir, name)
        if os.path.exists(p):
            return p
    return None


def _find_logo_esa_path():
    """Trouve le chemin du logo CESA."""
    logo_dir = _resource_path('CESA', 'logo')
    for name in ('logo_CESA.png', 'logo_CESA.jpg'):
        p = os.path.join(logo_dir, name)
        if os.path.exists(p):
            return p
    return None


def _find_icon_path():
    logo_dir = _resource_path('CESA', 'logo')
    ico = os.path.join(logo_dir, 'Icone_CESA.ico')
    if os.path.exists(ico):
        return ico
    png = os.path.join(logo_dir, 'Icone_CESA.png')
    if os.path.exists(png):
        return png
    return None


def check_dependencies():
    """Verifie que toutes les dependances sont installees."""
    required_packages = {
        'mne': 'MNE-Python',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'PySide6': 'PySide6',
        'pyqtgraph': 'pyqtgraph',
    }

    missing = []
    for package, name in required_packages.items():
        ok = False
        if FROZEN:
            # Deja charge au demarrage (pyqtgraph / PySide6 avant MNE) : ne pas re-importer
            if package in sys.modules:
                ok = True
                continue
            # importlib.util.find_spec est souvent faux-negatif dans un .exe PyInstaller
            try:
                __import__(package)
                ok = True
            except Exception as exc:
                logging.warning("Import test %r a echoue: %s", package, exc)
                ok = False
        else:
            ok = importlib.util.find_spec(package) is not None
        if not ok:
            missing.append(name)

    if missing:
        msg = "Dependances manquantes:\n" + "\n".join(f"  - {p}" for p in missing)
        if FROZEN:
            msg += "\n\nVoir logs/cesa_launch.log"
        logging.error(msg)
        print(msg)
        if not FROZEN:
            print("\nPour installer les dependances:")
            print("   pip install -r requirements.txt")
        if FROZEN:
            _win_message_box("CESA — dependances", msg)
        return False
    return True


def _build_splash_pixmap():
    """Build a QPixmap for the splash screen from available logos."""
    from PySide6 import QtCore, QtGui

    logo_irba = _find_logo_path()
    logo_cesa = _find_logo_esa_path()

    max_w, max_h = 320, 240
    images = []

    for path in (logo_irba, logo_cesa):
        if path is None:
            continue
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            continue
        pm = pm.scaled(
            max_w, max_h,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        images.append(pm)

    if not images:
        pm = QtGui.QPixmap(400, 120)
        pm.fill(QtGui.QColor("black"))
        painter = QtGui.QPainter(pm)
        painter.setPen(QtGui.QColor("white"))
        painter.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Weight.Bold))
        painter.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "CESA 0.0beta1.1")
        painter.end()
        return pm

    total_w = sum(p.width() for p in images) + 16 * (len(images) - 1) + 16
    total_h = max(p.height() for p in images) + 16
    canvas = QtGui.QPixmap(total_w, total_h)
    canvas.fill(QtGui.QColor("black"))
    painter = QtGui.QPainter(canvas)
    x = 8
    for p in images:
        y = (total_h - p.height()) // 2
        painter.drawPixmap(x, y, p)
        x += p.width() + 16
    painter.end()
    return canvas


def main():
    """Fonction principale de lancement."""
    if BUNDLE_DIR not in sys.path:
        sys.path.insert(0, BUNDLE_DIR)
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    from core.telemetry import telemetry  # noqa: E402

    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    telemetry.configure(
        profile_io=True,
        profile_render=True,
        track_fps=True,
        csv_path=os.path.join(BASE_DIR, "logs", "telemetry.csv"),
        reset_on_start=True,
    )
    telemetry.expect_modes(["lazy", "precomputed"])

    print("CESA (Complex EEG Studio Analysis) 0.0beta1.1")
    print("=" * 50)

    logging.info("Checking dependencies...")
    if not check_dependencies():
        logging.error("Cannot launch application - missing dependencies")
        return 1

    logging.info("All dependencies available")

    try:
        logging.info("Launching CESA 0.0beta1.1...")

        from PySide6 import QtWidgets, QtGui, QtCore

        if FROZEN:
            # Evite ecran vide / plantage silencieux si le pilote OpenGL pose probleme
            try:
                QtCore.QCoreApplication.setAttribute(
                    QtCore.Qt.ApplicationAttribute.AA_UseSoftwareOpenGL,
                    True,
                )
            except Exception:
                pass

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        app.setApplicationName("CESA")
        app.setApplicationVersion("0.0beta1.1")

        icon_path = _find_icon_path()
        if icon_path:
            app.setWindowIcon(QtGui.QIcon(icon_path))

        splash_pixmap = _build_splash_pixmap()
        splash = QtWidgets.QSplashScreen(splash_pixmap)
        splash.show()
        app.processEvents()

        splash.showMessage(
            "Chargement de l'interface...",
            QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignHCenter,
            QtGui.QColor("white"),
        )
        app.processEvents()

        from CESA.app_controller import AppController
        from CESA.qt_viewer.main_window import EEGViewerMainWindow

        controller = AppController()
        window = EEGViewerMainWindow(theme_name="dark")
        window.set_app_controller(controller)

        if icon_path:
            window.setWindowIcon(QtGui.QIcon(icon_path))

        window.showMaximized()
        splash.finish(window)

        logging.info("Application ready")

        return app.exec()

    except KeyboardInterrupt:
        logging.info("Application closed by user")
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Error during launch: %s\n%s", e, tb)
        try:
            print(tb)
        except Exception:
            pass
        if FROZEN:
            try:
                with open(_frozen_log_path("cesa_crash.log"), "w", encoding="utf-8") as fh:
                    fh.write(tb)
            except Exception:
                pass
            _win_message_box(
                "CESA — erreur au demarrage",
                f"{e}\n\nDetails: logs/cesa_crash.log ou logs/cesa_launch.log",
            )
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        tb = traceback.format_exc()
        try:
            logging.critical(tb)
        except Exception:
            pass
        try:
            with open(_frozen_log_path("cesa_crash.log"), "w", encoding="utf-8") as fh:
                fh.write(tb)
        except Exception:
            pass
        if FROZEN:
            _win_message_box("CESA — exception", tb[:900])
        sys.exit(1)
