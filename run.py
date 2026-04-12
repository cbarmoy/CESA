#!/usr/bin/env python3
"""
CESA (Complex EEG Studio Analysis) 0.0beta1.0 - Main Launcher
===============================================================

Script de lancement principal pour l'application CESA.
Lance l'interface Qt unique avec verification des dependances.

Auteur: Come Barmoy (Unite Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.0
Date: 2026-04-05
"""

import sys
import os
import importlib.util
import logging
import io

# Force matplotlib to use the Qt backend before any other import touches it.
import matplotlib
matplotlib.use("QtAgg")

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

from core.telemetry import telemetry  # noqa: E402

telemetry.configure(
    profile_io=True,
    profile_render=True,
    track_fps=True,
    csv_path="logs/telemetry.csv",
    reset_on_start=True,
)
telemetry.expect_modes(["lazy", "precomputed"])


def _find_logo_path():
    """Trouve le chemin du logo IRBA."""
    base = os.path.dirname(os.path.abspath(__file__))
    logo_dir = os.path.join(base, 'CESA', 'logo')
    for name in ('logo_IRBA.jpg', 'logo_IRBA.png', 'logo.png', 'logo.jpg'):
        p = os.path.join(logo_dir, name)
        if os.path.exists(p):
            return p
    return None


def _find_logo_esa_path():
    """Trouve le chemin du logo CESA."""
    base = os.path.dirname(os.path.abspath(__file__))
    logo_dir = os.path.join(base, 'CESA', 'logo')
    for name in ('logo_CESA.png', 'logo_CESA.jpg'):
        p = os.path.join(logo_dir, name)
        if os.path.exists(p):
            return p
    return None


def _find_icon_path():
    base = os.path.dirname(os.path.abspath(__file__))
    ico = os.path.join(base, 'CESA', 'logo', 'Icone_CESA.ico')
    if os.path.exists(ico):
        return ico
    png = os.path.join(base, 'CESA', 'logo', 'Icone_CESA.png')
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
        if importlib.util.find_spec(package) is None:
            missing.append(name)

    if missing:
        print("Dependances manquantes:")
        for p in missing:
            print(f"   - {p}")
        print("\nPour installer les dependances:")
        print("   pip install -r requirements.txt")
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
        painter.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "CESA 0.0beta1.0")
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
    print("CESA (Complex EEG Studio Analysis) 0.0beta1.0")
    print("=" * 50)

    logging.info("Checking dependencies...")
    if not check_dependencies():
        logging.error("Cannot launch application - missing dependencies")
        return 1

    logging.info("All dependencies available")

    try:
        logging.info("Launching CESA 0.0beta1.0...")

        from PySide6 import QtWidgets, QtGui, QtCore

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        app.setApplicationName("CESA")
        app.setApplicationVersion("0.0beta1.0")

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
        logging.error(f"Error during launch: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
