"""Every interactive control of the batch dialog must carry a hover tooltip."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets


@pytest.fixture
def app():
    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing
    return QtWidgets.QApplication([])


_INTERACTIVE = (
    QtWidgets.QAbstractButton,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QLineEdit,
    QtWidgets.QListWidget,
    QtWidgets.QPlainTextEdit,
    QtWidgets.QProgressBar,
)


def _is_spin_or_combo_internal(widget: QtWidgets.QWidget) -> bool:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, (
            QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox,
            QtWidgets.QTabBar, QtWidgets.QScrollBar,
        )):
            return True
        parent = parent.parentWidget()
    return False


def _interactive_controls(root: QtWidgets.QWidget):
    for child in root.findChildren(QtWidgets.QWidget):
        if not isinstance(child, _INTERACTIVE):
            continue
        if _is_spin_or_combo_internal(child):
            continue
        yield child


def test_every_batch_control_has_a_tooltip(app):
    from CESA.qt_viewer.dialogs.batch_analysis_dialog import BatchAnalysisDialog

    dlg = BatchAnalysisDialog(None)
    missing = []
    for widget in _interactive_controls(dlg):
        tip = widget.toolTip().strip()
        if len(tip) < 20:
            missing.append(f"{type(widget).__name__} {widget.objectName()!r} {widget!r}: {tip!r}")
    dlg.close()
    assert missing == [], "controles sans infobulle:\n" + "\n".join(missing)


def test_batch_mode_combo_items_have_tooltips(app):
    from PySide6 import QtCore

    from CESA.qt_viewer.dialogs.batch_analysis_dialog import BatchAnalysisDialog

    dlg = BatchAnalysisDialog(None)
    assert dlg._mode_combo.count() == 4
    for i in range(dlg._mode_combo.count()):
        tip = dlg._mode_combo.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
        assert isinstance(tip, str) and len(tip) > 20, i
    dlg.close()


def _plain_tooltip_lines(tip: str):
    import re

    text = re.sub(r"<br\s*/?>", "\n", tip, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_batch_tooltips_wrap_instead_of_one_long_line(app):
    from CESA.qt_viewer.dialogs.batch_analysis_dialog import BatchAnalysisDialog

    dlg = BatchAnalysisDialog(None)
    too_long = []
    for widget in _interactive_controls(dlg):
        tip = widget.toolTip().strip()
        for line in _plain_tooltip_lines(tip) or [tip]:
            if len(line) > 60:
                too_long.append(f"{type(widget).__name__}: {line!r}")
    dlg.close()
    assert too_long == [], "infobulles trop longues:\n" + "\n".join(too_long)


def test_fft_fields_stay_enabled_and_hide_in_coupling_mode(app):
    from CESA.qt_viewer.dialogs.batch_analysis_dialog import (
        MODE_COUPLING, MODE_FFT, BatchAnalysisDialog,
    )

    dlg = BatchAnalysisDialog(None)
    assert not dlg._fft_box.isHidden()
    assert dlg._coupling_box.isHidden()
    assert dlg._fmin_spin.isEnabled()
    assert dlg._fmax_spin.isEnabled()
    assert dlg._nperseg_spin.isEnabled()
    assert dlg._method_combo.isEnabled()

    dlg._mode_combo.setCurrentIndex(MODE_COUPLING)
    assert dlg._fft_box.isHidden()
    assert not dlg._coupling_box.isHidden()

    dlg._mode_combo.setCurrentIndex(MODE_FFT)
    assert not dlg._fft_box.isHidden()
    assert dlg._coupling_box.isHidden()
    assert dlg._fmin_spin.isEnabled()
    dlg.close()
