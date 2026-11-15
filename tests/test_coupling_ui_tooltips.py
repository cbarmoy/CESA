"""New coupling options must be explained in the CESA dialogs themselves."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture
def app():
    from PySide6 import QtWidgets

    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing
    return QtWidgets.QApplication([])


def test_micro_events_coupling_options_have_tooltips(app):
    from CESA.qt_viewer.dialogs.micro_events_dialog import (
        MicroEventsDialog, _CLASSIC_HEADER_TIPS, _CLASSIC_HEADERS,
    )

    dlg = MicroEventsDialog(
        None, controller=None,
        eeg_channels=["C3"], eog_channels=[],
        all_channels=["C3"], channel_types={"C3": "eeg"},
        has_scoring=False,
    )
    for widget in (dlg._chk_perm, dlg._chk_circ, dlg._chk_avg, dlg._chk_tfr):
        tip = widget.toolTip()
        assert len(tip) > 40, widget.text()
    assert "Sorties couplage" in dlg._btn_figure.toolTip()
    assert len(_CLASSIC_HEADER_TIPS) == len(_CLASSIC_HEADERS)
    assert all(len(tip) > 10 for tip in _CLASSIC_HEADER_TIPS)
    header = dlg._classic_table.horizontalHeaderItem(11)  # V p
    assert header is not None
    assert "up-state" in header.toolTip()
    dlg.close()


def test_batch_coupling_options_have_tooltips_and_hint(app):
    from CESA.qt_viewer.dialogs.batch_analysis_dialog import (
        MODE_COUPLING, BatchAnalysisDialog,
    )

    dlg = BatchAnalysisDialog(None)
    dlg._mode_combo.setCurrentIndex(MODE_COUPLING)
    for widget in (
        dlg._perm_chk, dlg._circ_fig_chk, dlg._avg_fig_chk, dlg._tfr_chk,
        dlg._events_chk, dlg._min_coupled_spin,
    ):
        assert len(widget.toolTip()) > 40, widget
    hint = dlg._mode_hint.text()
    assert "CSV" in hint
    assert "TFR" in hint
    assert not dlg._coupling_box.isHidden()
    assert dlg._fft_box.isHidden()
    dlg._mode_combo.setCurrentIndex(0)
    assert dlg._coupling_box.isHidden()
    assert not dlg._fft_box.isHidden()
    dlg.close()


def test_coupling_plot_dialog_explains_figure_types(app):
    from PySide6 import QtCore

    from CESA.qt_viewer.dialogs.coupling_plot_dialog import (
        TYPE_AVERAGE, TYPE_PHASE, TYPE_TFR, CouplingPlotDialog,
    )

    dlg = CouplingPlotDialog(
        None,
        events=[{"channel": "C3", "metadata": {"so_phase_rad": 0.1}}],
        coupling=[{"channel": "C3", "n_coupled": 20}],
        enabled={TYPE_PHASE, TYPE_AVERAGE, TYPE_TFR},
    )
    tips = [
        dlg._type_combo.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
        for i in range(dlg._type_combo.count())
    ]
    assert len(tips) == 3
    assert all(isinstance(t, str) and len(t) > 40 for t in tips)
    assert "20°" in dlg._bins_spin.toolTip()
    dlg.close()
