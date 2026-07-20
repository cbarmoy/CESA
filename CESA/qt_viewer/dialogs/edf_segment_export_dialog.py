"""Qt dialog: export a time segment of the recording to EDF.

Crops the loaded ``mne.io.Raw`` to a chosen ``[start, start+duration]`` window
(optionally restricted to selected channels) and writes it via MNE's EDF
exporter. Replaces the Tk ``_export_edf_segment`` window.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6 import QtWidgets

logger = logging.getLogger(__name__)


class EdfSegmentExportDialog(QtWidgets.QDialog):
    """Export a cropped segment of the current recording to EDF."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], controller) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exporter un segment EDF")
        self.resize(440, 480)
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        total = float(getattr(self._ctrl, "total_duration_s", 0.0) or 0.0)
        cur = float(getattr(self._ctrl, "current_time", 0.0) or 0.0)

        self._start = QtWidgets.QDoubleSpinBox()
        self._start.setRange(0.0, max(total, 1e6))
        self._start.setDecimals(2)
        self._start.setSuffix(" s")
        self._start.setValue(cur)
        form.addRow("Debut :", self._start)

        self._dur = QtWidgets.QDoubleSpinBox()
        self._dur.setRange(0.1, max(total, 1e6))
        self._dur.setDecimals(2)
        self._dur.setSuffix(" s")
        self._dur.setValue(30.0)
        form.addRow("Duree :", self._dur)

        layout.addLayout(form)

        if total > 0:
            layout.addWidget(QtWidgets.QLabel(f"Enregistrement total : {total:.1f} s"))

        layout.addWidget(QtWidgets.QLabel("Canaux (aucun selectionne = tous) :"))
        self._chan_list = QtWidgets.QListWidget()
        self._chan_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        raw = getattr(self._ctrl, "raw", None)
        if raw is not None:
            self._chan_list.addItems(list(raw.ch_names))
        layout.addWidget(self._chan_list, stretch=1)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        export_btn = QtWidgets.QPushButton("Exporter...")
        export_btn.clicked.connect(self._do_export)
        btns.addButton(export_btn, QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _selected_channels(self) -> Optional[List[str]]:
        sel = [it.text() for it in self._chan_list.selectedItems()]
        return sel or None

    def _do_export(self) -> None:
        raw = getattr(self._ctrl, "raw", None)
        if raw is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Aucun enregistrement charge.")
            return
        start_s = float(self._start.value())
        dur_s = float(self._dur.value())
        if dur_s <= 0:
            QtWidgets.QMessageBox.warning(self, "Erreur", "La duree doit etre > 0.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer le segment EDF", "", "EDF (*.edf)"
        )
        if not path:
            return
        if not path.lower().endswith(".edf"):
            path += ".edf"

        try:
            import mne

            tmax_avail = float(raw.times[-1])
            end_s = min(start_s + dur_s, tmax_avail)
            seg = raw.copy().crop(tmin=max(0.0, start_s), tmax=end_s)
            chans = self._selected_channels()
            if chans:
                seg.pick(chans)
            seg.export(path, fmt="edf", overwrite=True)
            QtWidgets.QMessageBox.information(
                self, "Succes",
                f"Segment exporte ({end_s - start_s:.1f} s, "
                f"{len(seg.ch_names)} canaux):\n{path}",
            )
            self.accept()
        except Exception as exc:
            logger.error("EDF segment export failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Erreur",
                f"Echec de l'export du segment:\n{exc}\n\n"
                "L'export EDF requiert le paquet 'edfio' (pip install edfio).",
            )
