"""Qt dialog: editable manual scoring table.

Provides a row-by-row editable view of the active sleep scoring
(``app_controller.sleep_scoring_data``). Each epoch can be re-scored via a
combo-box, bulk edits are supported on the current selection, and the result
can be applied back to the recording or exported to CSV/JSON.

Replaces the Tk ``_open_manual_scoring_editor`` Treeview window.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_STAGES: List[str] = ["W", "N1", "N2", "N3", "R", "U"]

_STAGE_COLORS = {
    "W": "#f38ba8",
    "N1": "#fab387",
    "N2": "#a6e3a1",
    "N3": "#89b4fa",
    "R": "#cba6f7",
    "U": "#9399b2",
}


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class _StageDelegate(QtWidgets.QStyledItemDelegate):
    """Combo-box editor for the stage column."""

    def createEditor(self, parent, option, index):  # noqa: N802 (Qt API)
        combo = QtWidgets.QComboBox(parent)
        combo.addItems(_STAGES)
        return combo

    def setEditorData(self, editor, index):  # noqa: N802
        value = index.data(QtCore.Qt.ItemDataRole.EditRole) or "U"
        pos = editor.findText(str(value))
        editor.setCurrentIndex(max(0, pos))

    def setModelData(self, editor, model, index):  # noqa: N802
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)


class ScoringEditorDialog(QtWidgets.QDialog):
    """Editable manual scoring table backed by the application controller."""

    COL_EPOCH = 0
    COL_TIME = 1
    COL_HMS = 2
    COL_STAGE = 3

    def __init__(self, parent: Optional[QtWidgets.QWidget], controller) -> None:
        super().__init__(parent)
        self.setWindowTitle("Editeur de scoring manuel")
        self.resize(640, 720)
        self._ctrl = controller
        self._main = parent
        self._epoch_len = float(getattr(controller, "scoring_epoch_duration", 30.0) or 30.0)
        self._build_ui()
        self._reload()

    # -- UI -------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Filtrer:"))
        self._filter_combo = QtWidgets.QComboBox()
        self._filter_combo.addItems(["Tous"] + _STAGES)
        self._filter_combo.currentTextChanged.connect(self._apply_row_filter)
        top.addWidget(self._filter_combo)
        top.addSpacing(16)
        top.addWidget(QtWidgets.QLabel("Affecter la selection a:"))
        self._bulk_combo = QtWidgets.QComboBox()
        self._bulk_combo.addItems(_STAGES)
        top.addWidget(self._bulk_combo)
        bulk_btn = QtWidgets.QPushButton("Appliquer a la selection")
        bulk_btn.clicked.connect(self._bulk_assign)
        top.addWidget(bulk_btn)
        top.addStretch()
        layout.addLayout(top)

        self._table = QtWidgets.QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Epoque", "Temps (s)", "Heure", "Stade"])
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._table.setItemDelegateForColumn(self.COL_STAGE, _StageDelegate(self._table))
        self._table.itemDoubleClicked.connect(self._on_double_click)
        self._table.itemChanged.connect(self._on_item_changed)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._info = QtWidgets.QLabel("")
        layout.addWidget(self._info)

        btns = QtWidgets.QHBoxLayout()
        reload_btn = QtWidgets.QPushButton("Recharger")
        reload_btn.clicked.connect(self._reload)
        btns.addWidget(reload_btn)
        btns.addStretch()
        apply_btn = QtWidgets.QPushButton("Appliquer au scoring")
        apply_btn.clicked.connect(self._apply_back)
        btns.addWidget(apply_btn)
        exp_csv = QtWidgets.QPushButton("Exporter CSV...")
        exp_csv.clicked.connect(lambda: self._export("csv"))
        btns.addWidget(exp_csv)
        exp_json = QtWidgets.QPushButton("Exporter JSON...")
        exp_json.clicked.connect(lambda: self._export("json"))
        btns.addWidget(exp_json)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    # -- Data -----------------------------------------------------------

    def _active_df(self) -> Optional[pd.DataFrame]:
        df = getattr(self._ctrl, "sleep_scoring_data", None)
        if df is None or df.empty:
            return None
        return df

    def _reload(self) -> None:
        df = self._active_df()
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        if df is None:
            self._table.blockSignals(False)
            self._info.setText("Aucun scoring disponible. Lancez un scoring automatique ou importez-en un.")
            return
        df = df.reset_index(drop=True)
        self._epoch_len = float(getattr(self._ctrl, "scoring_epoch_duration", 30.0) or 30.0)
        self._table.setRowCount(len(df))
        for row in range(len(df)):
            time_s = float(df.loc[row, "time"]) if "time" in df.columns else row * self._epoch_len
            stage = str(df.loc[row, "stage"]) if "stage" in df.columns else "U"

            ep_item = QtWidgets.QTableWidgetItem(str(row))
            ep_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            t_item = QtWidgets.QTableWidgetItem(f"{time_s:.1f}")
            t_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            hms_item = QtWidgets.QTableWidgetItem(_fmt_hms(time_s))
            hms_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            st_item = QtWidgets.QTableWidgetItem(stage)
            st_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEditable
            )
            self._colorize_stage(st_item, stage)

            self._table.setItem(row, self.COL_EPOCH, ep_item)
            self._table.setItem(row, self.COL_TIME, t_item)
            self._table.setItem(row, self.COL_HMS, hms_item)
            self._table.setItem(row, self.COL_STAGE, st_item)
        self._table.blockSignals(False)
        self._apply_row_filter(self._filter_combo.currentText())
        self._update_info()

    @staticmethod
    def _colorize_stage(item: QtWidgets.QTableWidgetItem, stage: str) -> None:
        color = _STAGE_COLORS.get(stage.upper())
        if color:
            item.setForeground(QtGui.QColor(color))
            font = item.font()
            font.setBold(True)
            item.setFont(font)

    # -- Interaction ----------------------------------------------------

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() != self.COL_STAGE:
            return
        stage = item.text().strip().upper()
        if stage not in _STAGES:
            stage = "U"
            self._table.blockSignals(True)
            item.setText(stage)
            self._table.blockSignals(False)
        self._colorize_stage(item, stage)
        self._update_info()

    def _on_double_click(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() in (self.COL_EPOCH, self.COL_TIME, self.COL_HMS):
            self._navigate_to_row(item.row())

    def _navigate_to_row(self, row: int) -> None:
        t_item = self._table.item(row, self.COL_TIME)
        if t_item is None:
            return
        try:
            time_s = float(t_item.text())
        except ValueError:
            return
        main = self._main
        if main is not None and hasattr(main, "set_time_window"):
            try:
                dur = float(getattr(main._viewer, "duration_s", self._epoch_len))
                main.set_time_window(time_s, dur or self._epoch_len)
            except Exception:
                logger.debug("navigate to row failed", exc_info=True)

    def _bulk_assign(self) -> None:
        stage = self._bulk_combo.currentText()
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        if not rows:
            QtWidgets.QMessageBox.information(self, "Selection", "Aucune ligne selectionnee.")
            return
        self._table.blockSignals(True)
        for row in rows:
            item = self._table.item(row, self.COL_STAGE)
            if item is not None:
                item.setText(stage)
                self._colorize_stage(item, stage)
        self._table.blockSignals(False)
        self._update_info()

    def _apply_row_filter(self, stage: str) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self.COL_STAGE)
            visible = stage == "Tous" or (item is not None and item.text() == stage)
            self._table.setRowHidden(row, not visible)

    def _update_info(self) -> None:
        counts = {}
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self.COL_STAGE)
            if item is not None:
                counts[item.text()] = counts.get(item.text(), 0) + 1
        total = sum(counts.values())
        parts = [f"{s}: {counts.get(s, 0)}" for s in _STAGES if counts.get(s)]
        self._info.setText(f"{total} epoques  |  " + "  ".join(parts))

    # -- Apply / export -------------------------------------------------

    def _collect_stages(self) -> List[str]:
        return [
            (self._table.item(row, self.COL_STAGE).text()
             if self._table.item(row, self.COL_STAGE) else "U")
            for row in range(self._table.rowCount())
        ]

    def _apply_back(self) -> None:
        df = self._active_df()
        if df is None:
            return
        stages = self._collect_stages()
        if len(stages) != len(df):
            QtWidgets.QMessageBox.warning(
                self, "Incoherence",
                "Le nombre de lignes ne correspond plus au scoring. Rechargez.",
            )
            return
        try:
            new_df = df.reset_index(drop=True).copy()
            new_df["stage"] = stages
            self._ctrl.sleep_scoring_data = new_df
            self._ctrl.scoring_dirty = True
            main = self._main
            if main is not None:
                hyp = self._ctrl.build_hypnogram_tuple()
                if hyp:
                    main.set_hypnogram(hyp)
                try:
                    main.set_scoring_annotations(self._ctrl.build_scoring_annotations())
                except Exception:
                    pass
                if hasattr(main, "update_dashboard"):
                    main.update_dashboard()
            QtWidgets.QMessageBox.information(
                self, "Scoring", "Modifications appliquees au scoring.",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Application echouee:\n{exc}")

    def _export(self, fmt: str) -> None:
        # Apply edits to the in-memory DataFrame first so the export reflects them.
        df = self._active_df()
        if df is None:
            QtWidgets.QMessageBox.information(self, "Export", "Aucun scoring a exporter.")
            return
        self._apply_back()
        if fmt == "json":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Exporter scoring JSON", "", "JSON (*.json)"
            )
            if path:
                ok = self._ctrl.save_scoring_json(path)
                self._notify_export(ok, path)
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Exporter scoring CSV", "", "CSV (*.csv)"
            )
            if path:
                ok = self._ctrl.save_scoring_csv(path)
                self._notify_export(ok, path)

    def _notify_export(self, ok: bool, path: str) -> None:
        if ok:
            QtWidgets.QMessageBox.information(self, "Export", f"Scoring exporte:\n{path}")
        else:
            QtWidgets.QMessageBox.warning(self, "Export", "Echec de l'export.")
