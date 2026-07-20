"""Qt dialog: named temporal markers (add / remove / navigate).

Markers are stored on ``app_controller.temporal_markers`` as a list of
``{"description", "position", "color"}`` dicts and rendered on the EEG viewer
as labelled vertical lines. Replaces the Tk ``_show_markers`` window.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_COLORS = ["#f38ba8", "#89b4fa", "#a6e3a1", "#fab387", "#cba6f7", "#94e2d5", "#f9e2af"]


class MarkersDialog(QtWidgets.QDialog):
    """Manage named temporal markers for the current recording."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], controller) -> None:
        super().__init__(parent)
        self.setWindowTitle("Marqueurs temporels")
        self.resize(560, 540)
        self._ctrl = controller
        self._main = parent
        if not isinstance(getattr(controller, "temporal_markers", None), list):
            controller.temporal_markers = []
        self._build_ui()
        self._reload_table()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        add_box = QtWidgets.QGroupBox("Ajouter un marqueur")
        form = QtWidgets.QFormLayout(add_box)
        self._desc = QtWidgets.QLineEdit()
        form.addRow("Description :", self._desc)

        pos_row = QtWidgets.QHBoxLayout()
        self._pos = QtWidgets.QDoubleSpinBox()
        self._pos.setRange(0.0, max(float(getattr(self._ctrl, "total_duration_s", 0.0) or 0.0), 1e6))
        self._pos.setDecimals(2)
        self._pos.setSuffix(" s")
        self._pos.setValue(float(getattr(self._ctrl, "current_time", 0.0) or 0.0))
        pos_row.addWidget(self._pos)
        cur_btn = QtWidgets.QPushButton("Position actuelle")
        cur_btn.clicked.connect(self._use_current_position)
        pos_row.addWidget(cur_btn)
        form.addRow("Position :", pos_row)

        self._color = QtWidgets.QComboBox()
        for c in _COLORS:
            pix = QtGui.QPixmap(16, 16)
            pix.fill(QtGui.QColor(c))
            self._color.addItem(QtGui.QIcon(pix), c)
        form.addRow("Couleur :", self._color)

        add_btn = QtWidgets.QPushButton("Ajouter")
        add_btn.clicked.connect(self._add_marker)
        form.addRow(add_btn)
        layout.addWidget(add_box)

        self._table = QtWidgets.QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Description", "Position (s)", "Couleur"])
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, stretch=1)

        btns = QtWidgets.QHBoxLayout()
        goto_btn = QtWidgets.QPushButton("Aller au marqueur")
        goto_btn.clicked.connect(self._goto_selected)
        btns.addWidget(goto_btn)
        del_btn = QtWidgets.QPushButton("Supprimer")
        del_btn.clicked.connect(self._remove_selected)
        btns.addWidget(del_btn)
        clear_btn = QtWidgets.QPushButton("Tout effacer")
        clear_btn.clicked.connect(self._clear_all)
        btns.addWidget(clear_btn)
        btns.addStretch()
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    # -- Helpers --------------------------------------------------------

    def _markers(self) -> list:
        return self._ctrl.temporal_markers

    def _use_current_position(self) -> None:
        main = self._main
        t = float(getattr(self._ctrl, "current_time", 0.0) or 0.0)
        if main is not None and hasattr(main, "_viewer"):
            try:
                t = float(main._viewer.start_s)
            except Exception:
                pass
        self._pos.setValue(t)

    def _reload_table(self) -> None:
        markers = sorted(self._markers(), key=lambda m: float(m.get("position", 0.0)))
        self._table.setRowCount(len(markers))
        for r, m in enumerate(markers):
            self._table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(m.get("description", ""))))
            self._table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{float(m.get('position', 0.0)):.2f}"))
            color = m.get("color", _COLORS[0])
            citem = QtWidgets.QTableWidgetItem(str(color))
            citem.setForeground(QtGui.QColor(color))
            self._table.setItem(r, 2, citem)

    def _push_to_viewer(self) -> None:
        main = self._main
        if main is not None and hasattr(main, "_viewer"):
            try:
                main._viewer.set_temporal_markers(list(self._markers()))
            except Exception:
                logger.debug("push markers to viewer failed", exc_info=True)

    # -- Actions --------------------------------------------------------

    def _add_marker(self) -> None:
        desc = self._desc.text().strip() or f"Marqueur {len(self._markers()) + 1}"
        self._markers().append({
            "description": desc,
            "position": float(self._pos.value()),
            "color": self._color.currentText(),
        })
        self._desc.clear()
        self._reload_table()
        self._push_to_viewer()

    def _selected_marker(self) -> Optional[dict]:
        row = self._table.currentRow()
        if row < 0:
            return None
        desc_item = self._table.item(row, 0)
        pos_item = self._table.item(row, 1)
        if desc_item is None or pos_item is None:
            return None
        try:
            pos = float(pos_item.text())
        except ValueError:
            return None
        for m in self._markers():
            if (str(m.get("description", "")) == desc_item.text()
                    and abs(float(m.get("position", 0.0)) - pos) < 1e-6):
                return m
        return None

    def _on_double_click(self, _item: QtWidgets.QTableWidgetItem) -> None:
        self._goto_selected()

    def _goto_selected(self) -> None:
        m = self._selected_marker()
        if m is None:
            return
        main = self._main
        if main is not None and hasattr(main, "set_time_window"):
            try:
                dur = float(getattr(main._viewer, "duration_s", 30.0)) or 30.0
                main.set_time_window(float(m.get("position", 0.0)), dur)
            except Exception:
                logger.debug("goto marker failed", exc_info=True)

    def _remove_selected(self) -> None:
        m = self._selected_marker()
        if m is None:
            return
        try:
            self._markers().remove(m)
        except ValueError:
            return
        self._reload_table()
        self._push_to_viewer()

    def _clear_all(self) -> None:
        self._ctrl.temporal_markers = []
        self._reload_table()
        self._push_to_viewer()
