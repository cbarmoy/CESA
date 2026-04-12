"""Inspection mode panel: per-epoch features, ML probabilities, AASM rules.

Toggled from the main window, this dockable panel shows computed
features for the currently visible epoch and, if available, ML model
outputs and AASM rule decisions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class InspectionPanel(QtWidgets.QDockWidget):
    """Dockable inspection panel showing per-epoch analysis data."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Inspection", parent)
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

        container = QtWidgets.QWidget()
        self.setWidget(container)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Epoch header
        self._epoch_label = QtWidgets.QLabel("Epoch --")
        self._epoch_label.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.Bold))
        layout.addWidget(self._epoch_label)

        # Features table
        features_group = QtWidgets.QGroupBox("Features")
        features_lay = QtWidgets.QVBoxLayout(features_group)
        self._features_table = QtWidgets.QTableWidget(0, 2)
        self._features_table.setHorizontalHeaderLabels(["Feature", "Valeur"])
        self._features_table.horizontalHeader().setStretchLastSection(True)
        self._features_table.verticalHeader().setVisible(False)
        self._features_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._features_table.setMaximumHeight(200)
        features_lay.addWidget(self._features_table)
        layout.addWidget(features_group)

        # ML probabilities
        ml_group = QtWidgets.QGroupBox("Probabilites ML")
        ml_lay = QtWidgets.QVBoxLayout(ml_group)
        self._ml_table = QtWidgets.QTableWidget(0, 2)
        self._ml_table.setHorizontalHeaderLabels(["Stade", "Probabilite"])
        self._ml_table.horizontalHeader().setStretchLastSection(True)
        self._ml_table.verticalHeader().setVisible(False)
        self._ml_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._ml_table.setMaximumHeight(160)
        ml_lay.addWidget(self._ml_table)
        layout.addWidget(ml_group)

        # AASM rules
        rules_group = QtWidgets.QGroupBox("Regles AASM")
        rules_lay = QtWidgets.QVBoxLayout(rules_group)
        self._rules_list = QtWidgets.QListWidget()
        self._rules_list.setMaximumHeight(120)
        rules_lay.addWidget(self._rules_list)
        layout.addWidget(rules_group)

        # Decision summary
        self._decision_label = QtWidgets.QLabel("")
        self._decision_label.setWordWrap(True)
        layout.addWidget(self._decision_label)

        layout.addStretch()

        self._current_epoch: int = -1
        self._features_data: List[Dict[str, float]] = []
        self._ml_probs: List[Dict[str, float]] = []
        self._aasm_rules: List[str] = []
        self._epoch_decisions: List[str] = []

    # ----- public API ---------------------------------------------------

    def set_epoch(self, epoch_idx: int, epoch_len: float = 30.0) -> None:
        """Update display for the given epoch index."""
        self._current_epoch = epoch_idx
        t_start = epoch_idx * epoch_len
        h = int(t_start // 3600)
        m = int((t_start % 3600) // 60)
        s = int(t_start % 60)
        self._epoch_label.setText(
            f"Epoch {epoch_idx}  ({h:02d}:{m:02d}:{s:02d})"
        )
        self._refresh_features()
        self._refresh_ml()
        self._refresh_rules()

    def set_features(self, features: List[Dict[str, float]]) -> None:
        self._features_data = features

    def set_ml_probabilities(self, probs: List[Dict[str, float]]) -> None:
        self._ml_probs = probs

    def set_aasm_rules(self, rules: List[str]) -> None:
        self._aasm_rules = rules

    def set_epoch_decisions(self, decisions: List[str]) -> None:
        self._epoch_decisions = decisions

    # ----- internal -----------------------------------------------------

    def _refresh_features(self) -> None:
        self._features_table.setRowCount(0)
        if self._current_epoch < 0 or self._current_epoch >= len(self._features_data):
            return
        feats = self._features_data[self._current_epoch]
        self._features_table.setRowCount(len(feats))
        for i, (k, v) in enumerate(sorted(feats.items())):
            self._features_table.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self._features_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(f"{v:.4f}")
            )

    def _refresh_ml(self) -> None:
        self._ml_table.setRowCount(0)
        if self._current_epoch < 0 or self._current_epoch >= len(self._ml_probs):
            return
        probs = self._ml_probs[self._current_epoch]
        self._ml_table.setRowCount(len(probs))
        for i, (stage, p) in enumerate(
            sorted(probs.items(), key=lambda x: -x[1])
        ):
            self._ml_table.setItem(i, 0, QtWidgets.QTableWidgetItem(stage))
            bar_text = f"{p:.2%}"
            item = QtWidgets.QTableWidgetItem(bar_text)
            if p > 0.5:
                item.setForeground(QtGui.QColor("#A6E3A1"))
            self._ml_table.setItem(i, 1, item)

    def _refresh_rules(self) -> None:
        self._rules_list.clear()
        if self._current_epoch < 0:
            return
        # Show decision reason if available
        if self._current_epoch < len(self._epoch_decisions):
            reason = self._epoch_decisions[self._current_epoch]
            if reason:
                self._decision_label.setText(f"Decision: {reason}")
        for rule in self._aasm_rules:
            self._rules_list.addItem(rule)
