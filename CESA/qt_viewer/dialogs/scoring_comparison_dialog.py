"""Qt dialog: detailed auto-vs-manual scoring comparison.

Given the concordance DataFrame produced by
``AppController.compare_scoring`` (columns ``epoch``, ``auto``, ``manual``,
``match``), this dialog shows a summary tab (global agreement, Cohen's kappa,
per-stage accuracy), a confusion-matrix heatmap and an overlaid hypnogram view.

Replaces the Tk ``_show_detailed_comparison`` / ``_open_comparison_window``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

_STAGE_ORDER = ["W", "N1", "N2", "N3", "R", "U"]
_STAGE_Y = {"W": 5, "R": 4, "N1": 3, "N2": 2, "N3": 1, "U": 0}


def _cohens_kappa(auto: pd.Series, manual: pd.Series) -> float:
    """Cohen's kappa between two categorical series (best effort)."""
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(auto, manual))
    except Exception:
        pass
    labels = sorted(set(auto) | set(manual))
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(auto)
    if n == 0:
        return float("nan")
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for a, m in zip(auto, manual):
        cm[idx[a], idx[m]] += 1.0
    po = np.trace(cm) / n
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return (po - pe) / (1.0 - pe)


class ScoringComparisonDialog(QtWidgets.QDialog):
    """Detailed comparison between automatic and manual scoring."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        merged: pd.DataFrame,
        epoch_len: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Comparaison auto vs manuel (detaillee)")
        self.resize(960, 720)
        self._merged = merged.reset_index(drop=True)
        self._epoch_len = float(epoch_len)
        self._stages = [s for s in _STAGE_ORDER
                        if s in set(self._merged["auto"]) | set(self._merged["manual"])]
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, stretch=1)
        self._tabs.addTab(self._build_summary_tab(), "Resume")
        self._tabs.addTab(self._build_confusion_tab(), "Matrice de confusion")
        self._tabs.addTab(self._build_hypno_tab(), "Hypnogrammes")

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        save_btn = QtWidgets.QPushButton("Exporter CSV...")
        save_btn.clicked.connect(self._export_csv)
        btns.addWidget(save_btn)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    # -- Summary --------------------------------------------------------

    def _build_summary_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        n_total = len(self._merged)
        n_match = int(self._merged["match"].sum())
        acc = n_match / n_total * 100 if n_total else 0.0
        kappa = _cohens_kappa(self._merged["auto"], self._merged["manual"])

        header = QtWidgets.QLabel(
            f"<h3>Concordance globale: {n_match}/{n_total} ({acc:.1f}%)</h3>"
            f"<p>Cohen's kappa: <b>{kappa:.3f}</b> "
            f"({self._kappa_label(kappa)})</p>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(header)

        table = QtWidgets.QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(
            ["Stade (auto)", "Concordants", "Total", "Precision"]
        )
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        grp = self._merged.groupby("auto")["match"].agg(["sum", "count"])
        table.setRowCount(len(grp))
        for r, (stage, row) in enumerate(grp.iterrows()):
            s_acc = row["sum"] / row["count"] * 100 if row["count"] else 0.0
            for c, val in enumerate(
                [str(stage), str(int(row["sum"])), str(int(row["count"])), f"{s_acc:.1f}%"]
            ):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(val))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(table, stretch=1)
        return w

    @staticmethod
    def _kappa_label(kappa: float) -> str:
        if kappa != kappa:  # NaN
            return "indetermine"
        if kappa < 0.0:
            return "desaccord"
        if kappa < 0.20:
            return "tres faible"
        if kappa < 0.40:
            return "faible"
        if kappa < 0.60:
            return "modere"
        if kappa < 0.80:
            return "bon"
        return "excellent"

    # -- Confusion matrix ----------------------------------------------

    def _build_confusion_tab(self) -> QtWidgets.QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        fig = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(canvas, stretch=1)

        stages = self._stages or _STAGE_ORDER
        idx = {s: i for i, s in enumerate(stages)}
        cm = np.zeros((len(stages), len(stages)), dtype=int)
        for a, m in zip(self._merged["auto"], self._merged["manual"]):
            if a in idx and m in idx:
                cm[idx[a], idx[m]] += 1

        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(stages)))
        ax.set_yticks(range(len(stages)))
        ax.set_xticklabels(stages)
        ax.set_yticklabels(stages)
        ax.set_xlabel("Manuel")
        ax.set_ylabel("Automatique")
        ax.set_title("Matrice de confusion (nb d'epoques)")
        thresh = cm.max() / 2.0 if cm.max() else 0.5
        for i in range(len(stages)):
            for j in range(len(stages)):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        canvas.draw()
        self._cm = cm
        return w

    # -- Hypnogram overlay ---------------------------------------------

    def _build_hypno_tab(self) -> QtWidgets.QWidget:
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        fig = Figure(figsize=(10, 5), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, w)
        lay.addWidget(toolbar)
        lay.addWidget(canvas, stretch=1)

        df = self._merged.sort_values("epoch")
        epochs = df["epoch"].to_numpy() * (self._epoch_len / 3600.0)
        auto_y = [_STAGE_Y.get(s, 0) for s in df["auto"]]
        manual_y = [_STAGE_Y.get(s, 0) for s in df["manual"]]

        ax = fig.add_subplot(111)
        ax.step(epochs, auto_y, where="post", label="Auto", color="#89b4fa", alpha=0.9)
        ax.step(epochs, manual_y, where="post", label="Manuel", color="#f38ba8",
                alpha=0.7, linestyle="--")
        mism = ~df["match"].to_numpy()
        if mism.any():
            ax.scatter(
                epochs[mism], np.array(auto_y)[mism], color="#fab387",
                s=12, zorder=3, label="Desaccords",
            )
        ax.set_yticks(list(_STAGE_Y.values()))
        ax.set_yticklabels(list(_STAGE_Y.keys()))
        ax.set_xlabel("Temps (h)")
        ax.set_title("Hypnogrammes superposes")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        canvas.draw()
        return w

    # -- Export ---------------------------------------------------------

    def _export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter comparaison", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._merged.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Export", f"Comparaison exportee:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")
