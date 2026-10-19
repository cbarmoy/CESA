"""Qt dialog: SpecParam (FOOOF) aperiodic / periodic spectral fit."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

_STAGE_COLORS = {
    "W": "#f38ba8",
    "N1": "#fab387",
    "N2": "#a6e3a1",
    "N3": "#89b4fa",
    "R": "#cba6f7",
    "all": "#cdd6f4",
}


class SpecParamDialog(QtWidgets.QDialog):
    """Fit SpecParam on Welch PSD (global or by sleep stage)."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        scoring_df: Optional[pd.DataFrame] = None,
        epoch_seconds: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SpecParam (FOOOF / 1/f)")
        self.resize(1000, 720)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._scoring_df = scoring_df
        self._epoch_seconds = float(epoch_seconds)
        self._last_results: Dict[str, dict] = {}
        self._build_ui()
        if self._channels:
            self._run_fit()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Canal:"))
        self._channel_combo = QtWidgets.QComboBox()
        self._channel_combo.addItems(self._channels)
        ctrl.addWidget(self._channel_combo)

        ctrl.addWidget(QtWidgets.QLabel("Mode:"))
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems(["Global (nuit)", "Par stade"])
        if self._scoring_df is None or len(self._scoring_df) == 0:
            self._mode_combo.setCurrentIndex(0)
            self._mode_combo.setEnabled(False)
        else:
            self._mode_combo.setCurrentIndex(1)
        ctrl.addWidget(self._mode_combo)

        ctrl.addWidget(QtWidgets.QLabel("fmin:"))
        self._fmin = QtWidgets.QDoubleSpinBox()
        self._fmin.setRange(0.1, 100.0)
        self._fmin.setValue(0.5)
        ctrl.addWidget(self._fmin)

        ctrl.addWidget(QtWidgets.QLabel("fmax:"))
        self._fmax = QtWidgets.QDoubleSpinBox()
        self._fmax.setRange(1.0, 200.0)
        self._fmax.setValue(40.0)
        ctrl.addWidget(self._fmax)

        run_btn = QtWidgets.QPushButton("Calculer")
        run_btn.clicked.connect(self._run_fit)
        ctrl.addWidget(run_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        self._fig = Figure(figsize=(10, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        self._table = QtWidgets.QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels([
            "Stade", "Offset", "Exposant 1/f", "R2", "Erreur",
            "Pic CF (Hz)", "Pic PW", "Pic BW", "Info",
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setMinimumHeight(160)
        self._table.setMaximumHeight(220)
        # Texte lisible sur themes sombre/clair (sinon cases "vides")
        self._table.setStyleSheet(
            "QTableWidget { color: #CDD6F4; background: #1E1E2E; gridline-color: #45475A; }"
            "QTableWidget::item { color: #CDD6F4; padding: 2px; }"
            "QHeaderView::section { color: #CDD6F4; background: #313244; padding: 4px; }"
            "QTableWidget::item:alternate { background: #313244; }"
        )
        layout.addWidget(self._table)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        export_btn = QtWidgets.QPushButton("Exporter CSV")
        export_btn.clicked.connect(self._export_csv)
        bottom.addWidget(export_btn)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def _run_fit(self) -> None:
        from CESA.specparam_analysis import (
            SpecParamUnavailableError,
            fit_specparam_by_stage,
            fit_specparam_on_signal,
            results_to_rows,
        )

        channel = self._channel_combo.currentText()
        if channel not in self._signals:
            return
        signal, fs = self._signals[channel]
        signal = np.asarray(signal, dtype=float)
        freq_range = (float(self._fmin.value()), float(self._fmax.value()))
        by_stage = self._mode_combo.currentIndex() == 1

        try:
            if by_stage:
                results = fit_specparam_by_stage(
                    signal, fs, self._scoring_df, self._epoch_seconds, freq_range,
                )
            else:
                fit = fit_specparam_on_signal(signal, fs, freq_range)
                results = {"all": fit}
            self._last_results = results
            self._channel_used = channel
        except SpecParamUnavailableError as exc:
            self._status.setText(str(exc))
            QtWidgets.QMessageBox.warning(self, "SpecParam", str(exc))
            return
        except Exception as exc:
            logger.error("SpecParam dialog failed: %s", exc, exc_info=True)
            self._status.setText(f"Erreur: {exc}")
            return

        warnings = [f"{k}: {v.get('warning')}" for k, v in results.items() if v.get("warning")]
        ok_n = sum(1 for v in results.values() if v.get("ok"))
        self._status.setText(
            f"{ok_n}/{len(results)} fit(s) OK"
            + ((" | " + "; ".join(warnings[:2])) if warnings else "")
        )
        self._fill_table(results_to_rows(results, channel=channel))
        self._plot(results)

    def _fill_table(self, rows: List[dict]) -> None:
        self._table.setRowCount(0)
        if not rows:
            self._table.setRowCount(1)
            self._table.setItem(
                0, 0, QtWidgets.QTableWidgetItem("(aucun resultat)")
            )
            return
        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            info = ""
            if row.get("warning"):
                info = str(row.get("warning"))
            elif not row.get("ok"):
                info = "fit non valide"
            else:
                n_peaks = int(row.get("n_peaks") or 0)
                info = f"{n_peaks} pic(s)" if n_peaks else "pas de pic"
            vals = [
                str(row.get("stage", "") or ""),
                _fmt(row.get("aperiodic_offset")),
                _fmt(row.get("aperiodic_exponent")),
                _fmt(row.get("r_squared")),
                _fmt(row.get("error")),
                _fmt(row.get("peak_cf")),
                _fmt(row.get("peak_pw")),
                _fmt(row.get("peak_bw")),
                info,
            ]
            for c, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                item.setFlags(
                    item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
                )
                self._table.setItem(i, c, item)
        self._table.resizeColumnsToContents()

    def _plot(self, results: Dict[str, dict]) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        for stage, fit in results.items():
            freqs = np.asarray(fit.get("freqs", []), dtype=float)
            psd = np.asarray(fit.get("psd", []), dtype=float)
            if freqs.size == 0 or psd.size == 0:
                continue
            color = _STAGE_COLORS.get(stage, None)
            label = f"{stage}"
            if fit.get("ok") and np.isfinite(fit.get("aperiodic_exponent", float("nan"))):
                label += f" (exp={fit['aperiodic_exponent']:.2f})"
            ax.loglog(freqs, psd, color=color, alpha=0.7, label=label)
            modeled = np.asarray(fit.get("modeled_spectrum", []), dtype=float)
            if modeled.size == freqs.size:
                ax.loglog(freqs, modeled, color=color, linestyle="--", alpha=0.9)
        ax.set_xlabel("Frequence (Hz)")
        ax.set_ylabel("PSD (uV^2/Hz)")
        ax.set_title("SpecParam: PSD (plein) + fit (tirets)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

    def _export_csv(self) -> None:
        from CESA.specparam_analysis import results_to_rows

        if not self._last_results:
            QtWidgets.QMessageBox.information(self, "Export", "Aucun resultat.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter SpecParam", "specparam.csv", "CSV (*.csv)",
        )
        if not path:
            return
        rows = results_to_rows(
            self._last_results, channel=getattr(self, "_channel_used", ""),
        )
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        QtWidgets.QMessageBox.information(self, "Export", f"Ecrit: {path}")


def _fmt(val) -> str:
    if val is None or val == "":
        return "—"
    try:
        if isinstance(val, (float, int, np.floating, np.integer)):
            f = float(val)
        else:
            f = float(val)
        if not np.isfinite(f):
            return "—"
        return f"{f:.4g}"
    except (TypeError, ValueError):
        s = str(val).strip()
        return s if s else "—"
