"""Qt dialog: per-channel temporal-domain statistics.

Computes amplitude statistics (mean, RMS, min, max, peak-to-peak, std,
variance, skewness, kurtosis, zero-crossing rate) per channel, optionally
restricted to a sleep stage. Replaces the Tk ``_show_temporal_analysis``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtWidgets

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

_METRICS = [
    ("Moyenne", "mean"),
    ("RMS", "rms"),
    ("Min", "min"),
    ("Max", "max"),
    ("Crete-crete", "ptp"),
    ("Ecart-type", "std"),
    ("Variance", "var"),
    ("Asymetrie", "skew"),
    ("Kurtosis", "kurtosis"),
    ("Taux passage zero", "zcr"),
]


def _stage_signal(signal, fs, scoring_df, epoch_len, stage):
    if scoring_df is None or scoring_df.empty or "stage" not in scoring_df.columns:
        return signal
    n = len(signal)
    ep = int(round(epoch_len * fs))
    if ep <= 0:
        return signal
    chunks = []
    for _, row in scoring_df.iterrows():
        if str(row.get("stage")) != stage:
            continue
        i0 = int(round(float(row.get("time", 0.0)) * fs))
        i1 = min(i0 + ep, n)
        if 0 <= i0 < i1:
            chunks.append(signal[i0:i1])
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=float)


def _channel_stats(x: np.ndarray, fs: float) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for _, k in _METRICS}
    mean = float(np.mean(x))
    std = float(np.std(x))
    centered = x - mean
    skew = float(np.mean(centered ** 3) / (std ** 3)) if std > 0 else 0.0
    kurt = float(np.mean(centered ** 4) / (std ** 4) - 3.0) if std > 0 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.signbit(centered))))) * fs
    return {
        "mean": mean,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "std": std,
        "var": float(np.var(x)),
        "skew": skew,
        "kurtosis": kurt,
        "zcr": zcr,
    }


class TemporalAnalysisDialog(QtWidgets.QDialog):
    """Per-channel temporal statistics table + bar chart."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        scoring_df: Optional[pd.DataFrame] = None,
        epoch_seconds: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyse temporelle")
        self.resize(980, 700)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._scoring_df = scoring_df
        self._epoch = float(epoch_seconds)
        self._df: Optional[pd.DataFrame] = None
        self._build_ui()
        self._compute()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Stade:"))
        self._stage_combo = QtWidgets.QComboBox()
        self._stage_combo.addItems(["Tous", "W", "N1", "N2", "N3", "R"])
        self._stage_combo.setEnabled(self._scoring_df is not None and not self._scoring_df.empty)
        top.addWidget(self._stage_combo)
        top.addWidget(QtWidgets.QLabel("Metrique (graphe):"))
        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItems([label for label, _ in _METRICS])
        self._metric_combo.currentIndexChanged.connect(self._plot)
        top.addWidget(self._metric_combo)
        refresh = QtWidgets.QPushButton("Actualiser")
        refresh.clicked.connect(self._compute)
        top.addWidget(refresh)
        top.addStretch()
        layout.addLayout(top)

        split = QtWidgets.QSplitter()
        self._table = QtWidgets.QTableWidget(0, len(_METRICS) + 1)
        self._table.setHorizontalHeaderLabels(["Canal"] + [label for label, _ in _METRICS])
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        split.addWidget(self._table)

        plot_w = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_w)
        self._fig = Figure(figsize=(5, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        pv.addWidget(NavigationToolbar2QT(self._canvas, plot_w))
        pv.addWidget(self._canvas, stretch=1)
        split.addWidget(plot_w)
        split.setSizes([560, 420])
        layout.addWidget(split, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        exp = QtWidgets.QPushButton("Exporter CSV")
        exp.clicked.connect(self._export)
        bottom.addWidget(exp)
        layout.addLayout(bottom)

    def _compute(self) -> None:
        stage = self._stage_combo.currentText()
        rows = []
        for ch in self._channels:
            sig, fs = self._signals[ch]
            sig = np.asarray(sig, dtype=float)
            if stage != "Tous":
                sig = _stage_signal(sig, fs, self._scoring_df, self._epoch, stage)
            stats = _channel_stats(sig, fs)
            stats["channel"] = ch
            rows.append(stats)
        self._df = pd.DataFrame(rows).set_index("channel")
        self._fill_table()
        self._plot()

    def _fill_table(self) -> None:
        if self._df is None:
            return
        self._table.setRowCount(len(self._df))
        for r, (ch, row) in enumerate(self._df.iterrows()):
            self._table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(ch)))
            for c, (_label, key) in enumerate(_METRICS):
                val = row[key]
                self._table.setItem(r, c + 1, QtWidgets.QTableWidgetItem(f"{val:.4g}"))
        self._table.resizeColumnsToContents()

    def _plot(self) -> None:
        if self._df is None or self._df.empty:
            return
        label, key = _METRICS[self._metric_combo.currentIndex()]
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.bar(self._df.index.astype(str), self._df[key], color="#89b4fa")
        ax.set_ylabel(label)
        ax.set_title(f"{label} par canal")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        self._fig.tight_layout()
        self._canvas.draw()

    def _export(self) -> None:
        if self._df is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self._df.to_csv(path)
            QtWidgets.QMessageBox.information(self, "Export", f"Exporte:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")
