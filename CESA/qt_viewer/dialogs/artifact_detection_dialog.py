"""Qt dialog: artifact detection (muscular / ocular / cardiac).

Computes per-epoch band-power / amplitude metrics on a chosen channel and flags
epochs whose metric exceeds a robust threshold (median + k*MAD). Flagged
segments are listed and can be navigated to in the viewer. Native Qt port of
the Tk ``_show_artifact_detection`` window.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtWidgets

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

# (label -> (band_hz, description))
_ARTIFACT_TYPES = {
    "Musculaire (EMG)": ((20.0, 90.0), "Puissance haute frequence elevee"),
    "Oculaire (EOG)": ((0.3, 4.0), "Oscillations lentes de forte amplitude"),
    "Cardiaque (ECG)": ((5.0, 40.0), "Composantes type QRS / amplitude crete"),
}


def _bandpower_epoch(x: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return float("nan")
    try:
        from scipy.signal import welch

        nperseg = min(len(x), max(64, int(fs)))
        freqs, psd = welch(x, fs=fs, nperseg=nperseg)
        lo, hi = band
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            return float("nan")
        return float(_trapz(psd[mask], freqs[mask]))
    except Exception:
        return float(np.var(x))


class ArtifactDetectionDialog(QtWidgets.QDialog):
    """Detect artifacts per epoch on a single channel."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        epoch_seconds: float = 5.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detection d'artefacts")
        self.resize(960, 700)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._main = parent
        self._epoch = float(epoch_seconds)
        self._flagged: List[Tuple[float, float, float]] = []  # (t_start, t_end, value)
        self._build_ui()
        self._run()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Canal:"))
        self._channel = QtWidgets.QComboBox()
        self._channel.addItems(self._channels)
        top.addWidget(self._channel)
        top.addWidget(QtWidgets.QLabel("Type:"))
        self._type = QtWidgets.QComboBox()
        self._type.addItems(list(_ARTIFACT_TYPES.keys()))
        top.addWidget(self._type)
        top.addWidget(QtWidgets.QLabel("Fenetre:"))
        self._win = QtWidgets.QDoubleSpinBox()
        self._win.setRange(1.0, 30.0)
        self._win.setValue(self._epoch)
        self._win.setSuffix(" s")
        top.addWidget(self._win)
        top.addWidget(QtWidgets.QLabel("Seuil (MAD):"))
        self._k = QtWidgets.QDoubleSpinBox()
        self._k.setRange(1.0, 10.0)
        self._k.setSingleStep(0.5)
        self._k.setValue(3.5)
        top.addWidget(self._k)
        run = QtWidgets.QPushButton("Detecter")
        run.clicked.connect(self._run)
        top.addWidget(run)
        top.addStretch()
        layout.addLayout(top)

        split = QtWidgets.QSplitter()
        self._table = QtWidgets.QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Debut (s)", "Fin (s)", "Valeur"])
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemDoubleClicked.connect(self._goto)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        split.addWidget(self._table)

        plot_w = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_w)
        self._fig = Figure(figsize=(6, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        pv.addWidget(NavigationToolbar2QT(self._canvas, plot_w))
        pv.addWidget(self._canvas, stretch=1)
        split.addWidget(plot_w)
        split.setSizes([360, 600])
        layout.addWidget(split, stretch=1)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

    def _run(self) -> None:
        ch = self._channel.currentText()
        if ch not in self._signals:
            return
        sig, fs = self._signals[ch]
        sig = np.asarray(sig, dtype=float)
        band = _ARTIFACT_TYPES[self._type.currentText()][0]
        win_s = float(self._win.value())
        ep = max(1, int(round(win_s * fs)))
        n_epochs = len(sig) // ep
        if n_epochs < 3:
            self._status.setText("Signal trop court.")
            return

        times = []
        values = []
        for i in range(n_epochs):
            seg = sig[i * ep:(i + 1) * ep]
            times.append(i * win_s)
            values.append(_bandpower_epoch(seg, fs, band))
        values = np.asarray(values, dtype=float)
        times = np.asarray(times, dtype=float)

        finite = values[np.isfinite(values)]
        med = float(np.median(finite)) if finite.size else 0.0
        mad = float(np.median(np.abs(finite - med))) if finite.size else 0.0
        thresh = med + float(self._k.value()) * (mad * 1.4826 if mad > 0 else (np.std(finite) or 1.0))

        self._flagged = [
            (float(t), float(t + win_s), float(v))
            for t, v in zip(times, values)
            if np.isfinite(v) and v > thresh
        ]

        self._table.setRowCount(len(self._flagged))
        for r, (t0, t1, v) in enumerate(self._flagged):
            self._table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{t0:.1f}"))
            self._table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{t1:.1f}"))
            self._table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{v:.4g}"))

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.plot(times, values, color="#89b4fa", lw=0.8, label="Metrique")
        ax.axhline(thresh, color="#f38ba8", ls="--", label="Seuil")
        if self._flagged:
            ft = [f[0] for f in self._flagged]
            fv = [f[2] for f in self._flagged]
            ax.scatter(ft, fv, color="#fab387", s=14, zorder=3, label="Artefacts")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Puissance bande")
        ax.set_title(f"{self._type.currentText()} - {ch}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

        pct = len(self._flagged) / n_epochs * 100 if n_epochs else 0.0
        self._status.setText(
            f"{len(self._flagged)} / {n_epochs} fenetres marquees ({pct:.1f}%)."
        )

    def _goto(self, _item) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._flagged):
            return
        t0 = self._flagged[row][0]
        main = self._main
        if main is not None and hasattr(main, "set_time_window"):
            try:
                dur = float(getattr(main._viewer, "duration_s", 30.0)) or 30.0
                main.set_time_window(t0, dur)
            except Exception:
                logger.debug("goto artifact failed", exc_info=True)
