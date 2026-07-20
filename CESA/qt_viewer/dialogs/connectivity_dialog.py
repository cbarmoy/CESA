"""Qt dialog: inter-channel connectivity (correlation / coherence / PLV / ANOVA).

Native Qt port consolidating the Tk ``_show_correlation_analysis``,
``_show_coherence_analysis``, ``_show_plv_connectivity`` and
``_show_variance_analysis`` (ANOVA) windows.

- Correlation: Pearson/Spearman matrix between channels.
- Coherence: mean magnitude-squared coherence in a frequency band.
- PLV: phase-locking value from the Hilbert analytic signal in a band.
- ANOVA: one-way ANOVA of band power across sleep stages (per channel).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtWidgets

# np.trapz was removed in NumPy 2.0; use np.trapezoid when available.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]


def _resample_to_common(signals: Dict[str, Signal], channels: List[str]):
    """Return ``(matrix[n_ch, n], fs)`` truncated to the shortest channel."""
    fs = signals[channels[0]][1]
    arrays = [np.asarray(signals[c][0], dtype=float) for c in channels]
    n = min(a.size for a in arrays)
    mat = np.vstack([a[:n] for a in arrays])
    return mat, fs


class ConnectivityDialog(QtWidgets.QDialog):
    """Tabbed inter-channel connectivity analysis."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        scoring_df: Optional[pd.DataFrame] = None,
        epoch_seconds: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connectivite inter-canaux")
        self.resize(1020, 760)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._scoring_df = scoring_df
        self._epoch = float(epoch_seconds)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Canaux:"))
        self._chan_list = QtWidgets.QListWidget()
        self._chan_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self._chan_list.addItems(self._channels)
        for i in range(min(6, len(self._channels))):
            self._chan_list.item(i).setSelected(True)
        self._chan_list.setMaximumHeight(110)
        top.addWidget(self._chan_list)

        form = QtWidgets.QFormLayout()
        self._method = QtWidgets.QComboBox()
        self._method.addItems(["Correlation (Pearson)", "Correlation (Spearman)",
                               "Coherence", "PLV", "ANOVA par stade"])
        self._method.currentIndexChanged.connect(self._sync_controls)
        form.addRow("Methode:", self._method)
        self._fmin = QtWidgets.QDoubleSpinBox()
        self._fmin.setRange(0.0, 100.0)
        self._fmin.setValue(8.0)
        form.addRow("fmin (Hz):", self._fmin)
        self._fmax = QtWidgets.QDoubleSpinBox()
        self._fmax.setRange(1.0, 200.0)
        self._fmax.setValue(13.0)
        form.addRow("fmax (Hz):", self._fmax)
        top.addLayout(form)
        run = QtWidgets.QPushButton("Calculer")
        run.clicked.connect(self._run)
        top.addWidget(run)
        top.addStretch()
        layout.addLayout(top)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(NavigationToolbar2QT(self._canvas, self))
        layout.addWidget(self._canvas, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        self._status = QtWidgets.QLabel("")
        bottom.addWidget(self._status)
        bottom.addStretch()
        exp = QtWidgets.QPushButton("Exporter CSV")
        exp.clicked.connect(self._export)
        bottom.addWidget(exp)
        layout.addLayout(bottom)
        self._last_matrix: Optional[pd.DataFrame] = None
        self._sync_controls()

    def _sync_controls(self) -> None:
        band_methods = self._method.currentIndex() >= 2
        self._fmin.setEnabled(band_methods)
        self._fmax.setEnabled(band_methods)

    def _selected_channels(self) -> List[str]:
        return [it.text() for it in self._chan_list.selectedItems()]

    def _run(self) -> None:
        channels = self._selected_channels()
        if len(channels) < 2 and self._method.currentIndex() != 4:
            QtWidgets.QMessageBox.information(self, "Connectivite", "Selectionnez au moins 2 canaux.")
            return
        idx = self._method.currentIndex()
        try:
            if idx in (0, 1):
                self._compute_correlation(channels, spearman=(idx == 1))
            elif idx == 2:
                self._compute_coherence(channels)
            elif idx == 3:
                self._compute_plv(channels)
            else:
                self._compute_anova(channels)
        except Exception as exc:
            logger.error("connectivity failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Calcul echoue:\n{exc}")

    # -- matrices -------------------------------------------------------

    def _compute_correlation(self, channels: List[str], spearman: bool) -> None:
        mat, _fs = _resample_to_common(self._signals, channels)
        if spearman:
            order = np.argsort(np.argsort(mat, axis=1), axis=1).astype(float)
            cm = np.corrcoef(order)
        else:
            cm = np.corrcoef(mat)
        self._plot_matrix(cm, channels, "Correlation", vmin=-1, vmax=1, cmap="coolwarm")

    def _compute_coherence(self, channels: List[str]) -> None:
        from scipy.signal import coherence

        mat, fs = _resample_to_common(self._signals, channels)
        lo, hi = self._fmin.value(), self._fmax.value()
        n = len(channels)
        cm = np.eye(n)
        nperseg = min(mat.shape[1], max(128, int(fs * 2)))
        for i in range(n):
            for j in range(i + 1, n):
                f, cxy = coherence(mat[i], mat[j], fs=fs, nperseg=nperseg)
                band = (f >= lo) & (f <= hi)
                val = float(np.mean(cxy[band])) if band.any() else 0.0
                cm[i, j] = cm[j, i] = val
        self._plot_matrix(cm, channels, f"Coherence [{lo:.0f}-{hi:.0f} Hz]",
                          vmin=0, vmax=1, cmap="viridis")

    def _compute_plv(self, channels: List[str]) -> None:
        from scipy.signal import butter, filtfilt, hilbert

        mat, fs = _resample_to_common(self._signals, channels)
        lo, hi = self._fmin.value(), self._fmax.value()
        nyq = fs / 2.0
        b, a = butter(4, [max(0.01, lo / nyq), min(0.99, hi / nyq)], btype="band")
        phases = np.angle(hilbert(filtfilt(b, a, mat, axis=1), axis=1))
        n = len(channels)
        cm = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                dphi = phases[i] - phases[j]
                plv = float(np.abs(np.mean(np.exp(1j * dphi))))
                cm[i, j] = cm[j, i] = plv
        self._plot_matrix(cm, channels, f"PLV [{lo:.0f}-{hi:.0f} Hz]",
                          vmin=0, vmax=1, cmap="magma")

    def _plot_matrix(self, cm, channels, title, vmin, vmax, cmap) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        im = ax.imshow(cm, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(channels)))
        ax.set_yticks(range(len(channels)))
        ax.set_xticklabels(channels, rotation=60, fontsize=7)
        ax.set_yticklabels(channels, fontsize=7)
        ax.set_title(title)
        self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._fig.tight_layout()
        self._canvas.draw()
        self._last_matrix = pd.DataFrame(cm, index=channels, columns=channels)
        self._status.setText(f"{title}: matrice {len(channels)}x{len(channels)}.")

    # -- ANOVA ----------------------------------------------------------

    def _compute_anova(self, channels: List[str]) -> None:
        from scipy.signal import welch

        if self._scoring_df is None or self._scoring_df.empty:
            QtWidgets.QMessageBox.information(
                self, "ANOVA", "Un scoring est requis pour l'ANOVA par stade.")
            return
        try:
            from scipy.stats import f_oneway
        except Exception:
            QtWidgets.QMessageBox.warning(self, "ANOVA", "scipy.stats indisponible.")
            return

        lo, hi = self._fmin.value(), self._fmax.value()
        ep = self._epoch
        rows = []
        stages = ["W", "N1", "N2", "N3", "R"]
        for ch in channels:
            sig, fs = self._signals[ch]
            sig = np.asarray(sig, dtype=float)
            ep_n = int(round(ep * fs))
            groups = {s: [] for s in stages}
            for _, row in self._scoring_df.iterrows():
                st = str(row.get("stage"))
                if st not in groups:
                    continue
                i0 = int(round(float(row.get("time", 0.0)) * fs))
                seg = sig[i0:i0 + ep_n]
                if seg.size < 16:
                    continue
                f, psd = welch(seg, fs=fs, nperseg=min(len(seg), max(64, int(fs))))
                band = (f >= lo) & (f <= hi)
                if band.any():
                    groups[st].append(float(_trapz(psd[band], f[band])))
            valid = [np.asarray(v) for v in groups.values() if len(v) >= 2]
            if len(valid) >= 2:
                fstat, pval = f_oneway(*valid)
            else:
                fstat, pval = float("nan"), float("nan")
            rows.append({
                "channel": ch, "F": fstat, "p": pval,
                **{f"mean_{s}": (np.mean(groups[s]) if groups[s] else np.nan) for s in stages},
            })
        df = pd.DataFrame(rows).set_index("channel")
        self._last_matrix = df

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        means = df[[f"mean_{s}" for s in stages]].to_numpy()
        x = np.arange(len(stages))
        for r, ch in enumerate(df.index):
            ax.plot(x, means[r], marker="o", ms=3, label=str(ch))
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_ylabel(f"Puissance [{lo:.0f}-{hi:.0f} Hz]")
        ax.set_title("Puissance de bande par stade (ANOVA)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        self._fig.tight_layout()
        self._canvas.draw()
        sig_ch = df.index[df["p"] < 0.05].tolist()
        self._status.setText(
            f"ANOVA: {len(sig_ch)} canaux significatifs (p<0.05): "
            + ", ".join(map(str, sig_ch[:8]))
        )

    def _export(self) -> None:
        if self._last_matrix is None:
            QtWidgets.QMessageBox.information(self, "Export", "Aucun resultat.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self._last_matrix.to_csv(path)
            QtWidgets.QMessageBox.information(self, "Export", f"Exporte:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")
