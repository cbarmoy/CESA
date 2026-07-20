"""Qt dialog: EEG microstate analysis (K-means).

Extracts global field power (GFP) peak topographies across the selected
channels and clusters them with K-means to obtain *k* microstate maps, then
back-fits every sample to report each map's coverage and mean duration.
Native Qt port of the Tk ``_show_microstates_analysis`` window.

Without 2D electrode coordinates, maps are shown as per-channel bar profiles
rather than scalp topographies.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtWidgets

from .batch_analysis_dialog import _BatchWorker

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]


class MicrostatesDialog(QtWidgets.QDialog):
    """K-means microstate clustering over selected channels."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Micro-etats (K-means)")
        self.resize(1000, 740)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._worker: Optional[_BatchWorker] = None
        self._result = None
        self._build_ui()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Canaux:"))
        self._chan_list = QtWidgets.QListWidget()
        self._chan_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self._chan_list.addItems(self._channels)
        for i in range(len(self._channels)):
            self._chan_list.item(i).setSelected(True)
        self._chan_list.setMaximumHeight(110)
        top.addWidget(self._chan_list)

        form = QtWidgets.QFormLayout()
        self._k = QtWidgets.QSpinBox()
        self._k.setRange(2, 10)
        self._k.setValue(4)
        form.addRow("Nb micro-etats (k):", self._k)
        self._gfp_only = QtWidgets.QCheckBox("Sur pics GFP uniquement")
        self._gfp_only.setChecked(True)
        form.addRow(self._gfp_only)
        self._max_samples = QtWidgets.QSpinBox()
        self._max_samples.setRange(1000, 500000)
        self._max_samples.setValue(50000)
        form.addRow("Echantillons max:", self._max_samples)
        top.addLayout(form)
        self._run_btn = QtWidgets.QPushButton("Calculer")
        self._run_btn.clicked.connect(self._run)
        top.addWidget(self._run_btn)
        top.addStretch()
        layout.addLayout(top)

        self._fig = Figure(figsize=(8, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(NavigationToolbar2QT(self._canvas, self))
        layout.addWidget(self._canvas, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        bottom.addWidget(self._progress, stretch=1)
        self._status = QtWidgets.QLabel("")
        bottom.addWidget(self._status)
        layout.addLayout(bottom)

    def _run(self) -> None:
        channels = [it.text() for it in self._chan_list.selectedItems()]
        if len(channels) < 2:
            QtWidgets.QMessageBox.information(self, "Micro-etats", "Selectionnez au moins 2 canaux.")
            return
        k = int(self._k.value())
        gfp_only = self._gfp_only.isChecked()
        max_samples = int(self._max_samples.value())
        signals = self._signals

        def job(progress_cb, should_stop):
            progress_cb(5.0, "Preparation des donnees...")
            fs = signals[channels[0]][1]
            arrays = [np.asarray(signals[c][0], dtype=float) for c in channels]
            n = min(a.size for a in arrays)
            data = np.vstack([a[:n] for a in arrays])  # (n_ch, n_samples)
            # Average reference + GFP
            data = data - data.mean(axis=0, keepdims=True)
            gfp = data.std(axis=0)
            if gfp_only:
                # local maxima of GFP
                peaks = np.where(
                    (gfp[1:-1] > gfp[:-2]) & (gfp[1:-1] > gfp[2:])
                )[0] + 1
                samples = data[:, peaks] if peaks.size else data
            else:
                samples = data
            samples = samples.T  # (n_samples, n_ch)
            if samples.shape[0] > max_samples:
                sel = np.random.default_rng(0).choice(
                    samples.shape[0], max_samples, replace=False
                )
                samples = samples[sel]
            # Normalize each map to unit norm (polarity-invariant clustering uses abs corr,
            # here we use standard k-means on normalized vectors as an approximation).
            norms = np.linalg.norm(samples, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            samples = samples / norms
            progress_cb(35.0, f"K-means (k={k})...")
            if should_stop():
                return None
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=k, n_init=10, random_state=0)
                labels = km.fit_predict(samples)
                maps = km.cluster_centers_
            except Exception:
                maps, labels = _kmeans_numpy(samples, k)
            progress_cb(75.0, "Back-fitting...")
            # Back-fit all samples (full resolution) to maps via max abs correlation.
            full = data.T
            fn = np.linalg.norm(full, axis=1, keepdims=True)
            fn[fn == 0] = 1.0
            full_norm = full / fn
            corr = np.abs(full_norm @ maps.T)
            assign = np.argmax(corr, axis=1)
            counts = np.bincount(assign, minlength=k).astype(float)
            coverage = counts / counts.sum() * 100.0
            # Mean duration (ms) of contiguous segments per state
            durations = {i: [] for i in range(k)}
            cur = assign[0]
            run = 1
            for v in assign[1:]:
                if v == cur:
                    run += 1
                else:
                    durations[cur].append(run)
                    cur, run = v, 1
            durations[cur].append(run)
            mean_dur_ms = {
                i: (np.mean(durations[i]) / fs * 1000.0 if durations[i] else 0.0)
                for i in range(k)
            }
            progress_cb(100.0, "Termine.")
            return {
                "maps": maps, "channels": channels, "coverage": coverage,
                "mean_dur_ms": mean_dur_ms, "k": k,
            }

        self._start_worker(job)

    def _start_worker(self, job) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._worker = _BatchWorker(job, self)
        self._worker.progress.connect(
            lambda pct, msg: (self._progress.setValue(int(pct)), self._status.setText(msg))
        )

        def _done(result):
            self._progress.setVisible(False)
            self._run_btn.setEnabled(True)
            if result:
                self._plot(result)

        def _failed(msg):
            self._progress.setVisible(False)
            self._run_btn.setEnabled(True)
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Calcul echoue:\n{msg}")

        self._worker.finished_result.connect(_done)
        self._worker.failed.connect(_failed)
        self._worker.start()

    def _plot(self, result) -> None:
        self._result = result
        maps = result["maps"]
        channels = result["channels"]
        coverage = result["coverage"]
        mean_dur = result["mean_dur_ms"]
        k = result["k"]
        self._fig.clear()
        ncol = min(k, 4)
        nrow = int(np.ceil(k / ncol))
        x = np.arange(len(channels))
        for i in range(k):
            ax = self._fig.add_subplot(nrow, ncol, i + 1)
            ax.bar(x, maps[i], color="#89b4fa")
            ax.set_title(f"MS{i+1}: {coverage[i]:.1f}% | {mean_dur[i]:.0f} ms", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=90, fontsize=6)
            ax.axhline(0, color="#6c7086", lw=0.5)
        self._fig.suptitle("Profils des micro-etats (par canal)")
        self._fig.tight_layout()
        self._canvas.draw()
        self._status.setText(
            "Couverture: " + "  ".join(f"MS{i+1}={coverage[i]:.1f}%" for i in range(k))
        )


def _kmeans_numpy(samples: np.ndarray, k: int, iters: int = 50):
    """Minimal K-means fallback when scikit-learn is unavailable."""
    rng = np.random.default_rng(0)
    centers = samples[rng.choice(samples.shape[0], k, replace=False)]
    labels = np.zeros(samples.shape[0], dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            pts = samples[labels == j]
            if len(pts):
                centers[j] = pts.mean(axis=0)
    return centers, labels
