"""Qt dialog: standalone entropy analysis (MSE and renormalized entropy).

Computes multiscale entropy (MSE, per channel and optionally per sleep stage)
and Issartel renormalized entropy (REN) for the loaded recording, reusing
:mod:`CESA.entropy`. Computation runs in a background thread.

Standalone counterpart to the Tk ``_show_multiscale_entropy`` /
``_show_renormalized_entropy`` windows (REN/MSE were previously only reachable
through the group-analysis dialog).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtWidgets

from .batch_analysis_dialog import _BatchWorker

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

_STAGE_COLORS = {
    "W": "#f38ba8", "N1": "#fab387", "N2": "#a6e3a1",
    "N3": "#89b4fa", "R": "#cba6f7",
}


def _stage_signal(
    signal: np.ndarray, fs: float, scoring_df: Optional[pd.DataFrame],
    epoch_len: float, stage: str,
) -> np.ndarray:
    """Concatenate the samples that belong to *stage* according to scoring."""
    if scoring_df is None or scoring_df.empty or "stage" not in scoring_df.columns:
        return signal
    n = len(signal)
    ep_samples = int(round(epoch_len * fs))
    if ep_samples <= 0:
        return signal
    chunks = []
    for _, row in scoring_df.iterrows():
        if str(row.get("stage")) != stage:
            continue
        t = float(row.get("time", 0.0))
        i0 = int(round(t * fs))
        i1 = min(i0 + ep_samples, n)
        if 0 <= i0 < i1:
            chunks.append(signal[i0:i1])
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks)


class EntropyAnalysisDialog(QtWidgets.QDialog):
    """Standalone MSE + renormalized entropy dialog."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        scoring_df: Optional[pd.DataFrame] = None,
        epoch_seconds: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Entropie (MSE / Renormalisee)")
        self.resize(1020, 760)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._scoring_df = scoring_df
        self._epoch = float(epoch_seconds)
        self._worker: Optional[_BatchWorker] = None
        self._mse_result = None
        self._ren_rows: List[dict] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, stretch=1)
        self._tabs.addTab(self._build_mse_tab(), "Multiscale (MSE)")
        self._tabs.addTab(self._build_ren_tab(), "Renormalisee (REN)")

        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

    # -- channel picker -------------------------------------------------

    def _make_channel_list(self) -> QtWidgets.QListWidget:
        lw = QtWidgets.QListWidget()
        lw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        lw.addItems(self._channels)
        if self._channels:
            lw.item(0).setSelected(True)
        lw.setMaximumWidth(220)
        return lw

    @staticmethod
    def _selected(lw: QtWidgets.QListWidget) -> List[str]:
        return [it.text() for it in lw.selectedItems()]

    # -- MSE tab --------------------------------------------------------

    def _build_mse_tab(self) -> QtWidgets.QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Canaux:"))
        self._mse_channels = self._make_channel_list()
        side.addWidget(self._mse_channels)

        form = QtWidgets.QFormLayout()
        self._mse_m = QtWidgets.QSpinBox()
        self._mse_m.setRange(1, 10)
        self._mse_m.setValue(2)
        form.addRow("m (embedding):", self._mse_m)
        self._mse_r = QtWidgets.QDoubleSpinBox()
        self._mse_r.setRange(0.01, 2.0)
        self._mse_r.setSingleStep(0.05)
        self._mse_r.setValue(0.20)
        form.addRow("r (tolerance):", self._mse_r)
        self._mse_scales = QtWidgets.QSpinBox()
        self._mse_scales.setRange(2, 40)
        self._mse_scales.setValue(20)
        form.addRow("Echelles max:", self._mse_scales)
        self._mse_per_stage = QtWidgets.QCheckBox("Par stade de sommeil")
        self._mse_per_stage.setEnabled(self._scoring_df is not None and not self._scoring_df.empty)
        form.addRow(self._mse_per_stage)
        side.addLayout(form)

        self._mse_run = QtWidgets.QPushButton("Calculer MSE")
        self._mse_run.clicked.connect(self._run_mse)
        side.addWidget(self._mse_run)
        self._mse_export = QtWidgets.QPushButton("Exporter CSV")
        self._mse_export.clicked.connect(self._export_mse)
        side.addWidget(self._mse_export)
        side.addStretch()
        outer.addLayout(side)

        right = QtWidgets.QVBoxLayout()
        self._mse_fig = Figure(figsize=(8, 6), dpi=100)
        self._mse_canvas = FigureCanvasQTAgg(self._mse_fig)
        right.addWidget(NavigationToolbar2QT(self._mse_canvas, w))
        right.addWidget(self._mse_canvas, stretch=1)
        outer.addLayout(right, stretch=1)
        return w

    def _run_mse(self) -> None:
        channels = self._selected(self._mse_channels)
        if not channels:
            QtWidgets.QMessageBox.information(self, "MSE", "Selectionnez au moins un canal.")
            return
        m = int(self._mse_m.value())
        r = float(self._mse_r.value())
        max_scale = int(self._mse_scales.value())
        per_stage = self._mse_per_stage.isChecked()
        signals = self._signals
        scoring = self._scoring_df
        epoch = self._epoch

        def job(progress_cb, should_stop):
            from CESA.entropy import (
                MultiscaleEntropyConfig,
                compute_multiscale_entropy,
            )

            cfg = MultiscaleEntropyConfig(m=m, r=r, scales=tuple(range(1, max_scale + 1)))
            out: Dict[str, Dict[int, float]] = {}
            stages = ["W", "N1", "N2", "N3", "R"] if per_stage else [None]
            total = len(channels) * len(stages)
            done = 0
            for ch in channels:
                sig, fs = signals[ch]
                sig = np.asarray(sig, dtype=float)
                for st in stages:
                    if should_stop():
                        return out
                    data = sig if st is None else _stage_signal(sig, fs, scoring, epoch, st)
                    label = ch if st is None else f"{ch} [{st}]"
                    done += 1
                    progress_cb(done / total * 100.0, f"MSE {label}...")
                    if data.size < (m + 2):
                        continue
                    try:
                        res = compute_multiscale_entropy(data, fs, [ch], config=cfg)
                        out[label] = dict(res.channel_entropy.get(ch, res.entropy_by_scale))
                    except Exception as exc:
                        logger.warning("MSE failed for %s: %s", label, exc)
            return out

        self._start_worker(job, self._on_mse_done, self._mse_run)

    def _on_mse_done(self, result) -> None:
        self._mse_result = result or {}
        self._mse_fig.clear()
        ax = self._mse_fig.add_subplot(111)
        for label, by_scale in self._mse_result.items():
            if not by_scale:
                continue
            scales = sorted(by_scale.keys())
            vals = [by_scale[s] for s in scales]
            color = None
            for st, c in _STAGE_COLORS.items():
                if label.endswith(f"[{st}]"):
                    color = c
            ax.plot(scales, vals, marker="o", ms=3, label=label, color=color, alpha=0.9)
        ax.set_xlabel("Echelle (tau)")
        ax.set_ylabel("Sample entropy")
        ax.set_title("Multiscale entropy")
        ax.grid(True, alpha=0.3)
        if self._mse_result:
            ax.legend(fontsize=7, ncol=2)
        self._mse_fig.tight_layout()
        self._mse_canvas.draw()
        self._status.setText(f"MSE: {len(self._mse_result)} courbe(s).")

    def _export_mse(self) -> None:
        if not self._mse_result:
            QtWidgets.QMessageBox.information(self, "Export", "Aucun resultat MSE.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter MSE", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame(self._mse_result)
            df.index.name = "scale"
            df.to_csv(path)
            QtWidgets.QMessageBox.information(self, "Export", f"MSE exporte:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")

    # -- REN tab --------------------------------------------------------

    def _build_ren_tab(self) -> QtWidgets.QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Canaux:"))
        self._ren_channels = self._make_channel_list()
        side.addWidget(self._ren_channels)

        form = QtWidgets.QFormLayout()
        self._ren_window = QtWidgets.QDoubleSpinBox()
        self._ren_window.setRange(0.5, 60.0)
        self._ren_window.setValue(4.0)
        self._ren_window.setSuffix(" s")
        form.addRow("Fenetre:", self._ren_window)
        self._ren_overlap = QtWidgets.QDoubleSpinBox()
        self._ren_overlap.setRange(0.0, 0.95)
        self._ren_overlap.setSingleStep(0.1)
        self._ren_overlap.setValue(0.5)
        form.addRow("Recouvrement:", self._ren_overlap)
        self._ren_psi = QtWidgets.QComboBox()
        self._ren_psi.addItems(["powerlaw", "identity", "log", "adaptive"])
        form.addRow("Psi:", self._ren_psi)
        self._ren_per_stage = QtWidgets.QCheckBox("Par stade de sommeil")
        self._ren_per_stage.setEnabled(self._scoring_df is not None and not self._scoring_df.empty)
        form.addRow(self._ren_per_stage)
        side.addLayout(form)

        self._ren_run = QtWidgets.QPushButton("Calculer REN")
        self._ren_run.clicked.connect(self._run_ren)
        side.addWidget(self._ren_run)
        self._ren_export = QtWidgets.QPushButton("Exporter CSV")
        self._ren_export.clicked.connect(self._export_ren)
        side.addWidget(self._ren_export)
        side.addStretch()
        outer.addLayout(side)

        right = QtWidgets.QVBoxLayout()
        self._ren_fig = Figure(figsize=(8, 6), dpi=100)
        self._ren_canvas = FigureCanvasQTAgg(self._ren_fig)
        right.addWidget(NavigationToolbar2QT(self._ren_canvas, w))
        right.addWidget(self._ren_canvas, stretch=1)
        outer.addLayout(right, stretch=1)
        return w

    def _run_ren(self) -> None:
        channels = self._selected(self._ren_channels)
        if not channels:
            QtWidgets.QMessageBox.information(self, "REN", "Selectionnez au moins un canal.")
            return
        window = float(self._ren_window.value())
        overlap = float(self._ren_overlap.value())
        psi = self._ren_psi.currentText()
        per_stage = self._ren_per_stage.isChecked()
        signals = self._signals
        scoring = self._scoring_df
        epoch = self._epoch

        def job(progress_cb, should_stop):
            from CESA.entropy import RenormalizedEntropyConfig, compute_renormalized_entropy

            cfg = RenormalizedEntropyConfig(
                window_length=window, overlap=overlap, psi_name=psi,
            )
            # Multichannel REN: stack selected channels (common fs assumed).
            fs = signals[channels[0]][1]
            stages = ["W", "N1", "N2", "N3", "R"] if per_stage else [None]
            rows = []
            for i, st in enumerate(stages):
                if should_stop():
                    return rows
                progress_cb((i + 1) / len(stages) * 100.0,
                            f"REN {st or 'global'}...")
                arrays = []
                for ch in channels:
                    sig, _fs = signals[ch]
                    sig = np.asarray(sig, dtype=float)
                    arr = sig if st is None else _stage_signal(sig, _fs, scoring, epoch, st)
                    arrays.append(arr)
                min_len = min((a.size for a in arrays), default=0)
                if min_len < int(window * fs) * 2:
                    continue
                data = np.vstack([a[:min_len] for a in arrays])
                try:
                    res = compute_renormalized_entropy(data, fs, channels, config=cfg)
                    rows.append({
                        "groupe": st or "global",
                        "entropy_nats": res.entropy_nats,
                        "entropy_bits": res.entropy_bits,
                        "n_eigenvalues": int(np.asarray(res.psi_eigenvalues).size),
                        "eigenvalues": np.asarray(res.psi_eigenvalues).tolist(),
                    })
                except Exception as exc:
                    logger.warning("REN failed for %s: %s", st, exc)
            return rows

        self._start_worker(job, self._on_ren_done, self._ren_run)

    def _on_ren_done(self, rows) -> None:
        self._ren_rows = rows or []
        self._ren_fig.clear()
        if not self._ren_rows:
            self._status.setText("REN: aucun resultat (signal trop court ?).")
            self._ren_canvas.draw()
            return
        ax1 = self._ren_fig.add_subplot(121)
        ax2 = self._ren_fig.add_subplot(122)
        groups = [r["groupe"] for r in self._ren_rows]
        bits = [r["entropy_bits"] for r in self._ren_rows]
        colors = [_STAGE_COLORS.get(g, "#89b4fa") for g in groups]
        ax1.bar(groups, bits, color=colors)
        ax1.set_ylabel("Entropie (bits)")
        ax1.set_title("Entropie renormalisee")
        ax1.grid(True, alpha=0.3, axis="y")
        for r in self._ren_rows:
            eigs = sorted(r["eigenvalues"], reverse=True)
            ax2.plot(eigs, marker=".", label=r["groupe"],
                     color=_STAGE_COLORS.get(r["groupe"], None))
        ax2.set_xlabel("Indice")
        ax2.set_ylabel("Valeur propre psi")
        ax2.set_title("Spectre")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7)
        self._ren_fig.tight_layout()
        self._ren_canvas.draw()
        txt = "  |  ".join(f"{r['groupe']}: {r['entropy_bits']:.3f} bits" for r in self._ren_rows)
        self._status.setText(f"REN: {txt}")

    def _export_ren(self) -> None:
        if not self._ren_rows:
            QtWidgets.QMessageBox.information(self, "Export", "Aucun resultat REN.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter REN", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame([
                {k: v for k, v in r.items() if k != "eigenvalues"}
                for r in self._ren_rows
            ])
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Export", f"REN exporte:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")

    # -- worker plumbing ------------------------------------------------

    def _start_worker(self, job, on_done, run_btn) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._progress.setVisible(True)
        self._progress.setValue(0)
        run_btn.setEnabled(False)
        self._worker = _BatchWorker(job, self)
        self._worker.progress.connect(self._on_progress)

        def _done(result):
            self._progress.setVisible(False)
            run_btn.setEnabled(True)
            on_done(result)

        def _failed(msg):
            self._progress.setVisible(False)
            run_btn.setEnabled(True)
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Calcul echoue:\n{msg}")

        self._worker.finished_result.connect(_done)
        self._worker.failed.connect(_failed)
        self._worker.start()

    def _on_progress(self, pct: float, msg: str) -> None:
        self._progress.setValue(int(pct))
        self._status.setText(msg)
