"""Qt dialog: PSD-by-stage analysis for the current recording (FFT or Welch).

Provides per-sleep-stage PSD curves (median/mean +/- SEM) using the scoring of
the loaded recording, with optional before/after filtering comparison. Reuses
:mod:`CESA.spectral_analysis` for the maths and
:func:`CESA.batch_engine.export_psd_results_to_csv` for CSV export.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtWidgets

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]

_STAGE_COLORS = {
    "W": "#f38ba8",
    "N1": "#fab387",
    "N2": "#a6e3a1",
    "N3": "#89b4fa",
    "R": "#cba6f7",
}


def _filter_signal(
    x: np.ndarray,
    fs: float,
    band: Optional[Tuple[float, float]] = (0.3, 40.0),
    notch_hz: Optional[float] = 50.0,
) -> np.ndarray:
    """Apply an optional band-pass + notch filter (best effort)."""
    y = np.asarray(x, dtype=float)
    try:
        if band is not None:
            from scipy.signal import butter, filtfilt

            low = max(0.0, float(band[0]))
            high = min(float(band[1]), fs / 2.0 - 1e-6)
            if high > low > 0.0:
                b, a = butter(4, [low / (fs / 2.0), high / (fs / 2.0)], btype="band")
                y = filtfilt(b, a, y)
    except Exception:
        pass
    try:
        if notch_hz and notch_hz > 0.0:
            from scipy.signal import filtfilt, iirnotch

            b, a = iirnotch(float(notch_hz) / (fs / 2.0), Q=30.0)
            y = filtfilt(b, a, y)
    except Exception:
        pass
    return y


class StagePSDDialog(QtWidgets.QDialog):
    """PSD-by-stage dialog for a single recording."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        scoring_df: Optional[pd.DataFrame],
        epoch_seconds: float = 30.0,
        initial_method: str = "fft",
        initial_before_after: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PSD par stade")
        self.resize(1040, 740)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._scoring_df = scoring_df
        self._epoch_seconds = float(epoch_seconds)
        self._last_results: Dict[str, object] = {}
        self._build_ui()
        self._method_combo.setCurrentIndex(1 if initial_method == "welch" else 0)
        self._before_after.setChecked(bool(initial_before_after))
        if self._channels:
            self._compute_and_plot()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Canal:"))
        self._channel_combo = QtWidgets.QComboBox()
        self._channel_combo.addItems(self._channels)
        ctrl.addWidget(self._channel_combo)

        ctrl.addWidget(QtWidgets.QLabel("Methode:"))
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(["FFT (magnitude)", "Welch (uV2/Hz)"])
        ctrl.addWidget(self._method_combo)

        ctrl.addWidget(QtWidgets.QLabel("fmin:"))
        self._fmin = QtWidgets.QDoubleSpinBox()
        self._fmin.setRange(0.0, 500.0)
        self._fmin.setValue(0.5)
        ctrl.addWidget(self._fmin)

        ctrl.addWidget(QtWidgets.QLabel("fmax:"))
        self._fmax = QtWidgets.QDoubleSpinBox()
        self._fmax.setRange(1.0, 1000.0)
        self._fmax.setValue(45.0)
        ctrl.addWidget(self._fmax)

        self._before_after = QtWidgets.QCheckBox("Avant/apres filtrage")
        ctrl.addWidget(self._before_after)

        self._equalize = QtWidgets.QCheckBox("Egaliser")
        self._equalize.setChecked(True)
        ctrl.addWidget(self._equalize)

        self._robust = QtWidgets.QCheckBox("Robuste")
        self._robust.setChecked(True)
        ctrl.addWidget(self._robust)

        refresh = QtWidgets.QPushButton("Actualiser")
        refresh.clicked.connect(self._compute_and_plot)
        ctrl.addWidget(refresh)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._fig = Figure(figsize=(10, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        if self._scoring_df is None or len(self._scoring_df) == 0:
            warn = QtWidgets.QLabel("Aucun scoring: tout le signal est traite comme 'W'.")
            bottom.addWidget(warn)
        bottom.addStretch()
        save_fig = QtWidgets.QPushButton("Enregistrer figure")
        save_fig.clicked.connect(self._save_figure)
        bottom.addWidget(save_fig)
        save_csv = QtWidgets.QPushButton("Exporter CSV")
        save_csv.clicked.connect(self._export_csv)
        bottom.addWidget(save_csv)
        layout.addLayout(bottom)

    def _compute_results(self, signal: np.ndarray, fs: float) -> Dict[str, object]:
        from CESA.spectral_analysis import (
            compute_stage_psd_fft_for_array,
            compute_stage_psd_welch_for_array,
        )

        fmin = self._fmin.value()
        fmax = self._fmax.value()
        equalize = self._equalize.isChecked()
        robust = self._robust.isChecked()
        if self._method_combo.currentIndex() == 1:
            return compute_stage_psd_welch_for_array(
                signal=signal, fs=fs, scoring_df=self._scoring_df,
                epoch_len=self._epoch_seconds, fmin=fmin, fmax=fmax,
                equalize_epochs=equalize, robust_stats=robust,
                band_filter=None, notch_hz=None,
            )
        return compute_stage_psd_fft_for_array(
            signal=signal, fs=fs, scoring_df=self._scoring_df,
            epoch_len=self._epoch_seconds, fmin=fmin, fmax=fmax,
            equalize_epochs=equalize, robust_stats=robust,
        )

    def _compute_and_plot(self) -> None:
        channel = self._channel_combo.currentText()
        if channel not in self._signals:
            return
        signal, fs = self._signals[channel]
        signal = np.asarray(signal, dtype=float)

        is_welch = self._method_combo.currentIndex() == 1
        ylabel = "PSD (uV^2/Hz)" if is_welch else "Magnitude FFT"

        self._fig.clear()
        if self._before_after.isChecked():
            raw_res = self._compute_results(signal, fs)
            filt_res = self._compute_results(_filter_signal(signal, fs), fs)
            ax1 = self._fig.add_subplot(121)
            ax2 = self._fig.add_subplot(122, sharex=ax1)
            self._plot_results(ax1, raw_res, f"{channel} - brut", ylabel, is_welch)
            self._plot_results(ax2, filt_res, f"{channel} - filtre", ylabel, is_welch)
            self._last_results = {"brut": raw_res, "filtre": filt_res}
        else:
            res = self._compute_results(signal, fs)
            ax = self._fig.add_subplot(111)
            self._plot_results(ax, res, f"PSD par stade - {channel}", ylabel, is_welch)
            self._last_results = {"": res}

        self._fig.tight_layout()
        self._canvas.draw()

    def _plot_results(self, ax, results, title: str, ylabel: str, log_scale: bool) -> None:
        for stage, payload in results.items():
            if not payload or len(payload) != 4:
                continue
            freqs, mean_vals, sem_vals, n_epochs = payload
            color = _STAGE_COLORS.get(stage, None)
            label = f"{stage} (n={n_epochs})"
            if log_scale:
                ax.semilogy(freqs, mean_vals, label=label, color=color, alpha=0.9)
            else:
                ax.plot(freqs, mean_vals, label=label, color=color, alpha=0.9)
            ax.fill_between(
                freqs,
                np.maximum(mean_vals - sem_vals, 1e-12) if log_scale else mean_vals - sem_vals,
                mean_vals + sem_vals,
                color=color, alpha=0.2,
            )
        ax.set_xlabel("Frequence (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def _save_figure(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches="tight")

    def _export_csv(self) -> None:
        if not self._last_results:
            QtWidgets.QMessageBox.information(self, "Export", "Aucun resultat a exporter.")
            return
        from CESA.batch_engine import export_psd_results_to_csv

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        # If before/after, export to two files; otherwise single file.
        try:
            if len(self._last_results) == 1:
                results = next(iter(self._last_results.values()))
                export_psd_results_to_csv(results, path)
            else:
                base = path[:-4] if path.lower().endswith(".csv") else path
                for key, results in self._last_results.items():
                    export_psd_results_to_csv(results, f"{base}_{key}.csv")
            QtWidgets.QMessageBox.information(self, "Export", "Export termine.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec de l'export:\n{exc}")
