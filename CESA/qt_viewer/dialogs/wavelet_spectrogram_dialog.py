"""Qt dialog: Morlet wavelet spectrogram (time-frequency) with hypnogram overlay.

Port of the old Tk ``_show_wavelet_spectrogram_before_after`` to the Qt viewer.
Computes a Morlet TFR for the selected time window using
:func:`mne.time_frequency.tfr_array_morlet`, displayed in dB with a percentile
color scale, comparing raw vs filtered signal, with the hypnogram overlaid.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6 import QtWidgets

from .stage_psd_dialog import _filter_signal

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]
_STAGE_DEPTH = {"W": 0.0, "N1": 1.0, "N2": 2.0, "N3": 3.0, "R": 2.5}


class WaveletSpectrogramDialog(QtWidgets.QDialog):
    """Morlet wavelet spectrogram dialog (raw vs filtered)."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        scoring_df: Optional[pd.DataFrame],
        current_time: float = 0.0,
        duration: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectrogramme ondelettes (Morlet)")
        self.resize(1100, 760)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._scoring_df = scoring_df
        self._current_time = float(current_time)
        self._duration = float(duration)
        self._build_ui()
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

        ctrl.addWidget(QtWidgets.QLabel("Debut (s):"))
        self._start = QtWidgets.QDoubleSpinBox()
        self._start.setRange(0.0, 1e6)
        self._start.setValue(self._current_time)
        ctrl.addWidget(self._start)

        ctrl.addWidget(QtWidgets.QLabel("Duree (s):"))
        self._dur = QtWidgets.QDoubleSpinBox()
        self._dur.setRange(5.0, 600.0)
        self._dur.setValue(max(30.0, self._duration))
        ctrl.addWidget(self._dur)

        ctrl.addWidget(QtWidgets.QLabel("fmax (Hz):"))
        self._fmax = QtWidgets.QDoubleSpinBox()
        self._fmax.setRange(5.0, 100.0)
        self._fmax.setValue(40.0)
        ctrl.addWidget(self._fmax)

        refresh = QtWidgets.QPushButton("Actualiser")
        refresh.clicked.connect(self._compute_and_plot)
        ctrl.addWidget(refresh)
        ctrl.addStretch()
        save = QtWidgets.QPushButton("Enregistrer figure")
        save.clicked.connect(self._save_figure)
        ctrl.addWidget(save)
        layout.addLayout(ctrl)

        self._fig = Figure(figsize=(11, 7), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

    def _compute_tfr(self, x: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
        from mne.time_frequency import tfr_array_morlet

        n_cycles = np.maximum(freqs / 2.0, 1.0)
        arr = x[np.newaxis, np.newaxis, :]
        power = tfr_array_morlet(
            arr, sfreq=fs, freqs=freqs, n_cycles=n_cycles, output="power"
        )
        return power[0, 0]

    def _plot_panel(self, ax, t, freqs, power, title: str) -> None:
        eps = np.finfo(float).eps
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        power_db = 10.0 * np.log10(power + eps)
        vals = power_db[np.isfinite(power_db)]
        if vals.size == 0:
            vmin, vmax = -120.0, -60.0
        else:
            vmin, vmax = np.percentile(vals, [5, 99])
        im = ax.pcolormesh(t, freqs, power_db, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("Frequence (Hz)")
        self._fig.colorbar(im, ax=ax, label="Power (dB)")
        self._overlay_hypnogram(ax, t)

    def _overlay_hypnogram(self, ax, t) -> None:
        if self._scoring_df is None or len(self._scoring_df) == 0:
            return
        t0, t1 = float(t[0]), float(t[-1])
        df = self._scoring_df
        mask = (df["time"] >= t0) & (df["time"] <= t1)
        sub = df.loc[mask]
        if sub.empty:
            return
        times = sub["time"].values
        depths = [_STAGE_DEPTH.get(str(s).upper(), np.nan) for s in sub["stage"].values]
        ax2 = ax.twinx()
        ax2.step(times, depths, where="post", color="white", linewidth=1.0, alpha=0.7)
        ax2.set_ylim(3.5, -0.5)
        ax2.set_yticks([0, 1, 2, 2.5, 3])
        ax2.set_yticklabels(["W", "N1", "N2", "R", "N3"], fontsize=7)

    def _compute_and_plot(self) -> None:
        channel = self._channel_combo.currentText()
        if channel not in self._signals:
            return
        signal, fs = self._signals[channel]
        signal = np.asarray(signal, dtype=float)

        start_s = max(0.0, self._start.value())
        dur_s = max(5.0, self._dur.value())
        i0 = int(max(0, min(len(signal) - 2, start_s * fs)))
        i1 = int(max(i0 + 2, min(len(signal), (start_s + dur_s) * fs)))
        t = np.arange(i0, i1) / fs
        seg = signal[i0:i1]
        if seg.size < 8:
            QtWidgets.QMessageBox.information(self, "Spectrogramme", "Fenetre trop courte.")
            return

        freqs = np.linspace(0.5, max(5.0, self._fmax.value()), 60)

        self._fig.clear()
        try:
            p_raw = self._compute_tfr(seg, fs, freqs)
            p_filt = self._compute_tfr(_filter_signal(seg, fs), fs, freqs)
        except Exception as exc:
            logger.error("TFR computation failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.warning(self, "Spectrogramme", f"Echec du calcul:\n{exc}")
            return

        ax1 = self._fig.add_subplot(211)
        ax2 = self._fig.add_subplot(212, sharex=ax1)
        self._plot_panel(ax1, t, freqs, p_raw, f"{channel} - brut")
        self._plot_panel(ax2, t, freqs, p_filt, f"{channel} - filtre")
        ax2.set_xlabel("Temps (s)")
        self._fig.tight_layout()
        self._canvas.draw()

    def _save_figure(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches="tight")
