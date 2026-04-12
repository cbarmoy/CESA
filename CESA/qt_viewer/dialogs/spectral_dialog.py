"""Qt dialog for spectral analysis (PSD Welch) using matplotlib Qt backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]


class SpectralAnalysisDialog(QtWidgets.QDialog):
    """Dialog displaying PSD Welch analysis with FigureCanvasQTAgg."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        current_time: float = 0.0,
        duration: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyse spectrale - PSD Welch")
        self.resize(1000, 700)
        self._signals = signals
        self._channels = channels
        self._current_time = current_time
        self._duration = duration
        self._build_ui()
        self._compute_and_plot()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Debut (s):"))
        self._start_spin = QtWidgets.QDoubleSpinBox()
        self._start_spin.setRange(0, 1e6)
        self._start_spin.setValue(self._current_time)
        ctrl.addWidget(self._start_spin)

        ctrl.addWidget(QtWidgets.QLabel("Duree (s):"))
        self._dur_spin = QtWidgets.QDoubleSpinBox()
        self._dur_spin.setRange(1, 3600)
        self._dur_spin.setValue(self._duration)
        ctrl.addWidget(self._dur_spin)

        refresh_btn = QtWidgets.QPushButton("Actualiser")
        refresh_btn.clicked.connect(self._compute_and_plot)
        ctrl.addWidget(refresh_btn)

        save_btn = QtWidgets.QPushButton("Enregistrer figure")
        save_btn.clicked.connect(self._save_figure)
        ctrl.addWidget(save_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Matplotlib canvas
        self._fig = Figure(figsize=(10, 6), dpi=100)
        self._fig.set_facecolor('#1E1E2E')
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        # Results table
        self._table = QtWidgets.QTableWidget()
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)

    def _compute_and_plot(self) -> None:
        try:
            from CESA.spectral_analysis import compute_psd_welch, EEG_BANDS
        except ImportError:
            logger.error("spectral_analysis module not available")
            return

        start = self._start_spin.value()
        dur = self._dur_spin.value()

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor('#1E1E2E')
        ax.set_xlabel("Frequence (Hz)")
        ax.set_ylabel("PSD (uV^2/Hz)")
        ax.set_title("Analyse spectrale - PSD Welch")
        ax.set_xlim(0, 50)
        ax.grid(True, alpha=0.3)

        rows = []
        for ch in self._channels:
            if ch not in self._signals:
                continue
            data_arr, fs = self._signals[ch]
            i_start = max(0, int(start * fs))
            i_end = min(len(data_arr), int((start + dur) * fs))
            if i_end - i_start < 2:
                continue
            segment = data_arr[i_start:i_end]

            try:
                freqs, psd = compute_psd_welch(segment, fs)
                ax.semilogy(freqs, psd, label=ch, alpha=0.8)

                for band_name, (f_lo, f_hi) in EEG_BANDS.items():
                    mask = (freqs >= f_lo) & (freqs < f_hi)
                    if mask.any():
                        power = float(np.sum(psd[mask]))
                        rows.append((ch, band_name, f"{power:.4f}"))
            except Exception as exc:
                logger.warning("PSD computation failed for %s: %s", ch, exc)

        ax.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

        # Fill table
        self._table.clear()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Canal", "Bande", "Puissance"])
        self._table.setRowCount(len(rows))
        for i, (ch, band, power) in enumerate(rows):
            self._table.setItem(i, 0, QtWidgets.QTableWidgetItem(ch))
            self._table.setItem(i, 1, QtWidgets.QTableWidgetItem(band))
            self._table.setItem(i, 2, QtWidgets.QTableWidgetItem(power))
        self._table.resizeColumnsToContents()

    def _save_figure(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches='tight')
