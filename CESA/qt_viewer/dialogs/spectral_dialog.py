"""Qt dialog for spectral analysis (PSD Welch / FFT) using matplotlib Qt backend."""

from __future__ import annotations

import csv
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]


class SpectralAnalysisDialog(QtWidgets.QDialog):
    """Dialog displaying PSD (Welch or FFT) analysis with FigureCanvasQTAgg."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        signals: Dict[str, Signal],
        channels: List[str],
        current_time: float = 0.0,
        duration: float = 30.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyse spectrale - PSD")
        self.resize(1100, 720)
        self._signals = signals
        self._channels = [c for c in channels if c in signals] or list(signals.keys())
        self._current_time = current_time
        self._duration = duration
        self._rows: List[Tuple[str, str, float]] = []
        self._build_ui()
        self._compute_and_plot()

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QtWidgets.QHBoxLayout(self)

        # Left column: channel selection
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Canaux:"))
        self._channel_list = QtWidgets.QListWidget()
        for ch in self._channels:
            it = QtWidgets.QListWidgetItem(ch)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked)
            self._channel_list.addItem(it)
        self._channel_list.setMaximumWidth(180)
        left.addWidget(self._channel_list, stretch=1)
        layout.addLayout(left)

        # Right column: controls + canvas + table
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, stretch=1)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Methode:"))
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(["Welch (uV2/Hz)", "FFT (magnitude)"])
        ctrl.addWidget(self._method_combo)

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

        ctrl.addWidget(QtWidgets.QLabel("nperseg (s):"))
        self._nperseg_spin = QtWidgets.QDoubleSpinBox()
        self._nperseg_spin.setRange(0.5, 30.0)
        self._nperseg_spin.setValue(4.0)
        ctrl.addWidget(self._nperseg_spin)

        refresh_btn = QtWidgets.QPushButton("Actualiser")
        refresh_btn.clicked.connect(self._compute_and_plot)
        ctrl.addWidget(refresh_btn)
        ctrl.addStretch()
        right.addLayout(ctrl)

        self._fig = Figure(figsize=(10, 6), dpi=100)
        self._fig.set_facecolor('#1E1E2E')
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        right.addWidget(self._toolbar)
        right.addWidget(self._canvas, stretch=1)

        self._table = QtWidgets.QTableWidget()
        self._table.setMaximumHeight(200)
        right.addWidget(self._table)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        save_btn = QtWidgets.QPushButton("Enregistrer figure")
        save_btn.clicked.connect(self._save_figure)
        btns.addWidget(save_btn)
        csv_btn = QtWidgets.QPushButton("Exporter CSV bandes")
        csv_btn.clicked.connect(self._export_csv)
        btns.addWidget(csv_btn)
        right.addLayout(btns)

    def _selected_channels(self) -> List[str]:
        return [
            self._channel_list.item(i).text()
            for i in range(self._channel_list.count())
            if self._channel_list.item(i).checkState() == QtCore.Qt.CheckState.Checked
        ]

    def _compute_and_plot(self) -> None:
        try:
            from CESA.spectral_analysis import (
                EEG_BANDS,
                compute_band_powers,
                compute_psd_fft,
                compute_psd_welch,
            )
        except ImportError:
            logger.error("spectral_analysis module not available")
            return

        start = self._start_spin.value()
        dur = self._dur_spin.value()
        use_welch = self._method_combo.currentIndex() == 0
        nperseg = int(self._nperseg_spin.value())

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor('#1E1E2E')
        ax.set_xlabel("Frequence (Hz)")
        ax.set_ylabel("PSD (uV^2/Hz)" if use_welch else "Magnitude FFT")
        ax.set_title("Analyse spectrale - " + ("Welch" if use_welch else "FFT"))
        ax.set_xlim(0, 50)
        ax.grid(True, alpha=0.3)

        rows: List[Tuple[str, str, float]] = []
        for ch in self._selected_channels():
            if ch not in self._signals:
                continue
            data_arr, fs = self._signals[ch]
            i_start = max(0, int(start * fs))
            i_end = min(len(data_arr), int((start + dur) * fs))
            if i_end - i_start < 2:
                continue
            segment = np.asarray(data_arr[i_start:i_end], dtype=float)

            try:
                if use_welch:
                    freqs, spectrum = compute_psd_welch(
                        segment, fs, nperseg=int(nperseg * fs)
                    )
                    ax.semilogy(freqs, spectrum, label=ch, alpha=0.8)
                else:
                    freqs, spectrum = compute_psd_fft(segment, fs)
                    ax.plot(freqs, spectrum, label=ch, alpha=0.8)

                powers = compute_band_powers(freqs, spectrum, EEG_BANDS)
                for band_name, power in powers.items():
                    rows.append((ch, band_name, float(power)))
            except Exception as exc:
                logger.warning("PSD computation failed for %s: %s", ch, exc)

        ax.legend(fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw()

        self._rows = rows
        self._table.clear()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Canal", "Bande", "Puissance"])
        self._table.setRowCount(len(rows))
        for i, (ch, band, power) in enumerate(rows):
            self._table.setItem(i, 0, QtWidgets.QTableWidgetItem(ch))
            self._table.setItem(i, 1, QtWidgets.QTableWidgetItem(band))
            self._table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{power:.4f}"))
        self._table.resizeColumnsToContents()

    def _save_figure(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches='tight')

    def _export_csv(self) -> None:
        if not self._rows:
            QtWidgets.QMessageBox.information(self, "Export", "Aucune donnee a exporter.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter CSV bandes", "", "CSV (*.csv)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "band", "power"])
                for ch, band, power in self._rows:
                    writer.writerow([ch, band, f"{power:.6f}"])
            QtWidgets.QMessageBox.information(self, "Export", "Export termine.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")
