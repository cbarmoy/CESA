"""Integrated dashboard panel: sleep architecture, quality metrics, SNR.

A dockable widget that provides at-a-glance summary statistics for the
loaded recording, updated as scoring / annotations change.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class DashboardPanel(QtWidgets.QDockWidget):
    """Dockable dashboard showing sleep metrics and quality indicators."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Dashboard", parent)
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setMinimumWidth(240)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.setWidget(scroll)

        container = QtWidgets.QWidget()
        scroll.setWidget(container)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Sleep architecture
        self._arch_group = QtWidgets.QGroupBox("Architecture du Sommeil")
        arch_lay = QtWidgets.QFormLayout(self._arch_group)
        self._tst_label = QtWidgets.QLabel("--")
        self._se_label = QtWidgets.QLabel("--")
        self._sol_label = QtWidgets.QLabel("--")
        self._waso_label = QtWidgets.QLabel("--")
        self._rem_lat_label = QtWidgets.QLabel("--")
        arch_lay.addRow("TST:", self._tst_label)
        arch_lay.addRow("SE:", self._se_label)
        arch_lay.addRow("SOL:", self._sol_label)
        arch_lay.addRow("WASO:", self._waso_label)
        arch_lay.addRow("REM Lat:", self._rem_lat_label)
        layout.addWidget(self._arch_group)

        # Stage distribution
        self._dist_group = QtWidgets.QGroupBox("Distribution des Stades")
        dist_lay = QtWidgets.QVBoxLayout(self._dist_group)
        self._dist_bars: Dict[str, Tuple] = {}
        for stage, color in [
            ("W", "#EBA0AC"), ("N1", "#89B4FA"), ("N2", "#74C7EC"),
            ("N3", "#89DCEB"), ("REM", "#CBA6F7"),
        ]:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(stage)
            lbl.setFixedWidth(30)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat("%v%")
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {color}; }}"
            )
            row.addWidget(lbl)
            row.addWidget(bar)
            dist_lay.addLayout(row)
            self._dist_bars[stage] = (lbl, bar)
        layout.addWidget(self._dist_group)

        # Quality metrics
        self._quality_group = QtWidgets.QGroupBox("Qualite du Signal")
        quality_lay = QtWidgets.QFormLayout(self._quality_group)
        self._snr_table = QtWidgets.QTableWidget(0, 2)
        self._snr_table.setHorizontalHeaderLabels(["Canal", "SNR (dB)"])
        self._snr_table.horizontalHeader().setStretchLastSection(True)
        self._snr_table.verticalHeader().setVisible(False)
        self._snr_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._snr_table.setMaximumHeight(140)
        quality_lay.addRow(self._snr_table)
        layout.addWidget(self._quality_group)

        # Filter status
        self._filter_group = QtWidgets.QGroupBox("Filtres")
        filter_lay = QtWidgets.QFormLayout(self._filter_group)
        self._n_filters_label = QtWidgets.QLabel("0")
        self._filter_status_label = QtWidgets.QLabel("OFF")
        filter_lay.addRow("Actifs:", self._n_filters_label)
        filter_lay.addRow("Etat:", self._filter_status_label)
        layout.addWidget(self._filter_group)

        # Events summary
        self._events_group = QtWidgets.QGroupBox("Evenements")
        events_lay = QtWidgets.QFormLayout(self._events_group)
        self._n_artifacts_label = QtWidgets.QLabel("0")
        self._n_arousals_label = QtWidgets.QLabel("0")
        self._n_apneas_label = QtWidgets.QLabel("0")
        self._n_annotations_label = QtWidgets.QLabel("0")
        events_lay.addRow("Artefacts:", self._n_artifacts_label)
        events_lay.addRow("Arousals:", self._n_arousals_label)
        events_lay.addRow("Apnees:", self._n_apneas_label)
        events_lay.addRow("Annotations:", self._n_annotations_label)
        layout.addWidget(self._events_group)

        # Warnings
        self._warnings_group = QtWidgets.QGroupBox("Alertes")
        warn_lay = QtWidgets.QVBoxLayout(self._warnings_group)
        self._warnings_list = QtWidgets.QListWidget()
        self._warnings_list.setMaximumHeight(80)
        warn_lay.addWidget(self._warnings_list)
        layout.addWidget(self._warnings_group)

        layout.addStretch()

    # ---- public API -----------------------------------------------

    def set_sleep_metrics(
        self,
        tst_min: float = 0,
        se_pct: float = 0,
        sol_min: float = 0,
        waso_min: float = 0,
        rem_lat_min: float = 0,
    ) -> None:
        self._tst_label.setText(f"{tst_min:.1f} min")
        self._se_label.setText(f"{se_pct:.1f}%")
        self._sol_label.setText(f"{sol_min:.1f} min")
        self._waso_label.setText(f"{waso_min:.1f} min")
        self._rem_lat_label.setText(f"{rem_lat_min:.1f} min")

    def set_stage_distribution(self, dist: Dict[str, float]) -> None:
        """*dist* maps stage name -> percentage (0-100)."""
        for stage, (lbl, bar) in self._dist_bars.items():
            pct = dist.get(stage, 0.0)
            bar.setValue(int(pct))

    def set_snr(self, snr: Dict[str, float]) -> None:
        self._snr_table.setRowCount(len(snr))
        for i, (name, db) in enumerate(sorted(snr.items())):
            self._snr_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            item = QtWidgets.QTableWidgetItem(f"{db:.1f}")
            if db < 5:
                item.setForeground(QtGui.QColor("#F38BA8"))
            elif db < 10:
                item.setForeground(QtGui.QColor("#F9E2AF"))
            else:
                item.setForeground(QtGui.QColor("#A6E3A1"))
            self._snr_table.setItem(i, 1, item)

    def set_filter_info(self, n_active: int, enabled: bool) -> None:
        self._n_filters_label.setText(str(n_active))
        self._filter_status_label.setText("ON" if enabled else "OFF")
        self._filter_status_label.setStyleSheet(
            "color: #A6E3A1;" if enabled else "color: #F38BA8;"
        )

    def set_event_counts(
        self,
        artifacts: int = 0,
        arousals: int = 0,
        apneas: int = 0,
        annotations: int = 0,
    ) -> None:
        self._n_artifacts_label.setText(str(artifacts))
        self._n_arousals_label.setText(str(arousals))
        self._n_apneas_label.setText(str(apneas))
        self._n_annotations_label.setText(str(annotations))

    def set_warnings(self, warnings: List[str]) -> None:
        self._warnings_list.clear()
        for w in warnings:
            item = QtWidgets.QListWidgetItem(w)
            item.setForeground(QtGui.QColor("#F9E2AF"))
            self._warnings_list.addItem(item)

    def compute_snr(
        self, signals: Dict[str, Any], epoch_len: float = 30.0,
    ) -> Dict[str, float]:
        """Estimate SNR per channel as signal power / high-freq noise power."""
        snr = {}
        for name, (data, fs) in signals.items():
            if len(data) == 0 or fs <= 0:
                snr[name] = 0.0
                continue
            # Rough SNR: power of band 0.5-30 Hz vs 50+ Hz
            from scipy.signal import welch as _welch
            try:
                freqs, psd = _welch(data, fs, nperseg=min(len(data), int(fs * epoch_len)))
                sig_mask = (freqs >= 0.5) & (freqs <= 30)
                noise_mask = freqs >= 50
                sig_power = np.mean(psd[sig_mask]) if sig_mask.any() else 1e-12
                noise_power = np.mean(psd[noise_mask]) if noise_mask.any() else 1e-12
                snr[name] = 10 * np.log10(max(sig_power, 1e-12) / max(noise_power, 1e-12))
            except Exception:
                snr[name] = 0.0
        return snr

    def compute_stage_distribution(
        self, hypnogram: List[str],
    ) -> Dict[str, float]:
        if not hypnogram:
            return {}
        total = len(hypnogram)
        dist = {}
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            mapped = stage if stage != "REM" else "R"
            count = sum(1 for s in hypnogram if s.upper() in (stage, mapped))
            dist[stage] = (count / total) * 100
        return dist
