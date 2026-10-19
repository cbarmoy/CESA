"""Qt dialog: night signal QC report (flat / SNR / line noise)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional

from PySide6 import QtWidgets

logger = logging.getLogger(__name__)


class NightQcDialog(QtWidgets.QDialog):
    """Show detailed night QC table from :func:`CESA.prep_qc.compute_night_qc`."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        controller: Any,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("QC nuit")
        self.resize(820, 520)
        self._ctrl = controller
        self._report = None
        self._build_ui()
        self._run()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._status = QtWidgets.QLabel("Calcul QC...")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._thresh = QtWidgets.QLabel("")
        self._thresh.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        self._thresh.setWordWrap(True)
        layout.addWidget(self._thresh)

        self._table = QtWidgets.QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels([
            "Canal", "Type", "SNR (dB)", "Bruit secteur",
            "Plat", "Extreme", "p99.9 (uV)", "Raisons",
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { color: #CDD6F4; background: #1E1E2E; }"
            "QHeaderView::section { color: #CDD6F4; background: #313244; }"
        )
        layout.addWidget(self._table, 1)

        row = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Recalculer")
        refresh.clicked.connect(self._run)
        row.addWidget(refresh)
        export_btn = QtWidgets.QPushButton("Exporter CSV")
        export_btn.clicked.connect(self._export)
        row.addWidget(export_btn)
        row.addStretch()
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        layout.addLayout(row)

    def _run(self) -> None:
        from CESA.prep_qc import compute_night_qc

        ctrl = self._ctrl
        if ctrl is None or not ctrl.derivations:
            self._status.setText("Aucun enregistrement.")
            return
        try:
            self._report = compute_night_qc(
                ctrl.derivations,
                ctrl.sfreq,
                ctrl.channel_types_dict(),
            )
        except Exception as exc:
            logger.error("QC nuit failed: %s", exc, exc_info=True)
            self._status.setText(f"Erreur: {exc}")
            return
        r = self._report
        self._status.setText(
            f"{r.n_channels} canaux EEG — {len(r.bad_channels)} suspect(s). "
            + (" | ".join(r.warnings[:3]) if r.warnings else "OK")
        )
        thr = r.thresholds or {}
        self._thresh.setText(
            "Seuils: flat std < {flat} uV · extreme si p99.9 > {ext} uV "
            "et frac >= {frac:.2%} · SNR < {snr} dB · bruit 50 Hz ratio > {line}. "
            "Canaux Impedance/SpO2 exclus.".format(
                flat=thr.get("flat_std_uv", 0.5),
                ext=thr.get("extreme_amp_uv", 1500),
                frac=thr.get("extreme_frac_min", 0.001),
                snr=thr.get("snr_warn_db", 5),
                line=thr.get("line_noise_warn", 2),
            )
        )
        self._table.setRowCount(len(r.per_channel))
        for i, ch in enumerate(r.per_channel):
            vals = [
                ch.name,
                getattr(ch, "channel_type", "eeg"),
                f"{ch.snr_db:.1f}",
                f"{ch.line_noise_ratio:.2f}",
                "oui" if ch.is_flat else "",
                "oui" if ch.extreme_amp else "",
                f"{getattr(ch, 'p99_9_uv', 0.0):.0f}",
                "; ".join(ch.reasons),
            ]
            for c, v in enumerate(vals):
                self._table.setItem(i, c, QtWidgets.QTableWidgetItem(v))

    def _export(self) -> None:
        if self._report is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter QC nuit", "night_qc.csv", "CSV (*.csv)",
        )
        if not path:
            return
        rows = self._report.to_dict()["per_channel"]
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["reasons"] = "; ".join(row.get("reasons") or [])
                writer.writerow(row)
        QtWidgets.QMessageBox.information(self, "Export", f"Ecrit: {path}")
