"""Qt dialog: micro-architecture detection (spindles / SO / REM / PAC).

Wraps :meth:`CESA.app_controller.AppController.detect_micro_events`, shows
event tables + density / PAC summaries, exports CSV, and applies overlays
to the viewer :class:`~CESA.qt_viewer.annotations.AnnotationStore`.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PySide6 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

_TYPE_TO_ANN = {
    "spindle": "spindle",
    "slow_wave": "slow_wave",
    "rem_burst": "rem_burst",
    "k_complex": "k_complex",
}

_CLASSIC_KEYS = [
    "channel", "n_so", "n_spindle", "n_coupled",
    "preferred_phase_deg", "preferred_phase_rad", "resultant_length",
    "rayleigh_z", "rayleigh_p", "backend",
]
_CLASSIC_HEADERS = [
    "Canal", "N SO", "N fuseaux", "N couples",
    "Phase pref. (deg)", "Phase pref. (rad)", "R",
    "Rayleigh z", "Rayleigh p", "Backend",
]


class MicroEventsDialog(QtWidgets.QDialog):
    """Configure and run YASA micro-event detection on the loaded recording."""

    applied = QtCore.Signal(list)  # list of Annotation-ready dicts

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        controller: Any,
        eeg_channels: Sequence[str],
        eog_channels: Sequence[str],
        *,
        all_channels: Optional[Sequence[str]] = None,
        channel_types: Optional[Dict[str, str]] = None,
        has_scoring: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Micro-architecture (spindles / SO / REM)")
        self.resize(980, 680)
        self._ctrl = controller
        self._types = dict(channel_types or {})
        # Prefer full channel list so oddly-named PSG channels remain selectable
        base = list(all_channels) if all_channels else []
        if not base and controller is not None:
            base = list(getattr(controller, "derivations", {}) or {})
        hinted_eeg = list(eeg_channels)
        hinted_eog = list(eog_channels)
        for ch in hinted_eeg + hinted_eog:
            if ch not in base:
                base.append(ch)
        self._all_channels = base
        self._eeg_channels = hinted_eeg
        self._eog_channels = hinted_eog
        self._has_scoring = bool(has_scoring)
        self._last_result: Dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # -- Options -------------------------------------------------------
        opts = QtWidgets.QGroupBox("Options de detection")
        form = QtWidgets.QFormLayout(opts)

        kinds_row = QtWidgets.QHBoxLayout()
        self._chk_spindles = QtWidgets.QCheckBox("Spindles (N2)")
        self._chk_spindles.setChecked(True)
        self._chk_sw = QtWidgets.QCheckBox("Ondes lentes / SO (N2–N3)")
        self._chk_sw.setChecked(True)
        self._chk_rem = QtWidgets.QCheckBox("REM bursts (EOG)")
        self._chk_rem.setChecked(bool(self._eog_channels) and len(self._eog_channels) >= 2)
        kinds_row.addWidget(self._chk_spindles)
        kinds_row.addWidget(self._chk_sw)
        kinds_row.addWidget(self._chk_rem)
        kinds_row.addStretch(1)
        form.addRow("Types:", kinds_row)

        self._chk_hypno = QtWidgets.QCheckBox(
            "Restreindre aux stades (hypnogramme)"
        )
        self._chk_hypno.setChecked(self._has_scoring)
        self._chk_hypno.setEnabled(self._has_scoring)
        if not self._has_scoring:
            self._chk_hypno.setToolTip("Chargez ou calculez un scoring d'abord.")
        form.addRow("", self._chk_hypno)

        self._chk_coupling = QtWidgets.QCheckBox(
            "Calculer couplage SO–sigma (PAC) + co-occurrence"
        )
        self._chk_coupling.setChecked(True)
        form.addRow("", self._chk_coupling)

        self._chk_classic = QtWidgets.QCheckBox(
            "Couplage classique (filtres + Hilbert + phase preferee)"
        )
        self._chk_classic.setChecked(True)
        self._chk_classic.setToolTip(
            "Detection independante de YASA: bandes 0.5-1.25 Hz (SO) et "
            "12-16 Hz (fuseaux), phase instantanee par transformee de Hilbert, "
            "phase preferee par statistiques circulaires."
        )
        form.addRow("", self._chk_classic)

        self._eeg_list = QtWidgets.QListWidget()
        self._eeg_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self._eeg_list.setMaximumHeight(140)
        eeg_hint = set(self._eeg_channels)
        for ch in self._all_channels:
            ctype = self._types.get(ch, "")
            label = f"{ch}  [{ctype}]" if ctype else ch
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, ch)
            self._eeg_list.addItem(item)
            item.setSelected(ch in eeg_hint or ctype == "eeg")
        form.addRow("Canaux EEG (tous listes):", self._eeg_list)

        self._eog_list = QtWidgets.QListWidget()
        self._eog_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self._eog_list.setMaximumHeight(100)
        eog_hint = set(self._eog_channels)
        n_eog_sel = 0
        for ch in self._all_channels:
            ctype = self._types.get(ch, "")
            label = f"{ch}  [{ctype}]" if ctype else ch
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, ch)
            self._eog_list.addItem(item)
            if ch in eog_hint or ctype == "eog":
                if n_eog_sel < 2 or ch in eog_hint:
                    item.setSelected(True)
                    n_eog_sel += 1
        form.addRow("Canaux EOG / LOC-ROC (tous listes):", self._eog_list)

        hint = QtWidgets.QLabel(
            "Les canaux au type inconnu (noms atypiques) restent selectionnables. "
            "Choisissez manuellement ceux a utiliser pour EEG ou EOG."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        form.addRow("", hint)

        layout.addWidget(opts)

        # -- Actions -------------------------------------------------------
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_run = QtWidgets.QPushButton("Lancer la detection")
        self._btn_run.clicked.connect(self._run_detection)
        btn_row.addWidget(self._btn_run)
        self._btn_export = QtWidgets.QPushButton("Exporter CSV...")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(self._btn_export)
        self._btn_apply = QtWidgets.QPushButton("Appliquer au viewer")
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._apply_to_viewer)
        btn_row.addWidget(self._btn_apply)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._status = QtWidgets.QLabel("Pret.")
        layout.addWidget(self._status)

        # -- Results tabs --------------------------------------------------
        self._tabs = QtWidgets.QTabWidget()
        self._events_table = QtWidgets.QTableWidget(0, 6)
        self._events_table.setHorizontalHeaderLabels([
            "Type", "Canal", "Onset (s)", "Duree (s)", "Confiance", "Details",
        ])
        self._events_table.horizontalHeader().setStretchLastSection(True)
        self._events_table.setAlternatingRowColors(True)
        self._tabs.addTab(self._events_table, "Evenements")

        self._summary_table = QtWidgets.QTableWidget(0, 8)
        self._summary_table.setHorizontalHeaderLabels([
            "Type", "Canal", "Stade", "Count", "Duree (min)",
            "Densite /min", "Duree moy (s)", "Freq moy (Hz)",
        ])
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.setAlternatingRowColors(True)
        self._tabs.addTab(self._summary_table, "Densites")

        self._pac_table = QtWidgets.QTableWidget(0, 6)
        self._pac_table.setHorizontalHeaderLabels([
            "Canal", "N SO", "N co-occ.", "% co-occ.",
            "ndPAC moy.", "Phase@sigma moy.",
        ])
        self._pac_table.horizontalHeader().setStretchLastSection(True)
        self._pac_table.setAlternatingRowColors(True)
        self._tabs.addTab(self._pac_table, "PAC / co-occurrence")

        self._classic_table = QtWidgets.QTableWidget(0, len(_CLASSIC_HEADERS))
        self._classic_table.setHorizontalHeaderLabels(_CLASSIC_HEADERS)
        self._classic_table.horizontalHeader().setStretchLastSection(True)
        self._classic_table.setAlternatingRowColors(True)
        self._tabs.addTab(self._classic_table, "Phase preferee (classique)")

        layout.addWidget(self._tabs, 1)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Fermer")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

    def _selected_items(self, list_widget: QtWidgets.QListWidget) -> List[str]:
        names = []
        for item in list_widget.selectedItems():
            name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            names.append(str(name) if name is not None else item.text())
        return names

    def _run_detection(self) -> None:
        kinds: List[str] = []
        if self._chk_spindles.isChecked():
            kinds.append("spindles")
        if self._chk_sw.isChecked():
            kinds.append("slow_waves")
        if self._chk_rem.isChecked():
            kinds.append("rem_bursts")
        run_classic = self._chk_classic.isChecked()
        if not kinds and not run_classic:
            QtWidgets.QMessageBox.warning(
                self, "Attention", "Selectionnez au moins un type d'evenement."
            )
            return

        eeg = self._selected_items(self._eeg_list)
        eog = self._selected_items(self._eog_list)
        use_hypno = self._chk_hypno.isChecked()

        progress = QtWidgets.QProgressDialog(
            "Detection micro-architecture...", None, 0, 0, self,
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()
        QtWidgets.QApplication.processEvents()

        def _report(msg: str) -> None:
            progress.setLabelText(msg)
            QtWidgets.QApplication.processEvents()

        result: Dict[str, Any] = {}
        classic: Dict[str, Any] = {}
        try:
            if kinds:
                result = self._ctrl.detect_micro_events(
                    kinds,
                    channels=eeg or None,
                    eog_channels=eog or None,
                    use_hypnogram=use_hypno,
                    coupling=self._chk_coupling.isChecked(),
                    progress_callback=_report,
                ) or {}
            if run_classic:
                classic = self._ctrl.detect_so_spindle_coupling(
                    channels=eeg or None,
                    use_hypnogram=use_hypno,
                    progress_callback=_report,
                ) or {}
        except Exception as exc:
            progress.close()
            logger.error("Micro-event detection failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Erreur", f"Detection echouee:\n{exc}"
            )
            return
        progress.close()

        events = list(result.get("events") or []) + list(classic.get("events") or [])
        warnings = list(result.get("warnings") or []) + list(classic.get("warnings") or [])
        self._last_result = {
            "events": events,
            "summary": result.get("summary") or [],
            "pac": result.get("pac") or [],
            "coupling": classic.get("coupling") or [],
            "warnings": warnings,
            "used_hypnogram": bool(
                result.get("used_hypnogram") or classic.get("used_hypnogram")
            ),
            "backend": classic.get("backend", "none"),
        }
        self._fill_events_table(events)
        self._fill_summary_table(self._last_result["summary"])
        self._fill_pac_table(self._last_result["pac"])
        self._fill_classic_table(self._last_result["coupling"])

        has_data = bool(events) or bool(self._last_result["coupling"])
        self._btn_export.setEnabled(has_data)
        self._btn_apply.setEnabled(bool(events))

        msg = f"{len(events)} evenement(s) detecte(s)."
        if self._last_result["used_hypnogram"]:
            msg += " Hypnogramme utilise."
        if run_classic:
            n_coupled = sum(
                int(row.get("n_coupled", 0) or 0)
                for row in self._last_result["coupling"]
            )
            msg += (
                f" Couplage classique: {n_coupled} fuseau(x) couple(s) "
                f"({self._last_result['backend']})."
            )
        if warnings:
            msg += "  |  " + " ; ".join(str(w) for w in warnings[:3])
        self._status.setText(msg)

        if warnings and not events:
            QtWidgets.QMessageBox.information(
                self, "Detection", "\n".join(str(w) for w in warnings),
            )

    def _fill_events_table(self, events: List[Dict[str, Any]]) -> None:
        self._events_table.setRowCount(0)
        self._events_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            etype = str(ev.get("type", ""))
            ch = str(ev.get("channel", ""))
            onset = float(ev.get("onset", 0.0))
            dur = float(ev.get("duration", 0.0))
            conf = float(ev.get("confidence", 1.0))
            meta = ev.get("metadata") or {}
            details_parts = []
            for key in (
                "Frequency", "ndPAC", "CooccurringSpindle", "PTP", "Stage",
                "so_phase_deg", "detector",
            ):
                if key in meta and meta[key] is not None:
                    details_parts.append(f"{key}={meta[key]}")
            details = ", ".join(details_parts)
            for col, val in enumerate([
                etype, ch, f"{onset:.2f}", f"{dur:.2f}", f"{conf:.2f}", details,
            ]):
                self._events_table.setItem(i, col, QtWidgets.QTableWidgetItem(val))

    def _fill_summary_table(self, rows: List[Dict[str, Any]]) -> None:
        self._summary_table.setRowCount(0)
        self._summary_table.setRowCount(len(rows))
        keys = [
            "type", "channel", "stage", "count", "duration_min",
            "density_per_min", "mean_duration_s", "mean_frequency_hz",
        ]
        for i, row in enumerate(rows):
            for col, key in enumerate(keys):
                val = row.get(key, "")
                text = "" if val is None else str(val)
                self._summary_table.setItem(i, col, QtWidgets.QTableWidgetItem(text))

    def _fill_pac_table(self, rows: List[Dict[str, Any]]) -> None:
        self._pac_table.setRowCount(0)
        self._pac_table.setRowCount(len(rows))
        keys = [
            "channel", "n_sw", "n_cooccurring", "pct_cooccurring",
            "mean_ndPAC", "mean_phase_at_sigma_peak",
        ]
        for i, row in enumerate(rows):
            for col, key in enumerate(keys):
                val = row.get(key, "")
                text = "" if val is None else str(val)
                self._pac_table.setItem(i, col, QtWidgets.QTableWidgetItem(text))

    def _fill_classic_table(self, rows: List[Dict[str, Any]]) -> None:
        self._classic_table.setRowCount(0)
        self._classic_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for col, key in enumerate(_CLASSIC_KEYS):
                val = row.get(key, "")
                if isinstance(val, float):
                    text = "" if val != val else f"{val:.4g}"  # NaN -> blank
                else:
                    text = "" if val is None else str(val)
                self._classic_table.setItem(i, col, QtWidgets.QTableWidgetItem(text))

    def _export_csv(self) -> None:
        if not self._last_result.get("events") and not self._last_result.get("coupling"):
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Exporter micro-evenements",
            "micro_events.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        base = Path(path)
        stem = base.with_suffix("")
        try:
            self._write_events_csv(stem.with_name(stem.name + "_events.csv"))
            summary = self._last_result.get("summary") or []
            if summary:
                self._write_records_csv(
                    stem.with_name(stem.name + "_densities.csv"), summary,
                )
            pac = self._last_result.get("pac") or []
            if pac:
                self._write_records_csv(
                    stem.with_name(stem.name + "_pac.csv"), pac,
                )
            coupling = self._last_result.get("coupling") or []
            if coupling:
                self._write_records_csv(
                    stem.with_name(stem.name + "_coupling.csv"), coupling,
                )
            self._status.setText(f"Export CSV: {stem.parent}")
            QtWidgets.QMessageBox.information(
                self,
                "Export",
                f"Fichiers ecrits dans:\n{stem.parent}\n"
                f"({stem.name}_events.csv"
                + (", *_densities.csv" if summary else "")
                + (", *_pac.csv" if pac else "")
                + (", *_coupling.csv" if coupling else "")
                + ")",
            )
        except Exception as exc:
            logger.error("CSV export failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Export echoue:\n{exc}")

    def _write_events_csv(self, path: Path) -> None:
        events = self._last_result.get("events") or []
        fieldnames = [
            "type", "channel", "onset", "duration", "confidence",
            "detector", "Frequency", "ndPAC", "PhaseAtSigmaPeak",
            "CooccurringSpindle", "PTP", "Stage",
            "so_phase_rad", "so_phase_deg", "coupled_so_onset_s",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for ev in events:
                meta = ev.get("metadata") or {}
                writer.writerow({
                    "type": ev.get("type", ""),
                    "channel": ev.get("channel", ""),
                    "onset": ev.get("onset", ""),
                    "duration": ev.get("duration", ""),
                    "confidence": ev.get("confidence", ""),
                    "detector": meta.get("detector", "yasa"),
                    "Frequency": meta.get("Frequency", ""),
                    "ndPAC": meta.get("ndPAC", ""),
                    "PhaseAtSigmaPeak": meta.get("PhaseAtSigmaPeak", ""),
                    "CooccurringSpindle": meta.get("CooccurringSpindle", ""),
                    "PTP": meta.get("PTP", ""),
                    "Stage": meta.get("Stage", ""),
                    "so_phase_rad": meta.get("so_phase_rad", ""),
                    "so_phase_deg": meta.get("so_phase_deg", ""),
                    "coupled_so_onset_s": meta.get("coupled_so_onset_s", ""),
                })

    @staticmethod
    def _write_records_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _apply_to_viewer(self) -> None:
        events = self._last_result.get("events") or []
        if not events:
            return
        payload = []
        for ev in events:
            etype = str(ev.get("type", "custom"))
            ann_type = _TYPE_TO_ANN.get(etype, etype)
            meta = ev.get("metadata") or {}
            bits = [ann_type]
            if meta.get("Frequency") is not None:
                bits.append(f"{float(meta['Frequency']):.1f}Hz")
            if meta.get("ndPAC") is not None:
                try:
                    bits.append(f"ndPAC={float(meta['ndPAC']):.2f}")
                except (TypeError, ValueError):
                    pass
            if meta.get("so_phase_deg") is not None:
                try:
                    bits.append(f"phiSO={float(meta['so_phase_deg']):.0f}deg")
                except (TypeError, ValueError):
                    pass
            payload.append({
                "onset_s": float(ev.get("onset", 0.0)),
                "duration_s": float(ev.get("duration", 0.0)),
                "ann_type": ann_type,
                "channel": str(ev.get("channel", "") or ""),
                "text": " ".join(bits),
                "confidence": float(ev.get("confidence", 1.0)),
                "source": "auto",
            })
        self.applied.emit(payload)
        self._status.setText(f"{len(payload)} annotation(s) envoyees au viewer.")
        QtWidgets.QMessageBox.information(
            self,
            "Viewer",
            f"{len(payload)} evenement(s) ajoutes aux annotations (source=auto).",
        )
