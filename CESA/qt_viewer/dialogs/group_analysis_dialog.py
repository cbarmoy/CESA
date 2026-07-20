"""Qt dialog: multi-subject group analysis (MSE / renormalized entropy / HRV).

Drives :mod:`CESA.group_analysis` natively from the Qt viewer. Designs can be
imported from a CSV (``subject, condition, edf_path[, scoring_path]``) or built
by assigning recordings to the AVANT/APRES conditions. Computation runs in a
background thread; results are exported as CSV and figures.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

from PySide6 import QtCore, QtWidgets

from core.raw_loader import recording_extensions_for_scan

from .batch_analysis_dialog import _BatchWorker

logger = logging.getLogger(__name__)


class GroupAnalysisDialog(QtWidgets.QDialog):
    """Tabbed group-analysis dialog (design / parameters / run + export)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyse de groupe (MSE / REN / HRV)")
        self.resize(980, 740)
        self._worker: Optional[_BatchWorker] = None
        self._result = None  # GroupAnalysisResult or GroupHRVResult
        self._is_hrv = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, stretch=1)
        self._tabs.addTab(self._build_design_tab(), "1. Design")
        self._tabs.addTab(self._build_params_tab(), "2. Parametres")
        self._tabs.addTab(self._build_run_tab(), "3. Traitement et export")

    def _build_design_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Importer design CSV...")
        import_btn.clicked.connect(self._import_design_csv)
        add_before = QtWidgets.QPushButton("Ajouter fichiers AVANT...")
        add_before.clicked.connect(lambda: self._add_files("AVANT"))
        add_after = QtWidgets.QPushButton("Ajouter fichiers APRES...")
        add_after.clicked.connect(lambda: self._add_files("APRES"))
        remove_btn = QtWidgets.QPushButton("Supprimer")
        remove_btn.clicked.connect(self._remove_rows)
        top.addWidget(import_btn)
        top.addWidget(add_before)
        top.addWidget(add_after)
        top.addStretch()
        top.addWidget(remove_btn)
        v.addLayout(top)

        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["subject", "condition", "edf_path", "scoring_path"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        v.addWidget(self._table, stretch=1)

        hint = QtWidgets.QLabel(
            "Astuce: 'subject' identifie l'individu (paire AVANT/APRES); 'condition'"
            " doit correspondre aux libelles avant/apres choisis dans Parametres."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        return w

    def _build_params_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItems([
            "MSE (entropie multi-echelle)",
            "Entropie renormalisee",
            "HRV (cardiaque)",
        ])
        form.addRow("Metrique:", self._metric_combo)

        self._before_edit = QtWidgets.QLineEdit("AVANT")
        form.addRow("Libelle avant:", self._before_edit)
        self._after_edit = QtWidgets.QLineEdit("APRES")
        form.addRow("Libelle apres:", self._after_edit)

        self._channels_edit = QtWidgets.QLineEdit()
        self._channels_edit.setPlaceholderText("ex: C3-A2, C4-A1 (vide = auto)")
        form.addRow("Canaux:", self._channels_edit)

        self._stages_edit = QtWidgets.QLineEdit("W, N2, N3, R")
        form.addRow("Stades:", self._stages_edit)

        self._epoch_spin = QtWidgets.QDoubleSpinBox()
        self._epoch_spin.setRange(1.0, 120.0)
        self._epoch_spin.setValue(30.0)
        form.addRow("Duree d'epoque (s):", self._epoch_spin)

        self._stats_chk = QtWidgets.QCheckBox("Tests statistiques (Wilcoxon + permutation)")
        self._stats_chk.setChecked(True)
        form.addRow("", self._stats_chk)

        return w

    def _build_run_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        btns = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Lancer")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._start)
        self._stop_btn = QtWidgets.QPushButton("Arreter")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        btns.addWidget(self._run_btn)
        btns.addWidget(self._stop_btn)
        btns.addStretch()
        v.addLayout(btns)

        self._progress = QtWidgets.QProgressBar()
        v.addWidget(self._progress)
        self._status_lbl = QtWidgets.QLabel("Pret.")
        v.addWidget(self._status_lbl)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        v.addWidget(self._log, stretch=1)

        exp = QtWidgets.QHBoxLayout()
        self._export_profiles_btn = QtWidgets.QPushButton("Exporter profils CSV")
        self._export_profiles_btn.clicked.connect(self._export_profiles)
        self._export_stats_btn = QtWidgets.QPushButton("Exporter stats CSV")
        self._export_stats_btn.clicked.connect(self._export_stats)
        self._export_fig_btn = QtWidgets.QPushButton("Enregistrer figures")
        self._export_fig_btn.clicked.connect(self._export_figures)
        for b in (self._export_profiles_btn, self._export_stats_btn, self._export_fig_btn):
            b.setEnabled(False)
            exp.addWidget(b)
        exp.addStretch()
        v.addLayout(exp)
        return w

    # ------------------------------------------------------------------
    # Design table helpers
    # ------------------------------------------------------------------

    def _add_row(self, subject: str, condition: str, edf_path: str, scoring_path: str = "") -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        for c, val in enumerate((subject, condition, edf_path, scoring_path)):
            self._table.setItem(r, c, QtWidgets.QTableWidgetItem(val))

    def _add_files(self, condition: str) -> None:
        exts = " ".join(f"*{e}" for e in recording_extensions_for_scan())
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, f"Fichiers {condition}", "",
            f"Enregistrements ({exts});;Tous les fichiers (*.*)",
        )
        label = self._before_edit.text() if condition == "AVANT" else self._after_edit.text()
        for p in paths:
            subject = os.path.splitext(os.path.basename(p))[0]
            self._add_row(subject, label, p)

    def _remove_rows(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _import_design_csv(self) -> None:
        from CESA.group_analysis import load_design_csv

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importer design CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            entries = load_design_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Design", f"Import impossible:\n{exc}")
            return
        self._table.setRowCount(0)
        for e in entries:
            self._add_row(
                e.subject, e.condition, str(e.edf_path),
                str(e.scoring_path) if e.scoring_path else "",
            )
        self._status_lbl.setText(f"{len(entries)} entree(s) importee(s).")

    def _collect_entries(self):
        from CESA.group_analysis import GroupDesignEntry

        entries = []
        for r in range(self._table.rowCount()):
            def _cell(c: int) -> str:
                item = self._table.item(r, c)
                return item.text().strip() if item else ""

            subject = _cell(0)
            condition = _cell(1)
            edf_path = _cell(2)
            scoring_path = _cell(3)
            if not subject or not condition or not edf_path:
                continue
            entries.append(GroupDesignEntry(
                subject=subject,
                condition=condition,
                edf_path=Path(edf_path),
                scoring_path=Path(scoring_path) if scoring_path else None,
            ))
        return entries

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _build_config(self):
        from CESA.group_analysis import GroupAnalysisConfig

        metric_idx = self._metric_combo.currentIndex()
        metric = {0: "mse", 1: "renorm", 2: "hrv"}[metric_idx]
        channels = [c.strip() for c in self._channels_edit.text().split(",") if c.strip()]
        stages = [s.strip() for s in self._stages_edit.text().split(",") if s.strip()]
        return GroupAnalysisConfig(
            metric=metric if metric != "hrv" else "mse",
            before_label=self._before_edit.text().strip() or "AVANT",
            after_label=self._after_edit.text().strip() or "APRES",
            display_before_label=self._before_edit.text().strip() or "AVANT",
            display_after_label=self._after_edit.text().strip() or "APRES",
            stages=stages or ["W", "N2", "N3", "R"],
            epoch_seconds=self._epoch_spin.value(),
            channel_names=channels or None,
        )

    def _start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        entries = self._collect_entries()
        if len(entries) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Design", "Ajoutez au moins deux entrees (AVANT et APRES)."
            )
            return
        config = self._build_config()
        self._is_hrv = self._metric_combo.currentIndex() == 2
        run_stats = self._stats_chk.isChecked() and not self._is_hrv

        self._tabs.setCurrentIndex(2)
        self._log.clear()
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        for b in (self._export_profiles_btn, self._export_stats_btn, self._export_fig_btn):
            b.setEnabled(False)

        is_hrv = self._is_hrv

        def job(progress_cb, should_stop):
            from CESA.group_analysis import (
                StatsConfig,
                compute_group_hrv,
                compute_group_profiles,
                run_statistical_tests,
            )

            # group_analysis expects progress_cb(message, fraction)
            def gp(msg, frac):
                progress_cb(float(frac) * 100.0, str(msg))

            if is_hrv:
                return compute_group_hrv(entries, config, progress_cb=gp)
            result = compute_group_profiles(entries, config, progress_cb=gp)
            if run_stats:
                progress_cb(95.0, "Tests statistiques...")
                try:
                    result.stats = run_statistical_tests(result, StatsConfig())
                except Exception as exc:
                    logger.warning("Stats failed: %s", exc)
            return result

        self._worker = _BatchWorker(job, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._log.appendPlainText("Arret demande (effectif en fin d'entree)...")

    def _on_progress(self, pct: float, msg: str) -> None:
        self._progress.setValue(int(pct))
        self._status_lbl.setText(msg)
        self._log.appendPlainText(msg)

    def _on_failed(self, msg: str) -> None:
        self._log.appendPlainText(f"ERREUR: {msg}")
        self._status_lbl.setText("Echec.")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_finished(self, result) -> None:
        self._result = result
        self._progress.setValue(100)
        self._status_lbl.setText("Termine.")
        self._log.appendPlainText("Calcul termine.")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._export_profiles_btn.setEnabled(True)
        if not self._is_hrv:
            self._export_stats_btn.setEnabled(True)
            self._export_fig_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_profiles(self) -> None:
        if self._result is None:
            return
        from CESA.group_analysis import export_hrv_to_csv, export_profiles_to_csv

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            if self._is_hrv:
                export_hrv_to_csv(self._result, path)
            else:
                export_profiles_to_csv(self._result, path)
            self._log.appendPlainText(f"-> {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")

    def _export_stats(self) -> None:
        if self._result is None or self._is_hrv:
            return
        from CESA.group_analysis import export_stats_to_csv

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter stats CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            export_stats_to_csv(self._result, path)
            self._log.appendPlainText(f"-> {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec:\n{exc}")

    def _export_figures(self) -> None:
        if self._result is None or self._is_hrv:
            return
        from CESA.group_analysis import plot_stage_profiles

        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Dossier des figures")
        if not directory:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            saved = 0
            for stage in self._result.available_stages():
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_stage_profiles(self._result, stage, ax=ax)
                out = os.path.join(directory, f"group_profile_{stage}.png")
                fig.savefig(out, dpi=200, bbox_inches="tight")
                plt.close(fig)
                self._log.appendPlainText(f"-> {out}")
                saved += 1
            QtWidgets.QMessageBox.information(self, "Figures", f"{saved} figure(s) enregistree(s).")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Figures", f"Echec:\n{exc}")
