"""Qt dialog: measure several automatic-scoring configurations side by side.

Presents the configurations worth testing on the loaded montage, runs the
selected ones against the manual reference, and ranks them. The controller's
scoring is restored afterwards, so nothing is lost; applying the winning
configuration is an explicit action.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from PySide6 import QtCore, QtWidgets

from CESA.qt_viewer.dialogs.auto_scoring_config_dialog import METHOD_LABELS
from CESA.scoring_sweep import (
    DEFAULT_RUN_SECONDS,
    SweepConfig,
    SweepResult,
    apply_config,
    build_backend_comparison,
    build_default_configs,
    estimate_duration_s,
    rank_results,
    run_sweep,
    sweep_summary,
    usable_backends,
)

logger = logging.getLogger(__name__)

_RANK_KEYS = (
    ("kappa", "Kappa de Cohen (recommande)"),
    ("accuracy", "Concordance brute"),
    ("macro_f1", "F1 macro"),
    ("deep_sleep_accuracy", "Concordance N2+N3 fusionnes"),
)


class ScoringSweepDialog(QtWidgets.QDialog):
    """Run and compare several scoring configurations."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], controller: Any) -> None:
        super().__init__(parent)
        self.setWindowTitle("Comparer des configurations de scoring")
        self.resize(1080, 700)
        self._ctrl = controller
        self._results: List[SweepResult] = []
        self._cancelled = False
        self._applied: Optional[SweepConfig] = None
        self._channels = list(getattr(self._ctrl.raw, "ch_names", []) or [])
        self._blocked = usable_backends(self._ctrl, list(METHOD_LABELS))
        self._configs: List[SweepConfig] = []
        self._build_ui()
        self._rebuild_configs()

    # -- Backend selection ----------------------------------------------

    def _default_method(self) -> str:
        """The controller's backend if it can run here, else the first that can.

        Sweeping with a backend whose prerequisites are missing only produces a
        column of identical failures.
        """
        current = str(getattr(self._ctrl, "sleep_scoring_method", "") or "").lower()
        if current and not self._blocked.get(current):
            return current
        for method in METHOD_LABELS:
            if not self._blocked.get(method):
                return method
        return current or "yasa"

    def _current_method(self) -> str:
        return str(self._method_combo.currentData() or "yasa")

    def _rebuild_configs(self) -> None:
        if getattr(self, "_rebuilding", False):
            return
        if not hasattr(self, "_compare_backends"):
            return
        self._rebuilding = True
        try:
            self._refresh_blocked()
            method = self._current_method()
            configs = build_default_configs(
                self._channels,
                method=method,
                age=getattr(self._ctrl, "yasa_age", None),
                male=getattr(self._ctrl, "yasa_male", None),
            )
            if self._compare_backends.isChecked():
                others = [
                    other for other in METHOD_LABELS
                    if other != method and not self._blocked.get(other)
                ]
                configs += build_backend_comparison(
                    self._channels, others,
                    age=getattr(self._ctrl, "yasa_age", None),
                    male=getattr(self._ctrl, "yasa_male", None),
                )
            self._configs = configs
            self._fill_config_table()
            self._update_method_note()
            self._update_estimate()
        finally:
            self._rebuilding = False

    def _refresh_blocked(self) -> None:
        """Re-evaluate prerequisites so a token or package installed this session counts."""
        if not hasattr(self, "_method_combo"):
            return
        self._blocked = usable_backends(self._ctrl, list(METHOD_LABELS))
        for index in range(self._method_combo.count()):
            method = self._method_combo.itemData(index)
            reason = self._blocked.get(method)
            label = METHOD_LABELS.get(method, method)
            text = label if not reason else f"{label} — indisponible"
            if self._method_combo.itemText(index) != text:
                self._method_combo.setItemText(index, text)
            self._method_combo.setItemData(
                index,
                reason or "",
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )

    def _update_method_note(self) -> None:
        from CESA.scoring_channels import channel_profile

        method = self._current_method()
        reason = self._blocked.get(method)
        if reason:
            self._method_note.setText(
                f"<span style='color:#f9a825'>Ce backend ne peut pas etre "
                f"utilise ici : {reason}</span>"
            )
            return
        profile = channel_profile(method)
        self._method_note.setText(f"<i>{profile.rationale}</i>")

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Chaque configuration est scoree puis comparee au <b>meme scoring "
            "manuel</b>, ce qui permet de choisir les canaux sur des mesures "
            "plutot qu'a tatons. Votre scoring actuel et vos reglages sont "
            "restaures a la fin : appliquer une configuration est une action "
            "separee, en bas de cette fenetre."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(intro)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._build_configs_tab(), "Configurations a tester")
        self._tabs.addTab(self._build_results_tab(), "Resultats")
        layout.addWidget(self._tabs, stretch=1)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QtWidgets.QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        btns = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Lancer la comparaison")
        self._run_btn.clicked.connect(self._on_run)
        btns.addWidget(self._run_btn)
        self._cancel_btn = QtWidgets.QPushButton("Interrompre")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btns.addWidget(self._cancel_btn)
        btns.addStretch()
        self._apply_btn = QtWidgets.QPushButton("Appliquer la configuration selectionnee")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        btns.addWidget(self._apply_btn)
        close_btn = QtWidgets.QPushButton("Fermer")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self._update_estimate()

    def _build_configs_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        box = QtWidgets.QGroupBox("Algorithme")
        form = QtWidgets.QFormLayout(box)
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.setMaxVisibleItems(12)
        for method, label in METHOD_LABELS.items():
            reason = self._blocked.get(method)
            text = label if not reason else f"{label} — indisponible"
            self._method_combo.addItem(text, method)
            if reason:
                self._method_combo.setItemData(
                    self._method_combo.count() - 1,
                    reason,
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )
        default = self._default_method()
        position = self._method_combo.findData(default)
        if position >= 0:
            self._method_combo.setCurrentIndex(position)
        # activated is what a click in the popup emits; currentIndexChanged
        # alone can be swallowed on Windows when neighbouring items are greyed.
        self._method_combo.activated.connect(lambda _i: self._rebuild_configs())
        self._method_combo.currentIndexChanged.connect(
            lambda _i: self._rebuild_configs()
        )
        form.addRow("Backend a comparer :", self._method_combo)

        self._compare_backends = QtWidgets.QCheckBox(
            "Ajouter une configuration par autre algorithme disponible"
        )
        self._compare_backends.setToolTip(
            "Compare aussi les algorithmes entre eux, chacun avec les canaux "
            "que son article recommande. Les backends dont les prerequis "
            "manquent sont exclus."
        )
        self._compare_backends.toggled.connect(lambda _c: self._rebuild_configs())
        form.addRow("", self._compare_backends)

        self._method_note = QtWidgets.QLabel()
        self._method_note.setWordWrap(True)
        self._method_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        form.addRow(self._method_note)
        lay.addWidget(box)

        lay.addWidget(QtWidgets.QLabel(
            "Decochez les configurations que vous ne voulez pas tester. Le "
            "balayage fait varier un element a la fois (derivation EEG, puis "
            "EOG, puis EMG, puis metadonnees), ce qui rend l'effet de chaque "
            "choix lisible. Les canaux proposes sont ceux attendus par "
            "l'algorithme choisi."
        ))

        headers = ["Tester", "Algorithme", "EEG", "EOG", "EMG", "Age / sexe"]
        self._config_table = QtWidgets.QTableWidget(0, len(headers))
        self._config_table.setHorizontalHeaderLabels(headers)
        header = self._config_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, len(headers)):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._config_table.itemChanged.connect(lambda _item: self._update_estimate())
        lay.addWidget(self._config_table, stretch=1)

        self._config_empty = QtWidgets.QLabel(
            "Aucune derivation EEG connue n'a ete trouvee dans ce montage pour "
            "cet algorithme : impossible de proposer des configurations."
        )
        self._config_empty.setWordWrap(True)
        self._config_empty.setVisible(False)
        lay.addWidget(self._config_empty)
        return w

    def _fill_config_table(self) -> None:
        from CESA.scoring_channels import channel_profile

        table = self._config_table
        table.blockSignals(True)
        table.setRowCount(len(self._configs))
        for row, config in enumerate(self._configs):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(
                QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            check.setCheckState(QtCore.Qt.CheckState.Checked)
            table.setItem(row, 0, check)
            sex = "" if config.male is None else (" H" if config.male else " F")
            metadata = (
                f"{config.age}{sex}" if config.age is not None or sex else "non renseigne"
            )
            emg = (
                config.emg or "aucun"
                if channel_profile(config.method).uses("emg")
                else "non utilise"
            )
            cells = [
                METHOD_LABELS.get(config.method, config.method).split(" (")[0],
                config.eeg, config.eog or "aucun", emg, metadata,
            ]
            for col, text in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(str(text))
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row, col, item)
        table.blockSignals(False)
        table.setVisible(bool(self._configs))
        self._config_empty.setVisible(not self._configs)

    def _build_results_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Classer par :"))
        self._rank_key = QtWidgets.QComboBox()
        for key, label in _RANK_KEYS:
            self._rank_key.addItem(label, key)
        self._rank_key.currentIndexChanged.connect(lambda _i: self._refresh_results())
        top.addWidget(self._rank_key)
        top.addStretch()
        lay.addLayout(top)

        headers = [
            "Configuration", "Concordance %", "Kappa", "F1 macro",
            "N2+N3 %", "F1 N3 %", "F1 R %", "Epoques", "Duree s",
        ]
        self._result_table = QtWidgets.QTableWidget(0, len(headers))
        self._result_table.setHorizontalHeaderLabels(headers)
        self._result_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._result_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._result_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._result_table.itemSelectionChanged.connect(self._on_selection_changed)
        lay.addWidget(self._result_table, stretch=1)

        self._summary = QtWidgets.QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(self._summary)

        export = QtWidgets.QHBoxLayout()
        export.addStretch()
        self._export_btn = QtWidgets.QPushButton("Exporter les resultats (CSV)...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        export.addWidget(self._export_btn)
        lay.addLayout(export)
        return w

    # -- Selection ------------------------------------------------------

    def _selected_configs(self) -> List[SweepConfig]:
        if not self._configs:
            return []
        out: List[SweepConfig] = []
        for row, config in enumerate(self._configs):
            item = self._config_table.item(row, 0)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(config)
        return out

    def _update_estimate(self) -> None:
        configs = self._selected_configs()
        runnable = [c for c in configs if not self._blocked.get(c.method)]
        per_run = self._measured_run_seconds()
        seconds = estimate_duration_s(len(runnable), per_run)
        self._run_btn.setEnabled(bool(runnable))
        if not runnable and configs:
            reason = self._blocked.get(self._current_method(), "prerequis manquants")
            self._status.setText(
                f"Rien a lancer : {reason} Choisissez un autre algorithme."
            )
            return
        self._status.setText(
            f"{len(runnable)} configuration(s) selectionnee(s), duree estimee "
            f"{seconds / 60.0:.1f} min (environ {per_run:.0f} s par scoring)."
        )

    def _measured_run_seconds(self) -> float:
        """Prefer the duration measured by the last real run over a guess."""
        params = getattr(self._ctrl, "auto_scoring_params", None) or {}
        measured = params.get("duree_calcul_s")
        try:
            value = float(measured)
        except (TypeError, ValueError):
            return DEFAULT_RUN_SECONDS
        return value if value > 0 else DEFAULT_RUN_SECONDS

    # -- Run ------------------------------------------------------------

    def _on_cancel(self) -> None:
        self._cancelled = True
        self._status.setText("Interruption demandee, fin du scoring en cours...")

    def _on_run(self) -> None:
        configs = self._selected_configs()
        if not configs:
            return
        self._cancelled = False
        self._results = []
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._progress.setRange(0, len(configs))
        self._progress.setValue(0)
        self._progress.setVisible(True)

        def progress(index: int, total: int, config: SweepConfig) -> None:
            self._progress.setValue(index)
            self._status.setText(
                f"Configuration {index + 1}/{total} : {config.label}"
            )
            QtWidgets.QApplication.processEvents()

        try:
            self._results = run_sweep(
                self._ctrl, configs,
                progress_cb=progress,
                should_cancel=lambda: self._cancelled,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Comparaison impossible", str(exc))
            self._status.setText(str(exc))
            return
        except Exception as exc:
            logger.exception("Sweep failed")
            QtWidgets.QMessageBox.warning(self, "Comparaison", f"Echec :\n{exc}")
            self._status.setText(f"Echec : {exc}")
            return
        finally:
            self._progress.setVisible(False)
            self._cancel_btn.setEnabled(False)
            self._run_btn.setEnabled(True)

        self._refresh_results()
        self._tabs.setCurrentIndex(1)

    # -- Results --------------------------------------------------------

    def _refresh_results(self) -> None:
        key = self._rank_key.currentData() or "kappa"
        ranked = rank_results(self._results, key)
        failed = [r for r in self._results if not r.ok]
        rows = ranked + failed
        self._ranked = rows

        self._result_table.setRowCount(len(rows))
        for row, result in enumerate(rows):
            if result.ok:
                values = [
                    result.config.label,
                    f"{result.accuracy:.1f}",
                    f"{result.kappa:.3f}",
                    f"{result.macro_f1:.3f}",
                    f"{result.deep_sleep_accuracy:.1f}",
                    self._fmt_f1(result.stage_f1.get("N3")),
                    self._fmt_f1(result.stage_f1.get("R")),
                    str(result.n_compared),
                    f"{result.duration_s:.0f}",
                ]
            else:
                values = [result.config.label, "echec", "", "", "", "", "", "",
                          f"{result.duration_s:.0f}"]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if not result.ok:
                    item.setToolTip(result.error or "")
                self._result_table.setItem(row, col, item)
        if rows:
            self._result_table.selectRow(0)
        self._export_btn.setEnabled(any(r.ok for r in self._results))

        notes = sweep_summary(self._results, key)
        cancelled = (
            "<br><i>Comparaison interrompue : seules les configurations deja "
            "testees apparaissent.</i>" if self._cancelled else ""
        )
        self._summary.setText("<ul><li>" + "</li><li>".join(notes) + "</li></ul>" + cancelled)

    @staticmethod
    def _fmt_f1(value: Optional[float]) -> str:
        if value is None or value != value:
            return "-"
        return f"{value * 100:.0f}"

    def _on_selection_changed(self) -> None:
        result = self._current_result()
        self._apply_btn.setEnabled(result is not None and result.ok)

    def _current_result(self) -> Optional[SweepResult]:
        rows = getattr(self, "_ranked", [])
        index = self._result_table.currentRow()
        if 0 <= index < len(rows):
            return rows[index]
        return None

    # -- Apply / export -------------------------------------------------

    def _on_apply(self) -> None:
        result = self._current_result()
        if result is None or not result.ok:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Appliquer la configuration",
            f"Appliquer cette configuration ?\n\n{result.config.label}\n\n"
            "L'algorithme, les listes de canaux et les metadonnees du scoring "
            "automatique seront remplaces. Relancez ensuite Sommeil > Scoring "
            "automatique pour produire le scoring correspondant.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        apply_config(self._ctrl, result.config)
        self._applied = result.config
        self.accept()

    def _on_export(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter les resultats", "comparaison_configurations.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        key = self._rank_key.currentData() or "kappa"
        try:
            self._write_csv(path, key)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Echec :\n{exc}")
            return
        QtWidgets.QMessageBox.information(self, "Export", f"Resultats exportes :\n{path}")

    def _write_csv(self, path: str, key: str) -> None:
        import csv

        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "Configuration", "Algorithme", "EEG", "EOG", "EMG", "Age", "Sexe",
                "Concordance %", "Kappa", "F1 macro", "N2+N3 %",
                "F1 N3 %", "F1 R %", "Epoques comparees", "Duree s", "Erreur",
            ])
            for result in rank_results(self._results, key) + [
                r for r in self._results if not r.ok
            ]:
                config = result.config
                sex = "" if config.male is None else ("homme" if config.male else "femme")
                writer.writerow([
                    config.label, config.method, config.eeg,
                    config.eog or "", config.emg or "",
                    config.age if config.age is not None else "", sex,
                    f"{result.accuracy:.2f}" if result.ok else "",
                    f"{result.kappa:.4f}" if result.ok else "",
                    f"{result.macro_f1:.4f}" if result.ok else "",
                    f"{result.deep_sleep_accuracy:.2f}" if result.ok else "",
                    self._fmt_f1(result.stage_f1.get("N3")) if result.ok else "",
                    self._fmt_f1(result.stage_f1.get("R")) if result.ok else "",
                    result.n_compared if result.ok else "",
                    f"{result.duration_s:.1f}",
                    result.error or "",
                ])
            writer.writerow([])
            for note in sweep_summary(self._results, key):
                writer.writerow([note])

    @property
    def applied_config(self) -> Optional[SweepConfig]:
        return self._applied
