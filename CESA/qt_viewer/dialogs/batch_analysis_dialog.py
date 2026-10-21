"""Qt-native batch analysis hub (FFT/PSD CSV, spaghetti, PSG metrics).

Reimplements the old Tk "Automatisation FFT en Lot" window natively for the Qt
viewer, driving the shared :mod:`CESA.batch_engine` and
:mod:`CESA.advanced_spaghetti_plots` without any GUI coupling in the maths.

Tabs
----
1. Fichiers   -- folder scan, multi-select, AVANT/APRES group marking
2. Parametres -- output mode + analysis parameters + channels
3. Traitement -- run/stop, progress bar, log
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable, List, Optional

from PySide6 import QtCore, QtWidgets

from core.raw_loader import open_raw_file, recording_extensions_for_scan
from CESA.sleep_pipeline.so_spindle_coupling import (
    MIN_COUPLED_EVENTS as MIN_COUPLED_DEFAULT,
)

logger = logging.getLogger(__name__)

# Output modes, in combo order.
MODE_FFT = 0
MODE_SPAGHETTI = 1
MODE_PSG = 2
MODE_COUPLING = 3
MODE_COUPLING_HEADBAND = 4

# Roles for files in a before/after design.
GROUP_NONE = "-"
GROUP_BEFORE = "AVANT"
GROUP_AFTER = "APRES"

# Keyword heuristics to auto-detect the AVANT/APRES group from a file name.
# Matched against the upper-cased name as whole tokens (separated by
# non-alphanumeric chars) to avoid false positives.
_BEFORE_TOKENS = {
    "AV", "AVANT", "BEFORE", "PRE", "BASELINE", "BASE", "NAP1", "J1", "T0",
    "PRETEST", "PRENAP",
}
_AFTER_TOKENS = {
    "AP", "APRES", "APRÈS", "AFTER", "POST", "RECOVERY", "RECUP", "NAP2",
    "J2", "T1", "POSTTEST", "POSTNAP",
}


def detect_group_from_name(name: str) -> str:
    """Infer the AVANT/APRES group from a recording file name.

    Returns :data:`GROUP_BEFORE`, :data:`GROUP_AFTER` or :data:`GROUP_NONE`.
    """
    stem = os.path.splitext(os.path.basename(name))[0].upper()
    tokens = set(re.split(r"[^A-Z0-9ÈÉÀ]+", stem))
    tokens.discard("")
    if tokens & _BEFORE_TOKENS:
        return GROUP_BEFORE
    if tokens & _AFTER_TOKENS:
        return GROUP_AFTER
    return GROUP_NONE


class _BatchWorker(QtCore.QThread):
    """Runs a blocking batch *job* in the background.

    The job receives ``(progress_cb, should_stop)`` and may return any object,
    which is forwarded through :pyattr:`finished_result`.
    """

    progress = QtCore.Signal(float, str)
    finished_result = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, job: Callable, parent=None) -> None:
        super().__init__(parent)
        self._job = job
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            result = self._job(
                lambda pct, msg: self.progress.emit(float(pct), str(msg)),
                lambda: self._stop,
            )
            self.finished_result.emit(result)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            logger.error("Batch worker error: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class BatchAnalysisDialog(QtWidgets.QDialog):
    """Tabbed dialog orchestrating the batch pipelines."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Automatisation en lot - FFT / Spaghetti / PSG")
        self.resize(960, 720)
        self._worker: Optional[_BatchWorker] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs, stretch=1)
        self._tabs.addTab(self._build_files_tab(), "1. Fichiers")
        self._tabs.addTab(self._build_params_tab(), "2. Parametres")
        self._tabs.addTab(self._build_run_tab(), "3. Traitement")

    def _build_files_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        scan_btn = QtWidgets.QPushButton("Scanner un dossier...")
        scan_btn.clicked.connect(self._scan_folder)
        add_btn = QtWidgets.QPushButton("Ajouter des fichiers...")
        add_btn.clicked.connect(self._add_files)
        top.addWidget(scan_btn)
        top.addWidget(add_btn)
        top.addStretch()
        v.addLayout(top)

        self._files_list = QtWidgets.QListWidget()
        self._files_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        v.addWidget(self._files_list, stretch=1)

        marks = QtWidgets.QHBoxLayout()
        mark_before = QtWidgets.QPushButton("Marquer AVANT")
        mark_before.clicked.connect(lambda: self._mark_selected(GROUP_BEFORE))
        mark_after = QtWidgets.QPushButton("Marquer APRES")
        mark_after.clicked.connect(lambda: self._mark_selected(GROUP_AFTER))
        mark_none = QtWidgets.QPushButton("Retirer le groupe")
        mark_none.clicked.connect(lambda: self._mark_selected(GROUP_NONE))
        remove_btn = QtWidgets.QPushButton("Supprimer de la liste")
        remove_btn.clicked.connect(self._remove_selected)
        marks.addWidget(mark_before)
        marks.addWidget(mark_after)
        marks.addWidget(mark_none)
        marks.addStretch()
        marks.addWidget(remove_btn)
        v.addLayout(marks)

        hint = QtWidgets.QLabel(
            "Le groupe AVANT/APRES est detecte automatiquement d'apres le nom du "
            "fichier (ex: _AV/_AP, avant/apres, pre/post, nap1/nap2, baseline/recovery)"
            " et reste modifiable manuellement. Utilise pour le mode 'Spaghetti'."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        return w

    def _build_params_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)

        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems([
            "FFT/PSD par canal (CSV)",
            "Spaghetti avant/apres",
            "Metriques PSG (CSV multi-sujets)",
            "Couplage SO-fuseau (CSV multi-sujets)",
            "Couplage SO-fuseau bandeau (CSV multi-sujets)",
        ])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Mode de sortie:", self._mode_combo)

        self._mode_hint = QtWidgets.QLabel()
        self._mode_hint.setWordWrap(True)
        form.addRow("", self._mode_hint)

        # Output directory
        out_row = QtWidgets.QHBoxLayout()
        self._out_edit = QtWidgets.QLineEdit()
        out_browse = QtWidgets.QPushButton("...")
        out_browse.setMaximumWidth(40)
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit)
        out_row.addWidget(out_browse)
        out_container = QtWidgets.QWidget()
        out_container.setLayout(out_row)
        form.addRow("Dossier de sortie:", out_container)

        # Channels
        ch_row = QtWidgets.QHBoxLayout()
        self._channels_edit = QtWidgets.QLineEdit()
        self._channels_edit.setPlaceholderText("ex: C3-A2, C4-A1, O1-A2")
        detect_btn = QtWidgets.QPushButton("Detecter")
        detect_btn.clicked.connect(self._detect_channels)
        ch_row.addWidget(self._channels_edit)
        ch_row.addWidget(detect_btn)
        ch_container = QtWidgets.QWidget()
        ch_container.setLayout(ch_row)
        form.addRow("Canaux (separes par virgule):", ch_container)
        self._channels_widget = ch_container

        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(["FFT (magnitude)", "Welch (uV2/Hz)"])
        form.addRow("Methode:", self._method_combo)

        self._fmin_spin = QtWidgets.QDoubleSpinBox()
        self._fmin_spin.setRange(0.0, 500.0)
        self._fmin_spin.setValue(0.5)
        form.addRow("Frequence min (Hz):", self._fmin_spin)

        self._fmax_spin = QtWidgets.QDoubleSpinBox()
        self._fmax_spin.setRange(1.0, 1000.0)
        self._fmax_spin.setValue(45.0)
        form.addRow("Frequence max (Hz):", self._fmax_spin)

        self._epoch_spin = QtWidgets.QDoubleSpinBox()
        self._epoch_spin.setRange(1.0, 120.0)
        self._epoch_spin.setValue(30.0)
        form.addRow("Duree d'epoque (s):", self._epoch_spin)

        self._nperseg_spin = QtWidgets.QDoubleSpinBox()
        self._nperseg_spin.setRange(0.5, 30.0)
        self._nperseg_spin.setValue(4.0)
        form.addRow("Fenetre Welch nperseg (s):", self._nperseg_spin)

        self._equalize_chk = QtWidgets.QCheckBox("Egaliser le nombre d'epoques par stade")
        self._equalize_chk.setChecked(True)
        form.addRow("", self._equalize_chk)

        self._robust_chk = QtWidgets.QCheckBox("Statistiques robustes (mediane + MAD)")
        self._robust_chk.setChecked(True)
        form.addRow("", self._robust_chk)

        self._scoring_chk = QtWidgets.QCheckBox("Utiliser le scoring (sidecar CSV/Excel/JSON/EDF)")
        self._scoring_chk.setChecked(True)
        form.addRow("", self._scoring_chk)

        self._min_coupled_spin = QtWidgets.QSpinBox()
        self._min_coupled_spin.setRange(1, 1000)
        self._min_coupled_spin.setValue(MIN_COUPLED_DEFAULT)
        self._min_coupled_spin.setToolTip(
            "En dessous de ce nombre de fuseaux couples, la phase preferee "
            "n'est pas publiee pour le canal (effectifs quand meme exportes)."
        )
        form.addRow("Fuseaux couples minimum:", self._min_coupled_spin)

        self._events_chk = QtWidgets.QCheckBox(
            "Exporter aussi un CSV d'evenements par fichier (phase de chaque fuseau)"
        )
        form.addRow("", self._events_chk)

        self._figures_chk = QtWidgets.QCheckBox(
            "Tracer les figures circulaires (phase par sujet, decalage AVANT/APRES)"
        )
        self._figures_chk.setChecked(True)
        self._figures_chk.setToolTip(
            "Figures polaires PNG a cote du CSV : une phase preferee par sujet "
            "et par canal, puis le decalage apparie si les deux conditions "
            "sont marquees dans l'onglet Fichiers."
        )
        form.addRow("", self._figures_chk)

        self._params_form = form
        self._on_mode_changed()
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
        self._progress.setRange(0, 100)
        v.addWidget(self._progress)

        self._status_lbl = QtWidgets.QLabel("Pret.")
        self._status_lbl.setWordWrap(True)
        v.addWidget(self._status_lbl)
        self._clock = QtCore.QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick_clock)
        self._run_t0 = 0.0
        self._step_t0 = 0.0
        self._step_msg = ""

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        v.addWidget(self._log, stretch=1)
        return w

    # ------------------------------------------------------------------
    # File list helpers
    # ------------------------------------------------------------------

    def _existing_paths(self) -> set:
        return {
            self._files_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(self._files_list.count())
        }

    def _add_path(self, path: str, group: Optional[str] = None) -> None:
        if path in self._existing_paths():
            return
        if group is None:
            group = detect_group_from_name(path)
        item = QtWidgets.QListWidgetItem(self._format_item(path, group))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
        item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, group)
        self._files_list.addItem(item)

    @staticmethod
    def _format_item(path: str, group: str) -> str:
        tag = "" if group == GROUP_NONE else f"  [{group}]"
        return f"{os.path.basename(path)}{tag}"

    def _scan_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir un dossier")
        if not directory:
            return
        exts = tuple(e.lower() for e in recording_extensions_for_scan())
        found = 0
        before = after = 0
        for root, _dirs, names in os.walk(directory):
            for name in names:
                if name.lower().endswith(exts) and "hypnogram" not in name.lower():
                    grp = detect_group_from_name(name)
                    self._add_path(os.path.join(root, name), group=grp)
                    found += 1
                    if grp == GROUP_BEFORE:
                        before += 1
                    elif grp == GROUP_AFTER:
                        after += 1
        self._status_lbl.setText(
            f"{found} fichier(s) trouve(s) - auto: {before} AVANT, {after} APRES "
            f"(modifiable manuellement)."
        )

    def _add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in recording_extensions_for_scan())
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choisir des enregistrements", "",
            f"Enregistrements ({exts});;Tous les fichiers (*.*)",
        )
        for p in paths:
            self._add_path(p)

    def _mark_selected(self, group: str) -> None:
        for item in self._files_list.selectedItems():
            path = item.data(QtCore.Qt.ItemDataRole.UserRole)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, group)
            item.setText(self._format_item(path, group))

    def _remove_selected(self) -> None:
        for item in self._files_list.selectedItems():
            self._files_list.takeItem(self._files_list.row(item))

    def _all_files(self) -> List[str]:
        return [
            self._files_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(self._files_list.count())
        ]

    def _files_in_group(self, group: str) -> List[str]:
        out = []
        for i in range(self._files_list.count()):
            item = self._files_list.item(i)
            if item.data(QtCore.Qt.ItemDataRole.UserRole + 1) == group:
                out.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return out

    # ------------------------------------------------------------------
    # Param helpers
    # ------------------------------------------------------------------

    def _show_param(self, widget: QtWidgets.QWidget, visible: bool) -> None:
        self._params_form.setRowVisible(widget, visible)

    def _on_mode_changed(self) -> None:
        mode = self._mode_combo.currentIndex()
        is_fft = mode == MODE_FFT
        is_spaghetti = mode == MODE_SPAGHETTI
        is_psg = mode == MODE_PSG
        is_headband = mode == MODE_COUPLING_HEADBAND
        is_coupling = mode == MODE_COUPLING or is_headband
        # Only the fields the selected analysis actually reads stay on screen.
        self._show_param(self._channels_widget, not is_psg)
        self._show_param(self._method_combo, is_fft)
        self._show_param(self._fmin_spin, is_fft)
        self._show_param(self._fmax_spin, is_fft)
        self._show_param(self._nperseg_spin, is_fft)
        self._show_param(self._equalize_chk, is_fft)
        self._show_param(self._robust_chk, is_fft)
        self._show_param(self._epoch_spin, True)
        self._show_param(self._scoring_chk, is_fft or is_coupling)
        self._show_param(self._min_coupled_spin, is_coupling)
        self._show_param(self._events_chk, is_coupling)
        self._show_param(self._figures_chk, is_coupling)
        hints = {
            MODE_FFT: "FFT/PSD : methode, bandes de frequence, epoques et scoring.",
            MODE_SPAGHETTI: (
                "Spaghetti : canaux, duree d'epoque, et au moins un fichier "
                "marque AVANT et un fichier marque APRES."
            ),
            MODE_PSG: (
                "Metriques PSG : duree d'epoque. Le scoring sidecar est "
                "obligatoire, les canaux ne sont pas utilises."
            ),
            MODE_COUPLING: (
                "Couplage SO-fuseau : seuls les canaux EEG sont analyses "
                "(EOG, ECG et frequence cardiaque sont ignores). Champ vide "
                "= tous les EEG. Le scoring est cherche a cote du fichier, "
                "puis dans les annotations de l'EDF. Chaque canal d'une nuit "
                "complete peut prendre plusieurs minutes."
            ),
            MODE_COUPLING_HEADBAND: (
                "Couplage SO-fuseau bandeau : seules les derivations frontales "
                "sont analysees (champ vide = toutes les frontales) et les "
                "clignements sont masques, sans canal EOG. Sortie dans "
                "so_spindle_coupling_headband.csv, avec les colonnes "
                "n_blinks et blink_rejected_s."
            ),
        }
        self._mode_hint.setText(hints.get(mode, ""))

    def _browse_output(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if directory:
            self._out_edit.setText(directory)

    def _detect_channels(self) -> None:
        files = self._all_files()
        if not files:
            QtWidgets.QMessageBox.information(self, "Detection", "Ajoutez d'abord des fichiers.")
            return
        try:
            raw = open_raw_file(files[0], preload=False, verbose="ERROR")
            channels = list(raw.ch_names)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Detection", f"Lecture impossible:\n{exc}")
            return
        picked = self._pick_channels(channels)
        if picked is not None:
            self._channels_edit.setText(", ".join(picked))

    def _pick_channels(self, channels: List[str]) -> Optional[List[str]]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Selection des canaux")
        dlg.resize(360, 480)
        lay = QtWidgets.QVBoxLayout(dlg)
        lst = QtWidgets.QListWidget()
        for ch in channels:
            it = QtWidgets.QListWidgetItem(ch)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            lst.addItem(it)
        lay.addWidget(lst)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return [
            lst.item(i).text()
            for i in range(lst.count())
            if lst.item(i).checkState() == QtCore.Qt.CheckState.Checked
        ]

    def _parsed_channels(self) -> List[str]:
        raw = self._channels_edit.text().strip()
        return [c.strip() for c in raw.split(",") if c.strip()]

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "Sortie", "Choisissez un dossier de sortie.")
            return

        mode = self._mode_combo.currentIndex()
        try:
            if mode == MODE_FFT:
                job = self._make_fft_job(out_dir)
            elif mode == MODE_SPAGHETTI:
                job = self._make_spaghetti_job(out_dir)
            elif mode in (MODE_COUPLING, MODE_COUPLING_HEADBAND):
                job = self._make_coupling_job(
                    out_dir, headband=mode == MODE_COUPLING_HEADBAND,
                )
            else:
                job = self._make_psg_job(out_dir)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Configuration", str(exc))
            return

        self._tabs.setCurrentIndex(2)
        self._log.clear()
        self._progress.setValue(0)
        self._step_msg = ""
        self._run_t0 = time.time()
        self._step_t0 = self._run_t0
        self._clock.start()
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = _BatchWorker(job, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._append_log("Arret demande...")

    def _on_progress(self, pct: float, msg: str) -> None:
        self._progress.setValue(int(pct))
        if msg != self._step_msg:
            self._step_msg = msg
            self._step_t0 = time.time()
            self._append_log(msg)
        self._tick_clock()

    def _tick_clock(self) -> None:
        if not self._step_msg or self._run_t0 <= 0:
            return
        self._status_lbl.setText(
            f"{self._step_msg}  —  {time.time() - self._step_t0:.0f} s "
            f"sur cette etape, {time.time() - self._run_t0:.0f} s au total"
        )

    def _on_failed(self, msg: str) -> None:
        self._append_log(f"ERREUR: {msg}")
        self._status_lbl.setText("Echec.")
        self._finish_ui()

    def _on_finished(self, result) -> None:
        from CESA.batch_engine import BatchResult

        if isinstance(result, BatchResult):
            self._append_log(
                f"Termine: {result.processed} reussi(s), {result.failed} echec(s)."
            )
            for err in result.errors:
                self._append_log(f"  - {err}")
            for warn in getattr(result, "warnings", []):
                self._append_log(f"  ! {warn}")
            for out in result.outputs:
                self._append_log(f"  -> {out}")
        elif isinstance(result, (list, tuple)):
            self._append_log(f"Termine: {len(result)} fichier(s) genere(s).")
            for out in result:
                self._append_log(f"  -> {out}")
        else:
            self._append_log("Termine.")
        self._status_lbl.setText("Termine.")
        self._finish_ui()

    def _finish_ui(self) -> None:
        self._clock.stop()
        self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Job factories
    # ------------------------------------------------------------------

    def _make_fft_job(self, out_dir: str) -> Callable:
        from CESA.batch_engine import BatchFFTConfig, run_batch_fft

        files = self._all_files()
        if not files:
            raise ValueError("Aucun fichier a traiter.")
        channels = self._parsed_channels()
        if not channels:
            raise ValueError("Indiquez au moins un canal.")
        config = BatchFFTConfig(
            output_dir=out_dir,
            method="welch" if self._method_combo.currentIndex() == 1 else "fft",
            fmin=self._fmin_spin.value(),
            fmax=self._fmax_spin.value(),
            epoch_len=self._epoch_spin.value(),
            nperseg_sec=self._nperseg_spin.value(),
            equalize_epochs=self._equalize_chk.isChecked(),
            robust_stats=self._robust_chk.isChecked(),
            include_scoring=self._scoring_chk.isChecked(),
        )

        def job(progress_cb, should_stop):
            return run_batch_fft(
                files, channels, config,
                progress_cb=progress_cb, should_stop=should_stop,
            )

        return job

    def _make_psg_job(self, out_dir: str) -> Callable:
        from CESA.batch_engine import BatchPSGConfig, run_batch_psg

        files = self._all_files()
        if not files:
            raise ValueError("Aucun fichier a traiter.")
        config = BatchPSGConfig(output_dir=out_dir, epoch_seconds=self._epoch_spin.value())

        def job(progress_cb, should_stop):
            return run_batch_psg(
                files, config, progress_cb=progress_cb, should_stop=should_stop,
            )

        return job

    def _make_coupling_job(self, out_dir: str, *, headband: bool = False) -> Callable:
        from CESA.batch_engine import (
            BatchCouplingConfig, run_batch_so_spindle_coupling,
        )

        files = self._all_files()
        if not files:
            raise ValueError("Aucun fichier a traiter.")
        config = BatchCouplingConfig(
            output_dir=out_dir,
            channels=self._parsed_channels(),
            epoch_seconds=self._epoch_spin.value(),
            include_scoring=self._scoring_chk.isChecked(),
            min_coupled=self._min_coupled_spin.value(),
            export_events=self._events_chk.isChecked(),
            export_figures=self._figures_chk.isChecked(),
            headband=headband,
        )
        groups = {
            path: group
            for group in (GROUP_BEFORE, GROUP_AFTER)
            for path in self._files_in_group(group)
        }

        def job(progress_cb, should_stop):
            return run_batch_so_spindle_coupling(
                files, config,
                groups=groups,
                progress_cb=progress_cb, should_stop=should_stop,
            )

        return job

    def _make_spaghetti_job(self, out_dir: str) -> Callable:
        from CESA.advanced_spaghetti_plots import generate_spaghetti_from_edf_file_lists

        before = self._files_in_group(GROUP_BEFORE)
        after = self._files_in_group(GROUP_AFTER)
        if not before or not after:
            raise ValueError(
                "Marquez au moins un fichier AVANT et un fichier APRES (onglet Fichiers)."
            )
        channels = self._parsed_channels() or None

        def job(progress_cb, should_stop):
            progress_cb(5.0, "Generation des graphiques spaghetti...")
            outputs = generate_spaghetti_from_edf_file_lists(
                before_files=before,
                after_files=after,
                output_dir=out_dir,
                selected_channels=channels,
                epoch_len=self._epoch_spin.value(),
            )
            progress_cb(100.0, "Termine")
            return outputs

        return job
