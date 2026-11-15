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

import html
import logging
import os
import re
import textwrap
from typing import Callable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

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

# Roles for files in a before/after design.
GROUP_NONE = "-"
GROUP_BEFORE = "AVANT"
GROUP_AFTER = "APRES"

_ROLE_PATH = QtCore.Qt.ItemDataRole.UserRole
_ROLE_GROUP = QtCore.Qt.ItemDataRole.UserRole + 1
_ROLE_SCORING = QtCore.Qt.ItemDataRole.UserRole + 2
_SCORING_MISSING_COLOR = QtGui.QColor("#C47F17")

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


_TIP_WIDTH = 48


def _format_tooltip(text: str) -> str:
    """Wrap a hover tip so the balloon stays a readable block, not one long line.

    A single-line rectangle used to cover neighbouring spin boxes and eat the
    first click; HTML + a max-width keeps the balloon to the right of the
    cursor instead of across the whole dialog.
    """
    tip = " ".join(str(text).split())
    if not tip:
        return ""
    if len(tip) <= _TIP_WIDTH:
        return tip
    wrapped = textwrap.fill(
        tip, width=_TIP_WIDTH,
        break_long_words=False, break_on_hyphens=False,
    )
    body = html.escape(wrapped).replace("\n", "<br/>")
    return (
        "<html><body>"
        f"<p style='margin:0;max-width:280px;white-space:normal;'>{body}</p>"
        "</body></html>"
    )


def _explain(widget: QtWidgets.QWidget, text: str) -> None:
    """Attach a wrapped hover tooltip; empty text is ignored.

    Spin boxes forward the same tip to their inner line edit, otherwise the
    hover is stolen by that child and the explanation never appears.
    """
    tip = _format_tooltip(text)
    if not tip:
        return
    widget.setToolTip(tip)
    line_edit = getattr(widget, "lineEdit", None)
    if callable(line_edit):
        inner = line_edit()
        if inner is not None:
            inner.setToolTip(tip)


def _form_row(
    form: QtWidgets.QFormLayout, label: str, widget: QtWidgets.QWidget, tip: str,
) -> QtWidgets.QLabel:
    """Form row whose field carries the hover explanation.

    The label is left without a tooltip: on Windows a balloon attached to the
    label opens on top of the spin box and swallows the click.
    """
    lbl = QtWidgets.QLabel(label)
    _explain(widget, tip)
    form.addRow(lbl, widget)
    return lbl


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
        self.setWindowTitle("Automatisation en lot - FFT / Spaghetti / PSG / Couplage")
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
        self._tabs.setTabToolTip(
            0,
            _format_tooltip(
                "Ajouter les enregistrements et, si besoin, les marquer "
                "AVANT / APRES (spaghetti et decalage de couplage)."
            ),
        )
        self._tabs.setTabToolTip(
            1,
            _format_tooltip(
                "Choisir le mode. Seuls les parametres du mode courant "
                "sont affiches."
            ),
        )
        self._tabs.setTabToolTip(
            2,
            _format_tooltip(
                "Lancer ou arreter le lot. Le journal note chaque etape "
                "(fichier, scoring, canal)."
            ),
        )

    def _build_files_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self._scan_btn = QtWidgets.QPushButton("Scanner un dossier...")
        self._scan_btn.clicked.connect(self._scan_folder)
        _explain(
            self._scan_btn,
            "Ajoute les EDF du dossier (sous-dossiers inclus). Detecte le "
            "scoring et le groupe AVANT/APRES d'apres le nom, sans lancer "
            "de calcul.",
        )
        self._add_btn = QtWidgets.QPushButton("Ajouter des fichiers...")
        self._add_btn.clicked.connect(self._add_files)
        _explain(
            self._add_btn,
            "Ajoute des fichiers un par un. Le scoring est detecte a l'ajout."
        )
        top.addWidget(self._scan_btn)
        top.addWidget(self._add_btn)
        top.addStretch()
        v.addLayout(top)

        self._files_list = QtWidgets.QListWidget()
        self._files_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        _explain(
            self._files_list,
            "Liste a traiter. Ctrl/Maj pour selectionner. Tags [AVANT]/[APRES] "
            "et [scoring]. Survolez une ligne pour le sidecar."
        )
        v.addWidget(self._files_list, stretch=1)

        marks = QtWidgets.QHBoxLayout()
        self._mark_before_btn = QtWidgets.QPushButton("Marquer AVANT")
        self._mark_before_btn.clicked.connect(
            lambda: self._mark_selected(GROUP_BEFORE)
        )
        _explain(
            self._mark_before_btn,
            "Groupe AVANT : obligatoire pour le spaghetti, sert aussi "
            "a apparier les nuits pour le decalage de phase.",
        )
        self._mark_after_btn = QtWidgets.QPushButton("Marquer APRES")
        self._mark_after_btn.clicked.connect(
            lambda: self._mark_selected(GROUP_AFTER)
        )
        _explain(
            self._mark_after_btn,
            "Attribue le groupe APRES aux fichiers selectionnes.",
        )
        self._mark_none_btn = QtWidgets.QPushButton("Retirer le groupe")
        self._mark_none_btn.clicked.connect(
            lambda: self._mark_selected(GROUP_NONE)
        )
        _explain(
            self._mark_none_btn,
            "Enleve le tag AVANT/APRES. Le fichier reste dans le lot.",
        )
        self._remove_btn = QtWidgets.QPushButton("Supprimer de la liste")
        self._remove_btn.clicked.connect(self._remove_selected)
        _explain(
            self._remove_btn,
            "Retire les fichiers selectionnes de la liste, sans les effacer du disque.",
        )
        marks.addWidget(self._mark_before_btn)
        marks.addWidget(self._mark_after_btn)
        marks.addWidget(self._mark_none_btn)
        marks.addStretch()
        marks.addWidget(self._remove_btn)
        v.addLayout(marks)

        self._files_status = QtWidgets.QLabel(
            "Aucun fichier. Scanner un dossier ou ajouter des enregistrements."
        )
        self._files_status.setWordWrap(True)
        _explain(
            self._files_status,
            "Resume de la liste : nombre de fichiers, combien ont un scoring "
            "detecte, combien de groupes AVANT/APRES. Mis a jour a chaque scan "
            "ou ajout, avant tout calcul.",
        )
        v.addWidget(self._files_status)

        hint = QtWidgets.QLabel(
            "Le groupe AVANT/APRES est detecte automatiquement d'apres le nom du "
            "fichier (ex: _AV/_AP, avant/apres, pre/post, nap1/nap2, baseline/recovery) "
            "et reste modifiable manuellement. Le tag [scoring] / [sans scoring] "
            "est detecte en meme temps (sidecar .csv/.xlsx/.json du meme nom, ou "
            "*Hypnogram*.edf dans le dossier). Survolez une ligne pour le chemin."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)
        return w

    def _build_params_tab(self) -> QtWidgets.QWidget:
        outer = QtWidgets.QWidget()
        outer_l = QtWidgets.QVBoxLayout(outer)
        outer_l.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)

        form = QtWidgets.QFormLayout()
        v.addLayout(form)

        self._mode_combo = QtWidgets.QComboBox()
        mode_tips = [
            "Un CSV de PSD par enregistrement, par canal et par stade.",
            "Spaghetti AVANT vs APRES (marquez les deux groupes).",
            "Une ligne de metriques PSG par fichier.",
            "Couplage SO-fuseau : CSV multi-sujets, plus les figures cochees.",
        ]
        self._mode_combo.addItems([
            "FFT/PSD par canal (CSV)",
            "Spaghetti avant/apres",
            "Metriques PSG (CSV multi-sujets)",
            "Couplage SO-fuseau (CSV multi-sujets)",
        ])
        for i, tip in enumerate(mode_tips):
            self._mode_combo.setItemData(
                i, _format_tooltip(tip), QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        _form_row(
            form, "Mode de sortie:", self._mode_combo,
            "Type de lot. Les parametres FFT ou couplage n'apparaissent "
            "que pour le mode choisi.",
        )

        out_row = QtWidgets.QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        self._out_edit = QtWidgets.QLineEdit()
        _explain(
            self._out_edit,
            "Dossier ou ecrire les CSV et les figures. Cree s'il n'existe pas.",
        )
        self._out_browse_btn = QtWidgets.QPushButton("...")
        self._out_browse_btn.setMaximumWidth(40)
        self._out_browse_btn.clicked.connect(self._browse_output)
        _explain(self._out_browse_btn, "Choisir le dossier de sortie.")
        out_row.addWidget(self._out_edit)
        out_row.addWidget(self._out_browse_btn)
        out_container = QtWidgets.QWidget()
        out_container.setLayout(out_row)
        _form_row(form, "Dossier de sortie:", out_container, "")

        ch_row = QtWidgets.QHBoxLayout()
        ch_row.setContentsMargins(0, 0, 0, 0)
        self._channels_edit = QtWidgets.QLineEdit()
        self._channels_edit.setPlaceholderText("ex: C3-A2, C4-A1, O1-A2")
        _explain(
            self._channels_edit,
            "Canaux separes par des virgules. FFT : obligatoire. "
            "Couplage / spaghetti : vide = tous les EEG.",
        )
        self._detect_btn = QtWidgets.QPushButton("Detecter")
        self._detect_btn.clicked.connect(self._detect_channels)
        _explain(
            self._detect_btn,
            "Lit le premier fichier et propose ses canaux a cocher.",
        )
        ch_row.addWidget(self._channels_edit)
        ch_row.addWidget(self._detect_btn)
        self._channels_widget = QtWidgets.QWidget()
        self._channels_widget.setLayout(ch_row)
        self._channels_label = _form_row(
            form, "Canaux (separes par virgule):", self._channels_widget, "",
        )

        self._epoch_spin = QtWidgets.QDoubleSpinBox()
        self._epoch_spin.setRange(1.0, 120.0)
        self._epoch_spin.setValue(30.0)
        _form_row(
            form, "Duree d'epoque (s):", self._epoch_spin,
            "Epoque de scoring. 30 s = AASM. Lu par tous les modes.",
        )

        self._scoring_chk = QtWidgets.QCheckBox(
            "Utiliser le scoring (sidecar CSV/Excel/JSON/EDF)"
        )
        self._scoring_chk.setChecked(True)
        _explain(
            self._scoring_chk,
            "Hypnogramme a cote de chaque EDF (meme nom .csv/.xlsx/.json, "
            "ou *Hypnogram*.edf). Sans scoring, l'enregistrement entier "
            "est analyse.",
        )
        form.addRow("", self._scoring_chk)
        self._scoring_label = form.labelForField(self._scoring_chk)

        self._fft_box = QtWidgets.QGroupBox("Parametres FFT")
        fft_form = QtWidgets.QFormLayout(self._fft_box)
        _explain(
            self._fft_box,
            "Bornes et estimateur de la PSD. Affiche seulement en FFT "
            "et spaghetti.",
        )

        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(["FFT (magnitude)", "Welch (uV2/Hz)"])
        self._method_combo.setItemData(
            0,
            _format_tooltip("Spectre en magnitude. Plus rapide, moins lisse."),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self._method_combo.setItemData(
            1,
            _format_tooltip(
                "Densite spectrale Welch (uV2/Hz). Plus stable ; "
                "utilise la fenetre nperseg."
            ),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        _form_row(
            fft_form, "Methode:", self._method_combo,
            "Estimateur spectral : FFT brute ou Welch.",
        )

        self._fmin_spin = QtWidgets.QDoubleSpinBox()
        self._fmin_spin.setRange(0.0, 500.0)
        self._fmin_spin.setValue(0.5)
        _form_row(
            fft_form, "Frequence min (Hz):", self._fmin_spin,
            "Borne basse de la PSD. Defaut 0,5 Hz.",
        )

        self._fmax_spin = QtWidgets.QDoubleSpinBox()
        self._fmax_spin.setRange(1.0, 1000.0)
        self._fmax_spin.setValue(45.0)
        _form_row(
            fft_form, "Frequence max (Hz):", self._fmax_spin,
            "Borne haute de la PSD. Defaut 45 Hz (sous le 50 Hz).",
        )

        self._nperseg_spin = QtWidgets.QDoubleSpinBox()
        self._nperseg_spin.setRange(0.5, 30.0)
        self._nperseg_spin.setValue(4.0)
        _form_row(
            fft_form, "Fenetre Welch nperseg (s):", self._nperseg_spin,
            "Longueur des segments Welch. Plus long = plus de resolution "
            "en frequence. Ignore en FFT brute.",
        )

        self._equalize_chk = QtWidgets.QCheckBox(
            "Egaliser le nombre d'epoques par stade"
        )
        self._equalize_chk.setChecked(True)
        _explain(
            self._equalize_chk,
            "Meme nombre d'epoques dans chaque stade (le minimum "
            "disponible) avant de moyenner la PSD.",
        )
        fft_form.addRow("", self._equalize_chk)

        self._robust_chk = QtWidgets.QCheckBox(
            "Statistiques robustes (mediane + MAD)"
        )
        self._robust_chk.setChecked(True)
        _explain(
            self._robust_chk,
            "Mediane ± MAD a la place de moyenne ± ecart-type. "
            "Moins sensible aux artefacts.",
        )
        fft_form.addRow("", self._robust_chk)
        v.addWidget(self._fft_box)

        self._coupling_box = QtWidgets.QGroupBox("Parametres couplage")
        cpl_form = QtWidgets.QFormLayout(self._coupling_box)
        _explain(
            self._coupling_box,
            "Options du couplage SO-fuseau. Masquees en FFT, spaghetti "
            "et PSG.",
        )

        self._min_coupled_spin = QtWidgets.QSpinBox()
        self._min_coupled_spin.setRange(1, 1000)
        self._min_coupled_spin.setValue(MIN_COUPLED_DEFAULT)
        _form_row(
            cpl_form, "Fuseaux couples minimum:", self._min_coupled_spin,
            "En dessous, phase / R / Rayleigh / V / R_z restent vides. "
            "Les effectifs sont quand meme exportes. Defaut 8.",
        )

        self._events_chk = QtWidgets.QCheckBox(
            "CSV d'evenements par fichier (phase de chaque fuseau)"
        )
        _explain(
            self._events_chk,
            "Un {sujet}_coupling_events.csv par enregistrement. "
            "N'allonge presque pas le calcul.",
        )
        cpl_form.addRow("", self._events_chk)

        self._coupling_out_lbl = QtWidgets.QLabel("Sorties optionnelles")
        self._coupling_out_lbl.setStyleSheet("font-weight: 600;")
        _explain(
            self._coupling_out_lbl,
            "Le CSV de stats est toujours ecrit. Cochez seulement "
            "ce que vous voulez en plus.",
        )
        cpl_form.addRow(self._coupling_out_lbl)

        self._perm_chk = QtWidgets.QCheckBox(
            "Force permutee R_z (500 tirages dans les cycles SO)"
        )
        self._perm_chk.setChecked(False)
        _explain(
            self._perm_chk,
            "Z-score de R contre des pics places au hasard dans les "
            "cycles SO. Ajoute r_z et r_z_p. Un peu plus long.",
        )
        cpl_form.addRow("", self._perm_chk)

        self._circ_fig_chk = QtWidgets.QCheckBox(
            "Figures circulaires (phase / decalage)"
        )
        self._circ_fig_chk.setChecked(True)
        _explain(
            self._circ_fig_chk,
            "coupling_phases.png et coupling_decalage.png a cote du CSV. "
            "Calcule depuis la table, sans relire les EDF.",
        )
        cpl_form.addRow("", self._circ_fig_chk)

        self._avg_fig_chk = QtWidgets.QCheckBox(
            "Moyenne fuseau-verrouillee + histogramme peri-evenement"
        )
        self._avg_fig_chk.setChecked(False)
        _explain(
            self._avg_fig_chk,
            "Grand average de cohorte ±2,5 s autour du pic des fuseaux. "
            "Relit le signal. Decoche par defaut.",
        )
        cpl_form.addRow("", self._avg_fig_chk)

        self._tfr_chk = QtWidgets.QCheckBox(
            "TFR alignee sur le creux de l'onde lente (lourd)"
        )
        self._tfr_chk.setChecked(False)
        _explain(
            self._tfr_chk,
            "Spectrogramme 5-30 Hz autour de chaque creux SO. "
            "Calcul le plus long : un PNG par cohorte.",
        )
        cpl_form.addRow("", self._tfr_chk)
        v.addWidget(self._coupling_box)

        self._mode_hint = QtWidgets.QLabel()
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        v.addWidget(self._mode_hint)
        v.addStretch()

        scroll.setWidget(inner)
        outer_l.addWidget(scroll)
        self._on_mode_changed()
        return outer

    def _build_run_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        btns = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Lancer")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._start)
        _explain(
            self._run_btn,
            "Lance le mode choisi sur tous les fichiers. Un dossier de "
            "sortie est obligatoire.",
        )
        self._stop_btn = QtWidgets.QPushButton("Arreter")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        _explain(
            self._stop_btn,
            "Arrete a la fin du fichier en cours. Le CSV deja ecrit est conserve."
        )
        btns.addWidget(self._run_btn)
        btns.addWidget(self._stop_btn)
        btns.addStretch()
        v.addLayout(btns)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        _explain(self._progress, "Avancement du lot, en pourcentage de fichiers.")
        v.addWidget(self._progress)

        self._status_lbl = QtWidgets.QLabel("Pret.")
        self._status_lbl.setWordWrap(True)
        _explain(
            self._status_lbl,
            "Etape en cours. La derniere ligne du journal dit ou le calcul "
            "s'est arrete si CESA se ferme.",
        )
        v.addWidget(self._status_lbl)

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        _explain(
            self._log,
            "Journal du lot : une ligne par etape (chargement, scoring, "
            "canal). ! = avertissement, ERREUR = echec.",
        )
        v.addWidget(self._log, stretch=1)
        return w

    # ------------------------------------------------------------------
    # File list helpers
    # ------------------------------------------------------------------

    def _existing_paths(self) -> set:
        return {
            self._files_list.item(i).data(_ROLE_PATH)
            for i in range(self._files_list.count())
        }

    def _add_path(self, path: str, group: Optional[str] = None) -> None:
        if path in self._existing_paths():
            return
        if group is None:
            group = detect_group_from_name(path)
        from CESA.batch_engine import find_scoring_sidecar

        sidecar = find_scoring_sidecar(path) or ""
        item = QtWidgets.QListWidgetItem()
        item.setData(_ROLE_PATH, path)
        item.setData(_ROLE_GROUP, group)
        item.setData(_ROLE_SCORING, sidecar)
        self._apply_item_display(item)
        self._files_list.addItem(item)

    @staticmethod
    def _format_item(path: str, group: str, has_scoring: bool = False) -> str:
        tags = []
        if group != GROUP_NONE:
            tags.append(group)
        tags.append("scoring" if has_scoring else "sans scoring")
        tag = "  ".join(f"[{t}]" for t in tags)
        return f"{os.path.basename(path)}  {tag}"

    def _apply_item_display(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(_ROLE_PATH)
        group = item.data(_ROLE_GROUP)
        sidecar = item.data(_ROLE_SCORING) or ""
        item.setText(self._format_item(path, group, bool(sidecar)))
        if sidecar:
            item.setToolTip(_format_tooltip(f"Scoring detecte : {sidecar}"))
            item.setForeground(self._files_list.palette().color(
                QtGui.QPalette.ColorRole.Text,
            ))
        else:
            item.setToolTip(_format_tooltip(
                "Aucun scoring a cote de ce fichier. FFT et couplage "
                "analyseront l'enregistrement entier."
            ))
            item.setForeground(_SCORING_MISSING_COLOR)

    def _refresh_files_status(self) -> None:
        n = self._files_list.count()
        with_scoring = sum(
            1 for i in range(n)
            if self._files_list.item(i).data(_ROLE_SCORING)
        )
        without = n - with_scoring
        before = len(self._files_in_group(GROUP_BEFORE))
        after = len(self._files_in_group(GROUP_AFTER))
        if n == 0:
            msg = "Aucun fichier. Scanner un dossier ou ajouter des enregistrements."
        else:
            msg = (
                f"{n} fichier(s) — {with_scoring} avec scoring, "
                f"{without} sans scoring"
            )
            if before or after:
                msg += f" — {before} AVANT, {after} APRES"
            if without:
                msg += (
                    "  |  Les fichiers sans scoring seront analyses "
                    "sans restriction de stade."
                )
        self._files_status.setText(msg)
        self._status_lbl.setText(msg)

    def _scan_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir un dossier")
        if not directory:
            return
        exts = tuple(e.lower() for e in recording_extensions_for_scan())
        for root, _dirs, names in os.walk(directory):
            for name in names:
                if name.lower().endswith(exts) and "hypnogram" not in name.lower():
                    grp = detect_group_from_name(name)
                    self._add_path(os.path.join(root, name), group=grp)
        self._refresh_files_status()

    def _add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in recording_extensions_for_scan())
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choisir des enregistrements", "",
            f"Enregistrements ({exts});;Tous les fichiers (*.*)",
        )
        for p in paths:
            self._add_path(p)
        if paths:
            self._refresh_files_status()

    def _mark_selected(self, group: str) -> None:
        for item in self._files_list.selectedItems():
            item.setData(_ROLE_GROUP, group)
            self._apply_item_display(item)
        self._refresh_files_status()

    def _remove_selected(self) -> None:
        for item in self._files_list.selectedItems():
            self._files_list.takeItem(self._files_list.row(item))
        self._refresh_files_status()

    def _all_files(self) -> List[str]:
        return [
            self._files_list.item(i).data(_ROLE_PATH)
            for i in range(self._files_list.count())
        ]

    def _files_in_group(self, group: str) -> List[str]:
        out = []
        for i in range(self._files_list.count()):
            item = self._files_list.item(i)
            if item.data(_ROLE_GROUP) == group:
                out.append(item.data(_ROLE_PATH))
        return out

    # ------------------------------------------------------------------
    # Param helpers
    # ------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        mode = self._mode_combo.currentIndex()
        is_psg = mode == MODE_PSG
        is_spaghetti = mode == MODE_SPAGHETTI
        is_coupling = mode == MODE_COUPLING
        is_spectral = mode in (MODE_FFT, MODE_SPAGHETTI)
        # Hide unused blocks instead of greying them: FFT fields stay clickable
        # in FFT/spaghetti, and disappear entirely in coupling/PSG.
        self._fft_box.setVisible(is_spectral)
        self._coupling_box.setVisible(is_coupling)
        self._channels_widget.setVisible(not is_psg)
        self._channels_label.setVisible(not is_psg)
        self._scoring_chk.setVisible(not is_spaghetti)
        if self._scoring_label is not None:
            self._scoring_label.setVisible(not is_spaghetti)
        if is_coupling:
            self._mode_hint.setText(
                "Le CSV de statistiques (effectifs, pourcentages, densite, "
                "amplitude, phase, R, Rayleigh, test V) est toujours ecrit. "
                "Cochez ci-dessus uniquement les calculs et figures en plus. "
                "Survolez une case pour le detail et le cout. La TFR est le "
                "calcul le plus long."
            )
        elif is_psg:
            self._mode_hint.setText(
                "Une ligne de metriques PSG par fichier (durees de stades, "
                "efficacite, latences)."
            )
        elif is_spaghetti:
            self._mode_hint.setText(
                "Graphiques spaghetti AVANT/APRÈS : marquez au moins un fichier "
                "dans chaque groupe a l'onglet Fichiers."
            )
        else:
            self._mode_hint.setText(
                "PSD par canal et par stade, un CSV par enregistrement."
            )

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
            elif mode == MODE_COUPLING:
                job = self._make_coupling_job(out_dir)
            else:
                job = self._make_psg_job(out_dir)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Configuration", str(exc))
            return

        self._tabs.setCurrentIndex(2)
        self._log.clear()
        self._progress.setValue(0)
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
        self._status_lbl.setText(msg)
        self._append_log(msg)

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

    def _make_coupling_job(self, out_dir: str) -> Callable:
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
            compute_permutation=self._perm_chk.isChecked(),
            export_figures=self._circ_fig_chk.isChecked(),
            export_event_averages=self._avg_fig_chk.isChecked(),
            export_tfr=self._tfr_chk.isChecked(),
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
