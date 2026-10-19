"""Qt dialog: advanced automatic-scoring configuration.

Lets the user pick the scoring backend and tune the parameters consumed by
``AppController.run_auto_scoring`` before launching, with a contextual help
panel that documents each backend (principle, required channels, prerequisites,
expected performance) and checks the loaded recording against the channels the
selected backend actually needs.

Only the parameters relevant to the selected backend are shown, and a
pre-flight check blocks backends that cannot possibly run (missing package,
missing model weights, no matching EEG channel) instead of letting the scoring
fail with an opaque error.

Replaces the Tk ``_open_sleep_scoring_settings`` window.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

from PySide6 import QtCore, QtWidgets

from CESA.scoring_channels import (
    emg_warning,
    find_submental_emg_candidates,
    order_channel_selection,
)

logger = logging.getLogger(__name__)

#: Human-readable backend names, shared with the sweep dialog.
METHOD_LABELS = {
    "pftsleep": "PFTSleep (deep learning multicanal)",
    "yasa": "YASA (features + gradient boosting)",
    "usleep": "U-Sleep (deep learning)",
    "aasm_rules": "Regles AASM (rule-based)",
    "ml": "ML (gradient boosting CESA)",
    "ml_hmm": "ML + HMM (modele de sequence)",
    "rules_hmm": "Regles AASM + HMM",
}

#: Per-backend documentation shown in the help panel.
#:
#: ``eeg``/``eog``/``emg``/``other`` describe the role of each modality for that
#: backend: "requis", "recommande" or "optionnel".
_METHOD_INFO: Dict[str, Dict[str, object]] = {
    "yasa": {
        "principe": (
            "Extraction de descripteurs par epoque (puissances spectrales, "
            "entropies, statistiques temporelles, lissage du contexte voisin) "
            "puis classification par gradient boosting pre-entraine sur plusieurs "
            "milliers de nuits."
        ),
        "eeg": "requis",
        "eog": "recommande",
        "emg": "recommande",
        "canaux": (
            "1 EEG central suffit (C4-M1, sinon C3-M2). Ajouter 1 EOG et 1 EMG "
            "menton ameliore surtout N1 et REM."
        ),
        "prerequis": "Paquet Python 'yasa' (pip install yasa). Pas de GPU necessaire.",
        "epoque": "30 s obligatoire en pratique (le modele a ete entraine ainsi).",
        "perf": (
            "Accord typique avec un scoreur humain : 80-88 % d'epoques, kappa "
            "0,70-0,80. Point faible attendu : N1."
        ),
        "quand": (
            "Choix par defaut recommande pour un montage PSG classique : rapide, "
            "robuste, sans poids a telecharger, et fournit une confiance par epoque."
        ),
        "limites": (
            "Sensible a la qualite de la derivation EEG choisie ; les artefacts ne "
            "sont pas rejetes automatiquement."
        ),
        "params": ("epoch", "sfreq", "conf", "age", "sex"),
    },
    "pftsleep": {
        "principe": (
            "Transformer applique a la nuit entiere, entraine en auto-supervise "
            "sur de la PSG multicanal ; exploite EEG, EOG, EMG, ECG et, quand ils "
            "existent, SpO2 et sangles respiratoires."
        ),
        "eeg": "requis",
        "eog": "recommande",
        "emg": "recommande",
        "other": "ECG recommande ; SpO2 et sangles thoraciques/abdominales optionnelles",
        "canaux": (
            "Derivations referencees attendues : EEG C4-M1 (ou C3-M2), EOG gauche "
            "E1-M2, EMG Chin1-Chin2, ECG. Les canaux absents sont remplaces par un "
            "vecteur nul ('dummy'), ce qui degrade la prediction sans la bloquer."
        ),
        "prerequis": (
            "PyTorch + paquet 'PFTSleep' installe, poids du modele presents dans le "
            "dossier indique (telechargement Hugging Face, depot soumis a "
            "acceptation : un token peut etre necessaire). Relit le fichier EDF "
            "d'origine sur le disque."
        ),
        "epoque": "30 s impose par le modele.",
        "perf": (
            "Le plus performant des backends disponibles quand tous les canaux sont "
            "presents (kappa souvent > 0,80), au prix d'un calcul bien plus long."
        ),
        "quand": (
            "Montage PSG complet (EEG + EOG + EMG + ECG) et temps de calcul "
            "acceptable ; utile comme second avis face a YASA."
        ),
        "limites": (
            "Lourd (GPU conseille) ; inutilisable sur un montage reduit a l'EEG, ou "
            "les canaux 'dummy' dominent l'entree."
        ),
        "params": ("epoch", "pft"),
    },
    "usleep": {
        "principe": (
            "Reseau convolutif entierement convolutionnel (U-Net temporel) "
            "appliquant une segmentation dense du signal, robuste aux montages "
            "et aux frequences d'echantillonnage varies. Les poids publies ne "
            "sont pas distribues : CESA appelle le service officiel "
            "sleep.ai.ku.dk (U-Sleep v1.0)."
        ),
        "eeg": "requis",
        "eog": "recommande",
        "emg": "non utilise",
        "canaux": (
            "1 EEG (F4-M1 en priorite) et 1 EOG (E1-M2). Seuls ces deux canaux "
            "sont extraits et envoyes, pas le reste de la PSG."
        ),
        "prerequis": (
            "Un token API (compte sleep.ai.ku.dk, valable 12 h) ou la variable "
            "USLEEP_API_TOKEN, et le paquet 'usleep-api'. Un checkpoint local "
            "reste possible mais n'est pas le modele de l'article."
        ),
        "epoque": "30 s ; le service reechantillonne a 128 Hz.",
        "perf": (
            "80-88 % face a un consensus d'experts selon l'article, sur 1 EEG "
            "et 1 EOG, sans EMG."
        ),
        "quand": (
            "Second avis face a YASA, surtout sur un montage reduit (bandeau : "
            "EEG frontal + EOG, pas de menton)."
        ),
        "limites": (
            "Envoi d'un extrait anonymise au Danemark. Le token expire au bout "
            "de 12 h. Un balayage de canaux re-uploade a chaque configuration."
        ),
        "params": ("epoch", "usleep"),
    },
    "aasm_rules": {
        "principe": (
            "Implementation directe des criteres AASM : detection alpha/theta/delta, "
            "fuseaux et complexes K, mouvements oculaires, tonus mentonnier, puis "
            "arbre de decision epoque par epoque."
        ),
        "eeg": "requis",
        "eog": "recommande",
        "emg": "recommande",
        "canaux": (
            "1 EEG (central de preference), 1 EOG et 1 EMG fortement conseilles : "
            "sans EOG ni EMG, la distinction REM / N1 devient tres incertaine."
        ),
        "prerequis": "Aucune dependance externe : toujours disponible.",
        "epoque": "Parametrable (30 s recommande).",
        "perf": (
            "Accord typiquement 65-78 % : inferieur aux modeles appris, mais chaque "
            "decision est tracable et explicable."
        ),
        "quand": (
            "Aucun modele installe, ou besoin d'un scoring transparent et "
            "reproductible pour justifier une decision."
        ),
        "limites": "Sensible aux seuils et a la qualite du signal.",
        "params": ("epoch", "sfreq"),
    },
    "rules_hmm": {
        "principe": (
            "Regles AASM suivies d'un decodage de Viterbi sur un modele de Markov "
            "cache : les transitions physiologiquement improbables sont lissees."
        ),
        "eeg": "requis",
        "eog": "recommande",
        "emg": "recommande",
        "canaux": "Identiques aux regles AASM.",
        "prerequis": "Aucune dependance externe.",
        "epoque": "Parametrable (30 s recommande).",
        "perf": (
            "Quelques points de mieux que les regles seules, hypnogramme moins "
            "fragmente (moins de stades isoles d'une epoque)."
        ),
        "quand": "A preferer systematiquement aux regles AASM seules.",
        "limites": (
            "Le lissage peut effacer de vrais micro-eveils ou de courtes periodes REM."
        ),
        "params": ("epoch", "sfreq"),
    },
    "ml": {
        "principe": (
            "Descripteurs du pipeline CESA classes par un modele entraine par vous "
            "(RandomForest / HistGradientBoosting) sur vos propres nuits scorees."
        ),
        "eeg": "requis",
        "eog": "recommande",
        "emg": "recommande",
        "canaux": "Comme les regles AASM ; le montage doit etre celui de l'entrainement.",
        "prerequis": (
            "Un fichier de modele entraine : aucun modele n'est fourni avec CESA. "
            "Utilisez les scripts d'entrainement du pipeline."
        ),
        "epoque": "Celle utilisee a l'entrainement (30 s en general).",
        "perf": (
            "Peut depasser YASA sur votre propre population si le modele a ete "
            "entraine sur un montage et un protocole identiques."
        ),
        "quand": "Etudes internes avec des nuits deja scorees pour l'entrainement.",
        "limites": "Sans modele, le backend echoue immediatement.",
        "params": ("epoch", "sfreq", "ml"),
    },
    "ml_hmm": {
        "principe": "Modele ML CESA suivi d'un decodage HMM de la sequence de stades.",
        "eeg": "requis",
        "eog": "recommande",
        "emg": "recommande",
        "canaux": "Identiques au backend ML.",
        "prerequis": "Meme modele entraine que le backend ML.",
        "epoque": "Celle utilisee a l'entrainement.",
        "perf": "Generalement meilleur que le ML seul, hypnogramme plus lisse.",
        "quand": "A preferer au ML seul quand vous disposez d'un modele.",
        "limites": "Memes limites que le ML, plus le lissage HMM.",
        "params": ("epoch", "sfreq", "ml"),
    },
}

#: Channel resolution priority used by the backends, for the "canal retenu"
#: preview. Mirrors ``SleepScorer._score_pftsleep``.
_PFT_PRIORITY = {
    "eeg": ["C4-M1", "C3-M2", "FPZ-CZ", "PZ-OZ"],
    "eog": ["E1-M2", "E2-M1", "EOG L", "EOG LEFT", "EOG GAUCHE"],
    "emg": ["CHIN1-CHIN2", "CHIN1-CHIN3", "EMG CHIN", "CHIN", "MENTON"],
    "ecg": ["ECG", "ECG1", "ECG2", "EKG"],
}


class AutoScoringConfigDialog(QtWidgets.QDialog):
    """Configure backend + parameters for automatic scoring."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        controller,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configuration du scoring automatique")
        self.resize(1080, 760)
        self._ctrl = controller
        self._channels = list(channel_names or [])
        self._selected_method: Optional[str] = None
        self._build_ui()
        self._on_method_changed()

    # -- UI construction ------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_help_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Lancer le scoring")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("Backend :"))
        self._method_combo = QtWidgets.QComboBox()
        self._methods = list(self._ctrl.SUPPORTED_SCORING_METHODS)
        self._method_combo.addItems([METHOD_LABELS.get(m, m) for m in self._methods])
        cur = getattr(self._ctrl, "sleep_scoring_method", "pftsleep")
        if cur in self._methods:
            self._method_combo.setCurrentIndex(self._methods.index(cur))
        self._method_combo.setToolTip(
            "Le panneau de droite decrit le modele selectionne et verifie que "
            "l'enregistrement contient les canaux dont il a besoin."
        )
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo, stretch=1)
        layout.addLayout(method_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_lay = QtWidgets.QVBoxLayout(inner)
        self._groups: Dict[str, QtWidgets.QWidget] = {}
        inner_lay.addWidget(self._build_common_group())
        inner_lay.addWidget(self._build_yasa_group())
        inner_lay.addWidget(self._build_usleep_group())
        inner_lay.addWidget(self._build_pft_group())
        inner_lay.addWidget(self._build_ml_group())
        inner_lay.addWidget(self._build_channels_group(), stretch=1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)
        return container

    def _register_group(self, key: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self._groups[key] = widget
        return widget

    def _build_common_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Parametres generaux")
        form = QtWidgets.QFormLayout(box)

        self._epoch = QtWidgets.QDoubleSpinBox()
        self._epoch.setRange(1.0, 120.0)
        self._epoch.setSingleStep(1.0)
        self._epoch.setSuffix(" s")
        self._epoch.setValue(float(getattr(self._ctrl, "auto_scoring_epoch_length", 30.0)))
        self._epoch.setToolTip(
            "Duree d'une epoque de scoring. 30 s est le standard AASM et la seule "
            "valeur acceptee par YASA, U-Sleep et PFTSleep."
        )
        form.addRow("Duree d'epoque :", self._epoch)

        self._target_sfreq = QtWidgets.QDoubleSpinBox()
        self._target_sfreq.setRange(0.0, 1000.0)
        self._target_sfreq.setSingleStep(10.0)
        self._target_sfreq.setSuffix(" Hz")
        self._target_sfreq.setValue(float(getattr(self._ctrl, "yasa_target_sfreq", 100.0)))
        self._target_sfreq.setToolTip(
            "Frequence de reechantillonnage avant extraction des descripteurs "
            "(100 Hz convient a l'EEG de sommeil). 0 = pas de reechantillonnage."
        )
        self._sfreq_row_label = QtWidgets.QLabel("Reechantillonnage cible :")
        form.addRow(self._sfreq_row_label, self._target_sfreq)
        return self._register_group("common", box)

    def _build_yasa_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("YASA")
        form = QtWidgets.QFormLayout(box)

        self._conf = QtWidgets.QDoubleSpinBox()
        self._conf.setRange(0.0, 1.0)
        self._conf.setSingleStep(0.05)
        self._conf.setDecimals(2)
        self._conf.setValue(float(getattr(self._ctrl, "yasa_confidence_threshold", 0.80)))
        self._conf.setToolTip(
            "Seuil sous lequel une epoque est consideree comme peu fiable. Sert a "
            "signaler les epoques a revoir a la main, pas a les supprimer."
        )
        form.addRow("Seuil de confiance :", self._conf)

        self._age = QtWidgets.QSpinBox()
        self._age.setRange(0, 120)
        self._age.setSpecialValueText("non renseigne")
        self._age.setValue(int(getattr(self._ctrl, "yasa_age", 0) or 0))
        self._age.setToolTip(
            "Metadonnee optionnelle. Renseigner l'age ameliore legerement le "
            "classement (l'architecture du sommeil varie avec l'age)."
        )
        form.addRow("Age du sujet :", self._age)

        self._sex = QtWidgets.QComboBox()
        self._sex.addItem("non renseigne", None)
        self._sex.addItem("homme", True)
        self._sex.addItem("femme", False)
        male = getattr(self._ctrl, "yasa_male", None)
        self._sex.setCurrentIndex({None: 0, True: 1, False: 2}.get(male, 0))
        form.addRow("Sexe du sujet :", self._sex)

        self._n3_fix = QtWidgets.QCheckBox(
            "Corriger le N3 (critere AASM d'ondes lentes, voie frontale)"
        )
        self._n3_fix.setChecked(bool(getattr(self._ctrl, "n3_aasm_correction", False)))
        self._n3_fix.setToolTip(
            "YASA sous-detecte souvent le N3. Cette option releve en N3 les "
            "epoques predites N2 dont au moins 20 % sont des ondes lentes "
            "> 75 uV, mesurees de preference sur F4-M1. L'eveil et le REM ne "
            "sont pas modifies. Decochez pour mesurer le backend tout seul."
        )
        form.addRow("", self._n3_fix)
        return self._register_group("yasa", box)

    def _build_usleep_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("U-Sleep")
        form = QtWidgets.QFormLayout(box)

        self._usleep_token = QtWidgets.QLineEdit(
            str(getattr(self._ctrl, "usleep_api_token", "") or "")
        )
        self._usleep_token.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._usleep_token.setPlaceholderText(
            "token API (12 h) — laissez vide pour USLEEP_API_TOKEN"
        )
        self._usleep_token.setToolTip(
            "Generez un token sur https://sleep.ai.ku.dk (Account -> Generate "
            "API Token). Il identifie votre compte et expire au bout de 12 h : "
            "CESA ne l'ecrit jamais sur le disque."
        )
        self._usleep_token.textChanged.connect(self._refresh_help)
        form.addRow("Token API :", self._usleep_token)

        hint = QtWidgets.QLabel(
            '<a href="https://sleep.ai.ku.dk/login">sleep.ai.ku.dk</a> → Account '
            "→ Generate API Token. Seuls l'EEG et l'EOG choisis sont extraits "
            "et anonymises avant l'envoi."
        )
        hint.setOpenExternalLinks(True)
        hint.setWordWrap(True)
        form.addRow("", hint)

        self._usleep_eog = QtWidgets.QCheckBox("Ajouter un canal EOG a l'entree")
        self._usleep_eog.setChecked(bool(getattr(self._ctrl, "usleep_use_eog", True)))
        form.addRow("", self._usleep_eog)

        self._usleep_n3_fix = QtWidgets.QCheckBox(
            "Corriger le N3 (critere AASM d'ondes lentes, voie frontale)"
        )
        self._usleep_n3_fix.setChecked(
            bool(getattr(self._ctrl, "n3_aasm_correction", False))
        )
        self._usleep_n3_fix.setToolTip(
            "Releve en N3 les epoques predites N2 dont au moins 20 % sont des "
            "ondes lentes > 75 uV, mesurees de preference sur F4-M1. L'eveil "
            "et le REM ne sont pas modifies."
        )
        form.addRow("", self._usleep_n3_fix)

        self._usleep_ckpt = QtWidgets.QLineEdit(
            str(getattr(self._ctrl, "usleep_checkpoint_path", "") or "")
        )
        self._usleep_ckpt.setPlaceholderText(
            "optionnel : poids locaux (.ckpt / .pt) si vous n'utilisez pas l'API"
        )
        self._usleep_ckpt.setToolTip(
            "Inutile pour le modele de l'article : les auteurs ne distribuent "
            "pas les poids. Ne renseignez un fichier que si vous en avez "
            "entraine ou obtenu un."
        )
        self._usleep_ckpt.textChanged.connect(self._refresh_help)
        form.addRow("Checkpoint (optionnel) :", self._path_row(self._usleep_ckpt, "file"))

        self._usleep_sfreq = QtWidgets.QDoubleSpinBox()
        self._usleep_sfreq.setRange(32.0, 512.0)
        self._usleep_sfreq.setSuffix(" Hz")
        self._usleep_sfreq.setValue(float(getattr(self._ctrl, "usleep_sfreq", 128.0)))
        self._usleep_sfreq.setToolTip("128 Hz correspond au modele publie.")
        form.addRow("Reechantillonnage :", self._usleep_sfreq)

        self._usleep_device = QtWidgets.QLineEdit(
            str(getattr(self._ctrl, "usleep_device", "") or "")
        )
        self._usleep_device.setPlaceholderText(
            "auto (checkpoint local uniquement ; l'API tourne au Danemark)"
        )
        form.addRow("Peripherique :", self._usleep_device)

        self._usleep_zscore = QtWidgets.QCheckBox(
            "Normaliser chaque canal (z-score, checkpoint local)"
        )
        self._usleep_zscore.setChecked(bool(getattr(self._ctrl, "usleep_zscore", True)))
        form.addRow("", self._usleep_zscore)
        return self._register_group("usleep", box)

    def _build_pft_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("PFTSleep")
        form = QtWidgets.QFormLayout(box)

        self._pft_dir = QtWidgets.QLineEdit(
            str(getattr(self._ctrl, "pft_models_dir", "") or "")
        )
        self._pft_dir.setPlaceholderText("dossier des poids (defaut : PFTSleep/)")
        form.addRow("Dossier des poids :", self._path_row(self._pft_dir, "dir"))

        self._pft_device = QtWidgets.QLineEdit(
            str(getattr(self._ctrl, "pft_device", "auto") or "auto")
        )
        self._pft_device.setPlaceholderText("auto")
        form.addRow("Peripherique :", self._pft_device)

        self._pft_token = QtWidgets.QLineEdit(
            str(getattr(self._ctrl, "pft_hf_token", "") or "")
        )
        self._pft_token.setPlaceholderText("token Hugging Face (si telechargement requis)")
        self._pft_token.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        form.addRow("Token HF :", self._pft_token)

        self._pft_channels: Dict[str, QtWidgets.QComboBox] = {}
        for key, label in (
            ("eeg", "EEG (C4-M1) :"),
            ("eog", "EOG gauche (E1-M2) :"),
            ("emg", "EMG menton :"),
            ("ecg", "ECG :"),
        ):
            combo = QtWidgets.QComboBox()
            combo.addItem("auto (detection par priorite)", None)
            for name in self._channels:
                combo.addItem(name, name)
            current = getattr(self._ctrl, f"pft_{key}_channel", None)
            if current:
                pos = combo.findData(current)
                if pos >= 0:
                    combo.setCurrentIndex(pos)
            combo.setToolTip(
                "Laisser sur 'auto' pour la detection par ordre de priorite. "
                "Un canal absent est remplace par un vecteur nul."
            )
            combo.currentIndexChanged.connect(self._refresh_help)
            self._pft_channels[key] = combo
            form.addRow(label, combo)
        return self._register_group("pft", box)

    def _build_ml_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Modele ML entraine")
        form = QtWidgets.QFormLayout(box)
        self._ml_path = QtWidgets.QLineEdit(
            str(getattr(self._ctrl, "ml_model_path", "") or "")
        )
        self._ml_path.setPlaceholderText("chemin du modele entraine (.joblib / .pkl)")
        self._ml_path.setToolTip(
            "Obligatoire pour les backends ML : aucun modele pre-entraine n'est "
            "fourni avec CESA."
        )
        form.addRow("Modele :", self._path_row(self._ml_path, "file"))
        return self._register_group("ml", box)

    def _path_row(self, line: QtWidgets.QLineEdit, kind: str) -> QtWidgets.QWidget:
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(line, stretch=1)
        btn = QtWidgets.QPushButton("Parcourir...")

        def _browse() -> None:
            if kind == "dir":
                path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir un dossier")
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir un fichier")
            if path:
                line.setText(path)
                self._refresh_help()

        btn.clicked.connect(_browse)
        lay.addWidget(btn)
        return wrap

    def _build_channels_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Canaux candidats (par ordre de priorite)")
        lay = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel(
            "Ces listes sont des <b>listes de candidats</b> : pour chaque modalite, "
            "le backend retient le <b>premier canal disponible</b>, pas l'ensemble "
            "de la selection. En selectionner davantage n'ajoute pas d'information, "
            "cela definit seulement des solutions de repli. Selection vide = liste "
            "par defaut.<br><br>"
            "L'EMG attendu est <b>exclusivement mentonnier</b> (Chin1-Chin2, "
            "Menton) : le REM se definit par l'atonie du menton. Un EMG de "
            "<b>jambe</b> (tibial, PLM) n'est pas un substitut et degrade la "
            "detection du REM ; mieux vaut laisser la liste EMG vide."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(hint)

        self._profile_note = QtWidgets.QLabel()
        self._profile_note.setWordWrap(True)
        self._profile_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(self._profile_note)

        chan_box = QtWidgets.QHBoxLayout()
        self._eeg_list = self._make_channel_list([])
        self._eog_list = self._make_channel_list([])
        self._emg_list = self._make_channel_list([])
        self._chan_columns = {}
        for modality, label, widget in (
            ("eeg", "EEG", self._eeg_list),
            ("eog", "EOG", self._eog_list),
            ("emg", "EMG (menton)", self._emg_list),
        ):
            column, title = self._labeled_list(label, widget)
            chan_box.addLayout(column)
            self._chan_columns[modality] = (title, widget)
        lay.addLayout(chan_box, stretch=1)
        return self._register_group("channels", box)

    def _reload_channel_lists(self, method: str) -> None:
        """Show the candidates of *method*, not a single YASA-shaped list."""
        from CESA.scoring_channels import ROLE_UNUSED, channel_profile

        profile = channel_profile(method)
        defaults = (
            self._ctrl.channel_candidates_for(method)
            if hasattr(self._ctrl, "channel_candidates_for")
            else {
                "eeg": list(profile.eeg),
                "eog": list(profile.eog),
                "emg": list(profile.emg),
            }
        )
        for modality, (title, widget) in self._chan_columns.items():
            role = profile.role(modality)
            unused = role == ROLE_UNUSED
            title.setText(
                f"{title.property('modality_label')} — {role}"
                if not unused else
                f"{title.property('modality_label')} — non utilise"
            )
            widget.setEnabled(not unused)
            widget.setToolTip(
                "Ce backend n'exploite pas cette modalite : la selection n'a "
                "aucun effet." if unused else
                "Liste de candidats : le premier canal disponible est retenu."
            )
            self._preselect(widget, [] if unused else defaults.get(modality, []))
        self._profile_note.setText(
            f"<i>{profile.rationale}</i>" if profile.rationale else ""
        )

    @staticmethod
    def _preselect(widget: QtWidgets.QListWidget, wanted: Sequence[str]) -> None:
        upper = {str(w).upper().strip() for w in wanted}
        widget.blockSignals(True)
        for row in range(widget.count()):
            item = widget.item(row)
            item.setSelected(item.text().upper().strip() in upper)
        widget.blockSignals(False)

    def _build_help_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container)
        lay.addWidget(QtWidgets.QLabel("<b>Aide au choix du modele</b>"))
        self._help = QtWidgets.QTextBrowser()
        self._help.setOpenExternalLinks(False)
        lay.addWidget(self._help, stretch=1)
        doc_btn = QtWidgets.QPushButton("Ouvrir la documentation complete")
        doc_btn.clicked.connect(self._open_doc)
        lay.addWidget(doc_btn)
        return container

    def _labeled_list(
        self, label: str, widget: QtWidgets.QListWidget,
    ) -> tuple[QtWidgets.QVBoxLayout, QtWidgets.QLabel]:
        v = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(label)
        title.setProperty("modality_label", label)
        v.addWidget(title)
        v.addWidget(widget)
        return v, title

    def _make_channel_list(self, preselected: List[str]) -> QtWidgets.QListWidget:
        lw = QtWidgets.QListWidget()
        lw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        # Show available channels first; fall back to known candidates.
        items = self._channels or list(preselected)
        pre = {p.upper() for p in preselected}
        for name in items:
            it = QtWidgets.QListWidgetItem(name)
            lw.addItem(it)
            if name.upper() in pre:
                it.setSelected(True)
        lw.itemSelectionChanged.connect(self._refresh_help)
        return lw

    # -- Help panel -----------------------------------------------------

    def _current_method(self) -> str:
        return self._methods[self._method_combo.currentIndex()]

    def _on_method_changed(self) -> None:
        if not getattr(self, "_groups", None):
            return  # signal fired before the parameter groups existed
        method = self._current_method()
        wanted = set(_METHOD_INFO.get(method, {}).get("params", ()))  # type: ignore[arg-type]
        self._groups["yasa"].setVisible("conf" in wanted or "age" in wanted)
        self._groups["usleep"].setVisible("usleep" in wanted)
        self._groups["pft"].setVisible("pft" in wanted)
        self._groups["ml"].setVisible("ml" in wanted)
        show_sfreq = "sfreq" in wanted
        self._target_sfreq.setVisible(show_sfreq)
        self._sfreq_row_label.setVisible(show_sfreq)
        self._epoch.setEnabled(method not in ("yasa", "usleep", "pftsleep"))
        if not self._epoch.isEnabled():
            self._epoch.setValue(30.0)
        # PFTSleep resolves its own channels through dedicated selectors.
        self._groups["channels"].setVisible(method != "pftsleep")
        if method != "pftsleep":
            self._reload_channel_lists(method)
        self._refresh_help()

    def _refresh_help(self) -> None:
        if not hasattr(self, "_help"):
            return  # signal fired while the left panel was still being built
        method = self._current_method()
        info = _METHOD_INFO.get(method, {})
        label = METHOD_LABELS.get(method, method)
        rows = [f"<h3>{label}</h3>"]
        for title, key in (
            ("Principe", "principe"),
            ("Canaux", "canaux"),
            ("Epoque", "epoque"),
            ("Prerequis", "prerequis"),
            ("Performance attendue", "perf"),
            ("Quand le choisir", "quand"),
            ("Limites", "limites"),
        ):
            value = info.get(key)
            if value:
                rows.append(f"<p><b>{title}</b><br>{value}</p>")
        rows.append(self._channel_report_html(method, info))
        rows.append(self._preflight_html(method))
        rows.append(
            "<p><i>Conseil general : comparez toujours le resultat automatique a "
            "un scoring humain sur au moins une nuit avant d'exploiter les "
            "hypnogrammes (menu Sommeil > Comparer auto vs manuel).</i></p>"
        )
        self._help.setHtml("".join(rows))

    def _channel_report_html(self, method: str, info: Dict[str, object]) -> str:
        if not self._channels:
            return (
                "<p><b>Canaux de l'enregistrement</b><br>"
                "Aucun enregistrement charge : impossible de verifier les canaux.</p>"
            )
        lines: List[str] = ["<p><b>Verification des canaux disponibles</b><br>"]
        if method == "pftsleep":
            lines.extend(self._pft_channel_lines())
        else:
            from CESA.scoring_channels import channel_profile

            profile = channel_profile(method)
            modality_rows = [
                ("EEG", profile.role("eeg"),
                 self._selected(self._eeg_list, profile.candidates("eeg")),
                 list(profile.candidates("eeg"))),
                ("EOG", profile.role("eog"),
                 self._selected(self._eog_list, profile.candidates("eog")),
                 list(profile.candidates("eog"))),
                ("EMG", profile.role("emg"),
                 self._selected(self._emg_list, profile.candidates("emg")),
                 list(profile.candidates("emg"))),
            ]
            for name, role, selected, defaults in modality_rows:
                if role == "non utilise":
                    lines.append(f"{name} : non utilise par ce backend.<br>")
                    continue
                resolved = self._resolve_channel(selected or list(defaults))
                if resolved:
                    lines.append(f"OK {name} ({role}) : canal retenu <b>{resolved}</b><br>")
                    if name == "EMG":
                        issue = emg_warning(resolved)
                        if issue:
                            lines.append(
                                f"<span style='color:#f9a825'>ATTENTION : {issue}</span><br>"
                            )
                        better = find_submental_emg_candidates(self._channels)
                        if issue and better:
                            lines.append(
                                "Canal mentonnier disponible : <b>"
                                + ", ".join(better) + "</b><br>"
                            )
                elif role == "requis":
                    lines.append(
                        f"MANQUANT {name} ({role}) : aucun candidat present dans "
                        "l'enregistrement, le scoring va echouer.<br>"
                    )
                else:
                    lines.append(
                        f"Absent {name} ({role}) : aucun candidat trouve, le backend "
                        "fonctionnera de facon degradee.<br>"
                    )
        lines.append("</p>")
        return "".join(lines)

    def _pft_channel_lines(self) -> List[str]:
        """PFTSleep resolves its own channels, with 'dummy' when one is absent."""
        lines: List[str] = []
        for key, label in (("eeg", "EEG"), ("eog", "EOG gauche"),
                           ("emg", "EMG menton"), ("ecg", "ECG")):
            forced = self._pft_channels[key].currentData()
            resolved = forced or self._resolve_channel(_PFT_PRIORITY[key])
            if resolved:
                lines.append(f"OK {label} : canal retenu <b>{resolved}</b><br>")
            else:
                lines.append(
                    f"Absent {label} : remplace par un vecteur nul ('dummy'), "
                    "prediction degradee.<br>"
                )
        for label, names in (("SpO2", ["SPO2", "SAO2"]),
                             ("Sangle abdominale", ["ABDO", "ABD", "ABDOMEN"]),
                             ("Sangle thoracique", ["THOR", "THX", "THORAX"])):
            resolved = self._resolve_channel(names)
            state = f"<b>{resolved}</b>" if resolved else "absent (vecteur nul)"
            lines.append(f"{label} (optionnel) : {state}<br>")
        return lines

    def _resolve_channel(self, candidates: Sequence[str]) -> Optional[str]:
        """First available channel, mirroring the backends' selection order."""
        available = {ch.upper().strip(): ch for ch in self._channels}
        for name in candidates:
            hit = available.get(str(name).upper().strip())
            if hit:
                return hit
        return None

    def _preflight_html(self, method: str) -> str:
        problems = self._collect_preflight(method)
        if not problems:
            return (
                "<p style='color:#7bd88f'><b>Pre-verification</b><br>"
                "Ce backend peut etre lance.</p>"
            )
        items = "".join(f"<li>{p}</li>" for p in problems)
        return (
            "<p style='color:#f38ba8'><b>Pre-verification : ce backend ne peut pas "
            f"demarrer</b><ul>{items}</ul></p>"
        )

    def _collect_preflight(self, method: str) -> List[str]:
        """Run the controller pre-flight against the values currently in the form."""
        if not hasattr(self._ctrl, "auto_scoring_preflight"):
            return []
        snapshot = {
            "usleep_checkpoint_path": getattr(self._ctrl, "usleep_checkpoint_path", None),
            "usleep_api_token": getattr(self._ctrl, "usleep_api_token", None),
            "ml_model_path": getattr(self._ctrl, "ml_model_path", None),
            "yasa_eeg_candidates": list(getattr(self._ctrl, "yasa_eeg_candidates", [])),
        }
        try:
            self._ctrl.usleep_checkpoint_path = self._usleep_ckpt.text().strip() or None
            self._ctrl.usleep_api_token = self._usleep_token.text().strip() or None
            self._ctrl.ml_model_path = self._ml_path.text().strip() or None
            eeg = self._selected(self._eeg_list)
            if eeg:
                self._ctrl.yasa_eeg_candidates = eeg
            return list(self._ctrl.auto_scoring_preflight(method))
        except Exception:
            logger.debug("preflight failed", exc_info=True)
            return []
        finally:
            for key, value in snapshot.items():
                setattr(self._ctrl, key, value)

    def _open_doc(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        roots = [
            os.path.abspath(os.path.join(here, "..", "..", "..")),
            getattr(sys, "_MEIPASS", ""),
            os.path.dirname(os.path.abspath(sys.executable)),
        ]
        path = ""
        for root in roots:
            if not root:
                continue
            candidate = os.path.join(root, "docs", "Choix_Modele_Scoring.md")
            if os.path.isfile(candidate):
                path = candidate
                break
        if not path:
            QtWidgets.QMessageBox.information(
                self, "Documentation",
                "Document 'docs/Choix_Modele_Scoring.md' introuvable dans "
                "l'installation.",
            )
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Documentation", f"Lecture impossible :\n{exc}")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Choix du modele de scoring automatique")
        dlg.resize(860, 700)
        lay = QtWidgets.QVBoxLayout(dlg)
        browser = QtWidgets.QTextBrowser()
        try:
            browser.setMarkdown(text)
        except Exception:
            browser.setPlainText(text)
        lay.addWidget(browser)
        close = QtWidgets.QPushButton("Fermer")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.exec()

    # -- Accept ---------------------------------------------------------

    @staticmethod
    def _selected(
        lw: QtWidgets.QListWidget,
        priority: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Selected names, in *priority* order when one is given.

        The widget lists channels in montage order. Without reordering, the
        first selected EEG of a 90-channel PSG would always be C4-M1 even when
        the backend prefers F4-M1.
        """
        picked = [it.text() for it in lw.selectedItems()]
        if not priority:
            return picked
        return order_channel_selection(picked, priority)

    def _on_accept(self) -> None:
        method = self._current_method()
        self._ctrl.sleep_scoring_method = method
        self._ctrl.auto_scoring_epoch_length = float(self._epoch.value())
        self._ctrl.yasa_target_sfreq = float(self._target_sfreq.value())
        self._ctrl.yasa_confidence_threshold = float(self._conf.value())
        self._ctrl.yasa_age = int(self._age.value()) or None
        self._ctrl.yasa_male = self._sex.currentData()
        if method == "usleep":
            self._ctrl.n3_aasm_correction = bool(self._usleep_n3_fix.isChecked())
        else:
            self._ctrl.n3_aasm_correction = bool(self._n3_fix.isChecked())

        self._ctrl.usleep_api_token = self._usleep_token.text().strip() or None
        self._ctrl.usleep_checkpoint_path = self._usleep_ckpt.text().strip() or None
        self._ctrl.usleep_sfreq = float(self._usleep_sfreq.value())
        self._ctrl.usleep_device = self._usleep_device.text().strip() or None
        self._ctrl.usleep_use_eog = bool(self._usleep_eog.isChecked())
        self._ctrl.usleep_zscore = bool(self._usleep_zscore.isChecked())

        self._ctrl.pft_models_dir = self._pft_dir.text().strip() or None
        self._ctrl.pft_device = self._pft_device.text().strip() or "auto"
        self._ctrl.pft_hf_token = self._pft_token.text().strip() or None
        for key, combo in self._pft_channels.items():
            setattr(self._ctrl, f"pft_{key}_channel", combo.currentData())

        self._ctrl.ml_model_path = self._ml_path.text().strip() or None

        # Remembered per backend: switching algorithm must not carry over a
        # selection made for another model's expectations.
        if method != "pftsleep" and hasattr(self._ctrl, "set_channel_candidates_for"):
            from CESA.scoring_channels import channel_profile

            profile = channel_profile(method)
            remembered = (
                self._ctrl.channel_candidates_for(method)
                if hasattr(self._ctrl, "channel_candidates_for")
                else {}
            )
            self._ctrl.set_channel_candidates_for(
                method,
                eeg=self._selected(
                    self._eeg_list,
                    remembered.get("eeg") or profile.candidates("eeg"),
                ) or None,
                eog=self._selected(
                    self._eog_list,
                    remembered.get("eog") or profile.candidates("eog"),
                ) or None,
                emg=self._selected(
                    self._emg_list,
                    remembered.get("emg") or profile.candidates("emg"),
                ) or None,
            )
        else:
            for values, attr in (
                (self._selected(self._eeg_list), "yasa_eeg_candidates"),
                (self._selected(self._eog_list), "yasa_eog_candidates"),
                (self._selected(self._emg_list), "yasa_emg_candidates"),
            ):
                if values:
                    setattr(self._ctrl, attr, values)

        problems = []
        if hasattr(self._ctrl, "auto_scoring_preflight"):
            try:
                problems = list(self._ctrl.auto_scoring_preflight(method))
            except Exception:
                logger.debug("preflight failed on accept", exc_info=True)
        if problems:
            QtWidgets.QMessageBox.warning(
                self, "Backend non utilisable",
                "Le backend selectionne ne peut pas etre lance :\n\n- "
                + "\n- ".join(problems)
                + "\n\nCorrigez la configuration ou choisissez un autre backend "
                  "(les regles AASM et 'Regles AASM + HMM' fonctionnent sans "
                  "modele a installer).",
            )
            return

        warnings: List[str] = []
        if hasattr(self._ctrl, "auto_scoring_warnings"):
            try:
                warnings = list(self._ctrl.auto_scoring_warnings(method))
            except Exception:
                logger.debug("warnings check failed on accept", exc_info=True)
        if warnings:
            answer = QtWidgets.QMessageBox.warning(
                self, "Configuration a risque",
                "Le scoring peut etre lance, mais la configuration va degrader "
                "le resultat :\n\n- " + "\n- ".join(warnings)
                + "\n\nLancer quand meme ?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self._selected_method = method
        self.accept()

    @property
    def selected_method(self) -> Optional[str]:
        return self._selected_method
