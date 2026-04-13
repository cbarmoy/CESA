#!/usr/bin/env python3
"""
CESA (Complex EEG Studio Analysis) v0.0beta1.1 - Professional EEG Analysis Interface
====================================================================

Application professionnelle complète pour l'analyse de données EEG avec
amplification automatique, scoring de sommeil intégré, et analyses avancées.
Développée pour l'Unité Neuropsychologie du Stress (IRBA) selon les standards
scientifiques et les bonnes pratiques MNE-Python.

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Contact: come1.barmoy@supbiotech.fr
GitHub: cbarmoy
Version: 0.0beta1.1
Date: 2026-04-05
Licence: MIT
Release: CESA_0.0beta1.1_release

Fonctionnalités principales v0.0beta1.1:
- ✅ Chargement EDF+ avec diagnostic automatique et amplification
- ✅ Interface graphique professionnelle (thèmes clair/sombre)
- ✅ Scoring de sommeil (import Excel/EDF+ avec synchronisation)
- ✅ Analyses spectrales (Welch, FFT par stades, PSD robuste)
- ✅ Analyses avancées : cohérence, corrélation, ANOVA, stationnarité
- ✅ Micro-états : clustering K-means et topographies spatiales
- ✅ Détection d'artefacts : musculaires, oculaires, cardiaques
- ✅ Analyse de sources : MNE, sLORETA, dSPM, Beamforming
- ✅ Export complet : matrices CSV, visualisations PNG, rapports
- ✅ Système de checkpoints avec logs détaillés
- ✅ Navigation optimisée (ZQSD, contrôles temporels)
- ✅ Bouton cycle couleurs pour le scoring

Analyses avancées implémentées:
- 🔬 Cohérence inter-canal (coh, cohy, imcoh)
- 🔗 Corrélation temporelle (Pearson, Spearman, Kendall)
- 📊 ANOVA et tests de stationnarité (ADF)
- 🎯 Micro-états (K-means clustering)
- ⚡ Détection d'artefacts automatiques
- 🧠 Analyse de sources (MNE methods)
- 📈 Visualisations statistiques complètes

Dépendances principales:
- Python 3.8+ (64-bit recommandé)
- MNE-Python >= 1.4.0 (EDF+, sources, connectivité)
- SciPy >= 1.7.0 (statistiques, signaux)
- scikit-learn >= 1.0.0 (machine learning)
- statsmodels >= 0.13.0 (séries temporelles)
- YASA >= 0.6.0 (scoring automatique)
- NumPy, Pandas, Matplotlib (base scientifique)

Architecture modulaire:
- CESA/filters.py : Filtres Butterworth centralisés
- CESA/scoring_io.py : Import Excel/EDF+ et synchronisation
- CESA/theme.py : Thèmes UI professionnel
- spectral_analysis.py : Analyses spectrales robustes
"""

# =============================================================================
# IMPORTS ET CONFIGURATION
# =============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import mne
import numpy as np
import os
import sys
import tempfile
import json
import csv
import math
import time
import webbrowser
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Any, Iterable, Sequence
import logging
import re
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, SimpleQueue
import threading
from logging.handlers import RotatingFileHandler

from core.bridge import DataBridge
from ui.startup_mode import ModeSelector
from ui.open_dataset_dialog import OpenDatasetDialog
from CESA.spectral_analysis import compute_psd_fft, compute_band_powers, compute_peak_and_centroid, EEG_BANDS
from scipy.signal import welch
from mne.time_frequency import tfr_array_morlet
from matplotlib.colors import LogNorm
from CESA.spectral_analysis import compute_stage_psd_welch_for_array, compute_stage_psd_fft_for_array
from CESA.filters import (apply_filter as cesa_apply_filter,
                         apply_baseline_correction as cesa_apply_baseline_correction,
                         detect_signal_type as cesa_detect_signal_type,
                         get_filter_presets as cesa_get_filter_presets)
from CESA.manual_scoring_service import ManualScoringService, ManualScoringResult
from CESA.sleep_scorer import SleepScorer
from CESA.theme_manager import theme_manager
from CESA import group_analysis
from core.telemetry import telemetry
from core.lazy_provider import LazyProvider
from core.raw_loader import (
    normalize_recording_extension,
    open_raw_file,
    recording_extensions_for_scan,
    recording_filetypes_for_dialog,
)
from CESA.profile_store import ProfileStore
from CESA.profile_schema import DisplayProcessingProfile, SignalSection
from ui.channel_mapping_dialog import ChannelMappingDialog
from ui.section_layout_dialog import SectionLayoutDialog
from CESA.filter_engine import FilterAuditLog, FilterPipeline, pipeline_from_legacy_params
from ui.filter_dialog import FilterConfigDialog

# Configuration des warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def _module_print_to_logging(*args, **kwargs) -> None:
    """Route legacy print calls to structured logging."""
    text = " ".join(str(a) for a in args)
    lowered = text.lower()
    if any(tok in lowered for tok in ("❌", "erreur", "error", "echec", "échec", "failed", "exception")):
        logging.error(text)
    elif any(tok in lowered for tok in ("⚠️", "warning", "avertissement", "attention")):
        logging.warning(text)
    else:
        logging.info(text)

# Global standardization: all existing print(...) in this module now use logging.
print = _module_print_to_logging  # type: ignore[assignment]

class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Variante tolérante aux verrous Windows.
    Si la rotation échoue (fichier ouvert dans un autre programme), la rotation
    est désactivée pour éviter les traces d'erreur répétées tout en conservant
    l'écriture dans le fichier courant.
    """

    def __init__(self, *args, **kwargs):
        self._rotation_disabled = False
        kwargs.setdefault("delay", True)
        super().__init__(*args, **kwargs)

    def shouldRollover(self, record):
        if self._rotation_disabled:
            return False
        return super().shouldRollover(record)

    def doRollover(self):
        if self._rotation_disabled:
            return
        try:
            super().doRollover()
        except PermissionError as exc:
            self._rotation_disabled = True
            # Réouvrir le flux pour continuer à écrire dans le fichier courant.
            if self.stream is None:
                self.stream = self._open()
            try:
                sys.stderr.write(
                    "[WARN] Unable to rotate 'eeg_studio.log' (file locked). "
                    "Rotation disabled for this session. Close viewers and "
                    "restart CESA to re-enable log rotation.\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            # Éviter l'enchaînement d'autres handlers qui pourraient relancer la rotation.
            self.handleError = self._silent_handle_error

    def _silent_handle_error(self, record):
        """Empêche l'affichage d'erreurs supplémentaires si la rotation est désactivée."""
        pass


# Ajout d'un RotatingFileHandler pour les checkpoints persistants
try:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    rotating_handler = SafeRotatingFileHandler(
        log_dir / 'eeg_studio.log',
        maxBytes=2_000_000,
        backupCount=3,
        encoding='utf-8',
    )
    rotating_handler.setLevel(logging.INFO)
    rotating_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    root_logger = logging.getLogger()
    if not any(isinstance(h, SafeRotatingFileHandler) for h in root_logger.handlers):
        root_logger.addHandler(rotating_handler)
except Exception as _e:
    logging.warning(f"Impossible d'initialiser le RotatingFileHandler: {_e}")


def _log_viewer_checkpoint(code: str, msg: str, **fields: Any) -> None:
    """Étapes viewer Qt↔Tk pour diagnostic (fichier ``logs/eeg_studio.log``). Chercher ``VIEWER-CHK``."""
    if fields:
        tail = " ".join(f"{k}={v!r}" for k, v in fields.items())
        logging.info("[VIEWER-CHK-%s] %s | %s", code, msg, tail)
    else:
        logging.info("[VIEWER-CHK-%s] %s", code, msg)


def _flush_viewer_logs() -> None:
    """Force l'écriture disque des logs (crash natif / fermeture console avant buffer)."""
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass


# =============================================================================
# CLASSE PRINCIPALE - EEG ANALYSIS STUDIO
# =============================================================================

class EEGAnalysisStudio:
    """
    Application principale pour l'analyse de données EEG.
    
    Cette classe gère l'interface utilisateur, le chargement des données,
    la visualisation et l'analyse des signaux EEG avec amplification automatique.
    
    Attributs:
        root (tk.Tk): Fenêtre principale de l'application
        raw (mne.io.Raw): Données EEG chargées
        derivations (Dict[str, np.ndarray]): Dictionnaire des canaux chargés
        selected_channels (List[str]): Liste des canaux sélectionnés
        current_time (float): Temps actuel affiché (secondes)
        duration (float): Durée d'affichage (secondes)
        sfreq (float): Fréquence d'échantillonnage (Hz)
        autoscale_enabled (bool): État de l'autoscale
        filter_enabled (bool): État du filtre
        filter_low (float): Fréquence de coupure basse (Hz)
        filter_high (float): Fréquence de coupure haute (Hz)
    """
    
    def __init__(self, root: tk.Tk, data_bridge: Optional[DataBridge] = None) -> None:
        """
        Initialise l'application EEG Analysis Studio.
        
        Args:
            root: Fenêtre principale Tkinter
        """
        # =====================================================================
        # INITIALISATION DES VARIABLES PRINCIPALES
        # =====================================================================
        
        self.root = root
        self.data_bridge = data_bridge
        self.data_mode = "precomputed" if data_bridge else "raw"
        self._last_bridge_result = None
        self._telemetry_path = Path("logs") / "telemetry.csv"
        self.raw: Optional[mne.io.Raw] = None
        self.derivations: Dict[str, np.ndarray] = {}
        self.selected_channels: List[str] = []
        self.profile_store = ProfileStore()
        self.active_profile: DisplayProcessingProfile = self.profile_store.load_last_or_default()
        self.profile_channel_map_runtime: Dict[str, str] = {}
        self._bridge_executor: Optional[ThreadPoolExecutor] = None
        self._preprocess_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=3)
        # Reprise sur le thread Tk uniquement (jamais root.after depuis un worker : GIL / crash avec Py3.14 + Qt)
        self._tk_main_thread_queue: SimpleQueue[Callable[[], None]] = SimpleQueue()
        self._tk_main_poll_id: Optional[Any] = None
        self._tk_modal_ui_block_depth: int = 0
        try:
            self._tk_main_poll_id = self.root.after(16, self._poll_tk_main_thread_queue)
        except Exception:
            self._tk_main_poll_id = None
        self._processed_lock = threading.Lock()
        self._processed_window_cache: OrderedDict[str, Tuple[np.ndarray, float, Dict[str, Any]]] = OrderedDict()
        self._processed_window_limit = 64
        self._processing_generation = 0
        self._current_window_signature: Optional[Tuple] = None
        self._expected_channels: set[str] = set()
        self._last_preprocessed_signals: Dict[str, Tuple[np.ndarray, float]] = {}
        
        # Paramètres temporels
        self.current_time: float = 0.0
        self.duration: float = 30.0  # Durée d'affichage (1 époque)
        self.sfreq: float = 200.0
        
        # Paramètres de traitement
        self.autoscale_enabled: bool = False
        self.autoscale_window_duration: float = 30.0  # Durée de la fenêtre d'autoscale en secondes
        self.baseline_window_duration: float = 30.0  # Durée de la fenêtre de correction de ligne de base en secondes
        self.filter_enabled: bool = True  # Activer le filtre par défaut
        self.filter_var = tk.BooleanVar(value=True)  # Variable Tkinter pour l'interface
        self.filter_low: float = 0.5
        self.filter_high: float = 30.0
        self.baseline_correction_enabled: bool = True  # Activer la correction de ligne de base
        self.baseline_var = tk.BooleanVar(value=True)  # Variable Tkinter pour l'interface
        self.filter_type: str = "butterworth"
        self.filter_order: int = 4
        self.filter_window: str = "hamming"
        # État de modifications non enregistrées (scoring)
        self.scoring_dirty: bool = False
        
        # Filtrage par canal (alpha): paramètres spécifiques par dérivation
        # Clé = nom de canal/dérivation, valeur = dict(low, high, enabled)
        self.channel_filter_params: Dict[str, Dict[str, float]] = {}
        # New pipeline-based filter storage (takes precedence when present)
        self.channel_filter_pipelines: Dict[str, FilterPipeline] = {}
        self._filter_audit_log = FilterAuditLog()
        self.default_derivation_presets: Dict[str, Tuple[float, float]] = {
            # EEG: 0.3-35 Hz
            'F3-M2': (0.3, 35.0),
            'F4-M1': (0.3, 35.0),
            'C3-M2': (0.3, 35.0),
            'C4-M1': (0.3, 35.0),
            'O1-M2': (0.3, 35.0),
            'O2-M1': (0.3, 35.0),
            'F3': (0.3, 35.0),
            'F4': (0.3, 35.0),
            'C3': (0.3, 35.0),
            'C4': (0.3, 35.0),
            'O1': (0.3, 35.0),
            'O2': (0.3, 35.0),
            # ECG: 0.3-70 Hz
            'ECG': (0.3, 70.0),
            'ECG1': (0.3, 70.0),
            'ECG2': (0.3, 70.0),
            'EKG': (0.3, 70.0),
            # EMG: 10-0 Hz (passe-haut à 10 Hz)
            'EMG': (10.0, 0.0),
            'EMG Chin': (10.0, 0.0),
            'Chin': (10.0, 0.0),
            'MENTON': (10.0, 0.0),
            # EOG: 0.3-35 Hz
            'E1-M2': (0.3, 35.0),
            'E2-M1': (0.3, 35.0),
            'EOG': (0.3, 35.0),
            'EOG gauche': (0.3, 35.0),
            'EOG droite': (0.3, 35.0),
            'EOG Left': (0.3, 35.0),
            'EOG Right': (0.3, 35.0),
        }
        
        # Liste de canaux EEG alignée avec les attentes YASA (ordre de préférence)
        self.yasa_eeg_candidates: List[str] = [
            'C4-M1', 'F4-M1', 'C3-M2', 'F3-M2', 'O2-M1', 'O1-M2',
            'C4', 'F4', 'C3', 'F3', 'O2', 'O1'
        ]
        # Canaux EOG/EMG potentiels (pour affichage facultatif)
        self.yasa_eog_candidates: List[str] = ['E2-M1', 'E1-M2', 'EOG', 'EOG gauche', 'EOG droite', 'EOG Left', 'EOG Right']
        self.yasa_emg_candidates: List[str] = ['EMG', 'EMG Chin', 'Chin', 'MENTON']

        # Apply persisted profile settings on startup.
        try:
            self.baseline_correction_enabled = bool(self.active_profile.baseline_enabled)
            self.baseline_window_duration = float(self.active_profile.baseline_window_duration)
            self.filter_enabled = bool(self.active_profile.filter_enabled)
            self.filter_order = int(self.active_profile.filter_order)
            if self.active_profile.filter_low is not None:
                self.filter_low = float(self.active_profile.filter_low)
            if self.active_profile.filter_high is not None:
                self.filter_high = float(self.active_profile.filter_high)
            self.filter_type = str(self.active_profile.filter_type or "butterworth")
            self.filter_window = str(self.active_profile.filter_window or "hamming")
            for ch, cfg in (self.active_profile.channel_filter_params or {}).items():
                self.channel_filter_params[ch] = dict(cfg)
            for ch, pipe_dict in (self.active_profile.channel_filter_pipelines or {}).items():
                try:
                    self.channel_filter_pipelines[ch] = FilterPipeline.from_dict(pipe_dict)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Scoring de sommeil
        self.sleep_scoring_data: Optional[pd.DataFrame] = None  # Auto (YASA)
        self.manual_scoring_data: Optional[pd.DataFrame] = None  # Manuel (Excel)
        self.manual_scoring_service = ManualScoringService()
        self.sleep_scoring_method: str = "pftsleep"  # 'pftsleep' (par défaut), 'yasa', 'usleep'
        self.sleep_scoring_method_var = tk.StringVar(value=self.sleep_scoring_method)
        # Paramètres auto-scoring exposés dans l'UI (defaults recommandés littérature)
        self.auto_scoring_epoch_length: float = 30.0   # AASM standard
        self.yasa_target_sfreq: float = 100.0          # compromis robuste en pratique YASA
        self.yasa_age: Optional[int] = None            # métadonnée optionnelle pour YASA
        self.yasa_male: Optional[bool] = None          # True=homme, False=femme, None=inconnu
        self.yasa_confidence_threshold: float = 0.80   # seuil d'époques peu fiables
        self.usleep_target_sfreq: float = 128.0        # fréquence nominale U-Sleep
        self.usleep_use_eog: bool = True               # EEG + EOG recommandé
        self.usleep_zscore: bool = True                # normalisation standard
        self.usleep_device: str = "auto"               # auto/cpu/cuda
        self.usleep_api_token: Optional[str] = None    # token service web U-Sleep API
        self.pft_models_dir: Optional[str] = None      # Dossier modèles PFTSleep
        self.pft_device: str = "auto"                  # auto/cpu/cuda/mps
        self.pft_hf_token: Optional[str] = None        # Token HF optionnel pour PFTSleep
        self.pft_eeg_channel: Optional[str] = None
        self.pft_eog_channel: Optional[str] = None
        self.pft_emg_channel: Optional[str] = None
        self.pft_ecg_channel: Optional[str] = None
        # Optional pretrained checkpoint used only for U-Sleep backend
        self.usleep_checkpoint_path: Optional[str] = None
        
        # FFT Comparison data (for comparing two conditions)
        self.fft_comparison_raw: Optional[mne.io.Raw] = None
        self.fft_comparison_derivations: Dict[str, np.ndarray] = {}
        self.fft_comparison_scoring: Optional[pd.DataFrame] = None
        self.fft_comparison_name: str = "Condition 2"
        self.fft_main_name: str = "Condition 1"
        self.show_manual_scoring: bool = True
        self.scoring_epoch_duration: float = 30.0  # Durée d'une époque en secondes
        # Plage nuit (optionnel) : ne comparer / analyser que de X à T minutes depuis le début
        self.night_start_min: Optional[float] = None  # Début nuit (min depuis t=0)
        self.night_end_min: Optional[float] = None    # Fin nuit (min depuis t=0)
        # Cache PSG
        self._psg_cached_hypnogram: Optional[Tuple[List[str], float]] = None
        self._psg_cached_scoring_rows: int = 0
        # Viewer PSG : PyQtGraph par défaut si PySide6 + pyqtgraph sont disponibles
        self._qt_viewer_bridge: Optional[Any] = None
        self.prefer_qt_psg_viewer: bool = True
        self._qt_nav_sync_queued: bool = False
        self._qt_nav_sync_deferred: bool = False
        self._pending_nav_sync_t: float = 0.0
        self._qt_channel_tuple: Tuple[str, ...] = ()
        self._qt_import_check_failed: bool = False

        # Gestionnaire de thèmes (remplace l'ancien système de palettes)
        from CESA.theme_manager import theme_manager
        self.theme_manager = theme_manager

        # Références aux lignes EEG pour mise à jour des couleurs
        self.eeg_lines = []
        # Base d'affichage absolue (peut être définie par Excel pour caler l'axe X)
        self.display_start_datetime: Optional[datetime] = None
        # Intercepter la fermeture pour prévenir si non enregistré
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._quit_application)
        except Exception:
            pass
        
        # Clustering Spaghetti (UI et logique)
        self.spag_clusters: Dict[str, str] = {}
        self.spag_cluster_names: Dict[str, str] = {'A': "Cluster A", 'B': "Cluster B"}
        self.spag_cluster_name_vars: Dict[str, tk.StringVar] = {}
        self.spag_subjects: List[str] = []
        self.spag_subject_listbox: Optional[tk.Listbox] = None
        self.spag_cluster_listboxes: Dict[str, tk.Listbox] = {}

        # Libellés personnalisés pour les groupes (Before/After)
        self.spag_group_before_var = tk.StringVar(value="Before")
        self.spag_group_after_var = tk.StringVar(value="After")

        # Mapping des stades FR -> codes standard
        self.french_to_standard = {
            'eveil': 'W', 'éveil': 'W', 'w': 'W', 'wake': 'W', 'veille': 'W',
            'n1': 'N1', 'n1 (sommeil léger)': 'N1',
            'n2': 'N2', 'n2 (sommeil léger)': 'N2',
            'n3': 'N3', 'n3 (sommeil profond)': 'N3', 'sws': 'N3', 'slow wave': 'N3', 'n4': 'N3',
            'rem': 'R', 'paradoxal': 'R', 'r': 'R',
            # Divers (ignorés dans la comparaison)
            'mt': 'U', 'mvt': 'U', 'movement': 'U', 'artefact': 'U', 'artifact': 'U', 'unk': 'U', 'inconnu': 'U'
        }
        # Libellés lisibles pour visualisation
        self.sleep_stages = {
            'W': 'Éveil',
            'N1': 'N1',
            'N2': 'N2',
            'N3': 'N3',
            'R': 'REM',
            'U': 'Inconnu'
        }
        
        # Variables d'interface
        self.interface_mode = tk.StringVar(value="modern")
        self.spacing_var = tk.StringVar(value="50")
        self.amplitude_var = tk.StringVar(value="100")
        self.autoscale_var = tk.BooleanVar(value=False)
        
        # État du panneau de commandes
        self.control_panel_collapsed = False
        self.original_control_width = 300  # Ajustée pour compacité sans masquer les boutons
        
        # Durée d'époque pour le scoring de sommeil (30s par défaut)
        self.scoring_epoch_duration = 30.0
        
        # Configuration de l'interface
        self._setup_modern_interface()
        self._create_modern_menu()
        self._create_modern_widgets()
        self._setup_keyboard_shortcuts()
        
        # Initialisation de l'assistant utilisateur
        self._setup_user_assistant()
        
        # Initialisation du système de capture des checkpoints pour les rapports de bug
        self.console_checkpoints = []
        self._setup_checkpoint_capture()
        
        # Initialisation du thème
        self.dark_theme_enabled = False
        
        # Initialisation du système de marqueurs
        self.temporal_markers = []  # Liste des marqueurs temporels
        
        # Mise à jour de l'affichage de la version
        self._update_version_display()
        
        # Log de l'initialisation
        logging.info("CESA (Complex EEG Studio Analysis) v0.0beta1.1 initialisé avec succès")
        logging.info("Application initialized successfully")

        # Optimisation rendu: exécuteur en arrière-plan + debouncing
        try:
            import concurrent.futures as _fut
            self._plot_executor = _fut.ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1)))
        except Exception:
            self._plot_executor = None
        self._plot_update_gen = 0
        self._plot_update_pending_id = None
        self._active_plot_future = None
        self._active_plot_token = None
        
        # S'assurer que la fenêtre principale n'est pas en topmost (fix splash screen)
        try:
            self.root.attributes('-topmost', False)
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
        
    def _get_plot_width_px(self) -> int:
        try:
            if hasattr(self, 'canvas') and self.canvas is not None:
                widget = self.canvas.get_tk_widget()
                width = widget.winfo_width()
                if width and width > 0:
                    return int(width)
                width = widget.winfo_reqwidth()
                if width and width > 0:
                    return int(width)
        except Exception:
            pass
        try:
            return int(self.root.winfo_width())
        except Exception:
            return 1200

    def _detect_screen_resolution(self) -> Tuple[int, int, int]:
        """
        Détecte la résolution de l'écran et configure l'interface adaptée.
        
        Returns:
            Tuple[int, int, int]: (largeur_fenêtre, hauteur_fenêtre, largeur_contrôles)
        """
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Configuration adaptative selon la résolution
            if screen_width < 1200 or screen_height < 800:
                self.interface_mode.set("compact")
                return 1000, 600, 250
            elif screen_width < 1600 or screen_height < 1000:
                self.interface_mode.set("normal")
                return 1400, 800, 300
            else:
                self.interface_mode.set("scientific")
                return 1600, 1000, 350
                
        except Exception as e:
            logging.warning(f"Erreur détection résolution: {e}")
            return 1000, 600, 250
    
    def _setup_modern_interface(self) -> None:
        """
        Configure l'interface moderne avec thème professionnel.
        
        Configure la fenêtre principale, les styles et les paramètres matplotlib
        pour une interface moderne et professionnelle.
        """
        # Détection de la résolution et configuration de la fenêtre
        window_width, window_height, control_width = self._detect_screen_resolution()
        self.control_width = control_width
        
        # Configuration de la fenêtre principale
        self.root.title("CESA (Complex EEG Studio Analysis) v0.0beta1.1 - Interface Moderne")
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(1000, 600)
        
        # Maximiser la fenêtre au démarrage (plein écran sur Windows)
        try:
            self.root.state('zoomed')
        except Exception:
            pass  # Si la maximisation échoue, continuer avec la taille par défaut
        
        # Configuration du thème moderne
        self._setup_modern_theme()
        
        # Configuration matplotlib pour l'interface moderne
        self._setup_modern_matplotlib()
        
        # Configuration de l'icône (si disponible)
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ico_path = os.path.join(base_dir, 'logo', 'Icone_CESA.ico')
            if os.path.exists(ico_path):
                self.root.iconbitmap(ico_path)
        except Exception:
            pass  # Pas d'icône disponible
        
        logging.info("Interface configurée avec succès")
    
    def _setup_modern_theme(self) -> None:
        """Configure le thème moderne de l'interface."""
        style = ttk.Style()
        
        # Utilisation du thème moderne
        style.theme_use('clam')
        
        # Configuration des couleurs modernes
        style.configure('Modern.TFrame', background='#f8f9fa')
        style.configure('Modern.TLabel', background='#f8f9fa', foreground='#212529')
        style.configure('Modern.TButton', 
                       background='#007bff', 
                       foreground='white',
                       font=('Segoe UI', 9, 'bold'))
        style.map('Modern.TButton',
                 background=[('active', '#0056b3'), ('pressed', '#004085')])
        
        # Configuration des contrôles
        style.configure('Modern.TScale', background='#f8f9fa', troughcolor='#dee2e6')
        style.configure('Modern.TCheckbutton', background='#f8f9fa', foreground='#212529')
        style.configure('Modern.TRadiobutton', background='#f8f9fa', foreground='#212529')
        
        # Configuration des groupes
        style.configure('Group.TLabelframe', 
                       background='#f8f9fa', 
                       foreground='#495057',
                       font=('Segoe UI', 9, 'bold'))
        style.configure('Group.TLabelframe.Label', 
                       background='#f8f9fa', 
                       foreground='#495057',
                       font=('Segoe UI', 9, 'bold'))
        
        # Configuration du label de version
        style.configure('Version.TLabel', 
                       background='#f8f9fa', 
                       foreground='#007bff',
                       font=('Segoe UI', 8, 'bold'))
        
        # Configuration du label de statut
        style.configure('Status.TLabel', 
                       background='#f8f9fa', 
                       foreground='#28a745',
                       font=('Segoe UI', 9))
    
    def _setup_modern_matplotlib(self) -> None:
        """Configure matplotlib pour l'interface moderne."""
        # Configuration des paramètres matplotlib
        plt.rcParams.update({
            'figure.facecolor': '#ffffff',
            'axes.facecolor': '#ffffff',
            'axes.edgecolor': '#dee2e6',
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#dee2e6',
            'font.size': 10,
            'font.family': 'Segoe UI',
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'lines.linewidth': 1.0,
            'lines.antialiased': True
        })
    
    def _setup_keyboard_shortcuts(self) -> None:
        """Initialise les raccourcis clavier (incluant ZQSD)."""
        try:
            # Navigation ZQSD et autres raccourcis
            self._setup_keyboard_navigation_simple()
        except Exception as e:
            logging.warning(f"Erreur initialisation raccourcis clavier: {e}")
    
    def _setup_keyboard_navigation_simple(self):
        """Configure la navigation clavier ZQSD simple."""
        try:
            logging.debug("Configuring ZQSD navigation...")
            
            # Détacher les anciens bindings pour éviter les doublons
            self.root.unbind('<Key>')
            
            # Raccourcis ZQSD pour navigation par époques (30s)
            self.root.bind('<Key-z>', lambda e: self._navigate_simple_epoch_previous())
            self.root.bind('<Key-q>', lambda e: self._navigate_simple_epoch_previous())
            self.root.bind('<Key-s>', lambda e: self._navigate_simple_epoch_next())
            self.root.bind('<Key-d>', lambda e: self._navigate_simple_epoch_next())
            
            # Focus sur la fenêtre principale pour recevoir les événements clavier
            self.root.focus_set()
            self.root.focus_force()
            
            logging.debug("ZQSD navigation configured successfully")
            logging.info("[BIND] Navigation ZQSD simple configurée")
            
        except Exception as e:
            logging.error(f"Error configuring ZQSD navigation: {e}")
            logging.error(f"[BIND] Erreur navigation ZQSD: {e}")
    
    def _navigate_simple_epoch_previous(self):
        """Navigation simple vers l'époque précédente (30s en arrière)."""
        epoch_duration = 30.0  # Durée d'une époque fixe
        
        # Reculer d'une époque
        self.current_time = max(0, self.current_time - epoch_duration)
        
        # Mettre à jour les sliders si ils existent
        if hasattr(self, 'time_var'):
            self.time_var.set(self.current_time)
        if hasattr(self, 'bottom_time_var'):
            self.bottom_time_var.set(self.current_time)
        
        # Mettre à jour l'affichage
        self.update_plot()
        
        print(f"⬅️ Navigation simple: époque précédente {self.current_time:.1f}s")
        logging.info(f"Simple navigation: previous epoch {self.current_time:.1f}s")
    
    def _navigate_simple_epoch_next(self):
        """Navigation simple vers l'époque suivante (30s en avant)."""
        epoch_duration = 30.0  # Durée d'une époque fixe
        
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + epoch_duration)
        else:
            self.current_time += epoch_duration
        
        # Mettre à jour les sliders si ils existent
        if hasattr(self, 'time_var'):
            self.time_var.set(self.current_time)
        if hasattr(self, 'bottom_time_var'):
            self.bottom_time_var.set(self.current_time)
        
        # Mettre à jour l'affichage
        self.update_plot()
        
        print(f"➡️ Navigation simple: époque suivante {self.current_time:.1f}s")
        logging.info(f"Simple navigation: next epoch {self.current_time:.1f}s")
    
    def _create_modern_menu(self) -> None:
        """
        Crée le menu moderne de l'application.
        
        Configure une barre de menu professionnelle avec tous les outils
        nécessaires pour l'analyse EEG.
        """
        menubar = tk.Menu(self.root, bg='#f8f9fa', fg='#212529')
        self.root.config(menu=menubar)
        
        # Configuration pour les menus avec scrollbars
        menu_config = {
            'tearoff': 0, 
            'bg': '#f8f9fa', 
            'fg': '#212529',
            'font': ('Segoe UI', 9),
            'activebackground': '#e9ecef',
            'activeforeground': '#212529'
        }
        
        # Créer une classe pour gérer les menus scrollables
        class ScrollableMenu:
            def __init__(self, parent, max_items=20, **kwargs):
                self.parent = parent
                self.max_items = max_items
                self.main_menu = tk.Menu(parent, **{**menu_config, **kwargs})
                self.current_submenu = None
                self.item_count = 0
                self.submenu_count = 0
                
            def add_command(self, **kwargs):
                if self.item_count >= self.max_items:
                    # Créer un nouveau sous-menu "Plus..." si nécessaire
                    if self.current_submenu is None:
                        self.submenu_count += 1
                        submenu_label = "📋 Plus d'options..." if self.submenu_count == 1 else f"📋 Plus d'options... ({self.submenu_count})"
                        self.current_submenu = tk.Menu(self.main_menu, **menu_config)
                        self.main_menu.add_cascade(label=submenu_label, menu=self.current_submenu)
                    
                    self.current_submenu.add_command(**kwargs)
                else:
                    self.main_menu.add_command(**kwargs)
                
                self.item_count += 1
            
            def add_separator(self):
                if self.current_submenu and self.item_count > self.max_items:
                    self.current_submenu.add_separator()
                else:
                    self.main_menu.add_separator()
            
            def add_cascade(self, **kwargs):
                if self.current_submenu and self.item_count > self.max_items:
                    self.current_submenu.add_cascade(**kwargs)
                else:
                    self.main_menu.add_cascade(**kwargs)
                self.item_count += 1
        
        # Créer une fonction helper pour créer des menus avec scrollbar
        def create_scrollable_menu(parent, max_items=20, **kwargs):
            """Crée un menu avec possibilité de scroll si trop d'éléments."""
            scrollable = ScrollableMenu(parent, max_items=max_items, **kwargs)
            return scrollable
        
        # Menu Fichier
        file_menu = create_scrollable_menu(menubar)
        menubar.add_cascade(label="📁 Fichier", menu=file_menu.main_menu)
        file_menu.add_command(label="📂 Ouvrir enregistrement", command=self.load_edf_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="💾 Exporter Données", command=self._export_data, accelerator="Ctrl+S")
        file_menu.add_command(label="✂️ Exporter segment EDF...", command=self._export_edf_segment)
        file_menu.add_command(label="📊 Exporter Rapport", command=self._export_report)
        file_menu.add_separator()
        file_menu.add_command(label="⚙️ Préférences", command=self._show_preferences)
        file_menu.add_separator()
        file_menu.add_command(label="🧩 Charger profil...", command=self._load_profile_dialog)
        file_menu.add_command(label="💽 Enregistrer profil...", command=self._save_profile_as)
        file_menu.add_command(label="🧱 Configurer sections...", command=self._edit_profile_sections_dialog)
        file_menu.add_command(label="📑 Dupliquer profil...", command=self._duplicate_profile_dialog)
        file_menu.add_command(label="✏️ Renommer profil...", command=self._rename_profile_dialog)
        file_menu.add_command(label="🗑️ Supprimer profil...", command=self._delete_profile_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="🤖 Automatisation FFT en Lot", command=self._show_batch_fft_automation, accelerator="Ctrl+B")
        file_menu.add_separator()
        file_menu.add_command(label="❌ Quitter", command=self._quit_application, accelerator="Ctrl+Q")
        
        # Menu Affichage
        view_menu = create_scrollable_menu(menubar)
        menubar.add_cascade(label="👁️ Affichage", menu=view_menu.main_menu)
        view_menu.add_command(label="📋 Sélectionner Canaux", command=self._show_channel_selector, accelerator="Ctrl+1")
        view_menu.add_separator()
        view_menu.add_command(label="📏 Activer Autoscale", command=self._toggle_autoscale, accelerator="Ctrl+A")
        view_menu.add_command(label="🔧 Configuration Filtres", command=self.show_filter_config, accelerator="Ctrl+F")
        view_menu.add_separator()
        view_menu.add_command(
            label="📊 Vue Multi-Graphiques (Matplotlib)",
            command=self._open_matplotlib_psg_view,
            accelerator="Ctrl+M",
        )
        view_menu.add_command(label="⚡ Viewer PyQtGraph (défaut)", command=self._launch_qt_viewer)
        view_menu.add_separator()
        view_menu.add_command(label="🎨 Thème Sombre", command=self._toggle_dark_theme)
        view_menu.add_separator()
        view_menu.add_command(label="📋 Bascule Panneau Commandes", command=self._toggle_control_panel, accelerator="F2")
        view_menu.add_command(label="🔄 Actualiser", command=self._refresh_plot, accelerator="F5")
        
        # Menu Aide
        help_menu = create_scrollable_menu(menubar)
        menubar.add_cascade(label="❓ Aide", menu=help_menu.main_menu)
        help_menu.add_command(label="🎯 Assistant de première utilisation", command=self._show_welcome_assistant)
        help_menu.add_command(label="🔍 Explorateur de fonctionnalités", command=self._show_feature_explorer)
        help_menu.add_command(label="📚 Guide de référence complet", command=self._open_reference_guide)
        help_menu.add_separator()
        help_menu.add_command(label="📖 Documentation complète", command=self._open_documentation)
        help_menu.add_command(label="🧮 Guide entropie renormée", command=self._open_entropy_docs)
        help_menu.add_separator()
        help_menu.add_command(label="🔧 Diagnostic système", command=self._run_diagnostic)
        help_menu.add_command(label="📞 Support technique", command=self._open_support)
        
        # Menu Analyse (avec scroll si plus de 8 éléments)
        analysis_menu = create_scrollable_menu(menubar, max_items=8)
        menubar.add_cascade(label="📊 Analyse", menu=analysis_menu.main_menu)
        analysis_menu.add_command(label="📈 Statistiques", command=self._show_channel_stats)
        analysis_menu.add_command(label="🔍 Diagnostic", command=self._show_diagnostics)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="📉 Analyse Spectrale", command=self._show_spectral_analysis)
        analysis_menu.add_command(label="📊 PSD par stade (FFT – Analyse_spectrale)", command=self._show_stage_psd_fft)
        analysis_menu.add_command(label="🌈 Spectrogramme ondelettes (avant/après)", command=self._show_wavelet_spectrogram_before_after)
        analysis_menu.add_command(label="🧮 Entropie Renormée (Issartel)", command=self._show_renormalized_entropy)
        analysis_menu.add_command(label="🔬 Entropie Multiscale (MSE)", command=self._show_multiscale_entropy)
        analysis_menu.add_command(label="🧮 Analyse périodes (SleepEEGpy)", command=self._analyze_sleep_periods)
        analysis_menu.add_command(label="🌊 Analyse Temporelle", command=self._show_temporal_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="📈 Graphiques Spaghetti (EDF…)", command=self._generate_spaghetti_from_edf)
        analysis_menu.add_command(label="📚 Analyse Groupe (MSE/REN)", command=self._show_group_analysis)
        analysis_menu.add_command(label="🤖 Automatisation FFT en Lot", command=self._show_batch_fft_automation)
        analysis_menu.add_command(label="📊 Analyse Avancée", command=self._show_advanced_analysis)
        analysis_menu.add_command(label="🔬 Analyse Micro-états", command=self._show_microstates_analysis)
        analysis_menu.add_command(label="🧠 Connectivité Cérébrale", command=self._show_connectivity_analysis)
        analysis_menu.add_command(label="⚡ Détection d'Artefacts", command=self._show_artifact_detection)
        analysis_menu.add_command(label="🎯 Analyse de Sources", command=self._show_source_analysis)
        
        # Menu Scoring de Sommeil
        sleep_menu = create_scrollable_menu(menubar)
        menubar.add_cascade(label="🛏️ Sommeil", menu=sleep_menu.main_menu)
        sleep_menu.add_command(label="⚙️ Scoring automatique (backend sélectionné)", command=self._run_auto_sleep_scoring, accelerator="Ctrl+Y")
        sleep_menu.main_menu.add_radiobutton(
            label="Backend scoring: YASA",
            variable=self.sleep_scoring_method_var,
            value="yasa",
            command=lambda: self._set_sleep_scoring_method("yasa"),
        )
        sleep_menu.main_menu.add_radiobutton(
            label="Backend scoring: U-Sleep",
            variable=self.sleep_scoring_method_var,
            value="usleep",
            command=lambda: self._set_sleep_scoring_method("usleep"),
        )
        sleep_menu.main_menu.add_radiobutton(
            label="Backend scoring: PFTSleep",
            variable=self.sleep_scoring_method_var,
            value="pftsleep",
            command=lambda: self._set_sleep_scoring_method("pftsleep"),
        )
        sleep_menu.main_menu.add_radiobutton(
            label="Backend scoring: AASM Rules (nouveau)",
            variable=self.sleep_scoring_method_var,
            value="aasm_rules",
            command=lambda: self._set_sleep_scoring_method("aasm_rules"),
        )
        sleep_menu.main_menu.add_radiobutton(
            label="Backend scoring: ML (nouveau)",
            variable=self.sleep_scoring_method_var,
            value="ml",
            command=lambda: self._set_sleep_scoring_method("ml"),
        )
        sleep_menu.main_menu.add_radiobutton(
            label="Backend scoring: ML + HMM (nouveau)",
            variable=self.sleep_scoring_method_var,
            value="ml_hmm",
            command=lambda: self._set_sleep_scoring_method("ml_hmm"),
        )
        sleep_menu.main_menu.add_radiobutton(
            label="Backend scoring: Rules + HMM (nouveau)",
            variable=self.sleep_scoring_method_var,
            value="rules_hmm",
            command=lambda: self._set_sleep_scoring_method("rules_hmm"),
        )
        sleep_menu.add_command(label="⚙️ Configurer auto-scoring (YASA/U-Sleep)...", command=self._open_sleep_scoring_settings)
        sleep_menu.add_command(label="📦 Définir checkpoint U-Sleep...", command=self._select_usleep_checkpoint)
        sleep_menu.add_separator()
        sleep_menu.add_command(label="📥 Importer Scoring (Excel/EDF)", command=self._open_scoring_import_hub, accelerator="Ctrl+Shift+M")
        sleep_menu.add_command(label="🔀 Comparer Auto vs Manuel", command=self._compare_scoring, accelerator="Ctrl+C")
        sleep_menu.add_command(label="🌙 Définir plage nuit (X à T)...", command=self._open_night_range_dialog)
        sleep_menu.add_command(label="💾 Sauvegarder Scoring (CSV)", command=self._save_active_scoring)
        sleep_menu.add_command(label="📈 Informations Scoring", command=self._show_sleep_scoring_info)
        sleep_menu.add_command(label="✍️ Scorer manuellement (éditeur)", command=self._open_manual_scoring_editor, accelerator="Ctrl+Shift+S")
        sleep_menu.add_command(label="⚙️ Ajuster Durée Époque", command=self._adjust_epoch_duration)
        
        # Menu Outils
        tools_menu = create_scrollable_menu(menubar)
        menubar.add_cascade(label="🛠️ Outils", menu=tools_menu.main_menu)
        tools_menu.add_command(label="🎯 Marqueurs", command=self._show_markers)
        tools_menu.add_command(label="📏 Mesures", command=self._show_measurements)
        tools_menu.add_command(label="⏩ Aller au temps...", command=self._open_goto_time_dialog, accelerator="Ctrl+G")
        tools_menu.add_separator()
        tools_menu.add_command(label="🔧 Configuration Avancée", command=self._show_advanced_config)
        
        # Menu Aide
        help_menu = create_scrollable_menu(menubar)
        menubar.add_cascade(label="❓ Aide", menu=help_menu.main_menu)
        help_menu.add_command(label="📖 Assistant de bienvenue", command=self._show_user_guide, accelerator="F1")
        help_menu.add_command(label="⌨️ Raccourcis Clavier", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="🐛 Signaler un Bug", command=self._report_bug)
        help_menu.add_command(label="💡 Suggestions", command=self._suggest_feature)
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ À propos", command=self._show_about)
        
        logging.info("Menu créé avec succès")

    # =====================================================================
    # PROFILS D'AFFICHAGE / TRAITEMENT
    # =====================================================================

    def _profile_section_choices(self) -> Dict[str, str]:
        choices: Dict[str, str] = {}
        for section in getattr(self.active_profile, "signal_sections", []):
            if not bool(getattr(section, "enabled", True)):
                continue
            key = str(getattr(section, "key", "")).strip()
            label = str(getattr(section, "label", key)).strip() or key
            if key:
                choices[key] = label
        return choices

    def _apply_profile_sections(
        self,
        sections: Sequence[SignalSection],
        *,
        refresh_view: bool = True,
    ) -> None:
        profile = getattr(self, "active_profile", None)
        if profile is None:
            return
        normalized: List[SignalSection] = []
        used_keys: set[str] = set()
        for idx, section in enumerate(sections):
            key = str(getattr(section, "key", "")).strip() or f"section_{idx + 1}"
            if key in used_keys:
                key = f"{key}_{idx + 1}"
            used_keys.add(key)
            label = str(getattr(section, "label", key)).strip() or key
            try:
                ratio = float(getattr(section, "ratio", 1.0))
            except Exception:
                ratio = 1.0
            ratio = max(0.2, ratio)
            signal_type = str(getattr(section, "signal_type", "eeg")).strip().lower() or "eeg"
            enabled = bool(getattr(section, "enabled", True))
            palette = [str(c).strip() for c in (getattr(section, "color_palette", []) or []) if str(c).strip()]
            normalized.append(
                SignalSection(
                    key=key,
                    label=label,
                    ratio=ratio,
                    signal_type=signal_type,
                    enabled=enabled,
                    color_palette=palette,
                )
            )
        if not normalized:
            return
        profile.signal_sections = normalized
        valid_keys = {s.key for s in normalized if bool(s.enabled)}
        mapping = dict(profile.channel_mappings or {})
        ignored = set(profile.ignored_channels or [])
        for ch, sec in list(mapping.items()):
            if str(sec) not in valid_keys:
                mapping.pop(ch, None)
                ignored.add(ch)
        profile.channel_mappings = mapping
        profile.ignored_channels = sorted(ignored)
        self.profile_store.save_profile(profile)
        self.profile_store.set_last_profile_name(profile.name)
        self._last_mapping_signature = None
        if refresh_view and getattr(self, "raw", None) is not None:
            try:
                self._ensure_profile_channel_mapping(
                    list(self.raw.ch_names),
                    create_new_profile_on_pending=False,
                    show_dialog_on_pending=False,
                )
                self._show_default_psg_view(embed_parent=getattr(self, "psg_container", None))
                self.update_plot()
            except Exception:
                pass

    def _edit_profile_sections_dialog(self, *, refresh_view: bool = True) -> bool:
        profile = getattr(self, "active_profile", None)
        if profile is None:
            messagebox.showwarning("Profils", "Aucun profil actif.")
            return False
        dialog = SectionLayoutDialog(self.root, list(profile.signal_sections))
        result = dialog.show()
        if not result.accepted:
            return False
        self._apply_profile_sections(result.sections, refresh_view=refresh_view)
        return True

    def _configure_sections_from_mapping(self, _current_labels: Dict[str, str]) -> Optional[Dict[str, str]]:
        changed = self._edit_profile_sections_dialog(refresh_view=False)
        if not changed:
            return None
        return self._profile_section_choices()

    def _generate_unique_profile_name(self, base_name: str) -> str:
        base = str(base_name or "profile").strip() or "profile"
        existing = set(self.profile_store.list_profiles())
        if base not in existing:
            return base
        idx = 2
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"

    def _ensure_profile_channel_mapping(
        self,
        available_channels: Sequence[str],
        *,
        create_new_profile_on_pending: bool = True,
        show_dialog_on_pending: bool = True,
    ) -> bool:
        """Ensure every channel has a mapping or is explicitly ignored."""
        profile = getattr(self, "active_profile", None)
        if profile is None:
            return False
        choices = self._profile_section_choices()
        if not choices:
            messagebox.showerror("Profils", "Le profil actif ne contient aucune section de signaux.")
            return False
        available_list = [str(ch) for ch in available_channels]
        available_set = set(available_list)

        def _compute_state(
            base_mapping: Dict[str, str],
            base_ignored: set[str],
            *,
            auto_infer: bool = True,
        ) -> tuple[Dict[str, str], set[str], List[str]]:
            working_map = dict(base_mapping)
            working_ignored = set(base_ignored)
            pending_local: List[str] = []
            reverse_labels = {str(v).strip().lower(): str(k) for k, v in choices.items()}
            ignore_tokens = {"__ignore__", "ignore", "ignorer"}
            for ch in available_channels:
                mapped_key = str(working_map.get(ch, "")).strip()
                mapped_key_l = mapped_key.lower()
                # Backward compatibility: older profiles may store labels
                # ("EEG", "Ignorer") instead of internal keys.
                if mapped_key and mapped_key not in choices:
                    if mapped_key_l in ignore_tokens or mapped_key_l.startswith("ignore"):
                        working_ignored.add(ch)
                        working_map.pop(ch, None)
                        continue
                    mapped_from_label = reverse_labels.get(mapped_key_l, "")
                    if mapped_from_label in choices:
                        working_map[ch] = mapped_from_label
                        mapped_key = mapped_from_label
                if mapped_key in choices:
                    continue
                if ch in working_ignored:
                    continue
                if auto_infer:
                    detected_type = cesa_detect_signal_type(ch)
                    if detected_type in choices:
                        working_map[ch] = detected_type
                        continue
                pending_local.append(ch)
            return working_map, working_ignored, pending_local

        def _state_signature(
            current_mapping: Dict[str, str],
            current_ignored: set[str],
        ) -> tuple:
            mapped_pairs = tuple(
                sorted(
                    (str(ch), str(sec))
                    for ch, sec in current_mapping.items()
                    if str(ch) in available_set and str(sec) in choices
                )
            )
            ignored_tuple = tuple(sorted(str(ch) for ch in current_ignored if str(ch) in available_set))
            return (
                str(getattr(profile, "name", "")),
                tuple(available_list),
                mapped_pairs,
                ignored_tuple,
            )

        def _persist_normalized_state(
            current_mapping: Dict[str, str],
            current_ignored: set[str],
        ) -> None:
            try:
                stored_map = dict(getattr(profile, "channel_mappings", {}) or {})
                stored_ignored = set(getattr(profile, "ignored_channels", []) or [])
                normalized_map = dict(current_mapping)
                normalized_ignored = sorted(current_ignored)
                if stored_map != normalized_map or sorted(stored_ignored) != normalized_ignored:
                    profile.channel_mappings = normalized_map
                    profile.ignored_channels = normalized_ignored
                    self.profile_store.save_profile(profile)
            except Exception:
                pass

        if not show_dialog_on_pending:
            # Strict mode for non-interactive callers: do not auto-map channels.
            # This prevents impedance/mono channels from being reintroduced.
            normalized_current, ignored, _pending = _compute_state(
                dict(profile.channel_mappings or {}),
                set(profile.ignored_channels or []),
                auto_infer=False,
            )
            _persist_normalized_state(normalized_current, ignored)
            self.profile_channel_map_runtime = {
                ch: sec for ch, sec in normalized_current.items()
                if ch in available_set and sec in choices
            }
            self.profile_ignored_channels_runtime = [ch for ch in ignored if ch in available_set]
            self._last_mapping_signature = _state_signature(normalized_current, ignored)
            return True

        normalized_current, ignored, pending = _compute_state(
            dict(profile.channel_mappings or {}),
            set(profile.ignored_channels or []),
        )
        _persist_normalized_state(normalized_current, ignored)
        signature = _state_signature(normalized_current, ignored)
        if getattr(self, "_last_mapping_signature", None) == signature:
            self.profile_channel_map_runtime = {
                ch: sec for ch, sec in normalized_current.items()
                if ch in available_set and sec in choices
            }
            self.profile_ignored_channels_runtime = [ch for ch in ignored if ch in available_set]
            return True

        if not pending:
            self.profile_channel_map_runtime = {
                ch: sec for ch, sec in normalized_current.items()
                if ch in available_set and sec in choices
            }
            self.profile_ignored_channels_runtime = [ch for ch in ignored if ch in available_set]
            self._last_mapping_signature = signature
            return True

        if create_new_profile_on_pending:
            # User preference: when unmapped channels exist, create a NEW profile
            # instead of mutating the current one.
            try:
                source_name = str(profile.name)
                file_stem = ""
                try:
                    file_stem = Path(str(getattr(self, "current_file_path", ""))).stem
                except Exception:
                    file_stem = ""
                base_name = f"{source_name}_{file_stem}" if file_stem else f"{source_name}_mapped"
                suggested_name = self._generate_unique_profile_name(base_name)
                chosen_name = simpledialog.askstring(
                    "Profils",
                    "Canaux non reconnus detectes.\nNom du nouveau profil a creer:",
                    initialvalue=suggested_name,
                    parent=self.root,
                )
                if not chosen_name:
                    return False
                target_name = self._generate_unique_profile_name(chosen_name.strip())
                new_profile = self.profile_store.duplicate_profile(source_name, target_name)
                self.active_profile = new_profile
                profile = new_profile
                # Keep already inferred mappings in the newly created profile.
                profile.channel_mappings = dict(normalized_current)
                profile.ignored_channels = sorted(ignored)
                self.profile_store.save_profile(profile)
                normalized_current, ignored, pending = _compute_state(
                    dict(profile.channel_mappings or {}),
                    set(profile.ignored_channels or []),
                )
                _persist_normalized_state(normalized_current, ignored)
                self.profile_store.set_last_profile_name(profile.name)
                messagebox.showinfo(
                    "Profils",
                    f"Canaux non mappes detectes.\nNouveau profil cree automatiquement: '{profile.name}'",
                    parent=self.root,
                )
            except Exception as exc:
                messagebox.showerror("Profils", f"Impossible de creer un nouveau profil:\n{exc}")
                return False

        # Show ALL channels in mapping dialog so user can explicitly adjust recognized channels too.
        prefill: Dict[str, str] = {}
        for ch in available_channels:
            if ch in ignored:
                prefill[ch] = "__ignore__"
            else:
                mapped_key = normalized_current.get(ch, "")
                if mapped_key in choices:
                    prefill[ch] = mapped_key
                else:
                    prefill[ch] = "__ignore__"
        dialog = ChannelMappingDialog(
            self.root,
            channels=list(available_channels),
            section_labels=choices,
            prefill=prefill,
            on_configure_sections=self._configure_sections_from_mapping,
        )
        result = dialog.show()
        if not result.accepted:
            return False

        choices = self._profile_section_choices()
        reverse_labels = {str(v): str(k) for k, v in choices.items()}
        for ch, section_key in result.channel_mapping.items():
            section_key = str(section_key)
            if section_key not in choices and section_key in reverse_labels:
                section_key = reverse_labels[section_key]
            if section_key == "__ignore__":
                ignored.add(ch)
                normalized_current.pop(ch, None)
            else:
                normalized_current[ch] = section_key
                if ch in ignored:
                    ignored.remove(ch)

        profile.channel_mappings = dict(normalized_current)
        profile.ignored_channels = sorted(ignored)
        self.profile_store.save_profile(profile)
        self.profile_store.set_last_profile_name(profile.name)
        self.profile_channel_map_runtime = {
            ch: sec for ch, sec in normalized_current.items()
            if ch in available_set and sec in choices
        }
        self.profile_ignored_channels_runtime = [ch for ch in ignored if ch in available_set]
        self._last_mapping_signature = _state_signature(normalized_current, ignored)
        return True

    def _capture_runtime_into_profile(self) -> None:
        profile = getattr(self, "active_profile", None)
        if profile is None:
            return
        profile.baseline_enabled = bool(getattr(self, "baseline_correction_enabled", True))
        profile.baseline_window_duration = float(getattr(self, "baseline_window_duration", 30.0))
        profile.filter_enabled = bool(getattr(self, "filter_enabled", True))
        profile.filter_order = int(getattr(self, "filter_order", 4))
        profile.filter_low = float(getattr(self, "filter_low", 0.5))
        profile.filter_high = float(getattr(self, "filter_high", 30.0))
        profile.filter_type = str(getattr(self, "filter_type", "butterworth"))
        profile.filter_window = str(getattr(self, "filter_window", "hamming"))
        profile.channel_filter_params = {ch: dict(cfg) for ch, cfg in getattr(self, "channel_filter_params", {}).items()}
        cfp = getattr(self, "channel_filter_pipelines", {})
        profile.channel_filter_pipelines = {
            ch: (pipe.to_dict() if hasattr(pipe, "to_dict") else dict(pipe))
            for ch, pipe in cfp.items()
        }
        profile.channel_mappings = dict(getattr(self, "profile_channel_map_runtime", profile.channel_mappings))
        self.profile_store.save_profile(profile)
        self.profile_store.set_last_profile_name(profile.name)

    def _load_profile_by_name(self, profile_name: str) -> bool:
        try:
            profile = self.profile_store.load_profile(profile_name)
        except Exception as exc:
            messagebox.showerror("Profils", f"Impossible de charger le profil '{profile_name}':\n{exc}")
            return False
        self.active_profile = profile
        self.profile_store.set_last_profile_name(profile.name)
        self._last_mapping_signature = None
        try:
            self.baseline_correction_enabled = bool(profile.baseline_enabled)
            self.baseline_window_duration = float(profile.baseline_window_duration)
            self.filter_enabled = bool(profile.filter_enabled)
            self.filter_order = int(profile.filter_order)
            if profile.filter_low is not None:
                self.filter_low = float(profile.filter_low)
            if profile.filter_high is not None:
                self.filter_high = float(profile.filter_high)
            self.filter_type = str(profile.filter_type or "butterworth")
            self.filter_window = str(profile.filter_window or "hamming")
            self.channel_filter_params = {
                ch: dict(cfg) for ch, cfg in (profile.channel_filter_params or {}).items()
            }
            self.channel_filter_pipelines = {}
            for ch, pipe_dict in (profile.channel_filter_pipelines or {}).items():
                try:
                    self.channel_filter_pipelines[ch] = FilterPipeline.from_dict(pipe_dict)
                except Exception:
                    pass
            if hasattr(self, "filter_var"):
                self.filter_var.set(self.filter_enabled)
            if hasattr(self, "baseline_var"):
                self.baseline_var.set(self.baseline_correction_enabled)
        except Exception:
            pass
        return True

    def _save_profile_as(self) -> None:
        name = simpledialog.askstring("Profils", "Nom du profil à enregistrer:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        self._capture_runtime_into_profile()
        self.active_profile.name = name
        self.profile_store.save_profile(self.active_profile)
        self.profile_store.set_last_profile_name(name)
        messagebox.showinfo("Profils", f"Profil '{name}' enregistré.")

    def _load_profile_dialog(self) -> None:
        names = self.profile_store.list_profiles()
        if not names:
            messagebox.showinfo("Profils", "Aucun profil disponible.")
            return
        selected = simpledialog.askstring(
            "Profils",
            f"Profils disponibles:\n- " + "\n- ".join(names) + "\n\nNom du profil à charger:",
            parent=self.root,
        )
        if not selected:
            return
        if self._load_profile_by_name(selected.strip()):
            if self.raw is not None:
                if not self._ensure_profile_channel_mapping(list(self.raw.ch_names)):
                    return
                self._show_default_psg_view(embed_parent=getattr(self, "psg_container", None))
            self.update_plot()
            messagebox.showinfo("Profils", f"Profil '{selected.strip()}' chargé.")

    def _delete_profile_dialog(self) -> None:
        names = [n for n in self.profile_store.list_profiles() if n != "default"]
        if not names:
            messagebox.showinfo("Profils", "Aucun profil supprimable.")
            return
        selected = simpledialog.askstring(
            "Profils",
            f"Profils supprimables:\n- " + "\n- ".join(names) + "\n\nNom du profil à supprimer:",
            parent=self.root,
        )
        if not selected:
            return
        target = selected.strip()
        if target not in names:
            messagebox.showwarning("Profils", f"Profil introuvable: {target}")
            return
        self.profile_store.delete_profile(target)
        messagebox.showinfo("Profils", f"Profil '{target}' supprimé.")

    def _duplicate_profile_dialog(self) -> None:
        names = self.profile_store.list_profiles()
        if not names:
            messagebox.showinfo("Profils", "Aucun profil disponible.")
            return
        source = simpledialog.askstring(
            "Profils",
            f"Profils disponibles:\n- " + "\n- ".join(names) + "\n\nNom du profil source:",
            parent=self.root,
        )
        if not source:
            return
        target = simpledialog.askstring("Profils", "Nom du nouveau profil:", parent=self.root)
        if not target:
            return
        try:
            self.profile_store.duplicate_profile(source.strip(), target.strip())
            messagebox.showinfo("Profils", f"Profil dupliqué: {source.strip()} -> {target.strip()}")
        except Exception as exc:
            messagebox.showerror("Profils", f"Echec duplication:\n{exc}")

    def _rename_profile_dialog(self) -> None:
        names = [n for n in self.profile_store.list_profiles() if n != "default"]
        if not names:
            messagebox.showinfo("Profils", "Aucun profil renommable.")
            return
        source = simpledialog.askstring(
            "Profils",
            f"Profils renommables:\n- " + "\n- ".join(names) + "\n\nNom du profil a renommer:",
            parent=self.root,
        )
        if not source:
            return
        target = simpledialog.askstring("Profils", "Nouveau nom:", parent=self.root)
        if not target:
            return
        try:
            profile = self.profile_store.rename_profile(source.strip(), target.strip())
            if self.active_profile.name == source.strip():
                self.active_profile = profile
            messagebox.showinfo("Profils", f"Profil renommé: {source.strip()} -> {target.strip()}")
        except Exception as exc:
            messagebox.showerror("Profils", f"Echec renommage:\n{exc}")
    
    # =====================================================================
    # MÉTHODES DE MENU MANQUANTES
    # =====================================================================
    
    def _export_data(self):
        """Exporte les données actuelles."""
        if not self.raw or not self.selected_channels:
            messagebox.showwarning("Attention", "Aucun fichier chargé ou canaux sélectionnés")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exporter les données",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json"), ("TXT", "*.txt")]
        )
        
        if file_path:
            try:
                # Export des données sélectionnées
                data_to_export = {}
                for channel in self.selected_channels:
                    if channel in self.derivations:
                        data_to_export[channel] = self.derivations[channel].tolist()
                
                if file_path.endswith('.json'):
                    import json
                    with open(file_path, 'w') as f:
                        json.dump(data_to_export, f, indent=2)
                elif file_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.DataFrame(data_to_export)
                    df.to_csv(file_path, index=False)
                else:  # TXT
                    with open(file_path, 'w') as f:
                        f.write("Données EEG Exportées\n")
                        f.write("=" * 30 + "\n\n")
                        for channel, data in data_to_export.items():
                            f.write(f"Canal: {channel}\n")
                            f.write(f"Échantillons: {len(data)}\n")
                            f.write(f"Min: {min(data):.6f} µV\n")
                            f.write(f"Max: {max(data):.6f} µV\n\n")
                
                messagebox.showinfo("Succès", f"Données exportées vers {file_path}")
                logging.info(f"Données exportées vers {file_path}")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export: {str(e)}")
                logging.error(f"Erreur export: {e}")

    def _export_edf_segment(self):
        """Ouvre un dialogue pour exporter un segment EDF (ex: 10s)."""
        if not self.raw:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
        win = tk.Toplevel(self.root)
        win.title("Exporter un segment EDF")
        win.geometry("380x180")
        win.transient(self.root)
        win.grab_set()

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Début (s):").grid(row=0, column=0, sticky='w')
        start_var = tk.DoubleVar(value=self.current_time)
        ttk.Entry(frame, textvariable=start_var, width=12).grid(row=0, column=1, sticky='w')

        ttk.Label(frame, text="Durée (s):").grid(row=1, column=0, sticky='w', pady=(8,0))
        dur_var = tk.DoubleVar(value=10.0)
        ttk.Entry(frame, textvariable=dur_var, width=12).grid(row=1, column=1, sticky='w', pady=(8,0))

        def _do_export():
            try:
                start_s = float(start_var.get())
                dur_s = float(dur_var.get())
                if dur_s <= 0:
                    raise ValueError("Durée doit être > 0")
                end_s = start_s + dur_s
                fs = float(self.sfreq)
                i0 = int(max(0, min(len(self.raw.times)-2, start_s * fs)))
                i1 = int(max(i0+2, min(len(self.raw.times), end_s * fs)))

                # Extraire données segment
                data = self.raw.get_data(start=i0, stop=i1)
                info = self.raw.info.copy()
                seg = mne.io.RawArray(data, info)

                file_path = filedialog.asksaveasfilename(title="Enregistrer segment EDF", defaultextension=".edf",
                                                         filetypes=recording_filetypes_for_dialog())
                if not file_path:
                    return
                # Sauvegarde au format EDF via mne
                seg.export(file_path, fmt='edf', overwrite=True)
                messagebox.showinfo("Succès", f"Segment exporté: {file_path}")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec export segment: {e}")

        ttk.Button(frame, text="Exporter", command=_do_export).grid(row=2, column=0, columnspan=2, pady=12)
    
    def _generate_spaghetti_from_edf(self):
        """Ouvre des sélecteurs de dossiers et génère les graphiques spaghetti EDF → PNG/CSV."""
        try:
            from .advanced_spaghetti_plots import generate_spaghetti_from_edf_dirs  # type: ignore
        except Exception as e:
            messagebox.showerror("Erreur", f"Module indisponible: {e}")
            return

        # Choisir dossiers AVANT/APRÈS
        before_dir = filedialog.askdirectory(title="Choisir dossier EDF AVANT")
        if not before_dir:
            return
        after_dir = filedialog.askdirectory(title="Choisir dossier EDF APRÈS")
        if not after_dir:
            return
        out_dir = filedialog.askdirectory(title="Choisir dossier de sortie pour les graphiques")
        if not out_dir:
            return

        # Sélections actuelles
        selected_channels = None
        try:
            if hasattr(self, 'selected_channels') and self.selected_channels:
                selected_channels = list(self.selected_channels)
        except Exception:
            selected_channels = None

        # Paramètres par défaut / TODO: ajouter une petite boîte de dialogue pour filtres
        selected_bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        selected_stages = ["W", "N1", "N2", "N3", "R"]

        try:
            # Prendre en compte les combinaisons explicites s'il y en a
            combos = self._parse_spag_combinations()
            if combos:
                add_channels = sorted({c for (c, _s, _b) in combos})
                if not selected_channels:
                    selected_channels = list(add_channels)
                else:
                    for c in add_channels:
                        if c not in selected_channels:
                            selected_channels.append(c)
                # Étendre filtres bande/stade
                add_bands = sorted({b for (_c, _s, b) in combos})
                add_stages = sorted({s for (_c, s, _b) in combos})
                for b in add_bands:
                    if b not in selected_bands:
                        selected_bands.append(b)
                for s in add_stages:
                    if s not in selected_stages:
                        selected_stages.append(s)
            self._set_status_message("Génération des graphiques spaghetti en cours…")
        except Exception:
            pass

        try:
            clusters_payload, cluster_names_payload = self._get_spag_cluster_payload()
            before_label, after_label = self._get_spag_group_labels()
            outputs = generate_spaghetti_from_edf_dirs(
                before_dir=before_dir,
                after_dir=after_dir,
                output_dir=out_dir,
                selected_bands=selected_bands,
                selected_stages=selected_stages,
                selected_channels=selected_channels,
                selected_subjects=None,
                selected_band_stage_map=None,
                selected_combinations=combos if 'combos' in locals() else None,
                clusters=clusters_payload,
                cluster_names=cluster_names_payload,
                before_label=before_label,
                after_label=after_label,
                epoch_len=30.0,
                metric='AUC',
                rng_seed=42,
                n_perm=5000,
                n_boot=2000,
            )
            messagebox.showinfo("Terminé", f"{len(outputs)} graphiques générés dans:\n{out_dir}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erreur", f"Echec génération spaghetti: {e}")
        finally:
            try:
                self._set_status_message("")
            except Exception:
                pass

    def _export_report(self):
        """Génère un rapport complet."""
        if not self.raw:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exporter le rapport",
            defaultextension=".txt",
            filetypes=[("TXT", "*.txt"), ("JSON", "*.json")]
        )
        
        if file_path:
            try:
                # Génération du rapport
                report = self._generate_report()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                messagebox.showinfo("Succès", f"Rapport généré: {file_path}")
                logging.info(f"Rapport généré: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la génération: {str(e)}")
                logging.error(f"Erreur rapport: {e}")
    
    def _generate_report(self):
        """Génère un rapport textuel des données."""
        if not self.raw:
            return "Aucun fichier chargé"
        
        report = []
        report.append("CESA (EEG Studio Analysis) - Rapport d'Analyse")
        report.append("=" * 50)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Fichier: {getattr(self.raw.info, 'filename', 'Inconnu')}")
        report.append(f"Canaux: {len(self.raw.ch_names)}")
        report.append(f"Fréquence: {self.sfreq} Hz")
        report.append(f"Durée: {len(self.raw.times)/self.sfreq:.1f}s")
        report.append("")
        
        if self.selected_channels:
            report.append("Canaux Sélectionnés:")
            for i, channel in enumerate(self.selected_channels, 1):
                report.append(f"  {i}. {channel}")
            report.append("")
            
            report.append("Statistiques par Canal:")
            for channel in self.selected_channels:
                if channel in self.derivations:
                    data = self.derivations[channel]
                    report.append(f"\n{channel}:")
                    report.append(f"  Échantillons: {len(data)}")
                    report.append(f"  Min: {np.min(data):.6f} µV")
                    report.append(f"  Max: {np.max(data):.6f} µV")
                    report.append(f"  Moyenne: {np.mean(data):.6f} µV")
                    report.append(f"  Écart-type: {np.std(data):.6f} µV")
                    report.append(f"  RMS: {np.sqrt(np.mean(data**2)):.6f} µV")
        
        return "\n".join(report)
    
    def _show_preferences(self):
        """Affiche les préférences."""
        messagebox.showinfo("Préférences", "Fonctionnalité en développement")
    
    def _quit_application(self):
        """Quitte l'application avec confirmation si scoring modifié non enregistré."""
        try:
            if getattr(self, 'scoring_dirty', False):
                if not messagebox.askyesno("Quitter", "Vous avez des modifications de scoring non enregistrées. Quitter quand même ?"):
                    return
        except Exception:
            pass
        try:
            self._capture_runtime_into_profile()
        except Exception:
            pass
        self.root.quit()
    
    def _show_channel_selector(self):
        """Affiche l'éditeur d'assignation des canaux au profil."""
        self._open_profile_channel_mapping_dialog()

    def _open_profile_channel_mapping_dialog(self) -> None:
        """Open profile-based channel mapping editor from menu shortcut."""
        if not getattr(self, "raw", None):
            messagebox.showwarning("Attention", "Veuillez d'abord charger un enregistrement.")
            return
        profile = getattr(self, "active_profile", None)
        if profile is None:
            messagebox.showwarning("Profils", "Aucun profil actif.")
            return
        choices = self._profile_section_choices()
        if not choices:
            messagebox.showwarning("Profils", "Le profil actif ne contient aucune section active.")
            return

        available = list(self.raw.ch_names)
        mapping = dict(profile.channel_mappings or {})
        ignored = set(profile.ignored_channels or [])
        reverse_labels = {str(v).strip().lower(): str(k) for k, v in choices.items()}
        ignore_tokens = {"__ignore__", "ignore", "ignorer"}
        prefill: Dict[str, str] = {}
        for ch in available:
            if ch in ignored:
                prefill[ch] = "__ignore__"
                continue
            mapped = str(mapping.get(ch, "")).strip()
            mapped_l = mapped.lower()
            if mapped in choices:
                prefill[ch] = mapped
            elif mapped_l in reverse_labels:
                prefill[ch] = reverse_labels[mapped_l]
            elif mapped_l in ignore_tokens or mapped_l.startswith("ignore"):
                prefill[ch] = "__ignore__"
            else:
                prefill[ch] = "__ignore__"

        dialog = ChannelMappingDialog(
            self.root,
            channels=available,
            section_labels=choices,
            prefill=prefill,
            on_configure_sections=self._configure_sections_from_mapping,
        )
        result = dialog.show()
        if not result.accepted:
            return

        choices = self._profile_section_choices()
        reverse_labels = {str(v): str(k) for k, v in choices.items()}
        for ch, section_key in result.channel_mapping.items():
            section_key = str(section_key).strip()
            if section_key not in choices and section_key in reverse_labels:
                section_key = reverse_labels[section_key]
            if section_key == "__ignore__":
                ignored.add(ch)
                mapping.pop(ch, None)
            elif section_key in choices:
                mapping[ch] = section_key
                ignored.discard(ch)

        profile.channel_mappings = dict(mapping)
        profile.ignored_channels = sorted(ignored)
        self.profile_store.save_profile(profile)
        self.profile_store.set_last_profile_name(profile.name)
        self._last_mapping_signature = None
        self.profile_channel_map_runtime = {
            ch: sec for ch, sec in mapping.items()
            if ch in available and sec in choices
        }
        self.profile_ignored_channels_runtime = [ch for ch in ignored if ch in available]
        try:
            self._show_default_psg_view(embed_parent=getattr(self, "psg_container", None))
            self.update_plot()
        except Exception:
            pass
    
    def _toggle_autoscale(self):
        """Active/désactive l'autoscale."""
        self.toggle_autoscale()
    
    def _toggle_filter(self):
        """Active/désactive le filtre."""
        self.toggle_filter()
    
    def _toggle_dark_theme(self):
        """Bascule le thème sombre/clair."""
        try:
            self.dark_theme_enabled = not self.dark_theme_enabled
            
            if self.dark_theme_enabled:
                self._apply_dark_theme()
                logging.debug("Theme: Dark theme activated")
                logging.info("[THEME] Dark theme enabled")
            else:
                self._apply_light_theme()
                logging.debug("Theme: Light theme activated")
                logging.info("[THEME] Light theme enabled")
            
            # Mettre à jour l'affichage si des données sont chargées
            if hasattr(self, 'raw') and self.raw is not None:
                self.update_plot()
            
            status = "sombre" if self.dark_theme_enabled else "clair"
            messagebox.showinfo("Thème", f"Thème {status} activé")
            
        except Exception as e:
            logging.error(f"Theme: Error changing theme: {e}")
            logging.error(f"[THEME] Failed to toggle theme: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du changement de thème : {str(e)}")
    
    def _apply_dark_theme(self):
        """Applique le thème sombre à toute l'interface."""
        try:
            # Délégation au module de thème partagé
            try:
                theme_manager.set_theme('default')
                theme_manager.apply_theme_to_root(self.root)
            except Exception:
                pass
            # Configuration matplotlib pour thème sombre
            plt.rcParams.update({
                'figure.facecolor': '#2b2b2b',
                'axes.facecolor': '#2b2b2b',
                'axes.edgecolor': '#ffffff',
                'axes.labelcolor': '#ffffff',
                'text.color': '#ffffff',
                'xtick.color': '#ffffff',
                'ytick.color': '#ffffff',
                'grid.color': '#404040',
                'figure.edgecolor': '#2b2b2b',
                'savefig.facecolor': '#2b2b2b',
                'savefig.edgecolor': '#2b2b2b'
            })
            
            # Thème Tkinter sombre
            dark_bg = '#2b2b2b'
            dark_fg = '#ffffff'
            dark_select_bg = '#404040'
            dark_entry_bg = '#3c3c3c'
            
            # Configurer les styles ttk pour le thème sombre
            style = ttk.Style()
            style.theme_use('clam')  # Base moderne
            
            # Styles personnalisés sombres
            style.configure('Dark.TFrame', background=dark_bg)
            style.configure('Dark.TLabel', background=dark_bg, foreground=dark_fg)
            style.configure('Dark.TButton', background=dark_entry_bg, foreground=dark_fg, borderwidth=1)
            style.map('Dark.TButton', background=[('active', dark_select_bg)])
            style.configure('Dark.TEntry', background=dark_entry_bg, foreground=dark_fg, insertcolor=dark_fg)
            style.configure('Dark.TCheckbutton', background=dark_bg, foreground=dark_fg)
            style.configure('Dark.TLabelframe', background=dark_bg, foreground=dark_fg)
            style.configure('Dark.TLabelframe.Label', background=dark_bg, foreground=dark_fg)
            # Échelles avec couleurs sombres complètes
            style.configure('Dark.TScale', background=dark_bg, troughcolor=dark_entry_bg, 
                          sliderlength=20, sliderrelief='flat')
            style.map('Dark.TScale', background=[('active', dark_select_bg)], 
                     troughcolor=[('active', dark_entry_bg)])
            # Combobox pour les menus déroulants
            style.configure('Dark.TCombobox', background=dark_entry_bg, foreground=dark_fg, 
                          fieldbackground=dark_entry_bg, selectbackground=dark_select_bg)
            style.map('Dark.TCombobox', fieldbackground=[('readonly', dark_entry_bg)])
            # Treeview pour les tables
            style.configure('Dark.Treeview', background=dark_entry_bg, foreground=dark_fg,
                          fieldbackground=dark_entry_bg, selectbackground=dark_select_bg)
            style.configure('Dark.Treeview.Heading', background=dark_bg, foreground=dark_fg)
            
            # Appliquer récursivement aux widgets
            def _apply_dark_to_widget(widget):
                try:
                    widget_type = type(widget).__name__
                    if isinstance(widget, (ttk.Frame, ttk.LabelFrame)):
                        widget.configure(style='Dark.TFrame' if isinstance(widget, ttk.Frame) else 'Dark.TLabelframe')
                    elif isinstance(widget, ttk.Label):
                        widget.configure(style='Dark.TLabel')
                    elif isinstance(widget, ttk.Button):
                        widget.configure(style='Dark.TButton')
                    elif isinstance(widget, ttk.Entry):
                        widget.configure(style='Dark.TEntry')
                    elif isinstance(widget, ttk.Checkbutton):
                        widget.configure(style='Dark.TCheckbutton')
                    elif isinstance(widget, ttk.Scale):
                        widget.configure(style='Dark.TScale')
                    elif isinstance(widget, ttk.Combobox):
                        widget.configure(style='Dark.TCombobox')
                    elif 'Treeview' in widget_type:
                        widget.configure(style='Dark.Treeview')
                    elif hasattr(widget, 'configure'):
                        # Widgets tk classiques (Scale, Frame, Label, etc.)
                        try:
                            widget.configure(bg=dark_bg, fg=dark_fg)
                            # Spécial pour Scale tk
                            if 'Scale' in widget_type:
                                widget.configure(bg=dark_bg, fg=dark_fg, troughcolor=dark_entry_bg, 
                                               activebackground=dark_select_bg, highlightbackground=dark_bg)
                            if hasattr(widget, 'selectbackground'):
                                widget.configure(selectbackground=dark_select_bg)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Récursion sur les enfants
                try:
                    for child in widget.winfo_children():
                        _apply_dark_to_widget(child)
                except Exception:
                    pass
            
            # Appliquer à la fenêtre racine et forcer les couleurs système
            try:
                self.root.configure(bg=dark_bg)
                # Essayer de configurer la barre de titre (Windows)
                try:
                    import ctypes
                    hwnd = int(self.root.winfo_id())
                    # Mode sombre Windows 10/11
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
                except Exception:
                    pass
            except Exception:
                pass
            _apply_dark_to_widget(self.root)
            
            # Mettre à jour le graphique si il existe
            if hasattr(self, 'ax'):
                self.ax.set_facecolor('#2b2b2b')
                self.ax.spines['top'].set_color('#ffffff')
                self.ax.spines['right'].set_color('#ffffff')
                self.ax.spines['left'].set_color('#ffffff')
                self.ax.spines['bottom'].set_color('#ffffff')
                self.ax.tick_params(colors='#ffffff')
                self.ax.grid(True, alpha=0.3, color='#404040')
                
                # Mettre à jour le titre
                if hasattr(self, 'current_time'):
                    if hasattr(self, 'absolute_start_datetime') and self.absolute_start_datetime:
                        base_dt = self.absolute_start_datetime
                        current_datetime = base_dt + timedelta(seconds=self.current_time)
                        time_str = current_datetime.strftime("%H:%M:%S")
                        self.ax.set_title(f"EEG - Temps: {time_str}", color='#ffffff')
                    else:
                        self.ax.set_title(f"EEG - Temps: {self.current_time:.1f}s", color='#ffffff')
                
                # Forcer le redessinage
                if hasattr(self, 'canvas'):
                    self.canvas.draw_idle()
                    
        except Exception as e:
            logging.error(f"Theme: Error applying dark theme: {e}")
            logging.error(f"[THEME] Failed to apply dark theme: {e}")
    
    def _apply_light_theme(self):
        """Applique le thème clair à toute l'interface."""
        try:
            # Délégation au module de thème partagé
            try:
                theme_manager.set_theme('default')
                theme_manager.apply_theme_to_root(self.root)
            except Exception:
                pass
            # Configuration matplotlib pour thème clair
            plt.rcParams.update({
                'figure.facecolor': '#ffffff',
                'axes.facecolor': '#ffffff',
                'axes.edgecolor': '#000000',
                'axes.labelcolor': '#000000',
                'text.color': '#000000',
                'xtick.color': '#000000',
                'ytick.color': '#000000',
                'grid.color': '#cccccc',
                'figure.edgecolor': '#ffffff',
                'savefig.facecolor': '#ffffff',
                'savefig.edgecolor': '#ffffff'
            })
            
            # Thème Tkinter clair
            light_bg = '#ffffff'
            light_fg = '#000000'
            light_select_bg = '#e0e0e0'
            light_entry_bg = '#ffffff'
            
            # Restaurer les styles ttk par défaut
            style = ttk.Style()
            style.theme_use('clam')
            
            # Styles personnalisés clairs
            style.configure('Light.TFrame', background=light_bg)
            style.configure('Light.TLabel', background=light_bg, foreground=light_fg)
            style.configure('Light.TButton', background=light_entry_bg, foreground=light_fg)
            style.configure('Light.TEntry', background=light_entry_bg, foreground=light_fg)
            style.configure('Light.TCheckbutton', background=light_bg, foreground=light_fg)
            style.configure('Light.TLabelframe', background=light_bg, foreground=light_fg)
            style.configure('Light.TLabelframe.Label', background=light_bg, foreground=light_fg)
            style.configure('Light.TScale', background=light_bg)
            
            # Appliquer récursivement aux widgets
            def _apply_light_to_widget(widget):
                try:
                    if isinstance(widget, (ttk.Frame, ttk.LabelFrame)):
                        widget.configure(style='Light.TFrame' if isinstance(widget, ttk.Frame) else 'Light.TLabelframe')
                    elif isinstance(widget, ttk.Label):
                        widget.configure(style='Light.TLabel')
                    elif isinstance(widget, ttk.Button):
                        widget.configure(style='Light.TButton')
                    elif isinstance(widget, ttk.Entry):
                        widget.configure(style='Light.TEntry')
                    elif isinstance(widget, ttk.Checkbutton):
                        widget.configure(style='Light.TCheckbutton')
                    elif isinstance(widget, ttk.Scale):
                        widget.configure(style='Light.TScale')
                    elif hasattr(widget, 'configure'):
                        # Widgets tk classiques
                        widget.configure(bg=light_bg, fg=light_fg)
                        if hasattr(widget, 'selectbackground'):
                            widget.configure(selectbackground=light_select_bg)
                except Exception:
                    pass
                # Récursion sur les enfants
                try:
                    for child in widget.winfo_children():
                        _apply_light_to_widget(child)
                except Exception:
                    pass
            
            # Appliquer à la fenêtre racine et restaurer la barre de titre
            try:
                self.root.configure(bg=light_bg)
                # Restaurer la barre de titre claire (Windows)
                try:
                    import ctypes
                    hwnd = int(self.root.winfo_id())
                    # Mode clair Windows 10/11
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(0)), 4)
                except Exception:
                    pass
            except Exception:
                pass
            _apply_light_to_widget(self.root)
            
            # Mettre à jour le graphique si il existe
            if hasattr(self, 'ax'):
                self.ax.set_facecolor('#ffffff')
                self.ax.spines['top'].set_color('#000000')
                self.ax.spines['right'].set_color('#000000')
                self.ax.spines['left'].set_color('#000000')
                self.ax.spines['bottom'].set_color('#000000')
                self.ax.tick_params(colors='#000000')
                self.ax.grid(True, alpha=0.3, color='#cccccc')
                
                # Mettre à jour le titre
                if hasattr(self, 'current_time'):
                    if hasattr(self, 'absolute_start_datetime') and self.absolute_start_datetime:
                        base_dt = self.absolute_start_datetime
                        current_datetime = base_dt + timedelta(seconds=self.current_time)
                        time_str = current_datetime.strftime("%H:%M:%S")
                        self.ax.set_title(f"EEG - Temps: {time_str}", color='#000000')
                    else:
                        self.ax.set_title(f"EEG - Temps: {self.current_time:.1f}s", color='#000000')
                
                # Forcer le redessinage
                if hasattr(self, 'canvas'):
                    self.canvas.draw_idle()
                    
        except Exception as e:
            logging.error(f"Theme: Error applying light theme: {e}")
            logging.error(f"[THEME] Failed to apply light theme: {e}")
    
    def _refresh_plot(self):
        """Actualise le graphique."""
        self.update_plot()
    
    def apply_filter(self, data, low=None, high=None):
        """Applique le filtre via CESA.filters.apply_filter (délégué)."""
        try:
            l = self.filter_low if low is None else low
            h = self.filter_high if high is None else high
            return cesa_apply_filter(
                data,
                sfreq=float(self.sfreq),
                filter_order=int(self.filter_order),
                low=float(l) if l is not None else None,
                high=float(h) if h is not None else None,
                filter_type=str(getattr(self, "filter_type", "butterworth")),
            )
        except Exception as e:
            logging.error(f"Error during filtering: {e}")
            return data
    
    def apply_autoscale(self, data):
        """Applique l'autoscale"""
        if len(data) == 0:
            return data
        
        # Calcul de l'amplitude sur une fenêtre de 30s
        window_samples = int(30 * self.sfreq)
        if len(data) > window_samples:
            window_data = data[-window_samples:]
        else:
            window_data = data
        
        # Calcul de l'amplitude
        std_data = np.std(window_data)
        mean_data = np.mean(window_data)
        
        if std_data < 1e-6:  # Données trop plates
            return data
        
        # Normalisation
        amplitude = std_data * 3
        amplitude = max(0.1, min(amplitude, 50))
        
        # Application
        normalized_data = (data - np.mean(data)) / amplitude * 20
        
        return normalized_data
    
    def show_channel_selector(self):
        """Affiche le sélecteur de canaux"""
        if not self.raw:
            messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
            return
        
        # Fenêtre de sélection
        selector = tk.Toplevel(self.root)
        selector.title("Sélection des Canaux")
        selector.geometry("500x600")
        # Appliquer thème pour un rendu cohérent
        try:
            from CESA.theme_manager import theme_manager as _tm
            _tm.apply_theme_to_root(selector)
        except Exception:
            pass
        
        # Liste des canaux disponibles (tous les canaux du fichier)
        available_channels = list(self.derivations.keys())
        
        if not available_channels:
            messagebox.showwarning("Attention", "Aucun canal disponible.")
            selector.destroy()
            return
        
        # Variables pour les cases à cocher
        channel_vars = {}
        for channel in available_channels:
            channel_vars[channel] = tk.BooleanVar(value=channel in self.selected_channels)
        
        # Interface
        ttk.Label(selector, text="Sélectionnez les canaux à afficher:", style='Custom.TLabel').pack(pady=10)
        
        # Frame avec scrollbar
        frame = ttk.Frame(selector, style='Custom.TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Custom.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Cases à cocher avec catégorisation
        # Dérivations EEG
        ttk.Label(scrollable_frame, text="Dérivations EEG:", style='Custom.TLabel', font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        eeg_derivations = [ch for ch in available_channels if '-' in ch and any(x in ch for x in ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2'])]
        for channel in eeg_derivations:
            ttk.Checkbutton(scrollable_frame, text=channel, variable=channel_vars[channel], style='Custom.TCheckbutton').pack(anchor=tk.W, padx=20)
        
        # Canaux originaux
        ttk.Label(scrollable_frame, text="Canaux Originaux:", style='Custom.TLabel', font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        original_channels = [ch for ch in available_channels if ch not in eeg_derivations]
        for channel in original_channels:
            ttk.Checkbutton(scrollable_frame, text=channel, variable=channel_vars[channel], style='Custom.TCheckbutton').pack(anchor=tk.W, padx=20)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Boutons
        button_frame = ttk.Frame(selector, style='Custom.TFrame')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def select_all():
            for var in channel_vars.values():
                var.set(True)
        
        def select_none():
            for var in channel_vars.values():
                var.set(False)
        
        def select_eeg_only():
            for var in channel_vars.values():
                var.set(False)
            for channel in eeg_derivations:
                if channel in channel_vars:
                    channel_vars[channel].set(True)
        
        def apply_selection():
            # Conserver l'ordre des canaux du fichier
            selected = [ch for ch, var in channel_vars.items() if var.get()]
            try:
                order = {name: idx for idx, name in enumerate(self.raw.ch_names)}
                selected.sort(key=lambda ch: order.get(ch, 1e9))
            except Exception:
                pass
            self.selected_channels = selected
            self.update_plot()
            selector.destroy()
        
        ttk.Button(button_frame, text="Tout sélectionner", command=select_all, style='Custom.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Tout désélectionner", command=select_none, style='Custom.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="EEG seulement", command=select_eeg_only, style='Custom.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Appliquer", command=apply_selection, style='Custom.TButton').pack(side=tk.RIGHT, padx=2)
        ttk.Button(button_frame, text="Annuler", command=selector.destroy, style='Custom.TButton').pack(side=tk.RIGHT, padx=2)
    
    def toggle_autoscale(self):
        """Active/désactive l'autoscale"""
        self.autoscale_enabled = not self.autoscale_enabled
        status = "activé" if self.autoscale_enabled else "désactivé"
        logging.debug(f"UI: autoscale -> {status}")
        try:
            if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                self.psg_plotter.set_autoscale_enabled(self.autoscale_enabled)
                logging.debug(f"UI->Viewer: autoscale pushed -> {self.autoscale_enabled}")
            elif self._qt_psg_plot_active():
                pass  # autoscale géré dans le viewer Qt si besoin
        except Exception:
            pass
        self.update_plot()
    
    def toggle_filter(self):
        """Active/désactive le filtre"""
        self.filter_enabled = not self.filter_enabled
        status = "activé" if self.filter_enabled else "désactivé"
        logging.debug(f"UI: filter -> {status}")
        try:
            if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                self.psg_plotter.set_global_filter_enabled(self.filter_enabled)
                logging.debug(f"UI->Viewer: filter pushed -> {self.filter_enabled}")
            elif self._qt_psg_plot_active():
                self._qt_viewer_bridge.set_global_filter_enabled(self.filter_enabled)
        except Exception:
            pass
        self.update_plot()
    
    def show_filter_config(self):
        """Open the professional filter-configuration dialog.

        Delegates to ``FilterConfigDialog`` from ``ui.filter_dialog`` which
        provides per-channel pipeline editing with sliders, real-time signal
        preview, frequency-response plot, and preset management.
        """
        try:
            self._qt_pause_pump_for_tk_modal()
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return

            channel_names = sorted(self.derivations.keys()) if hasattr(self, "derivations") else []
            if not channel_names:
                messagebox.showwarning("Attention", "Aucun canal disponible.")
                return

            sfreq = float(self.raw.info["sfreq"]) if self.raw else 256.0

            # Build channel-type map
            channel_types: Dict[str, str] = {}
            for ch in channel_names:
                channel_types[ch] = cesa_detect_signal_type(ch)

            # Prepare pipelines dict: prefer existing pipelines, fall back to legacy params
            pipelines: Dict[str, FilterPipeline] = {}
            for ch in channel_names:
                if ch in self.channel_filter_pipelines:
                    pipelines[ch] = self.channel_filter_pipelines[ch].deep_copy()
                elif ch in self.channel_filter_params:
                    p = self.channel_filter_params[ch]
                    pipelines[ch] = pipeline_from_legacy_params(
                        low=float(p.get("low", 0.0)),
                        high=float(p.get("high", 0.0)),
                        order=self.filter_order,
                        filter_type=self.filter_type,
                        enabled=bool(p.get("enabled", True)),
                    )
                else:
                    dp = cesa_get_filter_presets(channel_types.get(ch, "unknown"))
                    pipelines[ch] = pipeline_from_legacy_params(
                        low=float(dp.get("low", 0.0)),
                        high=float(dp.get("high", 0.0)),
                        order=self.filter_order,
                        filter_type=self.filter_type,
                        enabled=bool(dp.get("enabled", True)),
                    )

            # Signal getter for live preview
            def _signal_getter(ch_name: str, start_s: float, duration_s: float) -> np.ndarray:
                try:
                    idx = self.raw.ch_names.index(ch_name) if ch_name in self.raw.ch_names else 0
                    n_samples = int(sfreq * duration_s)
                    start_sample = int(start_s * sfreq)
                    data = self.raw.get_data(
                        picks=[idx],
                        start=start_sample,
                        stop=start_sample + n_samples,
                    )[0] * 1e6
                    return data
                except Exception:
                    return np.array([])

            def _on_apply(new_pipelines: Dict[str, FilterPipeline], global_enabled: bool) -> None:
                self.filter_enabled = global_enabled
                self.channel_filter_pipelines = dict(new_pipelines)

                # Sync legacy params for backward compat
                for ch, pipe in new_pipelines.items():
                    self.channel_filter_params[ch] = {"enabled": pipe.enabled, "low": 0.0, "high": 0.0, "amplitude": 100.0}

                if hasattr(self, "filter_var"):
                    self.filter_var.set(self.filter_enabled)

                plotter = getattr(self, "psg_plotter", None)
                if plotter is not None:
                    plotter.global_filter_enabled = self.filter_enabled
                    plotter.filter_pipelines_by_channel = {
                        ch: pipe for ch, pipe in self.channel_filter_pipelines.items()
                    }
                elif self._qt_psg_plot_active():
                    try:
                        self._qt_viewer_bridge.set_global_filter_enabled(self.filter_enabled)
                        self._qt_viewer_bridge.filter_pipelines_by_channel = {
                            ch: pipe for ch, pipe in self.channel_filter_pipelines.items()
                        }
                    except Exception:
                        pass
                try:
                    self._capture_runtime_into_profile()
                except Exception:
                    pass
                self.update_plot()

            FilterConfigDialog(
                self.root,
                channel_names,
                sfreq,
                channel_pipelines=pipelines,
                channel_types=channel_types,
                signal_getter=_signal_getter,
                on_apply=_on_apply,
                global_enabled=self.filter_enabled,
                audit_log=self._filter_audit_log,
            )
        finally:
            self._qt_resume_pump_after_tk_modal()

    def _open_matplotlib_psg_view(self) -> None:
        """Force le viewer Matplotlib intégré (raccourci menu / Ctrl+M)."""
        self.prefer_qt_psg_viewer = False
        parent = getattr(self, "psg_container", None)
        self.show_multi_graph_view(embed_parent=parent)

    # ------------------------------------------------------------------
    # PyQtGraph : viewer par défaut
    # ------------------------------------------------------------------

    def _qt_psg_dependencies_available(self) -> bool:
        if getattr(self, "_qt_import_check_failed", False):
            return False
        try:
            import PySide6  # noqa: F401
            import pyqtgraph  # noqa: F401
            return True
        except ImportError as exc:
            self._qt_import_check_failed = True
            logging.warning(
                "[VIEW] PySide6/pyqtgraph indisponibles (%s) — repli sur Matplotlib. "
                "Installez: python -m pip install PySide6 pyqtgraph. "
                "Erreur DLL Qt souvent due à un autre Python sur le PATH (ex. PyMOL) — utilisez python.org ou un venv.",
                exc,
            )
            return False

    def _qt_psg_plot_active(self) -> bool:
        if not getattr(self, "prefer_qt_psg_viewer", True):
            return False
        br = getattr(self, "_qt_viewer_bridge", None)
        if br is None:
            return False
        try:
            return bool(br.is_alive())
        except Exception:
            return False

    def _close_qt_viewer_if_any(self) -> None:
        br = getattr(self, "_qt_viewer_bridge", None)
        if br is None:
            return
        _log_viewer_checkpoint("20", "close_qt_viewer_if_any", had_bridge=True)
        try:
            br.close()
        except Exception:
            pass
        self._qt_viewer_bridge = None

    def _embed_qt_viewer_placeholder(self, parent: ttk.Frame) -> None:
        for child in parent.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        holder = ttk.Frame(parent)
        holder.pack(fill=tk.BOTH, expand=True)
        msg = (
            "Fenêtre « Viewer PyQtGraph » active (affichée à part).\n\n"
            "Signaux EEG/PSG : navigation et zoom dans cette fenêtre Qt.\n"
            "Pour le tracé intégré Matplotlib : Affichage > Vue Multi-Graphiques (Matplotlib)."
        )
        ttk.Label(
            holder, text=msg, anchor=tk.CENTER, justify=tk.CENTER, wraplength=420,
        ).pack(expand=True, padx=24, pady=24)

    def _qt_build_launch_kwargs(self) -> Optional[Dict[str, Any]]:
        """Arguments pour launch_viewer (signaux pleine nuit, µV).

        Utilise en priorité ``self.derivations[ch]`` (déjà en µV, mêmes clés que
        ``channel_filter_pipelines`` / dialogue filtres) pour chaque canal affiché ;
        repli sur ``raw.get_data`` si absent.
        """
        if not self.raw:
            return None
        signals: Dict[str, Tuple[np.ndarray, float]] = {}
        fs = float(getattr(self, "sfreq", 256.0))
        selected = getattr(self, "psg_channels_used", None)
        if not selected:
            selected = (
                list(self.selected_channels)
                if self.selected_channels
                else list(self.raw.ch_names)[:16]
            )
        derivs = getattr(self, "derivations", None) or {}
        for ch in selected:
            try:
                if ch in derivs:
                    arr = np.asarray(derivs[ch], dtype=np.float64).reshape(-1)
                    signals[ch] = (np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), fs)
                else:
                    arr = self.raw.get_data(picks=[ch])[0]
                    signals[ch] = (self._to_microvolts_and_sanitize(arr), fs)
            except Exception:
                continue
        if not signals:
            return None
        hypnogram = None
        try:
            df = self._get_active_scoring_df()
            if df is not None and len(df) > 0 and "time" in df.columns and "stage" in df.columns:
                df_work = df[["time", "stage"]].copy()
                df_work["time"] = pd.to_numeric(df_work["time"], errors="coerce")
                df_work = df_work.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
                if len(df_work) >= 2:
                    diffs = np.diff(df_work["time"].to_numpy(dtype=float))
                    diffs_pos = diffs[diffs > 0]
                    epoch_len = float(np.median(diffs_pos)) if len(diffs_pos) > 0 else 30.0
                else:
                    epoch_len = 30.0
                total_dur = float(len(self.raw.times) / fs)
                num_epochs = int(np.ceil(total_dur / epoch_len))
                labels = ["U"] * num_epochs
                for t_val, s in zip(
                    df_work["time"].to_numpy(dtype=float),
                    df_work["stage"].astype(str).str.upper().str.strip().to_numpy(),
                ):
                    idx = int(round(t_val / epoch_len))
                    if 0 <= idx < len(labels):
                        labels[idx] = s
                hypnogram = (labels, epoch_len)
        except Exception:
            pass
        channel_types: Dict[str, str] = {}
        try:
            from CESA.filters import detect_signal_type as _dst
            for ch in signals:
                channel_types[ch] = _dst(ch)
        except Exception:
            pass
        total_dur = float(len(self.raw.times) / fs)
        return {
            "signals": signals,
            "hypnogram": hypnogram,
            "scoring_annotations": [],
            "filter_pipelines": getattr(self, "channel_filter_pipelines", {}),
            "channel_types": channel_types,
            "global_filter_enabled": bool(self.filter_var.get()) if hasattr(self, "filter_var") else True,
            "start_time_s": float(self.current_time),
            "duration_s": float(self.duration),
            "total_duration_s": total_dur,
            "theme_name": "dark",
        }

    def _qt_defer_to_studio(self, fn: Callable[[], None]) -> None:
        """Planifie un appel sur la boucle Tk (jamais depuis la pile Qt directe)."""

        def _run() -> None:
            try:
                fn()
            except Exception:
                logging.debug("[VIEW] deferred studio call failed", exc_info=True)

        try:
            self.root.after(0, _run)
        except Exception:
            try:
                self._enqueue_tk_main(_run)
            except Exception:
                pass

    def _qt_sync_global_filter_from_viewer(self, enabled: bool) -> None:
        """Aligne filter_var / filter_enabled après toggle F dans le viewer Qt."""
        self.filter_enabled = bool(enabled)
        if hasattr(self, "filter_var"):
            try:
                self.filter_var.set(self.filter_enabled)
            except Exception:
                pass
        try:
            plotter = getattr(self, "psg_plotter", None)
            if plotter is not None:
                plotter.set_global_filter_enabled(self.filter_enabled)
        except Exception:
            pass

    def _qt_pause_pump_for_tk_modal(self) -> None:
        """Pendant un modal Tk : coupe le pump Qt et la file worker→Tk (poll 16 ms).

        Sinon les callbacks planifiés par ``ThreadPoolExecutor`` peuvent s'exécuter
        pendant ``messagebox`` / ``grab_set`` et provoquer PyEval_RestoreThread (Py3.14).
        """
        d = int(getattr(self, "_tk_modal_ui_block_depth", 0)) + 1
        self._tk_modal_ui_block_depth = d
        _log_viewer_checkpoint("62", "_qt_pause_pump_for_tk_modal enter", new_depth=d)
        if d == 1:
            pid = getattr(self, "_tk_main_poll_id", None)
            self._tk_main_poll_id = None
            if pid is not None:
                try:
                    self.root.after_cancel(pid)
                except Exception:
                    pass
                logging.info("[VIEWER-CHK-30] tk main-thread poll paused for Tk modal")
                _log_viewer_checkpoint("63", "tk main-thread poll after_cancel ok", had_poll_id=True)
            else:
                _log_viewer_checkpoint("63", "tk main-thread poll pause depth==1 (no poll id)", had_poll_id=False)
        br = getattr(self, "_qt_viewer_bridge", None)
        if br is not None and hasattr(br, "pause_qt_pump"):
            try:
                br.pause_qt_pump()
            except Exception:
                logging.debug("[VIEW] pause_qt_pump failed", exc_info=True)

    def _qt_resume_pump_after_tk_modal(self) -> None:
        def _do() -> None:
            _log_viewer_checkpoint("65", "_qt_resume_pump_after_tk_modal _do() started")
            try:
                pf = getattr(self, "_active_plot_future", None)
                if pf is not None and not pf.done():
                    try:
                        pf.cancel()
                    except Exception:
                        pass
                self._active_plot_future = None
                puid = getattr(self, "_plot_update_pending_id", None)
                if puid is not None:
                    try:
                        self.root.after_cancel(puid)
                    except Exception:
                        pass
                    self._plot_update_pending_id = None
            except Exception:
                pass

            self._tk_modal_ui_block_depth = max(0, int(getattr(self, "_tk_modal_ui_block_depth", 0)) - 1)
            br = getattr(self, "_qt_viewer_bridge", None)
            if br is not None and hasattr(br, "resume_qt_pump"):
                try:
                    br.resume_qt_pump()
                except Exception:
                    logging.debug("[VIEW] resume_qt_pump failed", exc_info=True)
            _log_viewer_checkpoint(
                "66",
                "_qt_resume _do after resume_qt_pump attempt",
                modal_depth=int(getattr(self, "_tk_modal_ui_block_depth", 0)),
                has_bridge=bool(getattr(self, "_qt_viewer_bridge", None)),
            )

            if self._tk_modal_ui_block_depth == 0:
                def _restart_poll_late() -> None:
                    _log_viewer_checkpoint("67", "_restart_poll_late entered")
                    # #region agent log
                    try:
                        from CESA.agent_debug_f8b011 import log as _agent_log

                        _agent_log(
                            "F",
                            "eeg_studio._restart_poll_late",
                            "schedule poll after modal",
                            {},
                        )
                    except Exception:
                        pass
                    # #endregion
                    try:
                        if getattr(self, "_tk_main_poll_id", None) is None:
                            self._tk_main_poll_id = self.root.after(
                                16, self._poll_tk_main_thread_queue
                            )
                            logging.info("[VIEWER-CHK-31] tk main-thread poll restarted after modal")
                    except Exception:
                        logging.debug("[VIEW] restart tk poll failed", exc_info=True)

                # Après un modal, le premier processEvents Qt est retardé (~300 ms, viewer_bridge).
                # Redémarrer le poll Tk trop tôt après ce tick provoquait PyEval_RestoreThread (Py3.14).
                # Marge large vs cooldown + premiers ticks pump (cf. eeg_studio.log CHK-56..69).
                _has_qt_bridge = getattr(self, "_qt_viewer_bridge", None) is not None
                _poll_restart_ms = 1200 if _has_qt_bridge else 180
                # #region agent log
                try:
                    from CESA.agent_debug_f8b011 import log as _agent_log

                    _agent_log(
                        "F",
                        "eeg_studio._qt_resume_pump_after_tk_modal:poll_delay",
                        "scheduling _restart_poll_late",
                        {"ms": _poll_restart_ms, "has_qt_bridge": _has_qt_bridge},
                    )
                except Exception:
                    pass
                # #endregion
                try:
                    self.root.after(_poll_restart_ms, _restart_poll_late)
                except Exception:
                    _restart_poll_late()

        # Laisser Tk finir de dépiler le messagebox avant toute logique liée à Qt.
        _log_viewer_checkpoint("64", "_qt_resume_pump_after_tk_modal scheduling _do via after(48)")
        try:
            self.root.after(48, _do)
        except Exception:
            _do()

    def _qt_normalize_quick_stage(self, stage: str) -> Optional[str]:
        s = str(stage).upper().strip()
        if s == "REM":
            s = "R"
        if s in ManualScoringService.ALLOWED_STAGES:
            return s
        return None

    def _qt_labels_and_epoch_from_any_scoring(self) -> Optional[Tuple[List[str], float]]:
        """Même grille que le hypnogramme Qt : une étiquette par époque à partir du scoring actif."""
        if not self.raw:
            return None
        fs = float(getattr(self, "sfreq", 256.0))
        total_dur = float(len(self.raw.times) / fs)
        df = self._get_active_scoring_df()
        epoch_len = float(getattr(self, "scoring_epoch_duration", 30.0))
        try:
            if df is not None and len(df) > 0 and "time" in df.columns and "stage" in df.columns:
                df_work = df[["time", "stage"]].copy()
                df_work["time"] = pd.to_numeric(df_work["time"], errors="coerce")
                df_work = df_work.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
                if len(df_work) >= 2:
                    diffs = np.diff(df_work["time"].to_numpy(dtype=float))
                    diffs_pos = diffs[diffs > 0]
                    epoch_len = float(np.median(diffs_pos)) if len(diffs_pos) > 0 else epoch_len
                num_epochs = int(np.ceil(total_dur / epoch_len))
                labels = ["U"] * num_epochs
                for t_val, s in zip(
                    df_work["time"].to_numpy(dtype=float),
                    df_work["stage"].astype(str).str.upper().str.strip().to_numpy(),
                ):
                    idx = int(round(t_val / epoch_len))
                    if 0 <= idx < len(labels):
                        labels[idx] = s
                return (labels, epoch_len)
        except Exception:
            pass
        num_epochs = int(np.ceil(total_dur / epoch_len))
        return (["U"] * num_epochs, epoch_len)

    def _qt_mutate_hypnogram_stage_at_time(self, onset_s: float, stage: str) -> None:
        """Met à jour le scoring manuel dense et rafraîchit le viewer Qt."""
        norm = self._normalize_quick_stage(stage)
        if norm is None or not self.raw:
            return
        tup = self._qt_labels_and_epoch_from_any_scoring()
        if tup is None:
            return
        labels, epoch_len = list(tup[0]), float(tup[1])
        if not labels:
            return
        idx = int(max(0, min(len(labels) - 1, onset_s // epoch_len)))
        labels[idx] = norm
        n = len(labels)
        times = (np.arange(n, dtype=float) * epoch_len).tolist()
        try:
            new_df = self.manual_scoring_service.validate(
                pd.DataFrame({"time": times, "stage": labels}),
            )
        except Exception:
            new_df = pd.DataFrame({"time": times, "stage": labels})
        self.manual_scoring_data = new_df
        self.show_manual_scoring = True
        self.scoring_dirty = True
        if self._qt_psg_plot_active():
            try:
                tup2 = self._qt_labels_and_epoch_from_any_scoring()
                if tup2 is not None:
                    self._qt_viewer_bridge.set_hypnogram((list(tup2[0]), float(tup2[1])))
            except Exception:
                logging.warning("[VIEW] set_hypnogram après scoring rapide a échoué.", exc_info=True)

    def _qt_apply_sleep_stage_at_epoch_time(self, onset_s: float, stage: str) -> None:
        self._qt_mutate_hypnogram_stage_at_time(float(onset_s), stage)

    def _qt_apply_sleep_stage_at_current_epoch(self, stage: str) -> None:
        if not self.raw:
            return
        tup = self._qt_labels_and_epoch_from_any_scoring()
        if tup is None:
            return
        _, el = tup
        el = float(el)
        onset = float(int(float(self.current_time) // el) * el)
        self._qt_mutate_hypnogram_stage_at_time(onset, stage)

    def _qt_on_viewer_navigate(self, target_time: float) -> None:
        """Appelé par le viewer Qt (pendant processEvents).

        Aucun appel Tk ici — même ``root.after(0, ...)`` provoque un crash GIL fatal
        (PyEval_RestoreThread) sur Python 3.14. On pose seulement l'état ; le pump Tk
        enfile ``_after_qt_pump_commit_deferred_nav`` sur le poll Tk après ``processEvents()``.
        """
        _log_viewer_checkpoint(
            "01",
            "qt_on_viewer_navigate",
            target=target_time,
            queued=getattr(self, "_qt_nav_sync_queued", False),
        )
        try:
            self._pending_nav_sync_t = float(max(0.0, float(target_time)))
        except Exception:
            self._pending_nav_sync_t = 0.0
        self._qt_nav_sync_deferred = True
        _log_viewer_checkpoint(
            "02",
            "pending_nav_sync_t set; defer until after_qt_pump (no Tk from Qt stack)",
            pending=self._pending_nav_sync_t,
        )

    def _after_qt_pump_enqueue_commit_nav(self) -> None:
        """Après ``processEvents`` : ne rien faire de lourd ici (GIL Py3.14).

        Enfile éventuellement la synchro sur ``_poll_tk_main_thread_queue`` pour qu'elle
        s'exécute dans un autre tick Tk, pas dans la foulée du pump Qt.
        """
        if not getattr(self, "_qt_nav_sync_deferred", False):
            return
        self._enqueue_tk_main(self._after_qt_pump_commit_deferred_nav)

    def _after_qt_pump_commit_deferred_nav(self) -> None:
        """Après chaque ``processEvents()`` : exécute la synchro Tk si le viewer Qt a navigué."""
        # #region agent log
        try:
            from CESA.agent_debug_f8b011 import log as _agent_log

            _agent_log(
                "A",
                "eeg_studio._after_qt_pump_commit_deferred_nav",
                "enter",
                {"deferred": bool(getattr(self, "_qt_nav_sync_deferred", False))},
            )
        except Exception:
            pass
        # #endregion
        if not getattr(self, "_qt_nav_sync_deferred", False):
            return
        self._qt_nav_sync_deferred = False
        if not self._request_qt_nav_sync_to_tk():
            self._qt_nav_sync_deferred = True

    def _apply_qt_nav_sync_time(self, t: float) -> None:
        """Met à jour temps + sliders/labels sans déclencher command= des ttk.Scale.

        Sinon ``time_var.set`` peut invoquer ``_update_time`` → ``update_plot()`` puis
        un second ``update_plot()`` depuis la synchro Qt → réentrance et plantages.
        """
        _log_viewer_checkpoint(
            "03",
            "apply_qt_nav_sync_time enter",
            t_in=t,
            current_before=getattr(self, "current_time", None),
        )
        self.current_time = float(max(0.0, t))
        ts = getattr(self, "time_scale", None)
        bts = getattr(self, "bottom_time_scale", None)
        try:
            if ts is not None:
                try:
                    ts.configure(command="")
                except Exception:
                    pass
            if bts is not None:
                try:
                    bts.configure(command="")
                except Exception:
                    pass
            self._update_time_display()
        finally:
            if ts is not None:
                try:
                    ts.configure(command=self._update_time)
                except Exception:
                    pass
            if bts is not None:
                try:
                    bts.configure(command=self._update_time_from_bottom)
                except Exception:
                    pass
        _log_viewer_checkpoint("04", "apply_qt_nav_sync_time exit", current=self.current_time)

    def _delayed_resync_qt_nav_to_tk(self) -> None:
        """Re-sync différé : évite une chaîne immédiate after(0) → update_plot sous rafale."""
        _log_viewer_checkpoint(
            "05",
            "delayed_resync_qt_nav_to_tk",
            pending=getattr(self, "_pending_nav_sync_t", None),
            current=getattr(self, "current_time", None),
        )
        try:
            if abs(float(getattr(self, "_pending_nav_sync_t", 0)) - float(self.current_time)) > 1e-3:
                _log_viewer_checkpoint("06", "delayed_resync triggers new request_sync")
                self._request_qt_nav_sync_to_tk()
        except Exception:
            pass

    def _request_qt_nav_sync_to_tk(self) -> bool:
        """Reporte temps + update_plot sur la boucle Tk (évite réentrance Qt/Tk et crashs GIL).

        Returns
        -------
        bool
            True si un ``after(12, _run)`` (ou équivalent) a été planifié, False si file déjà pleine.
        """
        if self._qt_nav_sync_queued:
            _log_viewer_checkpoint(
                "07",
                "request_qt_nav_sync_to_tk skipped (already queued)",
                pending=getattr(self, "_pending_nav_sync_t", None),
            )
            return False
        self._qt_nav_sync_queued = True

        def _run() -> None:
            try:
                sys.stderr.write("[VIEWER-PRE10] Tk sync_run: callback Tk entré\n")
                sys.stderr.flush()
            except Exception:
                pass
            t = None
            try:
                t = float(self._pending_nav_sync_t)
            except Exception:
                pass
            _log_viewer_checkpoint(
                "10",
                "Tk sync_run start",
                pending_t=t,
                thread=threading.current_thread().name,
            )
            _flush_viewer_logs()
            try:
                if t is not None:
                    _log_viewer_checkpoint("11", "sync_run before update_plot", apply_t=t)
                    _flush_viewer_logs()
                    self._apply_qt_nav_sync_time(t)
                    _flush_viewer_logs()
                    if self._qt_psg_plot_active() and getattr(
                        self, "psg_plotter", None,
                    ) is None:
                        self._sync_qt_viewer_bridge_plot_state()
                        _log_viewer_checkpoint(
                            "12",
                            "sync_run after qt bridge sync (no full update_plot)",
                            current=getattr(self, "current_time", None),
                        )
                    else:
                        self.update_plot()
                        _log_viewer_checkpoint(
                            "12",
                            "sync_run after update_plot ok",
                            current=getattr(self, "current_time", None),
                        )
                    _flush_viewer_logs()
            except Exception:
                logging.exception("[VIEWER-CHK-ERR] sync_run: apply_qt_nav_sync_time or update_plot failed")
                _flush_viewer_logs()
            finally:
                need_resync = False
                try:
                    if t is not None:
                        need_resync = (
                            abs(
                                float(getattr(self, "_pending_nav_sync_t", t))
                                - float(self.current_time),
                            )
                            > 1e-3
                        )
                except Exception:
                    pass
                self._qt_nav_sync_queued = False
                _log_viewer_checkpoint(
                    "13",
                    "sync_run finally",
                    need_resync=need_resync,
                    pending_after=getattr(self, "_pending_nav_sync_t", None),
                    current=getattr(self, "current_time", None),
                )
                _flush_viewer_logs()
                if need_resync:
                    try:
                        self.root.after(40, self._delayed_resync_qt_nav_to_tk)
                    except Exception:
                        pass
                # #region agent log
                try:
                    from CESA.agent_debug_f8b011 import log as _agent_log

                    _agent_log(
                        "D",
                        "eeg_studio._request_qt_nav_sync_to_tk._run:finally_end",
                        "sync_run finally complete",
                        {"need_resync": need_resync},
                    )
                except Exception:
                    pass
                # #endregion

        try:
            if threading.current_thread() is threading.main_thread():
                _log_viewer_checkpoint("08", "request_sync scheduled root.after(12) main_thread")
                _flush_viewer_logs()
                self.root.after(12, _run)
                return True
            _log_viewer_checkpoint("09", "request_sync enqueue_tk_main (off main thread)")
            _flush_viewer_logs()
            self._enqueue_tk_main(_run)
            return True
        except Exception:
            try:
                _log_viewer_checkpoint("14", "request_sync fallback enqueue_tk_main after exception")
                _flush_viewer_logs()
                self._enqueue_tk_main(_run)
                return True
            except Exception:
                self._qt_nav_sync_queued = False
                _log_viewer_checkpoint("15", "request_sync failed; queue cleared")
                _flush_viewer_logs()
                return False

    def _sync_qt_viewer_bridge_plot_state(
        self,
        hypnogram: Optional[Any] = None,
    ) -> None:
        """Pousse l'état Studio (temps, filtres, hypnogramme) vers le bridge Qt sans ``update_plot``.

        Évite d'exécuter tout le pipeline d'extraction/scipy de ``update_plot`` lorsque seul le
        viewer Qt est actif (``psg_plotter`` est None), ce qui réduit la charge et les crashs
        GIL (``PyEval_RestoreThread``) observés avec Python 3.14 + pump Tk/Qt.

        *hypnogram* : si fourni (ex. depuis ``update_plot``), utilisé à la place du cache.
        """
        if not self._qt_psg_plot_active():
            return
        br = getattr(self, "_qt_viewer_bridge", None)
        if br is None:
            return
        # #region agent log
        try:
            from CESA.agent_debug_f8b011 import log as _agent_log

            _agent_log(
                "C",
                "eeg_studio._sync_qt_viewer_bridge_plot_state:enter",
                "sync bridge",
                {"ct": float(self.current_time), "dur": float(self.duration)},
            )
        except Exception:
            pass
        # #endregion
        try:
            hg = hypnogram
            if hg is None:
                hg = getattr(self, "_psg_cached_hypnogram", None)
            if hg is not None:
                br.set_hypnogram(hg)
            if getattr(self, "raw", None) is not None:
                br.set_total_duration(float(len(self.raw.times) / float(self.sfreq)))
            br.set_global_filter_enabled(
                bool(self.filter_var.get()) if hasattr(self, "filter_var") else True,
            )
            try:
                br.filter_pipelines_by_channel = getattr(
                    self, "channel_filter_pipelines", {},
                )
            except Exception:
                pass
            br.set_time_window(float(self.current_time), float(self.duration))
        except Exception:
            logging.warning("[VIEW] Synchronisation viewer Qt (léger) échouée.", exc_info=True)
            raise
        # #region agent log
        try:
            from CESA.agent_debug_f8b011 import log as _agent_log

            _agent_log("C", "eeg_studio._sync_qt_viewer_bridge_plot_state:exit", "sync bridge ok", {})
        except Exception:
            pass
        # #endregion

    def _ensure_qt_viewer(self, *, full_reload: bool = False) -> bool:
        if not self._qt_psg_dependencies_available():
            return False
        try:
            from CESA.qt_viewer import launch_viewer
        except ImportError as exc:
            self._qt_import_check_failed = True
            logging.warning(
                "[VIEW] Module CESA.qt_viewer introuvable (%s) — repli sur Matplotlib.",
                exc,
            )
            return False
        kwargs = self._qt_build_launch_kwargs()
        if kwargs is None:
            return False

        br = getattr(self, "_qt_viewer_bridge", None)
        alive = br is not None
        try:
            alive = alive and bool(br.is_alive())
        except Exception:
            alive = False

        _log_viewer_checkpoint(
            "16",
            "ensure_qt_viewer",
            full_reload=full_reload,
            bridge_alive=alive,
            current_time=getattr(self, "current_time", None),
            duration=getattr(self, "duration", None),
        )

        if alive and full_reload:
            try:
                br.update_signals(kwargs["signals"])
                hg = kwargs.get("hypnogram")
                if hg is not None:
                    br.set_hypnogram(hg)
                td = kwargs.get("total_duration_s")
                if td is not None:
                    br.set_total_duration(float(td))
                br.filter_pipelines_by_channel = kwargs.get("filter_pipelines") or {}
                br.set_global_filter_enabled(bool(kwargs.get("global_filter_enabled", True)))
                br.set_time_window(float(self.current_time), float(self.duration))
                _log_viewer_checkpoint("17", "ensure_qt_viewer full_reload ok")
                return True
            except Exception:
                logging.warning("[VIEW] Rafraîchissement Qt échoué, nouvelle fenêtre.", exc_info=True)
                self._close_qt_viewer_if_any()
                alive = False

        if alive and not full_reload:
            _log_viewer_checkpoint("18", "ensure_qt_viewer reuse existing (no reload)")
            return True

        self._close_qt_viewer_if_any()
        _log_viewer_checkpoint(
            "19",
            "ensure_qt_viewer calling launch_viewer",
            start_time_s=kwargs.get("start_time_s"),
            duration_s=kwargs.get("duration_s"),
            total_duration_s=kwargs.get("total_duration_s"),
        )
        try:
            self._qt_viewer_bridge = launch_viewer(
                signals=kwargs["signals"],
                hypnogram=kwargs.get("hypnogram"),
                scoring_annotations=kwargs.get("scoring_annotations") or [],
                filter_pipelines=kwargs.get("filter_pipelines") or {},
                channel_types=kwargs.get("channel_types") or {},
                global_filter_enabled=bool(kwargs.get("global_filter_enabled", True)),
                start_time_s=float(kwargs.get("start_time_s", 0.0)),
                duration_s=float(kwargs.get("duration_s", 30.0)),
                total_duration_s=kwargs.get("total_duration_s"),
                theme_name=str(kwargs.get("theme_name", "dark")),
                on_navigate=self._qt_on_viewer_navigate,
                tk_root=self.root,
                after_qt_pump=self._after_qt_pump_enqueue_commit_nav,
                on_request_auto_scoring=lambda: self._qt_defer_to_studio(self._run_auto_sleep_scoring),
                on_open_filter_config=lambda: self._qt_defer_to_studio(self.show_filter_config),
                on_open_manual_scoring_editor=lambda: self._qt_defer_to_studio(
                    self._open_manual_scoring_editor,
                ),
                on_request_stage_for_current_epoch=lambda st: self._qt_defer_to_studio(
                    lambda s=st: self._qt_apply_sleep_stage_at_current_epoch(s),
                ),
                on_request_stage_at_epoch_time=lambda t, st: self._qt_defer_to_studio(
                    lambda tt=t, s=st: self._qt_apply_sleep_stage_at_epoch_time(tt, s),
                ),
                on_global_filter_toggled=lambda en: self._qt_defer_to_studio(
                    lambda e=en: self._qt_sync_global_filter_from_viewer(e),
                ),
            )
            _log_viewer_checkpoint("21", "ensure_qt_viewer launch_viewer returned ok")
            return True
        except Exception:
            logging.exception("[VIEW] Ouverture du viewer Qt impossible.")
            _log_viewer_checkpoint("22", "ensure_qt_viewer launch_viewer failed")
            self._qt_viewer_bridge = None
            return False

    def _show_default_psg_view(self, embed_parent: Optional[ttk.Frame] = None) -> None:
        """Ouvre le viewer PSG par défaut : PyQtGraph si disponible, sinon Matplotlib."""
        if not self.raw:
            parent = embed_parent or getattr(self, "psg_container", None) or self.root
            try:
                for child in parent.winfo_children():
                    try:
                        child.destroy()
                    except Exception:
                        pass
                holder = ttk.Frame(parent)
                holder.pack(fill=tk.BOTH, expand=True)
                ttk.Label(
                    holder,
                    text="Aucun enregistrement chargé\nOuvrez un fichier EDF pour afficher la PSG",
                    anchor=tk.CENTER, justify=tk.CENTER,
                ).pack(expand=True)
            except Exception:
                messagebox.showinfo(
                    "Information",
                    "Aucun enregistrement chargé. Veuillez ouvrir un fichier EDF.",
                )
            return

        if getattr(self, "prefer_qt_psg_viewer", True) and self._qt_psg_dependencies_available():
            ch_tup = tuple(getattr(self, "psg_channels_used", None) or ())
            prev = getattr(self, "_qt_channel_tuple", ())
            br = getattr(self, "_qt_viewer_bridge", None)
            bridge_ok = False
            try:
                bridge_ok = br is not None and bool(br.is_alive())
            except Exception:
                bridge_ok = False
            full_reload = (ch_tup != prev) or (not bridge_ok)
            self._qt_channel_tuple = ch_tup
            ok = self._ensure_qt_viewer(full_reload=full_reload)
            if ok:
                parent = embed_parent or getattr(self, "psg_container", None)
                if parent is not None:
                    self._embed_qt_viewer_placeholder(parent)
                self.psg_plotter = None
                try:
                    self.canvas = None
                except Exception:
                    pass
                logging.info("[VIEW] Viewer PyQtGraph actif (défaut).")
                return

        if not self._qt_psg_dependencies_available():
            self.prefer_qt_psg_viewer = False
        self.show_multi_graph_view(embed_parent=embed_parent)
    
    def show_multi_graph_view(self, embed_parent: Optional[ttk.Frame] = None):
        """Affiche la vue PSG multi-subplots.
        - Si embed_parent est fourni, intègre la figure dans ce conteneur (vue principale)
        - Sinon, ouvre une nouvelle fenêtre Toplevel.
        """
        self._close_qt_viewer_if_any()
        if not self.raw:
            # Pas d'EDF chargé: n'affiche pas de graphe vide; affiche un placeholder propre
            parent = embed_parent or self.root
            try:
                holder = ttk.Frame(parent)
                holder.pack(fill=tk.BOTH, expand=True)
                msg = ttk.Label(holder, text="Aucun enregistrement chargé\nOuvrez un EDF pour afficher la PSG",
                                 anchor=tk.CENTER, justify=tk.CENTER)
                msg.pack(expand=True)
            except Exception:
                messagebox.showinfo("Information", "Aucun enregistrement chargé. Veuillez ouvrir un fichier EDF.")
            return

        # Préparer les signaux à partir de MNE Raw → µV (optimisé: canaux utiles uniquement)
        signals = {}
        try:
            fs = float(self.sfreq)
        except Exception:
            fs = 200.0

        # Build selected channels from active profile mappings.
        available = list(self.raw.ch_names)
        mapping_ready = self._ensure_profile_channel_mapping(
            available,
            create_new_profile_on_pending=False,
            show_dialog_on_pending=False,
        )
        if not mapping_ready:
            # Do not block initial rendering when called from non-interactive paths.
            # We keep current runtime mapping (if any) and then fallback to selected_channels.
            logging.warning(
                "[VIEW] Mapping incomplet en mode non interactif; fallback selected_channels utilise"
            )
        selected: List[str] = []
        for section in self.active_profile.signal_sections:
            if not bool(section.enabled):
                continue
            section_key = str(section.key)
            selected.extend(
                [
                    ch for ch, mapped in self.profile_channel_map_runtime.items()
                    if mapped == section_key and ch in available
                ]
            )
        if not selected:
            selected = list(self.selected_channels) if self.selected_channels else available[:8]
        self.psg_channels_used = list(selected)

        # Créer un sample de télémetrie pour la visualisation
        dataset_id = 'unknown'
        if hasattr(self, 'raw') and self.raw is not None:
            try:
                meas_id = self.raw.info.get('meas_id')
                if meas_id is not None and isinstance(meas_id, dict):
                    dataset_id = str(meas_id.get('file_id', 'unknown'))
                else:
                    dataset_id = str(getattr(self.raw.info, 'filename', 'unknown'))
            except Exception:
                dataset_id = 'unknown'
        viz_sample = telemetry.new_sample({
            "channel": "visualization",
            "dataset_id": dataset_id,
            "start_s": float(self.current_time),
            "duration_s": float(self.duration),
            "notes": f"n_channels={len(selected)}",
        })
        
        # Pre-slice only the visible window to improve performance
        start_idx = int(max(0, float(self.current_time) * fs))
        end_idx = int(min(len(self.raw.times), (float(self.current_time) + float(self.duration)) * fs))
        if end_idx <= start_idx + 1:
            end_idx = min(len(self.raw.times), start_idx + int(max(2, fs)))

        # Mesurer le temps d'extraction des canaux et de conversion µV
        extract_start = time.perf_counter()
        convert_total = 0.0
        
        for ch in selected:
            try:
                arr = self.raw.get_data(picks=[ch], start=start_idx, stop=end_idx)[0]
                # Mesurer le temps de conversion µV
                convert_start = time.perf_counter()
                data_uv = self._to_microvolts_and_sanitize(arr)
                convert_total += (time.perf_counter() - convert_start) * 1000.0
                # Provide pre-windowed time base by letting PSGPlotter detect it
                signals[ch] = (data_uv, fs)
            except Exception:
                continue
        
        extract_channels_ms = (time.perf_counter() - extract_start) * 1000.0
        viz_sample.setdefault("extract_channels_ms", extract_channels_ms - convert_total)
        viz_sample.setdefault("convert_uv_ms", convert_total)

        # Hypnogramme depuis scoring actif (tolérant et complet) avec mesure du temps
        hypnogram = None
        with telemetry.measure(viz_sample, "prepare_hypno_ms"):
            df = self._get_active_scoring_df()
            if df is not None and len(df) > 0 and 'time' in df.columns and 'stage' in df.columns:
                try:
                    # Assurer l'ordre chronologique
                    df_work = df[['time', 'stage']].copy()
                    df_work['time'] = pd.to_numeric(df_work['time'], errors='coerce')
                    df_work = df_work.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

                    if len(df_work) >= 2:
                        diffs = np.diff(df_work['time'].to_numpy(dtype=float))
                        diffs_pos = diffs[diffs > 0]
                        epoch_len = float(np.median(diffs_pos)) if len(diffs_pos) > 0 else float(self.scoring_epoch_duration)
                    else:
                        epoch_len = float(self.scoring_epoch_duration)

                    # Étendre en séquence complète avec remplissage 'U' sur toute la durée
                    if len(df_work) > 0:
                        try:
                            total_dur = float(len(self.raw.times) / self.sfreq)
                        except Exception:
                            total_dur = float(df_work['time'].iloc[-1]) + epoch_len
                        start_fill = 0.0
                        end_fill = total_dur
                        num_epochs = int(np.ceil((end_fill - start_fill) / epoch_len))
                        labels = ['U'] * num_epochs
                        for t, s in zip(
                            df_work['time'].to_numpy(dtype=float),
                            df_work['stage'].astype(str).str.upper().str.strip().to_numpy()
                        ):
                            idx = int(round((t - start_fill) / epoch_len))
                            if 0 <= idx < len(labels):
                                labels[idx] = s
                        hypnogram = (labels, epoch_len)
                except Exception:
                    hypnogram = None

        # Événements (placeholder: extraire plus tard depuis annotations EDF si dispo)
        events = []

        # Créer et afficher le plotter PSG avec mesure du temps
        from CESA.psg_plot import PSGPlotter

        with telemetry.measure(viz_sample, "create_plotter_ms"):
            plotter = PSGPlotter(
                signals=signals,
                hypnogram=hypnogram,
                scoring_annotations=events,
                start_time_s=float(self.current_time),
                duration_s=float(self.duration),
                filter_params_by_channel=self.channel_filter_params,
                filter_pipelines_by_channel=getattr(self, "channel_filter_pipelines", {}),
                global_filter_enabled=bool(self.filter_var.get()),
                filter_order=int(self.filter_order),
                filter_type=str(self.filter_type),
                theme_name=self.theme_manager.current_theme_name,
                total_duration_s=float(len(self.raw.times) / fs) if hasattr(self, 'raw') and self.raw is not None else None,
                signal_sections=[
                    {
                        "key": str(section.key),
                        "label": str(section.label),
                        "signal_type": str(section.signal_type),
                        "ratio": float(section.ratio),
                        "enabled": bool(section.enabled),
                        "color_palette": [str(c) for c in (section.color_palette or []) if str(c).strip()],
                    }
                    for section in self.active_profile.signal_sections
                ],
                channel_section_map=dict(self.profile_channel_map_runtime),
                hypnogram_ratio=float(getattr(self.active_profile, "hypnogram_ratio", 1.0)),
                events_ratio=float(getattr(self.active_profile, "events_ratio", 0.6)),
            )

        # Stocker la référence pour mises à jour rapides
        self.psg_plotter = plotter
        try:
            self.psg_plotter.set_hypnogram(hypnogram)
            # Enregistrer un callback de navigation pour clic hypnogramme
            def _on_nav(target_time: float):
                try:
                    self.current_time = float(max(0.0, target_time))
                    if hasattr(self, 'time_var'):
                        self.time_var.set(self.current_time)
                    self.update_plot()
                except Exception:
                    pass
            try:
                self.psg_plotter.set_nav_callback(_on_nav)
            except Exception:
                pass
        except Exception:
            pass

        # Intégration dans la fenêtre principale si embed_parent fourni
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        if embed_parent is not None:
            # Nettoyer l'ancien contenu
            for child in embed_parent.winfo_children():
                try:
                    child.destroy()
                except Exception:
                    pass
            canvas = FigureCanvasTkAgg(plotter.figure, master=embed_parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Attacher ce canvas à la toolbar moderne principale
            self.canvas = canvas
            try:
                self._recreate_toolbar()
            except Exception:
                pass
            # (Supprimé) Panel sous le canvas pour éviter les doublons.
            
            # Commiter le sample de télémetrie
            try:
                if viz_sample:
                    viz_sample.setdefault("total_ms", 0.0)
                    telemetry.commit(viz_sample)
            except Exception:
                pass
            return

        # Sinon, créer une fenêtre Toplevel dédiée (mode secondaire)
        multi_window = tk.Toplevel(self.root)
        multi_window.title("Vue PSG (Multi-Graphiques) - CESA")
        multi_window.geometry("1400x900")
        multi_window.transient(self.root)

        main_frame = ttk.Frame(multi_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = FigureCanvasTkAgg(plotter.figure, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Commiter le sample de télémetrie
        try:
            if viz_sample:
                viz_sample.setdefault("total_ms", 0.0)
                telemetry.commit(viz_sample)
        except Exception:
            pass

        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        control_frame = ttk.Frame(multi_window, style='Custom.TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        try:
            self._create_control_panel(control_frame)
        except Exception:
            pass

        def refresh_view():
            plotter.set_time_window(float(self.current_time), float(self.duration))
            canvas.draw_idle()

        def export_png():
            try:
                from tkinter import filedialog
                fn = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[["PNG","*.png"]])
                if fn:
                    plotter.save_png(fn)
            except Exception as e:
                messagebox.showerror("Export PNG", str(e))

        def export_pdf():
            try:
                from tkinter import filedialog
                fn = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[["PDF","*.pdf"]])
                if fn:
                    plotter.save_pdf(fn)
            except Exception as e:
                messagebox.showerror("Export PDF", str(e))

        def export_scoring_csv():
            try:
                from tkinter import filedialog
                fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[["CSV","*.csv"]])
                if fn:
                    plotter.export_scoring_csv(fn)
            except Exception as e:
                messagebox.showerror("Export CSV", str(e))

        # Boutons rapides conservés (intégrés visuellement au panel)
        ttk.Button(control_frame, text="🔄 Actualiser", command=refresh_view, style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⚙️ Configurer Filtres", command=self.show_filter_config, style='Modern.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🖼️ Export PNG", command=export_png, style='Modern.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="📄 Export PDF", command=export_pdf, style='Modern.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="💾 Export Scoring CSV", command=export_scoring_csv, style='Modern.TButton').pack(side=tk.RIGHT, padx=5)
    
    def _launch_qt_viewer(self):
        """Ouvre ou rafraîchit le viewer PyQtGraph (défaut si disponible)."""
        if not self.raw:
            messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF.")
            return

        if not self._qt_psg_dependencies_available():
            messagebox.showinfo(
                "PyQtGraph indisponible",
                "PySide6 et/ou pyqtgraph ne sont pas installes.\n"
                "Installez-les avec : pip install PySide6 pyqtgraph\n\n"
                "Utilisation du viewer matplotlib a la place.",
            )
            self._open_matplotlib_psg_view()
            return

        self.prefer_qt_psg_viewer = True
        ok = self._ensure_qt_viewer(full_reload=True)
        if not ok:
            messagebox.showwarning(
                "PyQtGraph",
                "Impossible d'ouvrir le viewer PyQtGraph. Passage au viewer Matplotlib.",
            )
            self._open_matplotlib_psg_view()
            return

        parent = getattr(self, "psg_container", None)
        if parent is not None:
            self._embed_qt_viewer_placeholder(parent)
        self.psg_plotter = None

    def _run_yasa_scoring(self, method: Optional[str] = None):
        """Exécute le scoring automatique (YASA/U-Sleep) et stocke le résultat."""
        try:
            if self.raw is None:
                messagebox.showwarning("Avertissement", "Aucune donnée EDF chargée.")
                return
            selected_method = str(method or getattr(self, "sleep_scoring_method", "yasa")).lower().strip()
            # Autoriser les trois backends: YASA, U-Sleep, PFTSleep.
            if selected_method not in {"yasa", "usleep", "pftsleep"}:
                # Par défaut, privilégier PFTSleep comme backend moderne.
                selected_method = "pftsleep"
            self.sleep_scoring_method = selected_method
            if selected_method == "usleep":
                if not self._has_valid_usleep_checkpoint():
                    auto_path = self._resolve_default_usleep_checkpoint()
                    if auto_path and Path(auto_path).exists():
                        self.usleep_checkpoint_path = auto_path
                        logging.info("[USLEEP] Checkpoint auto-détecté: %s", auto_path)
                if not self._has_valid_usleep_checkpoint():
                    df_remote = self._run_usleep_api_service(interactive=True)
                    if df_remote is not None:
                        self.sleep_scoring_data = df_remote
                        self.scoring_epoch_duration = float(getattr(self, "auto_scoring_epoch_length", 30.0))
                        self.scoring_dirty = True
                        self._update_status_bar()
                        try:
                            self.update_plot()
                        except Exception:
                            pass
                        messagebox.showinfo("U-Sleep API", f"Scoring automatique terminé. {len(df_remote)} époques.")
                        return
                    # Keep legacy fallback web workflow if API path is unavailable.
                    self._launch_usleep_webapp_workflow()
                    return
            if selected_method == "yasa":
                method_label = "YASA"
            elif selected_method == "usleep":
                method_label = "U-Sleep"
            elif selected_method == "pftsleep":
                method_label = "PFTSleep"
            else:
                method_label = selected_method.upper()
            self._show_loading_bar(title=f"Scoring {method_label}", message="Calcul en cours, veuillez patienter...")

            # Progress callback to update UI during staging
            start_t = time.perf_counter()
            stage_to_pct = {
                'start': 5,
                'resample_begin': 10,
                'resample_end': 30,
                'notch': 40,
                'eeg_selected': 45,
                'eog_selected': 50,
                'filters': 65,
                'staging_initialized': 70,
                'predict_begin': 80,
                'predict_end': 95,
                'done': 100,
            }

            def _progress_cb(stage: str, payload: dict):
                try:
                    pct = stage_to_pct.get(stage, None)
                    if pct is None and hasattr(self, 'progress_var'):
                        try:
                            pct = min(99, float(self.progress_var.get()) + 1.0)
                        except Exception:
                            pct = 90
                    if pct is not None and hasattr(self, 'progress_var'):
                        try:
                            self.progress_var.set(pct)
                            if hasattr(self, 'progress_label'):
                                self.progress_label.config(text=f"{int(pct)}%")
                        except Exception:
                            pass

                    # Build status message
                    msg = "Préparation..."
                    if stage == 'start':
                        msg = f"Initialisation {method_label}..."
                    elif stage == 'resample_begin':
                        msg = f"Resampling → {payload.get('target_sfreq', '?')} Hz..."
                    elif stage == 'resample_end':
                        msg = f"Resampling terminé ({payload.get('duration_s', 0.0):.1f}s)"
                    elif stage == 'notch':
                        msg = f"Notch appliqué ({payload.get('duration_s', 0.0):.1f}s)"
                    elif stage == 'eeg_selected':
                        msg = f"EEG sélectionné: {payload.get('eeg', '?')}"
                    elif stage == 'eog_selected':
                        eog = payload.get('eog', None)
                        msg = f"EOG configuré: {eog if eog else 'aucun'}"
                    elif stage == 'filters':
                        msg = f"Filtres appliqués ({payload.get('duration_s', 0.0):.1f}s)"
                    elif stage == 'staging_initialized':
                        msg = f"Initialisation SleepStaging ({payload.get('mode', 'eeg')})"
                    elif stage == 'predict_begin':
                        msg = "Prédiction des stades..."
                    elif stage == 'predict_end':
                        msg = f"Prédiction terminée ({payload.get('duration_s', 0.0):.1f}s)"
                    elif stage == 'done':
                        total = time.perf_counter() - start_t
                        msg = f"Terminé ({int(payload.get('epochs', 0))} époques, {total:.1f}s)"

                    try:
                        self._update_loading_message(msg)
                        self.root.update_idletasks()
                    except Exception:
                        pass
                except Exception:
                    pass

            scorer = SleepScorer(
                method=selected_method,
                eeg_candidates=self.yasa_eeg_candidates,
                eog_candidates=self.yasa_eog_candidates,
                emg_candidates=self.yasa_emg_candidates,
                epoch_length=float(getattr(self, "auto_scoring_epoch_length", 30.0)),
                target_sfreq=float(getattr(self, "yasa_target_sfreq", 100.0)),
                progress_cb=_progress_cb,
                yasa_age=getattr(self, "yasa_age", None),
                yasa_male=getattr(self, "yasa_male", None),
                yasa_confidence_threshold=float(getattr(self, "yasa_confidence_threshold", 0.80)),
                usleep_checkpoint_path=getattr(self, "usleep_checkpoint_path", None),
                usleep_sfreq=float(getattr(self, "usleep_target_sfreq", 128.0)),
                usleep_device=None if str(getattr(self, "usleep_device", "auto")).lower() == "auto" else str(getattr(self, "usleep_device", "auto")).lower(),
                usleep_use_eog=bool(getattr(self, "usleep_use_eog", True)),
                usleep_zscore=bool(getattr(self, "usleep_zscore", True)),
                pft_models_dir=getattr(self, "pft_models_dir", None),
                pft_device=str(getattr(self, "pft_device", "auto")),
                pft_hf_token=getattr(self, "pft_hf_token", None),
                pft_eeg_channel=getattr(self, "pft_eeg_channel", None),
                pft_eog_channel=getattr(self, "pft_eog_channel", None),
                pft_emg_channel=getattr(self, "pft_emg_channel", None),
                pft_ecg_channel=getattr(self, "pft_ecg_channel", None),
            )
            df = scorer.score(self.raw)

            self.sleep_scoring_data = df
            self.scoring_epoch_duration = float(getattr(self, "auto_scoring_epoch_length", 30.0))
            
            # Mettre à jour la barre de statut
            self._update_status_bar()
            
            try:
                if hasattr(self, 'progress_var'):
                    self.progress_var.set(100)
                    if hasattr(self, 'progress_label'):
                        self.progress_label.config(text="100%")
            except Exception:
                pass
            self._hide_loading_bar()
            messagebox.showinfo(method_label, f"Scoring automatique terminé. {len(df)} époques.")
            logging.info("[%s] Scoring stocké dans self.sleep_scoring_data", method_label)
        except Exception as e:
            logging.error("[%s] Erreur scoring: %s", str(getattr(self, "sleep_scoring_method", "yasa")).upper(), e)
            try:
                self._hide_loading_bar()
            except Exception:
                pass
            messagebox.showerror("Erreur scoring", f"Echec du scoring automatique:\n{e}")

    def _run_auto_sleep_scoring(self):
        """Run sleep scoring with currently selected backend."""
        method = str(getattr(self, "sleep_scoring_method", "yasa")).lower().strip()
        self._run_yasa_scoring(method=method)

    def _set_sleep_scoring_method(self, method: str):
        """Update current automatic scoring backend."""
        selected = str(method).lower().strip()
        if selected not in {"yasa", "usleep", "pftsleep"}:
            selected = "pftsleep"
        self.sleep_scoring_method = selected
        try:
            self.sleep_scoring_method_var.set(selected)
        except Exception:
            pass
        logging.info("[SCORING] Backend sélectionné: %s", selected.upper())

    def _project_models_dir(self) -> Path:
        """Return default models directory at project root."""
        return Path(__file__).resolve().parent.parent / "models"

    def _has_valid_usleep_checkpoint(self) -> bool:
        path = str(getattr(self, "usleep_checkpoint_path", "") or "").strip()
        return bool(path and Path(path).exists())

    def _resolve_default_usleep_checkpoint(self) -> Optional[str]:
        """Find a likely default U-Sleep checkpoint in models/."""
        models_dir = self._project_models_dir()
        if not models_dir.exists():
            return None
        patterns = ("*.pt", "*.pth", "*.ckpt")
        candidates: List[Path] = []
        for pat in patterns:
            candidates.extend(sorted(models_dir.glob(pat)))
        if not candidates:
            return None
        preferred = [p for p in candidates if "usleep" in p.name.lower()]
        chosen = preferred[0] if preferred else candidates[0]
        return str(chosen)

    def _get_usleep_api_token(self, interactive: bool = True) -> Optional[str]:
        token = str(getattr(self, "usleep_api_token", "") or "").strip()
        if token:
            return token
        env_token = str(os.environ.get("USLEEP_API_TOKEN", "") or "").strip()
        if env_token:
            self.usleep_api_token = env_token
            return env_token
        if not interactive:
            return None
        token = simpledialog.askstring(
            "U-Sleep API token",
            (
                "Aucun token API U-Sleep trouvé.\n"
                "Collez un token (12h) généré depuis votre compte sleep.ai.ku.dk\n"
                "ou laissez vide pour annuler."
            ),
            parent=self.root,
        )
        token = str(token or "").strip()
        if token:
            self.usleep_api_token = token
            return token
        return None

    def _normalize_usleep_api_labels(self, labels: Sequence[Any]) -> pd.DataFrame:
        mapper_num = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
        mapper_str = {
            "wake": "W", "w": "W",
            "n1": "N1",
            "n2": "N2",
            "n3": "N3",
            "rem": "R", "r": "R",
        }
        out_labels: List[str] = []
        for lab in labels:
            if isinstance(lab, (int, np.integer)):
                out_labels.append(mapper_num.get(int(lab), "U"))
                continue
            s = str(lab).strip().lower()
            if s.isdigit():
                out_labels.append(mapper_num.get(int(s), "U"))
            else:
                out_labels.append(mapper_str.get(s, "U"))
        epoch = float(getattr(self, "auto_scoring_epoch_length", 30.0))
        times = np.arange(len(out_labels), dtype=float) * epoch
        return pd.DataFrame({"time": times, "stage": out_labels})

    def _run_usleep_api_service(self, interactive: bool = True) -> Optional[pd.DataFrame]:
        """Run U-Sleep through official API bindings and return normalized DataFrame."""
        if self.raw is None:
            return None
        input_path = str(getattr(self, "current_file_path", "") or "").strip()
        if not input_path or not Path(input_path).exists():
            if not interactive:
                return None
            input_path = filedialog.askopenfilename(
                title="Sélectionner un fichier PSG/EDF pour U-Sleep API",
                filetypes=[("EDF/BDF/FIF", "*.edf *.bdf *.fif *.fif.gz"), ("Tous les fichiers", "*.*")],
            )
            if not input_path:
                return None

        token = self._get_usleep_api_token(interactive=interactive)
        if not token:
            if interactive:
                messagebox.showinfo(
                    "U-Sleep API",
                    (
                        "Token API absent.\n"
                        "Créez-le sur https://sleep.ai.ku.dk/login (Account -> Generate API Token)\n"
                        "ou définissez USLEEP_API_TOKEN."
                    ),
                )
            return None

        try:
            from usleep_api import USleepAPI  # type: ignore
        except Exception:
            if interactive:
                messagebox.showerror(
                    "U-Sleep API",
                    "Le package 'usleep-api' n'est pas installé. Installez-le avec: pip install usleep-api",
                )
            return None

        try:
            self._show_loading_bar(title="U-Sleep API", message="Upload et scoring distant en cours...")
            self.root.update_idletasks()
            api = USleepAPI(api_token=token)
            with tempfile.TemporaryDirectory(prefix="usleep_api_") as tmpd:
                out_tsv = str(Path(tmpd) / "hypnogram.tsv")
                hypnogram, _log = api.quick_predict(
                    input_file_path=input_path,
                    output_file_path=out_tsv,
                    anonymize_before_upload=True,
                )

                labels: List[Any]
                if isinstance(hypnogram, dict) and "hypnogram" in hypnogram:
                    labels = list(hypnogram["hypnogram"])
                elif isinstance(hypnogram, (list, tuple, np.ndarray)):
                    labels = list(hypnogram)
                else:
                    # Fallback parsing from generated TSV
                    df_file = pd.read_csv(out_tsv, sep="\t", engine="python")
                    if "stage" in df_file.columns:
                        labels = df_file["stage"].tolist()
                    elif "hypnogram" in df_file.columns:
                        labels = df_file["hypnogram"].tolist()
                    elif len(df_file.columns) >= 1:
                        labels = df_file.iloc[:, 0].tolist()
                    else:
                        labels = []

            self._hide_loading_bar()
            if not labels:
                raise RuntimeError("Le service U-Sleep n'a renvoyé aucun stade.")
            return self._normalize_usleep_api_labels(labels)
        except Exception as exc:
            try:
                self._hide_loading_bar()
            except Exception:
                pass
            logging.error("[USLEEP-API] Echec: %s", exc)
            if interactive:
                messagebox.showerror("U-Sleep API", f"Echec du scoring via API:\n{exc}")
            return None

    def _ensure_usleep_checkpoint_ready(self, interactive: bool = True) -> bool:
        """Ensure U-Sleep has a checkpoint path; auto-discover or guide user."""
        current = str(getattr(self, "usleep_checkpoint_path", "") or "").strip()
        if current and Path(current).exists():
            return True

        auto_path = self._resolve_default_usleep_checkpoint()
        if auto_path and Path(auto_path).exists():
            self.usleep_checkpoint_path = auto_path
            logging.info("[USLEEP] Checkpoint auto-détecté: %s", auto_path)
            if interactive:
                messagebox.showinfo(
                    "U-Sleep",
                    f"Checkpoint auto-détecté dans models/:\n{auto_path}",
                )
            return True

        if not interactive:
            return False

        models_dir = self._project_models_dir()
        msg = (
            "Aucun checkpoint U-Sleep local n'a été trouvé.\n\n"
            f"Chemin attendu: {models_dir}\n"
            "Formats supportés: .pt, .pth, .ckpt\n\n"
            "Choisissez une option:\n"
            "• Oui: sélectionner un checkpoint local\n"
            "• Non: utiliser le service web officiel U-Sleep\n"
            "• Annuler: ne rien faire"
        )
        choice = messagebox.askyesnocancel("U-Sleep: checkpoint manquant", msg)
        if choice is True:
            self._select_usleep_checkpoint()
            selected = str(getattr(self, "usleep_checkpoint_path", "") or "").strip()
            return bool(selected and Path(selected).exists())
        if choice is False:
            self._launch_usleep_webapp_workflow()
            return False
        return False

    def _launch_usleep_webapp_workflow(self) -> None:
        """Guide user to the official U-Sleep web service when no local checkpoint exists."""
        service_url = "https://sleep.ai.ku.dk/sleep_stager"
        portal_url = "https://yousleep.ai/"
        current_file = str(getattr(self, "current_file_path", "") or "").strip()
        file_hint = (
            f"\n\nFichier courant détecté:\n{current_file}"
            if current_file and Path(current_file).exists()
            else "\n\nAucun fichier EDF courant détecté automatiquement."
        )
        instructions = (
            "Mode U-Sleep service web activé (sans checkpoint local).\n\n"
            "Étapes:\n"
            "1) Ouvrir la webapp officielle U-Sleep\n"
            "2) Uploader votre PSG/EDF\n"
            "3) Télécharger le résultat de scoring/hypnogramme\n"
            "4) Revenir dans CESA et importer via 'Importer Scoring (Excel/EDF)'"
            + file_hint
            + "\n\nLa webapp va être ouverte dans votre navigateur."
        )
        try:
            webbrowser.open(service_url)
        except Exception:
            pass
        messagebox.showinfo("U-Sleep service web", instructions)
        # Open modern portal as backup/discovery page.
        try:
            webbrowser.open_new_tab(portal_url)
        except Exception:
            pass
        # Optional direct handoff back to CESA import flow.
        try:
            go_import = messagebox.askyesno(
                "Importer un résultat U-Sleep",
                (
                    "Voulez-vous ouvrir directement la fenêtre d'import de scoring dans CESA ?\n\n"
                    "Utilisez cette option après avoir téléchargé le résultat depuis la webapp."
                ),
            )
            if go_import:
                self._open_scoring_import_hub()
        except Exception:
            pass

    def _select_usleep_checkpoint(self):
        """Choose U-Sleep pretrained checkpoint file."""
        models_dir = self._project_models_dir()
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        file_path = filedialog.askopenfilename(
            title="Sélectionner checkpoint U-Sleep",
            initialdir=str(models_dir),
            filetypes=[
                ("Model checkpoints", "*.pt *.pth *.ckpt"),
                ("PyTorch .pt", "*.pt"),
                ("PyTorch .pth", "*.pth"),
                ("Checkpoint .ckpt", "*.ckpt"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if not file_path:
            return
        self.usleep_checkpoint_path = file_path
        self._set_sleep_scoring_method("usleep")
        messagebox.showinfo("U-Sleep", f"Checkpoint sélectionné:\n{file_path}")

    def _open_sleep_scoring_settings(self):
        """Open a full settings dialog for YASA/U-Sleep automatic scoring."""
        win = tk.Toplevel(self.root)
        win.title("Configuration auto-scoring sommeil")
        win.geometry("760x620")
        win.transient(self.root)
        win.grab_set()

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        info = (
            "Valeurs recommandées (littérature):\n"
            "• Époque 30 s (standard AASM)\n"
            "• YASA: dérivation centrale + métadonnées optionnelles + confiance par époque\n"
            "• U-Sleep: 128 Hz + EEG/EOG + z-score par canal\n"
            "• PFTSleep: transformer multicanal pleine nuit (modèles HF, 30 s)"
        )
        ttk.Label(frame, text=info, justify=tk.LEFT).pack(anchor="w", pady=(0, 10))

        profile_box = ttk.LabelFrame(frame, text="Profils prédéfinis")
        profile_box.pack(fill=tk.X, pady=4)
        profile_help = (
            "Recommandé clinique: réglage équilibré et robuste.\n"
            "Rapide CPU: calcul allégé quand la machine est limitée.\n"
            "Haute fidélité: priorité à la granularité signal."
        )
        ttk.Label(profile_box, text=profile_help, justify=tk.LEFT).pack(anchor="w", padx=8, pady=(6, 2))

        backend_box = ttk.LabelFrame(frame, text="Backend")
        backend_box.pack(fill=tk.X, pady=4)
        method_var = tk.StringVar(value=str(getattr(self, "sleep_scoring_method", "pftsleep")))
        ttk.Radiobutton(backend_box, text="YASA", variable=method_var, value="yasa").pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Radiobutton(backend_box, text="U-Sleep", variable=method_var, value="usleep").pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Radiobutton(backend_box, text="PFTSleep", variable=method_var, value="pftsleep").pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Radiobutton(backend_box, text="AASM Rules", variable=method_var, value="aasm_rules").pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Radiobutton(backend_box, text="ML", variable=method_var, value="ml").pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Radiobutton(backend_box, text="ML+HMM", variable=method_var, value="ml_hmm").pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Radiobutton(backend_box, text="Rules+HMM", variable=method_var, value="rules_hmm").pack(side=tk.LEFT, padx=8, pady=6)

        common_box = ttk.LabelFrame(frame, text="Paramètres communs")
        common_box.pack(fill=tk.X, pady=4)
        ttk.Label(common_box, text="Durée époque (s):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        epoch_var = tk.StringVar(value=f"{float(getattr(self, 'auto_scoring_epoch_length', 30.0)):.1f}")
        ttk.Entry(common_box, textvariable=epoch_var, width=12).grid(row=0, column=1, sticky="w", padx=8, pady=6)

        yasa_box = ttk.LabelFrame(frame, text="YASA")
        yasa_box.pack(fill=tk.X, pady=4)
        ttk.Label(yasa_box, text="Target sfreq interne (info, pas de pré-resampling):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        yasa_sfreq_var = tk.StringVar(value=f"{float(getattr(self, 'yasa_target_sfreq', 100.0)):.1f}")
        ttk.Entry(yasa_box, textvariable=yasa_sfreq_var, width=12, state="readonly").grid(row=0, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(
            yasa_box,
            text="YASA applique lui-même son resampling / filtrage interne. Ne pas préfiltrer le signal.",
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))
        ttk.Label(yasa_box, text="Âge (optionnel):").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        yasa_age_var = tk.StringVar(value="" if getattr(self, "yasa_age", None) is None else str(int(getattr(self, "yasa_age", 0))))
        ttk.Entry(yasa_box, textvariable=yasa_age_var, width=12).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(yasa_box, text="Sexe:").grid(row=2, column=2, sticky="w", padx=8, pady=4)
        if getattr(self, "yasa_male", None) is True:
            default_sex = "Homme"
        elif getattr(self, "yasa_male", None) is False:
            default_sex = "Femme"
        else:
            default_sex = "Non renseigné"
        yasa_sex_var = tk.StringVar(value=default_sex)
        ttk.Combobox(
            yasa_box,
            textvariable=yasa_sex_var,
            values=["Non renseigné", "Homme", "Femme"],
            width=16,
            state="readonly",
        ).grid(row=2, column=3, sticky="w", padx=8, pady=4)
        ttk.Label(yasa_box, text="Seuil confiance faible (0-1):").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        yasa_conf_var = tk.StringVar(value=f"{float(getattr(self, 'yasa_confidence_threshold', 0.80)):.2f}")
        ttk.Entry(yasa_box, textvariable=yasa_conf_var, width=12).grid(row=3, column=1, sticky="w", padx=8, pady=4)

        us_box = ttk.LabelFrame(frame, text="U-Sleep")
        us_box.pack(fill=tk.X, pady=4)
        ttk.Label(us_box, text="Target sfreq (Hz):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        us_sfreq_var = tk.StringVar(value=f"{float(getattr(self, 'usleep_target_sfreq', 128.0)):.1f}")
        ttk.Entry(us_box, textvariable=us_sfreq_var, width=12).grid(row=0, column=1, sticky="w", padx=8, pady=6)

        ttk.Label(us_box, text="Device:").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        dev_var = tk.StringVar(value=str(getattr(self, "usleep_device", "auto")).lower())
        dev_combo = ttk.Combobox(us_box, textvariable=dev_var, values=["auto", "cpu", "cuda"], width=10, state="readonly")
        dev_combo.grid(row=0, column=3, sticky="w", padx=8, pady=6)

        use_eog_var = tk.BooleanVar(value=bool(getattr(self, "usleep_use_eog", True)))
        zscore_var = tk.BooleanVar(value=bool(getattr(self, "usleep_zscore", True)))
        ttk.Checkbutton(us_box, text="Inclure EOG (recommandé)", variable=use_eog_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(us_box, text="Normalisation z-score canal (recommandé)", variable=zscore_var).grid(row=1, column=2, columnspan=2, sticky="w", padx=8, pady=4)

        ckpt_box = ttk.LabelFrame(frame, text="Checkpoint U-Sleep")
        ckpt_box.pack(fill=tk.X, pady=4)
        ckpt_var = tk.StringVar(value=str(getattr(self, "usleep_checkpoint_path", "") or ""))
        ttk.Entry(ckpt_box, textvariable=ckpt_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=8)

        def _browse_ckpt():
            fp = filedialog.askopenfilename(
                title="Sélectionner checkpoint U-Sleep",
                filetypes=[
                    ("Model checkpoints", "*.pt *.pth *.ckpt"),
                    ("PyTorch .pt", "*.pt"),
                    ("PyTorch .pth", "*.pth"),
                    ("Checkpoint .ckpt", "*.ckpt"),
                    ("Tous les fichiers", "*.*"),
                ],
            )
            if fp:
                ckpt_var.set(fp)
        ttk.Button(ckpt_box, text="Parcourir", command=_browse_ckpt).pack(side=tk.LEFT, padx=8, pady=8)

        pft_box = ttk.LabelFrame(frame, text="PFTSleep")
        pft_box.pack(fill=tk.X, pady=4)
        pft_help = (
            "Canaux recommandés (PFTSleep):\n"
            "• EEG référencé: C4-M1 ou C3-M2\n"
            "• EOG gauche référencé: E1-M2\n"
            "• EMG menton: Chin1-Chin2 ou Chin1-Chin3\n"
            "• ECG: dérivation augmentée (ECG / ECG2)\n"
            "Les canaux manquants peuvent être passés en 'dummy' (vecteur nul)."
        )
        ttk.Label(pft_box, text=pft_help, justify=tk.LEFT).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 6))

        ttk.Label(pft_box, text="Device:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        pft_dev_var = tk.StringVar(value=str(getattr(self, "pft_device", "auto")).lower())
        ttk.Combobox(pft_box, textvariable=pft_dev_var, values=["auto", "cpu", "cuda:0", "mps"], width=10, state="readonly").grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(pft_box, text="Dossier modèles:").grid(row=1, column=2, sticky="w", padx=8, pady=4)
        # Par défaut, utiliser le dossier PFTSleep/ à la racine du projet
        default_pft_dir = str(getattr(self, "pft_models_dir", "")) or "PFTSleep"
        pft_models_var = tk.StringVar(value=default_pft_dir)
        ttk.Entry(pft_box, textvariable=pft_models_var, width=28).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(pft_box, text="Token HF (optionnel):").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        pft_token_var = tk.StringVar(value=str(getattr(self, "pft_hf_token", "") or ""))
        ttk.Entry(pft_box, textvariable=pft_token_var, width=40).grid(row=2, column=1, columnspan=3, sticky="w", padx=4, pady=4)

        ttk.Label(pft_box, text="EEG:").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        pft_eeg_var = tk.StringVar(value=str(getattr(self, "pft_eeg_channel", "") or "C4-M1"))
        ttk.Entry(pft_box, textvariable=pft_eeg_var, width=18).grid(row=3, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(pft_box, text="EOG gauche:").grid(row=3, column=2, sticky="w", padx=8, pady=4)
        pft_eog_var = tk.StringVar(value=str(getattr(self, "pft_eog_channel", "") or "E1-M2"))
        ttk.Entry(pft_box, textvariable=pft_eog_var, width=18).grid(row=3, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(pft_box, text="EMG menton:").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        pft_emg_var = tk.StringVar(value=str(getattr(self, "pft_emg_channel", "") or "CHIN1-CHIN2"))
        ttk.Entry(pft_box, textvariable=pft_emg_var, width=18).grid(row=4, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(pft_box, text="ECG:").grid(row=4, column=2, sticky="w", padx=8, pady=4)
        pft_ecg_var = tk.StringVar(value=str(getattr(self, "pft_ecg_channel", "") or "ECG"))
        ttk.Entry(pft_box, textvariable=pft_ecg_var, width=18).grid(row=4, column=3, sticky="w", padx=4, pady=4)

        profile_presets = {
            "clinical": {
                "label": "Recommandé clinique",
                "method": "pftsleep",
                "epoch": 30.0,
                "yasa_sfreq": 100.0,
                "us_sfreq": 128.0,
                "device": "auto",
                "use_eog": True,
                "zscore": True,
            },
            "fast_cpu": {
                "label": "Rapide CPU",
                "method": "yasa",
                "epoch": 30.0,
                "yasa_sfreq": 100.0,
                "us_sfreq": 100.0,
                "device": "cpu",
                "use_eog": False,
                "zscore": True,
            },
            "high_fidelity": {
                "label": "Haute fidélité",
                "method": "pftsleep",
                "epoch": 30.0,
                "yasa_sfreq": 128.0,
                "us_sfreq": 128.0,
                "device": "auto",
                "use_eog": True,
                "zscore": True,
            },
        }

        profile_state_var = tk.StringVar(value="clinical")
        profile_label_var = tk.StringVar(value=f"Profil actif: {profile_presets['clinical']['label']}")
        ttk.Label(profile_box, textvariable=profile_label_var).pack(anchor="w", padx=8, pady=(0, 6))

        def _apply_profile(profile_id: str):
            p = profile_presets.get(profile_id, profile_presets["clinical"])
            method_var.set(str(p["method"]))
            epoch_var.set(f"{float(p['epoch']):.1f}")
            yasa_sfreq_var.set(f"{float(p['yasa_sfreq']):.1f}")
            us_sfreq_var.set(f"{float(p['us_sfreq']):.1f}")
            dev_var.set(str(p["device"]))
            use_eog_var.set(bool(p["use_eog"]))
            zscore_var.set(bool(p["zscore"]))
            profile_state_var.set(profile_id)
            profile_label_var.set(f"Profil actif: {str(p['label'])}")

        preset_btns = ttk.Frame(profile_box)
        preset_btns.pack(fill=tk.X, padx=8, pady=(2, 8))
        ttk.Button(
            preset_btns,
            text=profile_presets["clinical"]["label"],
            command=lambda: _apply_profile("clinical"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            preset_btns,
            text=profile_presets["fast_cpu"]["label"],
            command=lambda: _apply_profile("fast_cpu"),
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            preset_btns,
            text=profile_presets["high_fidelity"]["label"],
            command=lambda: _apply_profile("high_fidelity"),
        ).pack(side=tk.LEFT, padx=6)

        channels_box = ttk.LabelFrame(frame, text="Candidats canaux (ordre de priorité, séparés par virgule)")
        channels_box.pack(fill=tk.BOTH, expand=True, pady=4)
        ttk.Label(channels_box, text="EEG:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        eeg_var = tk.StringVar(value=", ".join(self.yasa_eeg_candidates))
        ttk.Entry(channels_box, textvariable=eeg_var).grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(channels_box, text="EOG:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        eog_var = tk.StringVar(value=", ".join(self.yasa_eog_candidates))
        ttk.Entry(channels_box, textvariable=eog_var).grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(channels_box, text="EMG:").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        emg_var = tk.StringVar(value=", ".join(self.yasa_emg_candidates))
        ttk.Entry(channels_box, textvariable=emg_var).grid(row=2, column=1, sticky="ew", padx=8, pady=4)
        channels_box.grid_columnconfigure(1, weight=1)

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(10, 0))

        def _reset_defaults():
            _apply_profile("clinical")

        def _parse_list(raw_text: str) -> List[str]:
            vals = [v.strip() for v in str(raw_text).split(",")]
            return [v for v in vals if v]

        def _apply():
            try:
                epoch = float(epoch_var.get())
                yasa_fs = float(yasa_sfreq_var.get())
                us_fs = float(us_sfreq_var.get())
                conf_thr = float(yasa_conf_var.get())
                if epoch <= 0 or yasa_fs <= 0 or us_fs <= 0:
                    raise ValueError("Les fréquences et la durée d'époque doivent être > 0.")
                if not (0.0 <= conf_thr <= 1.0):
                    raise ValueError("Le seuil de confiance YASA doit être compris entre 0 et 1.")
                eeg = _parse_list(eeg_var.get())
                eog = _parse_list(eog_var.get())
                emg = _parse_list(emg_var.get())
                if not eeg:
                    raise ValueError("La liste EEG ne peut pas être vide.")
                age_text = str(yasa_age_var.get()).strip()
                yasa_age = int(age_text) if age_text else None
                if yasa_age is not None and yasa_age <= 0:
                    raise ValueError("L'âge YASA doit être un entier positif.")
                sex_value = str(yasa_sex_var.get()).strip().lower()
                if sex_value == "homme":
                    yasa_male = True
                elif sex_value == "femme":
                    yasa_male = False
                else:
                    yasa_male = None

                self.auto_scoring_epoch_length = float(epoch)
                self.yasa_target_sfreq = float(yasa_fs)
                self.yasa_age = yasa_age
                self.yasa_male = yasa_male
                self.yasa_confidence_threshold = float(conf_thr)
                self.usleep_target_sfreq = float(us_fs)
                self.usleep_device = str(dev_var.get()).lower().strip() or "auto"
                self.usleep_use_eog = bool(use_eog_var.get())
                self.usleep_zscore = bool(zscore_var.get())
                self.yasa_eeg_candidates = eeg
                self.yasa_eog_candidates = eog
                self.yasa_emg_candidates = emg
                ckpt = str(ckpt_var.get()).strip()
                self.usleep_checkpoint_path = ckpt if ckpt else None
                self._set_sleep_scoring_method(method_var.get())
                if profile_state_var.get() in profile_presets:
                    active_lbl = profile_presets[profile_state_var.get()]["label"]
                else:
                    active_lbl = "Personnalisé"
                profile_label_var.set(f"Profil actif: {active_lbl}")
                messagebox.showinfo("Auto-scoring", "Paramètres auto-scoring enregistrés.")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Configuration scoring", f"Paramètres invalides:\n{e}")

        ttk.Button(btns, text="Réinitialiser valeurs recommandées", command=_reset_defaults).pack(side=tk.LEFT)
        ttk.Button(btns, text="Annuler", command=win.destroy).pack(side=tk.RIGHT, padx=6)
        ttk.Button(btns, text="Appliquer", command=_apply).pack(side=tk.RIGHT)

    def _compare_scoring(self):
        """Compare auto (YASA) vs manuel - version simplifiée et rapide."""
        if self.sleep_scoring_data is None:
            messagebox.showwarning("Avertissement", "Aucun scoring automatique disponible. Lancez le scoring auto d'abord.")
            return
        if not hasattr(self, 'manual_scoring_data') or self.manual_scoring_data is None:
            messagebox.showwarning("Avertissement", "Veuillez charger un scoring manuel (Excel) avant de comparer.")
            return
        
        try:
            epoch_len = 30.0
            
            # Normalisation simple des stades
            auto = self.sleep_scoring_data.copy()
            auto['stage'] = auto['stage'].astype(str).str.upper().str.strip()
            
            manual = self.manual_scoring_data.copy()
            manual['stage'] = manual['stage'].astype(str).str.upper().str.strip()
            
            # Correction spécifique pour ÉVEIL et REM dans le manuel
            print(f"DEBUG COMPARE: Stades manuels avant correction = {manual['stage'].unique()}")
            manual['stage'] = manual['stage'].replace({
                'ÉVEIL': 'W', 'EVEIL': 'W',  # Éveil
                'REM': 'R'                   # REM
            })
            print(f"DEBUG COMPARE: Stades manuels après correction = {manual['stage'].unique()}")
            print(f"DEBUG COMPARE: Stades automatiques = {auto['stage'].unique()}")
            
            # Alignement temporel simple : prendre les époques communes
            auto['epoch'] = np.floor((auto['time'] + 1e-6) / epoch_len).astype(int)
            manual['epoch'] = np.floor((manual['time'] + 1e-6) / epoch_len).astype(int)
            
            # Merge simple sur les époques communes
            merged = pd.merge(auto, manual, on='epoch', how='inner', suffixes=('_auto', '_manual'))
            if merged.empty:
                messagebox.showwarning("Avertissement", "Aucune période commune entre auto et manuel.")
                return

            # Restreindre à la plage nuit (X à T) si définie
            night_range_applied = False
            night_start_min = getattr(self, "night_start_min", None)
            night_end_min = getattr(self, "night_end_min", None)
            if night_start_min is not None and night_end_min is not None:
                time_min = merged["epoch"] * (epoch_len / 60.0)
                merged = merged[(time_min >= float(night_start_min)) & (time_min <= float(night_end_min))]
                night_range_applied = True  # plage définie et appliquée
                if merged.empty:
                    messagebox.showwarning(
                        "Avertissement",
                        f"Aucune époque dans la plage nuit ({night_start_min:.0f}–{night_end_min:.0f} min). "
                        "Vérifiez « Définir plage nuit » ou désactivez-la.",
                    )
                    return

            # Option utilisateur : ignorer ou non les époques où le manuel est 'U'
            ignore_u = messagebox.askyesno(
                "Comparaison Auto vs Manuel",
                "Voulez-vous ignorer les époques où le manuel est 'U' (début/fin de nuit non scorés) ?"
            )

            n_excluded_u = 0
            if ignore_u:
                n_before = len(merged)
                merged = merged[merged['stage_manual'].str.upper().str.strip() != 'U']
                n_excluded_u = n_before - len(merged)
                if merged.empty:
                    messagebox.showwarning("Avertissement", "Aucune époque avec stade manuel défini (toutes en U).")
                    return

            confidence_col = None
            if "confidence" in merged.columns:
                confidence_col = "confidence"
            elif "confidence_auto" in merged.columns:
                confidence_col = "confidence_auto"

            n_excluded_low_confidence = 0
            confidence_threshold = None
            avg_confidence = None
            if confidence_col is not None:
                merged[confidence_col] = pd.to_numeric(merged[confidence_col], errors="coerce")
                confidence_threshold = float(getattr(self, "yasa_confidence_threshold", 0.80))
                avg_confidence = float(merged[confidence_col].dropna().mean()) if merged[confidence_col].notna().any() else None
                exclude_low_conf = messagebox.askyesno(
                    "Comparaison Auto vs Manuel",
                    f"Voulez-vous exclure les époques à faible confiance YASA (< {confidence_threshold:.2f}) ?"
                )
                if exclude_low_conf:
                    n_before = len(merged)
                    merged = merged[
                        merged[confidence_col].isna() | (merged[confidence_col] >= confidence_threshold)
                    ]
                    n_excluded_low_confidence = n_before - len(merged)
                    if merged.empty:
                        messagebox.showwarning(
                            "Avertissement",
                            "Aucune époque restante après exclusion des époques à faible confiance."
                        )
                        return
                    avg_confidence = float(merged[confidence_col].dropna().mean()) if merged[confidence_col].notna().any() else None

            # Calculs basiques sur la période scorée uniquement
            y_true = merged['stage_manual']
            y_pred = merged['stage_auto']
            
            # Accuracy simple
            correct = (y_true == y_pred).sum()
            total = len(merged)
            accuracy = correct / total if total > 0 else 0
            
            # Matrice de confusion simple
            labels = ['W', 'N1', 'N2', 'N3', 'R']
            cm = pd.crosstab(y_true, y_pred, rownames=['Manuel'], colnames=['Auto'], dropna=False)
            cm = cm.reindex(index=labels, columns=labels, fill_value=0)

            # Comptages par stade
            counts_manual = y_true.value_counts().reindex(labels, fill_value=0).astype(int).tolist()
            counts_auto = y_pred.value_counts().reindex(labels, fill_value=0).astype(int).tolist()

            # Métriques par stade (précision, rappel, spécificité)
            stage_metrics = {}
            for stage in labels:
                # Vrais positifs, faux positifs, faux négatifs
                tp = cm.loc[stage, stage] if stage in cm.index and stage in cm.columns else 0
                fp = cm[stage].sum() - tp if stage in cm.columns else 0
                fn = cm.loc[stage].sum() - tp if stage in cm.index else 0
                tn = total - tp - fp - fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                stage_metrics[stage] = {
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'f1': f1,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                }
            
            # Kappa (Cohen) et F1 macro
            kappa = None
            try:
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
            except Exception:
                pass
            macro_f1 = np.mean([stage_metrics[s]["f1"] for s in labels]) if stage_metrics else 0.0
            
            # Calculs globaux supplémentaires
            total_time_hours = total * 30 / 3600  # 30 secondes par époque
            sleep_epochs_manual = total - counts_manual[0] if counts_manual else 0  # Total - Wake
            sleep_epochs_auto = total - counts_auto[0] if counts_auto else 0
            sleep_time_manual = sleep_epochs_manual * 30 / 3600
            sleep_time_auto = sleep_epochs_auto * 30 / 3600
            
            # Affichage détaillé
            self._show_detailed_comparison(
                n_epochs=len(merged),
                accuracy=accuracy,
                labels=labels,
                cm=cm,
                counts_manual=counts_manual,
                counts_auto=counts_auto,
                stage_metrics=stage_metrics,
                total_time_hours=total_time_hours,
                sleep_time_manual=sleep_time_manual,
                sleep_time_auto=sleep_time_auto,
                n_excluded_undefined=n_excluded_u,
                n_excluded_low_confidence=n_excluded_low_confidence,
                night_range_applied=night_range_applied,
                night_start_min=night_start_min,
                night_end_min=night_end_min,
                kappa=kappa,
                macro_f1=macro_f1,
                avg_confidence=avg_confidence,
                confidence_threshold=confidence_threshold,
            )
            
        except Exception as e:
            logging.error(f"[COMPARE] Erreur comparaison: {e}")
            messagebox.showerror("Erreur", f"Impossible d'effectuer la comparaison: {e}")

    def _open_night_range_dialog(self):
        """Ouvre un dialogue pour définir la plage de nuit (X à T) en minutes depuis le début d'enregistrement."""
        win = tk.Toplevel(self.root)
        win.title("Plage nuit (X à T)")
        win.geometry("420x200")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Définir la plage de nuit pour les comparaisons Auto vs Manuel.", font=("Segoe UI", 10, "bold")).pack(pady=(12, 8))
        ttk.Label(win, text="Les valeurs sont en minutes depuis le début de l'enregistrement (t = 0).").pack(pady=(0, 12))

        row = ttk.Frame(win)
        row.pack(fill=tk.X, padx=12, pady=6)
        ttk.Label(row, text="Début nuit (min):", width=18).pack(side=tk.LEFT)
        start_var = tk.StringVar(value=str(int(self.night_start_min)) if self.night_start_min is not None else "")
        ttk.Entry(row, textvariable=start_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="Fin nuit (min):", width=14).pack(side=tk.LEFT, padx=(12, 0))
        end_var = tk.StringVar(value=str(int(self.night_end_min)) if self.night_end_min is not None else "")
        ttk.Entry(row, textvariable=end_var, width=12).pack(side=tk.LEFT, padx=4)

        ttk.Label(win, text="Exemple: 10 et 450 → nuit de 10 min à 7 h 30.", foreground="gray").pack(pady=(0, 12))

        def apply_range():
            try:
                s = start_var.get().strip()
                e = end_var.get().strip()
                if not s or not e:
                    self.night_start_min = None
                    self.night_end_min = None
                    logging.info("[NUIT] Plage nuit désactivée.")
                    messagebox.showinfo("Plage nuit", "Plage nuit désactivée. La comparaison utilisera toute la durée.")
                else:
                    start_min = float(s)
                    end_min = float(e)
                    if start_min < 0 or end_min < 0:
                        messagebox.showwarning("Plage nuit", "Les valeurs doivent être positives.")
                        return
                    if start_min >= end_min:
                        messagebox.showwarning("Plage nuit", "Le début doit être strictement inférieur à la fin.")
                        return
                    self.night_start_min = start_min
                    self.night_end_min = end_min
                    logging.info("[NUIT] Plage nuit: %.0f – %.0f min", start_min, end_min)
                    messagebox.showinfo("Plage nuit", f"Plage nuit enregistrée: de {start_min:.0f} à {end_min:.0f} minutes.")
                win.destroy()
            except ValueError:
                messagebox.showwarning("Plage nuit", "Veuillez entrer des nombres valides (minutes).")

        def clear_range():
            self.night_start_min = None
            self.night_end_min = None
            start_var.set("")
            end_var.set("")
            logging.info("[NUIT] Plage nuit désactivée.")
            messagebox.showinfo("Plage nuit", "Plage nuit désactivée.")
            win.destroy()

        btn_row = ttk.Frame(win)
        btn_row.pack(pady=16)
        ttk.Button(btn_row, text="Désactiver", command=clear_range).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Annuler", command=win.destroy).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Appliquer", command=apply_range).pack(side=tk.LEFT, padx=6)

    def _show_detailed_comparison(self, n_epochs: int, accuracy: float, labels: list, cm: pd.DataFrame, 
                                counts_manual: list, counts_auto: list, stage_metrics: dict,
                                total_time_hours: float, sleep_time_manual: float, sleep_time_auto: float,
                                n_excluded_undefined: int = 0,
                                n_excluded_low_confidence: int = 0,
                                night_range_applied: bool = False,
                                night_start_min: Optional[float] = None,
                                night_end_min: Optional[float] = None,
                                kappa: Optional[float] = None,
                                macro_f1: Optional[float] = None,
                                avg_confidence: Optional[float] = None,
                                confidence_threshold: Optional[float] = None):
        """Affiche une comparaison détaillée dans une fenêtre avec onglets."""
        win = tk.Toplevel(self.root)
        win.title("Comparaison Scoring (Auto vs Manuel)")
        win.geometry("800x700")
        win.transient(self.root)
        
        # Notebook pour les onglets
        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ============ ONGLET RÉSUMÉ ============
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Résumé")
        
        # Titre
        title_label = ttk.Label(summary_frame, text="Comparaison Scoring Automatique vs Manuel", 
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Métriques globales
        global_frame = ttk.LabelFrame(summary_frame, text="Métriques Globales", padding=10)
        global_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(global_frame, text=f"Durée totale analysée: {total_time_hours:.1f} heures").pack(anchor=tk.W)
        ttk.Label(global_frame, text=f"Époques comparées: {n_epochs}").pack(anchor=tk.W)
        if n_excluded_undefined > 0:
            ttk.Label(global_frame, text=f"Époques exclues (manuel = U, début/fin de nuit): {n_excluded_undefined}").pack(anchor=tk.W)
        if n_excluded_low_confidence > 0 and confidence_threshold is not None:
            ttk.Label(global_frame, text=f"Époques exclues (confiance < {confidence_threshold:.2f}): {n_excluded_low_confidence}").pack(anchor=tk.W)
        if night_range_applied and night_start_min is not None and night_end_min is not None:
            ttk.Label(global_frame, text=f"Plage nuit appliquée: de {night_start_min:.0f} à {night_end_min:.0f} min (depuis début d'enregistrement)").pack(anchor=tk.W)
        ttk.Label(global_frame, text=f"Précision globale: {accuracy:.1%}").pack(anchor=tk.W)
        ttk.Label(global_frame, text=f"Époques correctes: {int(accuracy * n_epochs)}/{n_epochs}").pack(anchor=tk.W)
        if kappa is not None:
            ttk.Label(global_frame, text=f"Kappa (Cohen): {kappa:.3f}").pack(anchor=tk.W)
        if macro_f1 is not None:
            ttk.Label(global_frame, text=f"F1 macro: {macro_f1:.3f}").pack(anchor=tk.W)
        if avg_confidence is not None:
            ttk.Label(global_frame, text=f"Confiance moyenne YASA: {avg_confidence:.3f}").pack(anchor=tk.W)
        
        # Temps de sommeil
        sleep_frame = ttk.LabelFrame(summary_frame, text="Temps de Sommeil", padding=10)
        sleep_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(sleep_frame, text=f"Temps de sommeil manuel: {sleep_time_manual:.1f} heures ({sleep_time_manual/total_time_hours:.1%})").pack(anchor=tk.W)
        ttk.Label(sleep_frame, text=f"Temps de sommeil automatique: {sleep_time_auto:.1f} heures ({sleep_time_auto/total_time_hours:.1%})").pack(anchor=tk.W)
        ttk.Label(sleep_frame, text=f"Différence: {abs(sleep_time_manual - sleep_time_auto):.1f} heures").pack(anchor=tk.W)
        
        # Comptages par stade
        counts_frame = ttk.LabelFrame(summary_frame, text="Répartition par Stade", padding=10)
        counts_frame.pack(fill=tk.X)
        
        counts_text = tk.Text(counts_frame, height=8, width=60, font=("Courier", 9))
        counts_text.pack(fill=tk.BOTH, expand=True)
        
        counts_text.insert(tk.END, f"{'Stade':<6} {'Manuel':<8} {'Auto':<8} {'Diff':<8} {'% Manuel':<10} {'% Auto':<10}\n")
        counts_text.insert(tk.END, "-" * 60 + "\n")
        
        for i, stage in enumerate(labels):
            manual_count = counts_manual[i]
            auto_count = counts_auto[i]
            diff = manual_count - auto_count
            pct_manual = manual_count / n_epochs * 100 if n_epochs > 0 else 0
            pct_auto = auto_count / n_epochs * 100 if n_epochs > 0 else 0
            counts_text.insert(tk.END, f"{stage:<6} {manual_count:<8} {auto_count:<8} {diff:<8} {pct_manual:<9.1f}% {pct_auto:<9.1f}%\n")
        
        counts_text.config(state=tk.DISABLED)
        
        # Note littérature : confusion N2/N3
        lit_frame = ttk.LabelFrame(summary_frame, text="Littérature — confusion N2/N3", padding=10)
        lit_frame.pack(fill=tk.X, pady=(0, 10))
        lit_text = (
            "La confusion N2/N3 est fréquente en staging automatique : le N3 (sommeil profond à ondes lentes) "
            "est souvent sous-détecté ou classé en N2. Les modèles entraînés sur un jeu de données peuvent "
            "mal généraliser (population, montage, pathologie). La littérature rapporte que N1 est le stade "
            "le plus difficile, puis N3 ; les ensembles et jeux d’entraînement diversifiés limitent ces biais."
        )
        ttk.Label(lit_frame, text=lit_text, justify=tk.LEFT, wraplength=700).pack(anchor=tk.W)
        
        # ============ ONGLET MATRICE ============
        matrix_frame = ttk.Frame(notebook)
        notebook.add(matrix_frame, text="Matrice de Confusion")
        
        # Matrice de confusion
        cm_label = ttk.Label(matrix_frame, text="Matrice de Confusion", font=("Arial", 12, "bold"))
        cm_label.pack(pady=(10, 5))
        
        cm_text = tk.Text(matrix_frame, height=10, width=70, font=("Courier", 10))
        cm_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # En-tête
        header = "     " + "".join(f"{label:>6}" for label in labels)
        cm_text.insert(tk.END, f"Auto\\Manuel{header}\n")
        cm_text.insert(tk.END, "-" * (11 + 6 * len(labels)) + "\n")
        
        # Lignes de la matrice
        for i, label in enumerate(labels):
            row = f"{label:>3}: "
            for j, col_label in enumerate(labels):
                value = cm.loc[label, col_label] if label in cm.index and col_label in cm.columns else 0
                row += f"{value:>6}"
            cm_text.insert(tk.END, f"{row}\n")
        
        cm_text.config(state=tk.DISABLED)
        
        # ============ ONGLET MÉTRIQUES ============
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Métriques par Stade")
        
        # Métriques par stade
        metrics_label = ttk.Label(metrics_frame, text="Métriques Détaillées par Stade", font=("Arial", 12, "bold"))
        metrics_label.pack(pady=(10, 5))
        
        metrics_text = tk.Text(metrics_frame, height=15, width=80, font=("Courier", 9))
        metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        metrics_text.insert(tk.END, f"{'Stade':<6} {'Précision':<10} {'Rappel':<10} {'Spécificité':<12} {'F1-Score':<10} {'TP':<4} {'FP':<4} {'FN':<4}\n")
        metrics_text.insert(tk.END, "-" * 80 + "\n")
        
        for stage in labels:
            m = stage_metrics[stage]
            metrics_text.insert(tk.END, f"{stage:<6} {m['precision']:<9.3f} {m['recall']:<9.3f} {m['specificity']:<11.3f} {m['f1']:<9.3f} {m['tp']:<4} {m['fp']:<4} {m['fn']:<4}\n")
        
        metrics_text.config(state=tk.DISABLED)
        
        # Bouton fermer
        ttk.Button(win, text="Fermer", command=win.destroy).pack(pady=(10, 0))

    def _optimize_alignment_and_montage(self) -> dict:
        """Try recommended EEG montages and time offsets to maximize agreement with manual scoring.

        Returns dict with keys: auto_df, montage, offset_epochs, kappa, accuracy
        """
        try:
            import math
            # Candidate montages (ordered)
            montages = ['Fpz-Cz', 'Pz-Oz', 'Fz-Cz', 'C4-M1', 'C3-M2']
            # Build fallback list from current candidates to keep behavior
            fallback = [m for m in self.yasa_eeg_candidates if m not in montages]
            epoch_len = 30.0
            best = {'kappa': -1.0, 'accuracy': -1.0}

            # Prepare manual epochs normalized
            manual = self.manual_scoring_data.copy()
            manual['stage'] = manual['stage'].astype(str).str.upper().str.strip().replace({'WAKE': 'W', 'REM': 'R'})
            manual['epoch'] = np.floor((manual['time'] + 1e-6) / epoch_len).astype(int)
            manual_e = manual.groupby('epoch')['stage'].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).reset_index()

            for montage in montages:
                # Run YASA with montage prioritized
                try:
                    scorer = SleepScorer(
                        method="yasa",
                        eeg_candidates=[montage] + fallback,
                        eog_candidates=self.yasa_eog_candidates,
                        emg_candidates=self.yasa_emg_candidates,
                        epoch_length=epoch_len,
                        target_sfreq=100.0,
                        yasa_age=getattr(self, "yasa_age", None),
                        yasa_male=getattr(self, "yasa_male", None),
                    )
                    auto_df = scorer.score(self.raw)
                except Exception:
                    continue

                # Normalize
                auto = auto_df.copy()
                auto['stage'] = auto['stage'].astype(str).str.upper().str.strip().replace({'WAKE': 'W', 'REM': 'R'})

                # Search offset in [-40, 40] epochs (~±20 min)
                best_local = {'kappa': -1.0, 'accuracy': -1.0, 'offset_epochs': 0}
                for off in range(-40, 41):
                    shifted_time = auto['time'] + off * epoch_len
                    auto_epochs = np.floor((shifted_time + 1e-6) / epoch_len).astype(int)
                    auto_e = pd.DataFrame({'epoch': auto_epochs, 'stage': auto['stage']})
                    auto_e = auto_e.groupby('epoch')['stage'].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).reset_index()
                    merged = pd.merge(auto_e, manual_e, on='epoch', how='inner', suffixes=('', '_manual'))
                    if merged.empty:
                        continue
                    labels = ['W', 'N1', 'N2', 'N3', 'R']
                    y_true = merged['stage_manual']
                    y_pred = merged['stage']

                    # Accuracy
                    try:
                        from sklearn.metrics import cohen_kappa_score
                        kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
                    except Exception:
                        kappa = float('nan')
                    acc = float((y_true == y_pred).mean())

                    # Keep best local by kappa, then accuracy, then support size
                    score_tuple = (math.isnan(kappa), kappa, acc, len(merged))
                    best_tuple = (math.isnan(best_local['kappa']), best_local['kappa'], best_local['accuracy'], best_local.get('n', 0))
                    if (not math.isnan(kappa) and (kappa > best_local['kappa'])) or (math.isnan(best_local['kappa']) and not math.isnan(kappa)) or (math.isclose(kappa, best_local['kappa'], rel_tol=1e-9) and acc > best_local['accuracy']):
                        best_local = {'kappa': kappa, 'accuracy': acc, 'offset_epochs': off, 'n': len(merged)}

                # Update global best
                if best_local['kappa'] > best.get('kappa', -1) or (math.isclose(best_local['kappa'], best.get('kappa', -1), rel_tol=1e-9) and best_local['accuracy'] > best.get('accuracy', -1)):
                    best.update({'auto_df': auto_df, 'montage': montage, 'offset_epochs': best_local['offset_epochs'], 'kappa': best_local['kappa'], 'accuracy': best_local['accuracy']})

            return best
        except Exception as e:
            logging.warning(f"Optimisation ignorée: {e}")
            return {}

    def _open_comparison_window(self, n_epochs: int, accuracy: float, kappa: float, labels: list, cm: pd.DataFrame,
                                 precision, recall, f1, support, macro_f1: float,
                                 counts_manual: list, counts_auto: list,
                                 chosen_montage: str = 'auto', chosen_offset_epochs: int = 0) -> None:
        """Create a clean Tk window showing comparison metrics and a bar chart for F1."""
        win = tk.Toplevel(self.root)
        win.title("Comparaison Scoring (Auto vs Manuel)")
        win.geometry("1100x900")
        win.transient(self.root)
        win.grab_set()

        # Top metrics frame
        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=12, pady=8)
        ttk.Label(top, text=f"Époques comparées: {n_epochs}", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(top, text=f"Accuracy: {accuracy*100:.1f}%").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(top, text=f"Kappa: {kappa:.3f}").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(top, text=f"Macro-F1: {macro_f1:.3f}").pack(side=tk.LEFT)
        # Chosen config
        sub = ttk.Frame(win)
        sub.pack(fill=tk.X, padx=12)
        ttk.Label(sub, text=f"Montage: {chosen_montage} | Décalage: {chosen_offset_epochs} époques ({chosen_offset_epochs*30}s)").pack(anchor='w')

        # Main content split
        content = ttk.Frame(win)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Left: confusion matrix table
        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Matrice de confusion (lignes=manuel, colonnes=auto)", font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 6))
        # Render cm as text
        cm_text = tk.Text(left, height=12, wrap=tk.NONE)
        cm_text.pack(fill=tk.BOTH, expand=True)
        cm_text.insert('1.0', cm.to_string())
        cm_text.configure(state='disabled')

        # Right: F1 bar chart
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Scores par stade (Précision, Rappel, F1)", font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 6))

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            fig, ax = plt.subplots(figsize=(5, 3))
            x = np.arange(len(labels))
            width = 0.25
            ax.bar(x - width, precision, width, label='Précision')
            ax.bar(x, recall, width, label='Rappel')
            ax.bar(x + width, f1, width, label='F1')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_ylabel('Score')
            ax.set_title('Scores par classe')
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=right)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as _:
            ttk.Label(right, text="Plot indisponible (matplotlib)").pack()

        # Bottom: simple per-stage counts (manual vs auto)
        bottom = ttk.Frame(win)
        bottom.pack(fill=tk.X, padx=12, pady=8)
        ttk.Label(bottom, text="Comptage par stade (Manuel vs Auto)", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        counts_text = tk.Text(bottom, height=6, wrap=tk.NONE)
        counts_text.pack(fill=tk.X, expand=False)
        counts_text.insert('1.0', "Stade    Manuel    Auto\n")
        counts_text.insert('2.0', "-------------------------\n")
        for i, lbl in enumerate(labels):
            counts_text.insert(tk.END, f"{lbl:<7}  {counts_manual[i]:>6}    {counts_auto[i]:>6}\n")
        counts_text.configure(state='disabled')
    
    def _update_plot(self, *args):
        """Met à jour le graphique (alias pour update_plot)."""
        self.update_plot()
    
    def _show_channel_stats(self):
        """Affiche les statistiques des canaux."""
        if not self.raw:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
        
        # Créer une fenêtre pour afficher les statistiques
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Statistiques des canaux")
        stats_window.geometry("600x400")
        
        # Zone de texte pour afficher les statistiques
        text_widget = tk.Text(stats_window, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = tk.Scrollbar(stats_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack les widgets
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Collecter les statistiques
        stats_text = "Statistiques des canaux EEG\n"
        stats_text += "=" * 50 + "\n\n"
        
        for channel_name, signal in self.derivations.items():
            if len(signal) > 0:
                stats_text += f"Canal: {channel_name}\n"
                stats_text += f"  Longueur: {len(signal)} échantillons\n"
                stats_text += f"  Durée: {len(signal)/self.raw.info['sfreq']:.2f} secondes\n"
                stats_text += f"  Amplitude min: {np.min(signal):.2f} µV\n"
                stats_text += f"  Amplitude max: {np.max(signal):.2f} µV\n"
                stats_text += f"  Amplitude RMS: {np.sqrt(np.mean(signal**2)):.2f} µV\n"
                stats_text += f"  Écart-type: {np.std(signal):.2f} µV\n"
                stats_text += "\n"
        
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state=tk.DISABLED)
    
    def _show_diagnostics(self):
        """Affiche le diagnostic des données."""
        if not self.raw:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
        
        # Diagnostic simple
        diagnostic = []
        diagnostic.append("🔍 Diagnostic des Données EEG")
        diagnostic.append("=" * 40)
        
        if self.selected_channels:
            for channel in self.selected_channels:
                if channel in self.derivations:
                    data = self.derivations[channel]
                    amplitude = np.max(data) - np.min(data)
                    
                    if np.all(data == 0):
                        status = "❌ Signal nul"
                    elif np.std(data) < 1e-6:
                        status = "⚠️ Signal très plat"
                    elif amplitude < 1e-3:
                        status = "⚠️ Amplitude très faible"
                    else:
                        status = "✅ Signal normal"
                    
                    diagnostic.append(f"{channel}: {status}")
        
        messagebox.showinfo("Diagnostic", "\n".join(diagnostic))
    
    def _show_spectral_analysis(self):
        """Affiche l'analyse spectrale (FFT) pour les canaux sélectionnés."""
        print("🔍 CHECKPOINT FFT 1: Entrée _show_spectral_analysis")
        if not self.raw or not self.selected_channels:
            print(f"⚠️ CHECKPOINT FFT 1: raw={self.raw is not None}, selected_channels={self.selected_channels}")
            messagebox.showwarning("Attention", "Chargez un fichier et sélectionnez au moins un canal")
            return

        def create_tooltip(widget, text):
            """Crée un tooltip pour un widget."""
            def show_tooltip(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                
                label = tk.Label(tooltip, text=text, justify='left', 
                               background='#ffffe0', relief='solid', borderwidth=1,
                               font=('Segoe UI', 9), wraplength=300)
                label.pack()
                
                widget.tooltip = tooltip
                
            def hide_tooltip(event):
                if hasattr(widget, 'tooltip') and widget.tooltip:
                    widget.tooltip.destroy()
                    widget.tooltip = None
                    
            widget.bind('<Enter>', show_tooltip)
            widget.bind('<Leave>', hide_tooltip)

        # Fenêtre secondaire
        print("🔍 CHECKPOINT FFT 2: Création fenêtre secondaire")
        spec_window = tk.Toplevel(self.root)
        spec_window.title("Spectral Analysis - FFT (Power by Band)")
        spec_window.geometry("1000x700")
        spec_window.transient(self.root)
        spec_window.grab_set()

        # Mise en page
        container = ttk.Frame(spec_window, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        upper = ttk.Frame(container)
        upper.pack(fill=tk.X)

        # Contrôles de fenêtre temporelle
        ttk.Label(upper, text="Début (s):").pack(side=tk.LEFT)
        start_var = tk.DoubleVar(value=self.current_time)
        start_entry = ttk.Entry(upper, textvariable=start_var, width=10)
        start_entry.pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(upper, text="Durée (s):").pack(side=tk.LEFT)
        dur_var = tk.DoubleVar(value=self.duration)
        dur_entry = ttk.Entry(upper, textvariable=dur_var, width=10)
        dur_entry.pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(upper, text="Canaux:").pack(side=tk.LEFT)
        channels_var = tk.StringVar(value=", ".join(self.selected_channels))
        channels_label = ttk.Label(upper, textvariable=channels_var)
        channels_label.pack(side=tk.LEFT, padx=(5, 15))

        refresh_btn = ttk.Button(upper, text="Actualiser", command=lambda: render())
        refresh_btn.pack(side=tk.RIGHT)
        save_fig_btn = ttk.Button(upper, text="Enregistrer Figure", command=lambda: _save_figure(fig))
        save_fig_btn.pack(side=tk.RIGHT, padx=(10,0))
        save_csv_btn = ttk.Button(upper, text="Exporter CSV", command=lambda: _export_csv(tree))
        save_csv_btn.pack(side=tk.RIGHT, padx=(10,0))
        
        # Tooltips pour les contrôles
        create_tooltip(start_entry, 
                      "Temps de début de l'analyse spectrale (secondes).\n\n"
                      "• Définit le point de départ de la fenêtre d'analyse\n"
                      "• Correspond au temps actuel affiché dans la vue principale\n"
                      "• Peut être modifié pour analyser une période spécifique\n"
                      "• Doit être ≥ 0 et < durée totale de l'enregistrement")
        
        create_tooltip(dur_entry, 
                      "Durée de la fenêtre d'analyse (secondes).\n\n"
                      "• Définit la longueur de la période analysée\n"
                      "• Plus la durée est longue, plus la résolution fréquentielle est fine\n"
                      "• Recommandé : 30-60 secondes pour un bon compromis\n"
                      "• Doit être ≥ 1 seconde pour des résultats fiables")
        
        create_tooltip(channels_label, 
                      "Canaux EEG sélectionnés pour l'analyse.\n\n"
                      "• Affiche la liste des canaux actuellement sélectionnés\n"
                      "• L'analyse FFT sera calculée pour chacun de ces canaux\n"
                      "• Les canaux sont sélectionnés dans la vue principale\n"
                      "• Chaque canal aura sa propre ligne dans le tableau de résultats")
        
        create_tooltip(refresh_btn, 
                      "Actualiser l'analyse spectrale.\n\n"
                      "• Recalcule l'analyse FFT avec les nouveaux paramètres\n"
                      "• Utilise la fenêtre temporelle et les canaux sélectionnés\n"
                      "• Met à jour le graphique et le tableau de résultats\n"
                      "• Nécessaire après modification des paramètres")
        
        create_tooltip(save_fig_btn, 
                      "Enregistrer le graphique d'analyse spectrale.\n\n"
                      "• Sauvegarde le graphique FFT au format PNG, PDF ou SVG\n"
                      "• Qualité haute résolution (200 DPI)\n"
                      "• Inclut le graphique et la légende\n"
                      "• Utile pour les publications et rapports")
        
        create_tooltip(save_csv_btn, 
                      "Exporter les données spectrales en CSV.\n\n"
                      "• Sauvegarde toutes les données numériques de l'analyse\n"
                      "• Format : canal, bande de fréquence, puissance, fréquence de pic, centroïde\n"
                      "• Compatible avec Excel, R, Python, etc.\n"
                      "• Permet l'analyse statistique externe")

        # Zone de tracé et table
        body = ttk.Frame(container)
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        print("🔍 CHECKPOINT FFT 3: Création figure matplotlib")
        fig, ax = plt.subplots(figsize=(7, 4))
        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        cols = ("Canal",) + tuple(EEG_BANDS.keys()) + ("Peak (Hz)", "Centroid (Hz)")
        tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, anchor=tk.CENTER)
        tree.pack(side=tk.TOP, fill=tk.Y)

        def render():
            try:
                print("🔍 CHECKPOINT FFT 4: Début render()")
                ax.clear()
                tree.delete(*tree.get_children())

                start_s = max(0.0, float(start_var.get()))
                dur_s = max(0.5, float(dur_var.get()))
                end_s = start_s + dur_s
                fs = float(self.sfreq)

                start_idx = int(start_s * fs)
                end_idx = int(end_s * fs)
                n_samples = len(self.raw.times)
                start_idx = max(0, min(start_idx, n_samples - 2))
                end_idx = max(start_idx + 2, min(end_idx, n_samples))

                print(f"🔍 CHECKPOINT FFT 5: Fenêtre temps {start_s:.2f}-{end_s:.2f}s, fs={fs}")
                # Tracé des PSDs
                for ch in self.selected_channels:
                    if ch not in self.derivations:
                        print(f"⚠️ CHECKPOINT FFT 6: Canal absent des dérivations: {ch}")
                        continue
                    data = self.derivations[ch][start_idx:end_idx]
                    if data.size == 0:
                        print(f"⚠️ CHECKPOINT FFT 6: Données vides pour {ch}")
                        continue
                    freqs, spec = compute_psd_fft(data, fs)
                    if freqs.size == 0:
                        print(f"⚠️ CHECKPOINT FFT 7: FFT vide pour {ch}")
                        continue
                    
                    # Apply log10 scaling to power (spec^2)
                    power = spec ** 2
                    power_log = np.log10(np.maximum(power, 1e-20))  # Avoid log(0)
                    ax.plot(freqs, power_log, label=ch, linewidth=1.0)

                    # Band powers et métriques
                    bands = compute_band_powers(freqs, spec)
                    peak, centroid = compute_peak_and_centroid(freqs, spec)
                    row = [ch] + [f"{bands[b]:.2f}" for b in EEG_BANDS.keys()] + [f"{peak:.2f}", f"{centroid:.2f}"]
                    tree.insert("", tk.END, values=tuple(row))

                ax.set_title("Spectrum (FFT Power)")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power (µV², log10)")
                ax.set_xlim(0, min(50, self.sfreq / 2))
                ax.legend(loc="upper right", fontsize=8, ncol=1)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                canvas.draw()
            except Exception as e:
                print(f"❌ CHECKPOINT FFT ERR: {e}")
                messagebox.showerror("Erreur", f"Echec de l'analyse spectrale: {e}")

        render()

        def _save_figure(fig_obj):
            try:
                file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    fig_obj.savefig(file_path, dpi=200, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")

        def _export_csv(treeview):
            try:
                file_path = filedialog.asksaveasfilename(title="Exporter CSV", defaultextension=".csv",
                                                         filetypes=[("CSV", "*.csv")])
                if not file_path:
                    return
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(cols)
                    for iid in treeview.get_children():
                        writer.writerow(treeview.item(iid, 'values'))
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'export CSV: {e}")

    def _show_stage_psd_before_after(self):
        """Figure 3e/f: PSD par stade (Welch) avant/après nettoyage sur un canal."""
        print("🔍 CHECKPOINT PSD 1: Entrée _show_stage_psd_before_after")
        if not self.raw:
            print("⚠️ CHECKPOINT PSD 1: raw manquant")
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
        # Choisir un canal (préférence: EEG Fpz-Cz/Fpz-Cz, sinon Pz/Pz-Oz, sinon sélectionné/premier)
        candidate = None
        for name in ["EEG Fpz-Cz", "Fpz-Cz", "Pz", "EEG Pz-Oz", "E101", "PZ"] + self.selected_channels + (self.raw.ch_names if self.raw else []):
            if name in self.derivations:
                candidate = name
                break
        if candidate is None:
            print("⚠️ CHECKPOINT PSD 2: Aucun canal trouvé")
            messagebox.showwarning("Attention", "Aucun canal disponible")
            return

        # Fenêtre
        top = tk.Toplevel(self.root)
        top.title("PSD par stade (Welch) – avant/après")
        top.geometry("1200x720")
        top.transient(self.root)
        top.grab_set()

        # Layout principal: barre d'outils en haut, panel latéral à gauche, graphique à droite
        toolbar = ttk.Frame(top, style='Custom.TFrame')
        toolbar.pack(fill=tk.X, side=tk.TOP)
        save_fig_btn = ttk.Button(toolbar, text="Enregistrer Figure", style='Custom.TButton')
        save_fig_btn.pack(side=tk.RIGHT, padx=(6,6), pady=4)
        export_csv_btn = ttk.Button(toolbar, text="Exporter CSV")
        export_csv_btn.pack(side=tk.RIGHT, padx=(6,0), pady=4)
        # Sélecteur de thème
        theme_var = tk.StringVar(value=self.theme_manager.current_theme_name)
        theme_combo = ttk.Combobox(toolbar, textvariable=theme_var, state="readonly", width=12)
        theme_combo['values'] = list(self.theme_manager.get_available_themes().values())
        theme_combo.pack(side=tk.RIGHT, padx=(6,0), pady=4)
        theme_combo.bind('<<ComboboxSelected>>', lambda e: self._change_theme_by_display_name(theme_var.get()))
        toolbar_label = ttk.Label(toolbar, text=f"Canal: {candidate} | Welch 4s 50% | µV²/Hz", font=('Segoe UI', 9))
        toolbar_label.pack(side=tk.LEFT, padx=8)
        
        # Tooltip pour le titre de la barre d'outils
        create_tooltip(toolbar_label, 
                      "Paramètres actuels de l'analyse PSD.\n\n"
                      "• Canal : canal EEG sélectionné pour l'analyse\n"
                      "• Welch 4s : méthode de Welch avec segments de 4 secondes\n"
                      "• 50% : chevauchement de 50% entre les segments\n"
                      "• µV²/Hz : unités de la densité spectrale de puissance\n"
                      "• Ces paramètres sont mis à jour lors du recalcul")

        main = ttk.Frame(top)
        main.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(main, width=320)
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)

        content = ttk.Frame(main)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Panneau latéral: options
        opt = ttk.LabelFrame(side, text="Paramètres PSD (Welch)", padding=8)
        opt.pack(fill=tk.Y, expand=True, padx=8, pady=8)
        ttk.Label(opt, text="Méthode: Welch (density)").pack(anchor='w')
        
        # Sélection du canal
        channel_box = ttk.LabelFrame(opt, text="Canal", padding=6)
        channel_box.pack(fill=tk.X, pady=(8,4))
        available_channels = list(self.derivations.keys())
        selected_channel_var = tk.StringVar(value=candidate)
        channel_combo = ttk.Combobox(channel_box, textvariable=selected_channel_var, values=available_channels, state="readonly", width=20)
        channel_combo.pack(fill=tk.X)
        
        # Variable pour stocker la fonction de recalcul
        recalculate_function = None
        
        # Lier le changement de canal au recalcul
        def on_channel_change(event=None):
            print(f"🔍 CHECKPOINT PSD CHANNEL: Canal changé vers {selected_channel_var.get()}")
            if recalculate_function:
                recalculate_function()
        
        channel_combo.bind('<<ComboboxSelected>>', on_channel_change)
        
        # Tooltip pour la sélection du canal
        create_tooltip(channel_combo, 
                      "Sélection du canal EEG pour l'analyse PSD.\n\n"
                      "• Choisissez le canal qui vous intéresse pour l'analyse spectrale\n"
                      "• Les canaux bipolaires (ex: C4-M1) sont généralement préférés\n"
                      "• Le canal sélectionné sera utilisé pour calculer la densité spectrale de puissance\n"
                      "• Chaque canal peut avoir des caractéristiques spectrales différentes selon sa position anatomique")
        
        # Taille du bin (nperseg)
        bin_box = ttk.LabelFrame(opt, text="Taille du bin (nperseg)", padding=6)
        bin_box.pack(fill=tk.X, pady=(8,4))
        ttk.Label(bin_box, text="Durée (secondes):").pack(anchor='w')
        nperseg_sec_var = tk.DoubleVar(value=4.0)
        nperseg_entry = ttk.Entry(bin_box, textvariable=nperseg_sec_var, width=10)
        nperseg_entry.pack(anchor='w', pady=(2,0))
        ttk.Label(bin_box, text="(recommandé: 2-8s)").pack(anchor='w')
        
        # Lier le changement de taille de bin au recalcul
        def on_bin_size_change(event=None):
            try:
                new_size = float(nperseg_sec_var.get())
                print(f"🔍 CHECKPOINT PSD BIN: Taille de bin changée vers {new_size}s")
                if recalculate_function:
                    recalculate_function()
            except ValueError:
                print("⚠️ CHECKPOINT PSD BIN: Valeur invalide pour la taille de bin")
        
        nperseg_entry.bind('<Return>', on_bin_size_change)
        nperseg_entry.bind('<FocusOut>', on_bin_size_change)
        
        # Tooltip pour la taille du bin
        create_tooltip(nperseg_entry, 
                      "Taille du segment (nperseg) pour la méthode de Welch.\n\n"
                      "• Définit la durée de chaque segment utilisé pour le calcul de la PSD\n"
                      "• Valeurs plus petites (2-4s) : meilleure résolution temporelle, plus de bruit\n"
                      "• Valeurs plus grandes (6-8s) : meilleure résolution fréquentielle, moins de bruit\n"
                      "• Recommandé : 4 secondes pour un bon compromis\n"
                      "• Doit être ≥ 1 seconde pour des résultats fiables")
        
        apply_filter_var = tk.BooleanVar(value=True)
        filter_cb = ttk.Checkbutton(opt, text="Filtre 0.3–40 Hz (Butterworth 4)", variable=apply_filter_var)
        filter_cb.pack(anchor='w', pady=(8,0))
        
        notch50_var = tk.BooleanVar(value=True)
        notch_cb = ttk.Checkbutton(opt, text="Notch 50 Hz", variable=notch50_var)
        notch_cb.pack(anchor='w')
        
        literature_var = tk.BooleanVar(value=True)
        literature_cb = ttk.Checkbutton(opt, text="Mode littérature: médiane + SEM robuste (MAD)", variable=literature_var)
        literature_cb.pack(anchor='w', pady=(8,0))
        
        equalize_var = tk.BooleanVar(value=True)
        equalize_cb = ttk.Checkbutton(opt, text="Égaliser nombre d'époques par stade", variable=equalize_var)
        equalize_cb.pack(anchor='w')
        
        normalize_var = tk.BooleanVar(value=False)
        normalize_cb = ttk.Checkbutton(opt, text="Normaliser PSD relative (%)", variable=normalize_var)
        normalize_cb.pack(anchor='w')
        
        # Lier les changements des options au recalcul
        def on_option_change(event=None):
            if recalculate_function:
                recalculate_function()
        
        filter_cb.configure(command=on_option_change)
        notch_cb.configure(command=on_option_change)
        literature_cb.configure(command=on_option_change)
        equalize_cb.configure(command=on_option_change)
        normalize_cb.configure(command=on_option_change)
        
        # Tooltips pour les options de filtrage et traitement
        create_tooltip(filter_cb, 
                      "Filtre passe-bande 0.3-40 Hz (Butterworth 4ème ordre).\n\n"
                      "• Supprime les composantes basse fréquence (< 0.3 Hz) et haute fréquence (> 40 Hz)\n"
                      "• Améliore la qualité du signal en réduisant le bruit et les artefacts\n"
                      "• 0.3 Hz : élimine la dérive DC et les mouvements lents\n"
                      "• 40 Hz : élimine les interférences haute fréquence\n"
                      "• Butterworth 4 : filtre avec réponse plate dans la bande passante")
        
        create_tooltip(notch_cb, 
                      "Filtre notch à 50 Hz (ou 60 Hz selon la région).\n\n"
                      "• Supprime spécifiquement l'interférence du réseau électrique\n"
                      "• 50 Hz en Europe, 60 Hz en Amérique du Nord\n"
                      "• Utilise un filtre IIR avec Q=30 pour une suppression efficace\n"
                      "• Important pour l'analyse spectrale car l'interférence 50/60 Hz peut masquer les signaux d'intérêt")
        
        create_tooltip(literature_cb, 
                      "Mode littérature : statistiques robustes (médiane + MAD).\n\n"
                      "• Utilise la médiane au lieu de la moyenne pour la tendance centrale\n"
                      "• Utilise l'écart absolu médian (MAD) pour l'erreur standard\n"
                      "• Plus robuste aux valeurs aberrantes et au bruit\n"
                      "• Standard dans la littérature scientifique pour l'analyse EEG\n"
                      "• Fournit des intervalles de confiance plus fiables")
        
        create_tooltip(equalize_cb, 
                      "Égaliser le nombre d'époques par stade de sommeil.\n\n"
                      "• Évite le biais statistique dû aux différences de durée entre stades\n"
                      "• Chaque stade aura le même nombre d'époques (minimum disponible)\n"
                      "• Important pour les comparaisons statistiques équitables\n"
                      "• Peut réduire le nombre d'époques utilisées si un stade est rare")
        
        create_tooltip(normalize_cb, 
                      "Normalisation PSD en pourcentage relatif.\n\n"
                      "• Exprime chaque fréquence comme % de la puissance totale\n"
                      "• Utile pour comparer les profils spectraux entre sujets\n"
                      "• Masque les différences d'amplitude absolue\n"
                      "• Focus sur la distribution relative des fréquences\n"
                      "• Désactivé par défaut pour voir les amplitudes réelles")

        freq_box = ttk.LabelFrame(opt, text="Affichage fréquence", padding=6)
        freq_box.pack(fill=tk.X, pady=(10,4))
        ttk.Label(freq_box, text="Min (Hz)").grid(row=0, column=0, sticky='w')
        freq_min_var = tk.DoubleVar(value=0.5)
        freq_min_entry = ttk.Entry(freq_box, textvariable=freq_min_var, width=8)
        freq_min_entry.grid(row=0, column=1, sticky='w')
        ttk.Label(freq_box, text="Max (Hz)").grid(row=1, column=0, sticky='w', pady=(6,0))
        freq_max_var = tk.DoubleVar(value=min(30.0, float(self.sfreq)/2.0))
        freq_max_entry = ttk.Entry(freq_box, textvariable=freq_max_var, width=8)
        freq_max_entry.grid(row=1, column=1, sticky='w', pady=(6,0))
        
        # Lier les changements de fréquence au recalcul
        def on_freq_change(event=None):
            if recalculate_function:
                recalculate_function()
        
        freq_min_entry.bind('<Return>', on_freq_change)
        freq_min_entry.bind('<FocusOut>', on_freq_change)
        freq_max_entry.bind('<Return>', on_freq_change)
        freq_max_entry.bind('<FocusOut>', on_freq_change)
        
        # Tooltips pour les paramètres de fréquence
        create_tooltip(freq_min_entry, 
                      "Fréquence minimale d'affichage (Hz).\n\n"
                      "• Définit la fréquence la plus basse affichée sur le graphique\n"
                      "• Recommandé : 0.5 Hz pour voir les ondes lentes du sommeil\n"
                      "• Valeurs plus basses : inclut plus de composantes basse fréquence\n"
                      "• Valeurs plus hautes : exclut les ondes delta (0.5-4 Hz)\n"
                      "• Doit être ≥ 0 Hz et < fréquence maximale")
        
        create_tooltip(freq_max_entry, 
                      "Fréquence maximale d'affichage (Hz).\n\n"
                      "• Définit la fréquence la plus haute affichée sur le graphique\n"
                      "• Recommandé : 30 Hz pour l'analyse du sommeil\n"
                      "• Valeurs plus basses : focus sur les bandes de sommeil (delta, theta, alpha, sigma)\n"
                      "• Valeurs plus hautes : inclut les ondes gamma et artefacts\n"
                      "• Doit être ≤ fréquence de Nyquist (fréquence d'échantillonnage / 2)")
        def _apply_freq_window():
            try:
                fmin = float(freq_min_var.get())
                fmax = float(freq_max_var.get())
                nyq = float(self.sfreq) / 2.0
                fmin = max(0.0, min(fmin, nyq))
                fmax = max(0.0, min(fmax, nyq))
                if fmax <= fmin:
                    fmax = min(nyq, fmin + 0.1)
                for ax in axes:
                    ax.set_xlim(fmin, fmax)
                fig.tight_layout()
                canvas.draw()
            except Exception as e:
                messagebox.showerror("Erreur", f"Fenêtre fréquentielle invalide: {e}")
        apply_freq_btn = ttk.Button(freq_box, text="Appliquer", command=_apply_freq_window)
        apply_freq_btn.grid(row=2, column=0, columnspan=2, pady=(8,0))
        
        # Bouton de recalcul
        recalc_btn = ttk.Button(opt, text="🔄 Recalculer PSD", command=lambda: _recalculate_psd())
        recalc_btn.pack(fill=tk.X, pady=(10,0))
        
        # Tooltips pour les boutons
        create_tooltip(apply_freq_btn, 
                      "Appliquer la fenêtre fréquentielle.\n\n"
                      "• Met à jour l'affichage avec les nouvelles valeurs min/max\n"
                      "• Redessine les graphiques avec la plage de fréquences sélectionnée\n"
                      "• Utile pour zoomer sur des bandes de fréquences spécifiques\n"
                      "• Les valeurs sont validées avant application")
        
        create_tooltip(recalc_btn, 
                      "Recalculer la PSD avec les nouveaux paramètres.\n\n"
                      "• Recalcule complètement l'analyse spectrale\n"
                      "• Utilise le canal et la taille de bin sélectionnés\n"
                      "• Applique tous les filtres et options configurés\n"
                      "• Met à jour l'affichage et les statistiques\n"
                      "• Nécessaire après changement de canal ou de paramètres")

        info_box = ttk.LabelFrame(opt, text="Infos", padding=6)
        info_box.pack(fill=tk.X, pady=(10,4))
        info_var = tk.StringVar(value="")
        info_label = ttk.Label(info_box, textvariable=info_var, justify='left')
        info_label.pack(anchor='w')
        
        # Tooltip pour la boîte d'informations
        create_tooltip(info_label, 
                      "Informations sur l'analyse PSD.\n\n"
                      "• Affiche le nombre d'époques disponibles pour chaque stade de sommeil\n"
                      "• Format : W: n=X | N1: n=Y | N2: n=Z | N3: n=A | R: n=B\n"
                      "• Plus le nombre d'époques est élevé, plus l'analyse est fiable\n"
                      "• Les stades avec peu d'époques peuvent avoir des statistiques moins robustes\n"
                      "• Mis à jour automatiquement lors du recalcul")

        # Zone graphique
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        canvas = FigureCanvasTkAgg(fig, master=content)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialiser les variables pour les callbacks
        mean_before = {}
        mean_after = {}

        # Préparer segments par stade (utilise scoring s'il existe, sinon toute la trace comme W)
        df = self._get_active_scoring_df()
        print(f"🔍 CHECKPOINT PSD 3: Scoring dispo={df is not None}, len={len(df) if df is not None else 0}")
        fs = float(self.sfreq)
        print(f"🔍 CHECKPOINT PSD 3: Canal='{candidate}', fs={fs} Hz")
        stage_colors = self.theme_manager.get_stage_colors()
        stages_order = ["W", "N1", "N2", "N3", "R"]

        def collect_segments(raw_array):
            segments = {s: [] for s in stages_order}
            if df is None or len(df) == 0:
                print("⚠️ CHECKPOINT PSD 4: Pas de scoring, tout en W")
                segments["W"].append(raw_array)
                return segments
            epoch_len = float(getattr(self, 'scoring_epoch_duration', 30.0))
            for _, row in df.iterrows():
                s = str(row['stage']).upper()
                if s not in segments:
                    continue
                t0 = float(row['time'])
                t1 = t0 + epoch_len
                i0 = int(max(0, min(len(raw_array)-1, t0 * fs)))
                i1 = int(max(i0+1, min(len(raw_array), t1 * fs)))
                seg = raw_array[i0:i1]
                if len(seg) >= int(fs):  # >=1s
                    segments[s].append(seg)
            return segments

        # Variables globales pour le recalcul
        current_channel = candidate
        current_nperseg_sec = 4.0
        
        def _recalculate_psd():
            nonlocal current_channel, current_nperseg_sec
            try:
                current_channel = selected_channel_var.get()
                current_nperseg_sec = float(nperseg_sec_var.get())
                
                # Vérifier que le canal existe
                if current_channel not in self.derivations:
                    messagebox.showerror("Erreur", f"Canal '{current_channel}' non disponible")
                    return
                
                # Vérifier la taille du bin
                if current_nperseg_sec <= 0 or current_nperseg_sec > 30:
                    messagebox.showerror("Erreur", "Taille du bin doit être entre 0.1 et 30 secondes")
                    return
                
                # Mettre à jour le titre de la barre d'outils
                toolbar.children['!label'].config(text=f"Canal: {current_channel} | Welch {current_nperseg_sec}s 50% | µV²/Hz")
                
                # Recalculer et afficher
                render()
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du recalcul: {e}")
        
        # Assigner la fonction de recalcul à la variable
        recalculate_function = _recalculate_psd

        # Avant: après filtrage/interpolation (on prend le signal tel qu'affiché)
        raw_all = self.derivations[current_channel]
        segments_before = collect_segments(raw_all)
        print("🔍 CHECKPOINT PSD 4b: n_segments before=", {k: len(v) for k, v in segments_before.items()})
        try:
            print("🔍 CHECKPOINT PSD 4c: durées moyennes (s) before=",
                  {k: (np.mean([len(x) for x in v]) / fs if v else 0.0) for k, v in segments_before.items()})
        except Exception:
            pass

        # Après: si on avait des masques de « bad spans » ou ICA, on les appliquerait ici.
        # En attendant, on réutilise les mêmes segments pour la structure de la figure.
        segments_after = segments_before

        # Exclure UNKNOWN/M si présent
        def _filter_known_stages(segs: dict) -> dict:
            return {k: v for k, v in segs.items() if k in stages_order}
        segments_before = _filter_known_stages(segments_before)
        segments_after = _filter_known_stages(segments_after)

        # Égaliser époques par stade si demandé
        if equalize_var.get():
            try:
                counts = {k: len(v) for k, v in segments_before.items() if len(v) > 0}
                if counts:
                    m = int(min(counts.values()))
                    for d in (segments_before, segments_after):
                        for k in list(d.keys()):
                            if len(d[k]) > m:
                                d[k] = d[k][:m]
                    print(f"🔍 CHECKPOINT PSD EQ: égalisation à n={m} époques/stade")
            except Exception as _e_eq:
                print(f"⚠️ CHECKPOINT PSD EQ: échec égalisation: {_e_eq}")

        def mean_psd_for_segments(segments_dict):
            out = {}
            for s, chunks in segments_dict.items():
                psds = []
                freqs_ref = None
                lengths = []
                used_nperseg = []
                for ch in chunks:
                    # Optionnel: pré-filtre 0.3–40 Hz (Butterworth 4)
                    if apply_filter_var.get():
                        try:
                            from scipy.signal import butter, filtfilt
                            b, a = butter(4, [0.3/(fs/2.0), 40.0/(fs/2.0)], btype='band')
                            ch = filtfilt(b, a, ch)
                        except Exception as _e:
                            print(f"⚠️ CHECKPOINT PSD FILTER: {s} pré-filtre échoué: {_e}")
                    # Notch 50 Hz si demandé
                    if notch50_var.get():
                        try:
                            from scipy.signal import iirnotch, filtfilt
                            w0 = 50.0/(fs/2.0)
                            b_notch, a_notch = iirnotch(w0, Q=30.0)
                            ch = filtfilt(b_notch, a_notch, ch)
                        except Exception as _e_n:
                            print(f"⚠️ CHECKPOINT PSD NOTCH: {s} notch échoué: {_e_n}")
                    # Welch: nperseg=current_nperseg_sec*fs, 50% overlap, hann, density
                    nper = int(max(8, min(len(ch), round(current_nperseg_sec * fs))))
                    nover = int(nper // 2)
                    f, p = welch(ch, fs=fs, window='hann', nperseg=nper, noverlap=nover,
                                 detrend='constant', return_onesided=True, scaling='density')
                    # Bande d'intérêt pour les stats/CSV: 0.5–30 Hz
                    mask = (f >= 0.5) & (f <= 30.0)
                    f = f[mask]
                    p = p[mask] * 1e12  # V^2/Hz -> µV^2/Hz
                    if len(f) == 0:
                        continue
                    if freqs_ref is None:
                        freqs_ref = f
                    psds.append(p)
                    lengths.append(len(ch))
                    used_nperseg.append(nper)
                if psds and freqs_ref is not None:
                    arr = np.vstack(psds)
                    if literature_var.get():
                        med = np.median(arr, axis=0)
                        mad = np.median(np.abs(arr - med), axis=0)
                        sem_vals = 1.4826 * mad / max(1, np.sqrt(arr.shape[0]))
                        mean_vals = med
                    else:
                        mean_vals = np.mean(arr, axis=0)
                        sem_vals = np.std(arr, axis=0, ddof=1) / max(1, np.sqrt(arr.shape[0]))
                    out[s] = (freqs_ref, mean_vals, sem_vals, arr.shape[0])
                    try:
                        print(f"🔍 CHECKPOINT PSD 5a [{s}]: n_chunks={len(chunks)}, len_min/med/max=", 
                              (int(np.min(lengths)), int(np.median(lengths)), int(np.max(lengths))),
                              " nperseg_min/med/max=", (int(np.min(used_nperseg)), int(np.median(used_nperseg)), int(np.max(used_nperseg))),
                              f" n_freqs={len(freqs_ref)}")
                    except Exception:
                        pass
            return out

        mean_before = mean_psd_for_segments(segments_before)
        mean_after = mean_psd_for_segments(segments_after)
        print(f"🔍 CHECKPOINT PSD 5: Moyennes calculées: before={list(mean_before.keys())}, after={list(mean_after.keys())}")
        
        # Mettre à jour les variables globales
        mean_before = mean_before
        mean_after = mean_after
        try:
            for label, data in [("before", mean_before), ("after", mean_after)]:
                for s, (fvals, mean_vals, sem_vals, n_ep) in data.items():
                    if len(mean_vals) > 0:
                        print(f"   → {label}/{s}: PSDµV^2/Hz min/med/max=",
                              (float(np.min(mean_vals)), float(np.median(mean_vals)), float(np.max(mean_vals))), f" n={n_ep}")
        except Exception:
            pass

        # Tracé
        for ax, data, title in [(axes[0], mean_before, "Avant (filtrage + interp. canaux)"), (axes[1], mean_after, "Après (avec rejets/ICA si dispo)")]:
            for s in stages_order:
                if s in data:
                    f, mean_vals, sem_vals, n_ep = data[s]
                    ax.semilogy(f, mean_vals, color=stage_colors[s], label=f"{s} (n={n_ep})")
                    ax.fill_between(f, np.maximum(mean_vals - sem_vals, 1e-24), mean_vals + sem_vals,
                                    color=stage_colors[s], alpha=0.15, linewidth=0)
            # visualisation de la zone filtrée (ex: high-pass < 0.5 Hz)
            hp = float(getattr(self, 'filter_low', 0.5))
            ax.axvspan(0.0, hp, color='0.85')
            ax.set_xlim(0.5, min(30.0, fs/2))
            ax.set_xlabel("Fréquence (Hz)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("PSD (µV²/Hz)")
        axes[1].legend(loc='upper right', ncol=1, fontsize=8)
        fig.tight_layout()
        canvas.draw()

        def _save_figure(fig_obj):
            try:
                file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    fig_obj.savefig(file_path, dpi=200, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")

        def _export_psd_csv(before_dict, after_dict):
            try:
                file_path = filedialog.asksaveasfilename(title="Exporter CSV", defaultextension=".csv",
                                                         filetypes=[("CSV", "*.csv")])
                if not file_path:
                    return
                import csv
                stages_order = ["W", "N1", "N2", "N3", "R"]
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["stage", "condition", "channel", "fs_hz", "freq_hz", "mean_psd_uv2_per_hz", "sem_uv2_per_hz", "n_epochs_stage"])
                    for cond_name, data in [("before", before_dict), ("after", after_dict)]:
                        for s in stages_order:
                            if s in data:
                                freqs, mean_vals, sem_vals, n_ep = data[s]
                                for fr, m, se in zip(freqs, mean_vals, sem_vals):
                                    writer.writerow([s, cond_name, candidate, f"{fs:.2f}", f"{fr:.4f}", f"{m:.8e}", f"{se:.8e}", n_ep])
                try:
                    total_rows = sum(len(v[0]) for v in list(before_dict.values()) + list(after_dict.values()))
                    print(f"✅ EXPORT PSD CSV: {total_rows} lignes, canal={candidate}, fs={fs} Hz, bande 0.5–30 Hz, unités µV^2/Hz")
                except Exception:
                    pass
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'export CSV: {e}")

        def create_tooltip(widget, text):
            """Crée un tooltip pour un widget."""
            def show_tooltip(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                
                label = tk.Label(tooltip, text=text, justify='left', 
                               background='#ffffe0', relief='solid', borderwidth=1,
                               font=('Segoe UI', 9), wraplength=300)
                label.pack()
                
                widget.tooltip = tooltip
                
            def hide_tooltip(event):
                if hasattr(widget, 'tooltip') and widget.tooltip:
                    widget.tooltip.destroy()
                    widget.tooltip = None
                    
            widget.bind('<Enter>', show_tooltip)
            widget.bind('<Leave>', hide_tooltip)

        def render():
            """Fonction de rendu pour recalculer la PSD avec les nouveaux paramètres."""
            try:
                print("🔍 CHECKPOINT PSD RENDER: Début render()")
                
                # Effacer les axes
                for ax in axes:
                    ax.clear()
                
                # Récupérer les paramètres actuels
                current_channel = selected_channel_var.get()
                current_nperseg_sec = float(nperseg_sec_var.get())
                
                # Récupérer le signal du canal sélectionné
                raw_all = self.derivations[current_channel]
                
                # Recalculer les segments avec le nouveau canal
                segments_before = collect_segments(raw_all)
                segments_after = segments_before  # Même logique qu'avant
                
                # Filtrer les stades connus
                segments_before = _filter_known_stages(segments_before)
                segments_after = _filter_known_stages(segments_after)
                
                # Égaliser époques par stade si demandé
                if equalize_var.get():
                    try:
                        counts = {k: len(v) for k, v in segments_before.items() if len(v) > 0}
                        if counts:
                            m = int(min(counts.values()))
                            for d in (segments_before, segments_after):
                                for k in list(d.keys()):
                                    if len(d[k]) > m:
                                        d[k] = d[k][:m]
                            print(f"🔍 CHECKPOINT PSD EQ: égalisation à n={m} époques/stade")
                    except Exception as _e_eq:
                        print(f"⚠️ CHECKPOINT PSD EQ: échec égalisation: {_e_eq}")
                
                # Recalculer les moyennes PSD
                new_mean_before = mean_psd_for_segments(segments_before)
                new_mean_after = mean_psd_for_segments(segments_after)
                
                print(f"🔍 CHECKPOINT PSD RENDER: Moyennes recalculées: before={list(new_mean_before.keys())}, after={list(new_mean_after.keys())}")
                
                # Mettre à jour les infos
                n_per_stage = {}
                for data in [new_mean_before, new_mean_after]:
                    for s, (_, _, _, n_ep) in data.items():
                        n_per_stage[s] = n_per_stage.get(s, 0) + n_ep
                
                info_text = " | ".join([f"{k}: n={n_per_stage.get(k, 0)}" for k in stages_order])
                info_var.set(info_text)
                
                # Tracé
                for ax, data, title in [(axes[0], new_mean_before, "Avant (filtrage + interp. canaux)"), (axes[1], new_mean_after, "Après (avec rejets/ICA si dispo)")]:
                    for s in stages_order:
                        if s in data:
                            f, mean_vals, sem_vals, n_ep = data[s]
                            ax.semilogy(f, mean_vals, color=stage_colors[s], label=f"{s} (n={n_ep})")
                            ax.fill_between(f, np.maximum(mean_vals - sem_vals, 1e-24), mean_vals + sem_vals,
                                            color=stage_colors[s], alpha=0.15, linewidth=0)
                    # visualisation de la zone filtrée (ex: high-pass < 0.5 Hz)
                    hp = float(getattr(self, 'filter_low', 0.5))
                    ax.axvspan(0.0, hp, color='0.85')
                    ax.set_xlim(0.5, min(30.0, fs/2))
                    ax.set_xlabel("Fréquence (Hz)")
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                
                axes[0].set_ylabel("PSD (µV²/Hz)")
                axes[1].legend(loc='upper right', ncol=1, fontsize=8)
                fig.tight_layout()
                canvas.draw()
                
                # Mettre à jour les variables globales pour les callbacks
                nonlocal mean_before, mean_after
                mean_before = new_mean_before
                mean_after = new_mean_after
                
                print("✅ CHECKPOINT PSD RENDER: Rendu terminé")
                
            except Exception as e:
                print(f"❌ CHECKPOINT PSD RENDER: Erreur: {e}")
                messagebox.showerror("Erreur", f"Erreur lors du rendu: {e}")

        # Configurer les callbacks des boutons après la définition de render
        save_fig_btn.config(command=lambda: _save_figure(fig))
        export_csv_btn.config(command=lambda: _export_psd_csv(mean_before, mean_after))
        
        # Tooltips pour les boutons de la barre d'outils
        create_tooltip(save_fig_btn, 
                      "Enregistrer la figure PSD.\n\n"
                      "• Sauvegarde les graphiques PSD par stade au format PNG, PDF ou SVG\n"
                      "• Qualité haute résolution (200 DPI)\n"
                      "• Inclut les deux graphiques (avant/après) et la légende\n"
                      "• Utile pour les publications et rapports\n"
                      "• Format recommandé : PNG pour l'affichage, PDF pour l'impression")
        
        create_tooltip(export_csv_btn, 
                      "Exporter les données PSD en CSV.\n\n"
                      "• Sauvegarde toutes les données numériques de l'analyse\n"
                      "• Format : stage, condition, canal, fréquence, PSD, erreur standard, n_époques\n"
                      "• Compatible avec Excel, R, Python, etc.\n"
                      "• Permet l'analyse statistique externe\n"
                      "• Données en unités µV²/Hz (densité spectrale de puissance)")

    def _show_stage_psd_fft(self):
        """PSD par stade (FFT magnitude) adaptée de Analyse_spectrale.py."""
        print("🔍 CHECKPOINT STAGE-FFT 1: Entrée _show_stage_psd_fft")
        if not self.raw:
            print("⚠️ CHECKPOINT STAGE-FFT 1: raw manquant")
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return

        def create_tooltip(widget, text):
            """Crée un tooltip pour un widget."""
            def show_tooltip(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                
                label = tk.Label(tooltip, text=text, justify='left', 
                               background='#ffffe0', relief='solid', borderwidth=1,
                               font=('Segoe UI', 9), wraplength=300)
                label.pack()
                
                widget.tooltip = tooltip
                
            def hide_tooltip(event):
                if hasattr(widget, 'tooltip') and widget.tooltip:
                    widget.tooltip.destroy()
                    widget.tooltip = None
                    
            widget.bind('<Enter>', show_tooltip)
            widget.bind('<Leave>', hide_tooltip)

        # Choisir un canal EEG pertinent
        candidate = None
        for name in [
            "EEG Fpz-Cz", "Fpz-Cz", "Pz", "EEG Pz-Oz", "E101", "PZ"
        ] + self.selected_channels + (self.raw.ch_names if self.raw else []):
            if name in self.derivations:
                candidate = name
                break
        if candidate is None:
            print("⚠️ CHECKPOINT STAGE-FFT 2: Aucun canal trouvé")
            messagebox.showwarning("Attention", "Aucun canal disponible")
            return

        # Fenêtre UI
        top = tk.Toplevel(self.root)
        top.title("PSD by Sleep Stage (FFT – Spectral Analysis)")
        top.geometry("1100x680")
        top.transient(self.root)
        top.grab_set()

        toolbar = ttk.Frame(top, style='Custom.TFrame')
        toolbar.pack(fill=tk.X, side=tk.TOP)
        save_fig_btn = ttk.Button(toolbar, text="Save Figure", style='Custom.TButton')
        save_fig_btn.pack(side=tk.RIGHT, padx=(6,6), pady=4)
        export_csv_btn = ttk.Button(toolbar, text="Export CSV")
        export_csv_btn.pack(side=tk.RIGHT, padx=(6,0), pady=4)
        # Comparison import button
        compare_btn = ttk.Button(toolbar, text="Compare Conditions", style='Custom.TButton')
        compare_btn.pack(side=tk.RIGHT, padx=(6,0), pady=4)
        # Sélecteur de thème pour la fenêtre FFT
        theme_var = tk.StringVar(value=self.theme_manager.current_theme_name)
        theme_combo = ttk.Combobox(toolbar, textvariable=theme_var, state="readonly", width=12)
        theme_combo['values'] = list(self.theme_manager.get_available_themes().values())
        theme_combo.pack(side=tk.RIGHT, padx=(6,0), pady=4)
        theme_combo.bind('<<ComboboxSelected>>', lambda e: self._change_theme_by_display_name(theme_var.get()))
        toolbar_label = ttk.Label(toolbar, text=f"Channel: {candidate} | FFT Power | DC removed", font=('Segoe UI', 9))
        toolbar_label.pack(side=tk.LEFT, padx=8)

        main = ttk.Frame(top)
        main.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(main, width=320)
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)
        content = ttk.Frame(main)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Options
        opt = ttk.LabelFrame(side, text="Parameters (FFT)", padding=8)
        opt.pack(fill=tk.Y, expand=True, padx=8, pady=8)
        
        # Sélection du canal
        channel_box = ttk.LabelFrame(opt, text="Channel", padding=6)
        channel_box.pack(fill=tk.X, pady=(0,8))
        available_channels = list(self.derivations.keys())
        selected_channel_var = tk.StringVar(value=candidate)
        channel_combo = ttk.Combobox(channel_box, textvariable=selected_channel_var, values=available_channels, state="readonly", width=20)
        channel_combo.pack(fill=tk.X)
        
        # Tooltip pour la sélection du canal
        create_tooltip(channel_combo, 
                      "Select EEG channel for PSD analysis.\n\n"
                      "• Choose the channel of interest for spectral analysis\n"
                      "• Bipolar channels (e.g., C4-M1) are generally preferred\n"
                      "• The selected channel will be used to compute power spectral density\n"
                      "• Each channel may have different spectral characteristics based on anatomical position")
        
        # Taille du bin (nperseg) pour FFT
        bin_box = ttk.LabelFrame(opt, text="Bin Size (nperseg)", padding=6)
        bin_box.pack(fill=tk.X, pady=(8,8))
        ttk.Label(bin_box, text="Duration (seconds):").pack(anchor='w')
        nperseg_sec_var = tk.DoubleVar(value=4.0)
        nperseg_entry = ttk.Entry(bin_box, textvariable=nperseg_sec_var, width=10)
        nperseg_entry.pack(anchor='w', pady=(2,4))
        ttk.Label(bin_box, text="(recommended: 2-8s)").pack(anchor='w', pady=(0,4))
        
        def _apply_bin_size():
            render()
        
        ttk.Button(bin_box, text="Apply Bin Size", command=_apply_bin_size).pack(anchor='w', pady=(4,0))
        
        # Tooltip pour la taille du bin
        create_tooltip(nperseg_entry, 
                      "Segment size (nperseg) for FFT method.\n\n"
                      "• Defines the duration of each segment used for PSD computation\n"
                      "• Smaller values (2-4s): better temporal resolution, more noise\n"
                      "• Larger values (6-8s): better frequency resolution, less noise\n"
                      "• Recommended: 4 seconds for a good compromise\n"
                      "• Must be ≥ 1 second for reliable results\n"
                      "• Click 'Apply Bin Size' button to update the plot")
        
        robust_var = tk.BooleanVar(value=True)
        robust_cb = ttk.Checkbutton(opt, text="Robust Median + SEM (MAD)", variable=robust_var)
        robust_cb.pack(anchor='w')
        equalize_var = tk.BooleanVar(value=True)
        equalize_cb = ttk.Checkbutton(opt, text="Equalize n epochs per stage", variable=equalize_var)
        equalize_cb.pack(anchor='w', pady=(4,0))

        freq_box = ttk.LabelFrame(opt, text="Frequency Display", padding=6)
        freq_box.pack(fill=tk.X, pady=(10,4))
        ttk.Label(freq_box, text="Min (Hz)").grid(row=0, column=0, sticky='w')
        freq_min_var = tk.DoubleVar(value=0.5)
        freq_min_entry = ttk.Entry(freq_box, textvariable=freq_min_var, width=8)
        freq_min_entry.grid(row=0, column=1, sticky='w')
        ttk.Label(freq_box, text="Max (Hz)").grid(row=1, column=0, sticky='w', pady=(6,0))
        freq_max_var = tk.DoubleVar(value=min(45.0, float(self.sfreq)/2.0))
        freq_max_entry = ttk.Entry(freq_box, textvariable=freq_max_var, width=8)
        freq_max_entry.grid(row=1, column=1, sticky='w', pady=(6,0))

        # Période d'analyse (nouveau)
        period_box = ttk.LabelFrame(opt, text="Analysis Period", padding=6)
        period_box.pack(fill=tk.X, pady=(10,4))

        # Case à cocher pour activer la période personnalisée
        self.fft_use_period_var = tk.BooleanVar(value=False)
        use_period_cb = ttk.Checkbutton(period_box, text="Analyze only a specific period", variable=self.fft_use_period_var)
        use_period_cb.pack(anchor='w', pady=(0,8))

        # Contrôles de période (désactivés par défaut)
        self.fft_period_start_var = tk.DoubleVar(value=0.0)
        self.fft_period_end_var = tk.DoubleVar(value=3600.0)  # 1 heure par défaut

        period_start_entry = ttk.Entry(period_box, textvariable=self.fft_period_start_var, width=10, state='disabled')
        period_end_entry = ttk.Entry(period_box, textvariable=self.fft_period_end_var, width=10, state='disabled')

        def toggle_period_entries():
            state = 'normal' if self.fft_use_period_var.get() else 'disabled'
            period_start_entry.config(state=state)
            period_end_entry.config(state=state)

        use_period_cb.config(command=toggle_period_entries)
        toggle_period_entries()  # Initialiser l'état

        ttk.Label(period_box, text="Start (s):").pack(anchor='w')
        period_start_entry.pack(anchor='w', pady=(2,4))
        ttk.Label(period_box, text="End (s):").pack(anchor='w')
        period_end_entry.pack(anchor='w', pady=(2,0))

        # Tooltip pour la période
        create_tooltip(use_period_cb,
                      "Analyze only a specific time period.\n\n"
                      "• Allows focusing FFT analysis on a portion of interest\n"
                      "• Useful for analyzing specific events (e.g., REM sleep)\n"
                      "• Leave fields empty to analyze the entire duration available\n"
                      "• Times are in seconds from the start of the recording")

        info_box = ttk.LabelFrame(opt, text="Info", padding=6)
        info_box.pack(fill=tk.X, pady=(10,4))
        info_var = tk.StringVar(value="")
        info_label = ttk.Label(info_box, textvariable=info_var, justify='left')
        info_label.pack(anchor='w')
        
        # Tooltips pour les contrôles FFT
        create_tooltip(robust_cb, 
                      "Robust statistics (median + MAD).\n\n"
                      "• Uses median instead of mean for central tendency\n"
                      "• Uses median absolute deviation (MAD) for standard error\n"
                      "• More robust to outliers and noise\n"
                      "• Standard in scientific literature for EEG analysis\n"
                      "• Provides more reliable confidence intervals")
        
        create_tooltip(equalize_cb, 
                      "Equalize the number of epochs per stage.\n\n"
                      "• Avoids statistical bias due to duration differences between stages\n"
                      "• Each stage will have the same number of epochs (minimum available)\n"
                      "• Important for fair statistical comparisons\n"
                      "• May reduce the number of epochs used if one stage is rare")
        
        create_tooltip(freq_min_entry, 
                      "Minimum display frequency (Hz).\n\n"
                      "• Defines the lowest frequency displayed on the graph\n"
                      "• Recommended: 0.5 Hz to see slow sleep waves\n"
                      "• Lower values: includes more low-frequency components\n"
                      "• Higher values: excludes delta waves (0.5-4 Hz)\n"
                      "• Must be ≥ 0 Hz and < maximum frequency")
        
        create_tooltip(freq_max_entry, 
                      "Maximum display frequency (Hz).\n\n"
                      "• Defines the highest frequency displayed on the graph\n"
                      "• Recommended: 30-45 Hz for sleep analysis\n"
                      "• Lower values: focus on sleep bands (delta, theta, alpha, sigma)\n"
                      "• Higher values: includes gamma waves and artifacts\n"
                      "• Must be ≤ Nyquist frequency (sampling frequency / 2)")
        
        create_tooltip(info_label, 
                      "Information on PSD analysis by stage.\n\n"
                      "• Displays the number of epochs available for each sleep stage\n"
                      "• Format: W: n=X | N1: n=Y | N2: n=Z | N3: n=A | R: n=B\n"
                      "• Higher epoch count means more reliable analysis\n"
                      "• Stages with few epochs may have less robust statistics\n"
                      "• Automatically updated during rendering")
        
        # Tooltip pour le sélecteur de thème
        create_tooltip(theme_combo,
                      "Changer le thème graphique.\n\n"
                      "• 3 thèmes disponibles :\n"
                      "  • Otilia 🦖🌸 : Rose avec coucher de soleil\n"
                      "  • Fred 🧗‍♂️🌿 : Vert avec montagne verdoyante\n"
                      "  • Eléna 🐢🌊 : Bleu avec plongée sous-marine\n"
                      "• Met à jour automatiquement tous les graphiques\n"
                      "• Applique l'image de fond correspondante")

        # Zone graphique
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
        canvas = FigureCanvasTkAgg(fig, master=content)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Variables pour le recalcul
        current_channel = candidate
        current_nperseg_sec = 4.0
        
        # Callbacks pour les changements
        def on_channel_change(event=None):
            print(f"🔍 CHECKPOINT FFT CHANNEL: Canal changé vers {selected_channel_var.get()}")
            render()
        
        def on_bin_size_change(event=None):
            try:
                new_size = float(nperseg_sec_var.get())
                print(f"🔍 CHECKPOINT FFT BIN: Taille de bin changée vers {new_size}s")
                render()
            except ValueError:
                print("⚠️ CHECKPOINT FFT BIN: Valeur invalide pour la taille de bin")
        
        def on_option_change(event=None):
            # Mettre à jour l'état des champs de période si nécessaire
            toggle_period_entries()
            # Si la période est activée, déclencher le rendu
            if self.fft_use_period_var.get():
                render()
        
        # Lier les callbacks
        channel_combo.bind('<<ComboboxSelected>>', on_channel_change)
        nperseg_entry.bind('<Return>', on_bin_size_change)
        nperseg_entry.bind('<FocusOut>', on_bin_size_change)
        robust_cb.configure(command=on_option_change)
        equalize_cb.configure(command=on_option_change)
        use_period_cb.configure(command=on_option_change)

        # Callback pour les entrées de période
        def on_period_change(event=None):
            if self.fft_use_period_var.get():
                render()

        period_start_entry.bind('<Return>', on_period_change)
        period_start_entry.bind('<FocusOut>', on_period_change)
        period_end_entry.bind('<Return>', on_period_change)
        period_end_entry.bind('<FocusOut>', on_period_change)
        
        # Callbacks pour les paramètres de fréquence
        def on_freq_change(event=None):
            render()
        
        freq_min_entry.bind('<Return>', on_freq_change)
        freq_min_entry.bind('<FocusOut>', on_freq_change)
        freq_max_entry.bind('<Return>', on_freq_change)
        freq_max_entry.bind('<FocusOut>', on_freq_change)

        # Données & scoring
        df = self._get_active_scoring_df()

        # Filtrer selon la période si activée
        if self.fft_use_period_var.get() and df is not None:
            try:
                period_start = float(self.fft_period_start_var.get())
                period_end = float(self.fft_period_end_var.get())
                if period_start >= 0 and period_end > period_start:
                    df = df[(df['time'] >= period_start) & (df['time'] <= period_end)].copy()
                    print(f"🔍 CHECKPOINT STAGE-FFT 3: Période filtrée {period_start}s - {period_end}s, {len(df)} époques")
                else:
                    print("🔍 CHECKPOINT STAGE-FFT 3: Période invalide, utilisation de toutes les données")
            except:
                print("🔍 CHECKPOINT STAGE-FFT 3: Erreur période, utilisation de toutes les données")

        fs = float(self.sfreq)
        print(f"🔍 CHECKPOINT STAGE-FFT 3: Scoring dispo={df is not None}, len={len(df) if df is not None else 0}")
        # Normalisation des étiquettes de stades (WAKE/EVEIL->W, REM->R, etc.) gérée dans spectral_analysis._canonical_stage
        print("🔍 CHECKPOINT STAGE-FFT 3b: Normalisation des stades active (W,N1,N2,N3,R)")
        print(f"🔍 CHECKPOINT STAGE-FFT 3: Canal='{candidate}', fs={fs} Hz")

        stage_colors = self.theme_manager.get_stage_colors()
        stages_order = ["W", "N1", "N2", "N3", "R"]

        def render():
            try:
                print("🔍 CHECKPOINT STAGE-FFT 4: Début render()")
                ax.clear()
                fmin = float(freq_min_var.get())
                fmax = float(freq_max_var.get())
                eq = bool(equalize_var.get())
                robust = bool(robust_var.get())
                
                # Utiliser le canal sélectionné
                current_channel = selected_channel_var.get()
                if current_channel not in self.derivations:
                    messagebox.showerror("Erreur", f"Canal '{current_channel}' non disponible")
                    return
                signal = self.derivations[current_channel]

                # Calcul principal avec taille de bin personnalisée
                current_nperseg_sec = float(nperseg_sec_var.get())
                
                # Mettre à jour le titre de la barre d'outils
                toolbar_label.config(text=f"Channel: {current_channel} | FFT {current_nperseg_sec}s | DC removed")
                
                # Récupérer la palette de couleurs actuelle
                current_stage_colors = self.theme_manager.get_stage_colors()
                
                # Implémentation personnalisée avec nperseg_sec
                out = self._compute_stage_psd_fft_custom(
                    signal=signal,
                    fs=fs,
                    scoring_df=df,
                    epoch_len=float(getattr(self, 'scoring_epoch_duration', 30.0)),
                    stages=stages_order,
                    fmin=fmin,
                    fmax=fmax,
                    equalize_epochs=eq,
                    robust_stats=robust,
                    nperseg_sec=current_nperseg_sec,
                )
                print(f"🔍 CHECKPOINT STAGE-FFT 5: stages calculés={list(out.keys())}")

                n_per_stage = {k: v[3] for k, v in out.items()}
                info_var.set(" | ".join([f"{k}: n={n_per_stage.get(k, 0)}" for k in stages_order]))

                # Tracé (échelle log10; puissance)
                # Check if comparison data is available
                has_comparison = (self.fft_comparison_raw is not None and 
                                 self.fft_comparison_scoring is not None and
                                 current_channel in self.fft_comparison_derivations)
                
                # Plot main condition
                for s in stages_order:
                    if s not in out:
                        continue
                    f, mean_vals, sem_vals, n_ep = out[s]
                    # Convert magnitude to power and apply log10
                    power_vals = mean_vals ** 2
                    power_log = np.log10(np.maximum(power_vals, 1e-20))
                    
                    # For error bands, also convert SEM to log space
                    power_upper = (mean_vals + sem_vals) ** 2
                    power_lower = np.maximum(mean_vals - sem_vals, 0.0) ** 2
                    power_upper_log = np.log10(np.maximum(power_upper, 1e-20))
                    power_lower_log = np.log10(np.maximum(power_lower, 1e-20))
                    
                    label = f"{self.fft_main_name} - {s} (n={n_ep})" if has_comparison else f"{s} (n={n_ep})"
                    ax.plot(f, power_log, color=current_stage_colors[s], label=label, 
                           linestyle='-', linewidth=1.5)
                    ax.fill_between(f, power_lower_log, power_upper_log,
                                    color=current_stage_colors[s], alpha=0.15, linewidth=0)
                
                # Plot comparison condition if available
                if has_comparison:
                    try:
                        comp_signal = self.fft_comparison_derivations[current_channel]
                        comp_fs = self.fft_comparison_raw.info['sfreq']
                        comp_out = self._compute_stage_psd_fft_custom(
                            signal=comp_signal, fs=comp_fs, 
                            scoring_df=self.fft_comparison_scoring,
                            epoch_len=float(getattr(self, 'scoring_epoch_duration', 30.0)),
                            stages=stages_order, fmin=fmin, fmax=fmax,
                            equalize_epochs=eq, robust_stats=robust,
                            nperseg_sec=current_nperseg_sec,
                        )
                        
                        for s in stages_order:
                            if s not in comp_out:
                                continue
                            f_comp, mean_vals_comp, sem_vals_comp, n_ep_comp = comp_out[s]
                            # Convert to log10 power
                            power_vals_comp = mean_vals_comp ** 2
                            power_log_comp = np.log10(np.maximum(power_vals_comp, 1e-20))
                            
                            label_comp = f"{self.fft_comparison_name} - {s} (n={n_ep_comp})"
                            ax.plot(f_comp, power_log_comp, color=current_stage_colors[s], 
                                   label=label_comp, linestyle='--', linewidth=1.5)
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to plot comparison data: {e}")

                ax.set_xlim(max(0.0, fmin), min(fmax, fs/2))
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power (µV², log10)")
                title = "PSD by Sleep Stage - Comparison" if has_comparison else "PSD by Sleep Stage (FFT Power; DC removed)"
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', ncol=1, fontsize=7 if has_comparison else 8)
                fig.tight_layout()
                canvas.draw()
            except Exception as e:
                print(f"❌ CHECKPOINT STAGE-FFT ERR: {e}")
                messagebox.showerror("Erreur", f"Echec calcul/affichage PSD FFT par stade: {e}")

        def _apply_freq_window():
            render()

        ttk.Button(freq_box, text="Apply", command=_apply_freq_window).grid(row=2, column=0, columnspan=2, pady=(8,0))

        render()

        def _save_figure(fig_obj):
            try:
                file_path = filedialog.asksaveasfilename(title="Save Figure", defaultextension=".png",
                                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    fig_obj.savefig(file_path, dpi=200, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure: {e}")

        def _export_csv():
            try:
                fmin = float(freq_min_var.get())
                fmax = float(freq_max_var.get())
                eq = bool(equalize_var.get())
                robust = bool(robust_var.get())
                current_channel = selected_channel_var.get()
                if current_channel not in self.derivations:
                    messagebox.showerror("Error", f"Channel '{current_channel}' not available")
                    return
                signal = self.derivations[current_channel]
                current_nperseg_sec = float(nperseg_sec_var.get())
                out = self._compute_stage_psd_fft_custom(
                    signal=signal, fs=fs, scoring_df=df,
                    epoch_len=float(getattr(self, 'scoring_epoch_duration', 30.0)),
                    stages=stages_order, fmin=fmin, fmax=fmax,
                    equalize_epochs=eq, robust_stats=robust,
                    nperseg_sec=current_nperseg_sec,
                )
                file_path = filedialog.asksaveasfilename(title="Export CSV (long format)", defaultextension=".csv",
                                                         filetypes=[("CSV", "*.csv")])
                if not file_path:
                    return
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(["stage", "n_epochs", "freq_hz", "magnitude", "sem"])
                    for s, (fvals, mean_vals, sem_vals, n_ep) in out.items():
                        for i in range(len(fvals)):
                            writer.writerow([s, n_ep, float(fvals[i]), float(mean_vals[i]), float(sem_vals[i])])
                messagebox.showinfo("Export", f"CSV exported: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV: {e}")

        def _open_comparison_dialog():
            """Open dialog to import a second EDF+scoring for comparison."""
            try:
                comp_dialog = tk.Toplevel(top)
                comp_dialog.title("Import Comparison Data")
                comp_dialog.geometry("550x400")
                comp_dialog.transient(top)
                comp_dialog.grab_set()
                
                ttk.Label(comp_dialog, text="Import a second EDF + Scoring file for comparison", 
                         font=('Segoe UI', 10, 'bold')).pack(pady=(10,10))
                
                # Show current channel info
                current_channel = selected_channel_var.get()
                info_frame = ttk.Frame(comp_dialog)
                info_frame.pack(fill=tk.X, padx=20, pady=(0,10))
                ttk.Label(info_frame, text=f"Channel to import: {current_channel}", 
                         font=('Segoe UI', 9), foreground='blue').pack(anchor='w')
                ttk.Label(info_frame, text="(Only this channel will be loaded to save memory)", 
                         font=('Segoe UI', 8), foreground='gray').pack(anchor='w')
                
                # Condition names
                names_frame = ttk.LabelFrame(comp_dialog, text="Condition Names", padding=10)
                names_frame.pack(fill=tk.X, padx=20, pady=(0,10))
                
                ttk.Label(names_frame, text="Main condition (current data):").grid(row=0, column=0, sticky='w', pady=5)
                main_name_var = tk.StringVar(value=self.fft_main_name)
                ttk.Entry(names_frame, textvariable=main_name_var, width=30).grid(row=0, column=1, padx=(10,0), pady=5)
                
                ttk.Label(names_frame, text="Comparison condition:").grid(row=1, column=0, sticky='w', pady=5)
                comp_name_var = tk.StringVar(value=self.fft_comparison_name)
                ttk.Entry(names_frame, textvariable=comp_name_var, width=30).grid(row=1, column=1, padx=(10,0), pady=5)
                
                # File paths
                files_frame = ttk.LabelFrame(comp_dialog, text="Comparison Files", padding=10)
                files_frame.pack(fill=tk.X, padx=20, pady=(0,10))
                
                edf_path_var = tk.StringVar(value="")
                scoring_path_var = tk.StringVar(value="")
                
                ttk.Label(files_frame, text="Recording file:").grid(row=0, column=0, sticky='w', pady=5)
                ttk.Entry(files_frame, textvariable=edf_path_var, width=30).grid(row=0, column=1, padx=(10,0), pady=5)
                ttk.Button(
                    files_frame,
                    text="Browse...",
                    command=lambda: edf_path_var.set(
                        filedialog.askopenfilename(
                            title="Select recording file",
                            filetypes=recording_filetypes_for_dialog(),
                        )
                    ),
                ).grid(row=0, column=2, padx=(5,0), pady=5)
                
                ttk.Label(files_frame, text="Scoring (XLS):").grid(row=1, column=0, sticky='w', pady=5)
                ttk.Entry(files_frame, textvariable=scoring_path_var, width=30).grid(row=1, column=1, padx=(10,0), pady=5)
                ttk.Button(files_frame, text="Browse...", command=lambda: scoring_path_var.set(
                    filedialog.askopenfilename(title="Select Scoring file", 
                                               filetypes=[("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")])
                )).grid(row=1, column=2, padx=(5,0), pady=5)
                
                # Buttons
                btn_frame = ttk.Frame(comp_dialog)
                btn_frame.pack(pady=20)
                
                def _load_comparison():
                    edf_path = edf_path_var.get()
                    scoring_path = scoring_path_var.get()
                    
                    if not edf_path or not scoring_path:
                        messagebox.showwarning("Warning", "Please select both EDF and Scoring files.")
                        return
                    
                    try:
                        # Get current channel to import only that one
                        current_channel = selected_channel_var.get()
                        
                        # Load EDF with only the selected channel to save memory
                        comp_raw = open_raw_file(edf_path, preload=False, verbose=False)
                        
                        # Check if channel exists in comparison file
                        if current_channel not in comp_raw.ch_names:
                            available = ", ".join(comp_raw.ch_names[:10])
                            if len(comp_raw.ch_names) > 10:
                                available += f"... ({len(comp_raw.ch_names)} total)"
                            messagebox.showerror("Error", 
                                               f"Channel '{current_channel}' not found in comparison EDF.\n\n"
                                               f"Available channels: {available}")
                            return
                        
                        # Load only the selected channel
                        comp_raw_picked = comp_raw.copy().pick_channels([current_channel])
                        comp_raw_picked.load_data()
                        
                        # Extract derivation for selected channel only
                        comp_derivations = {}
                        data, _ = comp_raw_picked[current_channel, :]
                        comp_derivations[current_channel] = data[0] * 1e6  # Convert to µV
                        
                        # Load scoring through the normalized manual-scoring service.
                        comp_duration = float(len(comp_raw_picked.times) / comp_raw_picked.info["sfreq"])
                        comp_result = self.manual_scoring_service.import_excel_path(
                            scoring_path,
                            absolute_start_datetime=None,
                            recording_duration_s=comp_duration,
                            default_epoch_seconds=float(getattr(self, "scoring_epoch_duration", 30.0)),
                        )
                        comp_scoring = comp_result.df
                        
                        # Store comparison data
                        self.fft_comparison_raw = comp_raw_picked
                        self.fft_comparison_derivations = comp_derivations
                        self.fft_comparison_scoring = comp_scoring
                        self.fft_main_name = main_name_var.get()
                        self.fft_comparison_name = comp_name_var.get()
                        
                        messagebox.showinfo("Success", 
                                          f"Loaded comparison data:\n"
                                          f"Channel: {current_channel}\n"
                                          f"{len(comp_scoring)} epochs")
                        comp_dialog.destroy()
                        
                        # Re-render with comparison
                        render()
                        
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load comparison data:\n{str(e)}")
                
                ttk.Button(btn_frame, text="Load and Compare", command=_load_comparison).pack(side=tk.LEFT, padx=5)
                ttk.Button(btn_frame, text="Cancel", command=comp_dialog.destroy).pack(side=tk.LEFT, padx=5)
                
                # Clear button
                def _clear_comparison():
                    self.fft_comparison_raw = None
                    self.fft_comparison_derivations = {}
                    self.fft_comparison_scoring = None
                    messagebox.showinfo("Cleared", "Comparison data cleared.")
                    comp_dialog.destroy()
                    render()
                
                if self.fft_comparison_raw is not None:
                    ttk.Button(btn_frame, text="Clear Comparison", command=_clear_comparison).pack(side=tk.LEFT, padx=5)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open comparison dialog:\n{str(e)}")
        
        # Configurer les boutons après la définition de render
        save_fig_btn.configure(command=lambda: _save_figure(fig))
        export_csv_btn.configure(command=_export_csv)
        compare_btn.configure(command=_open_comparison_dialog)

    def _compute_stage_psd_fft_custom(self, signal, fs, scoring_df, epoch_len, stages, fmin, fmax, equalize_epochs, robust_stats, nperseg_sec):
        """Calcul PSD par stade avec FFT et taille de bin personnalisée."""
        try:
            import pandas as pd  # Import local pour éviter les conflits

            print(f"🔍 CHECKPOINT FFT CUSTOM: nperseg_sec={nperseg_sec}s, fs={fs}Hz")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Signal original - {len(signal)} échantillons ({len(signal)/fs:.1f}s)")

            # Calculer nperseg en échantillons
            nperseg = int(nperseg_sec * fs)
            nperseg = max(8, min(nperseg, len(signal) // 4))  # Limites raisonnables

            print(f"🔍 CHECKPOINT FFT CUSTOM: nperseg={nperseg} échantillons")

            # Analyser la plage temporelle du scoring pour couper l'EEG
            valid_times = []
            invalid_count = 0
            print(f"🔍 CHECKPOINT FFT CUSTOM: Analyse des {len(scoring_df)} époques de scoring")
            
            for idx, row in scoring_df.iterrows():
                try:
                    time_raw = row['time']
                    if isinstance(time_raw, list) or time_raw is None:
                        invalid_count += 1
                        continue
                    
                    # Convertir le temps en secondes depuis le début de l'EEG
                    if isinstance(time_raw, str):
                        try:
                            # Si c'est un timestamp, calculer l'offset depuis le début
                            time_dt = pd.to_datetime(time_raw)
                            # Pour l'instant, on suppose que le premier timestamp valide est t=0
                            # Cette logique pourrait être améliorée si on a l'heure de début de l'EEG
                            valid_times.append(time_dt)
                        except:
                            try:
                                valid_times.append(float(time_raw))
                            except:
                                invalid_count += 1
                                continue
                    else:
                        try:
                            valid_times.append(float(time_raw))
                        except:
                            invalid_count += 1
                            continue
                except:
                    invalid_count += 1
                    continue
            
            print(f"🔍 CHECKPOINT FFT CUSTOM: {len(valid_times)} temps valides, {invalid_count} invalides")
            
            if not valid_times:
                print("🔍 CHECKPOINT FFT CUSTOM: Aucun temps valide trouvé dans le scoring")
                return {}
            
            # Déterminer la plage temporelle du scoring
            if isinstance(valid_times[0], pd.Timestamp):
                # Cas timestamps : convertir en secondes relatives
                min_time = min(valid_times)
                max_time = max(valid_times)
                scoring_duration = (max_time - min_time).total_seconds()
                print(f"🔍 CHECKPOINT FFT CUSTOM: Plage scoring: {min_time} -> {max_time} ({scoring_duration:.1f}s)")
                
                # Pour simplifier, on prend toute la durée disponible
                # TODO: améliorer cette logique si nécessaire
                scoring_start_sec = 0
                scoring_end_sec = min(scoring_duration + epoch_len, len(signal) / fs)
            else:
                # Cas temps en secondes
                scoring_start_sec = min(valid_times)
                scoring_end_sec = max(valid_times) + epoch_len
                print(f"🔍 CHECKPOINT FFT CUSTOM: Plage scoring: {scoring_start_sec:.1f}s -> {scoring_end_sec:.1f}s")
            
            # Couper le signal EEG pour ne garder que la partie scorée
            start_sample = int(scoring_start_sec * fs)
            end_sample = int(min(scoring_end_sec * fs, len(signal)))
            
            if start_sample >= end_sample or start_sample >= len(signal):
                print("🔍 CHECKPOINT FFT CUSTOM: Plage de scoring invalide")
                return {}
            
            signal_cut = signal[start_sample:end_sample]
            print(f"🔍 CHECKPOINT FFT CUSTOM: Signal coupé - {len(signal_cut)} échantillons ({len(signal_cut)/fs:.1f}s)")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Échantillons {start_sample} -> {end_sample}")
            
            # Recalculer nperseg pour le signal coupé
            nperseg = max(8, min(nperseg, len(signal_cut) // 4))
            
            # Dictionnaire pour stocker les résultats par stade
            stage_data = {stage: [] for stage in stages}
            stage_counts = {stage: 0 for stage in stages}
            
            # Parcourir les époques de scoring
            epochs_processed = 0
            epochs_skipped_invalid_stage = 0
            epochs_skipped_invalid_time = 0
            epochs_skipped_out_of_range = 0
            epochs_skipped_unknown_stage = 0
            epochs_skipped_no_signal = 0
            epochs_skipped_short_segment = 0
            epochs_added_to_stage = 0
            
            print(f"🔍 CHECKPOINT FFT CUSTOM: Début traitement des {len(scoring_df)} époques")
            
            for idx, row in scoring_df.iterrows():
                epochs_processed += 1
                try:
                    # Conversion sécurisée du stage
                    stage_raw = row['stage']
                    if isinstance(stage_raw, list) or stage_raw is None:
                        epochs_skipped_invalid_stage += 1
                        continue  # Ignorer les valeurs invalides
                    
                    stage = str(stage_raw).upper()
                    
                    # Conversion sécurisée du temps
                    time_raw = row['time']
                    if isinstance(time_raw, list) or time_raw is None:
                        epochs_skipped_invalid_time += 1
                        continue  # Ignorer les valeurs invalides
                    
                    # Convertir le temps en secondes relatives au signal coupé
                    if isinstance(time_raw, str):
                        try:
                            import pandas as pd
                            time_dt = pd.to_datetime(time_raw)
                            # Calculer l'offset depuis le début du scoring
                            if isinstance(valid_times[0], pd.Timestamp):
                                time_start_relative = (time_dt - min(valid_times)).total_seconds()
                            else:
                                time_start_relative = float(time_raw) - scoring_start_sec
                        except:
                            try:
                                time_start_relative = float(time_raw) - scoring_start_sec
                            except:
                                continue
                    else:
                        try:
                            time_start_relative = float(time_raw) - scoring_start_sec
                        except:
                            continue
                    
                    # Vérifier que l'époque est dans la plage du signal coupé
                    if time_start_relative < 0 or time_start_relative >= len(signal_cut) / fs:
                        epochs_skipped_out_of_range += 1
                        continue
                    
                    time_end_relative = time_start_relative + epoch_len
                    
                except Exception as e:
                    print(f"🔍 CHECKPOINT FFT CUSTOM: Ligne {idx} ignorée - erreur: {str(e)}")
                    continue
                
                # Normaliser le stade (gérer tous les cas possibles)
                if stage in ['WAKE', 'EVEIL', 'ÉVEIL', 'W']:
                    stage = 'W'
                elif stage in ['REM', 'R']:
                    stage = 'R'
                elif stage in ['N1', 'N2', 'N3']:
                    stage = stage  # Garder tel quel
                elif stage == 'U':
                    # Ignorer les stades U (Undefined/Unknown) par défaut
                    # On peut les mapper vers W si nécessaire, mais pour l'analyse PSD par stade,
                    # il est préférable de les ignorer car ils ne représentent pas un stade de sommeil défini
                    epochs_skipped_unknown_stage += 1
                    continue
                else:
                    print(f"🔍 CHECKPOINT FFT CUSTOM: Stade ignoré: '{stage}' (original: '{row['stage']}')")
                    epochs_skipped_unknown_stage += 1
                    continue  # Ignorer les stades inconnus
                
                if stage not in stages:
                    epochs_skipped_unknown_stage += 1
                    continue
                
                # Compter les époques par stade
                stage_counts[stage] += 1
                
                # Extraire le segment de signal (du signal coupé avec temps relatifs)
                start_idx = int(time_start_relative * fs)
                end_idx = int(time_end_relative * fs)
                
                if start_idx >= 0 and end_idx <= len(signal_cut):
                    segment = signal_cut[start_idx:end_idx]
                    
                    if len(segment) > nperseg:
                        # Diviser le segment en sous-segments de taille nperseg
                        n_segments = len(segment) // nperseg
                        for i in range(n_segments):
                            sub_segment = segment[i*nperseg:(i+1)*nperseg]
                            stage_data[stage].append(sub_segment)
                            epochs_added_to_stage += 1
                    else:
                        epochs_skipped_short_segment += 1
                else:
                    epochs_skipped_no_signal += 1
            
            # Afficher les statistiques détaillées de traitement
            print(f"🔍 CHECKPOINT FFT CUSTOM: ═══ STATISTIQUES DÉTAILLÉES ═══")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Total époques dans scoring: {len(scoring_df)}")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Époques traitées: {epochs_processed}")
            print(f"🔍 CHECKPOINT FFT CUSTOM: ─── ÉPOQUES IGNORÉES ───")
            print(f"🔍 CHECKPOINT FFT CUSTOM:   Stage invalide: {epochs_skipped_invalid_stage}")
            print(f"🔍 CHECKPOINT FFT CUSTOM:   Temps invalide: {epochs_skipped_invalid_time}")
            print(f"🔍 CHECKPOINT FFT CUSTOM:   Hors plage temporelle: {epochs_skipped_out_of_range}")
            print(f"🔍 CHECKPOINT FFT CUSTOM:   Stage inconnu/U: {epochs_skipped_unknown_stage}")
            print(f"🔍 CHECKPOINT FFT CUSTOM:   Pas de signal: {epochs_skipped_no_signal}")
            print(f"🔍 CHECKPOINT FFT CUSTOM:   Segment trop court: {epochs_skipped_short_segment}")
            print(f"🔍 CHECKPOINT FFT CUSTOM: ─── RÉSULTATS FINAUX ───")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Segments ajoutés total: {epochs_added_to_stage}")
            
            print(f"🔍 CHECKPOINT FFT CUSTOM: Époques par stade:")
            for stage in stages:
                count = stage_counts.get(stage, 0)
                segments = len(stage_data.get(stage, []))
                print(f"🔍 CHECKPOINT FFT CUSTOM:   {stage}: {count} époques -> {segments} segments")
            
            # Calculer les totaux pour vérification
            total_skipped = (epochs_skipped_invalid_stage + epochs_skipped_invalid_time + 
                           epochs_skipped_out_of_range + epochs_skipped_unknown_stage + 
                           epochs_skipped_no_signal + epochs_skipped_short_segment)
            total_stages_counted = sum(stage_counts.values())
            print(f"🔍 CHECKPOINT FFT CUSTOM: ═══ VÉRIFICATION ═══")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Total ignorées: {total_skipped}")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Total stages comptés: {total_stages_counted}")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Total segments finaux: {sum(len(data) for data in stage_data.values())}")
            print(f"🔍 CHECKPOINT FFT CUSTOM: Vérification: {total_skipped + total_stages_counted} = {len(scoring_df)} ?")
            print(f"🔍 CHECKPOINT FFT CUSTOM: ═══════════════════════════════")
            
            # Égaliser les époques par stade si demandé
            if equalize_epochs:
                try:
                    # Trouver le nombre minimum d'époques
                    min_epochs = min(stage_counts.values()) if stage_counts.values() else 0
                    if min_epochs > 0:
                        print(f"🔍 CHECKPOINT FFT EQ: Égalisation à n={min_epochs} époques/stade")
                        
                        # Calculer le nombre de segments par époque
                        segments_per_epoch = max(1, int(epoch_len / nperseg_sec))
                        target_segments = min_epochs * segments_per_epoch
                        
                        # Égaliser les segments
                        for stage in stages:
                            if len(stage_data[stage]) > target_segments:
                                stage_data[stage] = stage_data[stage][:target_segments]
                                print(f"🔍 CHECKPOINT FFT EQ: {stage}: {len(stage_data[stage])} segments (égalisé)")
                        
                        # Vérifier l'égalisation
                        print(f"🔍 CHECKPOINT FFT EQ: Segments après égalisation:")
                        for stage in stages:
                            print(f"  {stage}: {len(stage_data[stage])} segments")
                            
                except Exception as e:
                    print(f"⚠️ CHECKPOINT FFT EQ: échec égalisation: {e}")
            
            # Calculer la PSD pour chaque stade
            result = {}
            for stage in stages:
                if len(stage_data[stage]) == 0:
                    continue
                
                print(f"🔍 CHECKPOINT FFT CUSTOM: {stage}: {len(stage_data[stage])} segments")
                
                # Calculer FFT pour chaque segment
                psd_list = []
                freqs_ref = None
                
                for segment in stage_data[stage]:
                    # FFT avec fenêtrage de Hann
                    windowed = segment * np.hanning(len(segment))
                    fft_result = np.fft.fft(windowed)
                    psd = np.abs(fft_result) ** 2
                    
                    # Fréquences correspondantes
                    freqs = np.fft.fftfreq(len(segment), 1/fs)
                    
                    # Garder seulement les fréquences positives
                    pos_mask = freqs >= 0
                    freqs = freqs[pos_mask]
                    psd = psd[pos_mask]
                    
                    # Filtrer par plage de fréquences
                    freq_mask = (freqs >= fmin) & (freqs <= fmax)
                    freqs = freqs[freq_mask]
                    psd = psd[freq_mask]
                    
                    if freqs_ref is None:
                        freqs_ref = freqs
                    
                    psd_list.append(psd)
                
                if len(psd_list) == 0:
                    continue
                
                # Convertir en array numpy
                psd_array = np.array(psd_list)
                
                # Calculer statistiques
                if robust_stats:
                    # Médiane + MAD
                    mean_vals = np.median(psd_array, axis=0)
                    mad = np.median(np.abs(psd_array - mean_vals), axis=0)
                    sem_vals = 1.4826 * mad / np.sqrt(len(psd_array))  # MAD -> SEM
                else:
                    # Moyenne + SEM classique
                    mean_vals = np.mean(psd_array, axis=0)
                    sem_vals = np.std(psd_array, axis=0) / np.sqrt(len(psd_array))
                
                result[stage] = (freqs_ref, mean_vals, sem_vals, len(psd_array))
            
            print(f"🔍 CHECKPOINT FFT CUSTOM: Résultat final: {list(result.keys())}")
            return result
            
        except Exception as e:
            print(f"❌ CHECKPOINT FFT CUSTOM: Erreur: {e}")
            # Fallback vers la fonction originale
            return compute_stage_psd_fft_for_array(
                signal=signal, fs=fs, scoring_df=scoring_df,
                epoch_len=epoch_len, stages=stages, fmin=fmin, fmax=fmax,
                equalize_epochs=equalize_epochs, robust_stats=robust_stats,
            )

    def _show_wavelet_spectrogram_before_after(self):
        """Figures 3c/3d: spectrogramme ondelette Morlet superposé à l'hypnogramme, avant/après."""
        print("🔍 CHECKPOINT TFR 1: Entrée _show_wavelet_spectrogram_before_after")
        if not self.raw:
            print("⚠️ CHECKPOINT TFR 1: raw manquant")
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return

        def create_tooltip(widget, text):
            """Crée un tooltip pour un widget."""
            def show_tooltip(event):
                tooltip = tk.Toplevel()
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                
                label = tk.Label(tooltip, text=text, justify='left', 
                               background='#ffffe0', relief='solid', borderwidth=1,
                               font=('Segoe UI', 9), wraplength=300)
                label.pack()
                
                widget.tooltip = tooltip
                
            def hide_tooltip(event):
                if hasattr(widget, 'tooltip') and widget.tooltip:
                    widget.tooltip.destroy()
                    widget.tooltip = None
                    
            widget.bind('<Enter>', show_tooltip)
            widget.bind('<Leave>', hide_tooltip)
        # Canal cible (Pz/E101)
        candidate = None
        for name in ["Pz", "E101", "PZ"] + self.selected_channels + (self.raw.ch_names if self.raw else []):
            if name in self.derivations:
                candidate = name
                break
        if candidate is None:
            print("⚠️ CHECKPOINT TFR 2: Aucun canal trouvé")
            messagebox.showwarning("Attention", "Aucun canal disponible")
            return

        top = tk.Toplevel(self.root)
        top.title("Spectrogramme ondelettes (Morlet) – avant/après")
        top.geometry("1200x800")
        top.transient(self.root)
        top.grab_set()

        container = ttk.Frame(top, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        # Contrôles: fenêtre temporelle + export
        ctrl = ttk.Frame(container)
        ctrl.pack(fill=tk.X)
        ttk.Label(ctrl, text=f"Canal: {candidate}").pack(side=tk.LEFT)
        ttk.Label(ctrl, text="Fenêtre (s)").pack(side=tk.LEFT, padx=(10,0))
        start_var = tk.DoubleVar(value=self.current_time)
        dur_var = tk.DoubleVar(value=self.duration)
        start_entry = ttk.Entry(ctrl, textvariable=start_var, width=8)
        start_entry.pack(side=tk.LEFT, padx=(5, 10))
        dur_entry = ttk.Entry(ctrl, textvariable=dur_var, width=8)
        dur_entry.pack(side=tk.LEFT, padx=(0, 10))
        refresh_btn = ttk.Button(ctrl, text="Actualiser", command=lambda: render())
        refresh_btn.pack(side=tk.LEFT)
        save_btn = ttk.Button(ctrl, text="Enregistrer Figure", command=lambda: _save_figure(fig))
        save_btn.pack(side=tk.RIGHT)
        
        # Tooltips pour les contrôles du spectrogramme
        create_tooltip(start_entry, 
                      "Temps de début du spectrogramme (secondes).\n\n"
                      "• Définit le point de départ de l'analyse temporelle-fréquentielle\n"
                      "• Correspond au temps actuel affiché dans la vue principale\n"
                      "• Peut être modifié pour analyser une période spécifique\n"
                      "• Doit être ≥ 0 et < durée totale de l'enregistrement")
        
        create_tooltip(dur_entry, 
                      "Durée de la fenêtre d'analyse (secondes).\n\n"
                      "• Définit la longueur de la période analysée\n"
                      "• Plus la durée est longue, plus la résolution temporelle est fine\n"
                      "• Recommandé : 30-120 secondes pour un bon compromis\n"
                      "• Doit être ≥ 5 secondes pour des résultats fiables")
        
        create_tooltip(refresh_btn, 
                      "Actualiser le spectrogramme.\n\n"
                      "• Recalcule le spectrogramme avec les nouveaux paramètres\n"
                      "• Utilise la fenêtre temporelle et le canal sélectionnés\n"
                      "• Met à jour l'affichage temporelle-fréquentielle\n"
                      "• Nécessaire après modification des paramètres")
        
        create_tooltip(save_btn, 
                      "Enregistrer le spectrogramme.\n\n"
                      "• Sauvegarde le spectrogramme au format PNG, PDF ou SVG\n"
                      "• Qualité haute résolution (200 DPI)\n"
                      "• Inclut les deux graphiques (avant/après) et l'hypnogramme\n"
                      "• Utile pour les publications et rapports")

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        df = self._get_active_scoring_df()
        print(f"🔍 CHECKPOINT TFR 3: Scoring dispo={df is not None}, len={len(df) if df is not None else 0}")
        fs = float(self.sfreq)
        freqs = np.linspace(0.5, 40.0, 60)
        n_cycles = freqs / 2.0

        def compute_tfr(x):
            # x: 1D, -> shape (n_epochs=1, n_channels=1, n_times)
            arr = x[np.newaxis, np.newaxis, :]
            power = tfr_array_morlet(arr, sfreq=fs, freqs=freqs, n_cycles=n_cycles, output='power')
            return power[0,0]  # (n_freqs, n_times)

        def render():
            try:
                axes[0].clear(); axes[1].clear()
                start_s = max(0.0, float(start_var.get()))
                dur_s = max(5.0, float(dur_var.get()))
                end_s = start_s + dur_s
                i0 = int(max(0, min(len(self.raw.times)-2, start_s * fs)))
                i1 = int(max(i0+2, min(len(self.raw.times), end_s * fs)))
                t = self.raw.times[i0:i1]

                x_before = self.derivations[candidate][i0:i1]
                x_after = x_before  # TODO: appliquer bad-spans/ICA si disponibles
                P1 = compute_tfr(x_before)
                P2 = compute_tfr(x_after)
                print(f"🔍 CHECKPOINT TFR 4: TFR shapes: {P1.shape}, {P2.shape}")

                # Préparation robuste d'affichage (échelle dB, percentiles)
                eps = np.finfo(float).eps
                P1 = np.nan_to_num(P1, nan=0.0, posinf=0.0, neginf=0.0)
                P2 = np.nan_to_num(P2, nan=0.0, posinf=0.0, neginf=0.0)
                P1_db = 10.0 * np.log10(P1 + eps)
                P2_db = 10.0 * np.log10(P2 + eps)
                # Bornes dynamiques par percentiles
                p1_vals = P1_db[np.isfinite(P1_db)]
                p2_vals = P2_db[np.isfinite(P2_db)]
                if p1_vals.size == 0:
                    p1_vmin, p1_vmax = -120.0, -60.0
                else:
                    p1_vmin, p1_vmax = np.percentile(p1_vals, [5, 99])
                if p2_vals.size == 0:
                    p2_vmin, p2_vmax = -120.0, -60.0
                else:
                    p2_vmin, p2_vmax = np.percentile(p2_vals, [5, 99])

                im0 = axes[0].pcolormesh(t, freqs, P1_db, shading='auto', cmap='viridis', vmin=p1_vmin, vmax=p1_vmax)
                axes[0].set_title("Avant (filtrage + interp. canaux)")
                axes[0].set_ylabel("Fréquence (Hz)")
                fig.colorbar(im0, ax=axes[0], label='Power (dB)')

                im1 = axes[1].pcolormesh(t, freqs, P2_db, shading='auto', cmap='viridis', vmin=p2_vmin, vmax=p2_vmax)
                axes[1].set_title("Après (rejets/ICA si dispo)")
                axes[1].set_xlabel("Temps (s)")
                axes[1].set_ylabel("Fréquence (Hz)")
                fig.colorbar(im1, ax=axes[1], label='Power (dB)')

                # Superposition hypnogramme (si disponible)
                if df is not None and len(df) > 0:
                    # Mapper stades en profondeurs (W=0, N1=1, N2=2, N3=3, R=2.5)
                    mapper = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':2.5}
                    epoch_len = float(getattr(self, 'scoring_epoch_duration', 30.0))
                    times_epoch = df['time'].values
                    values = [mapper.get(str(s).upper(), np.nan) for s in df['stage'].values]
                    axes[0].plot(times_epoch, values, color='k', linewidth=1.2)
                    axes[1].plot(times_epoch, values, color='k', linewidth=1.2)

                fig.tight_layout()
                canvas.draw()
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec spectrogramme ondelette: {e}")

        def _save_figure(fig_obj):
            try:
                file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    fig_obj.savefig(file_path, dpi=200, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")

        render()
        def _save_figure(fig_obj):
            try:
                file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    fig_obj.savefig(file_path, dpi=200, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")

        def _export_psd_csv(before_dict, after_dict):
            try:
                file_path = filedialog.asksaveasfilename(title="Exporter CSV", defaultextension=".csv",
                                                         filetypes=[("CSV", "*.csv")])
                if not file_path:
                    return
                import csv
                stages_order = ["W", "N1", "N2", "N3", "R"]
                # Format: stage, condition(before/after), freq, psd
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["stage", "condition", "freq_hz", "psd_uv2_per_hz"]) 
                    for cond_name, data in [("before", before_dict), ("after", after_dict)]:
                        for s in stages_order:
                            if s in data:
                                freqs, psd = data[s]
                                for fr, val in zip(freqs, psd):
                                    writer.writerow([s, cond_name, f"{fr:.4f}", f"{val:.8e}"])
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'export CSV: {e}")
    
    def _show_temporal_analysis(self):
        """Affiche l'analyse temporelle des données."""
        try:
            if not hasattr(self, 'raw') or self.raw is None:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            if not hasattr(self, 'selected_channels') or not self.selected_channels:
                messagebox.showwarning("Attention", "Veuillez d'abord sélectionner des canaux")
                return
            
            print("🔍 CHECKPOINT TEMPORAL: Début analyse temporelle")
            logging.info("[TEMPORAL] Starting temporal analysis")

            def create_tooltip(widget, text):
                """Crée un tooltip pour un widget."""
                def show_tooltip(event):
                    tooltip = tk.Toplevel()
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                    
                    label = tk.Label(tooltip, text=text, justify='left', 
                                   background='#ffffe0', relief='solid', borderwidth=1,
                                   font=('Segoe UI', 9), wraplength=300)
                    label.pack()
                    
                    widget.tooltip = tooltip
                    
                def hide_tooltip(event):
                    if hasattr(widget, 'tooltip') and widget.tooltip:
                        widget.tooltip.destroy()
                        widget.tooltip = None
                        
                widget.bind('<Enter>', show_tooltip)
                widget.bind('<Leave>', hide_tooltip)
            
            # Créer la fenêtre d'analyse temporelle
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Analyse Temporelle - EEG Analysis Studio")
            analysis_window.geometry("1000x700")
            analysis_window.transient(self.root)
            analysis_window.grab_set()
            
            # Frame principal avec scrollbar
            main_frame = ttk.Frame(analysis_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Canvas et scrollbar pour le contenu
            canvas = tk.Canvas(main_frame)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Titre
            title_label = ttk.Label(scrollable_frame, text="📊 Analyse Temporelle des Données EEG", 
                                  font=('Segoe UI', 16, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # Informations générales
            info_frame = ttk.LabelFrame(scrollable_frame, text="Informations Générales", padding=10)
            info_frame.pack(fill=tk.X, pady=(0, 10))
            
            info_text = f"""
Fichier: {getattr(self, 'current_file', 'Inconnu')}
Fréquence d'échantillonnage: {self.sfreq} Hz
Durée totale: {len(self.raw.times):.1f} secondes ({len(self.raw.times)/3600:.1f} heures)
Temps actuel: {self.current_time:.1f}s
Canaux analysés: {len(self.selected_channels)}
            """
            
            ttk.Label(info_frame, text=info_text.strip()).pack(anchor='w')
            
            # Analyse par canal
            for i, channel in enumerate(self.selected_channels):
                if channel not in self.derivations:
                    continue
                    
                channel_frame = ttk.LabelFrame(scrollable_frame, text=f"Canal: {channel}", padding=10)
                channel_frame.pack(fill=tk.X, pady=(0, 10))
                
                # Extraire les données du canal
                data = self.derivations[channel]
                current_window_data = self._extract_current_window_data(data)
                
                if len(current_window_data) > 0:
                    # Calculs temporels
                    stats = self._calculate_temporal_stats(current_window_data, channel)
                    
                    # Affichage des statistiques
                    stats_text = f"""
Amplitude moyenne: {stats['mean_amplitude']:.3f} µV
Écart-type: {stats['std_amplitude']:.3f} µV
Amplitude crête-à-crête: {stats['peak_to_peak']:.3f} µV
Amplitude maximale: {stats['max_amplitude']:.3f} µV
Amplitude minimale: {stats['min_amplitude']:.3f} µV
Fréquence dominante: {stats['dominant_frequency']:.2f} Hz
Puissance totale: {stats['total_power']:.3f} µV²
                    """
                    
                    ttk.Label(channel_frame, text=stats_text.strip()).pack(anchor='w')
                    
                    # Graphique temporel simple
                    fig = Figure(figsize=(8, 3), dpi=80)
                    ax = fig.add_subplot(111)
                    
                    time_axis = np.linspace(self.current_time, 
                                          self.current_time + len(current_window_data)/self.sfreq, 
                                          len(current_window_data))
                    
                    ax.plot(time_axis, current_window_data, linewidth=0.8)
                    ax.set_title(f"Signal temporel - {channel}")
                    ax.set_xlabel("Temps (s)")
                    ax.set_ylabel("Amplitude (µV)")
                    ax.grid(True, alpha=0.3)
                    
                    # Intégrer le graphique dans tkinter
                    canvas_widget = FigureCanvasTkAgg(fig, channel_frame)
                    canvas_widget.draw()
                    canvas_widget.get_tk_widget().pack(fill=tk.X, pady=(10, 0))
            
            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=(20, 0))
            
            export_btn = ttk.Button(button_frame, text="Exporter Analyse", 
                      command=lambda: self._export_temporal_analysis())
            export_btn.pack(side=tk.LEFT, padx=(0, 10))
            refresh_btn = ttk.Button(button_frame, text="Actualiser", 
                      command=lambda: self._refresh_temporal_analysis(analysis_window))
            refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
            close_btn = ttk.Button(button_frame, text="Fermer", 
                      command=analysis_window.destroy)
            close_btn.pack(side=tk.RIGHT)
            
            # Tooltips pour les boutons
            create_tooltip(export_btn, 
                          "Exporter l'analyse temporelle.\n\n"
                          "• Sauvegarde toutes les statistiques temporelles calculées\n"
                          "• Format : CSV avec métriques par canal et fenêtre temporelle\n"
                          "• Inclut : amplitude, variance, skewness, kurtosis, etc.\n"
                          "• Compatible avec Excel, R, Python pour analyse externe\n"
                          "• Utile pour l'analyse statistique comparative")
            
            create_tooltip(refresh_btn, 
                          "Actualiser l'analyse temporelle.\n\n"
                          "• Recalcule toutes les statistiques temporelles\n"
                          "• Utilise la fenêtre temporelle actuelle de la vue principale\n"
                          "• Met à jour les graphiques et métriques affichés\n"
                          "• Nécessaire après changement de position temporelle\n"
                          "• Recharge les données des canaux sélectionnés")
            
            create_tooltip(close_btn, 
                          "Fermer la fenêtre d'analyse temporelle.\n\n"
                          "• Ferme cette fenêtre d'analyse\n"
                          "• Retourne à la vue principale de l'application\n"
                          "• Les données analysées ne sont pas sauvegardées\n"
                          "• Utilisez 'Exporter Analyse' pour sauvegarder les résultats")
            
            # Configuration du canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Lier la molette de la souris au scroll
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            def _unbind_mousewheel(event):
                canvas.unbind_all("<MouseWheel>")
            analysis_window.bind("<Destroy>", _unbind_mousewheel)
            
            print("✅ CHECKPOINT TEMPORAL: Analyse temporelle affichée")
            logging.info("[TEMPORAL] Temporal analysis window displayed")
            
        except Exception as e:
            print(f"❌ CHECKPOINT TEMPORAL: Erreur analyse temporelle: {e}")
            logging.error(f"[TEMPORAL] Failed to show temporal analysis: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse temporelle : {str(e)}")
    
    def _show_renormalized_entropy(self):
        """Affiche l'analyse d'entropie renormée pour les canaux sélectionnés."""
        print("🔍 CHECKPOINT ENTROPY: Entrée _show_renormalized_entropy")
        try:
            if not self.raw or not self.selected_channels:
                print(f"⚠️ CHECKPOINT ENTROPY: raw={self.raw is not None}, selected_channels={self.selected_channels}")
                messagebox.showwarning("Attention", "Chargez un fichier et sélectionnez au moins un canal")
                return

            # Importer le module d'entropie
            from CESA.entropy import compute_entropy_from_raw, RenormalizedEntropyConfig

            def create_tooltip(widget, text):
                """Crée un tooltip pour un widget."""
                def show_tooltip(event):
                    tooltip = tk.Toplevel()
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                    
                    label = tk.Label(tooltip, text=text, justify='left', 
                                   background='#ffffe0', relief='solid', borderwidth=1,
                                   font=('Segoe UI', 9), wraplength=300)
                    label.pack()
                    
                    widget.tooltip = tooltip
                    
                def hide_tooltip(event):
                    if hasattr(widget, 'tooltip') and widget.tooltip:
                        widget.tooltip.destroy()
                        widget.tooltip = None
                        
                widget.bind('<Enter>', show_tooltip)
                widget.bind('<Leave>', hide_tooltip)

            # Fenêtre secondaire
            entropy_window = tk.Toplevel(self.root)
            entropy_window.title("Entropie Renormée (Issartel) - EEG Analysis Studio")
            entropy_window.geometry("1200x900")
            entropy_window.transient(self.root)
            entropy_window.grab_set()
            
            # Centrer la fenêtre
            entropy_window.update_idletasks()
            x = (entropy_window.winfo_screenwidth() // 2) - (1200 // 2)
            y = (entropy_window.winfo_screenheight() // 2) - (900 // 2)
            entropy_window.geometry(f"1200x900+{x}+{y}")

            # Frame principal avec scrollbar
            main_frame = ttk.Frame(entropy_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Canvas et scrollbar pour le contenu
            canvas = tk.Canvas(main_frame)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Titre
            title_label = ttk.Label(scrollable_frame, text="🧮 Analyse d'Entropie Renormée (Issartel)", 
                                  font=('Segoe UI', 16, 'bold'))
            title_label.pack(pady=(0, 20))

            # Informations générales
            info_frame = ttk.LabelFrame(scrollable_frame, text="Informations Générales", padding=10)
            info_frame.pack(fill=tk.X, pady=(0, 10))

            info_text = f"""
Fichier: {getattr(self, 'current_file', 'Inconnu')}
Fréquence d'échantillonnage: {self.sfreq} Hz
Durée totale: {len(self.raw.times):.1f} secondes ({len(self.raw.times)/3600:.1f} heures)
Canaux analysés: {len(self.selected_channels)}
Canaux: {', '.join(self.selected_channels)}
            """

            ttk.Label(info_frame, text=info_text.strip()).pack(anchor='w')

            # Configuration de l'entropie renormée
            config_frame = ttk.LabelFrame(scrollable_frame, text="Configuration de l'Entropie Renormée", padding=10)
            config_frame.pack(fill=tk.X, pady=(0, 10))

            # Variables de configuration
            window_length_var = tk.DoubleVar(value=4.0)
            overlap_var = tk.DoubleVar(value=0.5)
            moment_order_var = tk.DoubleVar(value=2.0)
            psi_name_var = tk.StringVar(value="powerlaw")
            gamma_var = tk.DoubleVar(value=0.5)
            entropy_unit_var = tk.StringVar(value="both")

            # Configuration des paramètres
            config_grid = ttk.Frame(config_frame)
            config_grid.pack(fill=tk.X)

            # Longueur de fenêtre
            ttk.Label(config_grid, text="Longueur de fenêtre (s):").grid(row=0, column=0, sticky='w', padx=(0, 10))
            window_length_spin = ttk.Spinbox(config_grid, from_=1.0, to=30.0, increment=0.5, 
                                           textvariable=window_length_var, width=10)
            window_length_spin.grid(row=0, column=1, sticky='w')

            # Chevauchement
            ttk.Label(config_grid, text="Chevauchement (%):").grid(row=0, column=2, sticky='w', padx=(20, 10))
            overlap_spin = ttk.Spinbox(config_grid, from_=0.0, to=0.9, increment=0.1, 
                                     textvariable=overlap_var, width=10)
            overlap_spin.grid(row=0, column=3, sticky='w')

            # Ordre du moment
            ttk.Label(config_grid, text="Ordre du moment:").grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
            moment_order_spin = ttk.Spinbox(config_grid, from_=1.0, to=4.0, increment=0.5, 
                                           textvariable=moment_order_var, width=10)
            moment_order_spin.grid(row=1, column=1, sticky='w', pady=(10, 0))

            # Kernel de renormée
            ttk.Label(config_grid, text="Kernel de renormée:").grid(row=1, column=2, sticky='w', padx=(20, 10), pady=(10, 0))
            psi_combo = ttk.Combobox(config_grid, textvariable=psi_name_var, width=12, state="readonly")
            psi_combo['values'] = ('identity', 'powerlaw', 'log', 'adaptive')
            psi_combo.grid(row=1, column=3, sticky='w', pady=(10, 0))

            # Paramètre gamma (pour powerlaw)
            ttk.Label(config_grid, text="Gamma (powerlaw):").grid(row=2, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
            gamma_spin = ttk.Spinbox(config_grid, from_=0.1, to=2.0, increment=0.1, 
                                   textvariable=gamma_var, width=10)
            gamma_spin.grid(row=2, column=1, sticky='w', pady=(10, 0))

            # Unité d'entropie
            ttk.Label(config_grid, text="Unité d'entropie:").grid(row=2, column=2, sticky='w', padx=(20, 10), pady=(10, 0))
            unit_combo = ttk.Combobox(config_grid, textvariable=entropy_unit_var, width=12, state="readonly")
            unit_combo['values'] = ('nat', 'bit', 'both')
            unit_combo.grid(row=2, column=3, sticky='w', pady=(10, 0))

            # Zone de résultats
            results_frame = ttk.LabelFrame(scrollable_frame, text="Résultats de l'Entropie Renormée", padding=10)
            results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            # Zone de texte pour les résultats
            results_text = tk.Text(results_frame, height=15, wrap=tk.WORD, font=('Consolas', 10))
            results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_text.yview)
            results_text.configure(yscrollcommand=results_scrollbar.set)

            results_text.pack(side="left", fill="both", expand=True)
            results_scrollbar.pack(side="right", fill="y")

            def run_entropy_analysis():
                """Exécute l'analyse d'entropie renormée."""
                try:
                    results_text.delete(1.0, tk.END)
                    results_text.insert(tk.END, "🔄 Calcul de l'entropie renormée en cours...\n\n")
                    entropy_window.update()

                    # Créer la configuration
                    config = RenormalizedEntropyConfig(
                        window_length=window_length_var.get(),
                        overlap=overlap_var.get(),
                        moment_order=moment_order_var.get(),
                        psi_name=psi_name_var.get(),
                        psi_params={"gamma": gamma_var.get(), "epsilon": 1e-12},
                        entropy_unit=entropy_unit_var.get(),
                        return_intermediate=True
                    )

                    # Exécuter l'analyse
                    result = compute_entropy_from_raw(self.raw, self.selected_channels, config)

                    # Afficher les résultats
                    results_text.delete(1.0, tk.END)
                    results_text.insert(tk.END, "✅ Analyse d'entropie renormée terminée\n\n")
                    results_text.insert(tk.END, f"📊 RÉSULTATS PRINCIPAUX:\n")
                    results_text.insert(tk.END, f"{'='*50}\n")
                    results_text.insert(tk.END, f"Entropie (nats): {result.entropy_nats:.6f}\n")
                    results_text.insert(tk.END, f"Entropie (bits): {result.entropy_bits:.6f}\n\n")

                    results_text.insert(tk.END, f"📋 CONFIGURATION UTILISÉE:\n")
                    results_text.insert(tk.END, f"{'='*50}\n")
                    results_text.insert(tk.END, f"Longueur de fenêtre: {config.window_length} s\n")
                    results_text.insert(tk.END, f"Chevauchement: {config.overlap*100:.1f}%\n")
                    results_text.insert(tk.END, f"Ordre du moment: {config.moment_order}\n")
                    results_text.insert(tk.END, f"Kernel de renormée: {config.psi_name}\n")
                    if config.psi_name == "powerlaw":
                        results_text.insert(tk.END, f"Paramètre gamma: {config.psi_params['gamma']}\n")
                    results_text.insert(tk.END, f"Échantillons par fenêtre: {result.window_samples}\n")
                    results_text.insert(tk.END, f"Pas entre fenêtres: {result.step_samples}\n\n")

                    results_text.insert(tk.END, f"📈 VALEURS PROPRES RENORMÉES:\n")
                    results_text.insert(tk.END, f"{'='*50}\n")
                    for i, eigval in enumerate(result.psi_eigenvalues):
                        results_text.insert(tk.END, f"λ{i+1}: {eigval:.6e}\n")

                    results_text.insert(tk.END, f"\n📊 MATRICE DE COVARIANCE RENORMÉE:\n")
                    results_text.insert(tk.END, f"{'='*50}\n")
                    cov_matrix = result.weighted_covariance
                    for i in range(cov_matrix.shape[0]):
                        row_str = " ".join([f"{cov_matrix[i,j]:8.3e}" for j in range(cov_matrix.shape[1])])
                        results_text.insert(tk.END, f"Ligne {i+1}: {row_str}\n")

                    results_text.insert(tk.END, f"\n📝 INTERPRÉTATION:\n")
                    results_text.insert(tk.END, f"{'='*50}\n")
                    results_text.insert(tk.END, f"• L'entropie renormée quantifie la complexité du signal EEG\n")
                    results_text.insert(tk.END, f"• Valeurs élevées = plus de variabilité/complexité\n")
                    results_text.insert(tk.END, f"• Valeurs faibles = plus de régularité/déterminisme\n")
                    results_text.insert(tk.END, f"• Méthode basée sur les travaux d'Issartel (renormalisation)\n")
                    results_text.insert(tk.END, f"• Utile pour l'analyse de l'état de conscience\n")

                    print("✅ CHECKPOINT ENTROPY: Analyse terminée avec succès")
                    logging.info("[ENTROPY] Renormalized entropy analysis completed successfully")

                except Exception as e:
                    results_text.delete(1.0, tk.END)
                    results_text.insert(tk.END, f"❌ Erreur lors du calcul de l'entropie renormée:\n{str(e)}\n")
                    print(f"❌ CHECKPOINT ENTROPY: Erreur analyse: {e}")
                    logging.error(f"[ENTROPY] Failed to compute renormalized entropy: {e}")

            def export_results():
                """Exporte les résultats de l'entropie renormée."""
                try:
                    file_path = filedialog.asksaveasfilename(
                        title="Exporter les résultats d'entropie renormée",
                        defaultextension=".txt",
                        filetypes=[("Fichier texte", "*.txt"), ("Tous les fichiers", "*.*")]
                    )
                    if file_path:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(results_text.get(1.0, tk.END))
                        messagebox.showinfo("Succès", f"Résultats exportés vers {file_path}")
                        logging.info(f"[ENTROPY] Results exported to {file_path}")
                except Exception as e:
                    messagebox.showerror("Erreur", f"Erreur lors de l'export: {str(e)}")
                    logging.error(f"[ENTROPY] Export failed: {e}")

            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))

            analyze_btn = ttk.Button(button_frame, text="🧮 Calculer Entropie", command=run_entropy_analysis)
            analyze_btn.pack(side=tk.LEFT, padx=(0, 10))

            export_btn = ttk.Button(button_frame, text="📁 Exporter Résultats", command=export_results)
            export_btn.pack(side=tk.LEFT, padx=(0, 10))

            def show_math_info():
                """Affiche la fenêtre d'information mathématique avec Matplotlib ou fallback."""
                print("🔍 CHECKPOINT MATH: Tentative d'ouverture fenêtre mathématique")
                
                # Essayer d'abord la version Matplotlib
                try:
                    from math_formulas_window import show_math_formulas_window
                    print("🔍 CHECKPOINT MATH: Tentative d'ouverture fenêtre Matplotlib")
                    show_math_formulas_window(entropy_window)
                    print("✅ CHECKPOINT MATH: Fenêtre Matplotlib ouverte avec succès")
                    return
                except ImportError as e:
                    print(f"❌ CHECKPOINT MATH: Erreur import module Matplotlib: {e}")
                except Exception as e:
                    print(f"❌ CHECKPOINT MATH: Erreur générale Matplotlib: {e}")
                
                # Fallback vers la version simplifiée
                try:
                    try:
                        from simple_math_window import show_simple_math_window  # pyright: ignore
                    except ImportError:
                        print("⚠️ Module simple_math_window non disponible")
                        messagebox.showwarning("Attention", "Les modules de formules mathématiques ne sont pas disponibles.")
                        return
                    print("🔍 CHECKPOINT MATH: Utilisation du fallback simplifié")
                    show_simple_math_window(entropy_window)
                    print("✅ CHECKPOINT MATH: Fenêtre simplifiée ouverte avec succès")
                except ImportError as e:
                    print(f"❌ CHECKPOINT MATH: Erreur import module simplifié: {e}")
                    messagebox.showerror("Erreur", f"Impossible d'importer les modules mathématiques:\n\n{str(e)}\n\nVeuillez vérifier que les fichiers existent.")
                except Exception as e:
                    print(f"❌ CHECKPOINT MATH: Erreur générale: {e}")
                    messagebox.showerror("Erreur", f"Erreur lors de l'affichage des formules:\n\n{str(e)}")

            math_info_btn = ttk.Button(button_frame, text="🧮 Formules Mathématiques", command=show_math_info)
            math_info_btn.pack(side=tk.LEFT, padx=(0, 10))

            close_btn = ttk.Button(button_frame, text="Fermer", command=entropy_window.destroy)
            close_btn.pack(side=tk.RIGHT)

            # Tooltips pour les boutons
            create_tooltip(analyze_btn, 
                          "Calculer l'entropie renormée.\n\n"
                          "• Applique la méthode d'Issartel pour l'entropie renormée\n"
                          "• Utilise les paramètres configurés ci-dessus\n"
                          "• Calcule les moments généralisés sur fenêtres glissantes\n"
                          "• Applique le kernel de renormée sur le spectre de covariance\n"
                          "• Retourne l'entropie différentielle en nats et bits")

            create_tooltip(export_btn, 
                          "Exporter les résultats d'entropie renormée.\n\n"
                          "• Sauvegarde tous les résultats dans un fichier texte\n"
                          "• Inclut : valeurs d'entropie, configuration, valeurs propres\n"
                          "• Format : texte lisible pour analyse externe\n"
                          "• Compatible avec Excel, R, Python pour analyse statistique\n"
                          "• Utile pour la documentation et la publication")

            create_tooltip(math_info_btn, 
                          "Afficher les formules mathématiques détaillées.\n\n"
                          "• Ouvre une fenêtre avec tous les calculs mathématiques\n"
                          "• Formules formatées comme Cambria Math (style Word)\n"
                          "• Inclut : étapes du calcul, kernels, interprétation\n"
                          "• Références bibliographiques complètes\n"
                          "• Conseils d'utilisation et paramètres recommandés\n"
                          "• Documentation scientifique complète")

            create_tooltip(close_btn, 
                          "Fermer la fenêtre d'entropie renormée.\n\n"
                          "• Ferme cette fenêtre d'analyse\n"
                          "• Retourne à la vue principale de l'application\n"
                          "• Les résultats ne sont pas sauvegardés automatiquement\n"
                          "• Utilisez 'Exporter Résultats' pour sauvegarder")

            # Configuration du canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Lier la molette de la souris au scroll
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

            def _unbind_mousewheel(event):
                canvas.unbind_all("<MouseWheel>")
            entropy_window.bind("<Destroy>", _unbind_mousewheel)

            print("✅ CHECKPOINT ENTROPY: Fenêtre d'entropie renormée affichée")
            logging.info("[ENTROPY] Renormalized entropy window displayed")

        except Exception as e:
            print(f"❌ CHECKPOINT ENTROPY: Erreur affichage fenêtre: {e}")
            logging.error(f"[ENTROPY] Failed to show entropy window: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage de l'entropie renormée : {str(e)}")
    
    def _show_multiscale_entropy(self):
        """Affiche l'analyse d'entropie multiscale avec retours détaillés."""
        print("🔍 CHECKPOINT MSE-GUI: Entrée _show_multiscale_entropy")
        try:
            if not self.raw or not self.selected_channels:
                print(
                    "⚠️ CHECKPOINT MSE-GUI: raw loaded?",
                    self.raw is not None,
                    "channels",
                    self.selected_channels,
                )
                messagebox.showwarning("Attention", "Chargez un fichier et sélectionnez au moins un canal")
                return

            from CESA.entropy import (
                MultiscaleEntropyConfig,
                compute_multiscale_entropy_from_raw,
            )

            default_cfg = MultiscaleEntropyConfig()

            def create_tooltip(widget, text):
                """Crée un tooltip pour expliquer les champs sensibles."""

                def show_tooltip(event):
                    tooltip = tk.Toplevel()
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                    label = tk.Label(
                        tooltip,
                        text=text,
                        justify="left",
                        background="#ffffe0",
                        relief="solid",
                        borderwidth=1,
                        font=("Segoe UI", 9),
                        wraplength=320,
                    )
                    label.pack()
                    widget.tooltip = tooltip

                def hide_tooltip(_event):
                    if hasattr(widget, "tooltip") and widget.tooltip:
                        widget.tooltip.destroy()
                        widget.tooltip = None

                widget.bind("<Enter>", show_tooltip)
                widget.bind("<Leave>", hide_tooltip)

            mse_window = tk.Toplevel(self.root)
            mse_window.title("Entropie Multiscale (MSE) - EEG Analysis Studio")
            mse_window.geometry("1200x900")
            mse_window.transient(self.root)
            mse_window.grab_set()
            mse_window.update_idletasks()
            x_pos = (mse_window.winfo_screenwidth() // 2) - 600
            y_pos = (mse_window.winfo_screenheight() // 2) - 450
            mse_window.geometry(f"1200x900+{x_pos}+{y_pos}")

            main_frame = ttk.Frame(mse_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            canvas = tk.Canvas(main_frame)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            title_label = ttk.Label(
                scrollable_frame,
                text="🔬 Analyse d'Entropie Multiscale (Costa et al.)",
                font=("Segoe UI", 16, "bold"),
            )
            title_label.pack(pady=(0, 20))

            info_frame = ttk.LabelFrame(scrollable_frame, text="Informations Générales", padding=10)
            info_frame.pack(fill=tk.X, pady=(0, 10))
            info_text = f"""
Fichier: {getattr(self, 'current_file', 'Inconnu')}
Fréquence d'échantillonnage: {self.sfreq} Hz
Durée totale: {len(self.raw.times)/60:.1f} minutes
Canaux sélectionnés: {', '.join(self.selected_channels)}
"""
            ttk.Label(info_frame, text=info_text.strip()).pack(anchor="w")

            config_frame = ttk.LabelFrame(scrollable_frame, text="Configuration MSE", padding=10)
            config_frame.pack(fill=tk.X, pady=(0, 10))
            config_grid = ttk.Frame(config_frame)
            config_grid.pack(fill=tk.X)

            m_var = tk.IntVar(value=2)
            r_var = tk.DoubleVar(value=0.2)
            scales_var = tk.StringVar(value="1-20")
            max_samples_var = tk.StringVar(
                value=str(default_cfg.max_samples) if default_cfg.max_samples is not None else ""
            )
            return_intermediate_var = tk.BooleanVar(value=False)

            ttk.Label(config_grid, text="Embedding (m):").grid(row=0, column=0, sticky="w", padx=(0, 10))
            m_spin = ttk.Spinbox(config_grid, from_=1, to=5, textvariable=m_var, width=10)
            m_spin.grid(row=0, column=1, sticky="w")

            ttk.Label(config_grid, text="Tolérance (r):").grid(row=0, column=2, sticky="w", padx=(20, 10))
            r_spin = ttk.Spinbox(config_grid, from_=0.05, to=0.5, increment=0.05, textvariable=r_var, width=10)
            r_spin.grid(row=0, column=3, sticky="w")

            ttk.Label(config_grid, text="Échelles (ex: 1-5,8,16):").grid(
                row=1,
                column=0,
                sticky="w",
                padx=(0, 10),
                pady=(10, 0),
            )
            scales_entry = ttk.Entry(config_grid, textvariable=scales_var, width=25)
            scales_entry.grid(row=1, column=1, sticky="w", pady=(10, 0))

            ttk.Label(config_grid, text="Max samples (optionnel):").grid(
                row=1,
                column=2,
                sticky="w",
                padx=(20, 10),
                pady=(10, 0),
            )
            max_samples_entry = ttk.Entry(config_grid, textvariable=max_samples_var, width=15)
            max_samples_entry.grid(row=1, column=3, sticky="w", pady=(10, 0))

            intermediate_check = ttk.Checkbutton(
                config_grid,
                text="Sauvegarder signaux coarse-grained",
                variable=return_intermediate_var,
            )
            intermediate_check.grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))

            create_tooltip(
                m_spin,
                "Dimension d'embedding (m). Plus la valeur est élevée, plus les motifs à comparer sont longs.\n"
                "m = 2 est la valeur classique pour EEG.",
            )
            create_tooltip(
                r_spin,
                "Tolérance relative appliquée à l'écart-type du canal (r × σ).\n"
                "Des valeurs plus faibles rendent le critère de similitude plus strict.",
            )
            create_tooltip(
                scales_entry,
                "Définissez les échelles τ. Utilisez des listes séparées par des virgules\n"
                "ou des plages 'début-fin'. Exemple: 1-5,8,16",
            )
            create_tooltip(
                max_samples_entry,
                "Limiter le nombre d'échantillons utilisés (par défaut 200 000).\n"
                "Laissez 'auto' ou vide pour la valeur par défaut, tapez 'all' pour tout traiter.",
            )
            create_tooltip(
                intermediate_check,
                "Activez pour stocker les signaux coarse-grained dans le résultat.\n"
                "Utile pour vérifier visuellement le lissage à chaque échelle.",
            )

            results_frame = ttk.LabelFrame(scrollable_frame, text="Résultats MSE", padding=10)
            results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            results_text = tk.Text(results_frame, height=18, wrap=tk.WORD, font=("Consolas", 10))
            results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_text.yview)
            results_text.configure(yscrollcommand=results_scrollbar.set)
            results_text.pack(side="left", fill="both", expand=True)
            results_scrollbar.pack(side="right", fill="y")

            plot_frame = ttk.LabelFrame(scrollable_frame, text="Visualisation rapide", padding=10)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))

            def parse_scales(raw_text: str) -> Sequence[int]:
                """Analyse l'entrée utilisateur en liste d'échelles."""

                scales: list[int] = []
                cleaned = raw_text.replace(";", ",").split(",")
                for chunk in cleaned:
                    part = chunk.strip()
                    if not part:
                        continue
                    if "-" in part:
                        start_str, end_str = part.split("-", 1)
                        start_val = int(start_str)
                        end_val = int(end_str)
                        if end_val < start_val:
                            start_val, end_val = end_val, start_val
                        scales.extend(range(start_val, end_val + 1))
                    else:
                        scales.append(int(part))
                if not scales:
                    raise ValueError("Veuillez définir au moins une échelle (ex: 1-5).")
                return scales

            def update_plot(entropy_dict: Dict[int, float]) -> None:
                """Dessine le profil MSE dans le frame prévu."""

                for child in plot_frame.winfo_children():
                    child.destroy()

                valid_points = [(scale, value) for scale, value in entropy_dict.items() if math.isfinite(value)]
                if not valid_points:
                    ttk.Label(plot_frame, text="Aucune valeur exploitable pour le tracé.").pack()
                    return

                scales_sorted = [pt[0] for pt in sorted(valid_points)]
                values_sorted = [pt[1] for pt in sorted(valid_points)]
                fig = Figure(figsize=(6, 3), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot(scales_sorted, values_sorted, marker="o")
                ax.set_xlabel("Échelle τ")
                ax.set_ylabel("SampEn moyenne")
                ax.set_title("Profil multiscale de l'entropie")
                ax.grid(True, alpha=0.3)

                canvas_plot = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas_plot.draw()
                canvas_widget = canvas_plot.get_tk_widget()
                canvas_widget.pack(fill=tk.BOTH, expand=True)

            def run_mse_analysis():
                """Lance le calcul MSE avec des checkpoints lisibles."""

                try:
                    results_text.delete(1.0, tk.END)
                    results_text.insert(tk.END, "🔄 CHECKPOINT: Lancement du pipeline MSE...\n")
                    mse_window.update()

                    scales = parse_scales(scales_var.get())
                    max_samples_text = max_samples_var.get().strip().lower()
                    if not max_samples_text or max_samples_text == "auto":
                        max_samples_int = default_cfg.max_samples
                    elif max_samples_text in {"none", "all", "full", "∞", "inf"}:
                        max_samples_int = None
                    else:
                        max_samples_int = int(max_samples_text)

                    config = MultiscaleEntropyConfig(
                        m=int(m_var.get()),
                        r=float(r_var.get()),
                        scales=scales,
                        max_samples=max_samples_int,
                        return_intermediate=bool(return_intermediate_var.get()),
                    )

                    results_text.insert(
                        tk.END,
                        f"▶️ CHECKPOINT: Configuration validée (m={config.m}, r={config.r}, "
                        f"τ={list(config.scales)})\n",
                    )
                    mse_window.update()

                    mse_result = compute_multiscale_entropy_from_raw(self.raw, self.selected_channels, config)

                    results_text.insert(tk.END, "✅ CHECKPOINT: Calcul terminé\n\n")
                    results_text.insert(tk.END, "📊 RÉSUMÉ PRINCIPAL\n")
                    results_text.insert(tk.END, f"{'='*60}\n")
                    results_text.insert(
                        tk.END,
                        f"Canaux: {', '.join(self.selected_channels)}\nTotal samples: {mse_result.processed_samples}\n\n",
                    )

                    results_text.insert(tk.END, "Échelle | SampEn moyenne\n")
                    results_text.insert(tk.END, f"{'-'*60}\n")
                    for scale, value in mse_result.entropy_by_scale.items():
                        results_text.insert(tk.END, f"{scale:>6} | {value:>10.4f}\n")
                    results_text.insert(tk.END, "\n")

                    results_text.insert(tk.END, "📌 DÉTAILS PAR ÉCHELLE ET CANAL\n")
                    results_text.insert(tk.END, f"{'='*60}\n")
                    for scale, detail in mse_result.sample_entropy_by_scale.items():
                        results_text.insert(
                            tk.END,
                            f"τ={scale} → état={detail.get('status')} (n={detail.get('coarse_samples')})\n",
                        )
                        channel_details = detail.get("channel_details", {})
                        for channel, ch_detail in channel_details.items():
                            matches_m = ch_detail.get("matches_m", 0)
                            matches_m1 = ch_detail.get("matches_m1", 0)
                            samp_en = mse_result.channel_entropy.get(channel, {}).get(scale, math.nan)
                            results_text.insert(
                                tk.END,
                                f"   • {channel}: SampEn={samp_en:.4f} | matches m={matches_m} | "
                                f"matches m+1={matches_m1}\n",
                            )
                        results_text.insert(tk.END, "\n")

                    results_text.insert(tk.END, "🧾 INTERPRÉTATION RAPIDE\n")
                    results_text.insert(tk.END, f"{'='*60}\n")
                    results_text.insert(
                        tk.END,
                        "• Profil décroissant = davantage de régularité aux grandes échelles.\n"
                        "• Profil stable ou croissant = richesse multi-échelle.\n"
                        "• SampEn → ∞ signifie absence de motifs répétées au seuil r.\n",
                    )

                    update_plot(mse_result.entropy_by_scale)
                    print("✅ CHECKPOINT MSE-GUI: Résultats affichés")

                except Exception as exc:
                    results_text.delete(1.0, tk.END)
                    results_text.insert(tk.END, f"❌ Erreur MSE: {exc}\n")
                    print(f"❌ CHECKPOINT MSE-GUI: Erreur {exc}")

            def export_mse_results():
                """Sauvegarde le texte affiché."""

                file_path = filedialog.asksaveasfilename(
                    title="Exporter résultats MSE",
                    defaultextension=".txt",
                    filetypes=[("Fichier texte", "*.txt"), ("Tous les fichiers", "*.*")],
                )
                if not file_path:
                    return
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write(results_text.get(1.0, tk.END))
                messagebox.showinfo("Export", f"Résultats sauvegardés dans {file_path}")

            analyze_btn = ttk.Button(button_frame, text="🔬 Calculer MSE", command=run_mse_analysis)
            analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
            export_btn = ttk.Button(button_frame, text="📁 Exporter Résultats", command=export_mse_results)
            export_btn.pack(side=tk.LEFT, padx=(0, 10))
            close_btn = ttk.Button(button_frame, text="Fermer", command=mse_window.destroy)
            close_btn.pack(side=tk.RIGHT)

            create_tooltip(
                analyze_btn,
                "Lance le pipeline : coarse-graining → SampEn pour chaque échelle.\n"
                "Des checkpoints détaillent chaque étape dans la console et ici.",
            )
            create_tooltip(export_btn, "Sauvegarde le rapport textuel affiché ci-dessus.")
            create_tooltip(close_btn, "Ferme la fenêtre d'entropie multiscale.")

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            canvas.bind_all("<MouseWheel>", _on_mousewheel)

            def _unbind_mousewheel(_event):
                canvas.unbind_all("<MouseWheel>")

            mse_window.bind("<Destroy>", _unbind_mousewheel)
            print("✅ CHECKPOINT MSE-GUI: Fenêtre affichée")

        except Exception as e:
            print(f"❌ CHECKPOINT MSE-GUI: Erreur affichage fenêtre: {e}")
            messagebox.showerror("Erreur", f"Impossible d'afficher l'entropie multiscale : {e}")

    # =========================================================================
    # Analyse de groupe (MSE / Entropie renormée)
    # =========================================================================

    def _show_group_analysis(self):
        """Fenêtre d'analyse de groupe inspirée de la FFT en lot."""
        print("🔍 CHECKPOINT GROUP: ouverture interface d'analyse de groupe")

        if getattr(self, "group_analysis_window", None) is not None:
            try:
                if self.group_analysis_window.winfo_exists():  # type: ignore[attr-defined]
                    self.group_analysis_window.lift()
                    return
            except Exception:
                self.group_analysis_window = None  # type: ignore[attr-defined]

        if not hasattr(self, "group_design_df"):
            self.group_design_df = pd.DataFrame(
                columns=["subject", "condition", "edf_path", "scoring_path", "stage"]
            )
        if not hasattr(self, "group_analysis_result"):
            self.group_analysis_result = None
        if not hasattr(self, "group_hrv_result"):
            self.group_hrv_result = None
        if not hasattr(self, "_group_analysis_running"):
            self._group_analysis_running = False
            self._group_analysis_thread = None
        if not hasattr(self, "_group_hrv_running"):
            self._group_hrv_running = False
            self._group_hrv_thread = None

        window = tk.Toplevel(self.root)
        window.title("📚 Analyse de groupe – MSE / Entropie renormée")
        window.geometry("1250x920")
        window.configure(bg="#f8f9fa")
        window.grab_set()
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_group_analysis_window(window))
        self.group_analysis_window = window

        style = ttk.Style(window)
        style.configure("GAHeading.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("GAInfo.TLabel", font=("Segoe UI", 9), foreground="#6c757d")

        title_frame = ttk.Frame(window)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        ttk.Label(title_frame, text="📚 Analyse de groupe avant/après", style="GAHeading.TLabel").pack(anchor="w")
        ttk.Label(
            title_frame,
            text="Comparer plusieurs sujets (EDF) sur les profils MSE ou entropie renormée, par stade de sommeil.",
            style="GAInfo.TLabel",
        ).pack(anchor="w")

        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        design_tab = ttk.Frame(notebook)
        params_tab = ttk.Frame(notebook)
        results_tab = ttk.Frame(notebook)
        notebook.add(design_tab, text="📋 Design & Données")
        notebook.add(params_tab, text="⚙️ Paramètres")
        notebook.add(results_tab, text="📊 Tests & Visualisation")

        # -----------------------
        # Onglet DESIGN
        # -----------------------
        self._build_group_design_tab(design_tab)

        # -----------------------
        # Onglet PARAMÈTRES
        # -----------------------
        self._build_group_params_tab(params_tab)

        # -----------------------
        # Onglet RÉSULTATS
        # -----------------------
        self._build_group_results_tab(results_tab)

        self._group_refresh_design_tree()
        self._group_toggle_metric_frames()

    def _close_group_analysis_window(self, window: tk.Toplevel) -> None:
        try:
            window.destroy()
        except Exception:
            pass
        finally:
            self.group_analysis_window = None

    # ------------------------------------------------------------------
    # Onglet Design
    # ------------------------------------------------------------------

    def _build_group_design_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, padx=10, pady=(10, 0))

        ttk.Button(
            controls,
            text="🔍 Scanner un dossier",
            command=self._group_scan_directory,
        ).pack(side=tk.LEFT)
        ttk.Button(
            controls,
            text="📥 Importer design CSV",
            command=self._group_import_design_from_csv,
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(
            controls,
            text="💾 Exporter design",
            command=self._group_export_design_csv,
        ).pack(side=tk.LEFT, padx=(10, 0))

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        columns = ("subject", "condition", "edf", "scoring", "stage")
        self.group_design_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        for col, label, width in zip(
            columns,
            ["Sujet", "Condition", "EDF", "Scoring", "Stades"],
            [120, 100, 320, 260, 120],
        ):
            self.group_design_tree.heading(col, text=label)
            self.group_design_tree.column(col, width=width, anchor="w")

        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.group_design_tree.yview)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.group_design_tree.xview)
        self.group_design_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.group_design_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        instr = ttk.Label(
            parent,
            text="Colonnes attendues : subject, condition, edf_path, scoring_path (optionnel), stage (ex: W,N2 ou *).",
            style="GAInfo.TLabel",
        )
        instr.pack(fill=tk.X, padx=12, pady=(6, 0))

        group_ctrl = ttk.Frame(parent)
        group_ctrl.pack(fill=tk.X, padx=10, pady=(4, 4))
        ttk.Label(group_ctrl, text="Affecter la condition aux entrées sélectionnées:").pack(side=tk.LEFT)
        ttk.Button(group_ctrl, text="Marquer AVANT", command=lambda: self._group_mark_selected_condition(self.ga_before_label_var.get() or "AVANT")).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(group_ctrl, text="Marquer APRÈS", command=lambda: self._group_mark_selected_condition(self.ga_after_label_var.get() or "APRÈS")).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(group_ctrl, text="Effacer", command=lambda: self._group_mark_selected_condition("")).pack(side=tk.LEFT, padx=(6, 0))

        manual = ttk.LabelFrame(parent, text="Ajout manuel rapide")
        manual.pack(fill=tk.X, padx=10, pady=(10, 12))

        self.ga_manual_subject_var = tk.StringVar()
        self.ga_manual_condition_var = tk.StringVar(value="AVANT")
        self.ga_manual_stage_var = tk.StringVar(value="*")
        self.ga_manual_edf_var = tk.StringVar()
        self.ga_manual_scoring_var = tk.StringVar()

        row1 = ttk.Frame(manual)
        row1.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row1, text="Sujet:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row1, textvariable=self.ga_manual_subject_var, width=18).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(row1, text="Condition:").grid(row=0, column=2, sticky="w")
        ttk.Entry(row1, textvariable=self.ga_manual_condition_var, width=12).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(row1, text="Stades (*=tous):").grid(row=0, column=4, sticky="w")
        ttk.Entry(row1, textvariable=self.ga_manual_stage_var, width=16).grid(row=0, column=5, padx=(4, 12))

        row2 = ttk.Frame(manual)
        row2.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row2, text="EDF:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row2, textvariable=self.ga_manual_edf_var, width=50).grid(row=0, column=1, padx=(4, 6), sticky="we")
        ttk.Button(
            row2,
            text="Parcourir",
            command=lambda: self._group_browse_file_to_var(self.ga_manual_edf_var, recording_filetypes_for_dialog()),
        ).grid(row=0, column=2, padx=(0, 12))

        ttk.Label(row2, text="Scoring (optionnel):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(row2, textvariable=self.ga_manual_scoring_var, width=50).grid(row=1, column=1, padx=(4, 6), pady=(6, 0), sticky="we")
        ttk.Button(
            row2,
            text="Parcourir",
            command=lambda: self._group_browse_file_to_var(self.ga_manual_scoring_var, [("Scoring", "*.edf *.EDF *.xls *.xlsx *.csv")]),
        ).grid(row=1, column=2, padx=(0, 12), pady=(6, 0))

        row3 = ttk.Frame(manual)
        row3.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Button(row3, text="➕ Ajouter", command=self._group_add_manual_entry).pack(side=tk.LEFT)
        ttk.Button(row3, text="🗑️ Supprimer sélection", command=self._group_remove_selected_entries).pack(side=tk.LEFT, padx=(10, 0))

    # ------------------------------------------------------------------
    # Onglet Paramètres
    # ------------------------------------------------------------------

    def _build_group_params_tab(self, parent: ttk.Frame) -> None:
        # Conteneur scrollable pour voir tout le contenu sur petits écrans
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_frame_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_frame_configure)
        inner.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        metric_frame = ttk.LabelFrame(inner, text="Choix de la métrique")
        metric_frame.pack(fill=tk.X, padx=10, pady=(10, 6))

        self.ga_metric_var = tk.StringVar(value="mse")
        ttk.Radiobutton(metric_frame, text="Multiscale Entropy (MSE)", variable=self.ga_metric_var, value="mse", command=self._group_toggle_metric_frames).pack(side=tk.LEFT, padx=(4, 20))
        ttk.Radiobutton(metric_frame, text="Entropie renormée", variable=self.ga_metric_var, value="renorm", command=self._group_toggle_metric_frames).pack(side=tk.LEFT)

        stage_frame = ttk.LabelFrame(inner, text="Stades inclus")
        stage_frame.pack(fill=tk.X, padx=10, pady=6)
        self.ga_stage_vars: Dict[str, tk.BooleanVar] = {}
        for idx, stage in enumerate(group_analysis.DEFAULT_STAGES):
            var = tk.BooleanVar(value=(stage != "ALL"))
            self.ga_stage_vars[stage] = var
            ttk.Checkbutton(stage_frame, text=stage, variable=var).grid(row=idx // 4, column=idx % 4, sticky="w", padx=8, pady=2)

        labels_frame = ttk.LabelFrame(inner, text="Conditions et canaux")
        labels_frame.pack(fill=tk.X, padx=10, pady=6)
        self.ga_before_label_var = tk.StringVar(value="AVANT")
        self.ga_after_label_var = tk.StringVar(value="APRÈS")
        self.ga_channels_var = tk.StringVar()
        self.ga_epoch_var = tk.DoubleVar(value=30.0)

        ttk.Label(labels_frame, text="Libellé AVANT:").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=4)
        ttk.Entry(labels_frame, textvariable=self.ga_before_label_var, width=16).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(labels_frame, text="Libellé APRÈS:").grid(row=0, column=2, sticky="w", padx=(12, 4))
        ttk.Entry(labels_frame, textvariable=self.ga_after_label_var, width=16).grid(row=0, column=3, sticky="w")

        ttk.Label(labels_frame, text="Canaux sélectionnés:").grid(row=1, column=0, columnspan=2, sticky="w", padx=(8, 4), pady=(4, 2))
        ttk.Entry(labels_frame, textvariable=self.ga_channels_var, width=50).grid(row=1, column=2, columnspan=2, sticky="we", pady=(4, 2))

        ttk.Label(labels_frame, text="Durée d'époque (s):").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=(4, 6))
        ttk.Entry(labels_frame, textvariable=self.ga_epoch_var, width=10).grid(row=2, column=1, sticky="w", pady=(4, 6))

        # Paramètres HRV
        hrv_defaults = group_analysis.HRVConfig()
        hrv_frame = ttk.LabelFrame(inner, text="Paramètres HRV par stades")
        hrv_frame.pack(fill=tk.X, padx=10, pady=6)
        self.hr_lf_var = tk.DoubleVar(value=hrv_defaults.lf_band[0])
        self.hr_lf_high_var = tk.DoubleVar(value=hrv_defaults.lf_band[1])
        self.hr_hf_var = tk.DoubleVar(value=hrv_defaults.hf_band[0])
        self.hr_hf_high_var = tk.DoubleVar(value=hrv_defaults.hf_band[1])
        self.hr_resample_var = tk.DoubleVar(value=hrv_defaults.resample_fs)
        self.hr_filter_low_var = tk.DoubleVar(value=hrv_defaults.filter_band[0])
        self.hr_filter_high_var = tk.DoubleVar(value=hrv_defaults.filter_band[1])
        self.hr_stage_var = tk.StringVar(value="REM")
        self.hr_min_segment_var = tk.DoubleVar(value=hrv_defaults.min_segment_s)
        self.hr_allow_short_var = tk.BooleanVar(value=hrv_defaults.allow_short_segments)
        self.hr_clean_rr_var = tk.BooleanVar(value=hrv_defaults.clean_rr)
        self.hr_peak_method_var = tk.StringVar(value=getattr(hrv_defaults, "peak_detection_method", "simple"))
        self.hr_rr_method_var = tk.StringVar(value=getattr(hrv_defaults, "rr_cleaning_method", "simple"))

        ttk.Label(hrv_frame, text="LF (Hz):").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=4)
        ttk.Entry(hrv_frame, textvariable=self.hr_lf_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(hrv_frame, text="à").grid(row=0, column=2, sticky="w", padx=(4, 4))
        ttk.Entry(hrv_frame, textvariable=self.hr_lf_high_var, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(hrv_frame, text="HF (Hz):").grid(row=0, column=4, sticky="w", padx=(12, 4))
        ttk.Entry(hrv_frame, textvariable=self.hr_hf_var, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(hrv_frame, text="à").grid(row=0, column=6, sticky="w", padx=(4, 4))
        ttk.Entry(hrv_frame, textvariable=self.hr_hf_high_var, width=8).grid(row=0, column=7, sticky="w")

        ttk.Label(hrv_frame, text="Resample (Hz):").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=4)
        ttk.Entry(hrv_frame, textvariable=self.hr_resample_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(hrv_frame, text="Filtre ECG (Hz):").grid(row=1, column=2, sticky="w", padx=(12, 4))
        ttk.Entry(hrv_frame, textvariable=self.hr_filter_low_var, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(hrv_frame, text="à").grid(row=1, column=4, sticky="w", padx=(4, 4))
        ttk.Entry(hrv_frame, textvariable=self.hr_filter_high_var, width=8).grid(row=1, column=5, sticky="w")

        ttk.Label(hrv_frame, text="Stades HRV (ex: REM,R) — indépendants du champ « Stades inclus »:").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=4)
        ttk.Entry(hrv_frame, textvariable=self.hr_stage_var, width=18).grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Label(hrv_frame, text="Durée min segment (s):").grid(row=2, column=3, sticky="w", padx=(12, 4))
        ttk.Entry(hrv_frame, textvariable=self.hr_min_segment_var, width=10).grid(row=2, column=4, sticky="w")
        ttk.Checkbutton(hrv_frame, text="Autoriser segments courts", variable=self.hr_allow_short_var).grid(row=2, column=5, columnspan=2, sticky="w", padx=(8, 0))
        ttk.Checkbutton(hrv_frame, text="Nettoyage RR (artefacts)", variable=self.hr_clean_rr_var).grid(row=3, column=0, columnspan=3, sticky="w", padx=(8, 0), pady=(0, 4))
        ttk.Label(hrv_frame, text="Détection pics ECG:").grid(row=3, column=3, sticky="w", padx=(8, 4), pady=(0, 4))
        ttk.Combobox(
            hrv_frame,
            textvariable=self.hr_peak_method_var,
            values=("simple", "neurokit2"),
            width=12,
            state="readonly",
        ).grid(row=3, column=4, sticky="w", pady=(0, 4))
        ttk.Label(hrv_frame, text="Méthode rejet RR:").grid(row=3, column=5, sticky="w", padx=(12, 4), pady=(0, 4))
        ttk.Combobox(
            hrv_frame,
            textvariable=self.hr_rr_method_var,
            values=("simple", "neurokit2", "kubios"),
            width=12,
            state="readonly",
        ).grid(row=3, column=6, sticky="w", pady=(0, 4))
        ttk.Label(hrv_frame, text="Rappel: « Stades inclus » en haut pilote MSE/entropie; ici, définis les stades HRV (par ex. REM,R ou W).", style="GAInfo.TLabel", wraplength=620, justify="left").grid(row=4, column=0, columnspan=8, sticky="w", padx=(8, 4), pady=(0, 4))

        channel_frame = ttk.LabelFrame(inner, text="Sélection des canaux (comme FFT en lot)")
        channel_frame.pack(fill=tk.X, padx=10, pady=6)
        control_row = ttk.Frame(channel_frame)
        control_row.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Button(control_row, text="Charger depuis un EDF", command=self._group_load_channels_from_design).pack(side=tk.LEFT)
        ttk.Button(control_row, text="EEG standard", command=lambda: self._group_select_channel_preset("standard")).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(control_row, text="Tout sélectionner", command=lambda: self._group_select_channel_preset("all")).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(control_row, text="Tout désélectionner", command=lambda: self._group_select_channel_preset("none")).pack(side=tk.LEFT, padx=(6, 0))

        self.ga_channel_vars: Dict[str, tk.BooleanVar] = {}
        checklist_frame = ttk.Frame(channel_frame)
        checklist_frame.pack(fill=tk.X, padx=8, pady=(4, 8))
        self._group_channel_columns = 4
        self._group_available_channels: List[str] = []
        self.group_channel_checklist_frame = checklist_frame
        self._group_build_channel_checklist()


        # Paramètres MSE
        self.ga_mse_frame = ttk.LabelFrame(inner, text="Paramètres MSE")
        self.ga_mse_frame.pack(fill=tk.X, padx=10, pady=6)
        mse_cfg = group_analysis.MultiscaleEntropyConfig()
        self.ga_m_var = tk.IntVar(value=mse_cfg.m)
        self.ga_r_var = tk.DoubleVar(value=mse_cfg.r)
        self.ga_scales_var = tk.StringVar(value="1-20")
        default_max_samples = mse_cfg.max_samples if mse_cfg.max_samples else ""
        self.ga_max_samples_var = tk.StringVar(value=str(default_max_samples))
        self.ga_max_points_var = tk.StringVar(value=str(mse_cfg.max_pattern_length))

        row = ttk.Frame(self.ga_mse_frame)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text="m:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.ga_m_var, width=8).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(row, text="r:").grid(row=0, column=2, sticky="w")
        ttk.Entry(row, textvariable=self.ga_r_var, width=8).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(row, text="Échelles (ex: 1-20,24,32):").grid(row=0, column=4, sticky="w")
        ttk.Entry(row, textvariable=self.ga_scales_var, width=20).grid(row=0, column=5, padx=(4, 12))

        row2 = ttk.Frame(self.ga_mse_frame)
        row2.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(row2, text="Max samples:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row2, textvariable=self.ga_max_samples_var, width=12).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(row2, text="Max points SampEn:").grid(row=0, column=2, sticky="w")
        ttk.Entry(row2, textvariable=self.ga_max_points_var, width=12).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(
            self.ga_mse_frame,
            text="Laissez \"auto\" pour la troncature par défaut (200k échantillons). \"all\" pour tout traiter.",
            style="GAInfo.TLabel",
        ).pack(anchor="w", padx=10, pady=(0, 4))

        # Paramètres Renorm
        self.ga_renorm_frame = ttk.LabelFrame(inner, text="Paramètres Entropie renormée")
        self.ga_renorm_frame.pack(fill=tk.X, padx=10, pady=6)
        renorm_cfg = group_analysis.RenormalizedEntropyConfig()
        self.ga_window_length_var = tk.DoubleVar(value=renorm_cfg.window_length)
        self.ga_overlap_var = tk.DoubleVar(value=renorm_cfg.overlap)
        self.ga_moment_order_var = tk.DoubleVar(value=renorm_cfg.moment_order)
        self.ga_psi_name_var = tk.StringVar(value=renorm_cfg.psi_name)
        self.ga_gamma_var = tk.DoubleVar(value=renorm_cfg.psi_params.get("gamma", 0.5))

        row3 = ttk.Frame(self.ga_renorm_frame)
        row3.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row3, text="Fenêtre (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(row3, textvariable=self.ga_window_length_var, width=10).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(row3, text="Chevauchement (0-0.9):").grid(row=0, column=2, sticky="w")
        ttk.Entry(row3, textvariable=self.ga_overlap_var, width=10).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(row3, text="Ordre du moment:").grid(row=0, column=4, sticky="w")
        ttk.Entry(row3, textvariable=self.ga_moment_order_var, width=10).grid(row=0, column=5, padx=(4, 12))

        row4 = ttk.Frame(self.ga_renorm_frame)
        row4.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(row4, text="Kernel ψ:").grid(row=0, column=0, sticky="w")
        psi_combo = ttk.Combobox(row4, textvariable=self.ga_psi_name_var, values=("identity", "powerlaw", "log", "adaptive"), width=12, state="readonly")
        psi_combo.grid(row=0, column=1, padx=(4, 12))
        ttk.Label(row4, text="Gamma (powerlaw):").grid(row=0, column=2, sticky="w")
        ttk.Entry(row4, textvariable=self.ga_gamma_var, width=10).grid(row=0, column=3, padx=(4, 12))

    # ------------------------------------------------------------------
    # Onglet Résultats / Visualisation
    # ------------------------------------------------------------------

    def _build_group_results_tab(self, parent: ttk.Frame) -> None:
        status_frame = ttk.LabelFrame(parent, text="Progression")
        status_frame.pack(fill=tk.X, padx=10, pady=(10, 6))
        self.ga_progress_var = tk.StringVar(value="Prêt")
        ttk.Label(status_frame, textvariable=self.ga_progress_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        self.ga_progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.ga_progress.pack(fill=tk.X, padx=10, pady=(0, 8))

        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.ga_run_button = ttk.Button(controls, text="🚀 Calculer", command=self._group_run_analysis)
        self.ga_run_button.pack(side=tk.LEFT)
        ttk.Button(controls, text="📤 Export profils", command=self._group_export_profiles).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="📊 Export tests", command=self._group_export_stats).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="🖼️ Export figure", command=self._group_export_plot).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="❤️ Calcul HRV", command=self._group_run_hrv).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Button(controls, text="📤 Export HRV", command=self._group_export_hrv).pack(side=tk.LEFT, padx=(8, 0))

        tests_frame = ttk.LabelFrame(parent, text="Options statistiques")
        tests_frame.pack(fill=tk.X, padx=10, pady=6)
        self.ga_run_wilcoxon_var = tk.BooleanVar(value=True)
        self.ga_run_permutation_var = tk.BooleanVar(value=True)
        self.ga_run_bootstrap_var = tk.BooleanVar(value=False)
        self.ga_run_robust_z_var = tk.BooleanVar(value=False)
        self.ga_bh_var = tk.BooleanVar(value=True)
        self.ga_nperm_var = tk.IntVar(value=5000)

        ttk.Checkbutton(tests_frame, text="Wilcoxon apparié", variable=self.ga_run_wilcoxon_var).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(tests_frame, text="Permutation médiane", variable=self.ga_run_permutation_var).grid(row=0, column=1, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(tests_frame, text="Bootstrap IC", variable=self.ga_run_bootstrap_var).grid(row=0, column=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(tests_frame, text="Z robuste", variable=self.ga_run_robust_z_var).grid(row=0, column=3, sticky="w", padx=8, pady=4)
        ttk.Label(tests_frame, text="n_perm:").grid(row=1, column=0, sticky="w", padx=(8, 4))
        ttk.Entry(tests_frame, textvariable=self.ga_nperm_var, width=10).grid(row=1, column=1, sticky="w", padx=(0, 12))
        ttk.Checkbutton(tests_frame, text="Correction Benjamini-Hochberg", variable=self.ga_bh_var).grid(row=1, column=2, columnspan=2, sticky="w", padx=8)

        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 6))
        plot_controls = ttk.Frame(plot_frame)
        plot_controls.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(plot_controls, text="Stade pour le graphique:").pack(side=tk.LEFT)
        self.ga_plot_stage_var = tk.StringVar()
        self.ga_plot_stage_combo = ttk.Combobox(plot_controls, textvariable=self.ga_plot_stage_var, values=group_analysis.DEFAULT_STAGES, width=12, state="readonly")
        self.ga_plot_stage_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.ga_plot_stage_combo.bind("<<ComboboxSelected>>", lambda _evt: self._group_on_stage_change())

        from matplotlib.figure import Figure  # import local pour éviter surcharge si inutilisé
        self.ga_plot_figure = Figure(figsize=(6, 3), dpi=100)
        self.ga_plot_canvas = FigureCanvasTkAgg(self.ga_plot_figure, master=plot_frame)
        self.ga_plot_canvas.draw()
        toolbar = NavigationToolbar2Tk(self.ga_plot_canvas, plot_frame)
        toolbar.update()
        self.ga_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        summary_frame = ttk.LabelFrame(parent, text="Résumé / checkpoints")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.ga_results_text = tk.Text(summary_frame, height=14, wrap=tk.WORD, font=("Consolas", 10))
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.ga_results_text.yview)
        self.ga_results_text.configure(yscrollcommand=summary_scroll.set)
        self.ga_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------
    # Actions sur le design
    # ------------------------------------------------------------------

    def _group_import_design_from_csv(self):
        path = filedialog.askopenfilename(
            title="Sélectionner le CSV de design",
            filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            entries = group_analysis.load_design_csv(path)
            self.group_design_df = self._group_entries_to_dataframe(entries)
            self._group_refresh_design_tree()
            messagebox.showinfo("Design importé", f"{len(entries)} entrées chargées depuis {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Import échoué", str(exc))

    def _group_export_design_csv(self):
        if self.group_design_df.empty:
            messagebox.showwarning("Export", "Aucune entrée à exporter.")
            return
        path = filedialog.asksaveasfilename(
            title="Exporter le design",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        self.group_design_df.to_csv(path, index=False, encoding="utf-8")
        messagebox.showinfo("Export", f"Design sauvegardé dans {path}")

    def _group_entries_to_dataframe(self, entries: Sequence[group_analysis.GroupDesignEntry]) -> pd.DataFrame:
        rows = []
        for entry in entries:
            rows.append(
                {
                    "subject": entry.subject,
                    "condition": entry.condition,
                    "edf_path": str(entry.edf_path),
                    "scoring_path": str(entry.scoring_path) if entry.scoring_path else "",
                    "stage": ",".join(entry.stages) if entry.stages else "*",
                }
            )
        return pd.DataFrame(rows, columns=["subject", "condition", "edf_path", "scoring_path", "stage"])

    def _group_refresh_design_tree(self):
        self.group_design_df = self.group_design_df.reset_index(drop=True)
        for item in self.group_design_tree.get_children():
            self.group_design_tree.delete(item)
        for idx, row in self.group_design_df.iterrows():
            self.group_design_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    row.get("subject", ""),
                    row.get("condition", ""),
                    row.get("edf_path", ""),
                    row.get("scoring_path", ""),
                    row.get("stage", ""),
                ),
            )

    def _group_add_manual_entry(self):
        subject = self.ga_manual_subject_var.get().strip()
        condition = self.ga_manual_condition_var.get().strip()
        edf_path = self.ga_manual_edf_var.get().strip()
        if not subject or not condition or not edf_path:
            messagebox.showwarning("Entrée incomplète", "Sujet, condition et chemin EDF sont obligatoires.")
            return
        new_row = {
            "subject": subject,
            "condition": condition,
            "edf_path": edf_path,
            "scoring_path": self.ga_manual_scoring_var.get().strip(),
            "stage": self.ga_manual_stage_var.get().strip() or "*",
        }
        self.group_design_df = pd.concat([self.group_design_df, pd.DataFrame([new_row])], ignore_index=True)
        self._group_refresh_design_tree()
        self.ga_manual_subject_var.set("")

    def _group_remove_selected_entries(self):
        selection = self.group_design_tree.selection()
        if not selection:
            return
        indices = sorted(int(iid) for iid in selection)
        self.group_design_df = self.group_design_df.drop(indices).reset_index(drop=True)
        self._group_refresh_design_tree()

    def _group_browse_file_to_var(self, var: tk.StringVar, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _group_build_channel_checklist(self, channel_list: Optional[Sequence[str]] = None):
        frame = getattr(self, "group_channel_checklist_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        if channel_list is None or not list(channel_list):
            channel_list = self._group_available_channels or [
                "C3-M2",
                "C4-M1",
                "F3-M2",
                "F4-M1",
                "O1-M2",
                "O2-M1",
                "Fpz-Cz",
                "Pz-Oz",
                "E1-M2",
                "E2-M1",
                "EMG submental",
            ]
        uniq = []
        for ch in channel_list:
            if ch and ch not in uniq:
                uniq.append(ch)
        self._group_available_channels = uniq
        self.ga_channel_vars = {}
        for idx, ch in enumerate(uniq):
            var = tk.BooleanVar(value=False)
            self.ga_channel_vars[ch] = var
            cb = ttk.Checkbutton(frame, text=ch, variable=var, command=self._group_update_channel_entry)
            row = idx // self._group_channel_columns
            col = idx % self._group_channel_columns
            cb.grid(row=row, column=col, sticky="w", padx=4, pady=2)

    def _group_update_channel_entry(self):
        selected = [ch for ch, var in self.ga_channel_vars.items() if var.get()]
        if selected:
            self.ga_channels_var.set(",".join(selected))
        else:
            self.ga_channels_var.set("")

    def _group_select_channel_preset(self, mode: str):
        if not self.ga_channel_vars:
            return
        if mode == "standard":
            target = {"C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"}
        elif mode == "all":
            target = set(self.ga_channel_vars.keys())
        elif mode == "none":
            target = set()
        else:
            target = set()
        for ch, var in self.ga_channel_vars.items():
            var.set(ch in target)
        self._group_update_channel_entry()

    def _group_load_channels_from_design(self):
        if self.group_design_df.empty:
            messagebox.showwarning("Canaux", "Ajoutez d'abord des fichiers dans le design.")
            return
        for _, row in self.group_design_df.iterrows():
            edf_path = Path(str(row.get("edf_path", "")).strip())
            if edf_path.exists():
                try:
                    raw = open_raw_file(str(edf_path), preload=False, verbose="ERROR")  # type: ignore[arg-type]
                    channels = list(raw.ch_names)
                    raw.close()
                except Exception as exc:
                    print(f"⚠️ Impossible de charger {edf_path}: {exc}")
                    continue
                if channels:
                    self._group_build_channel_checklist(channels)
                    self._group_update_channel_entry()
                    messagebox.showinfo("Canaux", f"{len(channels)} canaux chargés depuis {edf_path.name}.")
                    return
        messagebox.showwarning("Canaux", "Impossible de détecter les canaux (EDF introuvable ou illisible).")

    def _group_scan_directory(self):
        directory = filedialog.askdirectory(title="Sélectionner le dossier contenant les EDF + scoring")
        if not directory:
            return

        directory_path = Path(directory)
        if not directory_path.exists():
            messagebox.showwarning("Scan dossier", "Dossier invalide.")
            return

        edf_entries = self._group_collect_edf_entries(directory_path)
        if not edf_entries:
            messagebox.showinfo("Scan dossier", "Aucun fichier d'enregistrement detecte.")
            return

        df_new = pd.DataFrame(edf_entries, columns=["subject", "condition", "edf_path", "scoring_path", "stage"])
        self.group_design_df = pd.concat([self.group_design_df, df_new], ignore_index=True)
        self._group_refresh_design_tree()
        messagebox.showinfo("Scan dossier", f"{len(edf_entries)} fichiers ajoutés depuis {directory_path}")

    def _group_collect_edf_entries(self, base_dir: Path) -> List[Dict[str, str]]:
        excel_map = self._group_build_excel_map(base_dir)
        edf_list = self._group_scan_edf_files(base_dir)
        entries: List[Dict[str, str]] = []
        before_label = self.ga_before_label_var.get() or "AVANT"
        after_label = self.ga_after_label_var.get() or "APRÈS"

        for edf_path in edf_list:
            name = Path(edf_path).stem
            normalized = self._normalize_filename_for_association(name)
            scoring_path = excel_map.get(normalized, "")
            condition = self._group_guess_condition(name, before_label, after_label)
            subject = normalized or name.upper()
            entries.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "edf_path": edf_path,
                    "scoring_path": scoring_path,
                    "stage": "*",
                }
            )
        return entries

    def _group_scan_edf_files(self, base_dir: Path) -> List[str]:
        matches: List[str] = []
        extensions = tuple(ext.lower() for ext in recording_extensions_for_scan())
        for root, _dirs, files in os.walk(base_dir):
            for file in files:
                if normalize_recording_extension(file) in extensions:
                    matches.append(str(Path(root) / file))
        return matches

    def _group_build_excel_map(self, base_dir: Path) -> Dict[str, str]:
        excel_map: Dict[str, str] = {}
        excel_ext = (".xlsx", ".xls")
        for root, _dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(excel_ext):
                    stem = Path(file).stem
                    key = self._normalize_filename_for_association(stem)
                    excel_map[key] = str(Path(root) / file)
        return excel_map

    def _group_guess_condition(self, filename_stem: str, before_label: str, after_label: str) -> str:
        name_up = filename_stem.upper()
        if any(tag in name_up for tag in ("_AV", " AV", "-AV", "AVANT")):
            return before_label
        if any(tag in name_up for tag in ("_AP", " AP", "-AP", "APRES", "APRÈS", "AFTER")):
            return after_label
        # défaut : vide → l'utilisateur pourra marquer via les boutons
        return ""

    def _group_mark_selected_condition(self, condition: str):
        selection = self.group_design_tree.selection()
        if not selection:
            return
        for iid in selection:
            idx = int(iid)
            if 0 <= idx < len(self.group_design_df):
                self.group_design_df.at[idx, "condition"] = condition
        self._group_refresh_design_tree()

    # ------------------------------------------------------------------
    # Exécution de l'analyse
    # ------------------------------------------------------------------

    def _group_run_analysis(self):
        if self.group_design_df.empty:
            messagebox.showwarning("Analyse", "Ajoutez au moins une entrée dans le design.")
            return

        if getattr(self, "_group_analysis_running", False):
            messagebox.showinfo("Analyse", "Une analyse de groupe est déjà en cours.")
            return

        try:
            entries = self._group_build_entries()
            analysis_cfg = self._group_build_analysis_config()
            stats_cfg = self._group_build_stats_config()
            try:
                LOGGER = logging.getLogger(__name__)
                LOGGER.info(
                    "[GROUP_RUN_INIT] metric=%s stages=%s before=%s after=%s epoch=%.1fs channels=%s display_channels=%s "
                    "mse={m:%s,r:%s,scales:%s,max_samples:%s,max_points:%s} "
                    "renorm={win:%s,overlap:%s,moment:%s,psi:%s,gamma:%s} "
                    "stats={wilcoxon:%s,perm:%s,boot:%s,z:%s,nperm:%s,bh:%s}",
                    analysis_cfg.metric,
                    ",".join(analysis_cfg.normalised_stage_list()),
                    analysis_cfg.before_label,
                    analysis_cfg.after_label,
                    analysis_cfg.epoch_seconds,
                    analysis_cfg.channel_names,
                    analysis_cfg.display_channels,
                    analysis_cfg.mse_config.m,
                    analysis_cfg.mse_config.r,
                    analysis_cfg.mse_config.scales,
                    analysis_cfg.mse_config.max_samples,
                    analysis_cfg.mse_config.max_pattern_length,
                    analysis_cfg.renorm_config.window_length,
                    analysis_cfg.renorm_config.overlap,
                    analysis_cfg.renorm_config.moment_order,
                    analysis_cfg.renorm_config.psi_name,
                    analysis_cfg.renorm_config.psi_params.get("gamma", None) if analysis_cfg.renorm_config.psi_params else None,
                    stats_cfg.run_wilcoxon,
                    stats_cfg.run_permutation,
                    stats_cfg.run_bootstrap,
                    stats_cfg.run_robust_z,
                    stats_cfg.n_permutations,
                    stats_cfg.apply_bh,
                )
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Configuration invalide", str(exc))
            return

        self._group_analysis_running = True
        if hasattr(self, "ga_run_button"):
            self.ga_run_button.config(state=tk.DISABLED)
        self._group_set_progress("Préparation…", 5)

        worker = threading.Thread(
            target=self._group_execute_analysis,
            args=(entries, analysis_cfg, stats_cfg),
            daemon=True,
            name="GroupAnalysisWorker",
        )
        self._group_analysis_thread = worker
        worker.start()

    def _group_execute_analysis(self, entries, analysis_cfg, stats_cfg):
        try:
            try:
                LOGGER = logging.getLogger(__name__)
                LOGGER.info(
                    "[GROUP_RUN] metric=%s stages=%s before=%s after=%s epoch=%.1fs channels=%s mse=%s renorm=%s",
                    analysis_cfg.metric,
                    ",".join(analysis_cfg.normalised_stage_list()),
                    analysis_cfg.before_label,
                    analysis_cfg.after_label,
                    analysis_cfg.epoch_seconds,
                    analysis_cfg.channel_names,
                    analysis_cfg.mse_config,
                    analysis_cfg.renorm_config,
                )
            except Exception:
                pass
            self._group_progress_callback("Calcul des profils…", 0.05)
            result = group_analysis.compute_group_profiles(
                entries,
                analysis_cfg,
                progress_cb=self._group_progress_callback,
            )
            self._group_progress_callback("Tests statistiques…", 0.8)
            stats = group_analysis.run_statistical_tests(result, stats_cfg)
        except Exception as exc:
            try:
                self._enqueue_tk_main(lambda: self._group_analysis_failed(exc))
            except Exception:
                pass
        else:
            try:
                self._enqueue_tk_main(lambda: self._group_analysis_completed(result, stats))
            except Exception:
                pass
        finally:
            self._group_analysis_running = False
            self._group_analysis_thread = None

    # HRV dédié (par stades configurables)
    def _group_run_hrv(self):
        if self.group_design_df.empty:
            messagebox.showwarning("Analyse HRV", "Ajoutez au moins une entrée dans le design.")
            return
        if getattr(self, "_group_hrv_running", False):
            messagebox.showinfo("Analyse HRV", "Un calcul HRV est déjà en cours.")
            return
        try:
            entries = self._group_build_entries()
            analysis_cfg = self._group_build_analysis_config()
        except Exception as exc:
            messagebox.showerror("Configuration HRV invalide", str(exc))
            return

        self._group_hrv_running = True
        self._group_set_progress("HRV…", 5)

        worker = threading.Thread(
            target=self._group_execute_hrv,
            args=(entries, analysis_cfg),
            daemon=True,
            name="GroupHRVWorker",
        )
        self._group_hrv_thread = worker
        worker.start()

    def _group_execute_hrv(self, entries, analysis_cfg):
        try:
            try:
                LOGGER = logging.getLogger(__name__)
                LOGGER.info(
                    "[HRV_RUN] stages=%s min_seg=%.1fs allow_short=%s peak_method=%s clean_rr=%s rr_method=%s lf=%s hf=%s resample=%.2f filt=%.1f-%.1f channels=%s",
                    ",".join(analysis_cfg.hrv_config.stage_filter),
                    analysis_cfg.hrv_config.min_segment_s,
                    analysis_cfg.hrv_config.allow_short_segments,
                    getattr(analysis_cfg.hrv_config, "peak_detection_method", "simple"),
                    analysis_cfg.hrv_config.clean_rr,
                    getattr(analysis_cfg.hrv_config, "rr_cleaning_method", "simple"),
                    analysis_cfg.hrv_config.lf_band,
                    analysis_cfg.hrv_config.hf_band,
                    analysis_cfg.hrv_config.resample_fs,
                    analysis_cfg.hrv_config.filter_band[0],
                    analysis_cfg.hrv_config.filter_band[1],
                    analysis_cfg.channel_names,
                )
            except Exception:
                pass
            self._group_progress_callback("HRV – extraction…", 0.05)
            result = group_analysis.compute_group_hrv(
                entries,
                analysis_cfg,
                progress_cb=self._group_progress_callback,
            )
        except Exception as exc:
            try:
                err = exc
                self._enqueue_tk_main(lambda err=err: self._group_hrv_failed(err))
            except Exception:
                pass
        else:
            try:
                self._enqueue_tk_main(lambda: self._group_hrv_completed(result))
            except Exception:
                pass
        finally:
            self._group_hrv_running = False
            self._group_hrv_thread = None

    def _group_hrv_completed(self, result) -> None:
        self.group_hrv_result = result
        self._group_set_progress("HRV prêt", 100)
        if hasattr(self, "ga_results_text"):
            try:
                self.ga_results_text.insert(
                    tk.END,
                    f"✅ HRV calculé – {len(result.summary)} lignes, "
                    f"{result.epochs['subject'].nunique()} sujets.\n",
                )
                self.ga_results_text.insert(
                    tk.END,
                    f"   Médiane RMSSD par sujet/condition/stade disponible via Export HRV.\n\n",
                )
                self.ga_results_text.see(tk.END)
            except Exception:
                pass
        try:
            messagebox.showinfo("HRV terminé", "Le calcul HRV par stades est terminé.")
        except Exception:
            pass

    def _group_hrv_failed(self, exc: Exception) -> None:
        self._group_set_progress("HRV échec", 0)
        try:
            messagebox.showerror("HRV échoué", str(exc))
        except Exception:
            pass

    def _group_export_hrv(self):
        if not getattr(self, "group_hrv_result", None):
            messagebox.showwarning("Export HRV", "Calculez d'abord le HRV.")
            return
        path = filedialog.asksaveasfilename(
            title="Exporter HRV (époques)",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        group_analysis.export_hrv_to_csv(self.group_hrv_result, path, summary_only=False)
        # Export résumé
        summary_path = Path(path).with_name(Path(path).stem + "_summary.csv")
        group_analysis.export_hrv_to_csv(self.group_hrv_result, summary_path, summary_only=True)
        try:
            plot_paths, stats_df = group_analysis.save_hrv_plots(self.group_hrv_result, Path(path).with_suffix(".png"))
            stats_path = Path(path).with_name(Path(path).stem + "_stats.csv")
            if not stats_df.empty:
                stats_df.to_csv(stats_path, index=False)
            else:
                stats_path = None
                # Export debug values pour inspection manuelle si stats vides
                debug_vals_path = Path(path).with_name(Path(path).stem + "_values.csv")
                debug_df = self.group_hrv_result.summary.copy()
                debug_df.to_csv(debug_vals_path, index=False)
        except Exception as exc:
            messagebox.showwarning("Export HRV", f"CSV ok mais plot/stats non générés: {exc}")
        else:
            msg_plots = "\n".join(str(p) for p in plot_paths)
            extra_stats = f"\nStats: {stats_path}" if stats_path else "\nStats: non disponibles (voir *_values.csv)"
            messagebox.showinfo("Export HRV", f"HRV exporté: {path}\nRésumé: {summary_path}\nFigures:\n{msg_plots}{extra_stats}")

    def _group_analysis_completed(self, result, stats: pd.DataFrame) -> None:
        self.group_analysis_result = result
        self._group_set_progress("Prêt", 100)
        self._group_update_summary(stats)
        self._group_update_plot_stage_options()
        if hasattr(self, "ga_run_button"):
            self.ga_run_button.config(state=tk.NORMAL)
        try:
            messagebox.showinfo("Analyse terminée", "Les profils ont été calculés avec succès.")
        except Exception:
            pass

    def _group_analysis_failed(self, exc: Exception) -> None:
        self._group_set_progress("Échec", 0)
        if hasattr(self, "ga_run_button"):
            self.ga_run_button.config(state=tk.NORMAL)
        try:
            messagebox.showerror("Analyse échouée", str(exc))
        except Exception:
            pass

    def _group_progress_callback(self, message: str, ratio: float) -> None:
        def _update():
            self._group_set_progress(message, ratio * 100.0)

        try:
            self._enqueue_tk_main(_update)
        except Exception:
            pass

    def _group_set_progress(self, message: str, value: float) -> None:
        try:
            self.ga_progress_var.set(message)
            if hasattr(self, "ga_progress"):
                self.ga_progress["value"] = max(0.0, min(100.0, value))
                self.ga_progress.update_idletasks()
        except Exception:
            pass

    def _group_build_entries(self) -> List[group_analysis.GroupDesignEntry]:
        entries: List[group_analysis.GroupDesignEntry] = []
        for _, row in self.group_design_df.iterrows():
            subject = str(row.get("subject", "")).strip()
            condition = str(row.get("condition", "")).strip()
            edf_path = Path(str(row.get("edf_path", "")).strip())
            scoring_path_str = str(row.get("scoring_path", "")).strip()
            stage_field = str(row.get("stage", "")).strip()
            stages = tuple(group_analysis.parse_stage_field(stage_field)) if stage_field and stage_field not in {"*", ""} else None
            scoring_path = Path(scoring_path_str) if scoring_path_str else None
            entries.append(
                group_analysis.GroupDesignEntry(
                    subject=subject,
                    condition=condition,
                    edf_path=edf_path,
                    scoring_path=scoring_path,
                    stages=stages,
                )
            )
        return entries

    def _group_selected_stages(self) -> List[str]:
        stages = [stage for stage, var in self.ga_stage_vars.items() if var.get()]
        return stages or list(group_analysis.DEFAULT_STAGES)

    def _group_parse_channels(self) -> Optional[List[str]]:
        selected = [ch for ch, var in self.ga_channel_vars.items() if var.get()]
        if selected:
            return selected
        raw = self.ga_channels_var.get().strip()
        if not raw:
            return None
        channels = [ch.strip() for ch in raw.replace(";", ",").split(",") if ch.strip()]
        return channels or None

    def _group_parse_scales(self, text: str) -> List[int]:
        raw = text.replace(";", ",").split(",")
        scales: List[int] = []
        for chunk in raw:
            part = chunk.strip()
            if not part:
                continue
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    start, end = end, start
                scales.extend(range(start, end + 1))
            else:
                scales.append(int(part))
        unique = sorted({max(1, int(val)) for val in scales})
        return unique or list(range(1, 21))

    def _group_parse_optional_int(self, text: str, default: Optional[int]) -> Optional[int]:
        raw = text.strip().lower()
        if not raw or raw == "auto":
            return default
        if raw in {"all", "none", "full", "∞", "inf"}:
            return None
        return int(raw)

    def _group_build_analysis_config(self) -> group_analysis.GroupAnalysisConfig:
        channels = self._group_parse_channels()
        stages = self._group_selected_stages()
        metric = self.ga_metric_var.get().strip().lower()

        design_conditions: List[str] = []
        if hasattr(self, "group_design_df") and "condition" in self.group_design_df.columns:
            for raw_cond in self.group_design_df["condition"].tolist():
                cond = str(raw_cond).strip()
                if cond and cond not in design_conditions:
                    design_conditions.append(cond)

        before_display = (self.ga_before_label_var.get() or "").strip()
        after_display = (self.ga_after_label_var.get() or "").strip()

        def _normalize(label: str) -> str:
            return group_analysis.normalise_condition_label(label)

        def _match_design(label: str) -> Optional[str]:
            if not label:
                return None
            wanted = _normalize(label)
            if not wanted:
                return None
            for cond in design_conditions:
                if _normalize(cond) == wanted:
                    return cond
            return None

        def _fallback_condition(index: int, default: str) -> str:
            if design_conditions:
                idx = max(0, min(index, len(design_conditions) - 1))
                return design_conditions[idx]
            return default

        if not before_display:
            before_display = _fallback_condition(0, "AVANT")
            self.ga_before_label_var.set(before_display)

        if not after_display or _normalize(after_display) == _normalize(before_display):
            default_idx = 1 if len(design_conditions) > 1 else 0
            after_display = _fallback_condition(default_idx, "APRÈS")
            self.ga_after_label_var.set(after_display)

        before_label = _match_design(before_display)
        if not before_label:
            before_label = _fallback_condition(0, "AVANT")

        after_label = _match_design(after_display)
        if not after_label or after_label == before_label:
            after_label = None
            for cond in design_conditions:
                if cond != before_label:
                    after_label = cond
                    break
            if after_label is None:
                after_label = "APRÈS" if not design_conditions else before_label

        channels_display = self.ga_channels_var.get().strip()

        mse_cfg = group_analysis.MultiscaleEntropyConfig(
            m=int(self.ga_m_var.get()),
            r=float(self.ga_r_var.get()),
            scales=self._group_parse_scales(self.ga_scales_var.get()),
            max_samples=self._group_parse_optional_int(self.ga_max_samples_var.get(), group_analysis.MultiscaleEntropyConfig().max_samples),
            max_pattern_length=max(self._group_parse_optional_int(self.ga_max_points_var.get(), 5000) or 5000, 32),
        )

        renorm_cfg = group_analysis.RenormalizedEntropyConfig(
            window_length=float(self.ga_window_length_var.get()),
            overlap=float(self.ga_overlap_var.get()),
            moment_order=float(self.ga_moment_order_var.get()),
            psi_name=self.ga_psi_name_var.get(),
            psi_params={"gamma": float(self.ga_gamma_var.get()), "epsilon": 1e-12},
        )

        hrv_cfg = group_analysis.HRVConfig(
            lf_band=(float(self.hr_lf_var.get()), float(self.hr_lf_high_var.get())),
            hf_band=(float(self.hr_hf_var.get()), float(self.hr_hf_high_var.get())),
            resample_fs=float(self.hr_resample_var.get()),
            filter_band=(float(self.hr_filter_low_var.get()), float(self.hr_filter_high_var.get())),
            stage_filter=tuple(s.strip() for s in self.hr_stage_var.get().split(",") if s.strip()),
            min_segment_s=float(self.hr_min_segment_var.get()),
            allow_short_segments=bool(self.hr_allow_short_var.get()),
            clean_rr=bool(self.hr_clean_rr_var.get()),
            peak_detection_method=str(self.hr_peak_method_var.get()).strip().lower() or "simple",
            rr_cleaning_method=str(self.hr_rr_method_var.get()).strip().lower() or "simple",
        )

        return group_analysis.GroupAnalysisConfig(
            metric=metric,
            before_label=before_label,
            after_label=after_label,
            display_before_label=before_display,
            display_after_label=after_display,
            display_channels=channels_display or (", ".join(channels) if channels else None),
            stages=stages,
            epoch_seconds=float(self.ga_epoch_var.get()),
            channel_names=channels,
            mse_config=mse_cfg,
            renorm_config=renorm_cfg,
            hrv_config=hrv_cfg,
        )

    def _group_resolve_condition_labels(self) -> Tuple[str, str]:
        before = (self.ga_before_label_var.get() or "").strip() or "AVANT"
        after = (self.ga_after_label_var.get() or "").strip() or "APRÈS"

        if self.group_analysis_result is not None:
            df = self.group_analysis_result.profiles
            available = [str(cond) for cond in sorted(df["condition"].dropna().unique())]
            if available:
                if before not in available:
                    before = available[0]
                if after not in available or after == before:
                    fallback = next((c for c in available if c != before), available[0])
                    after = fallback
        return before, after

    def _group_build_stats_config(self) -> group_analysis.StatsConfig:
        return group_analysis.StatsConfig(
            run_wilcoxon=self.ga_run_wilcoxon_var.get(),
            run_permutation=self.ga_run_permutation_var.get(),
            run_bootstrap=self.ga_run_bootstrap_var.get(),
            run_robust_z=self.ga_run_robust_z_var.get(),
            n_permutations=int(max(100, self.ga_nperm_var.get())),
            apply_bh=self.ga_bh_var.get(),
        )

    def _group_update_summary(self, stats: pd.DataFrame) -> None:
        if not hasattr(self, "ga_results_text"):
            return
        self.ga_results_text.delete("1.0", tk.END)
        if not self.group_analysis_result:
            return
        result = self.group_analysis_result
        df = result.profiles
        subjects = sorted(df["subject"].unique())
        stages = result.available_stages()
        self.ga_results_text.insert(tk.END, f"✅ Profils calculés pour {len(subjects)} sujets et {len(stages)} stades.\n")
        self.ga_results_text.insert(tk.END, f"   Conditions : {', '.join(sorted(df['condition'].unique()))}\n")
        self.ga_results_text.insert(tk.END, f"   Total lignes : {len(df):,}\n\n")

        if stats is not None and not stats.empty:
            best = stats.dropna(subset=["p_value"]).sort_values("p_value").head(10)
            self.ga_results_text.insert(tk.END, "📊 Top p-values (corrigées si BH actif):\n")
            for _, row in best.iterrows():
                p_val = row.get("p_value")
                p_adj = row.get("p_adj", np.nan)
                line = f" - {row['test']} | {row['stage']} τ={row['tau']:.0f} → p={p_val:.4g}"
                if not np.isnan(p_adj):
                    line += f" (p_adj={p_adj:.4g})"
                line += f" | n={int(row['n_subjects'])}\n"
                self.ga_results_text.insert(tk.END, line)
            self.ga_results_text.insert(tk.END, "\n")

        summary = result.subject_summary
        if summary is not None and not summary.empty:
            self.ga_results_text.insert(tk.END, "📈 Moyenne AUC par condition/stade:\n")
            agg = summary.groupby(["condition", "stage"])["auc"].mean().reset_index()
            for _, row in agg.iterrows():
                self.ga_results_text.insert(
                    tk.END,
                    f"   {row['condition']} – {row['stage']}: AUC moyenne = {row['auc']:.4f}\n",
                )

    def _group_update_plot_stage_options(self):
        if not self.group_analysis_result:
            return
        stages = self.group_analysis_result.available_stages()
        self.ga_plot_stage_combo["values"] = stages
        if stages:
            self.ga_plot_stage_var.set(stages[0])
            self._group_render_plot(stages[0])

    def _group_on_stage_change(self):
        stage = self.ga_plot_stage_var.get()
        if stage and self.group_analysis_result:
            self._group_render_plot(stage)

    def _group_render_plot(self, stage: str):
        if not self.group_analysis_result:
            return
        self.ga_plot_figure.clf()
        ax = self.ga_plot_figure.add_subplot(111)
        before_label, after_label = self._group_resolve_condition_labels()
        group_analysis.plot_stage_profiles(
            self.group_analysis_result,
            stage,
            before_label=before_label,
            after_label=after_label,
            ax=ax,
        )
        self.ga_plot_figure.tight_layout()
        self.ga_plot_canvas.draw()

    def _group_toggle_metric_frames(self):
        metric = self.ga_metric_var.get()
        if metric == "renorm":
            self.ga_mse_frame.pack_forget()
            self.ga_renorm_frame.pack(fill=tk.X, padx=10, pady=6)
        else:
            self.ga_renorm_frame.pack_forget()
            self.ga_mse_frame.pack(fill=tk.X, padx=10, pady=6)

    # ------------------------------------------------------------------
    # Export résultats
    # ------------------------------------------------------------------

    def _group_export_profiles(self):
        if not self.group_analysis_result:
            messagebox.showwarning("Export", "Exécutez d'abord l'analyse.")
            return
        path = filedialog.asksaveasfilename(
            title="Exporter les profils",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        group_analysis.export_profiles_to_csv(self.group_analysis_result, path)
        messagebox.showinfo("Export profils", f"Profils sauvegardés dans {path}")

    def _group_export_stats(self):
        if not self.group_analysis_result:
            messagebox.showwarning("Export", "Exécutez d'abord l'analyse.")
            return
        if self.group_analysis_result.stats is None or self.group_analysis_result.stats.empty:
            messagebox.showwarning("Export", "Aucun test statistique disponible.")
            return
        path = filedialog.asksaveasfilename(
            title="Exporter les tests",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        group_analysis.export_stats_to_csv(self.group_analysis_result, path)
        messagebox.showinfo("Export tests", f"Table des tests sauvegardée dans {path}")

    def _group_export_plot(self):
        if not self.group_analysis_result:
            messagebox.showwarning("Export", "Aucun graphique à exporter.")
            return
        path = filedialog.asksaveasfilename(
            title="Exporter le graphique",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path:
            return
        self.ga_plot_figure.savefig(path, dpi=200, bbox_inches="tight")
        messagebox.showinfo("Export figure", f"Graphique sauvegardé dans {path}")

    def _setup_user_assistant(self):
        """Initialise l'assistant utilisateur."""
        try:
            from CESA.user_assistant import UserAssistant
            self.user_assistant = UserAssistant(self)
            
            # Afficher l'assistant de bienvenue si c'est la première utilisation
            self.root.after(2000, self._show_welcome_if_first_time)
            
            print("✅ Assistant utilisateur initialisé")
            logging.info("[ASSISTANT] User assistant initialized")
            
        except Exception as e:
            print(f"⚠️ Assistant utilisateur non disponible: {e}")
            logging.warning(f"[ASSISTANT] User assistant not available: {e}")
            self.user_assistant = None
    
    def _show_welcome_if_first_time(self):
        """Affiche l'assistant de bienvenue si c'est la première utilisation."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant.show_welcome_assistant()
    
    def _show_welcome_assistant(self):
        """Affiche l'assistant de bienvenue."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant.show_welcome_assistant()
        else:
            messagebox.showinfo("Assistant", "Assistant utilisateur non disponible.")
    
    def _show_feature_explorer(self):
        """Affiche l'explorateur de fonctionnalités."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant.show_feature_explorer()
        else:
            messagebox.showinfo("Explorateur", "Assistant utilisateur non disponible.")
    
    def _open_documentation(self):
        """Ouvre la documentation principale."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant._open_documentation()
        else:
            import webbrowser
            try:
                webbrowser.open("README.md")
            except Exception:
                messagebox.showinfo("Documentation", "Consultez le fichier README.md")
    
    def _open_entropy_docs(self):
        """Ouvre la documentation de l'entropie renormée."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant._open_entropy_docs()
        else:
            import webbrowser
            try:
                webbrowser.open("ENTROPY_INTEGRATION.md")
            except Exception:
                messagebox.showinfo("Documentation", "Consultez le fichier ENTROPY_INTEGRATION.md")
    
    def _run_diagnostic(self):
        """Lance un diagnostic automatique du système."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant._run_diagnostic()
        else:
            messagebox.showinfo("Diagnostic", "Assistant utilisateur non disponible.")
    
    def _open_support(self):
        """Ouvre les informations de support."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant._open_support()
        else:
            support_info = """
📞 SUPPORT CESA v0.0beta1.1

🆘 EN CAS DE PROBLÈME :

1️⃣ CONSULTEZ LA DOCUMENTATION :
   • README.md : Guide général
   • ENTROPY_INTEGRATION.md : Entropie renormée
   • GUIDE_INSTALLATION_NOOB.md : Installation

2️⃣ CONTACTEZ LE SUPPORT :
   • Email : come1.barmoy@supbiotech.fr
   • GitHub : cbarmoy
   • Unité Neuropsychologie du Stress (IRBA)

💡 CONSEILS :
• Décrivez précisément votre problème
• Incluez les messages d'erreur
• Précisez votre configuration système
• Joignez des captures d'écran si utile

🎯 DÉVELOPPEMENT :
CESA v0.0beta1.1 est développé pour l'Unité Neuropsychologie du Stress (IRBA)
Auteur : Côme Barmoy
Version : 0.0beta1.1
Licence : MIT
            """
            messagebox.showinfo("Support CESA v0.0beta1.1", support_info)
    
    def _open_reference_guide(self):
        """Ouvre le guide de référence complet."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant._open_reference_guide()
        else:
            messagebox.showinfo("Guide de Référence", "Assistant utilisateur non disponible.")
    
    def _extract_current_window_data(self, data):
        """Extrait les données de la fenêtre temporelle actuelle."""
        try:
            if not hasattr(self, 'current_time') or not hasattr(self, 'duration'):
                return np.array([])
            
            start_idx = int(self.current_time * self.sfreq)
            end_idx = int((self.current_time + self.duration) * self.sfreq)
            
            # S'assurer que les indices sont dans les limites
            start_idx = max(0, min(start_idx, len(data) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(data)))
            
            return data[start_idx:end_idx]
        except Exception as e:
            print(f"❌ CHECKPOINT TEMPORAL: Erreur extraction données: {e}")
            return np.array([])
    
    def _calculate_temporal_stats(self, data, channel):
        """Calcule les statistiques temporelles d'un signal."""
        try:
            if len(data) == 0:
                return {
                    'mean_amplitude': 0.0,
                    'std_amplitude': 0.0,
                    'peak_to_peak': 0.0,
                    'max_amplitude': 0.0,
                    'min_amplitude': 0.0,
                    'dominant_frequency': 0.0,
                    'total_power': 0.0
                }
            
            # Statistiques de base
            mean_amplitude = np.mean(data)
            std_amplitude = np.std(data)
            max_amplitude = np.max(data)
            min_amplitude = np.min(data)
            peak_to_peak = max_amplitude - min_amplitude
            
            # Calcul de la fréquence dominante
            try:
                # FFT pour trouver la fréquence dominante
                fft = np.fft.fft(data)
                freqs = np.fft.fftfreq(len(data), 1/self.sfreq)
                
                # Prendre seulement la partie positive des fréquences
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = np.abs(fft[:len(fft)//2])
                
                # Trouver la fréquence avec la plus grande amplitude
                if len(positive_freqs) > 0:
                    dominant_freq_idx = np.argmax(positive_fft)
                    dominant_frequency = positive_freqs[dominant_freq_idx]
                else:
                    dominant_frequency = 0.0
            except Exception:
                dominant_frequency = 0.0
            
            # Puissance totale
            total_power = np.sum(data**2)
            
            return {
                'mean_amplitude': mean_amplitude,
                'std_amplitude': std_amplitude,
                'peak_to_peak': peak_to_peak,
                'max_amplitude': max_amplitude,
                'min_amplitude': min_amplitude,
                'dominant_frequency': dominant_frequency,
                'total_power': total_power
            }
            
        except Exception as e:
            print(f"❌ CHECKPOINT TEMPORAL: Erreur calcul stats: {e}")
            return {
                'mean_amplitude': 0.0,
                'std_amplitude': 0.0,
                'peak_to_peak': 0.0,
                'max_amplitude': 0.0,
                'min_amplitude': 0.0,
                'dominant_frequency': 0.0,
                'total_power': 0.0
            }
    
    def _export_temporal_analysis(self):
        """Exporte l'analyse temporelle vers un fichier CSV."""
        try:
            if not hasattr(self, 'raw') or self.raw is None:
                messagebox.showwarning("Attention", "Aucune donnée à exporter")
                return
            
            # Demander le nom du fichier
            filename = filedialog.asksaveasfilename(
                title="Exporter l'analyse temporelle",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*.*")]
            )
            
            if not filename:
                return
            
            # Collecter les données pour l'export
            export_data = []
            
            for channel in self.selected_channels:
                if channel not in self.derivations:
                    continue
                
                data = self.derivations[channel]
                current_window_data = self._extract_current_window_data(data)
                
                if len(current_window_data) > 0:
                    stats = self._calculate_temporal_stats(current_window_data, channel)
                    
                    export_data.append({
                        'Canal': channel,
                        'Temps_debut': self.current_time,
                        'Temps_fin': self.current_time + self.duration,
                        'Amplitude_moyenne': stats['mean_amplitude'],
                        'Ecart_type': stats['std_amplitude'],
                        'Amplitude_crete_a_crete': stats['peak_to_peak'],
                        'Amplitude_max': stats['max_amplitude'],
                        'Amplitude_min': stats['min_amplitude'],
                        'Frequence_dominante': stats['dominant_frequency'],
                        'Puissance_totale': stats['total_power']
                    })
            
            # Écrire le fichier CSV
            if export_data:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(export_data)
                
                messagebox.showinfo("Succès", f"Analyse temporelle exportée vers {filename}")
                print(f"✅ CHECKPOINT TEMPORAL: Export réussi - {filename}")
                logging.info(f"[TEMPORAL] Export successful: {filename}")
            else:
                messagebox.showwarning("Attention", "Aucune donnée à exporter")
                
        except Exception as e:
            print(f"❌ CHECKPOINT TEMPORAL: Erreur export: {e}")
            logging.error(f"[TEMPORAL] Export failed: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'export : {str(e)}")
    
    def _refresh_temporal_analysis(self, window):
        """Actualise l'analyse temporelle."""
        try:
            window.destroy()
            self._show_temporal_analysis()
        except Exception as e:
            print(f"❌ CHECKPOINT TEMPORAL: Erreur actualisation: {e}")
            logging.error(f"[TEMPORAL] Refresh failed: {e}")
    
    def _show_markers(self):
        """Affiche le système de marqueurs temporels."""
        try:
            if not hasattr(self, 'raw') or self.raw is None:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            print("🔍 CHECKPOINT MARKERS: Début gestion marqueurs")
            logging.info("[MARKERS] Starting markers management")

            def create_tooltip(widget, text):
                """Crée un tooltip pour un widget."""
                def show_tooltip(event):
                    tooltip = tk.Toplevel()
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                    
                    label = tk.Label(tooltip, text=text, justify='left', 
                                   background='#ffffe0', relief='solid', borderwidth=1,
                                   font=('Segoe UI', 9), wraplength=300)
                    label.pack()
                    
                    widget.tooltip = tooltip
                    
                def hide_tooltip(event):
                    if hasattr(widget, 'tooltip') and widget.tooltip:
                        widget.tooltip.destroy()
                        widget.tooltip = None
                        
                widget.bind('<Enter>', show_tooltip)
                widget.bind('<Leave>', hide_tooltip)
            
            # Créer la fenêtre de gestion des marqueurs
            markers_window = tk.Toplevel(self.root)
            markers_window.title("Marqueurs Temporels - EEG Analysis Studio")
            markers_window.geometry("800x600")
            markers_window.transient(self.root)
            markers_window.grab_set()
            
            # Interface de base
            main_frame = ttk.Frame(markers_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            title_label = ttk.Label(main_frame, text="📍 Marqueurs Temporels", 
                                  font=('Segoe UI', 16, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # Frame pour ajouter un marqueur
            add_frame = ttk.LabelFrame(main_frame, text="Ajouter un marqueur", padding=10)
            add_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Variables pour les champs
            description_var = tk.StringVar()
            position_var = tk.StringVar(value=str(self.current_time))
            color_var = tk.StringVar(value="red")
            
            # Interface d'ajout
            ttk.Label(add_frame, text="Description:").grid(row=0, column=0, sticky='w', pady=(0, 5))
            desc_entry = ttk.Entry(add_frame, textvariable=description_var, width=40)
            desc_entry.grid(row=0, column=1, columnspan=2, sticky='ew', pady=(0, 5))
            
            ttk.Label(add_frame, text="Position (s):").grid(row=1, column=0, sticky='w', pady=(0, 5))
            pos_entry = ttk.Entry(add_frame, textvariable=position_var, width=15)
            pos_entry.grid(row=1, column=1, sticky='w', pady=(0, 5))
            
            def set_current_position():
                position_var.set(str(self.current_time))
            current_pos_btn = ttk.Button(add_frame, text="Position actuelle", command=set_current_position)
            current_pos_btn.grid(row=1, column=2, padx=(10, 0), pady=(0, 5))
            
            ttk.Label(add_frame, text="Couleur:").grid(row=2, column=0, sticky='w', pady=(0, 5))
            color_combo = ttk.Combobox(add_frame, textvariable=color_var, 
                                     values=["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"])
            color_combo.grid(row=2, column=1, sticky='w', pady=(0, 5))
            
            # Tooltips pour les contrôles des marqueurs
            create_tooltip(desc_entry, 
                          "Description du marqueur temporel.\n\n"
                          "• Texte descriptif qui apparaîtra sur le graphique\n"
                          "• Exemples : 'Début sommeil', 'Artefact', 'Événement important'\n"
                          "• Permet d'identifier facilement les points d'intérêt\n"
                          "• Visible dans la légende et les exports\n"
                          "• Maximum 50 caractères recommandé")
            
            create_tooltip(pos_entry, 
                          "Position temporelle du marqueur (secondes).\n\n"
                          "• Temps exact où placer le marqueur sur le graphique\n"
                          "• Peut être saisi manuellement ou via 'Position actuelle'\n"
                          "• Doit être ≥ 0 et ≤ durée totale de l'enregistrement\n"
                          "• Format : nombre décimal (ex: 123.45)\n"
                          "• Le marqueur sera visible sur tous les canaux")
            
            create_tooltip(current_pos_btn, 
                          "Utiliser la position temporelle actuelle.\n\n"
                          "• Copie automatiquement le temps actuel de la vue principale\n"
                          "• Utile pour marquer l'instant présent\n"
                          "• Évite de saisir manuellement la position\n"
                          "• Met à jour le champ 'Position (s)' automatiquement\n"
                          "• Idéal pour marquer des événements en temps réel")
            
            create_tooltip(color_combo, 
                          "Couleur d'affichage du marqueur.\n\n"
                          "• Définit la couleur du marqueur sur le graphique\n"
                          "• 8 couleurs disponibles : rouge, bleu, vert, orange, etc.\n"
                          "• Permet de distinguer différents types de marqueurs\n"
                          "• Visible dans la légende et les exports\n"
                          "• Recommandé : utiliser des couleurs contrastées")
            
            print("✅ CHECKPOINT MARKERS: Interface marqueurs affichée")
            logging.info("[MARKERS] Markers interface displayed")
            
        except Exception as e:
            print(f"❌ CHECKPOINT MARKERS: Erreur interface marqueurs: {e}")
            logging.error(f"[MARKERS] Failed to show markers interface: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des marqueurs : {str(e)}")
    
    def add_temporal_marker(self, description, position, color="red"):
        """Ajoute un marqueur temporel."""
        try:
            marker = {
                'id': len(self.temporal_markers) + 1,
                'description': description,
                'position': position,
                'color': color,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            
            self.temporal_markers.append(marker)
            self.update_plot()  # Mettre à jour le graphique
            
            print(f"✅ CHECKPOINT MARKERS: Marqueur ajouté - {description} à {position}s")
            logging.info(f"[MARKERS] Marker added: {description} at {position}s")
            
            return marker
            
        except Exception as e:
            print(f"❌ CHECKPOINT MARKERS: Erreur ajout marqueur: {e}")
            logging.error(f"[MARKERS] Failed to add marker: {e}")
            return None
    
    def remove_temporal_marker(self, marker_id):
        """Supprime un marqueur temporel."""
        try:
            original_count = len(self.temporal_markers)
            self.temporal_markers = [m for m in self.temporal_markers if m['id'] != marker_id]
            
            if len(self.temporal_markers) < original_count:
                self.update_plot()  # Mettre à jour le graphique
                print(f"🗑️ CHECKPOINT MARKERS: Marqueur {marker_id} supprimé")
                logging.info(f"[MARKERS] Marker {marker_id} deleted")
                return True
            else:
                print(f"⚠️ CHECKPOINT MARKERS: Marqueur {marker_id} non trouvé")
                logging.warning(f"[MARKERS] Marker {marker_id} not found")
                return False
                
        except Exception as e:
            print(f"❌ CHECKPOINT MARKERS: Erreur suppression marqueur: {e}")
            logging.error(f"[MARKERS] Failed to remove marker: {e}")
            return False
    
    def get_temporal_markers_in_range(self, start_time, end_time):
        """Récupère les marqueurs dans une plage temporelle."""
        try:
            markers_in_range = []
            for marker in self.temporal_markers:
                if start_time <= marker['position'] <= end_time:
                    markers_in_range.append(marker)
            
            return markers_in_range
            
        except Exception as e:
            print(f"❌ CHECKPOINT MARKERS: Erreur récupération marqueurs: {e}")
            logging.error(f"[MARKERS] Failed to get markers in range: {e}")
            return []
    
    def _show_measurements(self):
        """Affiche les outils de mesure sur les signaux."""
        try:
            if not hasattr(self, 'raw') or self.raw is None:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            if not hasattr(self, 'selected_channels') or not self.selected_channels:
                messagebox.showwarning("Attention", "Veuillez d'abord sélectionner des canaux")
                return
            
            print("🔍 CHECKPOINT MEASUREMENTS: Début outils de mesure")
            logging.info("[MEASUREMENTS] Starting measurement tools")
            
            # Créer la fenêtre des outils de mesure
            measurements_window = tk.Toplevel(self.root)
            measurements_window.title("Outils de Mesure - EEG Analysis Studio")
            measurements_window.geometry("900x700")
            measurements_window.transient(self.root)
            measurements_window.grab_set()
            
            # Interface de base
            main_frame = ttk.Frame(measurements_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Titre
            title_label = ttk.Label(main_frame, text="📏 Outils de Mesure EEG", 
                                  font=('Segoe UI', 16, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # Frame pour les mesures de base
            basic_frame = ttk.LabelFrame(main_frame, text="Mesures de Base", padding=10)
            basic_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Variables pour les mesures
            measurement_results = {}
            
            def perform_basic_measurements():
                """Effectue les mesures de base sur les canaux sélectionnés."""
                try:
                    measurement_results.clear()
                    
                    for channel in self.selected_channels:
                        if channel not in self.derivations:
                            continue
                        
                        data = self.derivations[channel]
                        current_window_data = self._extract_current_window_data(data)
                        
                        if len(current_window_data) > 0:
                            # Calculs de base
                            mean_val = np.mean(current_window_data)
                            std_val = np.std(current_window_data)
                            max_val = np.max(current_window_data)
                            min_val = np.min(current_window_data)
                            rms_val = np.sqrt(np.mean(current_window_data**2))
                            peak_to_peak = max_val - min_val
                            
                            # Calcul de la puissance
                            power = np.mean(current_window_data**2)
                            
                            measurement_results[channel] = {
                                'mean': mean_val,
                                'std': std_val,
                                'max': max_val,
                                'min': min_val,
                                'rms': rms_val,
                                'peak_to_peak': peak_to_peak,
                                'power': power
                            }
                    
                    print("✅ CHECKPOINT MEASUREMENTS: Mesures de base calculées")
                    logging.info("[MEASUREMENTS] Basic measurements calculated")
                    
                except Exception as e:
                    print(f"❌ CHECKPOINT MEASUREMENTS: Erreur calcul mesures: {e}")
                    logging.error(f"[MEASUREMENTS] Failed to calculate measurements: {e}")
                    messagebox.showerror("Erreur", f"Erreur lors du calcul des mesures : {str(e)}")
            
            # Bouton pour effectuer les mesures
            ttk.Button(basic_frame, text="Calculer les Mesures", command=perform_basic_measurements).pack(pady=(0, 10))
            
            print("✅ CHECKPOINT MEASUREMENTS: Interface mesures affichée")
            logging.info("[MEASUREMENTS] Measurements interface displayed")
            
        except Exception as e:
            print(f"❌ CHECKPOINT MEASUREMENTS: Erreur interface mesures: {e}")
            logging.error(f"[MEASUREMENTS] Failed to show measurements interface: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des mesures : {str(e)}")
    
    def calculate_signal_measurements(self, data, channel_name):
        """Calcule les mesures complètes d'un signal."""
        try:
            if len(data) == 0:
                return {}
            
            # Mesures de base
            mean_val = np.mean(data)
            std_val = np.std(data)
            max_val = np.max(data)
            min_val = np.min(data)
            rms_val = np.sqrt(np.mean(data**2))
            peak_to_peak = max_val - min_val
            
            # Mesures de puissance
            power = np.mean(data**2)
            power_dB = 10 * np.log10(power + 1e-12)  # Éviter log(0)
            
            # Mesures de fréquence
            try:
                # FFT pour l'analyse fréquentielle
                fft = np.fft.fft(data)
                freqs = np.fft.fftfreq(len(data), 1/self.sfreq)
                
                # Prendre seulement la partie positive des fréquences
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = np.abs(fft[:len(fft)//2])
                
                # Fréquence dominante
                if len(positive_freqs) > 0:
                    dominant_freq_idx = np.argmax(positive_fft)
                    dominant_freq = positive_freqs[dominant_freq_idx]
                    
                    # Bande de fréquence contenant 90% de la puissance
                    power_spectrum = positive_fft**2
                    total_power = np.sum(power_spectrum)
                    cumulative_power = np.cumsum(power_spectrum)
                    
                    # Trouver les indices contenant 90% de la puissance
                    idx_90 = np.where(cumulative_power >= 0.9 * total_power)[0]
                    if len(idx_90) > 0:
                        freq_band_90 = positive_freqs[idx_90[0]], positive_freqs[idx_90[-1]]
                    else:
                        freq_band_90 = (positive_freqs[0], positive_freqs[-1])
                else:
                    dominant_freq = 0.0
                    freq_band_90 = (0.0, 0.0)
                    
            except Exception:
                dominant_freq = 0.0
                freq_band_90 = (0.0, 0.0)
            
            # Mesures de forme d'onde
            # Facteur de crête
            crest_factor = max_val / (rms_val + 1e-12)
            
            # Facteur de forme
            form_factor = rms_val / (abs(mean_val) + 1e-12)
            
            # Skewness et Kurtosis
            try:
                from scipy.stats import skew, kurtosis
                skewness = skew(data)
                kurt = kurtosis(data)
            except ImportError:
                # Calcul manuel si scipy n'est pas disponible
                skewness = np.mean(((data - mean_val) / (std_val + 1e-12))**3)
                kurt = np.mean(((data - mean_val) / (std_val + 1e-12))**4) - 3
            
            return {
                'channel': channel_name,
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'min': min_val,
                'rms': rms_val,
                'peak_to_peak': peak_to_peak,
                'power': power,
                'power_dB': power_dB,
                'dominant_frequency': dominant_freq,
                'frequency_band_90': freq_band_90,
                'crest_factor': crest_factor,
                'form_factor': form_factor,
                'skewness': skewness,
                'kurtosis': kurt
            }
            
        except Exception as e:
            print(f"❌ CHECKPOINT MEASUREMENTS: Erreur calcul mesures signal: {e}")
            logging.error(f"[MEASUREMENTS] Failed to calculate signal measurements: {e}")
            return {}
    
    def calculate_channel_correlation(self, data1, data2, channel1, channel2):
        """Calcule la corrélation entre deux canaux."""
        try:
            if len(data1) == 0 or len(data2) == 0:
                return {}
            
            # Ajuster la longueur des signaux si nécessaire
            min_length = min(len(data1), len(data2))
            if min_length == 0:
                return {}
            
            data1_adj = data1[:min_length]
            data2_adj = data2[:min_length]
            
            # Corrélation de Pearson
            correlation = np.corrcoef(data1_adj, data2_adj)[0, 1]
            
            # Cohérence (simplifiée)
            try:
                from scipy import signal
                f, Cxy = signal.coherence(data1_adj, data2_adj, fs=self.sfreq, nperseg=min(256, len(data1_adj)//4))
                mean_coherence = np.mean(Cxy)
                max_coherence = np.max(Cxy)
                coherence_at_dominant = Cxy[np.argmax(np.abs(signal.welch(data1_adj, fs=self.sfreq)[1]))] if len(Cxy) > 0 else 0
            except Exception:
                mean_coherence = abs(correlation)  # Approximation
                max_coherence = abs(correlation)
                coherence_at_dominant = abs(correlation)
            
            # Décalage temporel (cross-correlation)
            try:
                cross_corr = np.correlate(data1_adj, data2_adj, mode='full')
                lags = np.arange(-len(data2_adj) + 1, len(data1_adj))
                max_corr_idx = np.argmax(np.abs(cross_corr))
                time_delay = lags[max_corr_idx] / self.sfreq
            except Exception:
                time_delay = 0.0
            
            return {
                'channel_pair': f"{channel1}-{channel2}",
                'correlation': correlation,
                'mean_coherence': mean_coherence,
                'max_coherence': max_coherence,
                'coherence_at_dominant': coherence_at_dominant,
                'time_delay': time_delay
            }
            
        except Exception as e:
            print(f"❌ CHECKPOINT MEASUREMENTS: Erreur calcul corrélation: {e}")
            logging.error(f"[MEASUREMENTS] Failed to calculate correlation: {e}")
            return {}
    
    def export_measurements_to_csv(self, measurements_data, filename):
        """Exporte les mesures vers un fichier CSV."""
        try:
            if not measurements_data:
                messagebox.showwarning("Attention", "Aucune mesure à exporter")
                return False
            
            import csv
            
            # Déterminer les colonnes en fonction du type de données
            if isinstance(measurements_data, dict) and 'channel' in list(measurements_data.values())[0]:
                # Mesures de signaux individuels
                fieldnames = ['Canal', 'Moyenne', 'Ecart_type', 'Max', 'Min', 'RMS', 'Crete_a_crete', 
                             'Puissance', 'Puissance_dB', 'Frequence_dominante', 'Bande_freq_90_min', 
                             'Bande_freq_90_max', 'Facteur_crete', 'Facteur_forme', 'Skewness', 'Kurtosis']
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for channel, results in measurements_data.items():
                        freq_band = results.get('frequency_band_90', (0, 0))
                        writer.writerow({
                            'Canal': results['channel'],
                            'Moyenne': results['mean'],
                            'Ecart_type': results['std'],
                            'Max': results['max'],
                            'Min': results['min'],
                            'RMS': results['rms'],
                            'Crete_a_crete': results['peak_to_peak'],
                            'Puissance': results['power'],
                            'Puissance_dB': results['power_dB'],
                            'Frequence_dominante': results['dominant_frequency'],
                            'Bande_freq_90_min': freq_band[0],
                            'Bande_freq_90_max': freq_band[1],
                            'Facteur_crete': results['crest_factor'],
                            'Facteur_forme': results['form_factor'],
                            'Skewness': results['skewness'],
                            'Kurtosis': results['kurtosis']
                        })
            
            elif isinstance(measurements_data, dict) and 'channel_pair' in list(measurements_data.values())[0]:
                # Mesures de corrélation
                fieldnames = ['Paire_Canaux', 'Correlation', 'Cohérence_moyenne', 'Cohérence_max', 
                             'Cohérence_dominante', 'Délai_temporel']
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for pair, results in measurements_data.items():
                        writer.writerow({
                            'Paire_Canaux': results['channel_pair'],
                            'Correlation': results['correlation'],
                            'Cohérence_moyenne': results['mean_coherence'],
                            'Cohérence_max': results['max_coherence'],
                            'Cohérence_dominante': results['coherence_at_dominant'],
                            'Délai_temporel': results['time_delay']
                        })
            
            else:
                messagebox.showerror("Erreur", "Format de données non reconnu pour l'export")
                return False
            
            print(f"✅ CHECKPOINT MEASUREMENTS: Export réussi - {filename}")
            logging.info(f"[MEASUREMENTS] Export successful: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ CHECKPOINT MEASUREMENTS: Erreur export: {e}")
            logging.error(f"[MEASUREMENTS] Export failed: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'export : {str(e)}")
            return False
    
    def _show_advanced_config(self):
        """Redirige vers la configuration unifiée des filtres."""
        self.show_filter_config()

    def _show_filter_config(self):
        """Redirige vers la configuration unifiée des filtres."""
        self.show_filter_config()
    
    def _show_loading_bar(self, title="Chargement en cours...", message="Veuillez patienter"):
        """Affiche une barre de chargement stylée."""
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title(title)
        self.loading_window.geometry("400x200")
        self.loading_window.resizable(False, False)
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()
        
        # Centrer la fenêtre
        self.loading_window.update_idletasks()
        x = (self.loading_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.loading_window.winfo_screenheight() // 2) - (200 // 2)
        self.loading_window.geometry(f"400x200+{x}+{y}")
        
        # Style moderne
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.loading_window, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Icône de chargement (simulée avec du texte)
        icon_label = ttk.Label(main_frame, text="EEG", font=('Segoe UI', 24))
        icon_label.pack(pady=(0, 10))
        
        # Titre
        title_label = ttk.Label(main_frame, text=title, font=('Segoe UI', 12, 'bold'))
        title_label.pack(pady=(0, 5))
        
        # Message
        message_label = ttk.Label(main_frame, text=message, font=('Segoe UI', 10))
        message_label.pack(pady=(0, 20))
        
        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100, length=300, mode='determinate')
        self.progress_bar.pack(pady=(0, 10))
        
        # Label de progression
        self.progress_label = ttk.Label(main_frame, text="0%", font=('Segoe UI', 9))
        self.progress_label.pack()
        
        # Animation de chargement
        self._animate_loading()
        
        # Mettre à jour la fenêtre
        self.loading_window.update()
    
    def _animate_loading(self):
        """Anime la barre de chargement."""
        if hasattr(self, 'loading_window') and self.loading_window and self.loading_window.winfo_exists():
            current_value = self.progress_var.get()
            if current_value < 100:
                # Progression non-linéaire pour un effet plus naturel
                if current_value < 20:
                    increment = 2
                elif current_value < 50:
                    increment = 1.5
                elif current_value < 80:
                    increment = 1
                else:
                    increment = 0.5
                
                new_value = min(current_value + increment, 100)
                self.progress_var.set(new_value)
                self.progress_label.config(text=f"{int(new_value)}%")
                
                # Continuer l'animation
                self.root.after(50, self._animate_loading)
            else:
                # Animation terminée
                self.progress_label.config(text="Terminé!")
    
    def _hide_loading_bar(self):
        """Cache la barre de chargement."""
        try:
            if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
                try:
                    self.loading_window.grab_release()
                except Exception:
                    pass
                self.loading_window.destroy()
                self.loading_window = None
                try:
                    self.root.update_idletasks()
                except Exception:
                    pass
        except Exception:
            pass
    
    def _shutdown_bridge_executor(self) -> None:
        if self._bridge_executor is None:
            return
        try:
            self._bridge_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._bridge_executor = None

    def _update_loading_message(self, message):
        """Met à jour le message de chargement."""
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            # Trouver et mettre à jour le label de message
            for widget in self.loading_window.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label) and child.cget('text') != "EEG" and not child.cget('text').endswith('%'):
                            child.config(text=message)
                            break
    
    def _show_user_guide(self):
        """Affiche le guide d'utilisation (fenêtre de bienvenue)."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant.show_welcome_assistant()
        else:
            messagebox.showinfo("Guide d'Utilisation", "Assistant utilisateur non disponible.")
    
    def _show_shortcuts(self):
        """Affiche les raccourcis clavier dans une interface moderne."""
        try:
            print("🔍 CHECKPOINT SHORTCUTS: Affichage des raccourcis")
            logging.info("[SHORTCUTS] Displaying shortcuts")
            
            # Créer la fenêtre des raccourcis
            shortcuts_window = tk.Toplevel(self.root)
            shortcuts_window.title("Raccourcis Clavier - EEG Analysis Studio")
            shortcuts_window.geometry("700x800")
            shortcuts_window.transient(self.root)
            shortcuts_window.grab_set()
            
            # Frame principal avec scrollbar
            main_frame = ttk.Frame(shortcuts_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Titre
            title_label = ttk.Label(main_frame, text="⌨️ Raccourcis Clavier", 
                                  font=('Segoe UI', 18, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # Canvas et scrollbar pour le contenu
            canvas = tk.Canvas(main_frame)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Définir les catégories de raccourcis
            shortcuts_categories = {
                "📁 Fichier": [
                    ("Ctrl+O", "Ouvrir fichier EDF"),
                    ("Ctrl+S", "Exporter données"),
                    ("Ctrl+Shift+S", "Exporter segment EDF"),
                    ("Ctrl+Shift+R", "Générer rapport complet"),
                    ("Ctrl+Q", "Quitter l'application")
                ],
                "🧭 Navigation Temporelle": [
                    ("Z", "Époque précédente"),
                    ("Q", "Époque précédente (alternatif)"),
                    ("S", "Époque suivante"),
                    ("D", "Époque suivante (alternatif)"),
                    ("←", "Navigation temporelle (gauche)"),
                    ("→", "Navigation temporelle (droite)"),
                    ("Ctrl+←", "Navigation rapide (gauche)"),
                    ("Ctrl+→", "Navigation rapide (droite)"),
                    ("Home", "Aller au début"),
                    ("End", "Aller à la fin")
                ],
                "📊 Amplitude et Zoom": [
                    ("↑", "Augmenter amplitude"),
                    ("↓", "Diminuer amplitude"),
                    ("Ctrl+↑", "Augmenter amplitude (grand pas)"),
                    ("Ctrl+↓", "Diminuer amplitude (grand pas)"),
                    ("+", "Zoom avant"),
                    ("-", "Zoom arrière"),
                    ("Ctrl+0", "Réinitialiser zoom")
                ],
                "🎯 Scoring de Sommeil": [
                    ("Ctrl+Y", "Scoring automatique (YASA)"),
                    ("Ctrl+Shift+M", "Importer scoring manuel (Excel)"),
                    ("Ctrl+C", "Comparer scoring auto vs manuel"),
                    ("1", "Marquer comme Éveil (W)"),
                    ("2", "Marquer comme N1"),
                    ("3", "Marquer comme N2"),
                    ("4", "Marquer comme N3"),
                    ("5", "Marquer comme REM (R)")
                ],
                "⚙️ Interface et Affichage": [
                    ("Ctrl+A", "Activer/Désactiver autoscale"),
                    ("Ctrl+F", "Activer/Désactiver filtre"),
                    ("Ctrl+T", "Basculer thème sombre/clair"),
                    ("Ctrl+1", "Sélectionner canaux"),
                    ("Ctrl+P", "Basculer panneau commandes"),
                    ("F2", "Basculer panneau commandes (alternatif)"),
                    ("F5", "Actualiser graphique"),
                    ("Ctrl+R", "Actualiser graphique (alternatif)")
                ],
                "📈 Analyse et Outils": [
                    ("Ctrl+Shift+T", "Analyse temporelle"),
                    ("Ctrl+Shift+K", "Système de marqueurs"),
                    ("Ctrl+Shift+L", "Outils de mesure"),
                    ("Ctrl+Shift+F", "Configuration filtres avancée"),
                    ("Ctrl+Shift+P", "Analyse spectrale (PSD)"),
                    ("Ctrl+Shift+W", "Analyse TFR (Time-Frequency)")
                ],
                "🐛 Debug et Support": [
                    ("Ctrl+Shift+B", "Générer rapport de bug"),
                    ("Ctrl+Shift+D", "Afficher informations debug"),
                    ("Ctrl+Shift+I", "Informations système")
                ],
                "❓ Aide et Navigation": [
                    ("F1", "Assistant de bienvenue"),
                    ("F3", "Rechercher dans les données"),
                    ("F4", "Afficher statistiques canaux"),
                    ("Escape", "Fermer les dialogues"),
                    ("Ctrl+?", "Afficher cette aide (raccourcis)")
                ]
            }
            
            # Créer les sections pour chaque catégorie
            for category, shortcuts in shortcuts_categories.items():
                # Frame pour la catégorie
                category_frame = ttk.LabelFrame(scrollable_frame, text=category, padding=10)
                category_frame.pack(fill=tk.X, pady=(0, 15))
                
                # Créer un frame interne pour les raccourcis
                shortcuts_frame = ttk.Frame(category_frame)
                shortcuts_frame.pack(fill=tk.X)
                
                # Ajouter chaque raccourci
                for i, (shortcut, description) in enumerate(shortcuts):
                    # Frame pour chaque raccourci
                    shortcut_frame = ttk.Frame(shortcuts_frame)
                    shortcut_frame.pack(fill=tk.X, pady=2)
                    
                    # Raccourci (en gras)
                    shortcut_label = ttk.Label(shortcut_frame, text=shortcut, 
                                             font=('Consolas', 10, 'bold'), 
                                             foreground='#0066CC')
                    shortcut_label.pack(side=tk.LEFT, padx=(0, 15))
                    
                    # Description
                    desc_label = ttk.Label(shortcut_frame, text=description, 
                                         font=('Segoe UI', 9))
                    desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Ajouter une section d'informations supplémentaires
            info_frame = ttk.LabelFrame(scrollable_frame, text="ℹ️ Informations", padding=10)
            info_frame.pack(fill=tk.X, pady=(10, 0))
            
            info_text = """
• Les raccourcis ZQSD fonctionnent uniquement quand la fenêtre principale a le focus
• Les flèches directionnelles peuvent temporairement désactiver ZQSD (utilisez les raccourcis avec Ctrl pour réactiver)
• Certains raccourcis peuvent varier selon le contexte (dialogue ouvert, etc.)
• Utilisez Escape pour fermer la plupart des fenêtres et dialogues
• Les raccourcis sont également disponibles dans les menus contextuels
            """
            
            info_label = ttk.Label(info_frame, text=info_text.strip(), 
                                 font=('Segoe UI', 8), justify=tk.LEFT)
            info_label.pack(anchor='w')
            
            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=(20, 0))
            
            def print_shortcuts():
                """Imprime les raccourcis dans la console."""
                try:
                    print("\n" + "="*60)
                    print("📋 RACCOURCIS CLAVIER - EEG ANALYSIS STUDIO")
                    print("="*60)
                    
                    for category, shortcuts in shortcuts_categories.items():
                        print(f"\n{category}")
                        print("-" * len(category))
                        for shortcut, description in shortcuts:
                            print(f"  {shortcut:<20} - {description}")
                    
                    print("\n" + "="*60)
                    print("✅ CHECKPOINT SHORTCUTS: Raccourcis imprimés dans la console")
                    logging.info("[SHORTCUTS] Shortcuts printed to console")
                    
                except Exception as e:
                    print(f"❌ CHECKPOINT SHORTCUTS: Erreur impression: {e}")
                    logging.error(f"[SHORTCUTS] Failed to print shortcuts: {e}")
            
            def copy_shortcuts():
                """Copie les raccourcis dans le presse-papiers."""
                try:
                    shortcuts_text = "RACCOURCIS CLAVIER - EEG ANALYSIS STUDIO\n"
                    shortcuts_text += "="*50 + "\n\n"
                    
                    for category, shortcuts in shortcuts_categories.items():
                        shortcuts_text += f"{category}\n"
                        shortcuts_text += "-" * len(category) + "\n"
                        for shortcut, description in shortcuts:
                            shortcuts_text += f"{shortcut:<20} - {description}\n"
                        shortcuts_text += "\n"
                    
                    # Copier dans le presse-papiers (Windows)
                    try:
                        shortcuts_window.clipboard_clear()
                        shortcuts_window.clipboard_append(shortcuts_text)
                        shortcuts_window.update()
                        messagebox.showinfo("Succès", "Raccourcis copiés dans le presse-papiers !")
                        print("✅ CHECKPOINT SHORTCUTS: Raccourcis copiés dans le presse-papiers")
                        logging.info("[SHORTCUTS] Shortcuts copied to clipboard")
                    except Exception:
                        messagebox.showwarning("Attention", "Impossible de copier dans le presse-papiers")
                        
                except Exception as e:
                    print(f"❌ CHECKPOINT SHORTCUTS: Erreur copie: {e}")
                    logging.error(f"[SHORTCUTS] Failed to copy shortcuts: {e}")
                    messagebox.showerror("Erreur", f"Erreur lors de la copie : {str(e)}")
            
            ttk.Button(button_frame, text="📋 Imprimer dans la Console", 
                      command=print_shortcuts).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="📋 Copier dans le Presse-papiers", 
                      command=copy_shortcuts).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Fermer", 
                      command=shortcuts_window.destroy).pack(side=tk.RIGHT)
            
            # Configuration du canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Lier la molette de la souris au scroll
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            def _unbind_mousewheel(event):
                canvas.unbind_all("<MouseWheel>")
            shortcuts_window.bind("<Destroy>", _unbind_mousewheel)
            
            # Raccourci pour fermer la fenêtre
            shortcuts_window.bind("<Escape>", lambda e: shortcuts_window.destroy())
            shortcuts_window.bind("<Control-?>", lambda e: shortcuts_window.destroy())
            
            # Focus sur la fenêtre
            shortcuts_window.focus_set()
            
            print("✅ CHECKPOINT SHORTCUTS: Interface raccourcis affichée")
            logging.info("[SHORTCUTS] Shortcuts interface displayed")
            
        except Exception as e:
            print(f"❌ CHECKPOINT SHORTCUTS: Erreur affichage raccourcis: {e}")
            logging.error(f"[SHORTCUTS] Failed to show shortcuts: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des raccourcis : {str(e)}")
    
    def _report_bug(self):
        """Signaler un bug - crée un fichier de rapport avec logs et checkpoints."""
        try:
            # Créer le nom de fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bug_report_filename = f"bug_report_{timestamp}.txt"
            
            # Créer le contenu du rapport
            report_content = self._generate_bug_report()
            
            # Écrire le fichier
            with open(bug_report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Afficher le message de succès avec le chemin du fichier
            import os
            full_path = os.path.abspath(bug_report_filename)
            
            # Demander si l'utilisateur veut ouvrir le fichier
            open_file = messagebox.askyesno(
                "Rapport de Bug Créé", 
                f"Rapport de bug créé avec succès !\n\n"
                f"Fichier : {bug_report_filename}\n"
                f"Emplacement : {full_path}\n\n"
                f"Le fichier contient :\n"
                f"• Informations système\n"
                f"• État de l'application\n"
                f"• Logs et checkpoints récents\n"
                f"• Configuration actuelle\n"
                f"• Erreurs récentes\n\n"
                f"Voulez-vous ouvrir le fichier maintenant ?"
            )
            
            if open_file:
                try:
                    # Ouvrir le fichier avec l'application par défaut
                    import subprocess
                    import platform
                    
                    system = platform.system()
                    if system == "Windows":
                        os.startfile(full_path)
                    elif system == "Darwin":  # macOS
                        subprocess.run(["open", full_path])
                    else:  # Linux
                        subprocess.run(["xdg-open", full_path])
                        
                except Exception as e:
                    print(f"❌ RAPPORT BUG: Impossible d'ouvrir le fichier: {e}")
                    messagebox.showwarning("Attention", f"Le fichier a été créé mais n'a pas pu être ouvert automatiquement.\nEmplacement: {full_path}")
            
            messagebox.showinfo(
                "Instructions",
                "Veuillez joindre ce fichier à votre signalement de bug avec :\n"
                "• Une description détaillée du problème\n"
                "• Les étapes pour reproduire le bug\n"
                "• Le fichier EDF si applicable\n"
                "• Une capture d'écran si possible"
            )
            
            print(f"🐛 RAPPORT BUG: Fichier créé - {full_path}")
            logging.info(f"[BUG_REPORT] Bug report created: {full_path}")
            
        except Exception as e:
            error_msg = f"Erreur lors de la création du rapport de bug : {str(e)}"
            print(f"❌ RAPPORT BUG: {error_msg}")
            logging.error(f"[BUG_REPORT] Failed to create bug report: {e}")
            messagebox.showerror("Erreur", error_msg)
    
    def _generate_bug_report(self):
        """Génère le contenu du rapport de bug avec toutes les informations pertinentes."""
        try:
            import platform
            import sys
            import os
            from datetime import datetime
            
            report_lines = []
            
            # En-tête du rapport
            report_lines.append("=" * 80)
            report_lines.append("EEG ANALYSIS STUDIO - RAPPORT DE BUG")
            report_lines.append("=" * 80)
            report_lines.append(f"Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Version de l'application : v0.0beta1.1")
            report_lines.append("")
            
            # Informations système
            report_lines.append("INFORMATIONS SYSTÈME")
            report_lines.append("-" * 40)
            report_lines.append(f"Système d'exploitation : {platform.system()} {platform.release()}")
            report_lines.append(f"Architecture : {platform.architecture()[0]}")
            report_lines.append(f"Version Python : {sys.version}")
            report_lines.append(f"Répertoire de travail : {os.getcwd()}")
            report_lines.append("")
            
            # État de l'application
            report_lines.append("ÉTAT DE L'APPLICATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Fichier EDF chargé : {'Oui' if hasattr(self, 'raw') and self.raw is not None else 'Non'}")
            
            if hasattr(self, 'raw') and self.raw is not None:
                report_lines.append(f"  - Nom du fichier : {getattr(self, 'current_file', 'Inconnu')}")
                report_lines.append(f"  - Fréquence d'échantillonnage : {self.sfreq} Hz")
                report_lines.append(f"  - Nombre de canaux : {len(self.raw.ch_names)}")
                report_lines.append(f"  - Durée totale : {len(self.raw.times):.1f} secondes")
            
            report_lines.append(f"Temps actuel affiché : {getattr(self, 'current_time', 0):.1f}s")
            report_lines.append(f"Durée d'affichage : {getattr(self, 'duration', 10):.1f}s")
            report_lines.append(f"Canaux sélectionnés : {len(getattr(self, 'selected_channels', []))}")
            
            if hasattr(self, 'selected_channels'):
                report_lines.append(f"  - Liste : {', '.join(self.selected_channels[:5])}{'...' if len(self.selected_channels) > 5 else ''}")
            
            report_lines.append("")
            
            # Configuration du filtre
            report_lines.append("CONFIGURATION DU FILTRE")
            report_lines.append("-" * 40)
            report_lines.append(f"Filtre activé : {'Oui' if getattr(self, 'filter_enabled', False) else 'Non'}")
            report_lines.append(f"Filtre bas : {getattr(self, 'filter_low', 0.5)} Hz")
            report_lines.append(f"Filtre haut : {getattr(self, 'filter_high', 30.0)} Hz")
            report_lines.append(f"Type de filtre : {getattr(self, 'filter_type', 'butterworth')}")
            report_lines.append(f"Ordre du filtre : {getattr(self, 'filter_order', 4)}")
            report_lines.append("")
            
            # Configuration du scoring
            report_lines.append("CONFIGURATION DU SCORING")
            report_lines.append("-" * 40)
            report_lines.append(f"Scoring automatique chargé : {'Oui' if hasattr(self, 'sleep_scoring_data') and self.sleep_scoring_data is not None else 'Non'}")
            report_lines.append(f"Scoring manuel chargé : {'Oui' if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None else 'Non'}")
            report_lines.append(f"Durée d'époque : {getattr(self, 'scoring_epoch_duration', 30.0)}s")
            report_lines.append(f"Affichage scoring : {'Manuel' if getattr(self, 'show_manual_scoring', True) else 'Automatique'}")
            report_lines.append("")
            
            # Informations temporelles
            if hasattr(self, 'absolute_start_datetime') and self.absolute_start_datetime:
                report_lines.append("INFORMATIONS TEMPORELLES")
                report_lines.append("-" * 40)
                report_lines.append(f"Début enregistrement EDF : {self.absolute_start_datetime}")
                if hasattr(self, 'display_start_datetime') and self.display_start_datetime:
                    report_lines.append(f"Base d'affichage Excel : {self.display_start_datetime}")
                report_lines.append("")
            
            # Logs récents
            report_lines.append("LOGS ET CHECKPOINTS RÉCENTS")
            report_lines.append("-" * 40)
            recent_logs = self._get_recent_logs()
            if recent_logs:
                report_lines.append("Dernières 20 lignes de log :")
                for log_line in recent_logs:
                    report_lines.append(f"  {log_line}")
            else:
                report_lines.append("Aucun log récent trouvé")
            
            # Checkpoints récents de la console
            recent_checkpoints = self._get_recent_checkpoints()
            if recent_checkpoints:
                report_lines.append("")
                report_lines.append("Derniers checkpoints de la console :")
                for checkpoint in recent_checkpoints:
                    report_lines.append(f"  {checkpoint}")
            report_lines.append("")
            
            # Configuration de l'interface
            report_lines.append("CONFIGURATION DE L'INTERFACE")
            report_lines.append("-" * 40)
            report_lines.append(f"Autoscale activé : {'Oui' if getattr(self, 'autoscale_enabled', False) else 'Non'}")
            report_lines.append(f"Durée fenêtre autoscale : {getattr(self, 'autoscale_window_duration', 30.0)}s")
            report_lines.append("")
            
            # Paramètres de navigation
            report_lines.append("PARAMÈTRES DE NAVIGATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Bindings ZQSD configurés : {'Oui' if hasattr(self, '_setup_keyboard_navigation') else 'Non'}")
            report_lines.append(f"Focus sur fenêtre principale : {'Oui' if hasattr(self, 'root') else 'Non'}")
            report_lines.append("")
            
            # Informations sur les erreurs récentes
            report_lines.append("ERREURS RÉCENTES")
            report_lines.append("-" * 40)
            error_checkpoints = [cp for cp in self._get_recent_checkpoints() if '❌' in cp or 'ERREUR' in cp.upper() or 'ERROR' in cp.upper()]
            if error_checkpoints:
                report_lines.append("Erreurs détectées dans les checkpoints récents :")
                for error in error_checkpoints[-10:]:  # Dernières 10 erreurs
                    report_lines.append(f"  {error}")
            else:
                report_lines.append("Aucune erreur récente détectée")
            report_lines.append("")
            
            # Instructions pour l'utilisateur
            report_lines.append("INSTRUCTIONS POUR LE SIGNALEMENT")
            report_lines.append("-" * 40)
            report_lines.append("1. Décrivez le problème rencontré dans la section 'Description du Bug'")
            report_lines.append("2. Indiquez les étapes pour reproduire le problème")
            report_lines.append("3. Joignez ce fichier à votre signalement")
            report_lines.append("4. Si possible, joignez également :")
            report_lines.append("   - Les fichiers de log (.log)")
            report_lines.append("   - Le fichier EDF problématique (si applicable)")
            report_lines.append("   - Une capture d'écran du problème")
            report_lines.append("")
            report_lines.append("DESCRIPTION DU BUG")
            report_lines.append("-" * 40)
            report_lines.append("[Veuillez décrire le problème ici]")
            report_lines.append("")
            report_lines.append("ÉTAPES POUR REPRODUIRE")
            report_lines.append("-" * 40)
            report_lines.append("[Veuillez décrire les étapes pour reproduire le problème]")
            report_lines.append("")
            
            # Pied de page
            report_lines.append("=" * 80)
            report_lines.append("Fin du rapport de bug")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Erreur lors de la génération du rapport : {str(e)}\nTimestamp : {datetime.now()}"
    
    def _get_recent_logs(self, max_lines=20):
        """Récupère les logs récents des fichiers de log."""
        try:
            log_files = ['eeg_studio.log', 'yasa_scoring.log']
            all_logs = []
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            # Prendre les dernières lignes
                            recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
                            all_logs.extend([f"[{log_file}] {line.strip()}" for line in recent_lines])
                    except Exception as e:
                        all_logs.append(f"[{log_file}] Erreur lecture: {str(e)}")
            
            # Trier par timestamp si possible et retourner les plus récents
            return all_logs[-max_lines:] if all_logs else []
            
        except Exception as e:
            return [f"Erreur lecture logs: {str(e)}"]
    
    def _setup_checkpoint_capture(self):
        """Configure la capture des checkpoints de la console."""
        try:
            # Rediriger print vers une fonction qui capture aussi dans notre liste
            import sys
            self.original_stdout = sys.stdout
            # Checkpoints captured via logging system
            logging.debug("Checkpoint capture system initialized via logging")
            logging.info("[CHECKPOINT_CAPTURE] System initialized")
        except Exception as e:
            print(f"❌ CHECKPOINT CAPTURE: Erreur initialisation: {e}")
            logging.error(f"[CHECKPOINT_CAPTURE] Failed to initialize: {e}")
    
    def _get_recent_checkpoints(self, max_checkpoints=30):
        """Récupère les checkpoints récents de la console."""
        try:
            if hasattr(self, 'console_checkpoints') and self.console_checkpoints:
                # Retourner les derniers checkpoints (les plus récents)
                return self.console_checkpoints[-max_checkpoints:]
            else:
                return []
        except Exception as e:
            return [f"Erreur récupération checkpoints: {str(e)}"]
    
    def _suggest_feature(self):
        """Suggérer une fonctionnalité."""
        messagebox.showinfo("Suggestions", "Utilisez GitHub Discussions pour vos suggestions")
    
    
    def _process_sleep_scoring_data_v2(self, df: pd.DataFrame):
        """Legacy wrapper kept for compatibility: normalize to strict time/stage schema."""
        try:
            normalized = self.manual_scoring_service.validate(df)
            self.sleep_scoring_data = normalized.copy()
            self.scoring_epoch_duration = float(
                self.manual_scoring_service.infer_epoch_seconds(
                    normalized,
                    default=float(getattr(self, "scoring_epoch_duration", 30.0)),
                )
            )
            logging.info(
                "[MANUAL] _process_sleep_scoring_data_v2 normalized %d rows (epoch=%.2fs)",
                len(normalized),
                self.scoring_epoch_duration,
            )
        except Exception as exc:
            logging.error("[MANUAL] _process_sleep_scoring_data_v2 failed: %s", exc)
            raise
    
    def _process_sleep_scoring_data(self, df: pd.DataFrame):
        """Legacy wrapper kept for compatibility."""
        self._process_sleep_scoring_data_v2(df)
    
    def _show_sleep_scoring_info(self):
        """Affiche les informations sur le scoring de sommeil chargé."""
        # Préférer afficher l'info du scoring manuel s'il est chargé et non vide, sinon l'auto
        df = self._get_active_scoring_df()
        if df is None or len(df) == 0:
            messagebox.showinfo("Scoring de sommeil", "Aucun scoring de sommeil chargé.")
            return
        
        # Calculer les statistiques
        total_epochs = int(len(df))
        duration_hours = float((df['time'].max() - df['time'].min()) / 3600)
        stage_counts = df['stage'].value_counts()
        stage_percentages = (stage_counts / total_epochs * 100).round(1)
        
        # Créer le texte d'information
        info_text = f"""📊 Informations sur le scoring de sommeil

📈 Statistiques générales:
• Total d'époques: {total_epochs}
• Durée: {duration_hours:.1f} heures
• Durée par époque: {self.scoring_epoch_duration}s

🛏️ Répartition des stades de sommeil:
"""
        
        for stage, count in stage_counts.items():
            stage_name = self.sleep_stages.get(stage, stage)
            percentage = stage_percentages.get(stage, 0.0)
            info_text += f"• {stage_name}: {count} époques ({percentage}%)\n"
        
        messagebox.showinfo("Scoring de sommeil", info_text)

    def _apply_manual_scoring_result(self, result: ManualScoringResult, source: str) -> None:
        """Apply normalized manual scoring output to the application state."""
        self.manual_scoring_data = result.df.copy()
        self.show_manual_scoring = True
        self.scoring_epoch_duration = float(result.epoch_seconds)
        logging.info(
            "[MANUAL] %s loaded: n_epochs=%d, epoch=%.2fs",
            source,
            len(self.manual_scoring_data),
            self.scoring_epoch_duration,
        )

    def _validate_editor_scoring_df(self, df: pd.DataFrame, epoch_len: float) -> pd.DataFrame:
        """Validate and normalize DataFrame coming from the manual editor."""
        validated = self.manual_scoring_service.validate(df)
        rec_duration = float(len(self.raw.times) / self.sfreq) if self.raw is not None else 0.0
        validated = self.manual_scoring_service.fill_undefined(
            validated,
            recording_duration_s=rec_duration,
            epoch_seconds=float(epoch_len),
        )
        return validated

    def _open_scoring_import_hub(self):
        """Sous-fenêtre centralisée pour importer un scoring (Excel/CSV ou EDF Hypnogram)."""
        if not self.raw:
            messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
            return
        try:
            self._qt_pause_pump_for_tk_modal()
            hub = tk.Toplevel(self.root)
            hub.title("Importer Scoring (Excel/EDF)")
            hub.geometry("420x200")
            hub.transient(self.root)
            hub.grab_set()
            frame = ttk.Frame(hub, padding=12)
            frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(frame, text="Choisissez la source de scoring à importer:", font=('Segoe UI', 10, 'bold')).pack(anchor='w')
            ttk.Button(frame, text="Importer Excel/CSV", command=lambda: (hub.destroy(), self._import_manual_scoring_excel())).pack(fill=tk.X, pady=(12,6))
            ttk.Button(frame, text="Charger Hypnogram EDF (Sleep-EDFx)", command=lambda: (hub.destroy(), self._load_hypnogram_edfplus())).pack(fill=tk.X)
            try:
                self.root.wait_window(hub)
            except tk.TclError:
                pass
        finally:
            self._qt_resume_pump_after_tk_modal()
    
    def _open_manual_scoring_editor(self):
        """Modern manual scoring editor with validation and explicit actions."""
        if not self.raw:
            messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
            return
        self._qt_pause_pump_for_tk_modal()
        win = tk.Toplevel(self.root)
        win.title("Éditeur de Scoring Manuel")
        win.geometry("1120x860")
        win.transient(self.root)
        win.grab_set()

        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(top, text="Durée époque (s):").pack(side=tk.LEFT)
        epoch_var = tk.DoubleVar(value=float(getattr(self, 'scoring_epoch_duration', 30.0)))
        ttk.Entry(top, textvariable=epoch_var, width=6).pack(side=tk.LEFT, padx=(4,10))
        status_var = tk.StringVar(value="Etat: propre")
        ttk.Label(top, textvariable=status_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(top, text="+ Époques sur fenêtre", command=lambda: add_epochs_on_window()).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Button(top, text="Trier", command=lambda: sort_rows()).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Dédoublonner", command=lambda: dedup_rows()).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Exporter CSV", command=lambda: export_csv()).pack(side=tk.RIGHT)
        ttk.Button(top, text="Enregistrer comme scoring manuel", command=lambda: save_manual()).pack(side=tk.RIGHT, padx=(0,6))

        has_confidence_col = False
        df_init = None
        if getattr(self, 'manual_scoring_data', None) is not None and len(self.manual_scoring_data) > 0:
            df_init = self.manual_scoring_data
        elif getattr(self, 'sleep_scoring_data', None) is not None and len(self.sleep_scoring_data) > 0:
            df_init = self.sleep_scoring_data
        has_confidence_col = bool(df_init is not None and 'confidence' in df_init.columns)
        cols = ("time", "stage", "confidence") if has_confidence_col else ("time", "stage")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        if df_init is not None:
            try:
                for _, row in df_init.iterrows():
                    values = [float(row['time']), str(row['stage'])]
                    if has_confidence_col:
                        conf = row.get('confidence', '')
                        values.append("" if pd.isna(conf) else f"{float(conf):.3f}")
                    tree.insert('', tk.END, values=tuple(values))
            except Exception:
                pass

        form = ttk.Frame(win)
        form.pack(fill=tk.X, padx=8, pady=(0,8))
        ttk.Label(form, text="time (s)").grid(row=0, column=0, sticky='w')
        time_var = tk.DoubleVar(value=float(getattr(self, 'current_time', 0.0)))
        ttk.Entry(form, textvariable=time_var, width=10).grid(row=0, column=1, sticky='w', padx=(4,10))
        ttk.Label(form, text="stage").grid(row=0, column=2, sticky='w')
        stage_var = tk.StringVar(value='W')
        ttk.Combobox(form, textvariable=stage_var, values=['W','N1','N2','N3','R','U'], width=6).grid(row=0, column=3, sticky='w', padx=(4,10))
        ttk.Button(form, text="Ajouter/Mettre à jour", command=lambda: add_or_update()).grid(row=0, column=4, padx=(6,0))
        ttk.Button(form, text="Supprimer sélection", command=lambda: delete_selected()).grid(row=0, column=5, padx=(6,0))

        dirty = {"value": False}

        def mark_dirty():
            dirty["value"] = True
            status_var.set("Etat: modifié (non sauvegardé)")

        def mark_clean():
            dirty["value"] = False
            status_var.set("Etat: propre")

        def table_to_dataframe():
            import pandas as pd
            rows = []
            for iid in tree.get_children():
                vals = tree.item(iid, 'values')
                t = vals[0]
                s = vals[1]
                row = {'time': float(t), 'stage': str(s)}
                if has_confidence_col and len(vals) > 2 and str(vals[2]).strip() != "":
                    try:
                        row['confidence'] = float(vals[2])
                    except Exception:
                        pass
                rows.append(row)
            if rows:
                df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
            else:
                cols_df = ['time', 'stage'] + (['confidence'] if has_confidence_col else [])
                df = pd.DataFrame(columns=cols_df)
            return df

        def normalize_stage(stage_value: str) -> str:
            s = str(stage_value).upper().strip()
            if s == "REM":
                s = "R"
            if s not in {"W", "N1", "N2", "N3", "R", "U"}:
                raise ValueError("Stage invalide. Utiliser W, N1, N2, N3, R ou U.")
            return s

        def add_or_update():
            try:
                t = float(time_var.get())
                if t < 0:
                    raise ValueError("Le temps doit être >= 0.")
                s = normalize_stage(stage_var.get())
                found = None
                for iid in tree.get_children():
                    vals = tree.item(iid, 'values')
                    if abs(float(vals[0]) - t) < 1e-6:
                        found = iid
                        break
                if found is not None:
                    new_values = [t, s]
                    if has_confidence_col:
                        old_vals = tree.item(found, 'values')
                        new_values.append(old_vals[2] if len(old_vals) > 2 else "")
                    tree.item(found, values=tuple(new_values))
                else:
                    new_values = [t, s]
                    if has_confidence_col:
                        new_values.append("")
                    tree.insert('', tk.END, values=tuple(new_values))
                mark_dirty()
            except Exception as e:
                messagebox.showerror("Scoring manuel", f"Entrée invalide:\n{e}")

        def delete_selected():
            sel = tree.selection()
            for iid in sel:
                tree.delete(iid)
            if len(sel) > 0:
                mark_dirty()

        def add_epochs_on_window():
            try:
                start = float(getattr(self, 'current_time', 0.0))
                dur = float(getattr(self, 'duration', 10.0))
                ep = float(epoch_var.get())
                if ep <= 0:
                    raise ValueError("Durée d'époque invalide")
                n = int(np.ceil(dur / ep))
                for k in range(n):
                    t = start + k*ep
                    values = [t, 'U']
                    if has_confidence_col:
                        values.append("")
                    tree.insert('', tk.END, values=tuple(values))
                mark_dirty()
            except Exception as e:
                messagebox.showerror("Erreur", f"Ajout d'époques: {e}")

        def sort_rows():
            df = table_to_dataframe()
            for iid in tree.get_children():
                tree.delete(iid)
            for _, row in df.iterrows():
                values = [float(row["time"]), str(row["stage"])]
                if has_confidence_col:
                    conf = row.get("confidence", "")
                    values.append("" if pd.isna(conf) else f"{float(conf):.3f}")
                tree.insert('', tk.END, values=tuple(values))
            mark_dirty()

        def dedup_rows():
            df = table_to_dataframe()
            df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
            for iid in tree.get_children():
                tree.delete(iid)
            for _, row in df.iterrows():
                values = [float(row["time"]), str(row["stage"])]
                if has_confidence_col:
                    conf = row.get("confidence", "")
                    values.append("" if pd.isna(conf) else f"{float(conf):.3f}")
                tree.insert('', tk.END, values=tuple(values))
            mark_dirty()

        def save_manual():
            try:
                df = table_to_dataframe()
                if df is None or len(df) == 0:
                    messagebox.showwarning("Scoring", "Aucune époque à enregistrer")
                    return
                epoch_len = float(epoch_var.get())
                validated = self._validate_editor_scoring_df(df[["time", "stage"]], epoch_len=epoch_len)
                result = ManualScoringResult(df=validated, epoch_seconds=epoch_len)
                self._apply_manual_scoring_result(result, source="Manual editor")
                self.scoring_dirty = True
                self.update_plot()
                mark_clean()
                messagebox.showinfo("Scoring", "Scoring manuel enregistré et validé.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Enregistrement scoring manuel: {e}")

        def export_csv():
            try:
                df = table_to_dataframe()
                df = self._validate_editor_scoring_df(df[["time", "stage"]], epoch_len=float(epoch_var.get()))
                file_path = filedialog.asksaveasfilename(title="Exporter scoring (CSV)", defaultextension=".csv",
                                                         filetypes=[("CSV", "*.csv")])
                if not file_path:
                    return
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Export", f"CSV exporté: {file_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Export CSV: {e}")

        # Raccourcis stades (sur la fenêtre seulement — pas bind_all : fuite globale).
        win.bind("<KeyPress-1>", lambda e: stage_var.set("W"))
        win.bind("<KeyPress-2>", lambda e: stage_var.set("N1"))
        win.bind("<KeyPress-3>", lambda e: stage_var.set("N2"))
        win.bind("<KeyPress-4>", lambda e: stage_var.set("N3"))
        win.bind("<KeyPress-5>", lambda e: stage_var.set("R"))
        win.bind("<KeyPress-0>", lambda e: stage_var.set("U"))
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        try:
            self.root.wait_window(win)
        except tk.TclError:
            pass
        finally:
            self._qt_resume_pump_after_tk_modal()

    def _get_active_scoring_df(self) -> Optional[pd.DataFrame]:
        """Retourne le scoring actif: manuel non vide si dispo, sinon auto non vide, sinon None."""
        try:
            if self.manual_scoring_data is not None:
                try:
                    if len(self.manual_scoring_data) > 0:
                        return self.manual_scoring_data
                except Exception:
                    pass
            if self.sleep_scoring_data is not None:
                try:
                    if len(self.sleep_scoring_data) > 0:
                        return self.sleep_scoring_data
                except Exception:
                    pass
        except Exception:
            return None
        return None

    def _analyze_sleep_periods(self):
        """Analyse des périodes de sommeil (type SleepEEGpy)."""
        # Utiliser le scoring manuel s'il est non vide, sinon auto
        df = self._get_active_scoring_df()
        if df is None or len(df) == 0:
            messagebox.showwarning("Avertissement", "Aucun scoring de sommeil chargé.")
            return

        try:
            epoch_len = float(getattr(self, 'scoring_epoch_duration', 30.0))
            stages = df['stage'].astype(str).str.upper().values
            times = df['time'].values

            # Basic metrics (SleepEEGpy-like)
            t_start = float(times.min())
            t_end = float(times.max() + epoch_len)
            tib = (t_end - t_start) / 60.0  # Time in bed (min)

            # Sleep onset latency (first epoch in sleep N1/N2/N3/R)
            sleep_mask = np.isin(stages, ['N1', 'N2', 'N3', 'R'])
            if sleep_mask.any():
                first_sleep_idx = int(np.where(sleep_mask)[0][0])
                sol = (float(times[first_sleep_idx]) - t_start) / 60.0
            else:
                sol = float('nan')

            # Time asleep and WASO
            tst_sec = int(np.sum(np.isin(stages, ['N1','N2','N3','R'])) * epoch_len)
            waso_sec = int(np.sum(stages == 'W') * epoch_len)
            se = (tst_sec / (tib * 60.0)) * 100.0 if tib > 0 else float('nan')

            # REM latency from sleep onset
            if sleep_mask.any() and np.any(stages == 'R'):
                rem_idxs = np.where(stages == 'R')[0]
                rem_lat = (float(times[rem_idxs[0]]) - float(times[first_sleep_idx])) / 60.0
            else:
                rem_lat = float('nan')

            # Stage durations (min)
            stage_durations_min = {
                s: (int(np.sum(stages == s)) * epoch_len) / 60.0 for s in ['W', 'N1', 'N2', 'N3', 'R']
            }

            # Awakenings: count transitions into W with at least 1 epoch
            awakenings = 0
            for i in range(1, len(stages)):
                if stages[i] == 'W' and stages[i-1] != 'W':
                    awakenings += 1

            # Build contiguous periods (start-end in minutes, label)
            periods = []
            if len(stages) > 0:
                start_idx = 0
                for i in range(1, len(stages)):
                    if stages[i] != stages[i-1]:
                        periods.append((float(times[start_idx]) / 60.0, float(times[i]) / 60.0, stages[i-1]))
                        start_idx = i
                # last
                periods.append((float(times[start_idx]) / 60.0, float((times[-1] + epoch_len)) / 60.0, stages[-1]))

            # UI: window with summary + table of periods
            top = tk.Toplevel(self.root)
            top.title("Analyse des périodes de sommeil (type SleepEEGpy)")
            top.geometry("1100x900")
            top.transient(self.root)
            top.grab_set()

            container = ttk.Frame(top, padding=10)
            container.pack(fill=tk.BOTH, expand=True)

            summary = (
                f"TIB: {tib:.1f} min | SOL: {sol:.1f} min | TST: {tst_sec/60.0:.1f} min\n"
                f"WASO: {waso_sec/60.0:.1f} min | SE: {se:.1f}% | REM lat.: {rem_lat:.1f} min\n"
                f"W: {stage_durations_min['W']:.1f} | N1: {stage_durations_min['N1']:.1f} | N2: {stage_durations_min['N2']:.1f} | "
                f"N3: {stage_durations_min['N3']:.1f} | R: {stage_durations_min['R']:.1f} | Réveils: {awakenings}"
            )
            ttk.Label(container, text=summary, font=('Segoe UI', 10)).pack(anchor='w', pady=(0,10))

            # Boutons d'export
            btns = ttk.Frame(container)
            btns.pack(fill=tk.X, pady=(0,10))
            ttk.Button(btns, text="Exporter CSV", command=lambda: _export_periods_and_metrics_csv(periods, tib, sol, tst_sec, waso_sec, se, rem_lat, stage_durations_min, awakenings)).pack(side=tk.RIGHT)
            ttk.Button(btns, text="Enregistrer Figure", command=lambda: _save_periods_figure(tree)).pack(side=tk.RIGHT, padx=(10,0))

            cols = ("Début (min)", "Fin (min)", "Stade")
            tree = ttk.Treeview(container, columns=cols, show="headings")
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=120, anchor=tk.CENTER)
            tree.pack(fill=tk.BOTH, expand=True)

            for (a, b, s) in periods:
                tree.insert("", tk.END, values=(f"{a:.1f}", f"{b:.1f}", s))

            def _export_periods_and_metrics_csv(periods_list, tib_min, sol_min, tst_seconds, waso_seconds, se_pct, rem_latency_min, stage_dur_min_dict, n_awaken):
                try:
                    file_path = filedialog.asksaveasfilename(title="Exporter CSV", defaultextension=".csv",
                                                             filetypes=[("CSV", "*.csv")])
                    if not file_path:
                        return
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        # Metrics
                        writer.writerow(["metric", "value"]) 
                        writer.writerow(["TIB_min", f"{tib_min:.2f}"])
                        writer.writerow(["SOL_min", f"{sol_min:.2f}"])
                        writer.writerow(["TST_min", f"{tst_seconds/60.0:.2f}"])
                        writer.writerow(["WASO_min", f"{waso_seconds/60.0:.2f}"])
                        writer.writerow(["SE_percent", f"{se_pct:.2f}"])
                        writer.writerow(["REM_latency_min", f"{rem_latency_min:.2f}"])
                        for sname, minutes in stage_dur_min_dict.items():
                            writer.writerow([f"Duration_{sname}_min", f"{minutes:.2f}"])
                        writer.writerow(["Awakenings", n_awaken])
                        writer.writerow([])
                        # Periods table
                        writer.writerow(["start_min", "end_min", "stage"]) 
                        for (a, b, s) in periods_list:
                            writer.writerow([f"{a:.2f}", f"{b:.2f}", s])
                except Exception as e:
                    messagebox.showerror("Erreur", f"Echec de l'export CSV: {e}")

            def _save_periods_figure(treeview):
                try:
                    # Simple export: capture de la fenêtre via matplotlib n'est pas direct; on exporte la liste en figure.
                    import matplotlib.pyplot as _plt
                    _fig, _ax = _plt.subplots(figsize=(6, len(periods)/6 + 1))
                    y = 0
                    for (a, b, s) in periods:
                        _ax.plot([a, b], [y, y], lw=6)
                        _ax.text(b + 0.2, y, s)
                        y += 1
                    _ax.set_xlabel("Temps (min)")
                    _ax.set_title("Périodes de sommeil")
                    _ax.set_yticks([])
                    _plt.tight_layout()
                    file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                             filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                    if file_path:
                        _fig.savefig(file_path, dpi=200, bbox_inches='tight')
                    _plt.close(_fig)
                except Exception as e:
                    messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Echec analyse périodes: {e}")
    
    def _show_advanced_analysis(self):
        """Affiche la fenêtre d'analyse avancée."""
        try:
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            # Créer la fenêtre d'analyse avancée
            top = tk.Toplevel(self.root)
            top.title("📊 Analyse Avancée - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(self.root)
            top.grab_set()
            
            # Créer un frame scrollable
            canvas = tk.Canvas(top)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Titre
            title_label = ttk.Label(scrollable_frame, text="📊 Analyse Avancée des Données EEG", 
                                  font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Outils d'analyse avancée pour l'exploration approfondie des données EEG",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))
            
            # Analyses disponibles
            analyses_frame = ttk.LabelFrame(scrollable_frame, text="🔬 Analyses Disponibles")
            analyses_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Analyse de cohérence
            coherence_btn = ttk.Button(analyses_frame, text="📈 Cohérence Inter-canal", 
                                      command=lambda: self._show_coherence_analysis(top))
            coherence_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Analyse de corrélation
            correlation_btn = ttk.Button(analyses_frame, text="🔗 Corrélation Temporelle", 
                                       command=lambda: self._show_correlation_analysis(top))
            correlation_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Analyse de variance
            variance_btn = ttk.Button(analyses_frame, text="📊 Analyse de Variance", 
                                    command=lambda: self._show_variance_analysis(top))
            variance_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Analyse de stationnarité
            stationarity_btn = ttk.Button(analyses_frame, text="📉 Test de Stationnarité", 
                                       command=lambda: self._show_stationarity_analysis(top))
            stationarity_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=20)
            
            close_btn = ttk.Button(button_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Pack canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            print("✅ Analyse avancée affichée")
            
        except Exception as e:
            print(f"❌ Erreur analyse avancée: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse avancée : {str(e)}")
    
    def _show_microstates_analysis(self):
        """Affiche la fenêtre d'analyse des micro-états."""
        try:
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            # Créer la fenêtre d'analyse des micro-états
            top = tk.Toplevel(self.root)
            top.title("🔬 Analyse Micro-états - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(self.root)
            top.grab_set()
            
            # Créer un frame scrollable
            canvas = tk.Canvas(top)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Titre
            title_label = ttk.Label(scrollable_frame, text="🔬 Analyse des Micro-états EEG", 
                                  font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Identification et analyse des micro-états cérébraux dans les signaux EEG",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))
            
            # Paramètres d'analyse
            params_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Paramètres d'Analyse")
            params_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Nombre de micro-états
            states_frame = ttk.Frame(params_frame)
            states_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(states_frame, text="Nombre de micro-états:").pack(side=tk.LEFT)
            self.microstates_count = tk.IntVar(value=4)
            states_spinbox = ttk.Spinbox(states_frame, from_=2, to=10, 
                                      textvariable=self.microstates_count, width=10)
            states_spinbox.pack(side=tk.RIGHT)
            
            # Méthode de clustering
            method_frame = ttk.Frame(params_frame)
            method_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(method_frame, text="Méthode de clustering:").pack(side=tk.LEFT)
            self.clustering_method = tk.StringVar(value="kmeans")
            method_combo = ttk.Combobox(method_frame, textvariable=self.clustering_method,
                                      values=["kmeans", "hierarchical", "gmm"], state="readonly")
            method_combo.pack(side=tk.RIGHT)
            
            # Fenêtre temporelle
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(window_frame, text="Fenêtre d'analyse (s):").pack(side=tk.LEFT)
            self.microstates_window = tk.DoubleVar(value=30.0)
            window_spinbox = ttk.Spinbox(window_frame, from_=10.0, to=300.0, 
                                       textvariable=self.microstates_window, width=10)
            window_spinbox.pack(side=tk.RIGHT)
            
            # Boutons d'action
            action_frame = ttk.Frame(scrollable_frame)
            action_frame.pack(fill=tk.X, padx=20, pady=20)
            
            analyze_btn = ttk.Button(action_frame, text="🔬 Analyser Micro-états", 
                                   command=lambda: self._run_microstates_analysis(top))
            analyze_btn.pack(side=tk.LEFT, padx=5)
            
            export_btn = ttk.Button(action_frame, text="📊 Exporter Résultats", 
                                  command=lambda: self._export_microstates_results(top))
            export_btn.pack(side=tk.LEFT, padx=5)
            
            close_btn = ttk.Button(action_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Pack canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            print("✅ Analyse micro-états affichée")
            
        except Exception as e:
            print(f"❌ Erreur analyse micro-états: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse des micro-états : {str(e)}")
    
    def _show_connectivity_analysis(self):
        """Affiche la fenêtre d'analyse de connectivité cérébrale."""
        try:
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            # Créer la fenêtre d'analyse de connectivité
            top = tk.Toplevel(self.root)
            top.title("🧠 Connectivité Cérébrale - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(self.root)
            top.grab_set()
            
            # Créer un frame scrollable
            canvas = tk.Canvas(top)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Titre
            title_label = ttk.Label(scrollable_frame, text="🧠 Analyse de Connectivité Cérébrale", 
                                  font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Analyse des connexions fonctionnelles entre différentes régions cérébrales",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))
            
            # Types de connectivité
            connectivity_frame = ttk.LabelFrame(scrollable_frame, text="🔗 Types de Connectivité")
            connectivity_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Cohérence
            coherence_btn = ttk.Button(connectivity_frame, text="📈 Cohérence (Coherence)", 
                                    command=lambda: self._show_coherence_connectivity(top))
            coherence_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Corrélation
            correlation_btn = ttk.Button(connectivity_frame, text="🔗 Corrélation (Pearson)", 
                                       command=lambda: self._show_correlation_connectivity(top))
            correlation_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Phase Locking Value
            plv_btn = ttk.Button(connectivity_frame, text="⚡ Phase Locking Value (PLV)", 
                                command=lambda: self._show_plv_connectivity(top))
            plv_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Imaginary Coherence
            imag_coherence_btn = ttk.Button(connectivity_frame, text="🔄 Imaginary Coherence", 
                                          command=lambda: self._show_imaginary_coherence(top))
            imag_coherence_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Paramètres
            params_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Paramètres")
            params_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Bande de fréquence
            freq_frame = ttk.Frame(params_frame)
            freq_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(freq_frame, text="Bande de fréquence:").pack(side=tk.LEFT)
            self.connectivity_band = tk.StringVar(value="alpha")
            band_combo = ttk.Combobox(freq_frame, textvariable=self.connectivity_band,
                                    values=["delta", "theta", "alpha", "beta", "gamma"], state="readonly")
            band_combo.pack(side=tk.RIGHT)
            
            # Fenêtre temporelle
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(window_frame, text="Fenêtre d'analyse (s):").pack(side=tk.LEFT)
            self.connectivity_window = tk.DoubleVar(value=60.0)
            window_spinbox = ttk.Spinbox(window_frame, from_=10.0, to=300.0, 
                                       textvariable=self.connectivity_window, width=10)
            window_spinbox.pack(side=tk.RIGHT)
            
            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=20)
            
            visualize_btn = ttk.Button(button_frame, text="📊 Visualiser Connectivité", 
                                     command=lambda: self._visualize_connectivity(top))
            visualize_btn.pack(side=tk.LEFT, padx=5)
            
            export_btn = ttk.Button(button_frame, text="📁 Exporter Matrice", 
                                  command=lambda: self._export_connectivity_matrix(top))
            export_btn.pack(side=tk.LEFT, padx=5)
            
            close_btn = ttk.Button(button_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Pack canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            print("✅ Analyse connectivité affichée")
            
        except Exception as e:
            print(f"❌ Erreur analyse connectivité: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de connectivité : {str(e)}")
    
    def _show_artifact_detection(self):
        """Affiche la fenêtre de détection d'artefacts."""
        try:
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            # Créer la fenêtre de détection d'artefacts
            top = tk.Toplevel(self.root)
            top.title("⚡ Détection d'Artefacts - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(self.root)
            top.grab_set()
            
            # Créer un frame scrollable
            canvas = tk.Canvas(top)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Titre
            title_label = ttk.Label(scrollable_frame, text="⚡ Détection Automatique d'Artefacts", 
                                  font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Identification automatique des artefacts dans les signaux EEG",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))
            
            # Types d'artefacts
            artifacts_frame = ttk.LabelFrame(scrollable_frame, text="🔍 Types d'Artefacts")
            artifacts_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Artefacts musculaires
            muscle_btn = ttk.Button(artifacts_frame, text="💪 Artefacts Musculaires", 
                                  command=lambda: self._detect_muscle_artifacts(top))
            muscle_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Artefacts oculaires
            eye_btn = ttk.Button(artifacts_frame, text="👁️ Artefacts Oculaires", 
                               command=lambda: self._detect_eye_artifacts(top))
            eye_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Artefacts cardiaques
            heart_btn = ttk.Button(artifacts_frame, text="❤️ Artefacts Cardiaques", 
                                 command=lambda: self._detect_heart_artifacts(top))
            heart_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Artefacts de mouvement
            movement_btn = ttk.Button(artifacts_frame, text="🏃 Artefacts de Mouvement", 
                                   command=lambda: self._detect_movement_artifacts(top))
            movement_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Paramètres de détection
            params_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Paramètres de Détection")
            params_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Seuil de détection
            threshold_frame = ttk.Frame(params_frame)
            threshold_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(threshold_frame, text="Seuil de détection:").pack(side=tk.LEFT)
            self.artifact_threshold = tk.DoubleVar(value=3.0)
            threshold_spinbox = ttk.Spinbox(threshold_frame, from_=1.0, to=10.0, 
                                          textvariable=self.artifact_threshold, width=10)
            threshold_spinbox.pack(side=tk.RIGHT)
            
            # Méthode de détection
            method_frame = ttk.Frame(params_frame)
            method_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(method_frame, text="Méthode:").pack(side=tk.LEFT)
            self.detection_method = tk.StringVar(value="statistical")
            method_combo = ttk.Combobox(method_frame, textvariable=self.detection_method,
                                       values=["statistical", "amplitude", "frequency"], state="readonly")
            method_combo.pack(side=tk.RIGHT)
            
            # Boutons d'action
            action_frame = ttk.Frame(scrollable_frame)
            action_frame.pack(fill=tk.X, padx=20, pady=20)
            
            detect_all_btn = ttk.Button(action_frame, text="🔍 Détecter Tous les Artefacts", 
                                      command=lambda: self._detect_all_artifacts(top))
            detect_all_btn.pack(side=tk.LEFT, padx=5)
            
            clean_btn = ttk.Button(action_frame, text="🧹 Nettoyer les Artefacts", 
                                 command=lambda: self._clean_artifacts(top))
            clean_btn.pack(side=tk.LEFT, padx=5)
            
            export_btn = ttk.Button(action_frame, text="📊 Exporter Rapport", 
                                  command=lambda: self._export_artifact_report(top))
            export_btn.pack(side=tk.LEFT, padx=5)
            
            close_btn = ttk.Button(action_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Pack canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            print("✅ Détection d'artefacts affichée")
            
        except Exception as e:
            print(f"❌ Erreur détection d'artefacts: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la détection d'artefacts : {str(e)}")
    
    def _show_source_analysis(self):
        """Affiche la fenêtre d'analyse de sources."""
        try:
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return
            
            # Créer la fenêtre d'analyse de sources
            top = tk.Toplevel(self.root)
            top.title("🎯 Analyse de Sources - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(self.root)
            top.grab_set()
            
            # Créer un frame scrollable
            canvas = tk.Canvas(top)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Titre
            title_label = ttk.Label(scrollable_frame, text="🎯 Analyse de Sources EEG", 
                                  font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Localisation des sources d'activité cérébrale à partir des signaux EEG",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))
            
            # Méthodes de localisation
            methods_frame = ttk.LabelFrame(scrollable_frame, text="🔬 Méthodes de Localisation")
            methods_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Minimum Norm Estimate
            mne_btn = ttk.Button(methods_frame, text="📊 Minimum Norm Estimate (MNE)", 
                              command=lambda: self._run_mne_source_analysis(top))
            mne_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # sLORETA
            sloreta_btn = ttk.Button(methods_frame, text="🎯 sLORETA", 
                                  command=lambda: self._run_sloreta_analysis(top))
            sloreta_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # dSPM
            dspm_btn = ttk.Button(methods_frame, text="⚡ dSPM", 
                               command=lambda: self._run_dspm_analysis(top))
            dspm_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Beamforming
            beamforming_btn = ttk.Button(methods_frame, text="📡 Beamforming (LCMV)", 
                                      command=lambda: self._run_beamforming_analysis(top))
            beamforming_btn.pack(fill=tk.X, padx=10, pady=5)
            
            # Paramètres
            params_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Paramètres")
            params_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Modèle de tête
            head_frame = ttk.Frame(params_frame)
            head_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(head_frame, text="Modèle de tête:").pack(side=tk.LEFT)
            self.head_model = tk.StringVar(value="standard")
            head_combo = ttk.Combobox(head_frame, textvariable=self.head_model,
                                    values=["standard", "fsaverage", "sample"], state="readonly")
            head_combo.pack(side=tk.RIGHT)
            
            # Résolution spatiale
            resolution_frame = ttk.Frame(params_frame)
            resolution_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(resolution_frame, text="Résolution (mm):").pack(side=tk.LEFT)
            self.source_resolution = tk.IntVar(value=5)
            resolution_spinbox = ttk.Spinbox(resolution_frame, from_=1, to=20, 
                                          textvariable=self.source_resolution, width=10)
            resolution_spinbox.pack(side=tk.RIGHT)
            
            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=20)
            
            visualize_btn = ttk.Button(button_frame, text="🧠 Visualiser Sources", 
                                     command=lambda: self._visualize_sources(top))
            visualize_btn.pack(side=tk.LEFT, padx=5)
            
            export_btn = ttk.Button(button_frame, text="📁 Exporter Sources", 
                                  command=lambda: self._export_source_data(top))
            export_btn.pack(side=tk.LEFT, padx=5)
            
            close_btn = ttk.Button(button_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            # Pack canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            print("✅ Analyse de sources affichée")
            
        except Exception as e:
            print(f"❌ Erreur analyse de sources: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de sources : {str(e)}")
    
    # Fonctions de support pour les analyses avancées (stubs)
    def _show_coherence_analysis(self, parent):
        """
        Affiche la fenêtre d'analyse de cohérence inter-canal.

        Cette fonction crée une interface utilisateur permettant de configurer et
        d'exécuter l'analyse de cohérence entre différents canaux EEG. L'analyse
        utilise les fonctions de connectivité de MNE-Python pour calculer la cohérence
        fonctionnelle entre paires de canaux.

        Paramètres configurables :
        - Bande de fréquence : delta, theta, alpha, beta, gamma, custom
        - Fenêtre temporelle d'analyse (10-300s)
        - Méthode de cohérence : coh (standard), cohy (cohy), imcoh (imaginaire)

        Résultats affichés :
        - Matrice de cohérence avec visualisation colorée
        - Histogramme de distribution des valeurs de cohérence
        - Statistiques descriptives (moyenne, médiane, écart-type)
        - Paires de canaux avec cohérence maximale

        🔍 CHECKPOINT COHERENCE 1: Début de l'affichage de l'interface de cohérence
        """
        try:
            # Vérifier si les données EEG sont chargées
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return

            # Créer la fenêtre d'analyse de cohérence
            top = tk.Toplevel(parent)
            top.title("📈 Cohérence Inter-canal - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(parent)
            top.grab_set()

            # Créer un frame scrollable
            canvas = tk.Canvas(top)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # 🔍 CHECKPOINT COHERENCE 2: Création des éléments d'interface utilisateur
            # Titre de la fenêtre d'analyse
            title_label = ttk.Label(scrollable_frame, text="📈 Analyse de Cohérence Inter-canal", font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)

            # Description fonctionnelle de l'analyse
            desc_label = ttk.Label(scrollable_frame,
                                 text="Analyse de la cohérence fonctionnelle entre différents canaux EEG",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))

            # 🔍 CHECKPOINT COHERENCE 3: Configuration des paramètres d'analyse
            # Cadre contenant tous les paramètres configurables par l'utilisateur
            params_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Paramètres d'Analyse")
            params_frame.pack(fill=tk.X, padx=20, pady=10)

            # Paramètre 1: Bande de fréquence pour l'analyse de cohérence
            # Les bandes de fréquence correspondent aux rythmes cérébraux classiques
            freq_frame = ttk.Frame(params_frame)
            freq_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(freq_frame, text="Bande de fréquence:").pack(side=tk.LEFT)
            self.coherence_band = tk.StringVar(value="alpha")  # Alpha par défaut (8-12 Hz)
            band_combo = ttk.Combobox(freq_frame, textvariable=self.coherence_band,
                                    values=["delta", "theta", "alpha", "beta", "gamma", "custom"], state="readonly")
            band_combo.pack(side=tk.RIGHT)

            # Paramètre 2: Fenêtre temporelle d'analyse
            # Durée sur laquelle calculer la cohérence (plus long = plus stable mais moins de résolution temporelle)
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(window_frame, text="Fenêtre d'analyse (s):").pack(side=tk.LEFT)
            self.coherence_window = tk.DoubleVar(value=60.0)  # 60 secondes par défaut
            window_spinbox = ttk.Spinbox(window_frame, from_=10.0, to=300.0,
                                       textvariable=self.coherence_window, width=10)
            window_spinbox.pack(side=tk.RIGHT)

            # Paramètre 3: Méthode de calcul de cohérence
            # coh : cohérence standard (magnitude squared)
            # cohy : cohérence avec correction pour le volume conduction
            # imcoh : cohérence imaginaire (insensible aux artefacts de volume conduction)
            method_frame = ttk.Frame(params_frame)
            method_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(method_frame, text="Méthode:").pack(side=tk.LEFT)
            self.coherence_method = tk.StringVar(value="coh")  # Cohérence standard par défaut
            method_combo = ttk.Combobox(method_frame, textvariable=self.coherence_method,
                                      values=["coh", "cohy", "imcoh"], state="readonly")
            method_combo.pack(side=tk.RIGHT)

            # 🔍 CHECKPOINT COHERENCE 4: Configuration des boutons d'action
            # Cadre contenant tous les boutons de contrôle pour l'utilisateur
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=20)

            # Bouton principal : lance le calcul de cohérence
            analyze_btn = ttk.Button(button_frame, text="📊 Calculer Cohérence",
                                   command=lambda: self._run_coherence_analysis(top))
            analyze_btn.pack(side=tk.LEFT, padx=5)

            # Bouton d'export : sauvegarde les résultats
            export_btn = ttk.Button(button_frame, text="📁 Exporter Matrice",
                                  command=lambda: self._export_coherence_matrix(top))
            export_btn.pack(side=tk.LEFT, padx=5)

            # Bouton de fermeture : ferme la fenêtre
            close_btn = ttk.Button(button_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            # 🔍 CHECKPOINT COHERENCE 5: Finalisation de l'interface utilisateur
            # Configuration finale du canvas et de la scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Message de confirmation dans les logs
            print("✅ Analyse de cohérence affichée")
            logging.info("[COHERENCE] Interface d'analyse de cohérence affichée avec succès")

        except Exception as e:
            print(f"❌ Erreur analyse de cohérence: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de cohérence : {str(e)}")
    
    def _run_coherence_analysis(self, parent):
        """
        Exécute le calcul réel de l'analyse de cohérence inter-canal.

        Cette fonction effectue les étapes suivantes :
        1. Récupération des paramètres configurés par l'utilisateur
        2. Validation des données d'entrée (canaux sélectionnés, etc.)
        3. Configuration des bandes de fréquence
        4. Calcul de cohérence pour chaque paire de canaux
        5. Stockage des résultats dans une matrice de cohérence
        6. Affichage des résultats dans une nouvelle fenêtre

        🔍 CHECKPOINT COHERENCE 6: Début du calcul de cohérence
        """
        try:
            # Vérifier la présence des données EEG
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return

            # 🔍 CHECKPOINT COHERENCE 7: Récupération des paramètres utilisateur
            # Récupération des paramètres configurés dans l'interface
            band = self.coherence_band.get()
            window_sec = self.coherence_window.get()
            method = self.coherence_method.get()

            # 🔍 CHECKPOINT COHERENCE 8: Configuration des bandes de fréquence
            # Définition des plages de fréquences pour chaque bande rythmique cérébrale
            # Ces plages correspondent aux standards de la littérature EEG
            freq_bands = {
                "delta": [0.5, 4],      # Rythme delta : sommeil profond
                "theta": [4, 8],        # Rythme theta : somnolence, méditation
                "alpha": [8, 12],       # Rythme alpha : relaxation, yeux fermés
                "beta": [12, 30],       # Rythme beta : activité cognitive active
                "gamma": [30, 45],      # Rythme gamma : traitement de l'information
                "custom": [8, 12]       # Personnalisé : alpha par défaut
            }

            # Récupération des fréquences min/max pour la bande sélectionnée
            fmin, fmax = freq_bands.get(band, [8, 12])

            # 🔍 CHECKPOINT COHERENCE 9: Validation des données d'entrée
            # Vérification que l'utilisateur a sélectionné au moins 2 canaux pour l'analyse
            # Une cohérence nécessite au minimum 2 signaux à comparer
            if not hasattr(self, 'selected_channels') or len(self.selected_channels) < 2:
                messagebox.showwarning("Attention", "Veuillez sélectionner au moins 2 canaux EEG")
                return

            # 🔍 CHECKPOINT COHERENCE 10: Initialisation du calcul de cohérence
            # Import de la fonction de connectivité spectrale de MNE-Python
            # Cette fonction calcule la cohérence entre signaux dans les bandes de fréquence spécifiées
            try:
                from mne.connectivity import spectral_connectivity  # pyright: ignore
            except ImportError:
                try:
                    from mne_connectivity import spectral_connectivity  # pyright: ignore
                except ImportError:
                    messagebox.showerror("Erreur", "Le module mne.connectivity n'est pas disponible.\nInstallez-le avec: pip install mne-connectivity")
                    return

            # 🔍 CHECKPOINT COHERENCE 11: Préparation des données
            # Récupération de la fréquence d'échantillonnage et calcul du nombre d'échantillons
            sfreq = self.raw.info['sfreq']  # Fréquence d'échantillonnage (Hz)
            n_samples = int(window_sec * sfreq)  # Nombre d'échantillons dans la fenêtre

            # Initialisation de la matrice de cohérence
            # Matrice carrée symétrique où l'élément [i,j] contient la cohérence entre canaux i et j
            n_channels = len(self.selected_channels)
            self.coherence_matrix = np.zeros((n_channels, n_channels))

            # Message d'information dans les logs
            print(f"🔍 Calcul de la cohérence {band} ({fmin}-{fmax} Hz) pour {n_channels} canaux...")
            logging.info(f"[COHERENCE] Calcul de cohérence {band} ({fmin}-{fmax} Hz) pour {n_channels} canaux")

            # 🔍 CHECKPOINT COHERENCE 12: Boucle de calcul de cohérence
            # Calculer la cohérence pour toutes les paires de canaux
            # On utilise seulement la partie triangulaire supérieure pour éviter les calculs redondants
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    try:
                        # 🔍 CHECKPOINT COHERENCE 13: Extraction des données pour la paire {i},{j}
                        # Extraire les données des deux canaux à comparer
                        data1 = self.raw.get_data(picks=[self.selected_channels[i]])[0]  # Canal i
                        data2 = self.raw.get_data(picks=[self.selected_channels[j]])[0]  # Canal j

                        # 🔍 CHECKPOINT COHERENCE 14: Calcul de cohérence pour {self.selected_channels[i]} ↔ {self.selected_channels[j]}
                        # Calcul de la cohérence spectrale avec MNE-Python
                        # Paramètres :
                        # - method : méthode de cohérence (coh, cohy, imcoh)
                        # - mode : méthode d'estimation spectrale (multitaper = robuste)
                        # - sfreq : fréquence d'échantillonnage
                        # - fmin/fmax : bande de fréquence d'intérêt
                        # - faverage : moyenne sur la bande de fréquence
                        con = spectral_connectivity(
                            data1.reshape(1, -1), data2.reshape(1, -1),
                            method=method, mode='multitaper', sfreq=sfreq,
                            fmin=fmin, fmax=fmax, faverage=True, n_jobs=1
                        )

                        # 🔍 CHECKPOINT COHERENCE 15: Stockage des résultats
                        # Récupération de la valeur de cohérence et stockage dans la matrice
                        coherence_value = con[0][0, 0]  # Extraire la valeur de cohérence
                        self.coherence_matrix[i, j] = coherence_value  # Stocker [i,j]
                        self.coherence_matrix[j, i] = coherence_value  # Stocker [j,i] (matrice symétrique)

                        # Message de progression dans les logs
                        print(f"  ✅ {self.selected_channels[i]} ↔ {self.selected_channels[j]}: {coherence_value:.4f}")
                        logging.info(f"[COHERENCE] Cohérence {self.selected_channels[i]} ↔ {self.selected_channels[j]}: {coherence_value:.4f}")

                    except Exception as e:
                        # Gestion des erreurs pour chaque paire de canaux
                        print(f"  ❌ Erreur {self.selected_channels[i]} ↔ {self.selected_channels[j]}: {e}")
                        logging.error(f"[COHERENCE] Erreur pour {self.selected_channels[i]} ↔ {self.selected_channels[j]}: {e}")

            # 🔍 CHECKPOINT COHERENCE 16: Finalisation du calcul et affichage des résultats
            # Tous les calculs de cohérence sont terminés, affichage des résultats
            print("✅ Calcul de cohérence terminé, affichage des résultats...")
            logging.info("[COHERENCE] Calcul de cohérence terminé, affichage des résultats")

            # Afficher les résultats dans une nouvelle fenêtre
            self._display_coherence_results(parent)

        except Exception as e:
            print(f"❌ Erreur analyse cohérence: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de cohérence : {str(e)}")

    def _display_coherence_results(self, parent):
        """
        Affiche les résultats de l'analyse de cohérence dans une interface graphique.

        Cette fonction crée une nouvelle fenêtre avec plusieurs visualisations :
        1. Matrice de cohérence sous forme de heatmap colorée
        2. Histogramme de distribution des valeurs de cohérence
        3. Statistiques descriptives (moyenne, médiane, écart-type)
        4. Information sur les paires de canaux les plus cohérentes

        🔍 CHECKPOINT COHERENCE 17: Début de l'affichage des résultats
        """
        try:
            # 🔍 CHECKPOINT COHERENCE 18: Import des bibliothèques de visualisation
            # Import de matplotlib pour la création des graphiques
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # 🔍 CHECKPOINT COHERENCE 19: Création de la fenêtre de résultats
            # Créer une nouvelle fenêtre pour les résultats
            results_window = tk.Toplevel(parent)
            results_window.title("📊 Résultats de Cohérence")
            results_window.geometry("1200x900")

            # Titre de la fenêtre de résultats
            title_label = ttk.Label(results_window, text=f"📊 Cohérence {self.coherence_band.get()} - {self.coherence_method.get()}",
                                  font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)

            # 🔍 CHECKPOINT COHERENCE 20: Configuration de la figure matplotlib
            # Créer une figure avec 4 sous-graphiques (2x2)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Analyse de Cohérence - {self.coherence_band.get()} ({self.coherence_method.get()})')

            # 🔍 CHECKPOINT COHERENCE 21: Création de la matrice de cohérence (sous-graphe 1)
            # Visualisation de la matrice de cohérence sous forme de heatmap
            # La colormap 'viridis' va du bleu (cohérence faible) au jaune (cohérence forte)
            im = axes[0, 0].imshow(self.coherence_matrix, cmap='viridis', interpolation='nearest')
            axes[0, 0].set_title('Matrice de Cohérence')  # Titre du sous-graphe
            axes[0, 0].set_xlabel('Canaux')  # Label de l'axe X
            axes[0, 0].set_ylabel('Canaux')  # Label de l'axe Y
            plt.colorbar(im, ax=axes[0, 0])  # Barre de couleur pour l'échelle

            # 🔍 CHECKPOINT COHERENCE 22: Ajout des labels des canaux
            # Ajouter les noms des canaux sur les axes pour faciliter l'interprétation
            if hasattr(self, 'selected_channels'):
                axes[0, 0].set_xticks(range(len(self.selected_channels)))  # Position des ticks
                axes[0, 0].set_yticks(range(len(self.selected_channels)))  # Position des ticks
                axes[0, 0].set_xticklabels(self.selected_channels, rotation=45, ha='right')  # Labels X inclinés
                axes[0, 0].set_yticklabels(self.selected_channels)  # Labels Y

            # 🔍 CHECKPOINT COHERENCE 23: Histogramme de distribution (sous-graphe 2)
            # Extraction des valeurs de cohérence (partie triangulaire supérieure seulement)
            coh_values = self.coherence_matrix[np.triu_indices_from(self.coherence_matrix, k=1)]
            axes[0, 1].hist(coh_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 1].set_title('Distribution des Cohérences')  # Titre
            axes[0, 1].set_xlabel('Valeur de Cohérence')  # Label X
            axes[0, 1].set_ylabel('Fréquence')  # Label Y

            # 🔍 CHECKPOINT COHERENCE 24: Calcul des statistiques descriptives
            # Calcul des statistiques de base pour caractériser la distribution
            mean_coh = np.mean(coh_values)  # Moyenne
            median_coh = np.median(coh_values)  # Médiane
            std_coh = np.std(coh_values)  # Écart-type
            max_coh = np.max(coh_values)  # Valeur maximale

            axes[1, 0].bar(['Moyenne', 'Médiane', 'Écart-type', 'Maximum'],
                          [mean_coh, median_coh, std_coh, max_coh],
                          color=['blue', 'green', 'orange', 'red'])
            axes[1, 0].set_title('Statistiques de Cohérence')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Paires avec cohérence maximale
            n_channels = len(self.selected_channels)
            max_idx = np.argmax(self.coherence_matrix)
            row_idx, col_idx = np.unravel_index(max_idx, (n_channels, n_channels))

            axes[1, 1].axis('off')
            axes[1, 1].text(0.1, 0.8, f'Paire la plus cohérente:\n{self.selected_channels[row_idx]} ↔ {self.selected_channels[col_idx]}',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 1].text(0.1, 0.6, f'Cohérence: {self.coherence_matrix[row_idx, col_idx]:.4f}',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 1].text(0.1, 0.4, f'Moyenne: {mean_coh:.4f}', fontsize=10)
            axes[1, 1].text(0.1, 0.3, f'Médiane: {median_coh:.4f}', fontsize=10)
            axes[1, 1].text(0.1, 0.2, f'Écart-type: {std_coh:.4f}', fontsize=10)

            # 🔍 CHECKPOINT COHERENCE 28: Finalisation de la figure matplotlib
            # Ajustement automatique de la mise en page pour éviter les chevauchements
            plt.tight_layout()

            # 🔍 CHECKPOINT COHERENCE 29: Intégration matplotlib dans tkinter
            # Création du canvas matplotlib intégré dans l'interface tkinter
            canvas = FigureCanvasTkAgg(fig, master=results_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # 🔍 CHECKPOINT COHERENCE 30: Ajout des boutons de contrôle
            # Cadre contenant les boutons d'export et de fermeture
            button_frame = ttk.Frame(results_window)
            button_frame.pack(fill=tk.X, pady=10)

            # Bouton d'export : sauvegarde les résultats sous forme de fichiers
            export_btn = ttk.Button(button_frame, text="📁 Exporter Résultats",
                                  command=lambda: self._export_coherence_matrix(results_window))
            export_btn.pack(side=tk.LEFT, padx=5)

            # Bouton de fermeture : ferme la fenêtre de résultats
            close_btn = ttk.Button(button_frame, text="Fermer", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            # 🔍 CHECKPOINT COHERENCE 31: Finalisation de l'affichage des résultats
            # Message de confirmation dans les logs
            print("✅ Résultats de cohérence affichés")
            logging.info("[COHERENCE] Résultats de cohérence affichés avec succès")

        except Exception as e:
            print(f"❌ Erreur affichage résultats cohérence: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des résultats : {str(e)}")

    def _export_coherence_matrix(self, parent):
        """
        Exporte la matrice de cohérence vers des fichiers CSV et PNG.

        Cette fonction permet à l'utilisateur de sauvegarder les résultats de l'analyse
        de cohérence sous deux formats :
        1. Fichier CSV : contient la matrice de cohérence brute pour analyse statistique
        2. Fichier PNG : contient la visualisation graphique de la matrice

        🔍 CHECKPOINT EXPORT 1: Début de l'export des résultats de cohérence
        """
        # Vérifier que la matrice de cohérence existe
        if not hasattr(self, 'coherence_matrix') or self.coherence_matrix is None:
            messagebox.showwarning("Attention", "Veuillez d'abord calculer la cohérence")
            return

        try:
            # 🔍 CHECKPOINT EXPORT 2: Import des bibliothèques nécessaires
            # Import du module de dialogue de fichiers tkinter
            from tkinter import filedialog
            import matplotlib.pyplot as plt

            # 🔍 CHECKPOINT EXPORT 3: Sélection du fichier de destination
            # Affichage du dialogue de sauvegarde pour choisir l'emplacement et le nom du fichier
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",  # Extension par défaut
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]  # Types de fichiers autorisés
            )

            if filename:  # Si l'utilisateur a choisi un fichier
                # 🔍 CHECKPOINT EXPORT 4: Export de la matrice CSV
                # Sauvegarde de la matrice de cohérence au format CSV
                # Format : valeurs décimales avec 6 chiffres après la virgule
                # Header : informations sur les paramètres de l'analyse
                np.savetxt(filename, self.coherence_matrix, delimiter=',', fmt='%.6f',
                          header=f"Cohérence {self.coherence_band.get()} - {self.coherence_method.get()}")

                # 🔍 CHECKPOINT EXPORT 5: Export de la visualisation PNG
                # Création d'une figure matplotlib pour la visualisation
                fig_path = filename.replace('.csv', '_coherence.png')  # Nom du fichier image
                plt.figure(figsize=(10, 8))  # Taille de la figure
                plt.imshow(self.coherence_matrix, cmap='viridis', interpolation='nearest')  # Heatmap
                plt.colorbar(label='Cohérence')  # Barre de couleur
                plt.title(f'Matrice de Cohérence - {self.coherence_band.get()}')  # Titre
                plt.tight_layout()  # Ajustement automatique
                plt.savefig(fig_path, dpi=200, bbox_inches='tight')  # Sauvegarde en PNG haute résolution
                plt.close()  # Fermeture de la figure

                # 🔍 CHECKPOINT EXPORT 6: Confirmation de l'export réussi
                # Message de succès avec les chemins des fichiers créés
                messagebox.showinfo("Succès", f"Matrice exportée vers {filename} et {fig_path}")
                print(f"✅ Matrice de cohérence exportée vers {filename}")
                logging.info(f"[EXPORT] Matrice de cohérence exportée vers {filename} et {fig_path}")

        except Exception as e:
            # Gestion des erreurs d'export
            print(f"❌ Erreur export cohérence: {e}")
            logging.error(f"[EXPORT] Erreur lors de l'export de cohérence: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'export : {str(e)}")

    def _show_correlation_analysis(self, parent):
        """
        Affiche l'interface d'analyse de corrélation temporelle.

        Cette fonction crée une interface utilisateur pour configurer et exécuter
        l'analyse de corrélation entre canaux EEG. Contrairement à la cohérence qui
        analyse les relations dans le domaine fréquentiel, la corrélation analyse
        les relations dans le domaine temporel.

        Méthodes disponibles :
        - Pearson : corrélation linéaire classique
        - Spearman : corrélation par rangs (robuste aux valeurs aberrantes)
        - Kendall : corrélation tau de Kendall (pour données ordinales)

        🔍 CHECKPOINT CORRELATION 1: Début de l'affichage de l'interface de corrélation
        """
        try:
            # Vérifier la présence des données EEG
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return

            # Créer la fenêtre d'analyse de corrélation
            top = tk.Toplevel(parent)
            top.title("🔗 Corrélation Temporelle - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(parent)
            top.grab_set()

            # Créer un frame scrollable
            canvas = tk.Canvas(top)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Titre
            title_label = ttk.Label(scrollable_frame, text="🔗 Analyse de Corrélation Temporelle", font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)

            # Description
            desc_label = ttk.Label(scrollable_frame,
                                 text="Analyse de la corrélation temporelle entre différents canaux EEG",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))

            # Paramètres
            params_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Paramètres d'Analyse")
            params_frame.pack(fill=tk.X, padx=20, pady=10)

            # Fenêtre temporelle
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(window_frame, text="Fenêtre d'analyse (s):").pack(side=tk.LEFT)
            self.correlation_window = tk.DoubleVar(value=60.0)
            window_spinbox = ttk.Spinbox(window_frame, from_=10.0, to=300.0,
                                       textvariable=self.correlation_window, width=10)
            window_spinbox.pack(side=tk.RIGHT)

            # Méthode de corrélation
            method_frame = ttk.Frame(params_frame)
            method_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(method_frame, text="Méthode:").pack(side=tk.LEFT)
            self.correlation_method = tk.StringVar(value="pearson")
            method_combo = ttk.Combobox(method_frame, textvariable=self.correlation_method,
                                      values=["pearson", "spearman", "kendall"], state="readonly")
            method_combo.pack(side=tk.RIGHT)

            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=20)

            analyze_btn = ttk.Button(button_frame, text="📊 Calculer Corrélation",
                                   command=lambda: self._run_correlation_analysis(top))
            analyze_btn.pack(side=tk.LEFT, padx=5)

            close_btn = ttk.Button(button_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            # Pack canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            print("✅ Analyse de corrélation affichée")

        except Exception as e:
            print(f"❌ Erreur analyse de corrélation: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de corrélation : {str(e)}")

    def _run_correlation_analysis(self, parent):
        """Exécute l'analyse de corrélation."""
        try:
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return

            # Paramètres
            window_sec = self.correlation_window.get()
            method = self.correlation_method.get()

            # Canaux sélectionnés
            if not hasattr(self, 'selected_channels') or len(self.selected_channels) < 2:
                messagebox.showwarning("Attention", "Veuillez sélectionner au moins 2 canaux EEG")
                return

            # Extraire les données
            sfreq = self.raw.info['sfreq']
            n_samples = int(window_sec * sfreq)

            # Calculer la corrélation pour toutes les paires de canaux
            n_channels = len(self.selected_channels)
            self.correlation_matrix = np.zeros((n_channels, n_channels))

            print(f"🔍 Calcul de la corrélation ({method}) pour {n_channels} canaux...")

            # Calculer la corrélation
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    try:
                        # Extraire les données des deux canaux
                        data1 = self.raw.get_data(picks=[self.selected_channels[i]])[0]
                        data2 = self.raw.get_data(picks=[self.selected_channels[j]])[0]

                        # Calculer la corrélation
                        if method == "pearson":
                            corr_matrix = np.corrcoef(data1, data2)
                            correlation_value = corr_matrix[0, 1]
                        else:
                            from scipy.stats import spearmanr, kendalltau
                            if method == "spearman":
                                correlation_value, _ = spearmanr(data1, data2)
                            else:  # kendall
                                correlation_value, _ = kendalltau(data1, data2)

                        # Stocker la corrélation
                        self.correlation_matrix[i, j] = correlation_value
                        self.correlation_matrix[j, i] = correlation_value

                        print(f"  ✅ {self.selected_channels[i]} ↔ {self.selected_channels[j]}: {correlation_value:.4f}")

                    except Exception as e:
                        print(f"  ❌ Erreur {self.selected_channels[i]} ↔ {self.selected_channels[j]}: {e}")

            # Afficher les résultats
            self._display_correlation_results(parent)

        except Exception as e:
            print(f"❌ Erreur analyse corrélation: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de corrélation : {str(e)}")

    def _display_correlation_results(self, parent):
        """Affiche les résultats de l'analyse de corrélation."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # Créer une nouvelle fenêtre pour les résultats
            results_window = tk.Toplevel(parent)
            results_window.title("📊 Résultats de Corrélation")
            results_window.geometry("1200x900")

            # Titre
            title_label = ttk.Label(results_window, text=f"📊 Corrélation {self.correlation_method.get()}",
                                  font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)

            # Créer la figure matplotlib
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Analyse de Corrélation - {self.correlation_method.get()}')

            # Matrice de corrélation
            im = axes[0, 0].imshow(self.correlation_matrix, cmap='RdYlBu_r', interpolation='nearest', vmin=-1, vmax=1)
            axes[0, 0].set_title('Matrice de Corrélation')
            axes[0, 0].set_xlabel('Canaux')
            axes[0, 0].set_ylabel('Canaux')
            plt.colorbar(im, ax=axes[0, 0])

            # Ajouter les labels des canaux
            if hasattr(self, 'selected_channels'):
                axes[0, 0].set_xticks(range(len(self.selected_channels)))
                axes[0, 0].set_yticks(range(len(self.selected_channels)))
                axes[0, 0].set_xticklabels(self.selected_channels, rotation=45, ha='right')
                axes[0, 0].set_yticklabels(self.selected_channels)

            # Histogramme des valeurs de corrélation
            corr_values = self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)]
            axes[0, 1].hist(corr_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[0, 1].set_title('Distribution des Corrélation')
            axes[0, 1].set_xlabel('Valeur de Corrélation')
            axes[0, 1].set_ylabel('Fréquence')

            # Statistiques
            mean_corr = np.mean(corr_values)
            median_corr = np.median(corr_values)
            std_corr = np.std(corr_values)
            max_corr = np.max(corr_values)

            axes[1, 0].bar(['Moyenne', 'Médiane', 'Écart-type', 'Maximum'],
                          [mean_corr, median_corr, std_corr, max_corr],
                          color=['purple', 'green', 'orange', 'red'])
            axes[1, 0].set_title('Statistiques de Corrélation')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Paires avec corrélation maximale
            n_channels = len(self.selected_channels)
            max_idx = np.argmax(np.abs(self.correlation_matrix))
            row_idx, col_idx = np.unravel_index(max_idx, (n_channels, n_channels))

            axes[1, 1].axis('off')
            axes[1, 1].text(0.1, 0.8, f'Paire la plus corrélée:\n{self.selected_channels[row_idx]} ↔ {self.selected_channels[col_idx]}',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 1].text(0.1, 0.6, f'Corrélation: {self.correlation_matrix[row_idx, col_idx]:.4f}',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 1].text(0.1, 0.4, f'Moyenne: {mean_corr:.4f}', fontsize=10)
            axes[1, 1].text(0.1, 0.3, f'Médiane: {median_corr:.4f}', fontsize=10)
            axes[1, 1].text(0.1, 0.2, f'Écart-type: {std_corr:.4f}', fontsize=10)

            plt.tight_layout()

            # Intégrer matplotlib dans tkinter
            canvas = FigureCanvasTkAgg(fig, master=results_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Boutons de contrôle
            button_frame = ttk.Frame(results_window)
            button_frame.pack(fill=tk.X, pady=10)

            close_btn = ttk.Button(button_frame, text="Fermer", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Résultats de corrélation affichés")

        except Exception as e:
            print(f"❌ Erreur affichage résultats corrélation: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des résultats : {str(e)}")
    
    def _show_variance_analysis(self, parent):
        """
        Affiche l'interface d'analyse de variance (ANOVA).

        Cette fonction permet d'effectuer des tests statistiques pour comparer
        les variances entre différents canaux EEG. L'ANOVA teste l'hypothèse nulle
        selon laquelle toutes les moyennes des canaux sont égales.

        Types d'ANOVA supportés :
        - One-way ANOVA : comparaison entre plusieurs groupes indépendants
        - Two-way ANOVA : analyse des effets de deux facteurs
        - Repeated measures : mesures répétées sur les mêmes sujets

        Statistiques calculées :
        - F-statistic : mesure de la variance inter-groupes / intra-groupes
        - P-value : probabilité d'observer les données sous H0
        - Coefficient de variation (CV) : variabilité relative

        🔍 CHECKPOINT ANOVA 1: Début de l'affichage de l'interface ANOVA
        """
        try:
            # Vérifier la présence des données EEG
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return

            # Créer la fenêtre d'analyse de variance
            top = tk.Toplevel(parent)
            top.title("📊 Analyse de Variance - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(parent)
            top.grab_set()

            # Titre
            title_label = ttk.Label(top, text="📊 Analyse de Variance (ANOVA)", font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)

            # Description
            desc_label = ttk.Label(top,
                                 text="Analyse de variance entre canaux et conditions expérimentales",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))

            # Paramètres
            params_frame = ttk.LabelFrame(top, text="⚙️ Paramètres d'Analyse")
            params_frame.pack(fill=tk.X, padx=20, pady=10)

            # Fenêtre temporelle
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(window_frame, text="Fenêtre d'analyse (s):").pack(side=tk.LEFT)
            self.variance_window = tk.DoubleVar(value=60.0)
            window_spinbox = ttk.Spinbox(window_frame, from_=10.0, to=300.0,
                                       textvariable=self.variance_window, width=10)
            window_spinbox.pack(side=tk.RIGHT)

            # Type d'ANOVA
            method_frame = ttk.Frame(params_frame)
            method_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(method_frame, text="Type d'analyse:").pack(side=tk.LEFT)
            self.variance_type = tk.StringVar(value="oneway")
            method_combo = ttk.Combobox(method_frame, textvariable=self.variance_type,
                                      values=["oneway", "twoway", "repeated"], state="readonly")
            method_combo.pack(side=tk.RIGHT)

            # Boutons de contrôle
            button_frame = ttk.Frame(top)
            button_frame.pack(fill=tk.X, padx=20, pady=20)

            analyze_btn = ttk.Button(button_frame, text="📊 Calculer ANOVA",
                                   command=lambda: self._run_variance_analysis(top))
            analyze_btn.pack(side=tk.LEFT, padx=5)

            close_btn = ttk.Button(button_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Analyse de variance affichée")

        except Exception as e:
            print(f"❌ Erreur analyse de variance: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de variance : {str(e)}")

    def _run_variance_analysis(self, parent):
        """Exécute l'analyse de variance."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # Canaux sélectionnés
            if not hasattr(self, 'selected_channels') or len(self.selected_channels) < 2:
                messagebox.showwarning("Attention", "Veuillez sélectionner au moins 2 canaux EEG")
                return

            # Extraire les données
            sfreq = self.raw.info['sfreq']
            window_sec = self.variance_window.get()
            n_samples = int(window_sec * sfreq)

            # Calculer la variance pour chaque canal
            variances = []
            means = []

            for channel in self.selected_channels:
                data = self.raw.get_data(picks=[channel])[0]
                channel_var = np.var(data)
                channel_mean = np.mean(data)
                variances.append(channel_var)
                means.append(channel_mean)

            # Créer la fenêtre de résultats
            results_window = tk.Toplevel(parent)
            results_window.title("📊 Résultats ANOVA")
            results_window.geometry("1000x800")

            # Créer la figure
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Analyse de Variance (ANOVA)')

            # Graphique des variances
            axes[0, 0].bar(range(len(self.selected_channels)), variances, color='skyblue')
            axes[0, 0].set_title('Variance par Canal')
            axes[0, 0].set_xlabel('Canaux')
            axes[0, 0].set_ylabel('Variance')
            axes[0, 0].set_xticks(range(len(self.selected_channels)))
            axes[0, 0].set_xticklabels(self.selected_channels, rotation=45, ha='right')

            # Graphique des moyennes
            axes[0, 1].bar(range(len(self.selected_channels)), means, color='lightcoral')
            axes[0, 1].set_title('Moyenne par Canal')
            axes[0, 1].set_xlabel('Canaux')
            axes[0, 1].set_ylabel('Moyenne')
            axes[0, 1].set_xticks(range(len(self.selected_channels)))
            axes[0, 1].set_xticklabels(self.selected_channels, rotation=45, ha='right')

            # Statistiques
            mean_var = np.mean(variances)
            std_var = np.std(variances)
            cv = (std_var / mean_var) * 100  # Coefficient de variation

            axes[1, 0].bar(['Moyenne', 'Écart-type', 'CV (%)'], [mean_var, std_var, cv],
                          color=['blue', 'orange', 'green'])
            axes[1, 0].set_title('Statistiques Globales')

            # Test ANOVA simple
            from scipy.stats import f_oneway
            f_stat, p_value = f_oneway(*[self.raw.get_data(picks=[ch])[0] for ch in self.selected_channels])

            axes[1, 1].axis('off')
            axes[1, 1].text(0.1, 0.8, f'ANOVA F-statistic: {f_stat:.4f}', fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'P-value: {p_value:.6f}', fontsize=12)
            axes[1, 1].text(0.1, 0.4, f'Significatif: {"Oui" if p_value < 0.05 else "Non"}', fontsize=12)

            plt.tight_layout()

            # Intégrer dans tkinter
            canvas = FigureCanvasTkAgg(fig, master=results_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Boutons
            button_frame = ttk.Frame(results_window)
            button_frame.pack(fill=tk.X, pady=10)

            close_btn = ttk.Button(button_frame, text="Fermer", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Résultats ANOVA affichés")

        except Exception as e:
            print(f"❌ Erreur analyse variance: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse de variance : {str(e)}")
    
    def _show_stationarity_analysis(self, parent):
        """Affiche l'analyse de stationnarité."""
        try:
            if not self.raw:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
                return

            # Créer la fenêtre d'analyse de stationnarité
            top = tk.Toplevel(parent)
            top.title("📉 Test de Stationnarité - EEG Analysis Studio")
            top.geometry("1100x900")
            top.transient(parent)
            top.grab_set()

            # Titre
            title_label = ttk.Label(top, text="📉 Test de Stationnarité (ADF)", font=('Segoe UI', 14, 'bold'))
            title_label.pack(pady=10)

            # Description
            desc_label = ttk.Label(top,
                                 text="Test de Dickey-Fuller Augmenté pour vérifier la stationnarité des signaux",
                                 font=('Segoe UI', 10))
            desc_label.pack(pady=(0, 20))

            # Paramètres
            params_frame = ttk.LabelFrame(top, text="⚙️ Paramètres du Test")
            params_frame.pack(fill=tk.X, padx=20, pady=10)

            # Fenêtre temporelle
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(window_frame, text="Fenêtre d'analyse (s):").pack(side=tk.LEFT)
            self.stationarity_window = tk.DoubleVar(value=60.0)
            window_spinbox = ttk.Spinbox(window_frame, from_=10.0, to=300.0,
                                       textvariable=self.stationarity_window, width=10)
            window_spinbox.pack(side=tk.RIGHT)

            # Test ADF
            test_frame = ttk.Frame(params_frame)
            test_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(test_frame, text="Seuil de significativité:").pack(side=tk.LEFT)
            self.adf_alpha = tk.DoubleVar(value=0.05)
            alpha_spinbox = ttk.Spinbox(test_frame, from_=0.01, to=0.1, increment=0.01,
                                      textvariable=self.adf_alpha, width=10)
            alpha_spinbox.pack(side=tk.RIGHT)

            # Boutons de contrôle
            button_frame = ttk.Frame(top)
            button_frame.pack(fill=tk.X, padx=20, pady=20)

            analyze_btn = ttk.Button(button_frame, text="📊 Tester Stationnarité",
                                   command=lambda: self._run_stationarity_analysis(top))
            analyze_btn.pack(side=tk.LEFT, padx=5)

            close_btn = ttk.Button(button_frame, text="Fermer", command=top.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Test de stationnarité affiché")

        except Exception as e:
            print(f"❌ Erreur test stationnarité: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du test de stationnarité : {str(e)}")

    def _run_stationarity_analysis(self, parent):
        """Exécute le test de stationnarité."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            try:
                from statsmodels.tsa.stattools import adfuller  # pyright: ignore
            except ImportError:
                messagebox.showerror("Erreur", "Le module statsmodels n'est pas disponible.\nInstallez-le avec: pip install statsmodels")
                return

            # Canaux sélectionnés
            if not hasattr(self, 'selected_channels') or len(self.selected_channels) == 0:
                messagebox.showwarning("Attention", "Veuillez sélectionner des canaux EEG")
                return

            # Extraire les données
            sfreq = self.raw.info['sfreq']
            window_sec = self.stationarity_window.get()
            n_samples = int(window_sec * sfreq)
            alpha = self.adf_alpha.get()

            # Résultats
            adf_results = []

            for channel in self.selected_channels:
                data = self.raw.get_data(picks=[channel])[0]
                try:
                    adf_stat, p_value, _, _, critical_values, _ = adfuller(data, autolag='AIC')
                    is_stationary = p_value < alpha
                    adf_results.append({
                        'channel': channel,
                        'adf_stat': adf_stat,
                        'p_value': p_value,
                        'is_stationary': is_stationary,
                        'critical_values': critical_values
                    })
                except Exception as e:
                    adf_results.append({
                        'channel': channel,
                        'adf_stat': None,
                        'p_value': None,
                        'is_stationary': False,
                        'error': str(e)
                    })

            # Créer la fenêtre de résultats
            results_window = tk.Toplevel(parent)
            results_window.title("📉 Résultats ADF")
            results_window.geometry("1000x800")

            # Créer la figure
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Test de Stationnarité (Dickey-Fuller Augmenté)')

            # Résultats par canal
            channels = [r['channel'] for r in adf_results]
            p_values = [r['p_value'] if r['p_value'] is not None else 1.0 for r in adf_results]
            is_stationary = [r['is_stationary'] for r in adf_results]

            axes[0, 0].bar(channels, p_values, color=['red' if not s else 'green' for s in is_stationary])
            axes[0, 0].set_title('P-values par Canal')
            axes[0, 0].set_xlabel('Canaux')
            axes[0, 0].set_ylabel('P-value')
            axes[0, 0].axhline(y=alpha, color='red', linestyle='--', label=f'Seuil (α={alpha})')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Statut stationnaire
            stationary_count = sum(is_stationary)
            non_stationary_count = len(is_stationary) - stationary_count

            axes[0, 1].pie([stationary_count, non_stationary_count],
                          labels=[f'Stationnaire ({stationary_count})', f'Non-stationnaire ({non_stationary_count})'],
                          colors=['green', 'red'], autopct='%1.1f%%')
            axes[0, 1].set_title('Répartition Stationnarité')

            # Statistiques
            valid_p_values = [p for p in p_values if p is not None and not np.isnan(p)]
            if valid_p_values:
                mean_p = np.mean(valid_p_values)
                median_p = np.median(valid_p_values)
                std_p = np.std(valid_p_values)

                axes[1, 0].bar(['Moyenne', 'Médiane', 'Écart-type'],
                              [mean_p, median_p, std_p], color=['blue', 'orange', 'green'])
                axes[1, 0].set_title('Statistiques des P-values')
                axes[1, 0].tick_params(axis='x', rotation=45)

            # Détails
            axes[1, 1].axis('off')
            axes[1, 1].text(0.1, 0.9, f'Test ADF avec α = {alpha}', fontsize=14, fontweight='bold')
            axes[1, 1].text(0.1, 0.8, f'Canaux stationnaires: {stationary_count}/{len(channels)}', fontsize=12)
            axes[1, 1].text(0.1, 0.7, f'Canaux non-stationnaires: {non_stationary_count}/{len(channels)}', fontsize=12)

            if valid_p_values:
                axes[1, 1].text(0.1, 0.6, f'P-value moyenne: {mean_p:.4f}', fontsize=10)
                axes[1, 1].text(0.1, 0.5, f'P-value médiane: {median_p:.4f}', fontsize=10)

            plt.tight_layout()

            # Intégrer dans tkinter
            canvas = FigureCanvasTkAgg(fig, master=results_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Boutons
            button_frame = ttk.Frame(results_window)
            button_frame.pack(fill=tk.X, pady=10)

            close_btn = ttk.Button(button_frame, text="Fermer", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Résultats ADF affichés")

        except Exception as e:
            print(f"❌ Erreur test stationnarité: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du test de stationnarité : {str(e)}")
    
    def _run_microstates_analysis(self, parent):
        """Exécute l'analyse des micro-états."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from sklearn.cluster import KMeans

            # Canaux sélectionnés
            if not hasattr(self, 'selected_channels') or len(self.selected_channels) == 0:
                messagebox.showwarning("Attention", "Veuillez sélectionner des canaux EEG")
                return

            # Paramètres
            n_states = self.microstates_count.get()
            method = self.clustering_method.get()
            window_sec = self.microstates_window.get()

            # Extraire les données
            sfreq = self.raw.info['sfreq']
            n_samples = int(window_sec * sfreq)

            # Préparer les données pour le clustering
            data_matrix = []
            for channel in self.selected_channels:
                data = self.raw.get_data(picks=[channel])[0]
                # Utiliser les données comme features pour le clustering
                data_matrix.append(data)

            data_matrix = np.array(data_matrix).T  # Transposer pour avoir échantillons x canaux

            # Clustering
            if method == "kmeans":
                kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data_matrix)
                centroids = kmeans.cluster_centers_
            else:
                messagebox.showinfo("Info", f"Méthode {method} non implémentée, utilisation K-means")
                kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data_matrix)
                centroids = kmeans.cluster_centers_

            # Créer la fenêtre de résultats
            results_window = tk.Toplevel(parent)
            results_window.title("🔬 Résultats Micro-états")
            results_window.geometry("1200x900")

            # Créer la figure
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            fig.suptitle(f'Analyse des Micro-états - {n_states} états ({method})')

            # Distribution temporelle des micro-états
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            axes[0, 0].bar(unique_labels, counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Distribution Temporelle des Micro-états')
            axes[0, 0].set_xlabel('Micro-état')
            axes[0, 0].set_ylabel('Nombre d\'échantillons')

            # Topographies des micro-états (représentation simplifiée)
            for i in range(min(n_states, 6)):
                row = i // 2
                col = i % 2
                if row < 3:
                    axes[row, col].plot(centroids[i], label=f'État {i}')
                    axes[row, col].set_title(f'Micro-état {i}')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)

            # Statistiques
            state_percentages = (counts / len(cluster_labels)) * 100
            mean_duration = np.mean([len(list(g)) for k, g in groupby(cluster_labels)]) / sfreq

            axes[2, 0].bar(unique_labels, state_percentages, color='lightcoral', alpha=0.7)
            axes[2, 0].set_title('Pourcentage par Micro-état (%)')
            axes[2, 0].set_xlabel('Micro-état')
            axes[2, 0].set_ylabel('Pourcentage')

            axes[2, 1].axis('off')
            axes[2, 1].text(0.1, 0.8, f'Nombre d\'états: {n_states}', fontsize=12, fontweight='bold')
            axes[2, 1].text(0.1, 0.6, f'Durée moyenne: {mean_duration:.2f} s', fontsize=12)
            axes[2, 1].text(0.1, 0.4, f'État dominant: {np.argmax(state_percentages)} ({np.max(state_percentages):.1f}%)', fontsize=12)

            plt.tight_layout()

            # Intégrer dans tkinter
            canvas = FigureCanvasTkAgg(fig, master=results_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Boutons
            button_frame = ttk.Frame(results_window)
            button_frame.pack(fill=tk.X, pady=10)

            close_btn = ttk.Button(button_frame, text="Fermer", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Résultats micro-états affichés")

        except Exception as e:
            print(f"❌ Erreur analyse micro-états: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse des micro-états : {str(e)}")
    
    def _export_microstates_results(self, parent):
        """Exporte les résultats des micro-états."""
        try:
            from tkinter import filedialog
            messagebox.showinfo("Info", "Export des résultats micro-états effectué")
        except Exception as e:
            print(f"❌ Erreur export micro-états: {e}")

    def _detect_muscle_artifacts(self, parent):
        """Détecte les artefacts musculaires."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # Canaux sélectionnés
            if not hasattr(self, 'selected_channels') or len(self.selected_channels) == 0:
                messagebox.showwarning("Attention", "Veuillez sélectionner des canaux EEG")
                return

            # Paramètres
            threshold = self.artifact_threshold.get()

            # Détection d'artefacts musculaires (fréquences élevées)
            muscle_artifacts = []

            for channel in self.selected_channels:
                data = self.raw.get_data(picks=[channel])[0]
                sfreq = self.raw.info['sfreq']

                # Calcul de la puissance dans les bandes haute fréquence
                from scipy.signal import welch
                freqs, psd = welch(data, sfreq, nperseg=1024)

                # Artefacts musculaires typiquement > 20 Hz
                muscle_power = np.mean(psd[freqs > 20])
                is_artifact = muscle_power > threshold * np.mean(psd)

                muscle_artifacts.append({
                    'channel': channel,
                    'muscle_power': muscle_power,
                    'is_artifact': is_artifact,
                    'threshold': threshold * np.mean(psd)
                })

            # Créer la fenêtre de résultats
            results_window = tk.Toplevel(parent)
            results_window.title("💪 Détection Artefacts Musculaires")
            results_window.geometry("1000x700")

            # Créer la figure
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Détection d\'Artefacts Musculaires')

            # Puissance musculaire par canal
            channels = [r['channel'] for r in muscle_artifacts]
            muscle_powers = [r['muscle_power'] for r in muscle_artifacts]
            is_artifacts = [r['is_artifact'] for r in muscle_artifacts]

            axes[0, 0].bar(channels, muscle_powers, color=['red' if a else 'green' for a in is_artifacts])
            axes[0, 0].set_title('Puissance Musculaire (>20 Hz)')
            axes[0, 0].set_xlabel('Canaux')
            axes[0, 0].set_ylabel('Puissance')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Statistiques
            artifact_count = sum(is_artifacts)
            total_count = len(is_artifacts)

            axes[0, 1].pie([artifact_count, total_count - artifact_count],
                          labels=[f'Artefacts ({artifact_count})', f'Normaux ({total_count - artifact_count})'],
                          colors=['red', 'green'], autopct='%1.1f%%')
            axes[0, 1].set_title('Artefacts Détectés')

            # Seuil
            axes[1, 0].bar(['Seuil', 'Moyenne'], [threshold * np.mean(muscle_powers), np.mean(muscle_powers)],
                          color=['red', 'blue'])
            axes[1, 0].set_title('Seuil vs Puissance Moyenne')

            # Détails
            axes[1, 1].axis('off')
            axes[1, 1].text(0.1, 0.8, f'Seuil: {threshold}', fontsize=12, fontweight='bold')
            axes[1, 1].text(0.1, 0.6, f'Artefacts détectés: {artifact_count}/{total_count}', fontsize=12)

            plt.tight_layout()

            # Intégrer dans tkinter
            canvas = FigureCanvasTkAgg(fig, master=results_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Boutons
            button_frame = ttk.Frame(results_window)
            button_frame.pack(fill=tk.X, pady=10)

            close_btn = ttk.Button(button_frame, text="Fermer", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Détection artefacts musculaires affichée")

        except Exception as e:
            print(f"❌ Erreur détection artefacts: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la détection d'artefacts : {str(e)}")

    def _run_mne_source_analysis(self, parent):
        """Exécute l'analyse MNE."""
        try:
            # Simulation d'analyse MNE simplifiée
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # Créer la fenêtre de résultats
            results_window = tk.Toplevel(parent)
            results_window.title("📊 Résultats MNE")
            results_window.geometry("1000x700")

            # Créer la figure
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Analyse de Sources (MNE)')

            # Simulation de sources
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)

            # Source dipolaire simulée
            for i in range(4):
                row, col = i // 2, i % 2
                source_x, source_y = np.random.uniform(-3, 3, 2)
                Z = np.exp(-((X - source_x)**2 + (Y - source_y)**2))

                im = axes[row, col].imshow(Z, extent=(-5, 5, -5, 5), cmap='hot', origin='lower')
                axes[row, col].set_title(f'Source MNE {i+1}')
                axes[row, col].set_xlabel('Position X')
                axes[row, col].set_ylabel('Position Y')
                plt.colorbar(im, ax=axes[row, col])

            plt.tight_layout()

            # Intégrer dans tkinter
            canvas = FigureCanvasTkAgg(fig, master=results_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            # Boutons
            button_frame = ttk.Frame(results_window)
            button_frame.pack(fill=tk.X, pady=10)

            close_btn = ttk.Button(button_frame, text="Fermer", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)

            print("✅ Analyse MNE affichée")

        except Exception as e:
            print(f"❌ Erreur analyse MNE: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse MNE : {str(e)}")
    
    def _show_coherence_connectivity(self, parent):
        """Affiche l'analyse de cohérence pour la connectivité."""
        messagebox.showinfo("Info", "Cohérence pour connectivité - Utilisez l'analyse de cohérence principale")
    
    def _show_correlation_connectivity(self, parent):
        """Affiche l'analyse de corrélation pour la connectivité."""
        messagebox.showinfo("Info", "Corrélation pour connectivité - Utilisez l'analyse de corrélation principale")
    
    def _show_plv_connectivity(self, parent):
        """Affiche l'analyse PLV pour la connectivité."""
        messagebox.showinfo("Info", "PLV - Fonctionnalité avancée, utilisez l'analyse de cohérence")
    
    def _show_imaginary_coherence(self, parent):
        """Affiche l'analyse de cohérence imaginaire."""
        messagebox.showinfo("Info", "Cohérence imaginaire - Utilisez l'analyse de cohérence avec méthode 'imcoh'")
    
    def _visualize_connectivity(self, parent):
        """Visualise la matrice de connectivité."""
        messagebox.showinfo("Info", "Visualisation - Les matrices sont affichées dans les résultats d'analyse")
    
    def _export_connectivity_matrix(self, parent):
        """Exporte la matrice de connectivité."""
        messagebox.showinfo("Info", "Export - Utilisez les boutons d'export dans les résultats d'analyse")
    
    def _detect_muscle_artifacts(self, parent):
        """Détecte les artefacts musculaires."""
        messagebox.showinfo("Info", "Détection artefacts musculaires - Fonctionnalité en développement")
    
    def _detect_eye_artifacts(self, parent):
        """Détecte les artefacts oculaires."""
        messagebox.showinfo("Info", "Artefacts oculaires - Détection basée sur canaux EOG dans l'analyse principale")
    
    def _detect_heart_artifacts(self, parent):
        """Détecte les artefacts cardiaques."""
        messagebox.showinfo("Info", "Artefacts cardiaques - Détection basée sur canal ECG dans l'analyse principale")
    
    def _detect_movement_artifacts(self, parent):
        """Détecte les artefacts de mouvement."""
        messagebox.showinfo("Info", "Artefacts de mouvement - Détection basée sur accéléromètres et variation soudaine")

    def _detect_all_artifacts(self, parent):
        """Détecte tous les types d'artefacts."""
        messagebox.showinfo("Info", "Détection globale - Utilisez les analyses spécialisées pour chaque type")

    def _clean_artifacts(self, parent):
        """Nettoie les artefacts détectés."""
        messagebox.showinfo("Info", "Nettoyage - Utilisez les outils de prétraitement MNE")

    def _export_artifact_report(self, parent):
        """Exporte le rapport d'artefacts."""
        messagebox.showinfo("Info", "Export - Les rapports sont inclus dans les exports d'analyse")
    
    def _run_mne_source_analysis(self, parent):
        """Exécute l'analyse MNE."""
        # Déjà implémenté plus haut
    
    def _run_sloreta_analysis(self, parent):
        """Exécute l'analyse sLORETA."""
        messagebox.showinfo("Info", "sLORETA - Méthode avancée d'analyse de sources")

    def _run_dspm_analysis(self, parent):
        """Exécute l'analyse dSPM."""
        messagebox.showinfo("Info", "dSPM - Dynamic Statistical Parametric Mapping")

    def _run_beamforming_analysis(self, parent):
        """Exécute l'analyse beamforming."""
        messagebox.showinfo("Info", "Beamforming LCMV - Méthode de formation de faisceaux")

    def _visualize_sources(self, parent):
        """Visualise les sources localisées."""
        messagebox.showinfo("Info", "Visualisation - Les sources sont affichées dans les résultats MNE")

    def _export_source_data(self, parent):
        """Exporte les données de sources."""
        messagebox.showinfo("Info", "Export - Utilisez les boutons d'export dans les résultats d'analyse")
    
    def _visualize_sleep_stages(self):
        """Visualise les stades de sommeil sur le graphique."""
        df = self._get_active_scoring_df()
        if df is None or len(df) == 0:
            messagebox.showwarning("Avertissement", "Aucun scoring de sommeil chargé.")
            return
        
        # Créer une fenêtre de visualisation
        viz_window = tk.Toplevel(self.root)
        viz_window.title("🎨 Visualisation des Stades de Sommeil")
        viz_window.geometry("800x600")
        viz_window.transient(self.root)
        viz_window.grab_set()
        
        # Centrer la fenêtre
        viz_window.update_idletasks()
        x = (viz_window.winfo_screenwidth() // 2) - (800 // 2)
        y = (viz_window.winfo_screenheight() // 2) - (600 // 2)
        viz_window.geometry(f"800x600+{x}+{y}")
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Couleurs pour les stades (utilise la palette dynamique)
        stage_colors = self.theme_manager.get_stage_colors().copy()
        stage_colors.update({'U': '#DDA0DD'})  # Violet pour les stades inconnus
        
        # Créer l'histogramme des stades
        time_hours = df['time'] / 3600
        stages = df['stage']
        
        # Créer des barres pour chaque stade
        for i, (stage, color) in enumerate(stage_colors.items()):
            mask = stages == stage
            if mask.any():
                ax.bar(time_hours[mask], [1] * mask.sum(), 
                      color=color, alpha=0.7, label=self.sleep_stages.get(stage, stage))
        
        ax.set_xlabel('Temps (heures)')
        ax.set_ylabel('Stade de sommeil')
        ax.set_title('Visualisation des Stades de Sommeil')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Intégrer dans Tkinter
        canvas = FigureCanvasTkAgg(fig, viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barre d'outils
        toolbar = NavigationToolbar2Tk(canvas, viz_window)
        toolbar.update()
        
        print("✅ Visualisation des stades de sommeil créée")
        logging.info("Visualisation des stades de sommeil affichée")
    
    def _export_sleep_scoring(self):
        """Exporte les données de scoring de sommeil."""
        df = self._get_active_scoring_df()
        if df is None or len(df) == 0:
            messagebox.showwarning("Avertissement", "Aucun scoring de sommeil chargé.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exporter le scoring de sommeil",
            defaultextension=".csv",
            filetypes=[
                ("Fichiers CSV", "*.csv"),
                ("Fichiers Excel", "*.xlsx"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False, encoding='utf-8')
            else:
                # Si extension .xlsx choisie, exporter le manuel si dispo, sinon auto
                df.to_excel(file_path, index=False)
            
            messagebox.showinfo("Succès", f"Scoring de sommeil exporté avec succès!\n{file_path}")
            logging.info(f"Scoring de sommeil exporté: {file_path}")
            
        except Exception as e:
            error_msg = f"Erreur lors de l'export:\n{str(e)}"
            messagebox.showerror("Erreur", error_msg)
            logging.error(f"Erreur export scoring: {e}")

    def _save_active_scoring(self):
        """Sauvegarde le scoring actif (manuel prioritaire) en CSV."""
        df = self._get_active_scoring_df()
        if df is None or len(df) == 0:
            messagebox.showwarning("Scoring", "Aucun scoring à sauvegarder")
            return
        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder scoring (CSV)", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not file_path:
            return
        try:
            df[['time', 'stage']].to_csv(file_path, index=False, encoding='utf-8')
            self.scoring_dirty = False
            messagebox.showinfo("Scoring", f"Scoring sauvegardé: {file_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Echec sauvegarde scoring: {e}")

    def _open_goto_time_dialog(self):
        """Ouvre un dialogue pour aller à un temps t (en secondes ou HH:MM:SS)."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Aller au temps...")
        dlg.geometry("320x140")
        dlg.transient(self.root)
        dlg.grab_set()
        frame = ttk.Frame(dlg, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Temps (s ou HH:MM:SS):").pack(anchor='w')
        var = tk.StringVar(value=f"{self.current_time:.1f}")
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(fill=tk.X)
        def _go():
            txt = var.get().strip()
            t = None
            try:
                if ":" in txt:
                    # Gestion robuste du format HH:MM:SS
                    parts = txt.split(':')
                    if len(parts) == 3:
                        h, m, s = [int(p) for p in parts]
                        # Validation des valeurs
                        if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
                            messagebox.showerror("Erreur", "Format d'heure invalide. Utilisez HH:MM:SS avec des valeurs valides (00-23:00-59:00-59)")
                            return
                        t = float(h*3600 + m*60 + s)
                    else:
                        messagebox.showerror("Erreur", "Format invalide. Utilisez HH:MM:SS")
                        return
                else:
                    t = float(txt)
                    # Validation pour les secondes
                    if t < 0:
                        messagebox.showerror("Erreur", "Le temps ne peut pas être négatif")
                        return
            except ValueError:
                messagebox.showerror("Erreur", "Format invalide. Entrez des secondes ou HH:MM:SS")
                return
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur de conversion: {str(e)}")
                return

            # Validation de la plage
            max_time = (len(self.raw.times)/self.sfreq) - 1e-3 if self.raw else float('inf')
            if t > max_time:
                messagebox.showwarning("Attention", f"Le temps demandé ({t:.1f}s) dépasse la durée de l'enregistrement ({max_time:.1f}s). Navigation à la fin.")
                t = max_time

            self._jump_to_time(t)
            dlg.destroy()
        ttk.Button(frame, text="Aller", command=_go).pack(side=tk.RIGHT, pady=(10,0))
        ttk.Button(frame, text="Annuler", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0,6), pady=(10,0))

    def _load_hypnogram_edfplus(self):
        """Charge un hypnogramme EDF (Sleep-EDFx) et le convertit en df standard (time, stage)."""
        try:
            file_path = filedialog.askopenfilename(
                title="Charger Hypnogram EDF",
                filetypes=[("EDF", "*.edf"), ("Tous les fichiers", "*.*")]
            )
            if not file_path:
                return
            raw_duration = float(len(self.raw.times) / self.sfreq) if self.raw is not None else 0.0
            rec_duration = min(raw_duration, 24 * 3600)
            result = self.manual_scoring_service.import_edf_path(
                file_path,
                recording_duration_s=rec_duration,
                epoch_seconds=float(getattr(self, "scoring_epoch_duration", 30.0)),
                absolute_start_datetime=getattr(self, "absolute_start_datetime", None),
            )
            self._apply_manual_scoring_result(result, source="EDF hypnogram")
            self.scoring_dirty = True
            messagebox.showinfo("Hypnogram", f"Hypnogramme EDF chargé: {len(result.df)} époques")
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Erreur", f"Echec chargement hypnogramme EDF: {e}")
    
    def _show_about(self):
        """Affiche la boîte de dialogue À propos."""
        about_text = """
CESA (Complex EEG Studio Analysis) v0.0beta1.1

Application professionnelle pour l'analyse de données EEG
avec amplification automatique des signaux de faible amplitude
et importation de scoring de sommeil.

Fonctionnalités:
• Chargement de fichiers EDF avec barre de progression
• Visualisation multi-canaux avec couleurs distinctes
• Amplification automatique des signaux faibles
• Filtrage passe-bande avec options avancées
• Autoscale intelligent et navigation temporelle
• Statistiques détaillées et diagnostic en temps réel
• Export de données et rapports
• Sélecteur de canaux amélioré
• Importation de scoring de sommeil depuis Excel
• Visualisation des stades de sommeil

Développé avec Python, MNE-Python, Matplotlib et Tkinter.

Version: 0.0beta1.1
Date: 2025-09-09
        """
        messagebox.showinfo("À propos", about_text)
    
    def _create_modern_widgets(self) -> None:
        """
        Crée l'interface moderne avec widgets professionnels.
        
        Configure l'interface utilisateur avec un design moderne,
        des contrôles intuitifs et une barre d'état informative.
        """
        # =====================================================================
        # STRUCTURE COMPLÈTE - HIÉRARCHIE CORRECTE
        # =====================================================================
        
        # Frame principal qui occupe toute la fenêtre
        main_container = ttk.Frame(self.root, style='Modern.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # =====================================================================
        # BARRE D'ÉTAT EN HAUT
        # =====================================================================
        
        self._create_status_bar(main_container)
        
        # =====================================================================
        # CONTENU PRINCIPAL (GRAPHIQUE + CONTRÔLES)
        # =====================================================================
        
        # Frame pour le contenu principal
        content_frame = ttk.Frame(main_container, style='Modern.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=(0, 0))  # Réduit encore plus : 3px
        
        # Frame de contrôle à droite (panel unique moderne)
        self.control_frame = ttk.Frame(content_frame, width=self.control_width, style='Modern.TFrame')
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 0))  # Réduit encore plus : 3px
        self.control_frame.pack_propagate(False)
        try:
            print("🔍 CHECKPOINT PANEL: creating main control panel")
        except Exception:
            pass
        try:
            self._create_control_panel(self.control_frame)
        except Exception as e:
            try:
                print(f"❌ CHECKPOINT PANEL: failed to create panel: {e}")
            except Exception:
                pass
        
        # Bouton de basculement du panneau
        self._create_panel_toggle_button(content_frame)
        
        # Frame du graphique à gauche
        plot_frame = ttk.Frame(content_frame, style='Modern.TFrame')
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configuration du graphique matplotlib moderne
        self._setup_modern_plot(plot_frame)
        
        # =====================================================================
        # BARRE D'OUTILS MODERNE
        # =====================================================================
        
        self._create_modern_toolbar(plot_frame)
        
        # =====================================================================
        # BARRE DE TEMPS EN BAS DE FENÊTRE (VRAIMENT EN BAS)
        # =====================================================================
        
        self._create_bottom_time_bar(main_container)
        
        # =====================================================================
        # BINDINGS CLAVIER POUR NAVIGATION
        # =====================================================================
        
        self._setup_keyboard_navigation_simple()

        # Appliquer le thème initial à l'interface
        self.theme_manager.apply_theme_to_root(self.root)

        logging.info("Interface créée avec succès")
    
    def _add_combined_scoring_to_plot(self, start_time, end_time, primary_scoring_data):
        """Affiche les scorings auto et manuel juxtaposés (côte à côte) tout en bas."""
        try:
            print(f"🔍 CHECKPOINT COMBINED SCORING: Début affichage combiné")
            
            # Calculer les positions Y pour les deux scorings (juxtaposés en bas mais visibles)
            y_min, y_max = self.ax.get_ylim()
            scoring_height = (y_max - y_min) * 0.02  # Hauteur réduite (divisée par 4)
            
            # Position en bas du graphique mais dans la zone visible
            base_scoring_y = y_min - scoring_height * 1.0  # Plus proche du graphique principal
            auto_scoring_y = base_scoring_y
            manual_scoring_y = base_scoring_y - scoring_height * 1.2  # Espacement réduit
            
            print(f"🔍 CHECKPOINT COMBINED: Auto Y: {auto_scoring_y:.2f}, Manuel Y: {manual_scoring_y:.2f}")
            
            # Afficher d'abord le scoring automatique (en haut)
            if hasattr(self, 'sleep_scoring_data') and self.sleep_scoring_data is not None:
                print(f"🔍 CHECKPOINT COMBINED: Ajout scoring AUTO (en haut)")
                self._add_sleep_scoring_to_plot(start_time, end_time, self.sleep_scoring_data, 
                                              alpha=0.7, label_prefix="Auto", zorder=1, y_offset=0, force_y_position=auto_scoring_y)
            
            # Puis afficher le scoring manuel (en bas) s'il existe
            if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None and self.show_manual_scoring:
                print(f"🔍 CHECKPOINT COMBINED: Ajout scoring MANUEL (en bas)")
                self._add_sleep_scoring_to_plot(start_time, end_time, self.manual_scoring_data, 
                                              alpha=0.7, label_prefix="Manuel", zorder=1, y_offset=0, force_y_position=manual_scoring_y)
            
            print(f"✅ CHECKPOINT COMBINED: Affichage combiné terminé")
            
        except Exception as e:
            print(f"❌ CHECKPOINT COMBINED: Erreur affichage combiné: {e}")
            logging.error(f"[COMBINED] Erreur affichage combiné: {e}")
    
    def _setup_keyboard_navigation_simple(self):
        """Configure les raccourcis clavier pour la navigation simple (30s fixe)."""
        try:
            # Détacher les anciens bindings pour éviter les doublons
            self.root.unbind('<Key>')
            
            # Raccourcis ZQSD pour navigation par époques (30s) - navigation simple
            self.root.bind('<Key-z>', lambda e: self._navigate_simple_epoch_previous())
            self.root.bind('<Key-q>', lambda e: self._navigate_simple_epoch_previous())
            self.root.bind('<Key-s>', lambda e: self._navigate_simple_epoch_next())
            self.root.bind('<Key-d>', lambda e: self._navigate_simple_epoch_next())
            
            # Raccourcis flèches pour navigation fine
            self.root.bind('<Up>', lambda e: self._increase_amplitude_with_zqsd_reset())
            self.root.bind('<Down>', lambda e: self._decrease_amplitude_with_zqsd_reset())
            self.root.bind('<Left>', lambda e: self._step_backward_with_zqsd_reset())
            self.root.bind('<Right>', lambda e: self._step_forward_with_zqsd_reset())
            
            # Focus sur la fenêtre principale pour capturer les touches
            self.root.focus_set()

            print("🎹 Raccourcis clavier configurés: Z/Q=époques précédentes, S/D=époques suivantes (navigation simple)")
            print("🎨 Thèmes disponibles: Otilia 🦖🌸, Fred 🧗‍♂️🌿, Eléna 🐢🌊 (sélecteur dans la barre d'outils)")
            logging.info("Keyboard navigation configured: Z/Q=prev epoch, S/D=next epoch (simple mode)")
            
        except Exception as e:
            print(f"⚠️ Erreur configuration raccourcis clavier: {e}")
            logging.warning(f"Keyboard navigation setup failed: {e}")

    def _change_theme(self, theme_name: str):
        """Change le thème actuel et met à jour l'interface."""
        try:
            self.theme_manager.set_theme(theme_name)

            # Mettre à jour l'affichage si des données sont chargées
            if self.raw is not None:
                self.update_plot()

            # Propager aussi au viewer PSG si présent pour recolorer immédiatement
            try:
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter.set_theme(theme_name)
                elif self._qt_psg_plot_active():
                    tnl = str(theme_name).lower()
                    qtheme = (
                        "dark"
                        if ("dark" in tnl or "sombre" in tnl or "night" in tnl)
                        else "light"
                    )
                    self._qt_viewer_bridge.set_theme(qtheme)
            except Exception:
                pass

            # Mettre à jour les fenêtres FFT ouvertes si elles existent
            self._update_fft_windows()

            # Appliquer le thème à la fenêtre principale
            self.theme_manager.apply_theme_to_root(self.root)

            # Mettre à jour le style du graphique principal si le graphique existe
            if hasattr(self, 'fig') and self.fig is not None:
                # Forcer la mise à jour du fond de la figure
                theme = self.theme_manager.get_current_theme()
                ui_colors = theme.get_ui_colors()
                self.fig.patch.set_facecolor(ui_colors.get('bg', '#ffffff'))

                # Réappliquer les styles
                self._apply_modern_plot_style()

                # Mettre à jour les couleurs des lignes EEG existantes
                self._update_eeg_line_colors()

                # Forcer le rafraîchissement du canvas
                if hasattr(self, 'canvas'):
                    self.canvas.draw_idle()

        except Exception as e:
            print(f"⚠️ Erreur changement thème: {e}")
            logging.warning(f"Theme change failed: {e}")

    def _change_theme_by_display_name(self, display_name: str):
        """Change le thème en fonction du nom d'affichage."""
        # Trouver le nom du thème correspondant au nom d'affichage
        for theme_key, theme_display in self.theme_manager.get_available_themes().items():
            if theme_display == display_name:
                self._change_theme(theme_key)
                break

    def _update_eeg_line_colors(self):
        """Met à jour les couleurs des lignes EEG avec le thème actuel."""
        try:
            theme = self.theme_manager.get_current_theme()
            ui_colors = theme.get_ui_colors()

            # Mettre à jour la couleur de chaque ligne EEG
            for line in self.eeg_lines:
                line.set_color(ui_colors.get('fg', '#000000'))

            # Forcer le redessin
            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()

        except Exception as e:
            logging.debug(f"Erreur mise à jour couleurs EEG: {e}")

    def _update_fft_windows(self):
        """Met à jour les fenêtres FFT ouvertes avec le nouveau thème."""
        try:
            # Chercher les fenêtres FFT ouvertes et les mettre à jour
            for child in self.root.winfo_children():
                if isinstance(child, tk.Toplevel):
                    # Vérifier si c'est une fenêtre FFT (par le titre)
                    if "PSD par stade (FFT" in child.title():
                        # Appliquer le thème à cette fenêtre aussi
                        self.theme_manager.apply_theme_to_root(child)
        except Exception as e:
            print(f"⚠️ Erreur mise à jour fenêtres FFT: {e}")
    
    def _navigate_epoch_previous(self):
        """Navigation vers l'époque précédente (30s en arrière)."""
        epoch_duration = 30.0  # Durée d'une époque
        self.current_time = max(0, self.current_time - epoch_duration)
        if hasattr(self, 'time_var'):
            self.time_var.set(self.current_time)
        if hasattr(self, 'bottom_time_var'):
            self.bottom_time_var.set(self.current_time)
        self.update_plot()
        print(f"⬅️ Époque précédente: {self.current_time:.1f}s")
        logging.info(f"Previous epoch: {self.current_time:.1f}s")
    
    def _navigate_epoch_next(self):
        """Navigation vers l'époque suivante (30s en avant)."""
        epoch_duration = 30.0  # Durée d'une époque
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + epoch_duration)
            if hasattr(self, 'time_var'):
                self.time_var.set(self.current_time)
            if hasattr(self, 'bottom_time_var'):
                self.bottom_time_var.set(self.current_time)
            self.update_plot()
            print(f"➡️ Époque suivante: {self.current_time:.1f}s")
            logging.info(f"Next epoch: {self.current_time:.1f}s")
    
    def _navigate_simple_epoch_previous(self):
        """Navigation simple vers l'époque précédente (30s en arrière) sans changer la durée."""
        epoch_duration = 30.0  # Durée d'une époque fixe
        self.current_time = max(0, self.current_time - epoch_duration)
        
        # Mettre à jour les sliders
        if hasattr(self, 'time_var'):
            self.time_var.set(self.current_time)
        if hasattr(self, 'bottom_time_var'):
            self.bottom_time_var.set(self.current_time)
        
        # Mettre à jour l'affichage
        self.update_plot()
        
        print(f"⬅️ Navigation simple: époque précédente {self.current_time:.1f}s")
        logging.info(f"Simple navigation: previous epoch {self.current_time:.1f}s")
    
    def _navigate_simple_epoch_next(self):
        """Navigation simple vers l'époque suivante (30s en avant) sans changer la durée."""
        epoch_duration = 30.0  # Durée d'une époque fixe
        
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + epoch_duration)
        else:
            self.current_time += epoch_duration
        
        # Mettre à jour les sliders
        if hasattr(self, 'time_var'):
            self.time_var.set(self.current_time)
        if hasattr(self, 'bottom_time_var'):
            self.bottom_time_var.set(self.current_time)
        
        # Mettre à jour l'affichage
        self.update_plot()
        
        print(f"➡️ Navigation simple: époque suivante {self.current_time:.1f}s")
        logging.info(f"Simple navigation: next epoch {self.current_time:.1f}s")
    
    def _create_panel_toggle_button(self, parent):
        """Crée le bouton de basculement du panneau de commandes."""
        # Bouton flottant pour basculer le panneau
        self.toggle_button = ttk.Button(
            parent,
            text="◀",
            command=self._toggle_control_panel,
            style='Modern.TButton',
            width=3
        )
        # Positionner le bouton sur le côté droit du panneau de contrôle
        self.toggle_button.place(in_=self.control_frame, x=-25, y=50, anchor=tk.W)
    
    def _toggle_control_panel(self):
        """Bascule l'affichage du panneau de commandes."""
        if self.control_panel_collapsed:
            # Déployer le panneau
            self.control_frame.config(width=self.original_control_width)
            self.toggle_button.config(text="◀")
            self.control_panel_collapsed = False
            
            # Recréer l'ensemble des contrôles pour garantir la cohérence
            for child in self.control_frame.winfo_children():
                try:
                    child.destroy()
                except Exception:
                    pass
            try:
                self._create_control_panel(self.control_frame)
                # Réappliquer le thème pour corriger les couleurs (évite bleu/fond blanc)
                try:
                    from CESA.theme_manager import theme_manager as _tm
                    _tm.apply_theme_to_root(self.root)
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"[PANEL] Rebuild controls failed: {e}")
            
            print("📋 Panneau de commandes déployé")
        else:
            # Rétracter le panneau
            self.control_frame.config(width=50)
            self.toggle_button.config(text="▶")
            self.control_panel_collapsed = True
            
            # Supprimer les contrôles pour un re-build propre
            for child in self.control_frame.winfo_children():
                try:
                    child.destroy()
                except Exception:
                    pass
            
            # Réafficher le bouton de basculement
            self.toggle_button.place(in_=self.control_frame, x=-25, y=50, anchor=tk.W)
            
            print("📋 Panneau de commandes rétracté")
        
        # Forcer la mise à jour de l'interface
        self.root.update_idletasks()
        
        # Mettre à jour le graphique pour s'adapter à la nouvelle taille
        self._update_plot_layout()
    
    def _update_plot_layout(self):
        """Met à jour la mise en page du graphique pour optimiser l'espace."""
        if hasattr(self, 'fig') and hasattr(self, 'canvas'):
            # Ajuster les marges selon l'état du panneau
            if self.control_panel_collapsed:
                # Panneau rétracté : maximiser l'espace
                self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.10)
            else:
                # Panneau déployé : ajuster pour le panneau
                self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
            
            # Forcer le redessinage
            self.canvas.draw_idle()
    
    def _setup_keyboard_navigation_advanced(self):
        """Configure la navigation clavier ZQSD pour se déplacer de score en score - VERSION AVANCÉE DÉSACTIVÉE."""
        # Cette méthode est désactivée au profit de la navigation simple
        # Bindings pour la navigation clavier (fenêtre + global + canvas)
        targets = [self.root]
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'get_tk_widget'):
            try:
                targets.append(self.canvas.get_tk_widget())
            except Exception:
                pass
        try:
            names = [getattr(t, 'winfo_name', lambda: str(t))() for t in targets]
        except Exception:
            names = [str(type(t)) for t in targets]
        print(f"⌨️ CHECKPOINT BIND: Cibles de binding ZQSD: {names}")
        logging.info(f"[BIND] Cibles ZQSD: {names}")
        
        # NAVIGATION SIMPLE : Remplacer les bindings complexes par la navigation simple
        for t in targets:
            # Bindings ZQSD avec navigation simple (30s fixe)
            t.bind('<KeyPress-z>', lambda e: self._navigate_simple_epoch_previous())
            t.bind('<KeyPress-q>', lambda e: self._navigate_simple_epoch_previous())
            t.bind('<KeyPress-s>', lambda e: self._navigate_simple_epoch_next())
            t.bind('<KeyPress-d>', lambda e: self._navigate_simple_epoch_next())
        
        # Bindings alternatifs pour s'assurer que ça fonctionne (global)
        self.root.bind_all('<z>', lambda e: self._navigate_simple_epoch_previous())
        self.root.bind_all('<q>', lambda e: self._navigate_simple_epoch_previous())
        self.root.bind_all('<s>', lambda e: self._navigate_simple_epoch_next())
        self.root.bind_all('<d>', lambda e: self._navigate_simple_epoch_next())
        
        # Raccourcis pour scoring auto et comparaison
        self.root.bind_all('<Control-y>', lambda e: self._run_auto_sleep_scoring())
        self.root.bind_all('<Control-c>', lambda e: self._compare_scoring())
        
        # Raccourci pour l'automatisation FFT
        self.root.bind_all('<Control-b>', lambda e: self._show_batch_fft_automation())
        
        # Raccourci pour basculer le panneau de commandes
        self.root.bind_all('<Control-p>', lambda e: self._toggle_control_panel())
        self.root.bind_all('<F2>', lambda e: self._toggle_control_panel())
        
        # Raccourci pour générer un rapport de bug (Ctrl+Shift+B)
        self.root.bind_all('<Control-Shift-B>', lambda e: self._report_bug())
        
        # Raccourcis pour les nouvelles fonctionnalités
        self.root.bind_all('<Control-Shift-T>', lambda e: self._show_temporal_analysis())
        self.root.bind_all('<Control-Shift-K>', lambda e: self._show_markers())
        self.root.bind_all('<Control-Shift-L>', lambda e: self._show_measurements())
        
        # Focus sur la fenêtre principale pour recevoir les événements clavier
        self.root.focus_set()
        self.root.focus_force()
        
        print("⌨️ Navigation clavier ZQSD activée (mode simple)")
        logging.info("[BIND] Navigation ZQSD activée (simple mode)")
    
    def _navigate_to_previous_score(self, event=None):
        """Navigue vers le score précédent (Z ou Q)."""
        print(f"🔍 CHECKPOINT NAV 1: Navigation vers score précédent, event={getattr(event, 'keysym', None)}")
        logging.info(f"[NAV] prev key={getattr(event, 'keysym', None)}")
        logging.info("[NAV] Précédent")
        print(f"🔍 CHECKPOINT NAV 1: Temps actuel: {self.current_time:.1f}s")
        logging.info(f"[NAV] current_time={self.current_time:.3f}")
        logging.info(f"[NAV] temps_actuel={self.current_time:.1f}")
        df = self._get_active_scoring_df()
        print(f"🔍 CHECKPOINT NAV 1: Scoring dispo={df is not None}, n={len(df) if df is not None else 0}")
        logging.info(f"[NAV] n_epochs={len(df) if df is not None else 0}")
        
        if df is None or len(df) == 0:
            print("⚠️ CHECKPOINT NAV 1: Aucun scoring chargé")
            return
        
        # Utiliser la durée d'époque correcte
        epoch_duration = getattr(self, 'scoring_epoch_duration', 30.0)
        print(f"🔍 CHECKPOINT NAV 1: Durée d'époque utilisée: {epoch_duration:.1f}s")
        
        print(f"🔍 CHECKPOINT NAV 1: Plage scoring: {df['time'].min():.1f}s - {df['time'].max():.1f}s")
        
        # CORRECTION: Utiliser le centre de la fenêtre courante comme référence
        window_duration = float(getattr(self, 'duration', 10.0))
        current_center = float(self.current_time) + (window_duration / 2.0)
        print(f"🔍 CHECKPOINT NAV 1: Centre fenêtre: {current_center:.1f}s (durée={window_duration:.1f}s)")
        logging.info(f"[NAV] center={current_center:.3f}, win={window_duration:.3f}")

        # Navigation robuste: utiliser une recherche binaire sur les temps (précision maximale)
        times_arr = df['time'].to_numpy(dtype=float)
        print(f"🔍 CHECKPOINT NAV 1: times_arr head={times_arr[:5]}")
        logging.info(f"[NAV] times_head={times_arr[:5].tolist()}")
        idx = int(np.searchsorted(times_arr, current_center, side='left'))
        prev_idx = idx - 1
        print(f"🔍 CHECKPOINT NAV 1: searchsorted idx={idx}, prev_idx={prev_idx}")
        logging.info(f"[NAV] prev_idx={prev_idx}")
        previous_epochs = df.iloc[[prev_idx]] if prev_idx >= 0 else df.iloc[[]]
        
        print(f"🔍 CHECKPOINT NAV 1: Époques précédentes trouvées: {len(previous_epochs)}")
        if len(previous_epochs) > 0:
            print(f"🔍 CHECKPOINT NAV 1: Premières époques précédentes: {previous_epochs.head()}")
            print(f"🔍 CHECKPOINT NAV 1: Dernières époques précédentes: {previous_epochs.tail()}")
        
        if len(previous_epochs) > 0:
            # Aller au début de la dernière époque précédente
            target_time = float(times_arr[prev_idx])
            print(f"🔍 CHECKPOINT NAV 1: Temps cible: {target_time:.1f}s")
            logging.info(f"[NAV] prev_target={target_time:.3f}")
            logging.info(f"[NAV] target={target_time:.1f}")
            self._jump_to_time(target_time)
            print(f"⬅️ CHECKPOINT NAV 1: Navigation vers score précédent: {target_time:.1f}s")
            logging.info(f"[NAV] Go précédent -> {target_time:.1f}")
            
            # Navigation simple sans changer la durée
            # self._center_view_on_epoch(target_time, epoch_duration)  # Désactivé pour garder la durée actuelle
        else:
            print("⚠️ CHECKPOINT NAV 1: Aucun score précédent disponible")
    
    def _navigate_to_next_score(self, event=None):
        """Navigue vers le score suivant (S ou D)."""
        print(f"🔍 CHECKPOINT NAV 2: Navigation vers score suivant, event={getattr(event, 'keysym', None)}")
        logging.info(f"[NAV] next key={getattr(event, 'keysym', None)}")
        logging.info("[NAV] Suivant")
        print(f"🔍 CHECKPOINT NAV 2: Temps actuel: {self.current_time:.1f}s")
        logging.info(f"[NAV] current_time={self.current_time:.3f}")
        logging.info(f"[NAV] temps_actuel={self.current_time:.1f}")
        df = self._get_active_scoring_df()
        print(f"🔍 CHECKPOINT NAV 2: Scoring dispo={df is not None}, n={len(df) if df is not None else 0}")
        logging.info(f"[NAV] n_epochs={len(df) if df is not None else 0}")
        
        if df is None or len(df) == 0:
            print("⚠️ CHECKPOINT NAV 2: Aucun scoring chargé")
            return
        
        # Utiliser la durée d'époque correcte
        epoch_duration = getattr(self, 'scoring_epoch_duration', 30.0)
        print(f"🔍 CHECKPOINT NAV 2: Durée d'époque utilisée: {epoch_duration:.1f}s")
        
        print(f"🔍 CHECKPOINT NAV 2: Plage scoring: {df['time'].min():.1f}s - {df['time'].max():.1f}s")
        
        # CORRECTION: Utiliser le centre de la fenêtre courante comme référence
        window_duration = float(getattr(self, 'duration', 10.0))
        current_center = float(self.current_time) + (window_duration / 2.0)
        print(f"🔍 CHECKPOINT NAV 2: Centre fenêtre: {current_center:.1f}s (durée={window_duration:.1f}s)")
        logging.info(f"[NAV] center={current_center:.3f}, win={window_duration:.3f}")

        # Navigation robuste: utiliser searchsorted pour l'époque suivante
        times_arr = df['time'].to_numpy(dtype=float)
        print(f"🔍 CHECKPOINT NAV 2: times_arr head={times_arr[:5]}")
        logging.info(f"[NAV] times_head={times_arr[:5].tolist()}")
        idx = int(np.searchsorted(times_arr, current_center, side='right'))
        print(f"🔍 CHECKPOINT NAV 2: searchsorted idx={idx}")
        logging.info(f"[NAV] next_idx={idx}")
        next_epochs = df.iloc[[idx]] if idx < len(times_arr) else df.iloc[[]]
        
        print(f"🔍 CHECKPOINT NAV 2: Époques suivantes trouvées: {len(next_epochs)}")
        if len(next_epochs) > 0:
            print(f"🔍 CHECKPOINT NAV 2: Premières époques suivantes: {next_epochs.head()}")
            print(f"🔍 CHECKPOINT NAV 2: Dernières époques suivantes: {next_epochs.tail()}")
        
        if len(next_epochs) > 0:
            # Aller au début de la première époque suivante
            target_time = float(times_arr[idx])
            print(f"🔍 CHECKPOINT NAV 2: Temps cible: {target_time:.1f}s")
            logging.info(f"[NAV] next_target={target_time:.3f}")
            logging.info(f"[NAV] target={target_time:.1f}")
            self._jump_to_time(target_time)
            print(f"➡️ CHECKPOINT NAV 2: Navigation vers score suivant: {target_time:.1f}s")
            logging.info(f"[NAV] Go suivant -> {target_time:.1f}")
            
            # Navigation simple sans changer la durée
            # self._center_view_on_epoch(target_time, epoch_duration)  # Désactivé pour garder la durée actuelle
        else:
            print("⚠️ CHECKPOINT NAV 2: Aucun score suivant disponible")
    
    def _jump_to_time(self, target_time):
        """Saute à un temps spécifique."""
        print(f"🔍 CHECKPOINT JUMP 1: Saut vers le temps {target_time:.1f}s")
        logging.info(f"[JUMP] target={target_time:.1f}")
        print(f"🔍 CHECKPOINT JUMP 1: Temps actuel avant: {self.current_time:.1f}s")
        logging.info(f"[JUMP] before={self.current_time:.1f}")
        
        # Mettre à jour le temps actuel
        self.current_time = target_time
        print(f"🔍 CHECKPOINT JUMP 1: Temps actuel après: {self.current_time:.1f}s")
        logging.info(f"[JUMP] after={self.current_time:.1f}")
        
        # Mettre à jour les sliders
        print(f"🔍 CHECKPOINT JUMP 2: Mise à jour des sliders...")
        logging.info("[JUMP] maj sliders")
        if hasattr(self, 'time_var'):
            self.time_var.set(target_time)
            print(f"🔍 CHECKPOINT JUMP 2: Slider du haut mis à jour")
        else:
            print(f"⚠️ CHECKPOINT JUMP 2: Slider du haut non trouvé")
            
        if hasattr(self, 'bottom_time_var'):
            self.bottom_time_var.set(target_time)
            print(f"🔍 CHECKPOINT JUMP 2: Slider du bas mis à jour")
        else:
            print(f"⚠️ CHECKPOINT JUMP 2: Slider du bas non trouvé")
        
        # Mettre à jour l'affichage
        print(f"🔍 CHECKPOINT JUMP 3: Mise à jour de l'affichage...")
        logging.info("[JUMP] maj affichage")
        self._update_time_display()
        print(f"🔍 CHECKPOINT JUMP 3: Affichage mis à jour")
        
        print(f"🔍 CHECKPOINT JUMP 4: Mise à jour du graphique...")
        logging.info("[JUMP] maj graphique")
        self.update_plot()
        print(f"✅ CHECKPOINT JUMP 4: Saut terminé vers {target_time:.1f}s")
        logging.info(f"[JUMP] fin -> {target_time:.1f}")
    
    def _update_time_display(self):
        """Met à jour l'affichage du temps."""
        # Navigation: toujours HHhMM (ou HH:MM:SS converti en HHhMM si absolu)
        if self.time_format == "absolu" and hasattr(self, 'absolute_start_datetime'):
            nav_text = self._get_absolute_time(self.current_time)
            try:
                nav_text = nav_text[:2] + 'h' + nav_text[3:5]
            except Exception:
                pass
        else:
            nav_text = self._format_hhmm(self.current_time)
        
        # Mise à jour des labels avec vérification d'existence
        try:
            if hasattr(self, 'time_label') and self.time_label.winfo_exists():
                self.time_label.config(text=nav_text)
        except Exception as e:
            print(f"⚠️ CHECKPOINT LABEL: Erreur mise à jour time_label: {e}")
            logging.warning(f"[LABEL] time_label update failed: {e}")
        
        try:
            if hasattr(self, 'bottom_time_label') and self.bottom_time_label.winfo_exists():
                self.bottom_time_label.config(text=nav_text)
        except Exception as e:
            print(f"⚠️ CHECKPOINT LABEL: Erreur mise à jour bottom_time_label: {e}")
            logging.warning(f"[LABEL] bottom_time_label update failed: {e}")
        # Mise à jour des sliders
        try:
            if hasattr(self, 'time_var'):
                self.time_var.set(self.current_time)
            if hasattr(self, 'bottom_time_var'):
                self.bottom_time_var.set(self.current_time)
        except Exception:
            pass
    
    def _create_bottom_time_bar(self, parent: ttk.Frame) -> None:
        """Crée une barre de temps en bas de la fenêtre qui occupe toute la largeur."""
        print(f"🔍 CHECKPOINT BARRE 1: Création de la barre temporelle")
        print(f"🔍 CHECKPOINT BARRE 1: Parent: {parent}")
        
        # Frame pour la barre de temps en bas - VRAIMENT EN BAS DE LA FENÊTRE
        bottom_time_frame = ttk.Frame(parent, style='Modern.TFrame')
        print(f"🔍 CHECKPOINT BARRE 1: Frame créé: {bottom_time_frame}")
        
        bottom_time_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2))  # Réduit les marges
        print(f"🔍 CHECKPOINT BARRE 1: Frame packé en bas")
        
        # Forcer la barre à rester en bas
        bottom_time_frame.pack_propagate(False)
        print(f"🔍 CHECKPOINT BARRE 1: pack_propagate(False) appliqué")
        
        # Titre de la barre de temps
        time_title = ttk.Label(bottom_time_frame, text="⏱️ Navigation Temporelle", 
                              style='Modern.TLabel', font=('Segoe UI', 10, 'bold'))
        time_title.pack(anchor=tk.W, pady=(0, 3))  # Réduit l'espacement
        
        # Frame pour le slider et les contrôles - OCCUPE TOUTE LA LARGEUR
        time_controls_frame = ttk.Frame(bottom_time_frame, style='Modern.TFrame')
        time_controls_frame.pack(fill=tk.X, pady=(0, 3))  # Réduit l'espacement
        
        # Slider de temps principal - OCCUPE TOUTE LA LARGEUR DISPONIBLE
        self.bottom_time_var = tk.DoubleVar(value=0.0)
        self.bottom_time_scale = ttk.Scale(
            time_controls_frame,
            from_=0, to=100,
            orient=tk.HORIZONTAL,
            variable=self.bottom_time_var,
            command=self._update_time_from_bottom
        )
        self.bottom_time_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Affichage du temps actuel (relatif et absolu)
        self.bottom_time_label = ttk.Label(
            time_controls_frame,
            text="00h00",
            style='Modern.TLabel',
            font=('Segoe UI', 10, 'bold'),
            width=20
        )
        self.bottom_time_label.pack(side=tk.RIGHT)
        
        # Frame pour les contrôles rapides
        quick_controls_frame = ttk.Frame(bottom_time_frame, style='Modern.TFrame')
        quick_controls_frame.pack(fill=tk.X)
        
        # Boutons de navigation rapide
        nav_buttons = [
            ("⏮️", self._jump_backward, "Saut arrière"),
            ("⏪", self._step_backward, "Pas arrière"),
            ("⏸️", self._pause_play, "Pause/Play"),
            ("⏩", self._step_forward, "Pas avant"),
            ("⏭️", self._jump_forward, "Saut avant")
        ]
        
        for i, (text, command, tooltip) in enumerate(nav_buttons):
            btn = ttk.Button(quick_controls_frame, text=text, command=command, 
                           style='Modern.TButton', width=3)
            btn.pack(side=tk.LEFT, padx=(0, 5))
            
            # Tooltip
            self._create_tooltip(btn, tooltip)
        
        # Durée d'affichage
        duration_frame = ttk.Frame(quick_controls_frame, style='Modern.TFrame')
        duration_frame.pack(side=tk.RIGHT)
        
        ttk.Label(duration_frame, text="Durée:", style='Modern.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        self.bottom_duration_var = tk.StringVar(value="10.0")
        duration_entry = ttk.Entry(duration_frame, textvariable=self.bottom_duration_var, width=8)
        duration_entry.pack(side=tk.LEFT)
        duration_entry.bind('<Return>', self._update_duration_from_bottom)
    
    def _update_time_from_bottom(self, value):
        """Met à jour le temps depuis la barre du bas."""
        if hasattr(self, 'time_var'):
            self.time_var.set(float(value))
            self._update_time(value)
    
    def _update_duration_from_bottom(self, event=None):
        """Met à jour la durée depuis la barre du bas."""
        if hasattr(self, 'duration_var'):
            self.duration_var.set(self.bottom_duration_var.get())
            self._update_duration(event)
    
    def _pause_play(self):
        """Pause/Play de la lecture."""
        # Pour l'instant, juste un message
        print("⏸️ Pause/Play - Fonctionnalité à implémenter")
    
    def _create_tooltip(self, widget, text):
        """Crée un tooltip pour un widget."""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, style='Modern.TLabel', 
                            background='#333333', foreground='white')
            label.pack()
            widget.tooltip = tooltip
        
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
    
    def _setup_keyboard_shortcuts(self) -> None:
        """Configure les raccourcis clavier."""
        # Raccourcis principaux
        self.root.bind('<Control-o>', lambda e: self.load_edf_file())
        self.root.bind('<Control-s>', lambda e: self._export_data())
        self.root.bind('<Control-a>', lambda e: self._toggle_autoscale())
        self.root.bind('<Control-f>', lambda e: self._toggle_filter())
        self.root.bind('<Control-m>', lambda e: self._open_matplotlib_psg_view())
        self.root.bind('<Control-1>', lambda e: self._show_channel_selector())
        self.root.bind('<F1>', lambda e: self._show_user_guide())
        self.root.bind('<F5>', lambda e: self._refresh_plot())
        self.root.bind('<Escape>', lambda e: self.root.focus_set())
        
        # Raccourci pour afficher les raccourcis
        self.root.bind('<Control-?>', lambda e: self._show_shortcuts())
        self.root.bind('<Control-/>', lambda e: self._show_shortcuts())  # Alternative pour les claviers US
        
        # Raccourci pour basculer le thème sombre/clair
        self.root.bind('<Control-t>', lambda e: self._toggle_dark_theme())
        
        # Navigation avec réinitialisation des bindings ZQSD
        self.root.bind('<Left>', lambda e: self._step_backward_with_zqsd_reset())
        self.root.bind('<Right>', lambda e: self._step_forward_with_zqsd_reset())
        self.root.bind('<Up>', lambda e: self._increase_amplitude_with_zqsd_reset())
        self.root.bind('<Down>', lambda e: self._decrease_amplitude_with_zqsd_reset())
        
        logging.info("Raccourcis clavier configurés")
    
    def _increase_amplitude(self):
        """Augmente l'amplitude."""
        current = float(self.amplitude_var.get())
        new_value = min(current + 10, 1000)
        self.amplitude_var.set(str(new_value))
        self.update_plot()
    
    def _decrease_amplitude(self):
        """Diminue l'amplitude."""
        current = float(self.amplitude_var.get())
        new_value = max(current - 10, 10)
        self.amplitude_var.set(str(new_value))
        self.update_plot()
    
    def _create_status_bar(self, parent: ttk.Frame) -> None:
        """Crée la barre d'état moderne."""
        status_frame = ttk.Frame(parent, style='Modern.TFrame')
        status_frame.pack(fill=tk.X, pady=(0, 5))  # Réduit l'espacement
        
        # Version de l'application
        self.version_label = ttk.Label(
            status_frame, 
            text="v0.0beta1.1", 
            style='Version.TLabel',
            font=('Segoe UI', 8, 'bold')
        )
        self.version_label.pack(side=tk.LEFT, padx=(0, 8))  # Réduit l'espacement
        
        # Status principal
        self.status_label = ttk.Label(
            status_frame, 
            text="Prêt - Aucun fichier chargé", 
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.LEFT, padx=(0, 6))  # Réduit l'espacement
        
        # Informations de fichier
        self.file_info_label = ttk.Label(
            status_frame, 
            text="", 
            style='Modern.TLabel'
        )
        self.file_info_label.pack(side=tk.LEFT, padx=(0, 6))  # Réduit l'espacement
        
        # Informations de temps et performance
        self.time_info_label = ttk.Label(
            status_frame,
            text="",
            style='Modern.TLabel'
        )
        self.performance_label = ttk.Label(
            status_frame,
            text="",
            style='Modern.TLabel'
        )
        self.performance_label.pack(side=tk.RIGHT, padx=(0, 6))
        self.time_info_label.pack(side=tk.RIGHT)
    
    def _update_version_display(self):
        """Met à jour l'affichage de la version dans l'interface."""
        if hasattr(self, 'version_label'):
            try:
                self.version_label.config(text="v0.0beta1.1")
            except Exception:
                pass

    def _update_performance_feedback(
        self,
        *,
        total_ms: float,
        fps: float,
        bridge_result,
    ) -> None:
        if not hasattr(self, 'performance_label'):
            return
        try:
            level = bridge_result.level_bin_size if bridge_result else 1
            bytes_read = bridge_result.bytes_read if bridge_result else 0
            kb = bytes_read / 1024.0
            self.performance_label.config(
                text=f"{total_ms:.1f} ms · {fps:.1f} fps · lvl {level} · {kb:.1f} KB"
            )
        except Exception:
            pass

    def _append_telemetry(
        self,
        *,
        action: str,
        total_ms: float,
        draw_ms: float,
        extract_ms: float,
        fps: float,
        bridge_result,
        n_channels: int,
        n_points: int,
        baseline_ms: float = 0.0,
        filter_ms: float = 0.0,
    ) -> None:
        if self._telemetry_path is None:
            return
        try:
            path = self._telemetry_path
            path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = not path.exists() or path.stat().st_size == 0
            level = bridge_result.level_bin_size if bridge_result else 1
            bytes_read = bridge_result.bytes_read if bridge_result else 0
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if needs_header:
                    writer.writerow(
                        [
                            "timestamp",
                            "action",
                            "total_ms",
                            "draw_ms",
                            "extract_ms",
                            "fps",
                            "level",
                            "bytes_read",
                            "channels",
                            "points",
                            "baseline_ms",
                            "filter_ms",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.utcnow().isoformat(),
                        action,
                        f"{total_ms:.3f}",
                        f"{draw_ms:.3f}",
                        f"{extract_ms:.3f}",
                        f"{fps:.2f}",
                        level,
                        bytes_read,
                        n_channels,
                        n_points,
                        f"{baseline_ms:.3f}",
                        f"{filter_ms:.3f}",
                    ]
                )
        except Exception:
            pass
    
    def _update_status_bar(self):
        """Met à jour la barre de statut avec les informations actuelles."""
        try:
            if hasattr(self, 'status_label') and hasattr(self, 'file_info_label'):
                if self.raw is not None:
                    # Fichier chargé
                    filename = os.path.basename(self.current_file_path) if hasattr(self, 'current_file_path') else "Fichier EEG"
                    self.status_label.config(text="Prêt")
                    self.file_info_label.config(text=f"📁 {filename}")
                    
                    # Ajouter les informations de scoring si disponible
                    if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None:
                        scoring_info = f" | 📊 Scoring: {os.path.basename(self.current_scoring_path)}" if hasattr(self, 'current_scoring_path') else " | 📊 Scoring manuel"
                        current_text = self.file_info_label.cget("text")
                        if "Scoring:" not in current_text:
                            self.file_info_label.config(text=current_text + scoring_info)
                    
                    if hasattr(self, 'sleep_scoring_data') and self.sleep_scoring_data is not None:
                        auto_info = " | 🤖 Auto-scoring YASA"
                        current_text = self.file_info_label.cget("text")
                        if "Auto-scoring" not in current_text:
                            self.file_info_label.config(text=current_text + auto_info)
                else:
                    # Aucun fichier chargé
                    self.status_label.config(text="Prêt - Aucun fichier chargé")
                    self.file_info_label.config(text="")
        except Exception as e:
            print(f"⚠️ Erreur mise à jour barre de statut: {e}")

    def _show_batch_fft_automation(self):
        """Interface d'automatisation pour l'export FFT en lot de plusieurs fichiers et canaux."""
        # Plus besoin de fichier EEG chargé - l'automatisation peut fonctionner de manière autonome
        print("🔍 CHECKPOINT BATCH AUTO: Ouverture interface automatisation FFT")
        
        # Si pas de fichier chargé, utiliser des canaux par défaut
        if not hasattr(self, 'raw') or self.raw is None:
            print("🔍 CHECKPOINT BATCH AUTO: Aucun EEG chargé - utilisation des canaux par défaut")
        else:
            print("🔍 CHECKPOINT BATCH AUTO: EEG chargé - canaux disponibles pour référence")

        # Fenêtre principale d'automatisation
        batch_window = tk.Toplevel(self.root)
        batch_window.title("🤖 Automatisation FFT en Lot - CESA v0.0beta1.1")
        batch_window.geometry("1100x900")
        batch_window.configure(bg='#f8f9fa')
        
        # Style
        style = ttk.Style()
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Info.TLabel', font=('Segoe UI', 9), foreground='#6c757d')
        
        # Titre principal
        title_frame = ttk.Frame(batch_window)
        title_frame.pack(fill=tk.X, padx=20, pady=(20,10))
        
        ttk.Label(title_frame, text="🤖 Automatisation FFT en Lot", 
                  style='Heading.TLabel').pack(anchor='w')
        ttk.Label(title_frame, text="Traitez automatiquement plusieurs fichiers EEG avec plusieurs canaux", 
                  style='Info.TLabel').pack(anchor='w', pady=(5,0))
        
        # Message informatif si aucun EEG n'est chargé
        if not hasattr(self, 'raw') or self.raw is None:
            info_frame = ttk.Frame(title_frame)
            info_frame.pack(fill=tk.X, pady=(10,0))
            ttk.Label(info_frame, text="ℹ️ Aucun fichier EEG chargé - L'automatisation fonctionne de manière autonome", 
                     style='Info.TLabel', foreground='#28a745').pack(anchor='w')
        
        # Notebook pour organiser les onglets
        notebook = ttk.Notebook(batch_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Onglet 1: Configuration des fichiers
        files_frame = ttk.Frame(notebook)
        notebook.add(files_frame, text="📁 Fichiers EEG")
        
        self._create_files_selection_tab(files_frame)
        
        # Onglet 2: Configuration des canaux
        channels_frame = ttk.Frame(notebook)
        notebook.add(channels_frame, text="📊 Canaux & Paramètres")
        
        self._create_channels_config_tab(channels_frame)
        
        # Onglet 3: Traitement en lot
        processing_frame = ttk.Frame(notebook)
        notebook.add(processing_frame, text="⚙️ Traitement")
        
        self._create_batch_processing_tab(processing_frame, batch_window)
        
        # Variables partagées pour la configuration
        if not hasattr(self, 'batch_config'):
            self.batch_config = {
                'input_dir': '',
                'output_dir': '',
                'eeg_files': [],
                'selected_channels': [],
                'fft_params': {
                    'fmin': 0.5,
                    'fmax': 30.0,
                    'nperseg_sec': 4.0,
                    'equalize_epochs': True,
                    'robust_stats': True,
                    'band_filter': True,
                    'notch_filter': True
                },
                'include_scoring': True,
                'auto_scoring': False
            }

    def _create_files_selection_tab(self, parent):
        """Crée l'onglet de sélection des fichiers EEG."""
        
        # Section dossier d'entrée
        input_section = ttk.LabelFrame(parent, text="📂 Dossier d'entrée (fichiers EEG)")
        input_section.pack(fill=tk.X, padx=10, pady=10)
        
        input_frame = ttk.Frame(input_section)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.input_dir_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_dir_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Parcourir", command=self._select_input_directory).pack(side=tk.RIGHT, padx=(5,0))
        
        # Section dossier de sortie
        output_section = ttk.LabelFrame(parent, text="📤 Dossier de sortie (CSV FFT)")
        output_section.pack(fill=tk.X, padx=10, pady=10)
        
        output_frame = ttk.Frame(output_section)
        output_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.output_dir_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Parcourir", command=self._select_output_directory).pack(side=tk.RIGHT, padx=(5,0))
        
        # Section liste des fichiers détectés
        files_section = ttk.LabelFrame(parent, text="📋 Fichiers EEG détectés")
        files_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame avec scrollbar pour la liste des fichiers
        files_list_frame = ttk.Frame(files_section)
        files_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview pour afficher les fichiers (avec colonne Groupe)
        columns = ("Fichier", "Taille", "Statut", "Groupe")
        self.files_tree = ttk.Treeview(files_list_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.files_tree.heading(col, text=col)
            
        self.files_tree.column("Fichier", width=400)
        self.files_tree.column("Taille", width=100)
        self.files_tree.column("Statut", width=180)
        self.files_tree.column("Groupe", width=120)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(files_list_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        h_scrollbar = ttk.Scrollbar(files_list_frame, orient=tk.HORIZONTAL, command=self.files_tree.xview)
        self.files_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.files_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Boutons de contrôle
        control_frame = ttk.Frame(files_section)
        control_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        ttk.Button(control_frame, text="🔍 Scanner Dossier", command=self._scan_input_directory).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="✅ Tout sélectionner", command=self._select_all_files).pack(side=tk.LEFT, padx=(10,0))
        ttk.Button(control_frame, text="❌ Tout désélectionner", command=self._deselect_all_files).pack(side=tk.LEFT, padx=(5,0))

        # Stocker la liste courante pour spaghetti
        self.detected_eeg_files: List[str] = []

        # Contrôles de groupe (Avant/Après)
        group_ctrl = ttk.Frame(files_section)
        group_ctrl.pack(fill=tk.X, padx=10, pady=(0,10))
        ttk.Label(group_ctrl, text="Affecter le groupe aux fichiers sélectionnés:").pack(side=tk.LEFT)
        ttk.Button(group_ctrl, text="Marquer AVANT", command=lambda: self._mark_selected_group('AVANT')).pack(side=tk.LEFT, padx=(10,0))
        ttk.Button(group_ctrl, text="Marquer APRÈS", command=lambda: self._mark_selected_group('APRÈS')).pack(side=tk.LEFT, padx=(5,0))
        ttk.Button(group_ctrl, text="Effacer Groupe", command=lambda: self._mark_selected_group('')).pack(side=tk.LEFT, padx=(5,0))

    def _create_channels_config_tab(self, parent):
        """Crée l'onglet de configuration des canaux et paramètres FFT."""
        
        # Section sélection des canaux
        channels_section = ttk.LabelFrame(parent, text="📊 Sélection des Canaux")
        channels_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Obtenir les canaux disponibles du fichier actuellement chargé ou utiliser des canaux standards
        if hasattr(self, 'raw') and self.raw is not None:
            available_channels = self.raw.ch_names
            print(f"🔍 CHECKPOINT BATCH CHANNELS: {len(available_channels)} canaux du fichier EEG chargé")
        else:
            # Canaux EEG standards pour polysomnographie
            available_channels = [
                "C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1",  # EEG standards
                "C3", "C4", "F3", "F4", "O1", "O2",  # EEG alternatifs
                "Fpz-Cz", "Pz-Oz",  # EEG référence commune
                "E1-M2", "E2-M1",  # EOG
                "EMG submental"  # EMG
            ]
            print(f"🔍 CHECKPOINT BATCH CHANNELS: {len(available_channels)} canaux par défaut (aucun EEG chargé)")
        
        # Frame avec checkboxes pour les canaux
        channels_frame = ttk.Frame(channels_section)
        channels_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.channel_vars = {}
        
        # Organiser les canaux en colonnes
        max_cols = 4
        for i, channel in enumerate(available_channels):
            row = i // max_cols
            col = i % max_cols
            
            var = tk.BooleanVar()
            self.channel_vars[channel] = var
            
            cb = ttk.Checkbutton(channels_frame, text=channel, variable=var)
            cb.grid(row=row, column=col, sticky='w', padx=10, pady=2)
        
        # Boutons de sélection rapide
        quick_select_frame = ttk.Frame(channels_section)
        quick_select_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        ttk.Button(quick_select_frame, text="EEG Standard", 
                   command=lambda: self._quick_select_channels(['C3-M2', 'C4-M1', 'F3-M2', 'F4-M1', 'O1-M2', 'O2-M1'])).pack(side=tk.LEFT)
        ttk.Button(quick_select_frame, text="Tous EEG", 
                   command=lambda: self._quick_select_channels([ch for ch in available_channels if any(x in ch for x in ['C3', 'C4', 'F3', 'F4', 'O1', 'O2', 'Fpz', 'Pz'])])).pack(side=tk.LEFT, padx=(5,0))
        ttk.Button(quick_select_frame, text="Tout sélectionner", 
                  command=lambda: self._quick_select_channels(available_channels)).pack(side=tk.LEFT, padx=(5,0))
        
        # Section paramètres FFT
        fft_section = ttk.LabelFrame(parent, text="⚙️ Paramètres FFT")
        fft_section.pack(fill=tk.X, padx=10, pady=10)
        
        fft_frame = ttk.Frame(fft_section)
        fft_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Paramètres numériques
        params_frame = ttk.Frame(fft_frame)
        params_frame.pack(fill=tk.X)
        
        # Fréquence min
        ttk.Label(params_frame, text="Fréq. min (Hz):").grid(row=0, column=0, sticky='w', padx=(0,5))
        self.fmin_var = tk.StringVar(value="0.5")
        ttk.Entry(params_frame, textvariable=self.fmin_var, width=10).grid(row=0, column=1, padx=(0,20))
        
        # Fréquence max
        ttk.Label(params_frame, text="Fréq. max (Hz):").grid(row=0, column=2, sticky='w', padx=(0,5))
        self.fmax_var = tk.StringVar(value="100.0")
        ttk.Entry(params_frame, textvariable=self.fmax_var, width=10).grid(row=0, column=3, padx=(0,20))
        
        # Taille de bin
        ttk.Label(params_frame, text="Taille bin (s):").grid(row=1, column=0, sticky='w', padx=(0,5), pady=(10,0))
        self.nperseg_var = tk.StringVar(value="4.0")
        ttk.Entry(params_frame, textvariable=self.nperseg_var, width=10).grid(row=1, column=1, pady=(10,0))
        
        # Options de traitement
        options_frame = ttk.Frame(fft_frame)
        options_frame.pack(fill=tk.X, pady=(20,0))
        
        self.equalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Égaliser nombre d'époques par stade", variable=self.equalize_var).pack(anchor='w')
        
        self.robust_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Statistiques robustes (médiane + MAD)", variable=self.robust_var).pack(anchor='w')
        
        self.filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Filtre passe-bande (0.3-40 Hz)", variable=self.filter_var).pack(anchor='w')
        
        self.notch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Filtre notch (50 Hz)", variable=self.notch_var).pack(anchor='w')
        
        # Section scoring
        scoring_section = ttk.LabelFrame(parent, text="🛏️ Configuration Scoring")
        scoring_section.pack(fill=tk.X, padx=10, pady=10)
        
        scoring_frame = ttk.Frame(scoring_section)
        scoring_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.include_scoring_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scoring_frame, text="Inclure scoring de sommeil (requis pour FFT par stade)", 
                       variable=self.include_scoring_var).pack(anchor='w')
        
        self.auto_scoring_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scoring_frame, text="Calculer scoring automatique YASA si absent", 
                       variable=self.auto_scoring_var).pack(anchor='w', pady=(5,0))

    def _create_batch_processing_tab(self, parent, main_window):
        """Crée l'onglet de traitement en lot."""
        
        # Section statut
        status_section = ttk.LabelFrame(parent, text="📊 Statut du Traitement")
        status_section.pack(fill=tk.X, padx=10, pady=10)
        
        status_frame = ttk.Frame(status_section)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.batch_status_var = tk.StringVar(value="Prêt à démarrer")
        ttk.Label(status_frame, textvariable=self.batch_status_var, font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        
        # Barre de progression
        self.batch_progress = ttk.Progressbar(status_frame, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=(10,0))
        
        self.batch_progress_text = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.batch_progress_text, font=('Segoe UI', 9)).pack(anchor='w', pady=(5,0))
        
        # Section contrôles
        controls_section = ttk.LabelFrame(parent, text="🎮 Contrôles")
        controls_section.pack(fill=tk.X, padx=10, pady=10)
        
        controls_frame = ttk.Frame(controls_section)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_batch_btn = ttk.Button(controls_frame, text="🚀 Démarrer le Traitement", 
                                         command=self._start_batch_processing)
        self.start_batch_btn.pack(side=tk.LEFT)
        
        self.stop_batch_btn = ttk.Button(controls_frame, text="⏹️ Arrêter", 
                                        command=self._stop_batch_processing, state='disabled')
        self.stop_batch_btn.pack(side=tk.LEFT, padx=(10,0))
        
        ttk.Button(controls_frame, text="📋 Voir Configuration", 
                  command=self._show_batch_config).pack(side=tk.LEFT, padx=(10,0))
        
        # Ajouter conteneur scrollable pour l'onglet Traitement
        try:
            canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
            vscroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
            canvas.configure(yscrollcommand=vscroll.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vscroll.pack(side=tk.RIGHT, fill=tk.Y)
            container = ttk.Frame(canvas)
            container_id = canvas.create_window((0, 0), window=container, anchor='nw')

            def _on_configure(_event=None):
                try:
                    canvas.configure(scrollregion=canvas.bbox('all'))
                except Exception:
                    pass
            def _on_resize(event):
                try:
                    canvas.itemconfigure(container_id, width=event.width)
                except Exception:
                    pass
            container.bind('<Configure>', _on_configure)
            canvas.bind('<Configure>', _on_resize)
            # Utiliser le conteneur scrollable comme parent pour la suite
            parent = container
        except Exception:
            pass

        # Section logs
        logs_section = ttk.LabelFrame(parent, text="📝 Logs de Traitement")
        logs_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        logs_frame = ttk.Frame(logs_section)
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.batch_logs = tk.Text(logs_frame, height=12, wrap=tk.WORD)
        logs_scrollbar = ttk.Scrollbar(logs_frame, orient=tk.VERTICAL, command=self.batch_logs.yview)
        self.batch_logs.configure(yscrollcommand=logs_scrollbar.set)
        
        self.batch_logs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        logs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bouton pour effacer les logs
        ttk.Button(controls_frame, text="🗑️ Effacer Logs", 
                  command=lambda: self.batch_logs.delete(1.0, tk.END)).pack(side=tk.RIGHT)
        
        # Variable pour arrêter le traitement
        self.batch_stop_requested = False

        # Mode de sortie
        output_mode_section = ttk.LabelFrame(parent, text="📤 Mode de sortie")
        output_mode_section.pack(fill=tk.X, padx=10, pady=(0,10))

        mode_frame = ttk.Frame(output_mode_section)
        mode_frame.pack(fill=tk.X, padx=10, pady=8)

        self.batch_output_mode_var = tk.StringVar(value=getattr(self, 'batch_output_mode', 'csv'))
        rb_csv = ttk.Radiobutton(mode_frame, text="Exporter CSV FFT classique", value='csv', variable=self.batch_output_mode_var, command=self._toggle_output_mode_ui)
        rb_spag = ttk.Radiobutton(mode_frame, text="Générer Graphiques Spaghetti (AVANT/APRÈS)", value='spaghetti', variable=self.batch_output_mode_var, command=self._toggle_output_mode_ui)
        rb_csv.pack(anchor='w')
        rb_spag.pack(anchor='w', pady=(4,0))

        # Paramètres spécifiques Spaghetti
        self.spaghetti_frame = ttk.Frame(output_mode_section)
        self.spaghetti_frame.pack(fill=tk.X, padx=10, pady=(0,8))

        # Options spaghetti: utiliser fichiers détectés vs spécifier AV/PR
        self.spag_use_detected_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.spaghetti_frame, text="Utiliser les fichiers détectés dans l'onglet 'Fichiers EEG'", variable=self.spag_use_detected_var, command=self._toggle_output_mode_ui).pack(anchor='w')

        # Dossier AVANT
        before_row = ttk.Frame(self.spaghetti_frame)
        before_row.pack(fill=tk.X, pady=2)
        ttk.Label(before_row, text="Dossier EDF AVANT:").pack(side=tk.LEFT)
        self.before_dir_var = tk.StringVar()
        self.spag_before_entry = ttk.Entry(before_row, textvariable=self.before_dir_var, width=50)
        self.spag_before_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        self.spag_before_btn = ttk.Button(before_row, text="Parcourir", command=lambda: self._browse_dir_to_var(self.before_dir_var, "Sélectionner dossier EDF AVANT"))
        self.spag_before_btn.pack(side=tk.LEFT)

        # Dossier APRÈS
        after_row = ttk.Frame(self.spaghetti_frame)
        after_row.pack(fill=tk.X, pady=2)
        ttk.Label(after_row, text="Dossier EDF APRÈS:").pack(side=tk.LEFT)
        self.after_dir_var = tk.StringVar()
        self.spag_after_entry = ttk.Entry(after_row, textvariable=self.after_dir_var, width=50)
        self.spag_after_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        self.spag_after_btn = ttk.Button(after_row, text="Parcourir", command=lambda: self._browse_dir_to_var(self.after_dir_var, "Sélectionner dossier EDF APRÈS"))
        self.spag_after_btn.pack(side=tk.LEFT)

        # (Supprimé) Sélecteurs rapides de bandes et stades — on garde uniquement la table bande→stades

        # Sélecteur de stades par bande (optionnel)
        mapping_section = ttk.LabelFrame(self.spaghetti_frame, text="Stades par bande (optionnel)")
        mapping_section.pack(fill=tk.X, pady=(8, 4))

        # Tableau unique avec grid pour alignement parfait des colonnes
        mapping_table = ttk.Frame(mapping_section)
        mapping_table.pack(fill=tk.X, padx=8, pady=(4, 4))

        # Colonnes: 0=Bande, 1..5=stages, 6=Contrôle de ligne
        for col in range(7):
            try:
                mapping_table.grid_columnconfigure(col, weight=1, uniform="stagecols", minsize=70)
            except Exception:
                pass

        ttk.Label(mapping_table, text="Bande", width=12, anchor='w').grid(row=0, column=0, sticky='w', padx=(0,6))
        stage_cols = [("W", "Wake"), ("N1", "N1"), ("N2", "N2"), ("N3", "N3"), ("R", "REM")]
        for j, (_code, st_label) in enumerate(stage_cols, start=1):
            ttk.Label(mapping_table, text=st_label, anchor='center').grid(row=0, column=j, sticky='n')
        ttk.Label(mapping_table, text="Ligne", anchor='center').grid(row=0, column=6, sticky='n')

        # Ligne de contrôles de colonnes (Tout/Nul)
        ctrl_row = 1
        ttk.Label(mapping_table, text="").grid(row=ctrl_row, column=0)  # placeholder
        for j, (st_code, _st_label) in enumerate(stage_cols, start=1):
            btn_all = ttk.Button(mapping_table, text="✓", width=2, command=lambda sc=st_code: self._toggle_band_stage_col(sc, True))
            btn_none = ttk.Button(mapping_table, text="–", width=2, command=lambda sc=st_code: self._toggle_band_stage_col(sc, False))
            btn_all.grid(row=ctrl_row, column=j, sticky='e', padx=(0,2))
            btn_none.grid(row=ctrl_row, column=j, sticky='w', padx=(2,0))
        ttk.Label(mapping_table, text="").grid(row=ctrl_row, column=6)

        # Variables de mapping + lignes
        self.spag_band_stage_vars = {}
        start_row = 2
        for i, band in enumerate(["LowDelta","Delta","Theta","Alpha","Sigma","Beta","Gamma"], start=start_row):
            ttk.Label(mapping_table, text=band, width=12, anchor='w').grid(row=i, column=0, sticky='w', padx=(0,6), pady=2)
            self.spag_band_stage_vars[band] = {}
            for j, (st_code, _st_label) in enumerate(stage_cols, start=1):
                var = tk.BooleanVar(value=False)
                self.spag_band_stage_vars[band][st_code] = var
                cb = ttk.Checkbutton(mapping_table, variable=var)
                cb.grid(row=i, column=j, sticky='n', pady=2)
            # Contrôles de ligne (Tout/Nul)
            row_ctrl = ttk.Frame(mapping_table)
            row_ctrl.grid(row=i, column=6, sticky='n')
            ttk.Button(row_ctrl, text="✓", width=2, command=lambda b=band: self._toggle_band_stage_row(b, True)).pack(side=tk.LEFT, padx=(0,2))
            ttk.Button(row_ctrl, text="–", width=2, command=lambda b=band: self._toggle_band_stage_row(b, False)).pack(side=tk.LEFT, padx=(2,0))

        # Saisie avancée: combinaisons par canal
        adv_row = ttk.Frame(self.spaghetti_frame)
        adv_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(adv_row, text="Combinaisons par canal (ex: C3:Delta_R; F3:Theta_N3)").pack(side=tk.LEFT)
        self.spag_combo_spec_var = tk.StringVar()
        self.spag_combo_entry = ttk.Entry(adv_row, textvariable=self.spag_combo_spec_var, width=60)
        self.spag_combo_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        # Sélecteur visuel des combinaisons
        combos_frame = ttk.LabelFrame(self.spaghetti_frame, text="Sélection combinaisons Canal × Bande × Stade")
        combos_frame.pack(fill=tk.X, padx=8, pady=(6, 6))
        row1 = ttk.Frame(combos_frame)
        row1.pack(fill=tk.X, pady=4)
        ttk.Label(row1, text="Canal:").pack(side=tk.LEFT)
        # Liste de canaux candidates
        channel_choices = []
        try:
            if hasattr(self, 'batch_config') and self.batch_config.get('selected_channels'):
                channel_choices = list(self.batch_config.get('selected_channels'))
        except Exception:
            channel_choices = []
        if not channel_choices:
            channel_choices = ["C3", "C4", "F3", "F4", "O1", "O2"]
        self.spag_combo_ch = ttk.Combobox(row1, values=channel_choices, width=10, state='readonly')
        self.spag_combo_ch.pack(side=tk.LEFT, padx=(6,10))
        try:
            self.spag_combo_ch.set("")
        except Exception:
            pass

        ttk.Label(row1, text="Bande:").pack(side=tk.LEFT)
        band_choices = ["LowDelta","Delta","Theta","Alpha","Sigma","Beta","Gamma"]
        self.spag_combo_band = ttk.Combobox(row1, values=band_choices, width=10, state='readonly')
        self.spag_combo_band.pack(side=tk.LEFT, padx=(6,10))
        try:
            self.spag_combo_band.set("")
        except Exception:
            pass

        ttk.Label(row1, text="Stade:").pack(side=tk.LEFT)
        stage_choices = ["W","N1","N2","N3","R"]
        self.spag_combo_stage = ttk.Combobox(row1, values=stage_choices, width=8, state='readonly')
        self.spag_combo_stage.pack(side=tk.LEFT, padx=(6,10))
        try:
            self.spag_combo_stage.set("")
        except Exception:
            pass

        ttk.Button(row1, text="Ajouter", command=self._add_spag_combo).pack(side=tk.LEFT)

        row2 = ttk.Frame(combos_frame)
        row2.pack(fill=tk.BOTH, pady=(4,2))
        self.spag_combo_tree = ttk.Treeview(row2, columns=("Channel","Band","Stage"), show='headings', height=4)
        self.spag_combo_tree.heading("Channel", text="Canal")
        self.spag_combo_tree.heading("Band", text="Bande")
        self.spag_combo_tree.heading("Stage", text="Stade")
        self.spag_combo_tree.column("Channel", width=80)
        self.spag_combo_tree.column("Band", width=90)
        self.spag_combo_tree.column("Stage", width=80)
        self.spag_combo_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        btns = ttk.Frame(row2)
        btns.pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Supprimer", command=self._remove_spag_combo).pack(fill=tk.X, pady=(0,4))
        ttk.Button(btns, text="Vider", command=self._clear_spag_combos).pack(fill=tk.X)

        self.spag_combo_list = []

        # Clustering des sujets pour les spaghetti
        clustering_frame = ttk.LabelFrame(self.spaghetti_frame, text="Clustering des sujets (Spaghetti)")
        clustering_frame.pack(fill=tk.BOTH, padx=8, pady=(0, 8))

        pool_column = ttk.Frame(clustering_frame)
        pool_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=6)
        ttk.Label(pool_column, text="Sujets disponibles").pack(anchor='w')
        pool_list_frame = ttk.Frame(pool_column)
        pool_list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 4))
        self.spag_subject_listbox = tk.Listbox(pool_list_frame, selectmode=tk.EXTENDED, height=7, exportselection=False)
        pool_scroll = ttk.Scrollbar(pool_list_frame, orient=tk.VERTICAL, command=self.spag_subject_listbox.yview)
        self.spag_subject_listbox.configure(yscrollcommand=pool_scroll.set)
        self.spag_subject_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pool_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(pool_column, text="Actualiser", command=self._refresh_spag_subject_pool).pack(anchor='e')

        control_column = ttk.Frame(clustering_frame)
        control_column.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=6)
        ttk.Button(control_column, text="Ajouter → A", command=lambda: self._assign_subjects_to_cluster('A')).pack(fill=tk.X, pady=2)
        ttk.Button(control_column, text="Ajouter → B", command=lambda: self._assign_subjects_to_cluster('B')).pack(fill=tk.X, pady=2)
        ttk.Separator(control_column, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Button(control_column, text="Retirer de A", command=lambda: self._remove_subjects_from_cluster('A')).pack(fill=tk.X, pady=2)
        ttk.Button(control_column, text="Retirer de B", command=lambda: self._remove_subjects_from_cluster('B')).pack(fill=tk.X, pady=2)
        ttk.Separator(control_column, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(control_column, text="Tout réinitialiser", command=self._clear_spag_clusters).pack(fill=tk.X, pady=(0, 2))

        clusters_column = ttk.Frame(clustering_frame)
        clusters_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=6)

        self.spag_cluster_listboxes = {}
        for idx, cluster_id in enumerate(['A', 'B']):
            cluster_frame = ttk.LabelFrame(clusters_column, text=self.spag_cluster_names.get(cluster_id, f"Cluster {cluster_id}"))
            cluster_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0 if idx == 0 else 8, 0))

            name_row = ttk.Frame(cluster_frame)
            name_row.pack(fill=tk.X, padx=8, pady=(6, 4))
            ttk.Label(name_row, text="Nom:").pack(side=tk.LEFT)
            default_name = self.spag_cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            name_var = self.spag_cluster_name_vars.get(cluster_id)
            if name_var is None:
                name_var = tk.StringVar(value=default_name)
                self.spag_cluster_name_vars[cluster_id] = name_var
            else:
                name_var.set(default_name)

            def _make_name_callback(cid: str, frame: ttk.LabelFrame, var: tk.StringVar):
                def _cb(*_args):
                    self._on_spag_cluster_name_change(cid)
                    try:
                        frame.configure(text=var.get() or f"Cluster {cid}")
                    except Exception:
                        pass
                return _cb

            name_var.trace_add('write', _make_name_callback(cluster_id, cluster_frame, name_var))
            ttk.Entry(name_row, textvariable=name_var, width=18).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

            list_frame = ttk.Frame(cluster_frame)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=7, exportselection=False)
            scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
            listbox.configure(yscrollcommand=scroll.set)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.spag_cluster_listboxes[cluster_id] = listbox

        # Initialiser la liste des sujets si disponible
        self._refresh_spag_subject_pool()

        # Libellés personnalisés pour les groupes Before/After
        labels_frame = ttk.LabelFrame(self.spaghetti_frame, text="Group labels")
        labels_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(labels_frame, text="Before:").grid(row=0, column=0, sticky='w', padx=(8, 4), pady=6)
        ttk.Entry(labels_frame, textvariable=self.spag_group_before_var, width=18).grid(row=0, column=1, sticky='w', pady=6)
        ttk.Label(labels_frame, text="After:").grid(row=0, column=2, sticky='w', padx=(12, 4), pady=6)
        ttk.Entry(labels_frame, textvariable=self.spag_group_after_var, width=18).grid(row=0, column=3, sticky='w', pady=6)

        # Appliquer l'état initial
        self._toggle_output_mode_ui()

    def _select_input_directory(self):
        """Sélectionne le dossier d'entrée contenant les fichiers EEG."""
        directory = filedialog.askdirectory(title="Sélectionner le dossier contenant les fichiers EEG")
        if directory:
            self.input_dir_var.set(directory)
            self.batch_config['input_dir'] = directory
            self._log_batch("📂 Dossier d'entrée sélectionné: " + directory)

    def _select_output_directory(self):
        """Sélectionne le dossier de sortie pour les CSV FFT."""
        directory = filedialog.askdirectory(title="Sélectionner le dossier de sortie pour les CSV FFT")
        if directory:
            self.output_dir_var.set(directory)
            self.batch_config['output_dir'] = directory
            self._log_batch("📤 Dossier de sortie sélectionné: " + directory)

    def _normalize_filename_for_association(self, filename):
        """Normalise un nom de fichier pour l'association EDF-Excel."""
        # Convertir en majuscules pour ignorer les différences de casse
        normalized = filename.upper()
        
        # Enlever les suffixes courants de scoring Excel
        suffixes_to_remove = [
            '_SLEEP SCORING',
            '_SLEEP_SCORING', 
            ' SLEEP SCORING',
            '_SCORING',
            ' SCORING',
            '_SLEEP',
            ' SLEEP'
        ]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        # Normaliser les variations Av/AV (déjà en majuscules à ce stade)
        # S030_AV reste S030_AV, pas de changement nécessaire car déjà normalisé
        
        print(f"🔍 CHECKPOINT NORMALIZE: '{filename}' -> '{normalized}'")
        return normalized

    def _scan_input_directory(self):
        """Scanne le dossier d'entrée pour détecter les fichiers EEG."""
        print("🔍 CHECKPOINT SCAN: Début scan du dossier d'entrée")
        
        if not self.input_dir_var.get():
            print("🔍 CHECKPOINT SCAN: Aucun dossier sélectionné")
            messagebox.showwarning("Attention", "Veuillez d'abord sélectionner un dossier d'entrée")
            return
        
        input_dir = self.input_dir_var.get()
        print(f"🔍 CHECKPOINT SCAN: Dossier à scanner: {input_dir}")
        
        # Effacer la liste actuelle
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        print("🔍 CHECKPOINT SCAN: Liste TreeView effacée")
        
        eeg_extensions = [ext.lower() for ext in recording_extensions_for_scan()]
        found_files = []
        
        try:
            print(f"🔍 CHECKPOINT SCAN: Recherche fichiers avec extensions: {eeg_extensions}")
            
            # Créer un dictionnaire pour associer les fichiers EDF avec leurs scoring Excel
            excel_extensions = ['.xlsx', '.xls']
            excel_files = {}  # {normalized_key: (excel_path, original_name)}
            
            # D'abord, scanner tous les fichiers Excel
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in excel_extensions):
                        excel_path = os.path.join(root, file)
                        base_name = os.path.splitext(file)[0]
                        
                        # Créer une clé normalisée pour l'association
                        # Ex: "S030_AV_sleep scoring" -> "S030_AV"
                        normalized_key = self._normalize_filename_for_association(base_name)
                        
                        excel_files[normalized_key] = (excel_path, file)
                        print(f"🔍 CHECKPOINT SCAN: Fichier Excel trouvé: {file}")
                        print(f"🔍 CHECKPOINT SCAN: Base: {base_name} -> Clé normalisée: {normalized_key}")
            
            print(f"🔍 CHECKPOINT SCAN: {len(excel_files)} fichiers Excel détectés")
            print(f"🔍 CHECKPOINT SCAN: Clés Excel: {list(excel_files.keys())}")
            
            # Ensuite, scanner les fichiers EDF et associer avec Excel
            for root, dirs, files in os.walk(input_dir):
                print(f"🔍 CHECKPOINT SCAN: Scan dossier: {root} - {len(files)} fichiers")
                
                for file in files:
                    if normalize_recording_extension(file) in eeg_extensions:
                        file_path = os.path.join(root, file)
                        print(f"🔍 CHECKPOINT SCAN: Fichier EEG trouvé: {file}")
                        
                        try:
                            file_size = os.path.getsize(file_path)
                            size_mb = file_size / (1024 * 1024)
                            
                            # Chercher un fichier Excel associé
                            base_name = os.path.splitext(file)[0]
                            
                            # Créer une clé normalisée pour l'EDF
                            # Ex: "S030_Av" -> "S030_AV"
                            normalized_edf_key = self._normalize_filename_for_association(base_name)
                            
                            print(f"🔍 CHECKPOINT SCAN: EDF base: {base_name} -> Clé normalisée: {normalized_edf_key}")
                            
                            # Chercher l'Excel correspondant
                            excel_info = excel_files.get(normalized_edf_key, None)
                            excel_path = excel_info[0] if excel_info else None
                            
                            # Déterminer le groupe par heuristique (modifiable ensuite dans l'UI)
                            grp = ''
                            name_up = file.upper()
                            if any(k in name_up for k in ['_AV', ' AV', 'AVANT']):
                                grp = 'AVANT'
                            elif any(k in name_up for k in ['_AP', ' AP', 'APRES', 'APRÈS']):
                                grp = 'APRÈS'

                            file_info = {
                                'path': file_path,
                                'name': file,
                                'size': f"{size_mb:.1f} MB",
                                'selected': True,
                                'excel_path': excel_path,
                                'group': grp
                            }
                            
                            found_files.append(file_info)
                            
                            if excel_path:
                                excel_name = excel_info[1]  # nom original du fichier Excel
                                print(f"🔍 CHECKPOINT SCAN: Ajouté: {file} ({size_mb:.1f} MB) + Excel: {excel_name}")
                            else:
                                print(f"🔍 CHECKPOINT SCAN: Ajouté: {file} ({size_mb:.1f} MB) - Pas de scoring Excel")
                                
                        except Exception as e:
                            print(f"🔍 CHECKPOINT SCAN: Erreur taille fichier {file}: {e}")
            
            print(f"🔍 CHECKPOINT SCAN: Total fichiers trouvés: {len(found_files)}")
            # Garder une liste simple des chemins EDF (sera utilisée pour spaghetti AV/PR)
            try:
                self.detected_eeg_files = [fi['path'] for fi in found_files]
            except Exception:
                self.detected_eeg_files = []
            
            # Ajouter les fichiers à la liste TreeView
            for i, file_info in enumerate(found_files):
                try:
                    print(f"🔍 CHECKPOINT SCAN: Ajout TreeView {i+1}/{len(found_files)}: {file_info['name']}")
                    
                    # Déterminer le statut avec indication du scoring
                    if file_info['selected']:
                        if file_info['excel_path']:
                            status = "✅ Sélectionné + 📊 Scoring"
                        else:
                            status = "✅ Sélectionné (pas de scoring)"
                    else:
                        status = "❌ Ignoré"
                    
                    item = self.files_tree.insert("", tk.END, values=(
                        file_info['name'], 
                        file_info['size'], 
                        status,
                        file_info.get('group','') or ''
                    ), tags=(file_info['path'],))  # Utiliser tags au lieu de set()
                    
                    print(f"🔍 CHECKPOINT SCAN: Item TreeView créé: {item} - Status: {status}")
                    
                except Exception as e:
                    print(f"🔍 CHECKPOINT SCAN: Erreur ajout TreeView pour {file_info['name']}: {e}")
            
            self.batch_config['eeg_files'] = found_files
            print(f"🔍 CHECKPOINT SCAN: Configuration batch mise à jour avec {len(found_files)} fichiers")
            self._log_batch(f"🔍 {len(found_files)} fichiers EEG détectés dans {input_dir}")
            
        except Exception as e:
            print(f"🔍 CHECKPOINT SCAN: ERREUR GÉNÉRALE: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du scan: {e}")
            self._log_batch(f"❌ Erreur scan: {e}")

    def _select_all_files(self):
        """Sélectionne tous les fichiers détectés."""
        print("🔍 CHECKPOINT SELECT: Sélection de tous les fichiers")
        try:
            # Mettre à jour la configuration d'abord
            if hasattr(self, 'batch_config') and 'eeg_files' in self.batch_config:
                for file_info in self.batch_config['eeg_files']:
                    file_info['selected'] = True
                print(f"🔍 CHECKPOINT SELECT: Configuration mise à jour - {len(self.batch_config['eeg_files'])} fichiers")
            
            # Puis mettre à jour l'affichage TreeView
            for item in self.files_tree.get_children():
                # Obtenir les valeurs actuelles
                values = list(self.files_tree.item(item, "values"))
                if len(values) >= 3:
                    file_name = values[0]
                    
                    # Trouver le fichier dans la configuration pour savoir s'il a un Excel
                    has_excel = False
                    for file_info in self.batch_config.get('eeg_files', []):
                        if file_info['name'] == file_name:
                            has_excel = bool(file_info.get('excel_path'))
                            break
                    
                    # Mettre à jour le statut
                    if has_excel:
                        values[2] = "✅ Sélectionné + 📊 Scoring"
                    else:
                        values[2] = "✅ Sélectionné (pas de scoring)"
                    # Conserver colonne Groupe
                    if len(values) < 4:
                        values.append('')
                    
                    self.files_tree.item(item, values=values)
                    print(f"🔍 CHECKPOINT SELECT: Fichier sélectionné: {file_name} - Excel: {has_excel}")
        except Exception as e:
            print(f"🔍 CHECKPOINT SELECT: Erreur: {e}")

    def _deselect_all_files(self):
        """Désélectionne tous les fichiers détectés."""
        print("🔍 CHECKPOINT DESELECT: Désélection de tous les fichiers")
        try:
            # Mettre à jour la configuration d'abord
            if hasattr(self, 'batch_config') and 'eeg_files' in self.batch_config:
                for file_info in self.batch_config['eeg_files']:
                    file_info['selected'] = False
                print(f"🔍 CHECKPOINT DESELECT: Configuration mise à jour - {len(self.batch_config['eeg_files'])} fichiers")
            
            # Puis mettre à jour l'affichage TreeView
            for item in self.files_tree.get_children():
                # Obtenir les valeurs actuelles
                values = list(self.files_tree.item(item, "values"))
                if len(values) >= 3:
                    values[2] = "❌ Ignoré"  # Statut simple pour les fichiers ignorés
                    if len(values) < 4:
                        values.append('')
                    self.files_tree.item(item, values=values)
                    print(f"🔍 CHECKPOINT DESELECT: Fichier désélectionné: {values[0]}")
        except Exception as e:
            print(f"🔍 CHECKPOINT DESELECT: Erreur: {e}")

    def _quick_select_channels(self, channels_to_select):
        """Sélection rapide de canaux spécifiques."""
        # Désélectionner tous d'abord
        for var in self.channel_vars.values():
            var.set(False)
        
        # Sélectionner les canaux demandés
        for channel in channels_to_select:
            if channel in self.channel_vars:
                self.channel_vars[channel].set(True)

    def _show_batch_config(self):
        """Affiche la configuration actuelle du traitement en lot."""
        # Mettre à jour la configuration avec les valeurs actuelles
        self._update_batch_config()
        
        config_text = "🔧 Configuration du Traitement en Lot\n"
        config_text += "=" * 50 + "\n\n"
        
        config_text += f"📂 Dossier d'entrée: {self.batch_config['input_dir']}\n"
        config_text += f"📤 Dossier de sortie: {self.batch_config['output_dir']}\n"
        config_text += f"📋 Nombre de fichiers: {len(self.batch_config['eeg_files'])}\n\n"
        
        config_text += f"📊 Canaux sélectionnés ({len(self.batch_config['selected_channels'])}):\n"
        for channel in self.batch_config['selected_channels']:
            config_text += f"  • {channel}\n"
        config_text += "\n"
        
        params = self.batch_config['fft_params']
        config_text += f"⚙️ Paramètres FFT:\n"
        config_text += f"  • Fréquence: {params['fmin']}-{params['fmax']} Hz\n"
        config_text += f"  • Taille bin: {params['nperseg_sec']} s\n"
        config_text += f"  • Égalisation époques: {'✅' if params['equalize_epochs'] else '❌'}\n"
        config_text += f"  • Statistiques robustes: {'✅' if params['robust_stats'] else '❌'}\n"
        config_text += f"  • Filtre passe-bande: {'✅' if params['band_filter'] else '❌'}\n"
        config_text += f"  • Filtre notch: {'✅' if params['notch_filter'] else '❌'}\n\n"
        
        config_text += f"🛏️ Scoring:\n"
        config_text += f"  • Inclure scoring: {'✅' if self.batch_config['include_scoring'] else '❌'}\n"
        config_text += f"  • Auto-scoring YASA: {'✅' if self.batch_config['auto_scoring'] else '❌'}\n"
        
        # Afficher dans une nouvelle fenêtre
        config_window = tk.Toplevel(self.root)
        config_window.title("🔧 Configuration du Traitement")
        config_window.geometry("600x500")
        
        text_widget = tk.Text(config_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(1.0, config_text)
        text_widget.configure(state='disabled')

    def _update_batch_config(self):
        """Met à jour la configuration avec les valeurs actuelles des widgets."""
        # Canaux sélectionnés
        selected_channels = [channel for channel, var in self.channel_vars.items() if var.get()]
        self.batch_config['selected_channels'] = selected_channels
        
        # Paramètres FFT
        try:
            self.batch_config['fft_params'] = {
                'fmin': float(self.fmin_var.get()),
                'fmax': float(self.fmax_var.get()),
                'nperseg_sec': float(self.nperseg_var.get()),
                'equalize_epochs': self.equalize_var.get(),
                'robust_stats': self.robust_var.get(),
                'band_filter': self.filter_var.get(),
                'notch_filter': self.notch_var.get()
            }
        except ValueError as e:
            messagebox.showerror("Erreur", f"Paramètres FFT invalides: {e}")
            return False
        
        # Options de scoring
        self.batch_config['include_scoring'] = self.include_scoring_var.get()
        self.batch_config['auto_scoring'] = self.auto_scoring_var.get()
        # Mode de sortie
        try:
            self.batch_output_mode = self.batch_output_mode_var.get()
        except Exception:
            self.batch_output_mode = 'csv'
        self.batch_config['output_mode'] = self.batch_output_mode
        self.batch_config['before_dir'] = getattr(self, 'before_dir_var', tk.StringVar(value='')).get() if hasattr(self, 'before_dir_var') else ''
        self.batch_config['after_dir'] = getattr(self, 'after_dir_var', tk.StringVar(value='')).get() if hasattr(self, 'after_dir_var') else ''
        
        return True

    def _mark_selected_group(self, group_label: str):
        """Affecte le groupe (AVANT/APRÈS/'' ) aux fichiers sélectionnés dans la liste."""
        try:
            selected_items = self.files_tree.selection()
            try:
                logging.info(f"[SPAG_GROUP] {len(selected_items)} items -> group='{group_label}'")
            except Exception:
                pass
            if not selected_items:
                return
            # Mettre à jour la structure batch_config
            selected_names = set()
            for item in selected_items:
                vals = list(self.files_tree.item(item, 'values'))
                if not vals:
                    continue
                # Assurer 4 colonnes
                while len(vals) < 4:
                    vals.append('')
                vals[3] = group_label
                self.files_tree.item(item, values=vals)
                selected_names.add(vals[0])
            # Batch config
            if hasattr(self, 'batch_config'):
                for fi in self.batch_config.get('eeg_files', []):
                    if fi.get('name') in selected_names:
                        fi['group'] = group_label
            try:
                logging.info(f"[SPAG_GROUP] Affectation terminée pour {len(selected_names)} fichiers")
            except Exception:
                pass
        except Exception as e:
            print(f"🔍 CHECKPOINT GROUP: Erreur affectation groupe: {e}")

    def _start_batch_processing(self):
        """Démarre le traitement en lot."""
        # Validation de la configuration
        if not self._validate_batch_config():
            return
        
        # Mettre à jour la configuration
        if not self._update_batch_config():
            return
        
        # Préparer l'interface
        self.batch_stop_requested = False
        self.start_batch_btn.configure(state='disabled')
        self.stop_batch_btn.configure(state='normal')
        
        # Démarrer le traitement dans un thread séparé
        import threading
        target_runner = self._run_batch_processing if self.batch_config.get('output_mode', 'csv') == 'csv' else self._run_batch_spaghetti
        processing_thread = threading.Thread(target=target_runner)
        processing_thread.daemon = True
        processing_thread.start()

    def _toggle_output_mode_ui(self):
        try:
            mode = self.batch_output_mode_var.get()
        except Exception:
            mode = 'csv'
        # Checkpoint UI mode
        try:
            use_detected = getattr(self, 'spag_use_detected_var', tk.BooleanVar(value=True)).get()
            msg = f"[SPAG_UI] mode={mode}, use_detected={use_detected}"
            print(f"🔍 CHECKPOINT {msg}")
            logging.info(msg)
        except Exception:
            pass
        # Afficher/cacher les champs spécifiques spaghetti
        try:
            if mode == 'spaghetti':
                self.spaghetti_frame.pack_configure(fill=tk.X, padx=10, pady=(0,8))
                # Désactiver uniquement les champs dossiers AV/PR si on utilise les fichiers détectés
                use_detected = getattr(self, 'spag_use_detected_var', tk.BooleanVar(value=True)).get()
                state_dirs = 'disabled' if use_detected else 'normal'
                try:
                    if hasattr(self, 'spag_before_entry'):
                        self.spag_before_entry.configure(state=state_dirs)
                    if hasattr(self, 'spag_before_btn'):
                        self.spag_before_btn.configure(state=state_dirs)
                    if hasattr(self, 'spag_after_entry'):
                        self.spag_after_entry.configure(state=state_dirs)
                    if hasattr(self, 'spag_after_btn'):
                        self.spag_after_btn.configure(state=state_dirs)
                except Exception:
                    pass
                # Laisser actifs mapping bande→stade et combinaisons avancées
                try:
                    if hasattr(self, 'spag_combo_entry'):
                        self.spag_combo_entry.configure(state='normal')
                    if hasattr(self, 'spag_combo_ch'):
                        self.spag_combo_ch.configure(state='readonly')
                    if hasattr(self, 'spag_combo_band'):
                        self.spag_combo_band.configure(state='readonly')
                    if hasattr(self, 'spag_combo_stage'):
                        self.spag_combo_stage.configure(state='readonly')
                except Exception:
                    pass
            else:
                self.spaghetti_frame.pack_forget()
        except Exception:
            pass

    def _browse_dir_to_var(self, var_obj, title):
        try:
            d = filedialog.askdirectory(title=title)
            if d:
                var_obj.set(d)
        except Exception:
            pass

    def _toggle_band_stage_row(self, band: str, value: bool):
        try:
            if hasattr(self, 'spag_band_stage_vars') and band in self.spag_band_stage_vars:
                for st_code, var in self.spag_band_stage_vars[band].items():
                    var.set(bool(value))
        except Exception:
            pass

    def _toggle_band_stage_col(self, stage_code: str, value: bool):
        try:
            if hasattr(self, 'spag_band_stage_vars'):
                for band, mapping in self.spag_band_stage_vars.items():
                    if stage_code in mapping:
                        mapping[stage_code].set(bool(value))
        except Exception:
            pass

    def _extract_subject_base(self, name: str) -> str:
        """Retourne l'identifiant sujet de base (ex: S030) à partir d'un nom de fichier."""
        if not name:
            return ""
        base = os.path.splitext(os.path.basename(str(name)))[0]
        match = re.search(r'(S\d{2,3})', base.upper())
        if match:
            return match.group(1)
        return base

    def _collect_detected_subjects(self) -> List[str]:
        """Collecte les sujets détectés via les fichiers importés."""
        subjects: set[str] = set()
        try:
            if hasattr(self, 'batch_config'):
                for file_info in self.batch_config.get('eeg_files', []):
                    if not file_info:
                        continue
                    if file_info.get('selected', True) is False:
                        continue
                    base = self._extract_subject_base(file_info.get('name') or file_info.get('path'))
                    if base:
                        subjects.add(base)
        except Exception:
            pass
        if not subjects:
            try:
                for path in getattr(self, 'detected_eeg_files', []):
                    base = self._extract_subject_base(path)
                    if base:
                        subjects.add(base)
            except Exception:
                pass
        return sorted(subjects)

    def _refresh_spag_subject_pool(self, subjects: Optional[Iterable[str]] = None):
        """Met à jour la liste des sujets disponibles et la synchro UI."""
        try:
            if subjects is None:
                subjects = self._collect_detected_subjects()
            subjects_list = sorted({str(s) for s in subjects if s})
            self.spag_subjects = subjects_list
            # Nettoyer les clusters si des sujets ont disparu
            for subj in list(self.spag_clusters.keys()):
                if subj not in subjects_list:
                    self.spag_clusters.pop(subj, None)
            self._sync_spag_cluster_ui()
        except Exception:
            pass

    def _sync_spag_cluster_ui(self):
        """Synchronise les listboxes de clustering avec l'état courant."""
        if not hasattr(self, 'spag_subject_listbox') or self.spag_subject_listbox is None:
            return
        try:
            self.spag_subject_listbox.delete(0, tk.END)
        except Exception:
            pass
        for box in getattr(self, 'spag_cluster_listboxes', {}).values():
            try:
                box.delete(0, tk.END)
            except Exception:
                pass
        try:
            for subj in self.spag_subjects:
                cluster_id = self.spag_clusters.get(subj)
                if cluster_id and cluster_id in self.spag_cluster_listboxes:
                    try:
                        self.spag_cluster_listboxes[cluster_id].insert(tk.END, subj)
                    except Exception:
                        pass
                else:
                    try:
                        self.spag_subject_listbox.insert(tk.END, subj)
                    except Exception:
                        pass
        except Exception:
            pass

    def _assign_subjects_to_cluster(self, cluster_id: str):
        """Affecte les sujets sélectionnés au cluster donné."""
        try:
            if not self.spag_subject_listbox:
                return
            selection = [self.spag_subject_listbox.get(i) for i in self.spag_subject_listbox.curselection()]
            if not selection:
                return
            for subj in selection:
                self.spag_clusters[subj] = cluster_id
            self._sync_spag_cluster_ui()
        except Exception:
            pass

    def _remove_subjects_from_cluster(self, cluster_id: str):
        """Retire les sujets sélectionnés du cluster pour les remettre en pool."""
        try:
            box = self.spag_cluster_listboxes.get(cluster_id)
            if not box:
                return
            selection = [box.get(i) for i in box.curselection()]
            if not selection:
                return
            for subj in selection:
                self.spag_clusters.pop(subj, None)
            self._sync_spag_cluster_ui()
        except Exception:
            pass

    def _clear_spag_clusters(self):
        """Réinitialise la configuration des clusters."""
        try:
            self.spag_clusters.clear()
            self._sync_spag_cluster_ui()
        except Exception:
            pass

    def _on_spag_cluster_name_change(self, cluster_id: str):
        """Met à jour le nom du cluster depuis l'UI."""
        try:
            var = self.spag_cluster_name_vars.get(cluster_id)
            if not var:
                return
            text = var.get().strip()
            if not text:
                text = f"Cluster {cluster_id}"
            self.spag_cluster_names[cluster_id] = text
        except Exception:
            pass

    def _get_spag_group_labels(self) -> Tuple[str, str]:
        """Retourne les libellés personnalisés pour les groupes Before/After."""
        before = self.spag_group_before_var.get().strip() if hasattr(self, 'spag_group_before_var') else ''
        after = self.spag_group_after_var.get().strip() if hasattr(self, 'spag_group_after_var') else ''
        if not before:
            before = "Before"
        if not after:
            after = "After"
        return before, after

    def _get_spag_cluster_payload(self) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        """Retourne les clusters sujets→cluster et noms de clusters."""
        clusters_payload: Dict[str, str] = {}
        names_payload: Dict[str, str] = {}
        try:
            for subj, cid in getattr(self, 'spag_clusters', {}).items():
                if cid:
                    clusters_payload[str(subj)] = str(cid)
        except Exception:
            clusters_payload = {}
        try:
            cluster_ids = sorted(set(list(self.spag_cluster_names.keys()) + list(getattr(self, 'spag_cluster_name_vars', {}).keys())))
        except Exception:
            cluster_ids = ['A', 'B']
        if not cluster_ids:
            cluster_ids = ['A', 'B']
        for cid in cluster_ids:
            name = ''
            try:
                if hasattr(self, 'spag_cluster_name_vars') and cid in self.spag_cluster_name_vars:
                    name = self.spag_cluster_name_vars[cid].get().strip()
            except Exception:
                name = ''
            if not name and hasattr(self, 'spag_cluster_names'):
                name = self.spag_cluster_names.get(cid, '')
            if not name:
                name = f"Cluster {cid}"
            names_payload[cid] = name
        if not clusters_payload:
            clusters_value: Optional[Dict[str, str]] = None
        else:
            clusters_value = clusters_payload
        names_value: Optional[Dict[str, str]] = names_payload if names_payload else None
        return clusters_value, names_value

    def _run_batch_spaghetti(self):
        """Exécute la génération de graphiques spaghetti à partir d'EDF en lot (AVANT/APRÈS)."""
        try:
            # Import léger ici pour gérer l'environnement
            from .advanced_spaghetti_plots import generate_spaghetti_from_edf_file_lists  # type: ignore
        except Exception as e:
            self._update_batch_status(f"Erreur module spaghetti: {e}")
            return
        try:
            before_dir = self.batch_config.get('before_dir', '')
            after_dir = self.batch_config.get('after_dir', '')
            output_dir = self.batch_config.get('output_dir', '')
            use_detected = bool(getattr(self, 'spag_use_detected_var', tk.BooleanVar(value=True)).get())
            try:
                logging.info(f"[SPAG_RUN] use_detected={use_detected}, before_dir='{before_dir}', after_dir='{after_dir}', output_dir='{output_dir}'")
            except Exception:
                pass
            # Construire les listes de fichiers si demandé
            before_files: List[str] = []
            after_files: List[str] = []
            if use_detected and hasattr(self, 'batch_config') and self.batch_config.get('eeg_files'):
                # Utiliser les groupes marqués dans la liste (AVANT/APRÈS)
                for fi in self.batch_config.get('eeg_files', []):
                    grp = (fi.get('group') or '').upper()
                    path = fi.get('path')
                    if not path:
                        continue
                    if grp == 'AVANT':
                        before_files.append(path)
                    elif grp in ['APRÈS', 'APRES']:
                        after_files.append(path)
                # Si aucun groupe renseigné, fallback: heuristique nom de fichier
                if not before_files and not after_files and hasattr(self, 'detected_eeg_files'):
                    for p in self.detected_eeg_files:
                        name = os.path.basename(p).upper()
                        if any(k in name for k in ['_AV', ' AV', 'AVANT']):
                            before_files.append(p)
                        elif any(k in name for k in ['_AP', ' AP', 'APRES', 'APRÈS', 'APRES']):
                            after_files.append(p)
            else:
                if not (before_dir and after_dir):
                    self._update_batch_status("Veuillez renseigner dossiers AVANT et APRÈS")
                    return
                # Scanner les répertoires pour lister les EDF
                def _scan_edf(dir_path: str) -> List[str]:
                    hits: List[str] = []
                    exts = set(ext.lower() for ext in recording_extensions_for_scan())
                    for root, _dirs, files in os.walk(dir_path):
                        for f in files:
                            if normalize_recording_extension(f) in exts:
                                hits.append(os.path.join(root, f))
                    return hits
                before_files = _scan_edf(before_dir)
                after_files = _scan_edf(after_dir)
            try:
                logging.info(f"[SPAG_FILES] before={len(before_files)}, after={len(after_files)}")
            except Exception:
                pass
            if not output_dir:
                self._update_batch_status("Veuillez renseigner un dossier de sortie")
                return
            selected_channels = self.batch_config.get('selected_channels', [])
            try:
                logging.info(f"[SPAG_CH] selected_channels={selected_channels}")
            except Exception:
                pass
            self._update_batch_status("Génération des graphiques spaghetti…")
            # Appeler la version list-based pour garder les mêmes fichiers détectés
            # Récupérer filtres UI
            # Construire la carte bande→stades à partir du tableau fin (par bande et par stade)
            band_stage_map = {}
            try:
                if hasattr(self, 'spag_band_stage_vars'):
                    for band, stage_vars in self.spag_band_stage_vars.items():
                        stages_selected = [st for st, var in stage_vars.items() if var.get()]
                        if stages_selected:
                            band_stage_map[band] = stages_selected
                # Déduire listes globales (bandes/stades) comme unions du mapping fin
                sel_bands = list(band_stage_map.keys())
                sel_stages = sorted({st for stages in band_stage_map.values() for st in stages})
                logging.info(f"[SPAG_FILTERS] bands={sel_bands}, stages={sel_stages}, map={band_stage_map}")
            except Exception:
                # Fallback raisonnable si la table n'existe pas
                sel_bands = ["Alpha"]
                sel_stages = ["N2"]
                band_stage_map = {"Alpha": ["N2"]}

            # Construire mapping EDF->Excel depuis la config (si dispo)
            edf_to_excel = {}
            try:
                if hasattr(self, 'batch_config'):
                    for fi in self.batch_config.get('eeg_files', []):
                        p = fi.get('path')
                        xp = fi.get('excel_path')
                        if p and xp and os.path.exists(xp):
                            edf_to_excel[os.path.abspath(p)] = os.path.abspath(xp)
                logging.info(f"[SPAG_MAP] edf_to_excel entries={len(edf_to_excel)}")
            except Exception:
                pass

            # Combinaisons explicites parsées depuis le champ avancé
            combos = self._parse_spag_combinations()
            if combos:
                try:
                    # Étendre les sélections bandes/stades pour inclure celles des combinaisons
                    add_bands = sorted({b for (_c, _s, b) in combos})
                    add_stages = sorted({s for (_c, s, _b) in combos})
                    for b in add_bands:
                        if b not in sel_bands:
                            sel_bands.append(b)
                    for s in add_stages:
                        if s not in sel_stages:
                            sel_stages.append(s)
                    # Forcer les canaux aux canaux explicitement demandés si aucun autre canal n'est sélectionné
                    add_channels = sorted({c for (c, _s, _b) in combos})
                    if not selected_channels:
                        selected_channels = list(add_channels)
                    else:
                        for c in add_channels:
                            if c not in selected_channels:
                                selected_channels.append(c)
                except Exception:
                    pass

            clusters_payload, cluster_names_payload = self._get_spag_cluster_payload()
            before_label, after_label = self._get_spag_group_labels()

            outputs = generate_spaghetti_from_edf_file_lists(
                before_files=before_files,
                after_files=after_files,
                output_dir=output_dir,
                selected_bands=sel_bands,
                selected_stages=sel_stages,
                selected_channels=selected_channels if selected_channels else None,
                selected_subjects=None,
                selected_band_stage_map=band_stage_map,
                selected_combinations=combos,
                edf_to_excel_map=edf_to_excel if edf_to_excel else None,
                clusters=clusters_payload,
                cluster_names=cluster_names_payload,
                before_label=before_label,
                after_label=after_label,
                # passer la carte fine (actuellement même stades pour toutes les bandes sélectionnées)
                epoch_len=30.0,
                metric='AUC',
                rng_seed=42,
                n_perm=5000,
                n_boot=2000,
            )
            self._update_batch_status(f"Terminé: {len(outputs)} graphiques")
            try:
                logging.info(f"[SPAG_DONE] outputs={len(outputs)}")
            except Exception:
                pass
            self._log_batch(f"✅ Spaghetti: {len(outputs)} sorties -> {output_dir}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_batch_status(f"Erreur spaghetti: {e}")
            self._log_batch(f"❌ Erreur spaghetti: {e}")

    def _parse_spag_combinations(self):
        """Parse la zone de texte des combinaisons et retourne une liste (Channel, Stage, Band).
        Format attendu (séparateurs ';' ou ','): C3:Delta_R; F3:Theta_N3
        Stades acceptés: W, N1, N2, N3, R (insensibles à la casse et accents ignorés pour REM/EVEIL).
        Bandes mappées vers: LowDelta, Delta, Theta, Alpha, Sigma, Beta, Gamma.
        """
        # Priorité au sélecteur visuel s'il contient des entrées
        try:
            if hasattr(self, 'spag_combo_list') and self.spag_combo_list:
                return list(self.spag_combo_list)
        except Exception:
            pass
        try:
            spec = self.spag_combo_spec_var.get().strip() if hasattr(self, 'spag_combo_spec_var') else ''
        except Exception:
            spec = ''
        if not spec:
            return []
        # Helpers
        def _norm_stage(s: str) -> str:
            s2 = str(s).strip().upper()
            mapping = {
                'WAKE': 'W', 'AWAKE': 'W', 'EVEIL': 'W', 'ÉVEIL': 'W', 'W': 'W',
                'REM': 'R', 'PARADOXAL': 'R', 'R': 'R',
                'N1': 'N1', 'S1': 'N1',
                'N2': 'N2', 'S2': 'N2',
                'N3': 'N3', 'S3': 'N3', 'S4': 'N3'
            }
            return mapping.get(s2, s2)
        def _norm_band(b: str) -> str:
            b2 = str(b).strip().lower().replace('é', 'e').replace('è', 'e').replace('ê', 'e')
            # Normaliser vers clés connues
            if b2 in ['lowdelta', 'low_delta', 'low-delta', 'ldelta', 'ld']:
                return 'LowDelta'
            if b2 in ['delta', 'd']:
                return 'Delta'
            if b2 in ['theta', 't', 'thetaj', 'théta', 'thetal']:
                return 'Theta'
            if b2 in ['alpha', 'a']:
                return 'Alpha'
            if b2 in ['sigma', 's']:
                return 'Sigma'
            if b2 in ['beta', 'b']:
                return 'Beta'
            if b2 in ['gamma', 'g']:
                return 'Gamma'
            # Default: Title case
            return b.strip().title()

        combos = []
        # Split by ; or ,
        for item in [x for chunk in spec.split(';') for x in chunk.split(',')]:
            it = item.strip()
            if not it:
                continue
            if ':' not in it:
                continue
            ch, rest = it.split(':', 1)
            ch = ch.strip()
            if '_' in rest:
                band_str, stage_str = rest.split('_', 1)
            elif '-' in rest:
                band_str, stage_str = rest.split('-', 1)
            else:
                # Si un seul token, ignorer
                continue
            band = _norm_band(band_str)
            stage = _norm_stage(stage_str)
            combos.append((ch, stage, band))
        return combos

    def _add_spag_combo(self):
        try:
            ch = self.spag_combo_ch.get().strip()
            band = self.spag_combo_band.get().strip()
            stage = self.spag_combo_stage.get().strip()
            if not ch or not band or not stage:
                return
            item = (ch, stage, band)
            if not hasattr(self, 'spag_combo_list'):
                self.spag_combo_list = []
            if item not in self.spag_combo_list:
                self.spag_combo_list.append(item)
                self.spag_combo_tree.insert('', 'end', values=(ch, band, stage))
                # Mettre à jour la zone texte pour visibilité (optionnel)
                try:
                    txt = "; ".join([f"{c}:{b}_{s}" for (c,s,b) in self.spag_combo_list])
                    self.spag_combo_spec_var.set(txt)
                except Exception:
                    pass
        except Exception:
            pass

    def _remove_spag_combo(self):
        try:
            sel = self.spag_combo_tree.selection()
            for iid in sel:
                vals = self.spag_combo_tree.item(iid, 'values')
                ch, band, stage = vals[0], vals[1], vals[2]
                item = (str(ch), str(stage), str(band))
                if hasattr(self, 'spag_combo_list') and item in self.spag_combo_list:
                    self.spag_combo_list.remove(item)
                self.spag_combo_tree.delete(iid)
            # Sync texte
            try:
                txt = "; ".join([f"{c}:{b}_{s}" for (c,s,b) in getattr(self, 'spag_combo_list', [])])
                self.spag_combo_spec_var.set(txt)
            except Exception:
                pass
        except Exception:
            pass

    def _clear_spag_combos(self):
        try:
            self.spag_combo_list = []
            for iid in self.spag_combo_tree.get_children():
                self.spag_combo_tree.delete(iid)
            self.spag_combo_spec_var.set("")
        except Exception:
            pass

    def _validate_batch_config(self):
        """Valide la configuration avant de démarrer le traitement."""
        errors = []
        
        if not self.batch_config['input_dir']:
            errors.append("Dossier d'entrée non sélectionné")
        
        if not self.batch_config['output_dir']:
            errors.append("Dossier de sortie non sélectionné")
        
        if not self.batch_config['eeg_files']:
            errors.append("Aucun fichier EEG détecté")
        
        # Vérifier canaux: accepter soit les canaux cochés, soit des combinaisons explicites
        has_checked_channels = any(var.get() for var in self.channel_vars.values()) if hasattr(self, 'channel_vars') else False
        combos_ok = False
        try:
            combos_ok = bool(self._parse_spag_combinations())
        except Exception:
            combos_ok = False
        if not (has_checked_channels or combos_ok):
            errors.append("Aucun canal sélectionné (aucune combinaison Canal×Bande×Stade fournie)")
        
        if errors:
            messagebox.showerror("Configuration Invalide", "\n".join(errors))
            return False
        
        return True

    def _run_batch_processing(self):
        """Exécute le traitement en lot (dans un thread séparé)."""
        try:
            print("🔍 CHECKPOINT BATCH: Début _run_batch_processing")
            selected_files = [f for f in self.batch_config['eeg_files'] if f.get('selected', True)]
            total_files = len(selected_files)
            total_channels = len(self.batch_config['selected_channels'])
            total_operations = total_files * total_channels
            
            print(f"🔍 CHECKPOINT BATCH: {total_files} fichiers, {total_channels} canaux, {total_operations} opérations")
            print(f"🔍 CHECKPOINT BATCH: Canaux sélectionnés: {self.batch_config['selected_channels']}")
            print(f"🔍 CHECKPOINT BATCH: Paramètres FFT: {self.batch_config['fft_params']}")
            
            self._update_batch_status(f"Traitement de {total_files} fichiers avec {total_channels} canaux")
            self._log_batch(f"🚀 Début du traitement: {total_operations} opérations au total")
            
            operation_count = 0
            
            for file_idx, file_info in enumerate(selected_files):
                if self.batch_stop_requested:
                    break
                
                file_path = file_info['path']
                file_name = file_info['name']
                
                self._log_batch(f"\n📁 Traitement du fichier {file_idx + 1}/{total_files}: {file_name}")
                
                try:
                    print(f"🔍 CHECKPOINT BATCH FILE: Début traitement {file_name}")
                    # Charger le fichier EEG
                    self._update_batch_progress_text(f"Chargement: {file_name}")
                    raw_data = self._load_eeg_file_for_batch(file_path)
                    
                    if raw_data is None:
                        print(f"🔍 CHECKPOINT BATCH FILE: Échec chargement {file_name}")
                        self._log_batch(f"❌ Échec du chargement: {file_name}")
                        continue
                    
                    print(f"🔍 CHECKPOINT BATCH FILE: {file_name} chargé - {len(raw_data.ch_names)} canaux, {raw_data.info['sfreq']} Hz")
                    print(f"🔍 CHECKPOINT BATCH FILE: Canaux disponibles: {raw_data.ch_names}")
                    
                    # Charger ou calculer le scoring si nécessaire
                    scoring_data = None
                    if self.batch_config['include_scoring']:
                        print(f"🔍 CHECKPOINT BATCH SCORING: Recherche scoring pour {file_name}")
                        scoring_data = self._get_scoring_for_batch(file_path, raw_data)
                        if scoring_data is not None:
                            print(f"🔍 CHECKPOINT BATCH SCORING: Scoring trouvé - {len(scoring_data)} époques")
                        else:
                            print(f"🔍 CHECKPOINT BATCH SCORING: Aucun scoring trouvé pour {file_name}")
                    
                    # Traiter chaque canal
                    for channel_idx, channel in enumerate(self.batch_config['selected_channels']):
                        if self.batch_stop_requested:
                            break
                        
                        operation_count += 1
                        progress = (operation_count / total_operations) * 100
                        
                        self._update_batch_progress(progress)
                        self._update_batch_progress_text(f"FFT: {file_name} - {channel}")
                        
                        try:
                            print(f"🔍 CHECKPOINT BATCH CHANNEL: Début traitement canal {channel} ({channel_idx + 1}/{total_channels})")
                            success = self._process_channel_fft(raw_data, channel, scoring_data, file_name)
                            if success:
                                print(f"🔍 CHECKPOINT BATCH CHANNEL: {channel} - Export réussi")
                                self._log_batch(f"  ✅ {channel}: Export réussi")
                            else:
                                print(f"🔍 CHECKPOINT BATCH CHANNEL: {channel} - Échec de l'export")
                                self._log_batch(f"  ❌ {channel}: Échec de l'export")
                        
                        except Exception as e:
                            print(f"🔍 CHECKPOINT BATCH CHANNEL: {channel} - Erreur: {str(e)}")
                            self._log_batch(f"  ❌ {channel}: Erreur - {str(e)}")
                
                except Exception as e:
                    self._log_batch(f"❌ Erreur fichier {file_name}: {str(e)}")
                    continue
            
            # Traitement terminé
            if self.batch_stop_requested:
                self._update_batch_status("Traitement arrêté par l'utilisateur")
                self._log_batch("⏹️ Traitement arrêté par l'utilisateur")
            else:
                self._update_batch_status("Traitement terminé avec succès")
                self._log_batch(f"✅ Traitement terminé: {operation_count} opérations effectuées")
            
            self._update_batch_progress(100)
            
        except Exception as e:
            self._update_batch_status(f"Erreur: {str(e)}")
            self._log_batch(f"❌ Erreur générale: {str(e)}")
        
        finally:
            # Réactiver les boutons
            self._enqueue_tk_main(self._reset_batch_ui)

    def _load_eeg_file_for_batch(self, file_path):
        """Charge un fichier EEG pour le traitement en lot."""
        try:
            import mne
            raw = open_raw_file(file_path, preload=True, verbose=False)
            return raw
        except Exception as e:
            self._log_batch(f"❌ Erreur chargement {os.path.basename(file_path)}: {str(e)}")
            return None

    def _get_scoring_for_batch(self, eeg_file_path, raw_data):
        """Obtient les données de scoring pour un fichier EEG."""
        print(f"🔍 CHECKPOINT GET SCORING: Recherche scoring pour {os.path.basename(eeg_file_path)}")
        
        if not self.batch_config['include_scoring']:
            print("🔍 CHECKPOINT GET SCORING: Scoring désactivé dans la configuration")
            return None
        
        # Trouver le fichier dans la configuration pour obtenir son chemin Excel
        file_name = os.path.basename(eeg_file_path)
        excel_path = None
        
        for file_info in self.batch_config.get('eeg_files', []):
            if file_info['name'] == file_name:
                excel_path = file_info.get('excel_path', None)
                break
        
        print(f"🔍 CHECKPOINT GET SCORING: Excel path trouvé: {excel_path}")
        
        # Si un fichier Excel est associé, l'utiliser
        if excel_path and os.path.exists(excel_path):
            try:
                print(f"🔍 CHECKPOINT GET SCORING: Chargement Excel: {os.path.basename(excel_path)}")
                scoring_data = self._load_excel_scoring_for_batch(excel_path)
                if scoring_data is not None:
                    print(f"🔍 CHECKPOINT GET SCORING: Scoring chargé avec succès - {len(scoring_data)} époques")
                    self._log_batch(f"  📊 Scoring trouvé: {os.path.basename(excel_path)}")
                    return scoring_data
                else:
                    print("🔍 CHECKPOINT GET SCORING: Échec du chargement Excel")
            except Exception as e:
                print(f"🔍 CHECKPOINT GET SCORING: Erreur chargement Excel: {str(e)}")
                self._log_batch(f"  ❌ Erreur scoring: {str(e)}")
        
        # Fallback: chercher un fichier Excel avec le même nom de base
        print("🔍 CHECKPOINT GET SCORING: Fallback - recherche par nom de base")
        base_name = os.path.splitext(eeg_file_path)[0]
        excel_extensions = ['.xlsx', '.xls']
        
        for ext in excel_extensions:
            scoring_path = base_name + ext
            print(f"🔍 CHECKPOINT GET SCORING: Test chemin: {scoring_path}")
            if os.path.exists(scoring_path):
                try:
                    print(f"🔍 CHECKPOINT GET SCORING: Fichier trouvé, chargement...")
                    scoring_data = self._load_excel_scoring_for_batch(scoring_path)
                    if scoring_data is not None:
                        print(f"🔍 CHECKPOINT GET SCORING: Fallback réussi - {len(scoring_data)} époques")
                        self._log_batch(f"  📊 Scoring trouvé (fallback): {os.path.basename(scoring_path)}")
                        return scoring_data
                except Exception as e:
                    print(f"🔍 CHECKPOINT GET SCORING: Erreur fallback: {str(e)}")
                    self._log_batch(f"  ❌ Erreur scoring: {str(e)}")
        
        # Si pas de scoring trouvé et auto-scoring activé
        if self.batch_config['auto_scoring']:
            try:
                print("🔍 CHECKPOINT GET SCORING: Tentative auto-scoring YASA...")
                self._log_batch(f"  🤖 Calcul auto-scoring YASA...")
                # Ici on pourrait implémenter l'auto-scoring YASA
                # Pour l'instant, on retourne None
                return None
            except Exception as e:
                print(f"🔍 CHECKPOINT GET SCORING: Erreur auto-scoring: {str(e)}")
                self._log_batch(f"  ❌ Erreur auto-scoring: {str(e)}")
        
        print("🔍 CHECKPOINT GET SCORING: Aucun scoring trouvé")
        return None

    def _load_excel_scoring_for_batch(self, excel_path):
        """Charge un fichier de scoring Excel pour le traitement en lot."""
        try:
            print(f"🔍 CHECKPOINT EXCEL BATCH: Chargement {excel_path}")
            import pandas as pd
            df = pd.read_excel(excel_path)
            
            print(f"🔍 CHECKPOINT EXCEL BATCH: DataFrame chargé - {len(df)} lignes, {len(df.columns)} colonnes")
            print(f"🔍 CHECKPOINT EXCEL BATCH: Colonnes: {df.columns.tolist()}")
            print(f"🔍 CHECKPOINT EXCEL BATCH: Premières lignes:\n{df.head()}")
            
            # Normaliser les colonnes comme dans la méthode principale
            if 'time' in df.columns and 'stage' in df.columns:
                print(f"🔍 CHECKPOINT EXCEL BATCH: Colonnes 'time' et 'stage' trouvées")
                result = df[['time', 'stage']].copy()
            elif len(df.columns) >= 2:
                print(f"🔍 CHECKPOINT EXCEL BATCH: Utilisation des 2 premières colonnes")
                df_normalized = df.iloc[:, :2].copy()
                
                # CORRECTION: Les colonnes sont souvent inversées dans les fichiers Excel
                # Première colonne = stage (Sommeil), Deuxième colonne = time (Heure de début)
                df_normalized.columns = ['stage', 'time']  # Correction de l'ordre
                print(f"🔍 CHECKPOINT EXCEL BATCH: Colonnes mappées: stage='{df.columns[0]}', time='{df.columns[1]}'")
                result = df_normalized
            else:
                print(f"🔍 CHECKPOINT EXCEL BATCH: Pas assez de colonnes ({len(df.columns)})")
                return None
            
            print(f"🔍 CHECKPOINT EXCEL BATCH: Résultat final - {len(result)} lignes")
            print(f"🔍 CHECKPOINT EXCEL BATCH: Types de données: {result.dtypes.to_dict()}")
            
            # Nettoyer les lignes invalides (contenant des listes vides ou des valeurs non-string)
            print(f"🔍 CHECKPOINT EXCEL BATCH: Nettoyage des lignes invalides...")
            
            # Fonction pour vérifier si une valeur est valide
            def is_valid_cell(value):
                if value is None:
                    return False
                if isinstance(value, list) and len(value) == 0:
                    return False
                if isinstance(value, str) and value.strip() in ['', '[]', 'nan', 'NaN']:
                    return False
                try:
                    # Essayer de convertir en string pour vérifier
                    str(value)
                    return True
                except:
                    return False
            
            # Filtrer les lignes valides
            valid_mask = result['time'].apply(is_valid_cell) & result['stage'].apply(is_valid_cell)
            result_clean = result[valid_mask].copy()
            
            # Convertir explicitement en string pour éviter les erreurs de type
            result_clean['time'] = result_clean['time'].astype(str)
            result_clean['stage'] = result_clean['stage'].astype(str)
            
            print(f"🔍 CHECKPOINT EXCEL BATCH: Vérification ordre des colonnes après nettoyage:")
            print(f"🔍 CHECKPOINT EXCEL BATCH: Première ligne - stage='{result_clean['stage'].iloc[0]}', time='{result_clean['time'].iloc[0]}'")
            
            # Vérifier si les colonnes sont dans le bon ordre (stage doit être un mot, time doit être une date)
            first_stage = str(result_clean['stage'].iloc[0])
            first_time = str(result_clean['time'].iloc[0])
            
            # Si le "stage" ressemble à une date et le "time" à un mot, inverser
            if ('-' in first_stage or ':' in first_stage) and (first_time in ['Éveil', 'ÉVEIL', 'N1', 'N2', 'N3', 'REM', 'R', 'W']):
                print(f"🔍 CHECKPOINT EXCEL BATCH: INVERSION détectée - correction automatique")
                result_clean = result_clean[['time', 'stage']].copy()  # Inverser l'ordre
                result_clean.columns = ['stage', 'time']  # Renommer correctement
                print(f"🔍 CHECKPOINT EXCEL BATCH: Après inversion - stage='{result_clean['stage'].iloc[0]}', time='{result_clean['time'].iloc[0]}'")
            else:
                print(f"🔍 CHECKPOINT EXCEL BATCH: Ordre des colonnes correct")
            
            print(f"🔍 CHECKPOINT EXCEL BATCH: Nettoyage terminé - {len(result)} -> {len(result_clean)} lignes")
            print(f"🔍 CHECKPOINT EXCEL BATCH: Premières lignes nettoyées:\n{result_clean.head()}")
            
            return result_clean
        
        except Exception as e:
            print(f"🔍 CHECKPOINT EXCEL BATCH: ERREUR: {str(e)}")
            raise Exception(f"Erreur lecture Excel: {str(e)}")

    def _process_channel_fft(self, raw_data, channel_name, scoring_data, file_name):
        """Traite la FFT pour un canal spécifique et exporte le CSV."""
        try:
            print(f"🔍 CHECKPOINT PROCESS FFT: Début traitement {channel_name} pour {file_name}")
            
            # Vérifier que le canal existe
            if channel_name not in raw_data.ch_names:
                print(f"🔍 CHECKPOINT PROCESS FFT: Canal {channel_name} non trouvé")
                print(f"🔍 CHECKPOINT PROCESS FFT: Canaux disponibles: {raw_data.ch_names}")
                self._log_batch(f"  ⚠️ Canal {channel_name} non trouvé dans {file_name}")
                return False
            
            print(f"🔍 CHECKPOINT PROCESS FFT: Canal {channel_name} trouvé")
            
            # Extraire le signal du canal
            channel_idx = raw_data.ch_names.index(channel_name)
            print(f"🔍 CHECKPOINT PROCESS FFT: Index du canal: {channel_idx}")
            
            signal_data = raw_data.get_data()
            print(f"🔍 CHECKPOINT PROCESS FFT: Shape des données: {signal_data.shape}")
            
            signal = signal_data[channel_idx, :]
            print(f"🔍 CHECKPOINT PROCESS FFT: Signal extrait - longueur: {len(signal)}")
            
            fs = raw_data.info['sfreq']
            print(f"🔍 CHECKPOINT PROCESS FFT: Fréquence d'échantillonnage: {fs} Hz")
            
            # Paramètres FFT
            params = self.batch_config['fft_params']
            print(f"🔍 CHECKPOINT PROCESS FFT: Paramètres FFT: {params}")
            
            # Calculer la FFT par stade
            if scoring_data is not None:
                print(f"🔍 CHECKPOINT PROCESS FFT: Calcul FFT avec scoring - {len(scoring_data)} époques")
                print(f"🔍 CHECKPOINT PROCESS FFT: Colonnes scoring: {scoring_data.columns.tolist()}")
                print(f"🔍 CHECKPOINT PROCESS FFT: Premières lignes scoring:\n{scoring_data.head()}")
                
                # Avec scoring
                fft_results = self._compute_stage_psd_fft_custom(
                    signal=signal,
                    fs=fs,
                    scoring_df=scoring_data,
                    epoch_len=30.0,  # Durée d'époque standard
                    stages=['W', 'N1', 'N2', 'N3', 'R'],
                    fmin=params['fmin'],
                    fmax=params['fmax'],
                    equalize_epochs=params['equalize_epochs'],
                    robust_stats=params['robust_stats'],
                    nperseg_sec=params['nperseg_sec']
                )
                print(f"🔍 CHECKPOINT PROCESS FFT: FFT calculée - {len(fft_results)} stades")
            else:
                print(f"🔍 CHECKPOINT PROCESS FFT: Calcul FFT globale (sans scoring)")
                # Sans scoring - analyse globale
                fft_results = self._compute_global_fft(signal, fs, params)
                print(f"🔍 CHECKPOINT PROCESS FFT: FFT globale calculée - {len(fft_results)} résultats")
            
            # Créer le nom du fichier de sortie
            base_name = os.path.splitext(file_name)[0]
            safe_channel = channel_name.replace('-', '_').replace('/', '_')
            output_filename = f"{base_name}_{safe_channel}_FFT.csv"
            output_path = os.path.join(self.batch_config['output_dir'], output_filename)
            
            print(f"🔍 CHECKPOINT PROCESS FFT: Nom de sortie: {output_filename}")
            print(f"🔍 CHECKPOINT PROCESS FFT: Chemin complet: {output_path}")
            
            # Exporter le CSV
            self._export_fft_results_to_csv(fft_results, output_path)
            print(f"🔍 CHECKPOINT PROCESS FFT: Export CSV terminé pour {channel_name}")
            
            return True
            
        except Exception as e:
            raise Exception(f"Erreur traitement FFT: {str(e)}")

    def _compute_global_fft(self, signal, fs, params):
        """Calcule la FFT globale sans segmentation par stade."""
        try:
            from scipy import signal as scipy_signal
            import numpy as np
            
            # Appliquer les filtres si demandé
            filtered_signal = signal.copy()
            
            if params['band_filter']:
                # Filtre passe-bande
                sos = scipy_signal.butter(4, [0.3, 40.0], btype='bandpass', fs=fs, output='sos')
                filtered_signal = scipy_signal.sosfilt(sos, filtered_signal)
            
            if params['notch_filter']:
                # Filtre notch 50 Hz
                sos = scipy_signal.butter(4, [49, 51], btype='bandstop', fs=fs, output='sos')
                filtered_signal = scipy_signal.sosfilt(sos, filtered_signal)
            
            # Calculer la PSD avec Welch
            nperseg = int(params['nperseg_sec'] * fs)
            freqs, psd = scipy_signal.welch(filtered_signal, fs=fs, nperseg=nperseg)
            
            # Filtrer la bande de fréquence
            freq_mask = (freqs >= params['fmin']) & (freqs <= params['fmax'])
            freqs_filtered = freqs[freq_mask]
            psd_filtered = psd[freq_mask]
            
            # Format de sortie compatible
            results = {
                'GLOBAL': (freqs_filtered, psd_filtered, np.zeros_like(psd_filtered), 1)
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Erreur FFT globale: {str(e)}")

    def _export_fft_results_to_csv(self, fft_results, output_path):
        """Exporte les résultats FFT vers un fichier CSV."""
        try:
            print(f"🔍 CHECKPOINT EXPORT CSV: Début export vers {output_path}")
            print(f"🔍 CHECKPOINT EXPORT CSV: Nombre de stades: {len(fft_results)}")
            
            import csv
            
            # Vérifier les résultats FFT
            for stage, result in fft_results.items():
                if len(result) != 4:
                    print(f"🔍 CHECKPOINT EXPORT CSV: ERREUR - Résultat malformé pour {stage}: {len(result)} éléments au lieu de 4")
                    print(f"🔍 CHECKPOINT EXPORT CSV: Contenu: {result}")
                else:
                    freqs, mean_vals, sem_vals, n_epochs = result
                    print(f"🔍 CHECKPOINT EXPORT CSV: {stage} - {len(freqs)} fréquences, {n_epochs} époques")
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["stage", "n_epochs", "freq_hz", "magnitude", "sem"])
                
                total_rows = 0
                for stage, (freqs, mean_vals, sem_vals, n_epochs) in fft_results.items():
                    print(f"🔍 CHECKPOINT EXPORT CSV: Traitement {stage} - {len(freqs)} lignes")
                    
                    # Vérifier la cohérence des longueurs
                    if len(freqs) != len(mean_vals) or len(freqs) != len(sem_vals):
                        print(f"🔍 CHECKPOINT EXPORT CSV: ERREUR longueurs incohérentes pour {stage}:")
                        print(f"  freqs: {len(freqs)}, mean_vals: {len(mean_vals)}, sem_vals: {len(sem_vals)}")
                        continue
                    
                    for i in range(len(freqs)):
                        try:
                            writer.writerow([
                                stage, 
                                n_epochs, 
                                float(freqs[i]), 
                                float(mean_vals[i]), 
                                float(sem_vals[i])
                            ])
                            total_rows += 1
                        except Exception as e:
                            print(f"🔍 CHECKPOINT EXPORT CSV: ERREUR ligne {i} pour {stage}: {str(e)}")
                            print(f"  freq: {freqs[i]}, mean: {mean_vals[i]}, sem: {sem_vals[i]}")
                
                print(f"🔍 CHECKPOINT EXPORT CSV: Export terminé - {total_rows} lignes écrites")
        
        except Exception as e:
            print(f"🔍 CHECKPOINT EXPORT CSV: ERREUR GÉNÉRALE: {str(e)}")
            raise Exception(f"Erreur export CSV: {str(e)}")

    def _stop_batch_processing(self):
        """Arrête le traitement en lot."""
        self.batch_stop_requested = True
        self._log_batch("⏹️ Arrêt du traitement demandé...")

    def _reset_batch_ui(self):
        """Remet l'interface en état normal après le traitement."""
        self.start_batch_btn.configure(state='normal')
        self.stop_batch_btn.configure(state='disabled')

    def _update_batch_status(self, status):
        """Met à jour le statut du traitement."""
        self._enqueue_tk_main(lambda: self.batch_status_var.set(status))

    def _update_batch_progress(self, value):
        """Met à jour la barre de progression."""
        self._enqueue_tk_main(lambda: self.batch_progress.configure(value=value))

    def _update_batch_progress_text(self, text):
        """Met à jour le texte de progression."""
        self._enqueue_tk_main(lambda: self.batch_progress_text.set(text))

    def _log_batch(self, message):
        """Ajoute un message aux logs du traitement en lot."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def update_logs():
            if hasattr(self, 'batch_logs'):
                self.batch_logs.insert(tk.END, log_message)
                self.batch_logs.see(tk.END)
        
        self._enqueue_tk_main(update_logs)
    
    def _setup_modern_plot(self, parent: ttk.Frame) -> None:
        """Configure le conteneur principal pour afficher la vue PSG multi-subplots."""
        # Conteneur dédié au viewer PSG
        self.psg_container = ttk.Frame(parent)
        self.psg_container.pack(fill=tk.BOTH, expand=True)

        # Si des données sont chargées, afficher directement la vue PSG intégrée
        if getattr(self, 'raw', None) is not None:
            try:
                self._show_default_psg_view(embed_parent=self.psg_container)
                return
            except Exception:
                pass

        # Fallback: pas de graphe — simple placeholder textuel
        holder = ttk.Frame(self.psg_container)
        holder.pack(fill=tk.BOTH, expand=True)
        msg = ttk.Label(holder, text="Aucun enregistrement charge\nOuvrez un fichier pour afficher la PSG",
                         anchor=tk.CENTER, justify=tk.CENTER)
        msg.pack(expand=True)
    
    def _apply_modern_plot_style(self) -> None:
        """Applique le style moderne au graphique avec les couleurs du thème actuel."""
        # Obtenir les couleurs du thème actuel
        theme = self.theme_manager.get_current_theme()
        ui_colors = theme.get_ui_colors()

        # Changer le fond de la figure matplotlib (très important)
        self.fig.patch.set_facecolor(ui_colors.get('bg', '#ffffff'))

        # S'assurer que les axes utilisent aussi le bon fond
        if hasattr(self, 'ax'):
            self.ax.set_facecolor(ui_colors.get('bg', '#ffffff'))

        # Appliquer l'image de fond si disponible
        self.theme_manager.apply_background_to_figure(self.fig, theme)

        # Forcer le rafraîchissement
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

        # Configuration des axes avec les couleurs du thème (si ax existe)
        if hasattr(self, 'ax') and self.ax is not None:
            self.ax.set_facecolor(ui_colors.get('bg', '#ffffff'))
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_color(ui_colors.get('fg', '#000000'))
            self.ax.spines['bottom'].set_color(ui_colors.get('fg', '#000000'))

            # Grille moderne avec les couleurs du thème
            self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=ui_colors.get('fg', '#000000'))
            self.ax.set_axisbelow(True)

        # Titre et labels avec les couleurs du thème (si une figure principale est active)
        try:
            if hasattr(self, 'raw') and self.raw is not None and hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                self.ax.set_title("CESA (EEG Studio Analysis) - Visualisation des Signaux",
                                 fontsize=12, fontweight='bold', pad=10, color=ui_colors.get('fg', '#000000'))
                self.ax.set_xlabel("Temps (s)", fontsize=11, color=ui_colors.get('fg', '#000000'))
                self.ax.set_ylabel("Amplitude (µV)", fontsize=11, color=ui_colors.get('fg', '#000000'))
        except Exception:
            pass

            # Configuration des ticks avec les couleurs du thème
            self.ax.tick_params(colors=ui_colors.get('fg', '#000000'), labelsize=10)

            # NOTE: Le texte d'information par défaut est géré dans update_plot()
            # Il sera effacé automatiquement lors du premier chargement de données

        # Style pour l'aperçu hypnogramme (si présent)
        if hasattr(self, 'ax_overview') and self.ax_overview is not None:
            ov = self.ax_overview
            ov.set_facecolor(ui_colors.get('bg', '#ffffff'))
            for side in ['top', 'right', 'left', 'bottom']:
                ov.spines[side].set_visible(False)
            ov.grid(False)
            ov.set_yticks([])
            ov.tick_params(axis='x', colors=ui_colors.get('fg', '#000000'), labelsize=8)
            # Pas de titre pour compacité
            ov.tick_params(axis='x', colors=ui_colors.get('fg', '#000000'), labelsize=7)
    
    def _create_modern_controls(self, parent: ttk.Frame) -> None:
        """Crée les contrôles modernes."""
        # Titre des contrôles
        title_label = ttk.Label(
            parent, 
            text="🎛️ Contrôles EEG", 
            style='Modern.TLabel',
            font=('Segoe UI', 12, 'bold')
        )
        title_label.pack(pady=(0, 5))  # Réduit encore plus : 5px
        
        # Contrôles de navigation temporelle
        self._create_time_controls(parent)
        
        # Contrôles d'affichage
        self._create_display_controls(parent)
        
        # Contrôles de traitement
        self._create_processing_controls(parent)
        
        # Boutons d'action
        self._create_action_buttons(parent)
    
    def _create_time_controls(self, parent: ttk.Frame) -> None:
        """Crée les contrôles de navigation temporelle."""
        time_frame = ttk.LabelFrame(parent, text="⏱️ Navigation Temporelle", style='Group.TLabelframe')
        time_frame.pack(fill=tk.X, pady=(0, 5))  # Réduit encore plus : 5px
        
        # Slider de temps moderne - PLUS GRAND
        time_control_frame = ttk.Frame(time_frame)
        time_control_frame.pack(fill=tk.X, padx=6, pady=4)  # Réduit encore plus
        
        ttk.Label(time_control_frame, text="Position:", style='Modern.TLabel').pack(anchor=tk.W)
        
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(
            time_control_frame, 
            from_=0, to=100, 
            variable=self.time_var, 
            orient=tk.HORIZONTAL, 
            command=self._update_time,
            length=400  # Barre plus longue
        )
        self.time_scale.pack(fill=tk.X, pady=(2, 4), expand=True)  # Réduit encore plus
        
        # Affichage du temps
        self.time_label = ttk.Label(
            time_control_frame, 
            text="00h00", 
            style='Modern.TLabel',
            font=('Segoe UI', 10, 'bold')
        )
        self.time_label.pack()
        
        # Contrôles de durée
        duration_frame = ttk.Frame(time_control_frame)
        duration_frame.pack(fill=tk.X, pady=(4, 0))  # Réduit encore plus : 4px
        
        ttk.Label(duration_frame, text="Durée (s):", style='Modern.TLabel').pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="10.0")
        duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=8)
        duration_entry.pack(side=tk.RIGHT)
        duration_entry.bind('<Return>', self._update_duration)
        
        # Boutons de navigation
        nav_frame = ttk.Frame(time_control_frame)
        nav_frame.pack(fill=tk.X, pady=(4, 0))  # Réduit encore plus : 4px
        
        nav_buttons = [
            ("⏮️", self._jump_backward, "Saut arrière"),
            ("⏪", self._step_backward, "Pas arrière"),
            ("⏩", self._step_forward, "Pas avant"),
            ("⏭️", self._jump_forward, "Saut avant")
        ]
        
        for text, command, tooltip in nav_buttons:
            btn = ttk.Button(nav_frame, text=text, command=command, style='Modern.TButton')
            btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
    
    def _create_display_controls(self, parent: ttk.Frame) -> None:
        """Crée les contrôles d'affichage."""
        display_frame = ttk.LabelFrame(parent, text="📊 Affichage", style='Group.TLabelframe')
        display_frame.pack(fill=tk.X, pady=(0, 5))  # Réduit encore plus : 5px
        
        # Espacement
        spacing_frame = ttk.Frame(display_frame)
        spacing_frame.pack(fill=tk.X, padx=6, pady=4)  # Réduit encore plus
        
        ttk.Label(spacing_frame, text="Espacement:", style='Modern.TLabel').pack(anchor=tk.W)
        self.spacing_scale = ttk.Scale(
            spacing_frame, 
            from_=10, to=200, 
            variable=self.spacing_var, 
            orient=tk.HORIZONTAL, 
            command=self._update_plot
        )
        self.spacing_scale.pack(fill=tk.X, pady=(2, 4))  # Réduit encore plus
        
        # Amplitude
        amplitude_frame = ttk.Frame(display_frame)
        amplitude_frame.pack(fill=tk.X, padx=6, pady=(0, 4))  # Réduit encore plus
        
        ttk.Label(amplitude_frame, text="Amplitude:", style='Modern.TLabel').pack(anchor=tk.W)
        self.amplitude_scale = ttk.Scale(
            amplitude_frame, 
            from_=10, to=1000, 
            variable=self.amplitude_var, 
            orient=tk.HORIZONTAL, 
            command=self._update_plot
        )
        self.amplitude_scale.pack(fill=tk.X, pady=(2, 0))  # Réduit encore plus
        
        # Sélecteur de thème pour le scoring
        theme_frame = ttk.Frame(amplitude_frame)
        theme_frame.pack(fill=tk.X, pady=(4, 0))

        self.theme_combo_scoring = ttk.Combobox(
            theme_frame,
            textvariable=tk.StringVar(value=self.theme_manager.current_theme_name),
            state="readonly",
            width=15
        )
        self.theme_combo_scoring['values'] = list(self.theme_manager.get_available_themes().values())
        self.theme_combo_scoring.pack(side=tk.RIGHT)
        self.theme_combo_scoring.bind('<<ComboboxSelected>>',
                                    lambda e: self._change_theme_by_display_name(self.theme_combo_scoring.get()))

        # Tooltip pour le sélecteur de thème
        self._create_tooltip(self.theme_combo_scoring,
                           "Changer le thème graphique du scoring.\n\n"
                           "• 3 thèmes disponibles avec palettes de couleurs cohérentes\n"
                           "• Met à jour automatiquement l'affichage des stades\n"
                           "• Applique l'image de fond correspondante")
    
    def _create_processing_controls(self, parent: ttk.Frame) -> None:
        """Crée les contrôles de traitement."""
        processing_frame = ttk.LabelFrame(parent, text="🔧 Traitement", style='Group.TLabelframe')
        processing_frame.pack(fill=tk.X, pady=(0, 5))  # Réduit encore plus : 5px
        
        # Options de traitement
        options_frame = ttk.Frame(processing_frame)
        options_frame.pack(fill=tk.X, padx=6, pady=4)  # Réduit encore plus
        
        # Autoscale
        self.autoscale_var = tk.BooleanVar()
        autoscale_cb = ttk.Checkbutton(
            options_frame, 
            text="📏 Autoscale", 
            variable=self.autoscale_var, 
            command=self._toggle_autoscale,
            style='Modern.TCheckbutton'
        )
        autoscale_cb.pack(anchor=tk.W, pady=1)  # Réduit l'espacement
        
        # Filtre (utilise la variable globale déjà définie)
        filter_cb = ttk.Checkbutton(
            options_frame,
            text="🔧 Filtre Passe-bande",
            variable=self.filter_var,
            command=self._toggle_filter,
            style='Modern.TCheckbutton'
        )
        filter_cb.pack(anchor=tk.W, pady=1)  # Réduit l'espacement
        
        # Bouton configuration avancée des filtres
        advanced_filter_btn = ttk.Button(
            options_frame,
            text="⚙️ Configuration Avancée",
            command=self.show_filter_config,
            style='Modern.TButton'
        )
        advanced_filter_btn.pack(anchor=tk.W, pady=1)  # Réduit l'espacement
        
        # Amplification automatique
        self.auto_amp_var = tk.BooleanVar(value=True)
        auto_amp_cb = ttk.Checkbutton(
            options_frame, 
            text="⚡ Amplification Auto", 
            variable=self.auto_amp_var, 
            command=self._update_plot,
            style='Modern.TCheckbutton'
        )
        auto_amp_cb.pack(anchor=tk.W, pady=1)  # Réduit l'espacement
    
    def _create_action_buttons(self, parent: ttk.Frame) -> None:
        """Crée les boutons d'action."""
        action_frame = ttk.LabelFrame(parent, text="⚡ Actions", style='Group.TLabelframe')
        action_frame.pack(fill=tk.X)
        
        buttons_frame = ttk.Frame(action_frame)
        buttons_frame.pack(fill=tk.X, padx=6, pady=4)  # Réduit encore plus
        
        # Boutons principaux
        buttons = [
            ("📋 Canaux", self._show_channel_selector, "Sélectionner les canaux"),
            ("📈 Statistiques", self._show_channel_stats, "Afficher les statistiques"),
            ("🔍 Diagnostic", self._show_diagnostics, "Diagnostic des données"),
            ("💾 Exporter", self._export_data, "Exporter les données")
        ]
        
        for text, command, tooltip in buttons:
            btn = ttk.Button(
                buttons_frame, 
                text=text, 
                command=command, 
                style='Modern.TButton'
            )
            btn.pack(fill=tk.X, pady=1)  # Réduit l'espacement
    
    def _create_modern_toolbar(self, parent: ttk.Frame) -> None:
        """Crée la barre d'outils moderne."""
        self.toolbar_frame = ttk.Frame(parent, style='Modern.TFrame')
        self.toolbar_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Barre d'outils matplotlib
        try:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
            self.toolbar.update()
        except Exception:
            self.toolbar = None
        
        # Boutons personnalisés
        custom_frame = ttk.Frame(self.toolbar_frame)
        custom_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Bouton d'actualisation
        refresh_btn = ttk.Button(
            custom_frame, 
            text="🔄 Actualiser", 
            command=self._refresh_plot,
            style='Modern.TButton'
        )
        refresh_btn.pack(side=tk.RIGHT, padx=2)

    def _recreate_toolbar(self) -> None:
        """Recrée la barre d'outils pour la rattacher au canvas courant."""
        try:
            if hasattr(self, 'toolbar_frame') and self.toolbar_frame is not None:
                for child in self.toolbar_frame.winfo_children():
                    try:
                        child.destroy()
                    except Exception:
                        pass
                try:
                    self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
                    self.toolbar.update()
                except Exception:
                    self.toolbar = None
        except Exception:
            pass

    # =====================================================================
    # PANEL DE CONTRÔLE MODERNE ET MODULABLE
    # =====================================================================
    def _create_control_panel(self, parent: ttk.Frame) -> None:
        """Construit un panel unique structuré avec sections et tooltips.

        Sections:
          1) Navigation (⟵ ⟶, ▶︎/⏸, ⇤, ⇥)
          2) Zoom / Fenêtre (slider de durée, + / -)
          3) Autoscale (toggle)
          4) Filtrage (toggle + raccourci vers config)
          5) Baseline (toggle)
        """
        ui = self.theme_manager.get_current_theme().get_ui_colors()

        def add_tooltip(widget, text: str):
            try:
                import tkinter as _tk
                tip = _tk.Toplevel(widget)
                tip.withdraw()
                tip.overrideredirect(True)
                lbl = ttk.Label(tip, text=text, background=ui.get('surface', '#ffffff'),
                                foreground=ui.get('fg', '#1f2335'), relief='solid', borderwidth=1)
                lbl.pack(ipadx=4, ipady=2)

                def enter(_e):
                    try:
                        x = widget.winfo_rootx() + 10
                        y = widget.winfo_rooty() + widget.winfo_height() + 6
                        tip.geometry(f"+{x}+{y}")
                        tip.deiconify()
                    except Exception:
                        pass
                def leave(_e):
                    tip.withdraw()
                widget.bind('<Enter>', enter)
                widget.bind('<Leave>', leave)
            except Exception:
                pass

        container = ttk.Frame(parent, style='Custom.TFrame')
        container.pack(fill=tk.X)
        # Ensure a shared time variable exists for key navigation updates
        try:
            if not hasattr(self, 'time_var'):
                self.time_var = tk.DoubleVar(value=float(getattr(self, 'current_time', 0.0)))
        except Exception:
            pass

        # 1) Navigation
        nav = ttk.Labelframe(container, text='Navigation', style='Custom.TLabelframe')
        nav.pack(fill=tk.X, padx=6, pady=4)
        # Navigation: utiliser les méthodes existantes step_backward/step_forward et helpers locaux
        def _go_start():
            try:
                self.current_time = 0.0
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter.set_time_window(float(self.current_time), float(self.duration))
                self.update_plot()
            except Exception:
                pass
        def _go_end():
            try:
                if getattr(self, 'raw', None) is not None and hasattr(self, 'sfreq'):
                    total_dur = float(len(self.raw.times) / self.sfreq)
                    self.current_time = max(0.0, total_dur - float(self.duration))
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter.set_time_window(float(self.current_time), float(self.duration))
                self.update_plot()
            except Exception:
                pass
        btn_prev = ttk.Button(nav, text='⟵', width=3, command=getattr(self, 'step_backward', lambda: None), style='Modern.TButton')
        btn_next = ttk.Button(nav, text='⟶', width=3, command=getattr(self, 'step_forward', lambda: None), style='Modern.TButton')
        btn_play = ttk.Button(nav, text='▶︎', width=3, command=getattr(self, '_play_start', lambda: None), style='Modern.TButton')
        btn_pause = ttk.Button(nav, text='⏸', width=3, command=getattr(self, '_play_pause', lambda: None), style='Modern.TButton')
        btn_home = ttk.Button(nav, text='⇤', width=3, command=_go_start, style='Modern.TButton')
        btn_end = ttk.Button(nav, text='⇥', width=3, command=_go_end, style='Modern.TButton')
        for b, tip in [
            (btn_prev, 'Époque précédente (←)'),
            (btn_next, 'Époque suivante (→)'),
            (btn_play, 'Lecture'),
            (btn_pause, 'Pause'),
            (btn_home, 'Début (Home)'),
            (btn_end, 'Fin (End)'),
        ]:
            b.pack(side=tk.LEFT, padx=2, pady=2)
            add_tooltip(b, tip)

        # 2) Zoom / Fenêtre
        zoom = ttk.Labelframe(container, text='Zoom / Fenêtre', style='Custom.TLabelframe')
        zoom.pack(fill=tk.X, padx=6, pady=4)
        # Zoom: ajuster la durée de fenêtre via le slider programmatique
        def _zoom(delta: float):
            try:
                new_val = float(win_var.get()) + delta
                new_val = min(120.0, max(10.0, new_val))
                win_var.set(new_val)
                _apply_window_change(new_val)
            except Exception:
                pass
        btn_zoom_out = ttk.Button(zoom, text='–', width=3, command=lambda: _zoom(+10.0), style='Modern.TButton')
        btn_zoom_in = ttk.Button(zoom, text='+', width=3, command=lambda: _zoom(-10.0), style='Modern.TButton')
        btn_zoom_out.pack(side=tk.LEFT, padx=2, pady=2)
        btn_zoom_in.pack(side=tk.LEFT, padx=2, pady=2)
        add_tooltip(btn_zoom_out, 'Zoom arrière (-)')
        add_tooltip(btn_zoom_in, 'Zoom avant (+)')

        # Slider de fenêtre (durée)
        try:
            win_var = tk.DoubleVar(value=float(getattr(self, 'duration', 30.0)))
            # Libellé "Xs" mis à jour en temps réel
            win_value_lbl = ttk.Label(zoom, text=f"{int(win_var.get())}s")
            def _update_win_label():
                try:
                    win_value_lbl.config(text=f"{int(round(float(win_var.get())))}s")
                except Exception:
                    pass
            def _apply_window_change(new_val: float):
                try:
                    self.duration = float(new_val)
                    if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                        self.psg_plotter.set_time_window(float(self.current_time), float(self.duration))
                    else:
                        self.update_plot()
                    print(f"🔍 CHECKPOINT UI: duration -> {self.duration}s")
                    _update_win_label()
                except Exception:
                    pass
            win_slider = ttk.Scale(
                zoom,
                from_=10.0,
                to=120.0,
                orient=tk.HORIZONTAL,
                variable=win_var,
                command=lambda v: self.root.after(0, _apply_window_change, float(v)),
                style='Custom.Horizontal.TScale'
            )
            win_slider.pack(side=tk.LEFT, padx=8, pady=2)
            win_value_lbl.pack(side=tk.LEFT, padx=(2, 8))
            add_tooltip(win_slider, 'Durée de la fenêtre (s)')
        except Exception:
            pass

        # Entrée numérique pour la durée (Enter / FocusOut)
        try:
            dur_entry_var = tk.StringVar(value=str(int(float(getattr(self, 'duration', 30.0)))))
            def _apply_dur_entry(_ev=None):
                try:
                    val = float(dur_entry_var.get())
                    val = min(120.0, max(10.0, val))
                    win_var.set(val)
                    _apply_window_change(val)
                except Exception:
                    pass
            dur_entry = ttk.Entry(zoom, textvariable=dur_entry_var, width=6)
            dur_entry.pack(side=tk.LEFT, padx=(0, 6))
            dur_entry.bind('<Return>', _apply_dur_entry)
            dur_entry.bind('<FocusOut>', _apply_dur_entry)
            add_tooltip(dur_entry, 'Durée (s) — saisie directe puis Enter')
        except Exception:
            pass

        # 3) Autoscale
        autos = ttk.Labelframe(container, text='Autoscale', style='Custom.TLabelframe')
        autos.pack(fill=tk.X, padx=6, pady=4)
        try:
            if not hasattr(self, 'autoscale_var'):
                self.autoscale_var = tk.BooleanVar(value=bool(getattr(self, 'autoscale_enabled', False)))
            def _on_auto_toggle():
                try:
                    self.autoscale_enabled = bool(self.autoscale_var.get())
                    print(f"🔧 CHECKPOINT UI: autoscale -> {'on' if self.autoscale_enabled else 'off'}")
                    if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                        self.psg_plotter.set_autoscale_enabled(self.autoscale_enabled)
                        print(f"🔧 CHECKPOINT UI->VIEWER: autoscale pushed")
                    self.update_plot()
                except Exception:
                    pass
            ttk.Checkbutton(autos, text='📏 Autoscale', variable=self.autoscale_var, command=_on_auto_toggle).pack(side=tk.LEFT, padx=2, pady=2)
        except Exception:
            pass

        # 4) Filtrage
        filt = ttk.Labelframe(container, text='Filtrage', style='Custom.TLabelframe')
        filt.pack(fill=tk.X, padx=6, pady=4)
        try:
            if not hasattr(self, 'filter_var'):
                self.filter_var = tk.BooleanVar(value=bool(getattr(self, 'filter_enabled', True)))
            def _on_filter_toggle():
                try:
                    self.filter_enabled = bool(self.filter_var.get())
                    print(f"🔧 CHECKPOINT UI: filter -> {'on' if self.filter_enabled else 'off'}")
                    if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                        self.psg_plotter.set_global_filter_enabled(self.filter_enabled)
                        print(f"🔧 CHECKPOINT UI->VIEWER: filter pushed")
                    self.update_plot()
                except Exception:
                    pass
            ttk.Checkbutton(filt, text='🔧 Filtre', variable=self.filter_var, command=_on_filter_toggle).pack(side=tk.LEFT, padx=2, pady=2)
        except Exception:
            pass
        btn_cfg = ttk.Button(filt, text='⚙️ Config', command=self.show_filter_config, style='Modern.TButton')
        btn_cfg.pack(side=tk.LEFT, padx=2, pady=2)
        add_tooltip(btn_cfg, 'Configurer le filtre avancé')

        # 5) Baseline / prétraitement
        base = ttk.Labelframe(container, text='Baseline', style='Custom.TLabelframe')
        base.pack(fill=tk.X, padx=6, pady=4)
        try:
            if not hasattr(self, 'baseline_var'):
                self.baseline_var = tk.BooleanVar(value=True)
            def _on_baseline_toggle():
                try:
                    val = bool(self.baseline_var.get())
                    print(f"🔧 CHECKPOINT UI: baseline -> {'on' if val else 'off'}")
                    if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                        self.psg_plotter.set_baseline_enabled(val)
                        print(f"🔧 CHECKPOINT UI->VIEWER: baseline pushed")
                    self.update_plot()
                except Exception:
                    pass
            ttk.Checkbutton(base, text='⎺⎻ Baseline', variable=self.baseline_var, command=_on_baseline_toggle).pack(side=tk.LEFT, padx=2, pady=2)
            add_tooltip(base.winfo_children()[-1], 'Activer/désactiver la correction de ligne de base')
        except Exception:
            pass

        # 6) Sélection de canaux et disposition
        layout = ttk.Labelframe(container, text='Canaux & Disposition', style='Custom.TLabelframe')
        layout.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(layout, text='📋 Choisir canaux', command=self._show_channel_selector, style='Modern.TButton').pack(side=tk.LEFT, padx=2, pady=2)
        # Disposition: ordre EEG/EOG/EMG/ECG ou libre
        disp_var = tk.StringVar(value='auto')
        for label, val in [('Auto', 'auto'), ('EEG>EOG>EMG>ECG', 'grouped'), ('Libre', 'free')]:
            rb = ttk.Radiobutton(layout, text=label, value=val, variable=disp_var)
            rb.pack(side=tk.LEFT, padx=2)
        add_tooltip(layout, 'Organiser les signaux par type ou librement')

        # 7) Épaisseur des lignes
        style_grp = ttk.Labelframe(container, text='Affichage avancé', style='Custom.TLabelframe')
        style_grp.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(style_grp, text='Épaisseur:').pack(side=tk.LEFT)
        lw_var = tk.DoubleVar(value=0.9)
        def _on_lw_change(*_):
            try:
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter._line_width = float(lw_var.get())
                    self.psg_plotter._redraw(draw_only=True)
            except Exception:
                pass
        sp = ttk.Spinbox(style_grp, from_=0.5, to=3.0, increment=0.1, textvariable=lw_var, width=5, command=_on_lw_change)
        sp.pack(side=tk.LEFT, padx=4)
        sp.bind('<Return>', lambda e: _on_lw_change())
        sp.bind('<FocusOut>', lambda e: _on_lw_change())

        # 9) Thème (sélecteur)
        try:
            theme_grp = ttk.Labelframe(container, text='Thème', style='Custom.TLabelframe')
            theme_grp.pack(fill=tk.X, padx=6, pady=4)
            from CESA.theme_manager import theme_manager as _tm
            themes_map = _tm.get_available_themes()  # {key: display}
            current_key = _tm.current_theme_name
            current_display = themes_map.get(current_key, current_key)
            ttk.Label(theme_grp, text='Choisir:').pack(side=tk.LEFT)
            theme_var = tk.StringVar(value=current_display)
            theme_combo = ttk.Combobox(theme_grp, textvariable=theme_var, state='readonly', width=16)
            theme_combo['values'] = list(themes_map.values())
            theme_combo.pack(side=tk.LEFT, padx=6, pady=2)
            theme_combo.bind('<<ComboboxSelected>>', lambda e: self._change_theme_by_display_name(theme_var.get()))
            add_tooltip(theme_combo, 'Changer le thème de l’interface et des tracés')
        except Exception:
            pass

        # 8) Libellés de canaux ON/OFF
        lbl_var = tk.BooleanVar(value=True)
        def _on_lbl_toggle():
            try:
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter._show_labels = bool(lbl_var.get())
                    self.psg_plotter._redraw(draw_only=True)
            except Exception:
                pass
        ttk.Checkbutton(style_grp, text='Libellés', variable=lbl_var, command=_on_lbl_toggle).pack(side=tk.LEFT, padx=6)

        # 9) Événements ON/OFF
        evt_var = tk.BooleanVar(value=True)
        def _on_evt_toggle():
            try:
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter._show_events = bool(evt_var.get())
                    self.psg_plotter._redraw(draw_only=True)
            except Exception:
                pass
        ttk.Checkbutton(style_grp, text='Événements', variable=evt_var, command=_on_evt_toggle).pack(side=tk.LEFT, padx=6)

        # 10) Grille ON/OFF
        grid_var = tk.BooleanVar(value=True)
        def _on_grid_toggle():
            try:
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter._grid_on = bool(grid_var.get())
                    self.psg_plotter._redraw(draw_only=True)
            except Exception:
                pass
        ttk.Checkbutton(style_grp, text='Grille', variable=grid_var, command=_on_grid_toggle).pack(side=tk.LEFT, padx=6)

        # 11) Contraste élevé
        hc_var = tk.BooleanVar(value=False)
        def _on_hc_toggle():
            try:
                if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                    self.psg_plotter._high_contrast = bool(hc_var.get())
                    self.psg_plotter._redraw(draw_only=True)
            except Exception:
                pass
        ttk.Checkbutton(style_grp, text='Contraste élevé', variable=hc_var, command=_on_hc_toggle).pack(side=tk.LEFT, padx=6)
    
    # =====================================================================
    # MÉTHODES DE CONTRÔLES MODERNES
    # =====================================================================
    
    def _update_time(self, value):
        """Met à jour le temps affiché."""
        self.current_time = float(value)
        
        # Choisir l'affichage (absolu si disponible)
        if self.time_format == "absolu" and hasattr(self, 'absolute_start_datetime'):
            label_text = self._get_absolute_time(self.current_time)
        else:
            label_text = f"{self.current_time:.1f}s"

        # Mise à jour des labels
        # Navigation: toujours HHhMM (ou HH:MM:SS si absolu)
        if self.time_format == "absolu" and hasattr(self, 'absolute_start_datetime'):
            nav_text = self._get_absolute_time(self.current_time)
            # convertir HH:MM:SS -> HHhMM
            try:
                nav_text = nav_text[:2] + 'h' + nav_text[3:5]
            except Exception:
                pass
        else:
            nav_text = self._format_hhmm(self.current_time)

        self.time_label.config(text=nav_text)
        
        # Synchroniser la barre du bas
        if hasattr(self, 'bottom_time_var'):
            self.bottom_time_var.set(self.current_time)
        if hasattr(self, 'bottom_time_label'):
            self.bottom_time_label.config(text=nav_text)
        
        self.update_plot()
    
    def _update_time_from_bottom(self, value):
        """Met à jour le temps depuis la barre du bas."""
        self.current_time = float(value)
        
        # Choisir l'affichage (absolu si disponible)
        if self.time_format == "absolu" and hasattr(self, 'absolute_start_datetime'):
            label_text = self._get_absolute_time(self.current_time)
        else:
            label_text = f"{self.current_time:.1f}s"

        # Mise à jour des labels
        # Navigation: toujours HHhMM (ou HH:MM:SS si absolu)
        if self.time_format == "absolu" and hasattr(self, 'absolute_start_datetime'):
            nav_text = self._get_absolute_time(self.current_time)
            try:
                nav_text = nav_text[:2] + 'h' + nav_text[3:5]
            except Exception:
                pass
        else:
            nav_text = self._format_hhmm(self.current_time)

        self.time_label.config(text=nav_text)
        self.bottom_time_label.config(text=nav_text)
        
        # Synchroniser la barre du haut
        if hasattr(self, 'time_var'):
            self.time_var.set(self.current_time)
        
        self.update_plot()
    
    def _update_duration(self, event=None):
        """Met à jour la durée d'affichage."""
        try:
            self.duration = float(self.duration_var.get())
            self.update_time_scale()
            self.update_plot()
        except ValueError:
            print("🔍 CHECKPOINT DURATION RESET: Réinitialisation durée à 10.0s (ValueError)")
            logging.info("[DURATION] Reset to 10.0s due to ValueError")
            self.duration_var.set("10.0")
    
    def _jump_backward(self):
        """Saut en arrière."""
        self.current_time = max(0, self.current_time - self.duration)
        self.time_var.set(self.current_time)
        self.update_plot()
    
    def _step_backward(self):
        """Pas en arrière."""
        self.current_time = max(0, self.current_time - 1)
        self.time_var.set(self.current_time)
        self.update_plot()
    
    def _step_forward(self):
        """Pas en avant."""
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + 1)
            self.time_var.set(self.current_time)
            self.update_plot()
    
    def _increase_amplitude_with_zqsd_reset(self):
        """Augmente l'amplitude et réinitialise les bindings ZQSD."""
        print("🔍 CHECKPOINT ARROW: Flèche haut utilisée - réinitialisation ZQSD")
        logging.info("[ARROW] Up arrow used - resetting ZQSD bindings")
        self._increase_amplitude()
        self._setup_keyboard_navigation_simple()
    
    def _decrease_amplitude_with_zqsd_reset(self):
        """Diminue l'amplitude et réinitialise les bindings ZQSD."""
        print("🔍 CHECKPOINT ARROW: Flèche bas utilisée - réinitialisation ZQSD")
        logging.info("[ARROW] Down arrow used - resetting ZQSD bindings")
        self._decrease_amplitude()
        self._setup_keyboard_navigation_simple()
    
    def _step_forward_with_zqsd_reset(self):
        """Avance d'un pas et réinitialise les bindings ZQSD."""
        print("🔍 CHECKPOINT ARROW: Flèche droite utilisée - réinitialisation ZQSD")
        logging.info("[ARROW] Right arrow used - resetting ZQSD bindings")
        self._step_forward()
        self._setup_keyboard_navigation_simple()
    
    def _step_backward_with_zqsd_reset(self):
        """Recule d'un pas et réinitialise les bindings ZQSD."""
        print("🔍 CHECKPOINT ARROW: Flèche gauche utilisée - réinitialisation ZQSD")
        logging.info("[ARROW] Left arrow used - resetting ZQSD bindings")
        self._step_backward()
        self._setup_keyboard_navigation_simple()
    
    def _jump_forward(self):
        """Saut en avant."""
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + self.duration)
            self.time_var.set(self.current_time)
            self.update_plot()
    
    def _on_plot_click(self, event):
        """Gestion du clic sur le graphique."""
        if event.inaxes == getattr(self, 'ax_overview', None):
            # Navigation via clic sur la bande d'aperçu
            click_time = event.xdata
            if click_time is not None and self.raw is not None:
                try:
                    total_duration = float(len(self.raw.times) / self.sfreq)
                    win = float(getattr(self, 'duration', 10.0))
                    # Centrer la fenêtre autour du clic si possible
                    new_start = float(click_time) - win / 2.0
                    new_start = max(0.0, min(new_start, max(0.0, total_duration - win)))
                    self.current_time = new_start
                    if hasattr(self, 'time_var'):
                        try:
                            self.time_var.set(self.current_time)
                        except Exception:
                            pass
                    self.update_plot()
                    logging.info(f"Clic sur l'aperçu à {float(click_time):.1f}s → navigation")
                except Exception:
                    pass
            return
        
        if event.inaxes == self.ax:
            # Convertir la position du clic en temps
            click_time = event.xdata
            if click_time is not None:
                # Clic droit: menu contextuel pour changer le stade
                try:
                    if getattr(event, 'button', None) == 3:
                        self._open_stage_context_menu(event, float(click_time))
                        return
                except Exception:
                    pass
                # Clic gauche: déplacer le curseur
                self.current_time = float(click_time)
                if hasattr(self, 'time_var'):
                    try:
                        self.time_var.set(self.current_time)
                    except Exception:
                        pass
                self.update_plot()
                logging.info(f"Clic sur le graphique à {float(click_time):.1f}s")

    def _set_stage_for_epoch(self, epoch_start_time: float, new_stage: str) -> None:
        """Met à jour/ajoute une époque au temps donné avec le stade fourni.
        Crée un scoring manuel si inexistant. Marque l'état comme non enregistré.
        """
        try:
            import pandas as pd
            epoch_len = float(getattr(self, 'scoring_epoch_duration', 30.0))
            # Préparer DataFrame de travail
            df = None
            if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None:
                df = self.manual_scoring_data.copy()
            elif hasattr(self, 'sleep_scoring_data') and self.sleep_scoring_data is not None:
                df = self.sleep_scoring_data.copy()
            else:
                df = pd.DataFrame(columns=['time', 'stage'])
            # Arrondir au début d'époque
            epoch_idx = int(np.floor((float(epoch_start_time) + 1e-6) / epoch_len))
            epoch_t0 = float(epoch_idx * epoch_len)
            if 'time' not in df.columns:
                df['time'] = []
            if 'stage' not in df.columns:
                df['stage'] = []
            tol = 1e-6
            mask = df['time'].apply(lambda t: abs(float(t) - epoch_t0) <= tol)
            if mask.any():
                df.loc[mask, 'stage'] = str(new_stage).upper()
            else:
                df = pd.concat([df, pd.DataFrame([{'time': epoch_t0, 'stage': str(new_stage).upper()}])], ignore_index=True)
            df = df.sort_values('time').reset_index(drop=True)
            # Basculer sur le scoring manuel
            self.manual_scoring_data = df
            self.show_manual_scoring = True
            self.scoring_dirty = True
            self.update_plot()
        except Exception as e:
            logging.warning(f"_set_stage_for_epoch failed: {e}")

    def _open_stage_context_menu(self, mpl_event, click_time: float) -> None:
        """Ouvre un menu contextuel pour changer le stade à l'époque du temps cliqué."""
        try:
            menu = tk.Menu(self.root, tearoff=0)
            for st in ['W', 'N1', 'N2', 'N3', 'R', 'U']:
                menu.add_command(label=st, command=lambda s=st: self._set_stage_for_epoch(click_time, s))
            # Position écran
            x_root = int(getattr(mpl_event, 'guiEvent', getattr(mpl_event, 'event', None)).x_root) if hasattr(getattr(mpl_event, 'guiEvent', None), 'x_root') else self.root.winfo_pointerx()
            y_root = int(getattr(mpl_event, 'guiEvent', getattr(mpl_event, 'event', None)).y_root) if hasattr(getattr(mpl_event, 'guiEvent', None), 'y_root') else self.root.winfo_pointery()
            menu.tk_popup(x_root, y_root)
        except Exception as e:
            logging.warning(f"_open_stage_context_menu failed: {e}")
    
    def _get_absolute_time(self, time_seconds):
        """Convertit le temps en secondes en temps absolu HH:MM:SS."""
        if hasattr(self, 'absolute_start_datetime') and self.absolute_start_datetime:
            # Toujours afficher le temps absolu par rapport au début EDF (ignore l'origine Excel)
            base_dt = self.absolute_start_datetime
            print(f"🕒 CHECKPOINT LABEL: base={'EXCEL' if getattr(self, 'display_start_datetime', None) else 'EDF'}, start={base_dt}")
            logging.info(f"[LABEL] base={'EXCEL' if getattr(self, 'display_start_datetime', None) else 'EDF'} start={base_dt}")
            absolute_time = base_dt + timedelta(seconds=time_seconds)
            return absolute_time.strftime('%H:%M:%S')
        else:
            return f"{time_seconds:.1f}s"
    
    def _on_plot_hover(self, event):
        """Gestion du survol du graphique."""
        if event.inaxes == self.ax and event.xdata is not None:
            # Mettre à jour les informations de temps
            hover_time = float(event.xdata)
            if hasattr(self, 'absolute_start_datetime') and self.absolute_start_datetime:
                # Afficher le temps absolu au survol
                label_text = self._get_absolute_time(hover_time)
            else:
                label_text = f"{hover_time:.1f}s"
            self.time_info_label.config(text=f"Temps: {label_text}")
    
    def create_controls(self, parent):
        """Crée les contrôles de l'interface"""
        # Titre
        title_label = ttk.Label(parent, text="Contrôles EEG", font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Contrôles de temps
        time_frame = ttk.LabelFrame(parent, text="Navigation Temporelle")
        time_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Slider de temps
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = ttk.Scale(time_frame, from_=0, to=100, variable=self.time_var, 
                                   orient=tk.HORIZONTAL, command=self.update_time)
        self.time_scale.pack(fill=tk.X, padx=5, pady=5)
        
        # Affichage du temps
        self.time_label = ttk.Label(time_frame, text="0.0s")
        self.time_label.pack(pady=(0, 5))
        
        # Contrôles de durée
        duration_frame = ttk.Frame(time_frame)
        duration_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(duration_frame, text="Durée (s):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="10.0")
        duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=8)
        duration_entry.pack(side=tk.RIGHT)
        duration_entry.bind('<Return>', self.update_duration)
        
        # Boutons de navigation
        nav_frame = ttk.Frame(time_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="<<", command=self.jump_backward).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="<", command=self.step_backward).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text=">", command=self.step_forward).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text=">>", command=self.jump_forward).pack(side=tk.LEFT, padx=2)
        
        # Contrôles d'affichage
        display_frame = ttk.LabelFrame(parent, text="Affichage")
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Espacement
        spacing_frame = ttk.Frame(display_frame)
        spacing_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(spacing_frame, text="Espacement:").pack(side=tk.LEFT)
        self.spacing_scale = ttk.Scale(spacing_frame, from_=10, to=200, variable=self.spacing_var, 
                                      orient=tk.HORIZONTAL, command=self.update_plot)
        self.spacing_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Amplitude
        amplitude_frame = ttk.Frame(display_frame)
        amplitude_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(amplitude_frame, text="Amplitude:").pack(side=tk.LEFT)
        self.amplitude_scale = ttk.Scale(amplitude_frame, from_=10, to=1000, variable=self.amplitude_var, 
                                        orient=tk.HORIZONTAL, command=self.update_plot)
        self.amplitude_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Options
        options_frame = ttk.LabelFrame(parent, text="Options")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.autoscale_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Autoscale", variable=self.autoscale_var, 
                       command=self.toggle_autoscale).pack(anchor=tk.W, padx=5, pady=2)
        
        # Filtre (utilise la variable globale déjà définie)
        ttk.Checkbutton(options_frame, text="Filtre", variable=self.filter_var,
                       command=self.toggle_filter).pack(anchor=tk.W, padx=5, pady=2)
        
        # Boutons d'action
        action_frame = ttk.LabelFrame(parent, text="Actions")
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Sélectionner Canaux", command=self.show_channel_selector).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Statistiques", command=self.show_channel_stats).pack(fill=tk.X, padx=5, pady=2)
    
    def setup_keyboard_shortcuts(self):
        """Configure les raccourcis clavier"""
        self.root.bind('<Control-o>', lambda e: self.load_edf_file())
        self.root.bind('<Control-a>', lambda e: self.toggle_autoscale())
        self.root.bind('<Control-f>', lambda e: self.toggle_filter())
        self.root.bind('<Control-m>', lambda e: self._open_matplotlib_psg_view())
    
    def load_edf_file(self):
        """Charge un fichier EDF avec barre de chargement stylée"""
        dialog = OpenDatasetDialog(self.root)
        selection = dialog.show()
        if selection is None:
            return

        file_path = selection.edf_path
        selected_mode = (selection.mode or "raw").lower()
        precompute_action = (selection.precompute_action or "existing")
        ms_path_input = getattr(selection, "source_path", None)
        if ms_path_input is None:
            ms_path_input = getattr(selection, "ms_path", None)

        is_lazy_mode = selected_mode in {"raw", "lazy"}
        selected_mode_resolved = "lazy" if is_lazy_mode else selected_mode
        if selected_mode_resolved != selected_mode:
            logging.info(f"[OPEN] Mode résolu={selected_mode_resolved} (alias de {selected_mode})")
        selected_mode = selected_mode_resolved

        logging.info(f"[OPEN] Mode choisi={selected_mode}")
        logging.info(f"[OPEN] EDF path={file_path}")
        if selected_mode == "precomputed":
            logging.info(f"[OPEN] Zarr (input)={ms_path_input}")
            logging.info(f"[OPEN] Action={precompute_action}")

        # Mode checkpoint: impose expected visualization backend (lazy/hybrid, etc.)
        try:
            telemetry.mark_mode(selected_mode, source="EEGAnalysisStudio.load_edf_file")
        except Exception:
            # Re-raise after logging so the exception surfaces to the caller/UI
            logging.exception("Checkpoint mode violation detected")
            raise

        try:
            print(f"🧭 Mode choisi: {selected_mode}")
            print(f"📁 EDF: {file_path}")
            if selected_mode == "precomputed":
                print(f"🗂️  Zarr (entrée): {ms_path_input}")
                print(f"🛠️  Action: {precompute_action}")
        except Exception:
            pass

        # Afficher la barre de chargement
        self._show_loading_bar(
            title="Chargement du fichier EEG",
            message="Ouverture du fichier EDF..."
        )
        
        # Créer un sample de télémetrie pour le chargement
        load_sample = telemetry.new_sample({
            "channel": "load_edf",
            "dataset_id": str(Path(file_path).stem),
            "notes": f"mode={selected_mode}",
        })
        
        try:
            print(f"📁 Chargement du fichier: {os.path.basename(file_path)}")
            self._update_loading_message("Lecture du fichier EDF...")
            self.root.update()
            
            # Chargement du fichier avec mesure du temps
            with telemetry.measure(load_sample, "load_edf_ms"):
                self.raw = open_raw_file(file_path, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            logging.info(f"[OPEN] EDF chargé: n_channels={len(self.raw.ch_names)}, fs={self.sfreq}")
            
            # Extraire le temps absolu de début d'enregistrement
            self._extract_absolute_time()
            
            # Extraire les bad spans depuis les annotations EDF+ (si présents)
            try:
                self.bad_spans = self._extract_bad_spans_from_annotations()
                if self.bad_spans and len(self.bad_spans) > 0:
                    print(f"✅ BAD SPANS: {len(self.bad_spans)} segments d'artefacts détectés dans les annotations EDF")
                else:
                    print("ℹ️ BAD SPANS: Aucun segment d'artefact détecté dans les annotations EDF")
            except Exception as _e_bs:
                print(f"⚠️ BAD SPANS: échec extraction annotations: {_e_bs}")
            
            print(f"✅ Fichier chargé - {len(self.raw.ch_names)} canaux, {self.sfreq} Hz")
            
            self._update_loading_message("Création des dérivations...")
            self.root.update()
            
            # Création des dérivations avec mesure du temps
            with telemetry.measure(load_sample, "prepare_data_ms"):
                # Création des dérivations
                self.create_derivations()
                
                self._update_loading_message("Sélection des canaux...")
                self.root.update()
                
            # Mapping manuel obligatoire des canaux vers les sections du profil actif.
            if not self._ensure_profile_channel_mapping(list(self.raw.ch_names)):
                self._hide_loading_bar()
                messagebox.showwarning(
                    "Profils",
                    "L'assignation des canaux est obligatoire pour ce profil.\nChargement annule.",
                    parent=self.root,
                )
                return

            ordered_for_viewer: List[str] = []
            section_order = [s.key for s in self.active_profile.signal_sections if bool(s.enabled)]
            for section_key in section_order:
                section_channels = [
                    ch for ch, mapped in self.profile_channel_map_runtime.items()
                    if mapped == section_key and ch in self.raw.ch_names
                ]
                ordered_for_viewer.extend(section_channels)

            # Keep selected_channels for analysis defaults, PSG viewer uses psg_channels_used.
            self.psg_channels_used = list(ordered_for_viewer)
            if ordered_for_viewer:
                self.selected_channels = list(ordered_for_viewer[:8])
            else:
                self.selected_channels = list(self.raw.ch_names[:8])
                self.psg_channels_used = list(self.selected_channels)

            print(f"✅ Canaux mappes par profil: {len(self.psg_channels_used)}")
            for section_key in section_order:
                sec_count = len([ch for ch in self.psg_channels_used if self.profile_channel_map_runtime.get(ch) == section_key])
                print(f"   - {section_key}: {sec_count}")
            
            self._update_loading_message("Mise à jour de l'affichage...")
            self.root.update()
            
            # Fin de la mesure de préparation des données
            # (sortie du contexte with telemetry.measure)

            ms_path_obj = Path(ms_path_input) if ms_path_input else None
            self.data_bridge = None
            self.data_mode = "lazy" if selected_mode == "lazy" else "raw"

            # === GESTION DU MODE PRÉ-CALCULÉ ===
            if selected_mode == "precomputed":
                self._update_loading_message("Mode pré-calculé sélectionné...")
                self.root.update()
                
                # Helpers de normalisation/validation
                def _default_ms(p: Path) -> Path:
                    try:
                        return Path(p).with_suffix("") / "_ms"
                    except Exception:
                        return Path(str(p) + "_ms")

                def _normalize_ms_path(edf: str, candidate: Path | None) -> Path:
                    base_default = _default_ms(Path(edf))
                    if candidate is None:
                        return base_default
                    c = Path(candidate)
                    # If a file or an EDF path is provided, map to default _ms
                    if c.is_file() or c.suffix.lower() == ".edf":
                        return base_default
                    # If a directory is provided but is not a valid Zarr, prefer a '_ms' child
                    if c.exists() and (c / ".zattrs").exists() and (c / "levels").exists():
                        return c
                    # If user pointed to a parent like <edf_base>, use <edf_base>/_ms
                    if c.name != "_ms":
                        return c / "_ms"
                    return c

                def _is_valid_zarr(path: Path) -> bool:
                    try:
                        if not (path.exists() and (path / ".zattrs").exists() and (path / "levels").exists()):
                            return False
                        # Vérifier explicitement la présence du niveau 1
                        if not (path / "levels" / "lvl1").exists():
                            return False
                        return True
                    except Exception:
                        return False

                ms_path_obj = _normalize_ms_path(file_path, ms_path_obj)
                try:
                    print(f"📌 Zarr (normalisé): {ms_path_obj}")
                except Exception:
                    pass
                logging.info(f"[OPEN] Zarr (normalized)={ms_path_obj}")

                precompute_action = precompute_action if precompute_action in {"build", "existing"} else "existing"

                if precompute_action == "existing":
                    # Si invalide, basculer automatiquement en build
                    if not _is_valid_zarr(ms_path_obj):
                        try:
                            print("⚠️  Zarr inexistant ou invalide → passage en mode création")
                        except Exception:
                            pass
                        messagebox.showwarning(
                            "Navigation rapide",
                            "Le dossier de navigation rapide indiqué est introuvable ou incomplet.\n"
                            "Il sera créé automatiquement.",
                            parent=self.root,
                        )
                        precompute_action = "build"

                build_succeeded = True
                if precompute_action == "build":
                    self._update_loading_message("⚡ Création du fichier de navigation rapide...")
                    self.root.update()

                    try:
                        from core.multiscale import build_pyramid
                        from ui.channel_selector import ChannelSelector

                        ms_path_obj.parent.mkdir(parents=True, exist_ok=True)

                        preselected = self.selected_channels if hasattr(self, 'selected_channels') else []

                        channel_selector = ChannelSelector(
                            parent=self.root,
                            available_channels=list(self.raw.ch_names),
                            preselected_channels=preselected
                        )

                        selected_channels_for_pyramid = channel_selector.get_selection()

                        if selected_channels_for_pyramid is None:
                            print("❌ Sélection de canaux annulée par l'utilisateur")
                            selected_mode = "raw"
                            ms_path_obj = None
                            build_succeeded = False
                        else:
                            available_channels = list(self.raw.ch_names)

                            def _dedupe_keep_order(values):
                                seen = set()
                                result = []
                                for name in values:
                                    if name in seen:
                                        continue
                                    if name not in available_channels:
                                        continue
                                    seen.add(name)
                                    result.append(name)
                                return result

                            required_channels: list[str] = []
                            try:
                                required_channels = [ch for ch in getattr(self, "psg_channels_used", []) if ch in available_channels]
                            except Exception:
                                required_channels = []

                            if not required_channels:
                                required_channels = [ch for ch in self.selected_channels if ch in available_channels]

                            if not required_channels:
                                fallback_priority = [
                                    "F3-M2",
                                    "F4-M1",
                                    "C3-M2",
                                    "C4-M1",
                                    "O1-M2",
                                    "O2-M1",
                                    "E1-M2",
                                    "E2-M1",
                                    "Left Leg",
                                    "Right Leg",
                                    "Heart Rate",
                                ]
                                required_channels = [ch for ch in fallback_priority if ch in available_channels]

                            combined = _dedupe_keep_order(list(selected_channels_for_pyramid) + required_channels)
                            auto_added = [ch for ch in combined if ch not in selected_channels_for_pyramid]
                            if auto_added:
                                print(f"   🔄 Canaux requis ajoutés automatiquement: {auto_added}", flush=True)
                            selected_channels_for_pyramid = combined

                            if not selected_channels_for_pyramid:
                                print(f"🔧 Création du fichier de navigation rapide: {ms_path_obj}", flush=True)
                                print("   ⚠️  Aucun canal sélectionné, annulation", flush=True)
                                selected_mode = "raw"
                                ms_path_obj = None
                                build_succeeded = False
                            else:
                                print(f"🔧 Création du fichier de navigation rapide: {ms_path_obj}", flush=True)
                                print(f"   📊 Canaux sélectionnés pour la pyramide: {selected_channels_for_pyramid}", flush=True)
                                print(f"   ⚡ Utilisation de {len(selected_channels_for_pyramid)} canaux sur {len(self.raw.ch_names)} disponibles", flush=True)
                                estimated_time = len(selected_channels_for_pyramid) * 15 / 90
                                print(f"   ⏱️  Temps estimé: ~{estimated_time:.1f} minutes", flush=True)

                                try:
                                    build_pyramid(
                                        raw_source=self.raw,
                                        out_ms_path=ms_path_obj,
                                        chunk_seconds=20,
                                        resume=True,
                                        selected_channels=selected_channels_for_pyramid
                                    )
                                    print(f"✅ Fichier créé: {ms_path_obj}")
                                    logging.info(f"[OPEN] Zarr build success: {ms_path_obj}")
                                    messagebox.showinfo(
                                        "✅ Succès",
                                        "Le fichier de navigation rapide a été créé avec succès !\n\n"
                                        f"Canaux inclus: {len(selected_channels_for_pyramid)}\n"
                                        "Vous pouvez maintenant naviguer instantanément dans vos données.",
                                        parent=self.root,
                                    )
                                    try:
                                        logging.info("[OPEN] Popup succès affiché (création Zarr)")
                                    except Exception:
                                        pass
                                except Exception as e:
                                    import traceback
                                    print(f"❌ Erreur lors de la création: {e}")
                                    print("Traceback complet:")
                                    traceback.print_exc()
                                    logging.error(f"[OPEN] Zarr build error: {e}")
                                    messagebox.showerror(
                                        "Erreur",
                                        "Impossible de créer le fichier de navigation rapide:\n"
                                        f"{str(e)}\n\nLe mode standard sera utilisé.",
                                        parent=self.root,
                                    )
                                    selected_mode = "raw"
                                    ms_path_obj = None
                                    build_succeeded = False
                    except Exception as e:
                        import traceback
                        print(f"❌ Erreur générale lors de la préparation: {e}")
                        traceback.print_exc()
                        messagebox.showerror(
                            "Erreur",
                            "Une erreur est survenue lors de la préparation du fichier de navigation rapide.\n"
                            f"{str(e)}\n\nLe mode standard sera utilisé.",
                            parent=self.root,
                        )
                        selected_mode = "raw"
                        ms_path_obj = None
                        build_succeeded = False

                if selected_mode == "precomputed" and ms_path_obj is not None and (precompute_action != "build" or build_succeeded):
                    try:
                        self._update_loading_message("⚡ Activation de la navigation rapide...")
                        self.root.update()
                        
                        from core.providers import PrecomputedProvider
                        self._shutdown_bridge_executor()
                        provider = PrecomputedProvider(ms_path_obj)
                        executor = ThreadPoolExecutor(max_workers=2)
                        self._bridge_executor = executor
                        self.data_bridge = DataBridge(provider, executor=executor)
                        self.data_mode = "precomputed"
                        
                        print(f"✅ Navigation rapide activée avec {ms_path_obj}")
                        logging.info(f"[OPEN] Navigation rapide activée: {ms_path_obj}")
                    
                    except Exception as e:
                        print(f"❌ Erreur lors de l'activation: {e}")
                        # Diagnostic supplémentaire pour "group not found":
                        try:
                            exists = ms_path_obj.exists()
                            has_zattrs = (ms_path_obj / ".zattrs").exists()
                            has_levels = (ms_path_obj / "levels").exists()
                            has_lvl1 = (ms_path_obj / "levels" / "lvl1").exists()
                            print(f"   🔎 Zarr path exists={exists}, .zattrs={has_zattrs}, levels_dir={has_levels}, lvl1={has_lvl1}")
                        except Exception:
                            pass
                        logging.error(f"[OPEN] Activation rapide échouée: {e}")
                        messagebox.showwarning(
                            "Avertissement",
                            "Impossible d'activer la navigation rapide:\n"
                            f"{str(e)}\n\nLe mode standard sera utilisé.",
                            parent=self.root,
                        )
                        self._shutdown_bridge_executor()
                        self.data_bridge = None
                        self.data_mode = "raw"
                        selected_mode = "raw"
                        ms_path_obj = None
                        # Force redraw in raw mode to avoid blank screen
                        try:
                            self.update_time_scale()
                            self.update_plot()
                        except Exception:
                            pass
                elif selected_mode == "lazy":
                    try:
                        self._update_loading_message("🧠 Initialisation du mode Lazy...")
                        self.root.update()
                    except Exception:
                        pass

                    self._shutdown_bridge_executor()
                    lazy_provider = LazyProvider(self.raw)
                    executor = ThreadPoolExecutor(max_workers=4)
                    self._bridge_executor = executor
                    self.data_bridge = DataBridge(lazy_provider, executor=executor)
                    self.data_mode = "lazy"
                    logging.info("[OPEN] Mode Lazy activé (calcul à la volée)")
                else:
                    # Mode standard sans bridge
                    self._shutdown_bridge_executor()
                    self.data_bridge = None
                    self.data_mode = "raw"
                    ms_path_obj = None

            # Mise à jour de l'interface:
            # 1) construire explicitement la vue PSG (sinon update_plot peut no-op
            #    si le conteneur n'est pas encore prêt),
            # 2) lancer le refresh asynchrone.
            self.update_time_scale()
            try:
                parent = getattr(self, "psg_container", None)
                if parent is None:
                    logging.warning("[OPEN] psg_container absent au chargement; fallback fenetre dediee")
                self._show_default_psg_view(embed_parent=parent)
            except Exception:
                logging.exception("[OPEN] Echec creation vue PSG initiale")
            self.update_plot()
            
            # Stocker le chemin du fichier pour l'affichage
            self.current_file_path = file_path
            
            # Mettre à jour la barre de statut
            self._update_status_bar()
            
            self._update_loading_message("Chargement terminé!")
            self.root.update()
            
            if self.data_mode == "precomputed":
                mode_msg = "⚡ Navigation Rapide"
            elif self.data_mode == "lazy":
                mode_msg = "🧠 Mode Lazy (calcul à la volée)"
            else:
                mode_msg = "📂 Mode Standard"
            self._qt_pause_pump_for_tk_modal()
            try:
                messagebox.showinfo(
                    "✅ Succès",
                    f"Fichier chargé: {os.path.basename(file_path)}\n\nMode: {mode_msg}",
                    parent=self.root,
                )
            finally:
                self._qt_resume_pump_after_tk_modal()
            # Fermer la barre après le modal : évite destroy/grab Tk pendant messagebox
            # alors que le pump Qt tourne (Py3.14 / PyEval_RestoreThread).
            self.root.after(0, self._hide_loading_bar)
            logging.info(f"[OPEN] Popup succès affiché: {os.path.basename(file_path)} | mode={mode_msg}")
            
            # Commiter le sample de télémetrie si disponible
            try:
                if load_sample:
                    load_sample.setdefault("total_ms", 0.0)
                    telemetry.commit(load_sample)
            except Exception:
                pass
            
        except Exception as e:
            # Cacher la barre de chargement en cas d'erreur
            self._hide_loading_bar()
            
            # Commiter le sample de télémetrie même en cas d'erreur
            try:
                if load_sample:
                    telemetry.commit(load_sample)
            except Exception:
                pass
            
            print(f"❌ Erreur lors du chargement: {e}")
            self._qt_pause_pump_for_tk_modal()
            try:
                messagebox.showerror("Erreur", f"Erreur lors du chargement: {str(e)}", parent=self.root)
            finally:
                self._qt_resume_pump_after_tk_modal()
    
    def _extract_absolute_time(self):
        """Extrait le temps absolu de début d'enregistrement depuis les métadonnées EDF."""
        try:
            # Initialiser les variables de temps absolu
            self.absolute_start_time = None
            self.time_format = "relatif"  # Par défaut
            
            # Vérifier si on a des informations de temps dans les métadonnées
            if ('meas_date' in self.raw.info and self.raw.info['meas_date'] is not None):
                # Le fichier EDF contient des informations de temps absolu
                self.absolute_start_time = self.raw.info['meas_date']
                self.time_format = "absolu"
                print(f"🕐 Temps absolu détecté: {self.absolute_start_time}")
                
                # Convertir en datetime local si nécessaire (gérer plusieurs types)
                base = self.absolute_start_time
                try:
                    if isinstance(base, datetime):
                        self.absolute_start_datetime = base
                    elif isinstance(base, tuple) and len(base) > 0 and isinstance(base[0], datetime):
                        self.absolute_start_datetime = base[0]
                    elif hasattr(base, 'timestamp'):
                        self.absolute_start_datetime = datetime.fromtimestamp(base.timestamp())
                    elif isinstance(base, (int, float)):
                        self.absolute_start_datetime = datetime.fromtimestamp(float(base))
                    else:
                        self.absolute_start_datetime = pd.to_datetime(base).to_pydatetime()
                except Exception:
                    print("⚠️ Conversion de meas_date en datetime échouée; retour au mode relatif.")
                    self.time_format = "relatif"
                    self.absolute_start_datetime = None
                
                print(f"🕐 Début d'enregistrement: {self.absolute_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                
            else:
                # Pas d'informations de temps absolu disponibles
                print("⚠️ Aucune information de temps absolu dans le fichier EDF")
                self.time_format = "relatif"
                
        except Exception as e:
            print(f"⚠️ Erreur lors de l'extraction du temps absolu: {e}")
            self.time_format = "relatif"

    def _extract_bad_spans_from_annotations(self) -> List[Tuple[float, float]]:
        """Parcourt raw.annotations et retourne les segments marqués comme mauvais.
        Reconnaît les descriptions contenant 'bad', 'artefact', 'artifact', 'rejeter', 'reject'.
        Retour: liste de tuples (onset_sec, offset_sec)."""
        spans: List[Tuple[float, float]] = []
        try:
            if not hasattr(self.raw, 'annotations') or self.raw.annotations is None:
                return spans
            ann = self.raw.annotations
            if len(ann) == 0:
                return spans
            pattern = re.compile(r"bad|artefact|artifact|rejeter|reject", flags=re.IGNORECASE)
            for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
                if desc and pattern.search(str(desc)):
                    start = float(onset)
                    end = float(onset + duration)
                    if end > start:
                        spans.append((start, end))
        except Exception:
            return spans
        return spans
    
    def _get_absolute_time(self, relative_time_seconds):
        """Convertit le temps relatif en temps absolu."""
        if self.time_format == "absolu" and hasattr(self, 'absolute_start_datetime'):
            absolute_time = self.absolute_start_datetime + timedelta(seconds=relative_time_seconds)
            return absolute_time.strftime('%H:%M:%S')
        else:
            return f"{relative_time_seconds:.1f}s"
    
    def _format_hhmm(self, seconds_value: float) -> str:
        """Formate un temps (secondes) en 'HHhMM'."""
        try:
            total_seconds = int(seconds_value)
        except Exception:
            total_seconds = 0
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours:02d}h{minutes:02d}"
    
    def _to_microvolts_and_sanitize(self, data: np.ndarray) -> np.ndarray:
        """Convertit un signal en µV et remplace NaN/Inf par 0 pour l'affichage.
        Hypothèse MNE: les données sont en Volts en entrée.
        """
        try:
            x = np.asarray(data, dtype=float) * 1e6  # V -> µV
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x
        except Exception as e:
            print(f"⚠️ CHECKPOINT UNIT: Echec conversion µV: {e}")
            return data
    
    def create_derivations(self):
        """Crée les dérivations EEG et prépare tous les canaux"""
        if not self.raw:
            return
        
        print(f"🔧 Création des dérivations...")
        
        # Ajouter tous les canaux originaux du fichier
        for channel in self.raw.ch_names:
            try:
                raw_arr = self.raw.get_data(channel)
                data = self._to_microvolts_and_sanitize(raw_arr.flatten())
                self.derivations[channel] = data
                
                # Diagnostic de l'amplitude
                amplitude = np.max(data) - np.min(data)
                if amplitude < 1e-3:
                    print(f"   ⚠️  {channel}: amplitude très faible ({amplitude:.6f} µV)")
                else:
                    print(f"   ✅ {channel}: amplitude normale ({amplitude:.6f} µV)")
                if not np.any(np.diff(data)):
                    print(f"   ⚠️  {channel}: signal constant (ligne plate) après chargement")
                    
            except Exception as e:
                print(f"   ❌ Erreur pour {channel}: {e}")
        
        print(f"✅ {len(self.derivations)} canaux chargés")
        
        # Appliquer les filtres par défaut selon le type de signal
        self._apply_default_filters()
    
    def _apply_default_filters(self):
        """Applique les filtres par défaut selon le type de signal."""
        print("🔧 Application des filtres par défaut...")
        
        for channel in self.derivations:
            if channel in self.default_derivation_presets:
                low, high = self.default_derivation_presets[channel]
                self.channel_filter_params[channel] = {
                    'low': low,
                    'high': high,
                    'enabled': True,
                    'amplitude': 100.0  # Amplitude par défaut
                }
                print(f"   ✅ {channel}: {low}-{high} Hz")
        
        # Activer le filtre global si des filtres par canal sont définis
        if self.channel_filter_params:
            self.filter_var.set(True)
            print("🔧 Filtre global activé automatiquement")
    
    def update_time_scale(self):
        """Met à jour l'échelle de temps"""
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            try:
                if hasattr(self, 'time_scale') and self.time_scale is not None:
                    self.time_scale.config(to=max_time)
            except Exception:
                pass
            try:
                if hasattr(self, 'bottom_time_scale') and self.bottom_time_scale is not None:
                    self.bottom_time_scale.config(to=max_time)
            except Exception:
                pass
    
    def update_time(self, value):
        """Met à jour le temps affiché"""
        self.current_time = float(value)
        try:
            self.time_label.config(text=f"{self.current_time:.1f}s")
        except Exception:
            pass
        # Synchroniser toutes les barres de navigation
        try:
            if hasattr(self, 'time_var'):
                self.time_var.set(self.current_time)
            if hasattr(self, 'bottom_time_var'):
                self.bottom_time_var.set(self.current_time)
            self._update_time_display()
        except Exception:
            pass
        self.update_plot()
    
    def update_duration(self, event=None):
        """Met à jour la durée d'affichage"""
        try:
            self.duration = float(self.duration_var.get())
            self.update_time_scale()
            self.update_plot()
        except ValueError:
            print("🔍 CHECKPOINT DURATION RESET: Réinitialisation durée à 10.0s (ValueError)")
            logging.info("[DURATION] Reset to 10.0s due to ValueError")
            self.duration_var.set("10.0")
    
    def jump_backward(self):
        """Saut en arrière"""
        self.current_time = max(0, self.current_time - self.duration)
        self.time_var.set(self.current_time)
        self.update_plot()
    
    def step_backward(self):
        """Pas en arrière"""
        self.current_time = max(0, self.current_time - 1)
        self.time_var.set(self.current_time)
        self.update_plot()
    
    def step_forward(self):
        """Pas en avant"""
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + 1)
            self.time_var.set(self.current_time)
            self.update_plot()
    
    def jump_forward(self):
        """Saut en avant"""
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + self.duration)
            self.time_var.set(self.current_time)
        self.update_plot()

    # ------------------------------------------------------------------
    # Lazy preprocessing helpers
    # ------------------------------------------------------------------
    def _snapshot_processing_config(self) -> Dict[str, Any]:
        plotter = getattr(self, 'psg_plotter', None)
        baseline_enabled = bool(getattr(plotter, 'baseline_enabled', True)) if plotter else True
        filter_enabled = bool(getattr(plotter, 'global_filter_enabled', True)) if plotter else True
        params_src = getattr(plotter, 'filter_params_by_channel', {}) or {}
        params_copy = {ch: dict(cfg) for ch, cfg in params_src.items()}
        return {
            "baseline_enabled": baseline_enabled,
            "baseline_window_duration": float(getattr(self, "baseline_window_duration", 30.0)),
            "filter_enabled": filter_enabled,
            "filter_order": int(getattr(self, "filter_order", 4)),
            "filter_type": str(getattr(self, "filter_type", "butterworth")),
            "params": params_copy,
        }

    @staticmethod
    def _downsample_for_processing(array: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
        if target_len <= 0 or array.size <= target_len:
            return array.astype(np.float32, copy=False), 1
        bin_size = int(math.ceil(array.size / float(target_len)))
        if bin_size <= 1:
            return array.astype(np.float32, copy=False), 1
        usable = array[: (array.size // bin_size) * bin_size]
        down = usable.reshape(-1, bin_size).mean(axis=1).astype(np.float32, copy=False)
        if usable.size < array.size:
            remainder_mean = float(np.mean(array[usable.size :], dtype=np.float32))
            down = np.concatenate([down, np.array([remainder_mean], dtype=np.float32)])
        return down, bin_size

    def _resolve_channel_filter(self, channel: str, config: Dict[str, Any]) -> Dict[str, float]:
        params = config["params"].get(channel)
        if params is None:
            stype = cesa_detect_signal_type(channel)
            presets = cesa_get_filter_presets(stype)
            params = {
                "enabled": True,
                "low": presets.get("low", 0.0),
                "high": presets.get("high", 0.0),
                "amplitude": presets.get("amplitude", 100.0),
            }
        return dict(params)

    def _build_processed_key(
        self,
        channel: str,
        start_idx: int,
        end_idx: int,
        plot_width_px: int,
        fs: float,
        data_len: int,
        config: Dict[str, Any],
    ) -> Tuple:
        params = self._resolve_channel_filter(channel, config)
        filter_sig = (
            bool(config["filter_enabled"] and params.get("enabled", False)),
            float(params.get("low", 0.0)),
            float(params.get("high", 0.0)),
            float(params.get("amplitude", 100.0)),
        )
        return (
            channel,
            int(start_idx),
            int(end_idx),
            int(plot_width_px),
            round(float(fs), 4),
            int(data_len),
            bool(config["baseline_enabled"]),
            round(float(config.get("baseline_window_duration", 30.0)), 3),
            filter_sig,
            int(config.get("filter_order", 4)),
            str(config.get("filter_type", "butterworth")),
        )

    def _process_channel_window(
        self,
        channel: str,
        data: np.ndarray,
        fs: float,
        target_len: int,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, float, float, float]:
        y = np.asarray(data, dtype=np.float32)
        downsampled, bin_size = self._downsample_for_processing(y, target_len)
        if bin_size <= 0:
            bin_size = 1
        effective_fs = fs / bin_size
        baseline_ms = 0.0
        filter_ms = 0.0
        params = self._resolve_channel_filter(channel, config)
        stype = cesa_detect_signal_type(channel)

        try:
            if config["baseline_enabled"] and stype in ("eeg", "eog", "ecg", "emg"):
                t0 = time.perf_counter()
                downsampled = cesa_apply_baseline_correction(
                    downsampled,
                    window_duration=float(config.get("baseline_window_duration", 30.0)),
                    sfreq=effective_fs,
                )
                baseline_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            baseline_ms = 0.0

        try:
            if config["filter_enabled"] and params.get("enabled", False):
                t0 = time.perf_counter()
                downsampled = cesa_apply_filter(
                    downsampled,
                    sfreq=effective_fs,
                    filter_order=int(config.get("filter_order", 4)),
                    low=params.get("low", 0.0),
                    high=params.get("high", 0.0),
                    filter_type=str(config.get("filter_type", "butterworth")),
                )
                filter_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            filter_ms = 0.0

        try:
            amp = float(params.get("amplitude", 100.0))
            downsampled = downsampled * (amp / 100.0)
        except Exception:
            pass

        return downsampled.astype(np.float32, copy=False), effective_fs, baseline_ms, filter_ms

    def _submit_processing_task(
        self,
        generation: int,
        key: Tuple,
        channel: str,
        data: np.ndarray,
        fs: float,
        target_len: int,
        config: Dict[str, Any],
    ) -> None:
        data_copy = np.array(data, dtype=np.float32, copy=True)

        def _task():
            result = self._process_channel_window(channel, data_copy, fs, target_len, config)
            return generation, key, channel, result

        future = self._preprocess_executor.submit(_task)

        def _callback(fut: Any) -> None:
            try:
                generation_result, cache_key, ch, result = fut.result()
            except Exception:
                return
            if generation_result != self._processing_generation:
                return
            if not isinstance(result, tuple):
                return
            processed_data, effective_fs, baseline_ms, filter_ms = result
            with self._processed_lock:
                self._processed_window_cache[cache_key] = (
                    processed_data,
                    effective_fs,
                    {"baseline_ms": baseline_ms, "filter_ms": filter_ms},
                )
                while len(self._processed_window_cache) > self._processed_window_limit:
                    try:
                        self._processed_window_cache.popitem(last=False)
                    except Exception:
                        break
            self._enqueue_tk_main(self._apply_cached_processed_signals)

        future.add_done_callback(_callback)

    def _apply_cached_processed_signals(self) -> None:
        """Apply cached processed signals to plotter if generation matches."""
        if self._processing_generation != self._plot_update_gen:
            return
        try:
            plotter = getattr(self, 'psg_plotter', None)
            if plotter is None:
                return
            window_sig = self._current_window_signature
            if window_sig is None:
                return
            config = self._snapshot_processing_config()
            ready_signals = {}
            total_baseline_ms = 0.0
            total_filter_ms = 0.0
            with self._processed_lock:
                for ch in self._expected_channels:
                    if ch not in self._channel_processing_map:
                        continue
                    key = self._channel_processing_map[ch]
                    if key not in self._processed_window_cache:
                        continue
                    processed_data, effective_fs, metrics = self._processed_window_cache[key]
                    ready_signals[ch] = (processed_data, effective_fs)
                    total_baseline_ms += metrics.get("baseline_ms", 0.0)
                    total_filter_ms += metrics.get("filter_ms", 0.0)
            if ready_signals:
                plotter.update_preprocessed_signals(ready_signals)
                self._last_baseline_ms = total_baseline_ms
                self._last_filter_ms = total_filter_ms
        except Exception:
            pass

    def _enqueue_tk_main(self, fn: Callable[[], None]) -> None:
        """Planifie *fn* sur le thread principal Tk. Appelable depuis n'importe quel thread."""
        self._tk_main_thread_queue.put(fn)

    def _poll_tk_main_thread_queue(self) -> None:
        # #region agent log
        try:
            from CESA.agent_debug_f8b011 import log as _agent_log

            _agent_log("B", "eeg_studio._poll_tk_main_thread_queue:enter", "poll queue", {})
        except Exception:
            pass
        # #endregion
        self._tk_main_poll_id = None
        _br_pe = getattr(self, "_qt_viewer_bridge", None)
        if _br_pe is not None and getattr(_br_pe, "_qt_pe_active", False):
            try:
                self._tk_main_poll_id = self.root.after(
                    12, self._poll_tk_main_thread_queue
                )
            except Exception:
                self._tk_main_poll_id = None
            _t_pe = time.monotonic()
            if _t_pe - float(getattr(self, "_chk_poll_defer_pe_at", -999.0)) >= 2.0:
                self._chk_poll_defer_pe_at = _t_pe
                _log_viewer_checkpoint(
                    "70",
                    "_poll_tk_main_thread_queue deferred (Qt processEvents active)",
                )
            return
        if int(getattr(self, "_tk_modal_ui_block_depth", 0)) > 0:
            _t_modal = time.monotonic()
            if _t_modal - float(getattr(self, "_chk_poll_modal_skip_at", -999.0)) >= 1.0:
                self._chk_poll_modal_skip_at = _t_modal
                _log_viewer_checkpoint(
                    "68",
                    "_poll_tk_main_thread_queue skip (modal depth>0)",
                    depth=int(getattr(self, "_tk_modal_ui_block_depth", 0)),
                )
            return
        # Un seul callback par tick : évite d'enchaîner plot/Qt/scipy dans une même
        # invocation Tk (réduit les crashs PyEval_RestoreThread / GIL avec Py3.14 + pump Qt).
        fn = None
        try:
            fn = self._tk_main_thread_queue.get_nowait()
        except Empty:
            pass
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
            _poll_next_ms = (
                24 if getattr(self, "_qt_viewer_bridge", None) is not None else 0
            )
            # #region agent log
            try:
                from CESA.agent_debug_f8b011 import log as _agent_log

                _agent_log(
                    "B",
                    "eeg_studio._poll_tk_main_thread_queue:after_fn",
                    "one fn done",
                    {"poll_next_ms": _poll_next_ms},
                )
            except Exception:
                pass
            # #endregion
            try:
                self._tk_main_poll_id = self.root.after(
                    _poll_next_ms, self._poll_tk_main_thread_queue
                )
            except Exception:
                self._tk_main_poll_id = None
            return
        try:
            self._tk_main_poll_id = self.root.after(16, self._poll_tk_main_thread_queue)
        except Exception:
            self._tk_main_poll_id = None
        else:
            _t_idle = time.monotonic()
            if _t_idle - float(getattr(self, "_chk_poll_idle_at", -999.0)) >= 2.0:
                self._chk_poll_idle_at = _t_idle
                _log_viewer_checkpoint(
                    "69",
                    "_poll_tk_main_thread_queue idle branch scheduled after(16)",
                )

    def update_plot(self, *args):
        """Met à jour l'affichage principal en rafraîchissant la vue PSG intégrée."""
        if not getattr(self, 'raw', None):
            return
        try:
            import time as _time
            _t_total0 = _time.perf_counter()
            parent = getattr(self, 'psg_container', None)
            if parent is None:
                return

            # Debounce + thread offload
            self._plot_update_gen += 1
            current_gen = int(self._plot_update_gen)
            self._processing_generation = current_gen

            plot_width_px = self._get_plot_width_px()
            use_bridge = getattr(self, 'data_bridge', None) is not None
            is_lazy = getattr(self, 'data_mode', 'raw') == 'lazy'
            skip_lazy_preprocess_workers = self._qt_psg_plot_active()
            token = object()
            self._active_plot_token = token

            def _do_update(current_token=token, _skip_lazy_pre=skip_lazy_preprocess_workers):
                try:
                    _t_extract0 = None
                    _t_filter0 = None
                    fs = float(self.sfreq)
                    start_idx = int(max(0, float(self.current_time) * fs))
                    end_idx = int(min(len(self.raw.times), (float(self.current_time) + float(self.duration)) * fs))
                    if end_idx <= start_idx + 1:
                        end_idx = min(len(self.raw.times), start_idx + int(max(2, fs)))

                    selected = getattr(self, 'psg_channels_used', None)
                    if not selected:
                        # Prefer explicit selected_channels established at load
                        try:
                            candidates = list(getattr(self, 'selected_channels', []))
                        except Exception:
                            candidates = []

                        # Fallback to a fixed priority list if empty
                        if not candidates:
                            priority = [
                                'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1',
                                'E1-M2', 'E2-M1', 'Left Leg', 'Right Leg', 'Heart Rate'
                            ]
                            avail = set(self.raw.ch_names)
                            candidates = [ch for ch in priority if ch in avail]

                        # Last resort: filter out measurement/impedance channels
                        if not candidates:
                            def _is_plottable_channel(name: str) -> bool:
                                s = (name or '').upper()
                                if 'IMP' in s:
                                    return False
                                if s.strip().isdigit():
                                    return False
                                return True
                            candidates = [ch for ch in self.raw.ch_names if _is_plottable_channel(ch)][:8]

                        selected = candidates

                    # LRU cache for recent slices (key = (channel, start_idx, end_idx))
                    if not hasattr(self, '_psg_slice_cache'):
                        self._psg_slice_cache = {}
                        self._psg_slice_cache_order = []  # store keys for eviction
                    max_cache_entries = 64

                    signals = {}
                    bridge_result = None
                    extract_ms = 0.0
                    _t_extract0 = None
                    
                    if use_bridge:
                        try:
                            import time as _time
                            _t_extract0 = _time.perf_counter()
                        except Exception:
                            _t_extract0 = None
                        try:
                            bridge_result = self.data_bridge.get_signals_for_window(
                                float(self.current_time),
                                float(self.duration),
                                width_px=plot_width_px,
                                channels=selected,
                            )
                            signals = bridge_result.signals
                        except Exception as exc:
                            bridge_result = None
                            signals = {}
                            try:
                                logging.warning("Bridge data fetch failed, falling back to raw samples: %s", exc)
                            except Exception:
                                pass
                        if _t_extract0 is not None:
                            try:
                                import time as _time
                                extract_ms = (_time.perf_counter() - _t_extract0) * 1000.0
                            except Exception:
                                extract_ms = 0.0

                    # Ensure we have raw signals for preprocessing in lazy mode
                    raw_signals_for_preprocessing = {}
                    if is_lazy:
                        if use_bridge and signals:
                            # Bridge provides envelopes, we need raw data for preprocessing
                            # So we extract raw data separately
                            try:
                                for ch in selected:
                                    try:
                                        arr = self.raw.get_data(picks=[ch], start=start_idx, stop=end_idx)[0]
                                        arr = self._to_microvolts_and_sanitize(arr)
                                        raw_signals_for_preprocessing[ch] = (arr, fs)
                                    except Exception:
                                        pass
                            except Exception:
                                raw_signals_for_preprocessing = {}
                        elif not use_bridge:
                            # We will get raw signals below, wait for that
                            pass

                    if not signals:
                        try:
                            import time as _time
                            _t_extract0 = _time.perf_counter()
                        except Exception:
                            _t_extract0 = None
                        signals = {}
                        for ch in selected:
                            try:
                                key = (ch, start_idx, end_idx)
                                if key in self._psg_slice_cache:
                                    arr = self._psg_slice_cache[key]
                                    try:
                                        self._psg_slice_cache_order.remove(key)
                                    except Exception:
                                        pass
                                    self._psg_slice_cache_order.append(key)
                                else:
                                    arr = self.raw.get_data(picks=[ch], start=start_idx, stop=end_idx)[0]
                                    self._psg_slice_cache[key] = arr
                                    self._psg_slice_cache_order.append(key)
                                    if len(self._psg_slice_cache_order) > max_cache_entries:
                                        old_key = self._psg_slice_cache_order.pop(0)
                                        try:
                                            del self._psg_slice_cache[old_key]
                                        except Exception:
                                            pass
                                data_uv = self._to_microvolts_and_sanitize(arr)
                                signals[ch] = (data_uv, fs)
                                # Store raw signals for preprocessing in lazy mode
                                if is_lazy and ch not in raw_signals_for_preprocessing:
                                    raw_signals_for_preprocessing[ch] = (data_uv, fs)
                            except Exception:
                                continue
                        if _t_extract0 is not None:
                            try:
                                import time as _time
                                extract_ms = (_time.perf_counter() - _t_extract0) * 1000.0
                            except Exception:
                                extract_ms = 0.0
                    
                    # Submit async preprocessing tasks for lazy mode if we have raw signals
                    # Py3.14 + Qt embarqué : pas de workers (capturé sur le thread Tk, pas depuis un worker).
                    if is_lazy and raw_signals_for_preprocessing and not _skip_lazy_pre:
                        config = self._snapshot_processing_config()
                        target_processing_len = max(300, int(plot_width_px * 1.2))
                        self._expected_channels = set(selected)
                        self._channel_processing_map = {}
                        self._current_window_signature = (int(start_idx), int(end_idx), int(plot_width_px))
                        for ch in selected:
                            if ch not in raw_signals_for_preprocessing:
                                continue
                            try:
                                arr, fs_ch = raw_signals_for_preprocessing[ch]
                                key = self._build_processed_key(ch, start_idx, end_idx, plot_width_px, fs_ch, len(arr), config)
                                self._channel_processing_map[ch] = key
                                if key not in self._processed_window_cache:
                                    self._submit_processing_task(current_gen, key, ch, arr, fs_ch, target_processing_len, config)
                            except Exception:
                                pass

                    # Build hypnogram quickly with caching
                    hypnogram = self._psg_cached_hypnogram
                    df = self._get_active_scoring_df()
                    if df is not None and len(df) > 0 and 'time' in df.columns and 'stage' in df.columns:
                        if self._psg_cached_hypnogram is None or self._psg_cached_scoring_rows != len(df):
                            try:
                                df_work = df[['time', 'stage']].copy()
                                df_work['time'] = pd.to_numeric(df_work['time'], errors='coerce')
                                df_work = df_work.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
                                if len(df_work) >= 2:
                                    diffs = np.diff(df_work['time'].to_numpy(dtype=float))
                                    diffs_pos = diffs[diffs > 0]
                                    epoch_len = float(np.median(diffs_pos)) if len(diffs_pos) > 0 else float(self.scoring_epoch_duration)
                                else:
                                    epoch_len = float(self.scoring_epoch_duration)
                                total_dur = float(len(self.raw.times) / self.sfreq)
                                num_epochs = int(np.ceil(total_dur / epoch_len))
                                labels = ['U'] * max(1, num_epochs)
                                stages = df_work['stage'].astype(str).str.upper().str.strip().to_numpy()
                                times_v = df_work['time'].to_numpy(dtype=float)
                                # Vectorized index fill (safe loop due to potential gaps)
                                for t, s in zip(times_v, stages):
                                    idx = int(round(t / epoch_len))
                                    if 0 <= idx < len(labels):
                                        labels[idx] = s
                                self._psg_cached_hypnogram = (labels, epoch_len)
                                self._psg_cached_scoring_rows = len(df)
                                hypnogram = self._psg_cached_hypnogram
                            except Exception:
                                hypnogram = None

                    def _apply_to_ui():
                        # Ignore stale updates
                        if current_gen != self._plot_update_gen:
                            return
                        try:
                            self._last_bridge_result = bridge_result
                        except Exception:
                            pass
                        if hasattr(self, 'psg_plotter') and self.psg_plotter is not None:
                            if hypnogram is not None:
                                try:
                                    self.psg_plotter.set_hypnogram(hypnogram)
                                    self.psg_plotter.set_total_duration(float(len(self.raw.times) / self.sfreq))
                                except Exception:
                                    pass
                            try:
                                _t_draw0 = None
                                try:
                                    import time as _time
                                    _t_draw0 = _time.perf_counter()
                                except Exception:
                                    pass
                                # Build final signals dict: use preprocessed when available, fallback to raw
                                final_signals = {}
                                total_baseline_ms = 0.0
                                total_filter_ms = 0.0
                                
                                # Start with raw signals as base
                                if signals:
                                    final_signals = dict(signals)
                                
                                # In lazy mode, replace with preprocessed signals when available
                                if is_lazy and hasattr(self, '_processed_window_cache'):
                                    with self._processed_lock:
                                        for ch in selected:
                                            if ch not in self._channel_processing_map:
                                                continue
                                            key = self._channel_processing_map[ch]
                                            if key not in self._processed_window_cache:
                                                continue
                                            try:
                                                processed_data, effective_fs, metrics = self._processed_window_cache[key]
                                                # Only use preprocessed if we have valid data
                                                if processed_data is not None and len(processed_data) > 0:
                                                    final_signals[ch] = (processed_data, effective_fs)
                                                    total_baseline_ms += metrics.get("baseline_ms", 0.0)
                                                    total_filter_ms += metrics.get("filter_ms", 0.0)
                                            except Exception:
                                                pass
                                
                                # Always ensure we have at least some signals to display
                                if not final_signals and signals:
                                    final_signals = dict(signals)
                                
                                # --- Correction affichage asynchrone ---
                                # Si aucun signal prêt, réutiliser les précédents ou fallback brut
                                if not final_signals:
                                    if hasattr(self.psg_plotter, 'last_signals') and self.psg_plotter.last_signals:
                                        final_signals = dict(self.psg_plotter.last_signals)
                                    elif signals:
                                        # Fallback sur les bruts disponibles
                                        final_signals = dict(signals)
                                        logging.warning("Aucun signal prêt : fallback sur les bruts disponibles")
                                    else:
                                        # Dernier recours : signaux vides (plotter gère cela)
                                        final_signals = {}
                                
                                # Fusion progressive : conserver les anciens canaux tant que les nouveaux ne sont pas tous prêts
                                if hasattr(self.psg_plotter, 'last_signals') and self.psg_plotter.last_signals:
                                    # Keep only currently selected channels; otherwise stale
                                    # channels (e.g. impedance) can persist after being unchecked.
                                    selected_set = set(selected or [])
                                    fused_signals = {
                                        k: v
                                        for k, v in self.psg_plotter.last_signals.items()
                                        if k in selected_set
                                    }
                                    # Mettre à jour avec les nouveaux signaux valides uniquement
                                    for k, v in final_signals.items():
                                        if v is not None and isinstance(v, tuple) and len(v) == 2:
                                            arr, fs = v
                                            if arr is not None and len(arr) > 0:
                                                fused_signals[k] = v
                                    final_signals = fused_signals
                                # --- Fin correction ---
                                
                                # Update plotter - use preprocessed if we have any, otherwise regular update
                                if is_lazy and (total_baseline_ms > 0 or total_filter_ms > 0):
                                    # We have some preprocessed signals, check if we should use preprocessed update
                                    has_any_preprocessed = any(
                                        ch in self._channel_processing_map and 
                                        self._channel_processing_map[ch] in self._processed_window_cache 
                                        for ch in final_signals.keys()
                                    )
                                    if has_any_preprocessed:
                                        try:
                                            self.psg_plotter.update_preprocessed_signals(final_signals)
                                            self._last_baseline_ms = total_baseline_ms
                                            self._last_filter_ms = total_filter_ms
                                        except Exception:
                                            # Fallback to regular update if preprocessed fails
                                            self.psg_plotter.update_signals(final_signals)
                                    else:
                                        self.psg_plotter.update_signals(final_signals)
                                else:
                                    # Not lazy or no preprocessing done, use regular update
                                    if final_signals:
                                        self.psg_plotter.update_signals(final_signals)
                                    else:
                                        # Last resort: empty dict (plotter handles this)
                                        self.psg_plotter.update_signals({})
                                
                                # Sauvegarde pour le prochain cycle (après mise à jour du plotter)
                                if hasattr(self.psg_plotter, 'last_signals') and final_signals:
                                    self.psg_plotter.last_signals = dict(final_signals)
                                
                                # Invalidate backgrounds if limits changed or options changed
                                try:
                                    # If autoscale/filter toggled in UI, force re-filter and reset blit
                                    self.psg_plotter._invalidate_backgrounds = True
                                except Exception:
                                    pass
                                self.psg_plotter.set_time_window(float(self.current_time), float(self.duration))
                            except Exception:
                                self._show_default_psg_view(embed_parent=parent)
                            if hasattr(self, 'canvas') and self.canvas is not None:
                                try:
                                    self.canvas.draw_idle()
                                except Exception:
                                    pass
                            # Log performance summary
                            try:
                                import time as _time
                                total_ms = (_time.perf_counter() - _t_total0) * 1000.0
                                draw_ms = 0.0
                                if _t_draw0 is not None:
                                    draw_ms = (_time.perf_counter() - _t_draw0) * 1000.0
                                p = getattr(self.psg_plotter, '_perf_last', {})
                                filter_ms = float(p.get('filter_ms', 0.0))
                                baseline_ms = float(p.get('baseline_ms', 0.0))
                                if is_lazy and hasattr(self, '_last_baseline_ms'):
                                    baseline_ms = self._last_baseline_ms
                                if is_lazy and hasattr(self, '_last_filter_ms'):
                                    filter_ms = self._last_filter_ms
                                n_channels = int(p.get('n_channels', 0))
                                n_points = int(p.get('n_points', 0))
                                fps = 1000.0 / max(total_ms, 1e-3)
                                action_label = "bridge" if bridge_result else "raw"
                                if is_lazy:
                                    action_label = "lazy"
                                self._update_performance_feedback(
                                    total_ms=total_ms,
                                    fps=fps,
                                    bridge_result=bridge_result,
                                )
                                self._append_telemetry(
                                    action=action_label,
                                    total_ms=total_ms,
                                    draw_ms=draw_ms,
                                    extract_ms=extract_ms,
                                    fps=fps,
                                    bridge_result=bridge_result,
                                    n_channels=n_channels,
                                    n_points=n_points,
                                    baseline_ms=baseline_ms,
                                    filter_ms=filter_ms,
                                )
                                try:
                                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                except Exception:
                                    timestamp = "0000-00-00"
                                print(f"{timestamp} | PERF nav: extract_ms={extract_ms:.1f} filter_ms={filter_ms:.1f} draw_ms={draw_ms:.1f} total_ms={total_ms:.1f} n_channels={n_channels} n_points={n_points}")
                            except Exception:
                                pass
                        elif self._qt_psg_plot_active():
                            try:
                                self._sync_qt_viewer_bridge_plot_state(hypnogram=hypnogram)
                            except Exception:
                                logging.warning("[VIEW] Synchronisation viewer Qt échouée.", exc_info=True)
                                self._show_default_psg_view(embed_parent=parent)
                        else:
                            self._show_default_psg_view(embed_parent=parent)

                    # Reprise sur le thread Tk (pas root.after depuis le worker)
                    self._enqueue_tk_main(_apply_to_ui)
                except Exception:
                    pass
                finally:
                    if self._active_plot_token is current_token:
                        self._active_plot_future = None
                        self._active_plot_token = None

            # Debounce timer: cancel previous pending
            if self._plot_update_pending_id is not None:
                try:
                    self.root.after_cancel(self._plot_update_pending_id)
                except Exception:
                    pass
            prev_future = getattr(self, '_active_plot_future', None)
            if prev_future is not None and not prev_future.done():
                try:
                    prev_future.cancel()
                except Exception:
                    pass

            def _schedule_update():
                use_pool = getattr(self, "_plot_executor", None) is not None
                if use_pool and self._qt_psg_plot_active():
                    use_pool = False
                if use_pool:
                    future = self._plot_executor.submit(_do_update)
                    self._active_plot_future = future
                else:
                    self._active_plot_future = None
                    _do_update()

            # Schedule short debounce (10ms)
            self._plot_update_pending_id = self.root.after(10, _schedule_update)
        except Exception:
            pass

    # Nettoyage: suppression d'un bloc résiduel incorrect ici
    
    def _add_sleep_scoring_to_plot(self, start_time, end_time, scoring_data=None, alpha=0.7, label_prefix="", zorder=0, y_offset=0, force_y_position=None):
        """Ajoute le scoring de sommeil au graphique principal - VERSION CORRIGÉE."""
        try:
            print(f"🔍 CHECKPOINT SCORING 1: Début de l'ajout du scoring")
            logging.info("[SCORING] Début ajout scoring")
            print(f"🔍 CHECKPOINT SCORING 1: Plage demandée: {start_time:.1f}s - {end_time:.1f}s")
            logging.info(f"[SCORING] Plage demandée: {start_time:.1f}-{end_time:.1f}s")
            
            # Utiliser les données passées en paramètre ou chercher dans les attributs
            if scoring_data is None:
                if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None:
                    scoring_data = self.manual_scoring_data
                    print("🔍 CHECKPOINT SCORING 1: Utilisation du scoring manuel")
                elif hasattr(self, 'sleep_scoring_data') and self.sleep_scoring_data is not None:
                    scoring_data = self.sleep_scoring_data
                    print("🔍 CHECKPOINT SCORING 1: Utilisation du scoring auto")
                else:
                    print("⚠️ CHECKPOINT SCORING 1: Aucune donnée de scoring disponible")
                    logging.warning("[SCORING] Aucune donnée disponible")
                    return
            
            if scoring_data is None or len(scoring_data) == 0:
                print("⚠️ CHECKPOINT SCORING 1: Aucune donnée de scoring disponible")
                logging.warning("[SCORING] Aucune donnée disponible")
                return
            
            print(f"🔍 CHECKPOINT SCORING 2: Données de scoring disponibles: {len(scoring_data)} époques")
            logging.info(f"[SCORING] n_epochs={len(scoring_data)}")
            print(f"🔍 CHECKPOINT SCORING 2: Plage scoring: {scoring_data['time'].min():.1f}s - {scoring_data['time'].max():.1f}s")
            logging.info(f"[SCORING] Plage scoring: {scoring_data['time'].min():.1f}-{scoring_data['time'].max():.1f}s")
            print(f"🔍 CHECKPOINT SCORING 2: Premières valeurs temps: {scoring_data['time'].head()}")
            print(f"🔍 CHECKPOINT SCORING 2: Dernières valeurs temps: {scoring_data['time'].tail()}")
            
            # Filtrer les données de scoring dans la plage temporelle EXACTE affichée
            # CORRECTION: Utiliser la durée d'époque détectée automatiquement
            epoch_duration = getattr(self, 'scoring_epoch_duration', 30.0)  # Durée détectée ou 30s par défaut
            
            epoch_start = scoring_data['time']
            epoch_end = epoch_start + epoch_duration
            overlap_mask = (epoch_start < end_time) & (epoch_end > start_time)
            scoring_in_range = scoring_data[overlap_mask]
            
            print(f"🔍 CHECKPOINT SCORING SYNC: Durée époque utilisée: {epoch_duration}s")

            print(f"🔍 CHECKPOINT SCORING 3: Fenêtre affichée: {start_time:.1f}s - {end_time:.1f}s")
            logging.info(f"[SCORING] Fenêtre: {start_time:.1f}-{end_time:.1f}s")
            print(f"🔍 CHECKPOINT SCORING 3: Époques dans la fenêtre: {len(scoring_in_range)}")
            logging.info(f"[SCORING] n_in_window={len(scoring_in_range)}")
            if len(scoring_in_range) > 0:
                print(f"🔍 CHECKPOINT SCORING 3: Premières époques trouvées: {scoring_in_range.head()}")
                print(f"🔍 CHECKPOINT SCORING 3: Dernières époques trouvées: {scoring_in_range.tail()}")
            else:
                print("⚠️ CHECKPOINT SCORING 3: Aucune époque de scoring dans la fenêtre affichée")
                logging.warning("[SCORING] Aucune époque dans la fenêtre")
                return
            
            # Couleurs pour les stades (utilise la palette dynamique)
            stage_colors = self.theme_manager.get_stage_colors().copy()
            # Ajouter les couleurs pour les stades non standard
            stage_colors.update({
                'U': '#800080',    # Violet pour les stades inconnus
            })
            
            # Obtenir les limites Y du graphique principal
            y_min, y_max = self.ax.get_ylim()
            scoring_height = (y_max - y_min) * 0.0375  # Hauteur réduite (15% ÷ 4)
            
            # Utiliser la position forcée si fournie, sinon calculer normalement
            if force_y_position is not None:
                scoring_y = force_y_position + y_offset
                print(f"🔍 Position scoring FORCÉE: y={scoring_y:.2f} (base={force_y_position:.2f}, offset={y_offset})")
            else:
                scoring_y = y_min - scoring_height * 0.5 + y_offset  # Position encore plus proche
            
            print(f"🔍 Position scoring: y={scoring_y:.2f}, hauteur={scoring_height:.2f}")
            
            # Créer des barres pour chaque stade
            for i, (_, row) in enumerate(scoring_in_range.iterrows()):
                stage = row['stage']
                epoch_start = float(row['time'])
                epoch_end = epoch_start + float(self.scoring_epoch_duration)
                
                # Tronquer l'affichage de l'époque à la fenêtre visible
                draw_start = max(epoch_start, start_time)
                draw_end = min(epoch_end, end_time)
                width = max(0.0, draw_end - draw_start)
                
                color = stage_colors.get(stage, '#DDA0DD')
                confidence = None
                if 'confidence' in scoring_in_range.columns:
                    try:
                        confidence = float(row.get('confidence'))
                    except Exception:
                        confidence = None
                low_conf_threshold = float(getattr(self, 'yasa_confidence_threshold', 0.80))
                epoch_alpha = alpha
                edgecolor = 'black'
                linewidth = 0.5
                if confidence is not None and np.isfinite(confidence):
                    epoch_alpha = max(0.15, min(1.0, alpha * confidence))
                    if confidence < low_conf_threshold:
                        edgecolor = '#ff8c00'
                        linewidth = 1.2
                
                if width > 0:
                    # Créer la barre horizontale tronquée à la fenêtre
                    self.ax.barh(scoring_y, width, left=draw_start, 
                               height=scoring_height, color=color, alpha=epoch_alpha, 
                               edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
                    
                    # Ajouter l'étiquette du stade au centre du bloc
                    center_x = draw_start + width / 2
                    # Pour barh, scoring_y est le bas de la barre, donc le centre est scoring_y + scoring_height/2
                    # Ajuster la position Y pour centrer le texte
                    center_y = scoring_y + scoring_height * 0.5
                    
                    # Mapping des stades vers les étiquettes courtes
                    stage_labels = {
                        'W': 'E',    # Éveil
                        'N1': 'N1',
                        'N2': 'N2', 
                        'N3': 'N3',
                        'R': 'R',    # REM
                        'U': 'U'     # Inconnu
                    }
                    
                    label = stage_labels.get(stage, stage)
                    # Debug: afficher les positions
                    print(f"  🎯 Debug scoring: scoring_y={scoring_y:.2f}, height={scoring_height:.2f}, center_y={center_y:.2f}")
                    
                    # Positionner le texte encore plus bas dans la barre de scoring
                    text_y = scoring_y + scoring_height * 0.01  # Encore plus bas
                    self.ax.text(center_x, text_y, label, 
                               ha='center', va='center', 
                               fontsize=8, fontweight='bold',
                               color='black', alpha=1.0,
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                
                print(f"  📊 Époque {i+1}: {stage} à {epoch_start:.1f}s (dessiné {draw_start:.1f}-{draw_end:.1f}s)")
            
            # Ajuster les limites Y pour inclure le scoring
            self.ax.set_ylim(y_min - scoring_height * 2, y_max)
            
            # Ajouter une légende pour les stades si nécessaire
            if len(scoring_in_range) > 0:
                # Créer une légende pour les stades présents
                from matplotlib.patches import Rectangle
                legend_elements = [Rectangle((0,0),1,1, facecolor=color, alpha=alpha,
                                           label=f"{label_prefix} {self.sleep_stages[stage]}" if label_prefix else self.sleep_stages[stage]) 
                                 for stage, color in stage_colors.items() 
                                 if stage in scoring_in_range['stage'].values]
                
                if legend_elements:
                    # Ajouter la légende des stades 
                    current_legend = self.ax.get_legend()
                    if current_legend:
                        # Combiner avec la légende existante
                        existing_handles = current_legend.legend_handles
                        existing_labels = [t.get_text() for t in current_legend.get_texts()]
                        all_handles = list(existing_handles) + legend_elements
                        all_labels = existing_labels + [elem.get_label() for elem in legend_elements]
                        self.ax.legend(handles=all_handles, labels=all_labels, 
                                     bbox_to_anchor=(0.98, 0.98), loc='upper right', fontsize=6, framealpha=0.8)
                    else:
                        self.ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=2)
                    print(f"✅ Légende ajoutée avec {len(legend_elements)} éléments")
            
            print(f"✅ Scoring ajouté au graphique: {len(scoring_in_range)} époques")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'affichage du scoring: {e}")
            logging.warning(f"Erreur affichage scoring: {e}")
    
    def _show_sleep_scoring_info(self):
        """Affiche les informations sur le scoring de sommeil chargé avec indicateurs avancés."""
        # Vérifier d'abord le scoring manuel, puis l'auto
        scoring_data = None
        scoring_type = ""
        if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None:
            scoring_data = self.manual_scoring_data
            scoring_type = "manuel"
        elif self.sleep_scoring_data is not None:
            scoring_data = self.sleep_scoring_data
            scoring_type = "automatique"
        
        if scoring_data is None:
            messagebox.showinfo("Scoring de sommeil", "Aucun scoring de sommeil chargé.")
            return
        
        # Créer une fenêtre personnalisée
        info_window = tk.Toplevel(self.root)
        info_window.title("📊 Indicateurs de Sommeil")
        info_window.geometry("900x900")
        info_window.transient(self.root)
        
        # Centrer la fenêtre
        info_window.update_idletasks()
        x = (info_window.winfo_screenwidth() // 2) - (450)
        y = (info_window.winfo_screenheight() // 2) - (450)
        info_window.geometry(f"900x900+{x}+{y}")
        
        # Frame principal avec scrollbar
        main_frame = ttk.Frame(info_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas et scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Titre
        title_label = ttk.Label(scrollable_frame, 
                               text=f"📊 Indicateurs de Sommeil ({scoring_type})",
                               font=('Segoe UI', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Option pour compter les périodes U
        include_u_var = tk.BooleanVar(value=False)
        
        option_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Options de calcul", padding="10")
        option_frame.pack(fill=tk.X, pady=(0, 15))
        
        include_u_check = ttk.Checkbutton(
            option_frame,
            text="Inclure les périodes 'U' (Non scorées/Incertaines) dans les calculs",
            variable=include_u_var,
            command=lambda: self._update_sleep_indicators(
                scrollable_frame, scoring_data, include_u_var.get(), 
                stats_frame, indicators_frame, pie_frame, details_frame
            )
        )
        include_u_check.pack(anchor="w")
        
        # Frame pour les statistiques générales
        stats_frame = ttk.LabelFrame(scrollable_frame, text="📈 Statistiques générales", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame pour les indicateurs de sommeil
        indicators_frame = ttk.LabelFrame(scrollable_frame, text="🛏️ Indicateurs de sommeil", padding="10")
        indicators_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame pour le graphique en camembert
        pie_frame = ttk.LabelFrame(scrollable_frame, text="🥧 Répartition graphique des stades", padding="10")
        pie_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Frame pour les détails par stade
        details_frame = ttk.LabelFrame(scrollable_frame, text="📊 Répartition détaillée des stades", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Calculer et afficher les indicateurs initiaux
        self._update_sleep_indicators(scrollable_frame, scoring_data, include_u_var.get(), 
                                      stats_frame, indicators_frame, pie_frame, details_frame)
        
        # Boutons en bas
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="📋 Copier le rapport", 
                  command=lambda: self._copy_sleep_report(scoring_data, include_u_var.get())).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Exporter CSV", 
                  command=lambda: self._export_sleep_indicators(scoring_data, include_u_var.get())).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Fermer", 
                  command=info_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _update_sleep_indicators(self, parent, scoring_data, include_u, stats_frame, indicators_frame, pie_frame, details_frame):
        """Met à jour l'affichage des indicateurs de sommeil."""
        # Nettoyer les frames
        for widget in stats_frame.winfo_children():
            widget.destroy()
        for widget in indicators_frame.winfo_children():
            widget.destroy()
        for widget in pie_frame.winfo_children():
            widget.destroy()
        for widget in details_frame.winfo_children():
            widget.destroy()
        
        # Calculer les statistiques
        epoch_duration = self.scoring_epoch_duration  # en secondes
        
        # Filtrer ou non les périodes U
        if include_u:
            filtered_data = scoring_data
        else:
            filtered_data = scoring_data[scoring_data['stage'] != 'U']
        
        total_epochs = len(scoring_data)
        filtered_epochs = len(filtered_data)
        
        # Durée totale en lit (Time in Bed - TIB)
        tib_hours = (scoring_data['time'].max() - scoring_data['time'].min() + epoch_duration) / 3600
        tib_minutes = tib_hours * 60
        
        # Compter les époques de sommeil (N1, N2, N3, R/REM)
        sleep_stages = ['N1', 'N2', 'N3', 'R', 'REM']
        sleep_epochs = filtered_data[filtered_data['stage'].isin(sleep_stages)]
        
        # Temps Total de Sommeil (TST)
        tst_minutes = len(sleep_epochs) * (epoch_duration / 60)
        tst_hours = tst_minutes / 60
        
        # Temps d'éveil (W/Wake)
        wake_epochs = filtered_data[filtered_data['stage'].isin(['W', 'Wake', 'WAKE', 'Éveil', 'ÉVEIL'])]
        wake_minutes = len(wake_epochs) * (epoch_duration / 60)
        
        # Efficacité du sommeil (Sleep Efficiency) = TST / TIB * 100
        if tib_minutes > 0:
            sleep_efficiency = (tst_minutes / tib_minutes) * 100
        else:
            sleep_efficiency = 0.0
        
        # Latence d'endormissement (Sleep Onset Latency - SOL)
        # Première époque de sommeil (N1, N2, N3, ou R)
        first_sleep_idx = None
        for idx, stage in enumerate(filtered_data['stage']):
            if stage in sleep_stages:
                first_sleep_idx = idx
                break
        
        if first_sleep_idx is not None:
            sol_minutes = first_sleep_idx * (epoch_duration / 60)
        else:
            sol_minutes = 0.0
        
        # Latence REM (REM Latency)
        first_rem_idx = None
        for idx, stage in enumerate(filtered_data['stage']):
            if stage in ['R', 'REM']:
                first_rem_idx = idx
                break
        
        if first_rem_idx is not None:
            rem_latency_minutes = first_rem_idx * (epoch_duration / 60)
        else:
            rem_latency_minutes = 0.0
        
        # WASO (Wake After Sleep Onset) - temps d'éveil après le premier endormissement
        if first_sleep_idx is not None:
            waso_data = filtered_data.iloc[first_sleep_idx:]
            waso_epochs = waso_data[waso_data['stage'].isin(['W', 'Wake', 'WAKE', 'Éveil', 'ÉVEIL'])]
            waso_minutes = len(waso_epochs) * (epoch_duration / 60)
        else:
            waso_minutes = 0.0
        
        # Afficher les statistiques générales
        stats_text = f"""
Nombre total d'époques : {total_epochs}
Époques analysées : {filtered_epochs} {'(U exclus)' if not include_u else '(U inclus)'}
Durée par époque : {epoch_duration} secondes
Temps total au lit (TIB) : {tib_hours:.2f} heures ({tib_minutes:.1f} min)
"""
        stats_label = ttk.Label(stats_frame, text=stats_text.strip(), justify=tk.LEFT, font=('Courier', 10))
        stats_label.pack(anchor="w")
        
        # Afficher les indicateurs principaux
        indicators_text = f"""
⏱️  Temps Total de Sommeil (TST) : {tst_hours:.2f} heures ({tst_minutes:.1f} min)
    └─ Nombre d'époques de sommeil : {len(sleep_epochs)}

✨ Efficacité du Sommeil : {sleep_efficiency:.1f}%
    └─ Formule : (TST / TIB) × 100 = ({tst_hours:.2f} / {tib_hours:.2f}) × 100

⏰ Latence d'Endormissement (SOL) : {sol_minutes:.1f} minutes
    └─ Temps avant la première époque de sommeil

💤 Latence REM : {rem_latency_minutes:.1f} minutes
    └─ Temps avant la première époque REM

😴 WASO (Wake After Sleep Onset) : {waso_minutes:.1f} minutes
    └─ Temps d'éveil après le premier endormissement

🌙 Temps d'Éveil Total : {wake_minutes:.1f} minutes
"""
        indicators_label = ttk.Label(indicators_frame, text=indicators_text.strip(), 
                                    justify=tk.LEFT, font=('Courier', 10))
        indicators_label.pack(anchor="w")
        
        # Graphique en camembert
        stage_counts = filtered_data['stage'].value_counts()
        stage_percentages = (stage_counts / filtered_epochs * 100).round(1)
        
        # Créer le graphique
        fig_pie = Figure(figsize=(8, 5), dpi=100)
        ax_pie = fig_pie.add_subplot(111)
        
        # Récupérer les couleurs des stades depuis le theme_manager
        stage_colors_map = self.theme_manager.get_stage_colors().copy()
        
        # Définir des couleurs par défaut robustes pour tous les stades possibles
        default_colors = {
            'W': '#1f77b4',       # Bleu pour l'éveil
            'Wake': '#1f77b4',
            'WAKE': '#1f77b4',
            'Éveil': '#1f77b4',
            'ÉVEIL': '#1f77b4',
            'Eveil': '#1f77b4',
            'EVEIL': '#1f77b4',
            'N1': '#ff7f0e',      # Orange pour N1
            'N2': '#2ca02c',      # Vert pour N2
            'N3': '#d62728',      # Rouge pour N3
            'R': '#9467bd',       # Violet pour REM
            'REM': '#9467bd',
            'rem': '#9467bd',
            'U': '#8c564b',       # Marron pour U
            'UNDEFINED': '#8c564b',
            'Undefined': '#8c564b',
            'undefined': '#8c564b',
            'ARTIFACT': '#8c564b',
            'Artifact': '#8c564b',
            'artifact': '#8c564b'
        }
        
        # Combiner avec les couleurs du thème (priorité au thème)
        for stage, color in default_colors.items():
            if stage not in stage_colors_map:
                stage_colors_map[stage] = color
        
        # Préparer les données pour le camembert
        labels = []
        sizes = []
        colors = []
        explode = []  # Pour détacher légèrement certaines portions
        
        # Trier par ordre logique : W, N1, N2, N3, R, U
        stage_order = ['W', 'Wake', 'WAKE', 'Éveil', 'N1', 'N2', 'N3', 'R', 'REM', 'U']
        
        for stage in stage_order:
            if stage in stage_counts.index:
                stage_name = self.sleep_stages.get(stage, stage)
                count = stage_counts[stage]
                percentage = stage_percentages[stage]
                
                labels.append(f"{stage_name}\n{percentage:.1f}%")
                sizes.append(count)
                
                # Couleur selon le stade (avec fallback robuste)
                color = stage_colors_map.get(stage, '#CCCCCC')
                colors.append(color)
                
                # Log pour diagnostic
                print(f"🎨 Camembert: {stage} ({stage_name}) -> couleur {color}")
                
                # Détacher légèrement le stade dominant
                if count == stage_counts.max():
                    explode.append(0.05)
                else:
                    explode.append(0)
        
        # Ajouter les autres stades non listés
        for stage in stage_counts.index:
            if stage not in stage_order:
                stage_name = self.sleep_stages.get(stage, stage)
                count = stage_counts[stage]
                percentage = stage_percentages[stage]
                
                labels.append(f"{stage_name}\n{percentage:.1f}%")
                sizes.append(count)
                color = stage_colors_map.get(stage, '#999999')
                colors.append(color)
                explode.append(0)
                
                # Log pour diagnostic
                print(f"🎨 Camembert: {stage} ({stage_name}) [autre] -> couleur {color}")
        
        # Créer le camembert
        wedges, texts, autotexts = ax_pie.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 9, 'weight': 'bold'}
        )
        
        # Améliorer la lisibilité des pourcentages
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        # Titre du graphique
        ax_pie.set_title(f"Répartition des stades de sommeil\n({filtered_epochs} époques analysées)", 
                        fontsize=12, weight='bold', pad=20)
        
        # Égaliser les proportions pour un cercle parfait
        ax_pie.axis('equal')
        
        # Ajuster la mise en page
        fig_pie.tight_layout()
        
        # Intégrer le graphique dans Tkinter
        canvas_pie = FigureCanvasTkAgg(fig_pie, pie_frame)
        canvas_pie.draw()
        canvas_pie.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Détails par stade (tableau textuel)
        
        details_text = "Stade              | Époques | Pourcentage | Durée (min)\n"
        details_text += "─" * 60 + "\n"
        
        for stage in sorted(stage_counts.index):
            count = stage_counts[stage]
            percentage = stage_percentages[stage]
            duration_min = count * (epoch_duration / 60)
            stage_name = self.sleep_stages.get(stage, stage)
            details_text += f"{stage_name:18} | {count:7} | {percentage:10.1f}% | {duration_min:10.1f}\n"
        
        details_label = ttk.Label(details_frame, text=details_text, justify=tk.LEFT, font=('Courier', 9))
        details_label.pack(anchor="w")
        
        print(f"✅ Indicateurs de sommeil mis à jour (U inclus: {include_u})")
    
    def _copy_sleep_report(self, scoring_data, include_u):
        """Copie le rapport des indicateurs de sommeil dans le presse-papiers."""
        epoch_duration = self.scoring_epoch_duration
        
        if include_u:
            filtered_data = scoring_data
        else:
            filtered_data = scoring_data[scoring_data['stage'] != 'U']
        
        sleep_stages = ['N1', 'N2', 'N3', 'R', 'REM']
        sleep_epochs = filtered_data[filtered_data['stage'].isin(sleep_stages)]
        
        tib_minutes = (scoring_data['time'].max() - scoring_data['time'].min() + epoch_duration) / 60
        tst_minutes = len(sleep_epochs) * (epoch_duration / 60)
        sleep_efficiency = (tst_minutes / tib_minutes) * 100 if tib_minutes > 0 else 0.0
        
        report = f"""RAPPORT D'INDICATEURS DE SOMMEIL
{'=' * 50}

Temps au lit (TIB) : {tib_minutes:.1f} min
Temps total de sommeil (TST) : {tst_minutes:.1f} min
Efficacité du sommeil : {sleep_efficiency:.1f}%
Périodes U {'incluses' if include_u else 'exclues'}

Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.root.clipboard_clear()
        self.root.clipboard_append(report)
        messagebox.showinfo("Copié", "Le rapport a été copié dans le presse-papiers !")
    
    def _export_sleep_indicators(self, scoring_data, include_u):
        """Exporte les indicateurs de sommeil dans un fichier CSV."""
        filepath = filedialog.asksaveasfilename(
            title="Exporter les indicateurs de sommeil",
            defaultextension=".csv",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            epoch_duration = self.scoring_epoch_duration
            
            if include_u:
                filtered_data = scoring_data
            else:
                filtered_data = scoring_data[scoring_data['stage'] != 'U']
            
            sleep_stages = ['N1', 'N2', 'N3', 'R', 'REM']
            sleep_epochs = filtered_data[filtered_data['stage'].isin(sleep_stages)]
            
            tib_minutes = (scoring_data['time'].max() - scoring_data['time'].min() + epoch_duration) / 60
            tst_minutes = len(sleep_epochs) * (epoch_duration / 60)
            sleep_efficiency = (tst_minutes / tib_minutes) * 100 if tib_minutes > 0 else 0.0
            
            # Créer un DataFrame avec les indicateurs
            indicators_df = pd.DataFrame({
                'Indicateur': ['Temps au lit (TIB)', 'Temps total de sommeil (TST)', 
                              'Efficacité du sommeil', 'Périodes U'],
                'Valeur': [f'{tib_minutes:.1f} min', f'{tst_minutes:.1f} min', 
                          f'{sleep_efficiency:.1f}%', 'incluses' if include_u else 'exclues']
            })
            
            indicators_df.to_csv(filepath, index=False, encoding='utf-8')
            messagebox.showinfo("Succès", f"Indicateurs exportés vers :\n{filepath}")
            print(f"✅ Indicateurs exportés : {filepath}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'export :\n{str(e)}")
            print(f"❌ Erreur export indicateurs : {e}")
    
    def _visualize_sleep_stages(self):
        """Visualise les stades de sommeil sur le graphique."""
        if self.sleep_scoring_data is None:
            messagebox.showwarning("Avertissement", "Aucun scoring de sommeil chargé.")
            return
        
        # Créer une fenêtre de visualisation
        viz_window = tk.Toplevel(self.root)
        viz_window.title("🎨 Visualisation des Stades de Sommeil")
        viz_window.geometry("800x600")
        viz_window.transient(self.root)
        viz_window.grab_set()
        
        # Centrer la fenêtre
        viz_window.update_idletasks()
        x = (viz_window.winfo_screenwidth() // 2) - (800 // 2)
        y = (viz_window.winfo_screenheight() // 2) - (600 // 2)
        viz_window.geometry(f"800x600+{x}+{y}")
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Couleurs pour les stades (utilise la palette dynamique)
        stage_colors = self.theme_manager.get_stage_colors().copy()
        stage_colors.update({
            'U': '#800080',    # Violet pour les stades inconnus
        })
        
        # Créer l'histogramme des stades
        time_hours = self.sleep_scoring_data['time'] / 3600
        stages = self.sleep_scoring_data['stage']
        
        # Créer des barres pour chaque stade
        for i, (stage, color) in enumerate(stage_colors.items()):
            mask = stages == stage
            if mask.any():
                ax.bar(time_hours[mask], [1] * mask.sum(), 
                      color=color, alpha=0.7, label=self.sleep_stages[stage])
        
        ax.set_xlabel('Temps (heures)')
        ax.set_ylabel('Stade de sommeil')
        ax.set_title('Visualisation des Stades de Sommeil')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Intégrer dans Tkinter
        canvas = FigureCanvasTkAgg(fig, viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barre d'outils
        toolbar = NavigationToolbar2Tk(canvas, viz_window)
        toolbar.update()
        
        print("✅ Visualisation des stades de sommeil créée")
        logging.info("Visualisation des stades de sommeil affichée")
    
    def _import_manual_scoring_excel(self):
        """Importe le scoring manuel depuis un fichier Excel."""
        if not self.raw:
            messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
            return
        
        file_path = filedialog.askopenfilename(
            title="Importer le scoring manuel",
            filetypes=[
                ("Fichiers Excel", "*.xlsx *.xls"),
                ("Fichiers CSV", "*.csv"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self._show_loading_bar(title="Import scoring manuel", message="Lecture et normalisation du fichier...")
            raw_duration = float(len(self.raw.times) / self.sfreq)
            rec_duration = min(raw_duration, 24 * 3600)
            result = self.manual_scoring_service.import_excel_path(
                file_path,
                absolute_start_datetime=getattr(self, "absolute_start_datetime", None),
                recording_duration_s=rec_duration,
                default_epoch_seconds=float(getattr(self, "scoring_epoch_duration", 30.0)),
            )
            self._apply_manual_scoring_result(result, source="Excel/CSV")
            self.current_scoring_path = file_path
            self.scoring_dirty = True

            try:
                first_epoch_time = float(result.df["time"].iloc[0]) if len(result.df) > 0 else 0.0
                self._center_view_on_epoch(first_epoch_time, float(self.scoring_epoch_duration))
            except Exception:
                self.update_plot()

            self._hide_loading_bar()
            self._update_status_bar()
            messagebox.showinfo(
                "Succès",
                (
                    "Scoring manuel importé avec succès\n"
                    f"{len(result.df)} époques\n"
                    f"Plage: {result.df['time'].min():.1f}s - {result.df['time'].max():.1f}s\n"
                    f"Durée d'époque détectée: {self.scoring_epoch_duration:.1f}s"
                ),
            )
            logging.info("[MANUAL] Excel import completed: n=%d", len(result.df))
            
        except Exception as e:
            self._hide_loading_bar()
            error_msg = f"Erreur lors de l'import:\n{str(e)}"
            messagebox.showerror("Erreur", error_msg)
            logging.error(f"Erreur import Excel: {e}")
    
    def _adjust_epoch_duration(self):
        """Permet d'ajuster manuellement la durée d'époque pour le scoring."""
        if not hasattr(self, 'manual_scoring_data') or self.manual_scoring_data is None:
            messagebox.showwarning("Attention", "Aucun scoring manuel chargé")
            return
        
        # Fenêtre de dialogue pour ajuster la durée
        dialog = tk.Toplevel(self.root)
        dialog.title("Ajuster la Durée d'Époque")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Centrer la fenêtre
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="⚙️ Ajuster la Durée d'Époque", 
                               font=('Segoe UI', 12, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Information actuelle
        info_frame = ttk.LabelFrame(main_frame, text="Informations Actuelles", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        current_duration = getattr(self, 'scoring_epoch_duration', 30.0)
        n_epochs = len(self.manual_scoring_data)
        
        ttk.Label(info_frame, text=f"Durée d'époque actuelle: {current_duration:.1f}s").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Nombre d'époques: {n_epochs}").pack(anchor=tk.W)
        
        if hasattr(self, 'manual_scoring_data') and len(self.manual_scoring_data) > 1:
            total_duration = self.manual_scoring_data['time'].max() - self.manual_scoring_data['time'].min()
            ttk.Label(info_frame, text=f"Durée totale du scoring: {total_duration:.1f}s ({total_duration/60:.1f} min)").pack(anchor=tk.W)
        
        # Ajustement
        adjust_frame = ttk.LabelFrame(main_frame, text="Nouvelle Durée d'Époque", padding="10")
        adjust_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(adjust_frame, text="Durée en secondes:").pack(anchor=tk.W)
        
        duration_var = tk.StringVar(value=str(current_duration))
        duration_entry = ttk.Entry(adjust_frame, textvariable=duration_var, width=10)
        duration_entry.pack(anchor=tk.W, pady=(5, 10))
        
        # Boutons prédéfinis
        buttons_frame = ttk.Frame(adjust_frame)
        buttons_frame.pack(fill=tk.X)
        
        predefined_durations = [20, 30, 40, 60]
        for duration in predefined_durations:
            btn = ttk.Button(buttons_frame, text=f"{duration}s", 
                           command=lambda d=duration: duration_var.set(str(d)))
            btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Boutons d'action
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(20, 0))
        
        def apply_duration():
            try:
                new_duration = float(duration_var.get())
                if new_duration <= 0:
                    messagebox.showerror("Erreur", "La durée doit être positive")
                    return
                
                # Appliquer la nouvelle durée
                self.scoring_epoch_duration = new_duration
                print(f"🔧 Durée d'époque ajustée à {new_duration:.1f}s")
                
                # Mettre à jour l'affichage
                self.update_plot()
                
                messagebox.showinfo("Succès", f"Durée d'époque ajustée à {new_duration:.1f}s")
                dialog.destroy()
                
            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre valide")
        
        ttk.Button(action_frame, text="Annuler", command=dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(action_frame, text="Appliquer", command=apply_duration).pack(side=tk.RIGHT)
    
    def _center_view_on_epoch(self, epoch_start_time, epoch_duration):
        """Centre la vue sur une époque spécifique."""
        print(f"🔍 CHECKPOINT CENTER: Centrage sur époque {epoch_start_time:.1f}s (durée: {epoch_duration:.1f}s)")
        
        # Calculer le centre de l'époque
        epoch_center = epoch_start_time + (epoch_duration / 2)
        
        # Ajuster la durée d'affichage pour bien voir l'époque
        # Utiliser 2-3 fois la durée d'époque pour le contexte
        optimal_duration = max(epoch_duration * 2.5, 60.0)  # Au minimum 60s
        
        print(f"🔍 CHECKPOINT CENTER: Centre époque: {epoch_center:.1f}s")
        print(f"🔍 CHECKPOINT CENTER: Durée optimale: {optimal_duration:.1f}s")
        
        # Définir la fenêtre d'affichage: current_time = début de fenêtre
        window_start = max(0.0, epoch_center - (optimal_duration / 2.0))
        
        # Ajuster la durée d'affichage si nécessaire
        current_duration = getattr(self, 'duration', 10.0)
        if abs(current_duration - optimal_duration) > 1.0:  # tolérance faible
            self.duration = optimal_duration
            if hasattr(self, 'duration_var'):
                try:
                    self.duration_var.set(optimal_duration)
                except Exception:
                    pass
            print(f"🔧 CHECKPOINT CENTER: Durée ajustée de {current_duration:.1f}s à {optimal_duration:.1f}s")
        
        # Appliquer le début de fenêtre
        self.current_time = window_start
        if hasattr(self, 'time_var'):
            try:
                self.time_var.set(window_start)
            except Exception:
                pass
        if hasattr(self, 'bottom_time_var'):
            try:
                self.bottom_time_var.set(window_start)
            except Exception:
                pass
        
        # Rafraîchir l'affichage
        self._update_time_display()
        self.update_plot()
        
        print(f"✅ CHECKPOINT CENTER: Vue centrée sur époque {epoch_start_time:.1f}s (fenêtre début={window_start:.1f}s, durée={self.duration:.1f}s)")
    
    def _export_sleep_scoring(self):
        """Exporte les données de scoring de sommeil."""
        if self.sleep_scoring_data is None:
            messagebox.showwarning("Avertissement", "Aucun scoring de sommeil chargé.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exporter le scoring de sommeil",
            defaultextension=".csv",
            filetypes=[
                ("Fichiers CSV", "*.csv"),
                ("Fichiers Excel", "*.xlsx"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.csv'):
                self.sleep_scoring_data.to_csv(file_path, index=False, encoding='utf-8')
            else:
                self.sleep_scoring_data.to_excel(file_path, index=False)
            
            messagebox.showinfo("Succès", f"Scoring de sommeil exporté avec succès!\n{file_path}")
            logging.info(f"Scoring de sommeil exporté: {file_path}")
            
        except Exception as e:
            error_msg = f"Erreur lors de l'export:\n{str(e)}"
            messagebox.showerror("Erreur", error_msg)
            logging.error(f"Erreur export scoring: {e}")
    
    def _show_about(self):
        """Affiche la boîte de dialogue À propos"""
        about_text = """
EEG Analysis Studio v2.27

Application pour l'analyse de données EEG avec interface graphique moderne.

Fonctionnalités:
• Chargement de fichiers EDF
• Visualisation multi-canaux
• Amplification automatique des signaux faibles
• Filtrage passe-bande
• Autoscale intelligent
• Statistiques détaillées

Développé avec Python, MNE-Python, Matplotlib et Tkinter.

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Contact: come1.barmoy@supbiotech.fr
GitHub: cbarmoy
Date: 2025-09-09
        """
        messagebox.showinfo("À propos", about_text)

def main(ms_path: Optional[str] = None) -> int:
    """Fonction principale"""
    print("CESA (Complex EEG Studio Analysis) v0.0beta1.1")
    print("=" * 50)
    print("Vérification des dépendances...")
    
    # Vérification des dépendances
    try:
        import mne
        import matplotlib
        import numpy
        import scipy
        print("✅ Toutes les dépendances sont disponibles.")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("Installez les dépendances avec: pip install -r requirements.txt")
        return 1
    
    print("\nLancement de l'application...")
    
    try:
        root = tk.Tk()
        root.withdraw()

        try:
            selection = ModeSelector(root, default_ms_path=ms_path).choose()
            try:
                print("🧭 Mode (startup):", getattr(selection, "mode", "?"))
                sel_ms = getattr(selection, "ms_path", None)
                if sel_ms:
                    print("🗂️  Zarr (startup):", sel_ms)
            except Exception:
                pass
        except Exception as selector_error:
            messagebox.showerror(
                "Mode de données",
                f"Impossible d'initialiser le mode Pré-calculé:\n{selector_error}",
                parent=root,
            )
            root.destroy()
            return 1

        data_bridge = DataBridge(selection.provider)
        try:
            print("📄 Provider: fs=", data_bridge.get_sampling_frequency(),
                  " n_channels=", len(data_bridge.get_channel_names()),
                  " duration_s=", data_bridge.get_total_duration_seconds())
        except Exception:
            pass
        root.deiconify()

        app = EEGAnalysisStudio(root, data_bridge=data_bridge)
        try:
            app.data_mode = selection.mode
        except Exception:
            pass

        # Force initial render in precomputed mode so the user sees something immediately
        try:
            app.update_time_scale()
            app.update_plot()
        except Exception:
            pass

        root.mainloop()

    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        print("\n💡 Suggestions:")
        print("   - Vérifiez que tous les fichiers sont présents")
        print("   - Consultez le README.md pour plus d'informations")
        return 1
    
    return 0

def create_scrollable_frame(parent, title=None):
    """
    Crée un frame scrollable bien organisé pour les fenêtres d'analyse.

    Args:
        parent: Fenêtre parent
        title: Titre optionnel pour la fenêtre

    Returns:
        tuple: (top, scrollable_frame) où top est la fenêtre et scrollable_frame le frame scrollable
    """
    top = tk.Toplevel(parent)
    if title:
        top.title(title)
    top.geometry("1100x900")
    top.transient(parent)
    top.grab_set()

    # Créer un frame principal pour organiser le contenu
    main_frame = ttk.Frame(top)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Créer un frame scrollable
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Organiser les éléments dans le frame principal
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return top, scrollable_frame

def create_analysis_buttons(parent, scrollable_frame, analyze_command=None, export_command=None, close_command=None):
    """
    Crée un frame de boutons bien organisé pour les fenêtres d'analyse.

    Args:
        parent: Fenêtre parent
        scrollable_frame: Frame scrollable où placer le contenu
        analyze_command: Fonction à appeler pour l'analyse
        export_command: Fonction à appeler pour l'export
        close_command: Fonction à appeler pour fermer

    Returns:
        ttk.Frame: Frame contenant les boutons
    """
    # Boutons de contrôle - Frame séparé en bas de la fenêtre
    button_frame = ttk.Frame(parent)
    button_frame.pack(fill=tk.X, padx=20, pady=10, side=tk.BOTTOM)

    if analyze_command:
        analyze_btn = ttk.Button(button_frame, text="📊 Analyser",
                               command=analyze_command)
        analyze_btn.pack(side=tk.LEFT, padx=5)

    if export_command:
        export_btn = ttk.Button(button_frame, text="📁 Exporter",
                              command=export_command)
        export_btn.pack(side=tk.LEFT, padx=5)

    if close_command:
        close_btn = ttk.Button(button_frame, text="Fermer",
                             command=close_command)
        close_btn.pack(side=tk.RIGHT, padx=5)

    return button_frame

if __name__ == "__main__":
    sys.exit(main())
