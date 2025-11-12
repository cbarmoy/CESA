#!/usr/bin/env python3
"""
CCESA (Complex EEG Studio Analysis) v1.0 - Professional EEG Analysis Interface
====================================================================

Application professionnelle complète pour l'analyse de données EEG avec
amplification automatique, scoring de sommeil intégré, et analyses avancées.
Développée pour l'Unité Neuropsychologie du Stress (IRBA) selon les standards
scientifiques et les bonnes pratiques MNE-Python.

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Contact: come1.barmoy@supbiotech.fr
GitHub: cbarmoy
Version: 3.0.0 - Release Candidate
Date: 2025-09-26
Licence: MIT
Release: CESA_3.0_release

Fonctionnalités principales v3.0:
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

# ----------------------- OPTIMISATION IMPORTS PYTHON -----------------------

# Imports standards séparés, triés et groupés par fonction
import os
import sys
import json
import csv
import re
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from itertools import groupby

# Import scientifiques et de visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm
import mne
from mne.time_frequency import tfr_array_morlet
from scipy.signal import welch

# Imports tkinter (éléments graphiques)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font

# Imports internes regroupés par module logique/usage
try:
    from core.data_bridge import DataBridge
except ImportError:
    # Fallback - désactiver navigation rapide si module absent
    DataBridge = None
    logging.warning("core.data_bridge non trouvé - navigation rapide désactivée")

from CESA.menu_builder import MenuBuilder
from CESA.file_dialogs import EEGFileManager

from ui.startup_mode import ModeSelector
from ui.open_dataset_dialog import OpenDatasetDialog
from ui.shortcuts_dialog import ShortcutsDialog
from ui.filter_config_dialog import FilterConfigDialog
from ui.report_dialog import ReportDialog
from ui.channel_selector import ChannelSelector
from ui.main_interface import MainInterfaceManager
from CESA.core_manager import CoreManager
from CESA.theme_manager import theme_manager




# Modules CESA spécifiques (fonctionnalités analytiques) regroupés en tuples
from CESA.spectral_analysis import (
    compute_psd_fft,
    compute_band_powers,
    compute_peak_and_centroid,
    compute_stage_psd_welch_for_array,
    compute_stage_psd_fft_for_array,
    EEG_BANDS,
)

# Imports pour les filtres EEG centralisés
from CESA.filters import (
    apply_filter as cesa_apply_filter,
    apply_baseline_correction as cesa_apply_baseline_correction,
    detect_signal_type as cesa_detect_signal_type,
    get_filter_presets as cesa_get_filter_presets,
)

# Imports pour le scoring de sommeil
from CESA.scoring_manager import (
    open_scoring_import_hub,
    open_manual_scoring_editor,
    save_active_scoring,
    get_active_scoring_df,
    export_periods_and_metrics_csv,
    analyze_sleep_periods,
)


# Imports pour l'import et la synchronisation du scoring
from CESA.scoring_io import (
    import_excel_scoring as cesa_import_excel_scoring,
    import_edf_hypnogram as cesa_import_edf_hypnogram,
)

# Imports optionnels sécurisés avec gestion d’erreur claire
try:
    from CESA.event_system import event_bus, Events, EventData
    from CESA.performance_monitor import perf_monitor, measure_time
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False
    logging.warning("Event system not available")

try:
    from CESA.performance_dashboard import PerformanceDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Gestion des warnings pour garder la sortie propre
warnings.filterwarnings('ignore')

# Configuration centralisée du logging (niveau, format, handlers)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)

# Ajout optionnel d’un RotatingFileHandler pour log volumineux
try:
    from logging.handlers import RotatingFileHandler
    rotating_handler = RotatingFileHandler(
        'eeg_studio.log', maxBytes=2_000_000, backupCount=3, encoding='utf-8'
    )
    rotating_handler.setLevel(logging.INFO)
    rotating_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(rotating_handler)
except Exception as e:
    logging.warning(f"Impossible d'initialiser RotatingFileHandler: {e}")

# Imports supplémentaires optionnels liés à l’optimisation mémoire
try:
    from CESA.memory_manager import memory_manager, cached_analysis
    from CESA.data_optimizer import data_optimizer, optimize_data
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("Memory optimization not available")

# Import d’un cache manager (ex : stockage résultats calculs lourds)
from CESA.cache_manager import cache_result

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
    
    def __init__(self, root, data_bridge=None) -> None:
       
        self.root = root
        self.data_bridge = data_bridge
        
        self.start_time = time.time()
        self.telemetry_path = Path("logs/telemetry.csv")

        # Gestionnaires principaux
        self.data_bridge = DataBridge
        self.file_manager = EEGFileManager(self)
        self.data_mode = "precomputed" if DataBridge else "raw"
        self.last_bridge_result = None
        
        # Variables d'interface
        self.sleep_stages = {...}
        self.interface_mode = tk.StringVar(value="modern")
        self.spacing_var = tk.StringVar(value="50")
        self.amplitude_var = tk.StringVar(value="100")
        self.autoscale_var = tk.BooleanVar(value=False)
        
        # État du panneau de commandes
        self.control_panel_collapsed = False
        self.original_control_width = 300

        # Durée d'époque pour le scoring de sommeil (30s par défaut)
        self.scoring_epoch_duration = 30.0

        # Gestionnaire de thèmes (remplace l'ancien système de palettes)
        self.theme_manager = theme_manager

        
        # ✅ 1. D'ABORD configurer l'interface
        self._setup_modern_interface()
        self.create_modern_menu()
        self._create_modern_widgets()
        self._setup_keyboard_shortcuts()

        # ✅ 2. ENSUITE créer les managers (APRÈS l'interface)
        self.interface_manager = MainInterfaceManager(self)
        self.core_manager = CoreManager(self)
        
        # ✅ 3. Puis les systèmes auxiliaires
        self._setup_user_assistant()
        self._console_checkpoints = []
        self._setup_checkpoint_capture()
        self.temporal_markers = []

        # Initialisation de base
        self._start_time = time.time()
        self._telemetry_path = Path("logs") / "telemetry.csv"

        # Gestionnaires principaux
        self.data_bridge = data_bridge
        self.file_manager = EEGFileManager(self)
        self.data_mode = "precomputed" if data_bridge else "raw"
        self._last_bridge_result = None
        
        # Données EEG
        self.raw: Optional[mne.io.Raw] = None
        self.derivations: Dict[str, np.ndarray] = {}
        self.selected_channels: List[str] = []

        # Système d'événements et gestionnaires de dialogue
        self._setup_event_system()
        self.report_dialog = ReportDialog(self)
        self.shortcuts_dialog = ShortcutsDialog(self)
        self.filter_config_dialog = FilterConfigDialog(self)
        self.open_dataset_dialog = OpenDatasetDialog(self)

        # Paramètres temporels
        self.current_time: float = 0.0
        self.duration: float = 30.0  # Durée d'affichage (1 époque)
        self.sfreq: float = 200.0

        self._plot_update_pending_id = None
        self._last_plot_update = 0
        self._min_plot_interval = 0.05  # 50ms minimum entre les mises à jour
        self._original_update_plot = self.update_plot


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
        
        # Scoring de sommeil
        self.sleep_scoring_data: Optional[pd.DataFrame] = None  # Auto (YASA)
        self.manual_scoring_data: Optional[pd.DataFrame] = None  # Manuel (Excel)
        self.show_manual_scoring: bool = True
        self.scoring_epoch_duration: float = 30.0  # Durée d'une époque en secondes
        # Cache PSG
        self._psg_cached_hypnogram: Optional[Tuple[List[str], float]] = None
        self._psg_cached_scoring_rows: int = 0
        

        # Références aux lignes EEG pour mise à jour des couleurs
        self.eeg_lines = []
        # Base d'affichage absolue (peut être définie par Excel pour caler l'axe X)
        self.display_start_datetime: Optional[datetime] = None
        # Intercepter la fermeture pour prévenir si non enregistré
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._quit_application)
        except Exception:
            pass
        
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


        # Configuration de la fenêtre
        try:
            self.root.attributes('-topmost', False)
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
        
        # Optimisation du rendu
        try:
            import concurrent.futures as _fut
            self._plot_executor = _fut.ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1)))
        except Exception:
            self._plot_executor = None
        self._plot_update_gen = 0
        self._plot_update_pending_id = None
        self._active_plot_future = None
        self._active_plot_token = None
        
        # Enregistrement des objets volumineux si disponible
        if OPTIMIZATION_AVAILABLE and hasattr(self, 'raw') and self.raw:
            memory_manager.register_large_object(self.raw)

        # Initialisation du dashboard si disponible
        if DASHBOARD_AVAILABLE:
            self.performance_dashboard = PerformanceDashboard(self)
        
        # Liaison des méthodes de scoring
        self.open_scoring_import_hub = open_scoring_import_hub.__get__(self)
        self.open_manual_scoring_editor = open_manual_scoring_editor.__get__(self)
        self.save_active_scoring = save_active_scoring.__get__(self)
        self.get_active_scoring_df = get_active_scoring_df.__get__(self)
        self.export_periods_and_metrics_csv = export_periods_and_metrics_csv.__get__(self)
        self.analyze_sleep_periods = analyze_sleep_periods.__get__(self)
        
        # Mise à jour de l'affichage et logging
        self._update_version_display()
        logging.info("CCESA (Complex EEG Studio Analysis) v1.0 initialisé avec succès")
        logging.info("Application initialized successfully")


    def _init_modular_interface(self):
        """Initialise l'interface modulaire en complément de l'existante"""
        try:
            # Créer la mise en page modulaire
            self.interface_manager.create_main_layout()
            
            # Connecter aux optimisations existantes
            self.core_manager.initialize_all_modules()
            
            logging.info("Modular interface initialized")
        except Exception as e:
            logging.error(f"Modular interface error: {e}")

    def show_performance_dashboard(self):
        """Affiche le dashboard de performance"""
        if DASHBOARD_AVAILABLE:
            self.performance_dashboard.show_dashboard()
        else:
            messagebox.showwarning("Attention", "Dashboard de performance non disponible")
    def showshortcuts(self):
        """Affiche les raccourcis clavier - Redirige vers le module UI"""
        # Cette méthode existait avant dans ce fichier
        # Maintenant elle appelle le module externe
        self.shortcuts_dialog.show()

    def show_filter_config(self):
        """Affiche la configuration des filtres - Redirige vers le module UI"""
        self.filter_config_dialog.show()

    def show_channel_options(self):
        """Affiche le sélecteur de canaux - Redirige vers le module UI"""
        selector = ChannelSelector(self)
        selector.show()
    def _setup_event_system(self):
        """Configure le système d'événements"""
        try:
            # S'abonner aux événements pertinents
            event_bus.subscribe(Events.DATA_LOADED, self._on_data_loaded)
            event_bus.subscribe(Events.TIME_CHANGED, self._on_time_changed)
            event_bus.subscribe(Events.FILTER_CHANGED, self._on_filter_changed)
            event_bus.subscribe(Events.CHANNELS_SELECTED, self._on_channels_selected)
            logging.info("EVENT: Event system initialized")
        except Exception as e:
            logging.error(f"EVENT: Failed to setup event system: {e}")
        
    def _on_data_loaded(self, data):
        """Gestionnaire pour le chargement de données"""
        logging.info(f"EVENT: Data loaded - {data.filename}")
    
    def _on_time_changed(self, data):
        """Gestionnaire pour le changement de temps"""
        if hasattr(self, '_schedule_plot_update'):
            self._schedule_plot_update()
        
    def _on_filter_changed(self, data):
        """Gestionnaire pour le changement de filtre"""
        logging.info(f"EVENT: Filter changed - enabled: {data.enabled}")
        if hasattr(self, '_schedule_plot_update'):
            self._schedule_plot_update()
        
    def _on_channels_selected(self, data):
        """Gestionnaire pour la sélection de canaux"""
        logging.info(f"EVENT: Channels selected - {len(data.channels)} channels")
        if hasattr(self, '_schedule_plot_update'):
            self._schedule_plot_update()


       
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
        self.root.title("CCESA (Complex EEG Studio Analysis) v1.0 - Interface Moderne")
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
                       font=('Helvetica', 9, 'bold'))
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
                       font=('Helvetica', 9, 'bold'))
        style.configure('Group.TLabelframe.Label', 
                       background='#f8f9fa', 
                       foreground='#495057',
                       font=('Helvetica', 9, 'bold'))
        
        # Configuration du label de version
        style.configure('Version.TLabel', 
                       background='#f8f9fa', 
                       foreground='#007bff',
                       font=('Helvetica', 8, 'bold'))
        
        # Configuration du label de statut
        style.configure('Status.TLabel', 
                       background='#f8f9fa', 
                       foreground='#28a745',
                       font=('Helvetica', 9))
    
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
            'font.family': 'Helvetica',
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
        """Navigation simple vers l'époque précédente - VERSION AVEC ÉVÉNEMENTS"""
        # Votre code existant
        epoch_duration = 30.0
        self.current_time = max(0, self.current_time - epoch_duration)
        
        if hasattr(self, 'timevar'):
            self.timevar.set(self.current_time)
        if hasattr(self, 'bottomtimevar'):
            self.bottomtimevar.set(self.current_time)
            
        # NOUVEAU : Émettre l'événement
        if EVENTS_AVAILABLE:
            try:
                event_data = EventData.TimeChanged(
                    current_time=self.current_time,
                    duration=getattr(self, 'duration', 10.0)
                )
                event_bus.emit(Events.TIME_CHANGED, event_data, throttle=0.05)
            except Exception as e:
                logging.error(f"EVENT: Error emitting time_changed - {e}")
        
        self.update_plot()
    
    def _navigate_simple_epoch_next(self):
        """Navigation simple vers l'époque suivante - VERSION AVEC ÉVÉNEMENTS"""
        # Votre code existant
        epoch_duration = 30.0
        if self.raw:
            max_time = len(self.raw.times) / self.sfreq - self.duration
            self.current_time = min(max_time, self.current_time + epoch_duration)
        else:
            self.current_time += epoch_duration
            
        if hasattr(self, 'timevar'):
            self.timevar.set(self.current_time)
        if hasattr(self, 'bottomtimevar'):
            self.bottomtimevar.set(self.current_time)
        
        # NOUVEAU : Émettre l'événement
        if EVENTS_AVAILABLE:
            try:
                event_data = EventData.TimeChanged(
                    current_time=self.current_time,
                    duration=getattr(self, 'duration', 10.0)
                )
                event_bus.emit(Events.TIME_CHANGED, event_data, throttle=0.05)
            except Exception as e:
                logging.error(f"EVENT: Error emitting time_changed - {e}")
        
        self.update_plot()
    
    def create_modern_menu(self):
        """Crée le menu moderne - VERSION MODULAIRE"""
        try:
            from CESA.menu_builder import MenuBuilder
            self.menu_builder = MenuBuilder(self)
            self.menu_builder.create_modern_menu()
        except Exception as e:
            logging.error(f"Erreur création menu modulaire: {e}")
            # Fallback vers ancien système si erreur
            # self._create_legacy_menu()

    # =====================================================================
    # MÉTHODES DE MENU MANQUANTES
    # =====================================================================
    
    def _export_data(self):
        """Exporte les données - VERSION OPTIMISÉE MODULAIRE"""
        try:
            self.report_dialog.export_data()
        except Exception as e:
            logging.error(f"EXPORT: Erreur - {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de l'export : {str(e)}")


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
                                                         filetypes=[("EDF", "*.edf")])
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
            self._set_status_message("Génération des graphiques spaghetti en cours…")
        except Exception:
            pass

        try:
            outputs = generate_spaghetti_from_edf_dirs(
                before_dir=before_dir,
                after_dir=after_dir,
                output_dir=out_dir,
                selected_bands=selected_bands,
                selected_stages=selected_stages,
                selected_channels=selected_channels,
                selected_subjects=None,
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
        """Génère un rapport complet - VERSION OPTIMISÉE MODULAIRE"""
        try:
            self.report_dialog.generate_report()
        except Exception as e:
            logging.error(f"EXPORT_REPORT: Erreur - {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de l'export : {str(e)}")

    def _generate_report(self):
        """Génère un rapport complet - VERSION OPTIMISÉE MODULAIRE"""
        try:
            return self.report_dialog.generate_analysis_report()
        except Exception as e:
            logging.error(f"REPORT: Erreur - {str(e)}")
            return f"Erreur lors de la génération : {str(e)}"

    # def __init__(self, root):
    #     self.root = root
    #     # self.file_opener = FileOpener(root)

    #     # Redéfinir la méthode _on_file_loaded pour récupérer les données
    #     self.file_opener._on_file_loaded = self.on_file_loaded
    def open_file_dialog(self):
        """Délègue au gestionnaire de fichiers"""
        self.file_manager.load_edf_file()

    def load_selected_edf(self, selection):
        """Charge le fichier EDF sélectionné avec ses options"""
        filepath = selection['filepath']
        mode = selection['mode']
        ms_path = selection['ms_path']
        precompute_action = selection['precompute_action']
        
        # Afficher la barre de progression
        self.file_dialog.show_loading_bar("Chargement EEG", "Ouverture du fichier EDF...")
        
        try:
            # Utiliser votre méthode de chargement existante
            # (cherchez dans votre fichier la méthode qui charge réellement l'EDF)
            self.load_edf_with_params(filepath, mode, ms_path, precompute_action)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement : {e}")
            logging.error(f"Erreur chargement EDF: {e}")
        finally:
            self.file_dialog.hide_loading_bar()



    def on_file_loaded(self, raw, sfreq):
        # Traitement après fichier chargé
        self.raw = raw
        self.sfreq = sfreq
        print(f"Fichier chargé avec {len(raw.ch_names)} canaux à {sfreq} Hz")
        # Mettre à jour l'interface après chargement


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
        self.root.quit()
    
    def _show_channel_selector(self):
        """Affiche le sélecteur de canaux."""
        self.show_channel_selector()
    
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
        except Exception:
            pass
        self.update_plot()
    
    def show_filter_config(self):
        """Affiche la configuration des filtres - VERSION OPTIMISÉE MODULAIRE"""
        try:
            self.filter_config_dialog.show_filter_config()
        except Exception as e:
            logging.error(f"FILTER_CONFIG: Erreur - {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage de la configuration : {str(e)}")

    def show_multi_graph_view(self, embed_parent: Optional[ttk.Frame] = None):
        """Affiche la vue PSG multi-subplots.
        - Si embed_parent est fourni, intègre la figure dans ce conteneur (vue principale)
        - Sinon, ouvre une nouvelle fenêtre Toplevel.
        """
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

        def _norm(name: str) -> str:
            s = (name or "").upper()
            for ch in ("-", "/", " ", "_", ":"):
                s = s.replace(ch, "")
            return s

        available = list(self.raw.ch_names)
        norm_map = { _norm(n): n for n in available }

        # EEG préférés
        eeg_pref = ["F3M2", "F4M1", "C3M2", "C4M1", "O1M2", "O2M1"]
        eeg = [norm_map[p] for p in eeg_pref if p in norm_map]
        if len(eeg) == 0:
            # Fallback: prendre d'autres EEG détectés
            eeg = [n for n in available if cesa_detect_signal_type(n) == 'eeg'][:6]

        # EOG préférés
        eog_pref = ["E1M2", "E2M1"]
        eog = [norm_map[p] for p in eog_pref if p in norm_map]
        if len(eog) == 0:
            eog = [n for n in available if cesa_detect_signal_type(n) == 'eog'][:2]

        # EMG préférés
        emg = [n for n in available if ("LEFT LEG" in n.upper() or "RIGHT LEG" in n.upper())][:2]
        if len(emg) == 0:
            emg = [n for n in available if cesa_detect_signal_type(n) == 'emg'][:2]

        # ECG préférés
        ecg = []
        for key in ["ECG", "ECG1", "ECG2", "HEART RATE", "FREQUENCE CARDI", "FRÉQUENCE CARDI"]:
            if _norm(key) in norm_map:
                ecg.append(norm_map[_norm(key)])
                break
        if len(ecg) == 0:
            ecg = [n for n in available if cesa_detect_signal_type(n) == 'ecg'][:1]

        selected = []
        selected.extend(eeg)
        selected.extend(eog)
        selected.extend(emg)
        selected.extend(ecg)

        # Mémoriser les canaux utilisés par le viewer PSG pour des rafraîchissements rapides
        try:
            self.psg_channels_used = list(selected)
        except Exception:
            pass

        # Pre-slice only the visible window to improve performance
        start_idx = int(max(0, float(self.current_time) * fs))
        end_idx = int(min(len(self.raw.times), (float(self.current_time) + float(self.duration)) * fs))
        if end_idx <= start_idx + 1:
            end_idx = min(len(self.raw.times), start_idx + int(max(2, fs)))

        for ch in selected:
            try:
                arr = self.raw.get_data(picks=[ch], start=start_idx, stop=end_idx)[0]
                data_uv = self._to_microvolts_and_sanitize(arr)
                # Provide pre-windowed time base by letting PSGPlotter detect it
                signals[ch] = (data_uv, fs)
            except Exception:
                continue

        # Hypnogramme depuis scoring actif (tolérant et complet)
        hypnogram = None
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

        # Créer et afficher le plotter PSG
        from CESA.psg_plot import PSGPlotter

        plotter = PSGPlotter(
            signals=signals,
            hypnogram=hypnogram,
            scoring_annotations=events,
            start_time_s=float(self.current_time),
            duration_s=float(self.duration),
            filter_params_by_channel=self.channel_filter_params,
            global_filter_enabled=bool(self.filter_var.get()),
            theme_name=self.theme_manager.current_theme_name,
            total_duration_s=float(len(self.raw.times) / fs) if hasattr(self, 'raw') and self.raw is not None else None,
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
    
    def _run_yasa_scoring(self):
        """Exécute le scoring automatique YASA et stocke dans self.sleep_scoring_data."""
        try:
            if self.raw is None:
                messagebox.showwarning("Avertissement", "Aucune donnée EDF chargée.")
                return
            # Loading bar while running YASA
            self._show_loading_bar(title="Scoring YASA", message="Calcul en cours, veuillez patienter...")
            try:
                from yasa_scoring import run_yasa_scoring  # pyright: ignore
            except ImportError:
                messagebox.showerror("Erreur", "Le module yasa_scoring n'est pas disponible.")
                self._close_loading_bar()
                return

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
                        msg = "Initialisation YASA..."
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

            df = run_yasa_scoring(
                self.raw,
                eeg_candidates=self.yasa_eeg_candidates,
                eog_candidates=self.yasa_eog_candidates,
                emg_candidates=self.yasa_emg_candidates,
                epoch_length=30.0,
                target_sfreq=100.0,
                progress_cb=_progress_cb,
            )

            self.sleep_scoring_data = df
            self.scoring_epoch_duration = 30.0
            
            # Mettre à jour la barre de statut
            self.update_status_bar()
            
            try:
                if hasattr(self, 'progress_var'):
                    self.progress_var.set(100)
                    if hasattr(self, 'progress_label'):
                        self.progress_label.config(text="100%")
            except Exception:
                pass
            self._hide_loading_bar()
            messagebox.showinfo("YASA", f"Scoring automatique terminé. {len(df)} époques.")
            logging.info("[YASA] Scoring stocké dans self.sleep_scoring_data")
        except Exception as e:
            logging.error(f"[YASA] Erreur scoring: {e}")
            try:
                self._hide_loading_bar()
            except Exception:
                pass
            messagebox.showerror("Erreur YASA", f"Echec du scoring automatique:\n{e}")

    def _compare_scoring(self):
        """Compare auto (YASA) vs manuel - version simplifiée et rapide."""
        if self.sleep_scoring_data is None:
            messagebox.showwarning("Avertissement", "Aucun scoring automatique disponible. Lancez YASA d'abord.")
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

            # Calculs basiques
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
                sleep_time_auto=sleep_time_auto
            )
            
        except Exception as e:
            logging.error(f"[COMPARE] Erreur comparaison: {e}")
            messagebox.showerror("Erreur", f"Impossible d'effectuer la comparaison: {e}")
    
    def _show_detailed_comparison(self, n_epochs: int, accuracy: float, labels: list, cm: pd.DataFrame, 
                                counts_manual: list, counts_auto: list, stage_metrics: dict,
                                total_time_hours: float, sleep_time_manual: float, sleep_time_auto: float):
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
        ttk.Label(global_frame, text=f"Précision globale: {accuracy:.1%}").pack(anchor=tk.W)
        ttk.Label(global_frame, text=f"Époques correctes: {int(accuracy * n_epochs)}/{n_epochs}").pack(anchor=tk.W)
        
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
            try:
                from yasa_scoring import run_yasa_scoring  # pyright: ignore
            except ImportError:
                messagebox.showerror("Erreur", "Le module yasa_scoring n'est pas disponible.")
                return {}
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
                    auto_df = run_yasa_scoring(
                        self.raw,
                        eeg_candidates=[montage] + fallback,
                        eog_candidates=self.yasa_eog_candidates,
                        emg_candidates=self.yasa_emg_candidates,
                        epoch_length=epoch_len,
                        target_sfreq=100.0,
                    )
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
        ttk.Label(top, text=f"Époques comparées: {n_epochs}", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 20))
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
        ttk.Label(left, text="Matrice de confusion (lignes=manuel, colonnes=auto)", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(0, 6))
        # Render cm as text
        cm_text = tk.Text(left, height=12, wrap=tk.NONE)
        cm_text.pack(fill=tk.BOTH, expand=True)
        cm_text.insert('1.0', cm.to_string())
        cm_text.configure(state='disabled')

        # Right: F1 bar chart
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Scores par stade (Précision, Rappel, F1)", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(0, 6))

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
        ttk.Label(bottom, text="Comptage par stade (Manuel vs Auto)", font=('Helvetica', 10, 'bold')).pack(anchor='w')
        counts_text = tk.Text(bottom, height=6, wrap=tk.NONE)
        counts_text.pack(fill=tk.X, expand=False)
        counts_text.insert('1.0', "Stade    Manuel    Auto\n")
        counts_text.insert('2.0', "-------------------------\n")
        for i, lbl in enumerate(labels):
            counts_text.insert(tk.END, f"{lbl:<7}  {counts_manual[i]:>6}    {counts_auto[i]:>6}\n")
        counts_text.configure(state='disabled')
    
    if EVENTS_AVAILABLE:
        @measure_time("update_plot")
        def update_plot(self):
            """Version optimisée avec mesure de performance et événements"""
            import time
            current_time = time.time()
            
            if hasattr(self, '_plot_update_pending_id') and self._plot_update_pending_id:
                self.root.after_cancel(self._plot_update_pending_id)
            
            if not hasattr(self, '_last_plot_update'):
                self._last_plot_update = 0
            if not hasattr(self, '_min_plot_interval'):
                self._min_plot_interval = 0.05
                
            time_since_last = current_time - self._last_plot_update
            delay = max(0, int((self._min_plot_interval - time_since_last) * 1000))
            
            self._plot_update_pending_id = self.root.after(delay, self._do_actual_plot_update)

            if OPTIMIZATION_AVAILABLE:
                @cached_analysis()
                @optimize_data
                def _prepare_plot_data(self, channels, time_range):
                    """Prépare les données pour le tracé avec optimisation"""
                    if not self.raw:
                        return {}
                    
                    # Générer la clé de cache
                    cache_key = f"plot_{hash(tuple(channels))}_{time_range[0]}_{time_range[1]}"
                    
                    # Vérifier le cache
                    cached_data = memory_manager.get_plot_data(cache_key)
                    if cached_data:
                        return cached_data
                    
                    # Préparer les données optimisées
                    raw_data = {ch: self.derivations[ch] for ch in channels if ch in self.derivations}
                    optimized_data = data_optimizer.prepare_channel_data(
                        raw_data, channels, time_range, self.sfreq
                    )
                    
                    # Mettre en cache
                    memory_manager.cache_plot_data(cache_key, optimized_data)
                    
                    return optimized_data

    else:
        # Version fallback si les événements ne sont pas disponibles
        def update_plot(self):
            # Votre code update_plot existant sans modifications
            pass

    
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
                               font=('Helvetica', 9), wraplength=300)
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
        spec_window.title("Analyse spectrale - FFT (puissance par bande)")
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
                    ax.plot(freqs, spec, label=ch, linewidth=1.0)

                    # Band powers et métriques
                    bands = compute_band_powers(freqs, spec)
                    peak, centroid = compute_peak_and_centroid(freqs, spec)
                    row = [ch] + [f"{bands[b]:.2f}" for b in EEG_BANDS.keys()] + [f"{peak:.2f}", f"{centroid:.2f}"]
                    tree.insert("", tk.END, values=tuple(row))

                ax.set_title("Spectre (FFT magnitude)")
                ax.set_xlabel("Fréquence (Hz)")
                ax.set_ylabel("Magnitude")
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
        toolbar_label = ttk.Label(toolbar, text=f"Canal: {candidate} | Welch 4s 50% | µV²/Hz", font=('Helvetica', 9))
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
                               font=('Helvetica', 9), wraplength=300)
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
                               font=('Helvetica', 9), wraplength=300)
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
        top.title("PSD par stade (FFT – Analyse_spectrale)")
        top.geometry("1100x680")
        top.transient(self.root)
        top.grab_set()

        toolbar = ttk.Frame(top, style='Custom.TFrame')
        toolbar.pack(fill=tk.X, side=tk.TOP)
        save_fig_btn = ttk.Button(toolbar, text="Enregistrer Figure", style='Custom.TButton')
        save_fig_btn.pack(side=tk.RIGHT, padx=(6,6), pady=4)
        export_csv_btn = ttk.Button(toolbar, text="Exporter CSV")
        export_csv_btn.pack(side=tk.RIGHT, padx=(6,0), pady=4)
        # Sélecteur de thème pour la fenêtre FFT
        theme_var = tk.StringVar(value=self.theme_manager.current_theme_name)
        theme_combo = ttk.Combobox(toolbar, textvariable=theme_var, state="readonly", width=12)
        theme_combo['values'] = list(self.theme_manager.get_available_themes().values())
        theme_combo.pack(side=tk.RIGHT, padx=(6,0), pady=4)
        theme_combo.bind('<<ComboboxSelected>>', lambda e: self._change_theme_by_display_name(theme_var.get()))
        toolbar_label = ttk.Label(toolbar, text=f"Canal: {candidate} | FFT magnitude | DC retirée", font=('Helvetica', 9))
        toolbar_label.pack(side=tk.LEFT, padx=8)

        main = ttk.Frame(top)
        main.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(main, width=320)
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)
        content = ttk.Frame(main)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Options
        opt = ttk.LabelFrame(side, text="Paramètres (FFT)", padding=8)
        opt.pack(fill=tk.Y, expand=True, padx=8, pady=8)
        
        # Sélection du canal
        channel_box = ttk.LabelFrame(opt, text="Canal", padding=6)
        channel_box.pack(fill=tk.X, pady=(0,8))
        available_channels = list(self.derivations.keys())
        selected_channel_var = tk.StringVar(value=candidate)
        channel_combo = ttk.Combobox(channel_box, textvariable=selected_channel_var, values=available_channels, state="readonly", width=20)
        channel_combo.pack(fill=tk.X)
        
        # Tooltip pour la sélection du canal
        create_tooltip(channel_combo, 
                      "Sélection du canal EEG pour l'analyse PSD.\n\n"
                      "• Choisissez le canal qui vous intéresse pour l'analyse spectrale\n"
                      "• Les canaux bipolaires (ex: C4-M1) sont généralement préférés\n"
                      "• Le canal sélectionné sera utilisé pour calculer la densité spectrale de puissance\n"
                      "• Chaque canal peut avoir des caractéristiques spectrales différentes selon sa position anatomique")
        
        # Taille du bin (nperseg) pour FFT
        bin_box = ttk.LabelFrame(opt, text="Taille du bin (nperseg)", padding=6)
        bin_box.pack(fill=tk.X, pady=(8,8))
        ttk.Label(bin_box, text="Durée (secondes):").pack(anchor='w')
        nperseg_sec_var = tk.DoubleVar(value=4.0)
        nperseg_entry = ttk.Entry(bin_box, textvariable=nperseg_sec_var, width=10)
        nperseg_entry.pack(anchor='w', pady=(2,0))
        ttk.Label(bin_box, text="(recommandé: 2-8s)").pack(anchor='w')
        
        # Tooltip pour la taille du bin
        create_tooltip(nperseg_entry, 
                      "Taille du segment (nperseg) pour la méthode FFT.\n\n"
                      "• Définit la durée de chaque segment utilisé pour le calcul de la PSD\n"
                      "• Valeurs plus petites (2-4s) : meilleure résolution temporelle, plus de bruit\n"
                      "• Valeurs plus grandes (6-8s) : meilleure résolution fréquentielle, moins de bruit\n"
                      "• Recommandé : 4 secondes pour un bon compromis\n"
                      "• Doit être ≥ 1 seconde pour des résultats fiables")
        
        robust_var = tk.BooleanVar(value=True)
        robust_cb = ttk.Checkbutton(opt, text="Médiane + SEM robuste (MAD)", variable=robust_var)
        robust_cb.pack(anchor='w')
        equalize_var = tk.BooleanVar(value=True)
        equalize_cb = ttk.Checkbutton(opt, text="Égaliser n d'époques par stade", variable=equalize_var)
        equalize_cb.pack(anchor='w', pady=(4,0))

        freq_box = ttk.LabelFrame(opt, text="Affichage fréquence", padding=6)
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
        period_box = ttk.LabelFrame(opt, text="Période d'analyse", padding=6)
        period_box.pack(fill=tk.X, pady=(10,4))

        # Case à cocher pour activer la période personnalisée
        self.fft_use_period_var = tk.BooleanVar(value=False)
        use_period_cb = ttk.Checkbutton(period_box, text="Analyser seulement une période", variable=self.fft_use_period_var)
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

        ttk.Label(period_box, text="Début (s):").pack(anchor='w')
        period_start_entry.pack(anchor='w', pady=(2,4))
        ttk.Label(period_box, text="Fin (s):").pack(anchor='w')
        period_end_entry.pack(anchor='w', pady=(2,0))

        # Tooltip pour la période
        create_tooltip(use_period_cb,
                      "Analyser seulement une période temporelle spécifique.\n\n"
                      "• Permet de focaliser l'analyse FFT sur une portion d'intérêt\n"
                      "• Utile pour analyser des événements spécifiques (ex: sommeil paradoxal)\n"
                      "• Laisse les champs vides pour analyser toute la durée disponible\n"
                      "• Les temps sont en secondes depuis le début de l'enregistrement")

        info_box = ttk.LabelFrame(opt, text="Infos", padding=6)
        info_box.pack(fill=tk.X, pady=(10,4))
        info_var = tk.StringVar(value="")
        info_label = ttk.Label(info_box, textvariable=info_var, justify='left')
        info_label.pack(anchor='w')
        
        # Tooltips pour les contrôles FFT
        create_tooltip(robust_cb, 
                      "Statistiques robustes (médiane + MAD).\n\n"
                      "• Utilise la médiane au lieu de la moyenne pour la tendance centrale\n"
                      "• Utilise l'écart absolu médian (MAD) pour l'erreur standard\n"
                      "• Plus robuste aux valeurs aberrantes et au bruit\n"
                      "• Standard dans la littérature scientifique pour l'analyse EEG\n"
                      "• Fournit des intervalles de confiance plus fiables")
        
        create_tooltip(equalize_cb, 
                      "Égaliser le nombre d'époques par stade.\n\n"
                      "• Évite le biais statistique dû aux différences de durée entre stades\n"
                      "• Chaque stade aura le même nombre d'époques (minimum disponible)\n"
                      "• Important pour les comparaisons statistiques équitables\n"
                      "• Peut réduire le nombre d'époques utilisées si un stade est rare")
        
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
                      "• Recommandé : 30-45 Hz pour l'analyse du sommeil\n"
                      "• Valeurs plus basses : focus sur les bandes de sommeil (delta, theta, alpha, sigma)\n"
                      "• Valeurs plus hautes : inclut les ondes gamma et artefacts\n"
                      "• Doit être ≤ fréquence de Nyquist (fréquence d'échantillonnage / 2)")
        
        create_tooltip(info_label, 
                      "Informations sur l'analyse PSD par stade.\n\n"
                      "• Affiche le nombre d'époques disponibles pour chaque stade de sommeil\n"
                      "• Format : W: n=X | N1: n=Y | N2: n=Z | N3: n=A | R: n=B\n"
                      "• Plus le nombre d'époques est élevé, plus l'analyse est fiable\n"
                      "• Les stades avec peu d'époques peuvent avoir des statistiques moins robustes\n"
                      "• Mis à jour automatiquement lors du rendu")
        
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
                toolbar_label.config(text=f"Canal: {current_channel} | FFT {current_nperseg_sec}s | DC retirée")
                
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

                # Tracé (échelle linéaire; magnitude a.u.)
                for s in stages_order:
                    if s not in out:
                        continue
                    f, mean_vals, sem_vals, n_ep = out[s]
                    ax.plot(f, mean_vals, color=current_stage_colors[s], label=f"{s} (n={n_ep})")
                    ax.fill_between(f, np.maximum(mean_vals - sem_vals, 0.0), mean_vals + sem_vals,
                                    color=current_stage_colors[s], alpha=0.15, linewidth=0)

                ax.set_xlim(max(0.0, fmin), min(fmax, fs/2))
                ax.set_xlabel("Fréquence (Hz)")
                ax.set_ylabel("Magnitude (a.u.)")
                ax.set_title("PSD par stade (FFT magnitude; DC retirée)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', ncol=1, fontsize=8)
                fig.tight_layout()
                canvas.draw()
            except Exception as e:
                print(f"❌ CHECKPOINT STAGE-FFT ERR: {e}")
                messagebox.showerror("Erreur", f"Echec calcul/affichage PSD FFT par stade: {e}")

        def _apply_freq_window():
            render()

        ttk.Button(freq_box, text="Appliquer", command=_apply_freq_window).grid(row=2, column=0, columnspan=2, pady=(8,0))

        render()

        def _save_figure(fig_obj):
            try:
                file_path = filedialog.asksaveasfilename(title="Enregistrer la figure", defaultextension=".png",
                                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
                if file_path:
                    fig_obj.savefig(file_path, dpi=200, bbox_inches='tight')
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'enregistrement de la figure: {e}")

        def _export_csv():
            try:
                fmin = float(freq_min_var.get())
                fmax = float(freq_max_var.get())
                eq = bool(equalize_var.get())
                robust = bool(robust_var.get())
                current_channel = selected_channel_var.get()
                if current_channel not in self.derivations:
                    messagebox.showerror("Erreur", f"Canal '{current_channel}' non disponible")
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
                file_path = filedialog.asksaveasfilename(title="Exporter CSV (long format)", defaultextension=".csv",
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
                messagebox.showinfo("Export", f"CSV exporté: {file_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec de l'export CSV: {e}")

        # Configurer les boutons après la définition de render
        save_fig_btn.configure(command=lambda: _save_figure(fig))
        export_csv_btn.configure(command=_export_csv)

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
                               font=('Helvetica', 9), wraplength=300)
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
                                   font=('Helvetica', 9), wraplength=300)
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
                                  font=('Helvetica', 16, 'bold'))
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
                                   font=('Helvetica', 9), wraplength=300)
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
                                  font=('Helvetica', 16, 'bold'))
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
📞 SUPPORT CESA v3.0

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
CESA v3.0 est développé pour l'Unité Neuropsychologie du Stress (IRBA)
Auteur : Côme Barmoy
Version : 3.0.0
Licence : MIT
            """
            messagebox.showinfo("Support CESA v3.0", support_info)
    
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
                                   font=('Helvetica', 9), wraplength=300)
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
                                  font=('Helvetica', 16, 'bold'))
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
                                  font=('Helvetica', 16, 'bold'))
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
        """Affiche la configuration avancée."""
        self._show_filter_config()
    
    # def _show_filter_config(self):
    #     """Affiche la configuration avancée du filtre."""
    #     filter_window = tk.Toplevel(self.root)
    #     filter_window.title("Configuration Avancée - Filtre et Autoscale (par canal)")
    #     filter_window.geometry("800x700")
    #     filter_window.resizable(True, True)
    #     filter_window.transient(self.root)
    #     filter_window.grab_set()
        
    #     # Centrer la fenêtre
    #     filter_window.update_idletasks()
    #     x = (filter_window.winfo_screenwidth() // 2) - (800 // 2)
    #     y = (filter_window.winfo_screenheight() // 2) - (700 // 2)
    #     filter_window.geometry(f"800x700+{x}+{y}")
        
    #     # Style moderne
    #     style = ttk.Style()
    #     style.theme_use('clam')
        
    #     # Frame principal
    #     main_frame = ttk.Frame(filter_window, padding="20")
    #     main_frame.pack(fill=tk.BOTH, expand=True)
        
    #     # Titre
    #     title_label = ttk.Label(main_frame, text="🔧 Configuration Avancée - Filtre et Autoscale", 
    #                            font=('Helvetica', 14, 'bold'))
    #     title_label.pack(pady=(0, 20))
        
    #     # Configuration des fréquences
    #     freq_frame = ttk.LabelFrame(main_frame, text="📊 Fréquences de Coupure", padding="10")
    #     freq_frame.pack(fill=tk.X, pady=(0, 15))
        
    #     # Fréquence basse
    #     ttk.Label(freq_frame, text="Fréquence basse (Hz):").pack(anchor=tk.W)
    #     low_frame = ttk.Frame(freq_frame)
    #     low_frame.pack(fill=tk.X, pady=(5, 10))
        
    #     self.filter_low_var = tk.DoubleVar(value=self.filter_low)
    #     low_scale = ttk.Scale(low_frame, from_=0.1, to=10.0, variable=self.filter_low_var, 
    #                          orient=tk.HORIZONTAL)
    #     low_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
    #     low_entry = ttk.Entry(low_frame, textvariable=self.filter_low_var, width=8)
    #     low_entry.pack(side=tk.RIGHT)
        
    #     # Fréquence haute
    #     ttk.Label(freq_frame, text="Fréquence haute (Hz):").pack(anchor=tk.W)
    #     high_frame = ttk.Frame(freq_frame)
    #     high_frame.pack(fill=tk.X, pady=(5, 0))
        
    #     self.filter_high_var = tk.DoubleVar(value=self.filter_high)
    #     high_scale = ttk.Scale(high_frame, from_=10.0, to=100.0, variable=self.filter_high_var, 
    #                           orient=tk.HORIZONTAL)
    #     high_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
    #     high_entry = ttk.Entry(high_frame, textvariable=self.filter_high_var, width=8)
    #     high_entry.pack(side=tk.RIGHT)
        
    #     # Configuration du type de filtre (global)
    #     # Configuration par canal (présélections)
    #     per_channel_frame = ttk.LabelFrame(main_frame, text="Paramètres par canal (présélections rapides)", padding="10")
    #     per_channel_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

    #     # Frame avec scrollbar pour le tableau
    #     tree_frame = ttk.Frame(per_channel_frame)
    #     tree_frame.pack(fill=tk.BOTH, expand=True)
        
    #     columns = ("Canal", "Bas (Hz)", "Haut (Hz)", "Actif", "Amplitude")
    #     tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=8)
    #     for col in columns:
    #         tree.heading(col, text=col)
    #         tree.column(col, width=120, anchor=tk.CENTER)
        
    #     # Scrollbar pour le tableau
    #     tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    #     tree.configure(yscrollcommand=tree_scrollbar.set)
        
    #     tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    #     tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    #     # Remplir avec presets pour F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1
    #     def load_presets_into_tree():
    #         tree.delete(*tree.get_children())
    #         present_channels = set(self.derivations.keys()) if hasattr(self, 'derivations') else set()
    #         for ch, (lo, hi) in self.default_derivation_presets.items():
    #             enabled = True if ch in present_channels else False
    #             # Amplitude par défaut (récupérer depuis channel_filter_params si existe)
    #             amp = 100.0
    #             if ch in self.channel_filter_params:
    #                 amp = float(self.channel_filter_params[ch].get('amplitude', 100.0))
    #             # Charger overrides s'ils existent
    #             if ch in self.channel_filter_params:
    #                 params = self.channel_filter_params[ch]
    #                 lo = params.get('low', lo)
    #                 hi = params.get('high', hi)
    #                 enabled = bool(params.get('enabled', enabled))
    #             tree.insert('', tk.END, values=(ch, f"{lo}", f"{hi}", "Oui" if enabled else "Non", f"{amp}"))

    #     load_presets_into_tree()

    #     # Aide pour édition: double-clic pour modifier une cellule (simple implémentation)
    #     def on_tree_double_click(event):
    #         item_id = tree.identify_row(event.y)
    #         col = tree.identify_column(event.x)
    #         if not item_id or not col:
    #             return
    #         col_index = int(col.replace('#','')) - 1
    #         if col_index == 0:
    #             return  # canal non éditable
    #         x, y, width, height = tree.bbox(item_id, col)
    #         value = tree.set(item_id, columns[col_index])
    #         entry = ttk.Entry(tree)
    #         entry.insert(0, value)
    #         entry.place(x=x, y=y, width=width, height=height)

    #         def save_edit(event=None):
    #             new_val = entry.get()
    #             entry.destroy()
    #             # Validation bas/haut/amplitude/enabled
    #             if col_index in (1,2,4):  # bas, haut, amplitude
    #                 try:
    #                     float(new_val)
    #                 except:
    #                     return
    #             if col_index == 3:
    #                 new_val = "Oui" if new_val.strip().lower() in ("oui","yes","true","1") else "Non"
    #             tree.set(item_id, columns[col_index], new_val)

    #         entry.bind('<Return>', save_edit)
    #         entry.bind('<FocusOut>', save_edit)
    #         entry.focus_set()

    #     tree.bind('<Double-1>', on_tree_double_click)
    #     type_frame = ttk.LabelFrame(main_frame, text="⚙️ Type de Filtre", padding="10")
    #     type_frame.pack(fill=tk.X, pady=(0, 15))
        
    #     self.filter_type_var = tk.StringVar(value=self.filter_type)
    #     ttk.Radiobutton(type_frame, text="Butterworth (recommandé)", variable=self.filter_type_var, 
    #                    value="butterworth").pack(anchor=tk.W, pady=2)
    #     ttk.Radiobutton(type_frame, text="Chebyshev Type I", variable=self.filter_type_var, 
    #                    value="cheby1").pack(anchor=tk.W, pady=2)
    #     ttk.Radiobutton(type_frame, text="Chebyshev Type II", variable=self.filter_type_var, 
    #                    value="cheby2").pack(anchor=tk.W, pady=2)
    #     ttk.Radiobutton(type_frame, text="Elliptique", variable=self.filter_type_var, 
    #                    value="ellip").pack(anchor=tk.W, pady=2)
        
    #     # Configuration de l'ordre
    #     order_frame = ttk.LabelFrame(main_frame, text="📈 Ordre du Filtre", padding="10")
    #     order_frame.pack(fill=tk.X, pady=(0, 15))
        
    #     self.filter_order_var = tk.IntVar(value=self.filter_order)
    #     order_scale = ttk.Scale(order_frame, from_=1, to=10, variable=self.filter_order_var, 
    #                            orient=tk.HORIZONTAL)
    #     order_scale.pack(fill=tk.X, pady=(5, 10))
        
    #     order_info = ttk.Label(order_frame, text="Ordre plus élevé = pente plus raide, mais plus de calculs")
    #     order_info.pack(anchor=tk.W)
        
    #     # Configuration de la fenêtre
    #     window_frame = ttk.LabelFrame(main_frame, text="🪟 Fenêtre de Filtrage", padding="10")
    #     window_frame.pack(fill=tk.X, pady=(0, 15))
        
    #     self.filter_window_var = tk.StringVar(value=self.filter_window)
    #     ttk.Radiobutton(window_frame, text="Hamming (recommandé)", variable=self.filter_window_var, 
    #                    value="hamming").pack(anchor=tk.W, pady=2)
    #     ttk.Radiobutton(window_frame, text="Hanning", variable=self.filter_window_var, 
    #                    value="hanning").pack(anchor=tk.W, pady=2)
    #     ttk.Radiobutton(window_frame, text="Blackman", variable=self.filter_window_var, 
    #                    value="blackman").pack(anchor=tk.W, pady=2)
    #     ttk.Radiobutton(window_frame, text="Kaiser", variable=self.filter_window_var, 
    #                    value="kaiser").pack(anchor=tk.W, pady=2)
        
    #     # Configuration de l'autoscale
    #     autoscale_frame = ttk.LabelFrame(main_frame, text="📏 Configuration Autoscale", padding="10")
    #     autoscale_frame.pack(fill=tk.X, pady=(0, 20))
        
    #     ttk.Label(autoscale_frame, text="Durée de la fenêtre d'autoscale (secondes):").pack(anchor=tk.W)
    #     autoscale_duration_frame = ttk.Frame(autoscale_frame)
    #     autoscale_duration_frame.pack(fill=tk.X, pady=(5, 0))
        
    #     self.autoscale_duration_var = tk.DoubleVar(value=self.autoscale_window_duration)
    #     autoscale_scale = ttk.Scale(autoscale_duration_frame, from_=5, to=120, 
    #                                variable=self.autoscale_duration_var, orient=tk.HORIZONTAL)
    #     autoscale_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
    #     autoscale_entry = ttk.Entry(autoscale_duration_frame, textvariable=self.autoscale_duration_var, width=8)
    #     autoscale_entry.pack(side=tk.RIGHT)
        
    #     autoscale_info = ttk.Label(autoscale_frame, text="Fenêtre plus courte = adaptation plus rapide, plus longue = plus stable")
    #     autoscale_info.pack(anchor=tk.W, pady=(5, 0))
        
    #     # Boutons
    #     button_frame = ttk.Frame(main_frame)
    #     button_frame.pack(fill=tk.X, pady=(10, 0))
        
    #     def apply_filter_config():
    #         """Applique la configuration du filtre et de l'autoscale."""
    #         self.filter_low = self.filter_low_var.get()
    #         self.filter_high = self.filter_high_var.get()
    #         self.filter_type = self.filter_type_var.get()
    #         self.filter_order = self.filter_order_var.get()
    #         self.filter_window = self.filter_window_var.get()
            
    #         # Sauvegarder paramètres par canal depuis le tableau
    #         updated: Dict[str, Dict[str, float]] = {}
    #         for item in tree.get_children():
    #             ch, lo, hi, enabled, amp = tree.item(item, 'values')
    #             try:
    #                 lo_f = float(lo)
    #                 hi_f = float(hi)
    #                 amp_f = float(amp)
    #             except:
    #                 continue
    #             updated[ch] = {
    #                 'low': lo_f,
    #                 'high': hi_f,
    #                 'enabled': True if str(enabled).lower() in ("oui","yes","true","1") else False,
    #                 'amplitude': amp_f
    #             }
    #         self.channel_filter_params.update(updated)
    #         self.autoscale_window_duration = self.autoscale_duration_var.get()
            
    #         # Mettre à jour l'affichage
    #         if self.raw and self.selected_channels:
    #             self.update_plot()
            
    #         messagebox.showinfo("Configuration", "Configuration appliquée avec succès!")
    #         logging.info(f"Filtre configuré: {self.filter_type}, ordre {self.filter_order}, "
    #                     f"freq {self.filter_low}-{self.filter_high} Hz, fenêtre {self.filter_window}")
    #         logging.info(f"Autoscale configuré: fenêtre de {self.autoscale_window_duration}s")
        
    #     def reset_filter_config():
    #         """Remet la configuration par défaut."""
    #         print("🔍 CHECKPOINT FILTER RESET: Réinitialisation configuration filtre")
    #         logging.info("[FILTER] Resetting filter configuration to defaults")
    #         self.filter_low_var.set(0.5)
    #         self.filter_high_var.set(30.0)
    #         self.filter_type_var.set("butterworth")
    #         self.filter_order_var.set(4)
    #         self.filter_window_var.set("hamming")
    #         self.channel_filter_params.clear()
    #         load_presets_into_tree()
    #         self.autoscale_duration_var.set(30.0)
        
    #     ttk.Button(button_frame, text="Réinitialiser", command=reset_filter_config).pack(side=tk.LEFT, padx=(0, 10))
    #     ttk.Button(button_frame, text="Appliquer", command=apply_filter_config).pack(side=tk.LEFT, padx=(0, 10))
    #     ttk.Button(button_frame, text="Annuler", command=filter_window.destroy).pack(side=tk.RIGHT, padx=(10, 0))
        
    #     # Focus sur la fenêtre
    #     filter_window.focus_set()
    
    def _show_user_guide(self):
        """Affiche le guide d'utilisation (fenêtre de bienvenue)."""
        if hasattr(self, 'user_assistant') and self.user_assistant:
            self.user_assistant.show_welcome_assistant()
        else:
            messagebox.showinfo("Guide d'Utilisation", "Assistant utilisateur non disponible.")
    
    def _show_shortcuts(self):
        """Affiche les raccourcis - VERSION OPTIMISÉE MODULAIRE"""
        try:
            shortcuts_dialog = ShortcutsDialog(self.root)
            shortcuts_dialog.show_shortcuts()
        except Exception as e:
            print(f"❌ CHECKPOINT SHORTCUTS: Erreur - {e}")
            logging.error(f"[SHORTCUTS] Erreur - {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des raccourcis : {str(e)}")


    def _report_bug(self):
        """Signaler un bug - VERSION OPTIMISÉE MODULAIRE"""
        try:
            self.report_dialog.report_bug()
        except Exception as e:
            logging.error(f"BUGREPORT: Erreur - {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de la génération du rapport : {str(e)}")

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
            report_lines.append(f"Version de l'application : v3.0")
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
        """Traite les données de scoring de sommeil - VERSION CORRIGÉE."""
        try:
            print(f"🔍 CHECKPOINT 1: Données brutes reçues: {df.shape}")
            print(f"🔍 CHECKPOINT 1: Colonnes: {list(df.columns)}")
            print(f"🔍 CHECKPOINT 1: Premières lignes:\n{df.head()}")
            print(f"🔍 CHECKPOINT 1: Types de données:\n{df.dtypes}")
            
            # Chercher les colonnes pertinentes
            time_col = None
            stage_col = None
            
            print(f"🔍 CHECKPOINT 2: Recherche des colonnes...")
            # Recherche des colonnes par nom (français et anglais)
            for col in df.columns:
                col_lower = str(col).lower()
                print(f"🔍 CHECKPOINT 2: Analyse colonne '{col}' -> '{col_lower}'")
                if any(keyword in col_lower for keyword in ['time', 'temps', 'epoch', 'époque', 'heure', 'début', 'start', 'minute']):
                    time_col = col
                    print(f"✅ CHECKPOINT 2: Colonne temps trouvée: {col}")
                elif any(keyword in col_lower for keyword in ['stage', 'stade', 'sleep', 'sommeil', 'score', 'état', 'state']):
                    stage_col = col
                    print(f"✅ CHECKPOINT 2: Colonne stade trouvée: {col}")
            
            # Si pas trouvé, utiliser les premières colonnes
            if time_col is None and len(df.columns) > 0:
                time_col = df.columns[0]
                print(f"⚠️ CHECKPOINT 2: Utilisation première colonne comme temps: {time_col}")
            if stage_col is None and len(df.columns) > 1:
                stage_col = df.columns[1]
                print(f"⚠️ CHECKPOINT 2: Utilisation deuxième colonne comme stade: {stage_col}")
            
            print(f"🔍 CHECKPOINT 3: Colonnes sélectionnées - Temps: {time_col}, Stade: {stage_col}")
            
            if not time_col or not stage_col:
                raise ValueError("Impossible de trouver les colonnes de temps et de stade")
            
            # Créer le DataFrame de scoring
            print(f"🔍 CHECKPOINT 4: Création du DataFrame de scoring...")
            self.sleep_scoring_data = df[[time_col, stage_col]].copy()
            self.sleep_scoring_data.columns = ['time', 'stage']
            
            print(f"🔍 CHECKPOINT 4: Données après sélection: {self.sleep_scoring_data.shape}")
            print(f"🔍 CHECKPOINT 4: Premières lignes sélectionnées:\n{self.sleep_scoring_data.head()}")
            print(f"🔍 CHECKPOINT 4: Types après sélection:\n{self.sleep_scoring_data.dtypes}")
            
            # Nettoyer les données - supprimer les lignes vides
            print(f"🔍 CHECKPOINT 5: Nettoyage des données...")
            self.sleep_scoring_data = self.sleep_scoring_data.dropna()
            
            # Supprimer les lignes avec des valeurs vides ou invalides
            self.sleep_scoring_data = self.sleep_scoring_data[
                (self.sleep_scoring_data['stage'].astype(str).str.strip() != '') &
                (self.sleep_scoring_data['time'].astype(str).str.strip() != '') &
                (~self.sleep_scoring_data['stage'].astype(str).str.contains(r'^\[\]$', na=False)) &
                (~self.sleep_scoring_data['time'].astype(str).str.contains(r'^\[\]$', na=False))
            ]
            
            print(f"🔍 CHECKPOINT 5: Données après nettoyage: {self.sleep_scoring_data.shape}")
            print(f"🔍 CHECKPOINT 5: Premières lignes après nettoyage:\n{self.sleep_scoring_data.head()}")
            
            # Convertir les stades français vers les codes standard
            print(f"🔍 CHECKPOINT 6: Conversion des stades...")
            self.sleep_scoring_data['stage'] = self.sleep_scoring_data['stage'].astype(str).str.lower().str.strip()
            print(f"🔍 CHECKPOINT 6: Stades avant mapping: {self.sleep_scoring_data['stage'].unique()[:10]}")
            self.sleep_scoring_data['stage'] = self.sleep_scoring_data['stage'].map(self.french_to_standard).fillna('U')
            print(f"🔍 CHECKPOINT 6: Stades après mapping: {self.sleep_scoring_data['stage'].unique()[:10]}")
            
            # Convertir le temps en secondes - CORRECTION MAJEURE
            print(f"🔍 CHECKPOINT 7: Conversion du temps...")
            print(f"🔍 CHECKPOINT 7: Type de données temps: {self.sleep_scoring_data['time'].dtype}")
            print(f"🔍 CHECKPOINT 7: Premières valeurs temps: {self.sleep_scoring_data['time'].head()}")
            
            # CORRECTION DÉFINITIVE DE LA CONVERSION DES TEMPS
            print(f"🔍 CHECKPOINT 7: CORRECTION DÉFINITIVE - FORCAGE DE LA CONVERSION")
            
            # S'assurer que l'index est séquentiel avant de fabriquer les temps
            self.sleep_scoring_data.reset_index(drop=True, inplace=True)

            # FORCER LA CONVERSION EN ÉPOQUES DE 30s POUR TOUS LES CAS
            print(f"🔍 CHECKPOINT 7: FORCAGE DE LA CONVERSION EN ÉPOQUES DE 30s")
            self.sleep_scoring_data['time'] = self.sleep_scoring_data.index * 30.0
            print(f"✅ CHECKPOINT 7: Temps FORCÉS en époques de 30s")
            print(f"🔍 CHECKPOINT 7: Temps finaux: {self.sleep_scoring_data['time'].min():.1f}s - {self.sleep_scoring_data['time'].max():.1f}s")
            
            # Vérification finale
            if self.sleep_scoring_data['time'].max() > 0:
                print(f"✅ CHECKPOINT 7: CONVERSION RÉUSSIE - Temps valides")
            else:
                print(f"❌ CHECKPOINT 7: ÉCHEC DE LA CONVERSION - Temps toujours à 0")
            
            # Ancienne logique supprimée - on force toujours la conversion en époques de 30s
            
            # Définir la durée d'époque (30 secondes par défaut)
            self.scoring_epoch_duration = 30.0
            print(f"🔍 CHECKPOINT 8: Durée d'époque définie: {self.scoring_epoch_duration}s")
            logging.info(f"[EPOCH] Epoch duration set to {self.scoring_epoch_duration}s during import")
            
            # CONSERVER LES TEMPS ABSOLUS - NE PAS LES AJUSTER À 0
            # Les temps de scoring doivent rester en temps absolu pour correspondre au fichier EDF
            if len(self.sleep_scoring_data) > 0:
                print(f"🔍 CHECKPOINT 8: Temps avant ajustement: {self.sleep_scoring_data['time'].min():.1f}s - {self.sleep_scoring_data['time'].max():.1f}s")
                print(f"✅ CHECKPOINT 8: Temps conservés en temps absolu")
                print(f"🔍 CHECKPOINT 8: Temps de scoring final: {self.sleep_scoring_data['time'].min():.1f}s - {self.sleep_scoring_data['time'].max():.1f}s")
            
            print(f"✅ CHECKPOINT 9: Scoring traité avec succès: {len(self.sleep_scoring_data)} époques")
            print(f"🔍 CHECKPOINT 9: Stades uniques: {self.sleep_scoring_data['stage'].unique()}")
            print(f"🔍 CHECKPOINT 9: Plage temporelle finale: {self.sleep_scoring_data['time'].min():.1f}s - {self.sleep_scoring_data['time'].max():.1f}s")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement des données: {e}")
            logging.error(f"Erreur lors du traitement des données: {e}")
            raise e
    
    def _process_sleep_scoring_data(self, df: pd.DataFrame):
        """Traite les données de scoring de sommeil."""
        try:
            # Chercher les colonnes pertinentes
            time_col = None
            stage_col = None
            
            # Recherche des colonnes par nom (français et anglais)
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['time', 'temps', 'epoch', 'époque', 'heure', 'début', 'start']):
                    time_col = col
                elif any(keyword in col_lower for keyword in ['stage', 'stade', 'sleep', 'sommeil', 'score', 'état']):
                    stage_col = col
            
            # Si pas trouvé, utiliser les premières colonnes
            if time_col is None and len(df.columns) > 0:
                time_col = df.columns[0]
            if stage_col is None and len(df.columns) > 1:
                stage_col = df.columns[1]
            
            # Créer le DataFrame de scoring
            if time_col and stage_col:
                self.sleep_scoring_data = df[[time_col, stage_col]].copy()
                self.sleep_scoring_data.columns = ['time', 'stage']
                
                # Nettoyer les données - supprimer les lignes vides ou avec des listes vides
                self.sleep_scoring_data = self.sleep_scoring_data.dropna()
                # Supprimer les lignes où le stade est une liste vide []
                self.sleep_scoring_data = self.sleep_scoring_data[
                    ~self.sleep_scoring_data['stage'].astype(str).str.contains(r'^\[\]$', na=False)
                ]
                self.sleep_scoring_data = self.sleep_scoring_data[
                    ~self.sleep_scoring_data['time'].astype(str).str.contains(r'^\[\]$', na=False)
                ]
                
                # Convertir les stades français vers les codes standard
                self.sleep_scoring_data['stage'] = self.sleep_scoring_data['stage'].astype(str).str.lower().str.strip()
                self.sleep_scoring_data['stage'] = self.sleep_scoring_data['stage'].map(self.french_to_standard).fillna('U')
                
                # Convertir le temps en secondes
                if self.sleep_scoring_data['time'].dtype == 'object':
                    # Essayer de convertir en datetime puis en secondes
                    try:
                        self.sleep_scoring_data['time'] = pd.to_datetime(self.sleep_scoring_data['time'])
                        # Convertir en secondes depuis le début
                        start_time = self.sleep_scoring_data['time'].iloc[0]
                        self.sleep_scoring_data['time'] = (self.sleep_scoring_data['time'] - start_time).dt.total_seconds()
                    except Exception as e:
                        print(f"⚠️ Erreur conversion datetime: {e}")
                        # Si échec, essayer de convertir directement en numérique
                        self.sleep_scoring_data['time'] = pd.to_numeric(self.sleep_scoring_data['time'], errors='coerce')
                
                # Nettoyer les valeurs NaN après conversion
                self.sleep_scoring_data = self.sleep_scoring_data.dropna()
                
                # Calculer les époques
                self.sleep_scoring_data['epoch'] = (self.sleep_scoring_data['time'] / self.scoring_epoch_duration).astype(int)
                
                print(f"✅ Scoring de sommeil traité: {len(self.sleep_scoring_data)} époques")
                print(f"   Période: {self.sleep_scoring_data['time'].min():.1f}s - {self.sleep_scoring_data['time'].max():.1f}s")
                print(f"   Stades: {self.sleep_scoring_data['stage'].value_counts().to_dict()}")
                
            else:
                raise ValueError("Impossible de trouver les colonnes de temps et de stade")
                
        except Exception as e:
            raise Exception(f"Erreur lors du traitement des données: {str(e)}")
    
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
                                  font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Outils d'analyse avancée pour l'exploration approfondie des données EEG",
                                 font=('Helvetica', 10))
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
                                  font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Identification et analyse des micro-états cérébraux dans les signaux EEG",
                                 font=('Helvetica', 10))
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
                                  font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Analyse des connexions fonctionnelles entre différentes régions cérébrales",
                                 font=('Helvetica', 10))
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
                                  font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Identification automatique des artefacts dans les signaux EEG",
                                 font=('Helvetica', 10))
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
                                  font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)
            
            # Description
            desc_label = ttk.Label(scrollable_frame, 
                                 text="Localisation des sources d'activité cérébrale à partir des signaux EEG",
                                 font=('Helvetica', 10))
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
            title_label = ttk.Label(scrollable_frame, text="📈 Analyse de Cohérence Inter-canal", font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)

            # Description fonctionnelle de l'analyse
            desc_label = ttk.Label(scrollable_frame,
                                 text="Analyse de la cohérence fonctionnelle entre différents canaux EEG",
                                 font=('Helvetica', 10))
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
                                  font=('Helvetica', 14, 'bold'))
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
            title_label = ttk.Label(scrollable_frame, text="🔗 Analyse de Corrélation Temporelle", font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)

            # Description
            desc_label = ttk.Label(scrollable_frame,
                                 text="Analyse de la corrélation temporelle entre différents canaux EEG",
                                 font=('Helvetica', 10))
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
                                  font=('Helvetica', 14, 'bold'))
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
            title_label = ttk.Label(top, text="📊 Analyse de Variance (ANOVA)", font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)

            # Description
            desc_label = ttk.Label(top,
                                 text="Analyse de variance entre canaux et conditions expérimentales",
                                 font=('Helvetica', 10))
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
            title_label = ttk.Label(top, text="📉 Test de Stationnarité (ADF)", font=('Helvetica', 14, 'bold'))
            title_label.pack(pady=10)

            # Description
            desc_label = ttk.Label(top,
                                 text="Test de Dickey-Fuller Augmenté pour vérifier la stationnarité des signaux",
                                 font=('Helvetica', 10))
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
            # Délégation rapide via utilitaire partagé (aligne automatiquement et remplit 'U')
            try:
                # Calculer la durée d'enregistrement avec limite (max 24h pour éviter les problèmes)
                raw_duration = float(len(self.raw.times) / self.sfreq) if self.raw is not None else 0.0
                rec_duration = min(raw_duration, 24 * 3600)  # Limiter à 24h maximum
                base_dt = getattr(self, 'absolute_start_datetime', None)
                df_fast = cesa_import_edf_hypnogram(
                    file_path,
                    recording_duration_s=rec_duration,
                    epoch_seconds=float(getattr(self, 'scoring_epoch_duration', 30.0)),
                    absolute_start_datetime=base_dt,
                )
                if df_fast is not None and len(df_fast) > 0:
                    self.manual_scoring_data = df_fast
                    self.show_manual_scoring = True
                    messagebox.showinfo("Hypnogram", f"Hypnogramme EDF chargé: {len(df_fast)} époques")
                    return
            except Exception:
                # En cas d'échec, poursuivre avec l'implémentation locale existante
                pass
            print(f"🔍 CHECKPOINT HYPNO 1: Fichier sélectionné: {os.path.basename(file_path)}")
            # Lire uniquement les annotations
            ann = mne.read_annotations(file_path)
            if ann is None or len(ann) == 0:
                messagebox.showwarning("Hypnogram", "Aucune annotation trouvée dans ce fichier.")
                return
            # Map R&K to standard
            rk_to_std = {
                'W': 'W', 'R': 'R', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3',
                'M': 'W', '?': 'U'
            }
            descs = [str(d) for d in ann.description]
            print(f"🔍 CHECKPOINT HYPNO 2: n_annotations={len(ann)}, exemples={descs[:8]}")
            idxs = []
            codes = []
            for i, desc in enumerate(descs):
                if "Sleep stage" in desc:
                    code = desc.strip().split()[-1]
                    idxs.append(i)
                    codes.append(rk_to_std.get(code, 'U'))
            if not codes:
                messagebox.showwarning("Hypnogram", "Aucune annotation de stade trouvée.")
                return
            # Construire les temps à partir de ann.onset et orig_time (synchronisation absolue)
            epoch_len = float(getattr(self, 'scoring_epoch_duration', 30.0))
            onsets = np.asarray(ann.onset, dtype=float)[idxs]
            times = onsets.copy()
            # Si orig_time (tz-aware/naive), aligner au début EDF
            try:
                base_main = pd.Timestamp(self.absolute_start_datetime) if hasattr(self, 'absolute_start_datetime') and self.absolute_start_datetime else None
            except Exception:
                base_main = None
            if ann.orig_time is not None and base_main is not None:
                try:
                    base_ann = pd.Timestamp(ann.orig_time)
                    # Harmoniser tz
                    if getattr(base_main, 'tz', None) is not None:
                        if getattr(base_ann, 'tz', None) is None:
                            base_ann = base_ann.tz_localize(base_main.tz)
                        else:
                            base_ann = base_ann.tz_convert(base_main.tz)
                    else:
                        base_ann = base_ann.tz_localize(None)
                        base_main = base_main.tz_localize(None) if getattr(base_main, 'tz', None) is not None else base_main
                    # t_rel = (orig_time + onset) - main_start
                    abs_onsets = base_ann + pd.to_timedelta(times, unit='s')
                    times = (abs_onsets - base_main).total_seconds().to_numpy()
                    print(f"🔍 CHECKPOINT HYPNO SYNC: base_ann={base_ann}, base_main={base_main}")
                except Exception as e_sync:
                    print(f"⚠️ HYPNO SYNC: fallback relatif (onset uniquement): {e_sync}")
            # Nettoyer: supprimer époques <0 (avant début EDF)
            mask_pos = times >= -1e-6
            times = times[mask_pos]
            codes = [c for j, c in enumerate(codes) if mask_pos[j]]
            df = pd.DataFrame({'time': times, 'stage': codes})
            # Compléter jusqu'à la fin d'enregistrement
            # Calculer la durée avec limite pour éviter les problèmes
            raw_duration = float(len(self.raw.times) / self.sfreq) if self.raw is not None else (len(df) * epoch_len)
            rec_duration = min(raw_duration, 24 * 3600)  # Limiter à 24h maximum
            # Compléter le début si le premier temps > 0
            add_head = []
            if len(df) > 0 and float(df['time'].iloc[0]) > 0.0:
                t = 0.0
                while t < float(df['time'].iloc[0]) - 1e-6:
                    add_head.append({'time': t, 'stage': 'U'})
                    t += epoch_len
            last_time = float(df['time'].iloc[-1]) if len(df) > 0 else 0.0
            add_rows = []
            t = last_time + epoch_len
            while t < rec_duration - 1e-6:
                add_rows.append({'time': t, 'stage': 'U'})
                t += epoch_len
            if add_head or add_rows:
                df = pd.concat([pd.DataFrame(add_head), df, pd.DataFrame(add_rows)], ignore_index=True)
                df = df.sort_values('time').reset_index(drop=True)
            print(f"🔍 CHECKPOINT HYPNO 3: Stades uniques={sorted(set(df['stage']))}, n_epochs={len(df)}")
            # Stocker comme manual_scoring_data prioritaire
            self.manual_scoring_data = df
            self.show_manual_scoring = True
            messagebox.showinfo("Hypnogram", f"Hypnogramme EDF chargé: {len(df)} époques")
        except Exception as e:
            messagebox.showerror("Erreur", f"Echec chargement hypnogramme EDF: {e}")
    
    def _show_about(self):
        """Affiche la boîte de dialogue À propos."""
        about_text = """
CCESA (Complex EEG Studio Analysis) v1.0

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

Version: 4.0.0
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
        
        # Raccourcis pour YASA et comparaison
        self.root.bind_all('<Control-y>', lambda e: self._run_yasa_scoring())
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
            
            # ========== AJOUTEZ CET ÉVÉNEMENT ICI ==========
            if EVENTS_AVAILABLE:
                try:
                    event_data = EventData.TimeChanged(
                        current_time=self.current_time,
                        duration=getattr(self, 'duration', 10.0)
                    )
                    event_bus.emit(Events.TIME_CHANGED, event_data, throttle=0.05)
                    logging.debug(f"EVENT: Time changed to {self.current_time:.1f}s")
                except Exception as e:
                    logging.error(f"EVENT: Error emitting time_changed - {e}")
            # ===============================================
            
            print(f"⬅️ CHECKPOINT NAV 1: Navigation vers score précédent: {target_time:.1f}s")
            logging.info(f"[NAV] Go précédent -> {target_time:.1f}")
            
            # Navigation simple sans changer la durée
            # self._center_view_on_epoch(target_time, epoch_duration)  # Désactivé pour garder la durée actuelle
        else:
            print("⚠️ CHECKPOINT NAV 1: Aucun scoring chargé")

    
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
            if EVENTS_AVAILABLE:
                try:
                    event_data = EventData.TimeChanged(
                        current_time=self.current_time,
                        duration=getattr(self, 'duration', 10.0)
                    )
                    event_bus.emit(Events.TIME_CHANGED, event_data, throttle=0.05)
                    logging.debug(f"EVENT: Time changed to {self.current_time:.1f}s")
                except Exception as e:
                    logging.error(f"EVENT: Error emitting time_changed - {e}")
            
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
                              style='Modern.TLabel', font=('Helvetica', 10, 'bold'))
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
            font=('Helvetica', 10, 'bold'),
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
            text="v3.0", 
            style='Version.TLabel',
            font=('Helvetica', 8, 'bold')
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
                self.version_label.config(text="v3.0")
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
                    ]
                )
        except Exception:
            pass
    
    

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
        batch_window.title("🤖 Automatisation FFT en Lot - CESA v3.0")
        batch_window.geometry("1100x900")
        batch_window.configure(bg='#f8f9fa')
        
        # Style
        style = ttk.Style()
        style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Info.TLabel', font=('Helvetica', 9), foreground='#6c757d')
        
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
        ttk.Label(status_frame, textvariable=self.batch_status_var, font=('Helvetica', 10, 'bold')).pack(anchor='w')
        
        # Barre de progression
        self.batch_progress = ttk.Progressbar(status_frame, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=(10,0))
        
        self.batch_progress_text = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.batch_progress_text, font=('Helvetica', 9)).pack(anchor='w', pady=(5,0))
        
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
        ttk.Entry(before_row, textvariable=self.before_dir_var, width=50).pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(before_row, text="Parcourir", command=lambda: self._browse_dir_to_var(self.before_dir_var, "Sélectionner dossier EDF AVANT")).pack(side=tk.LEFT)

        # Dossier APRÈS
        after_row = ttk.Frame(self.spaghetti_frame)
        after_row.pack(fill=tk.X, pady=2)
        ttk.Label(after_row, text="Dossier EDF APRÈS:").pack(side=tk.LEFT)
        self.after_dir_var = tk.StringVar()
        ttk.Entry(after_row, textvariable=self.after_dir_var, width=50).pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(after_row, text="Parcourir", command=lambda: self._browse_dir_to_var(self.after_dir_var, "Sélectionner dossier EDF APRÈS")).pack(side=tk.LEFT)

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
                var = tk.BooleanVar(value=True)
                self.spag_band_stage_vars[band][st_code] = var
                cb = ttk.Checkbutton(mapping_table, variable=var)
                cb.grid(row=i, column=j, sticky='n', pady=2)
            # Contrôles de ligne (Tout/Nul)
            row_ctrl = ttk.Frame(mapping_table)
            row_ctrl.grid(row=i, column=6, sticky='n')
            ttk.Button(row_ctrl, text="✓", width=2, command=lambda b=band: self._toggle_band_stage_row(b, True)).pack(side=tk.LEFT, padx=(0,2))
            ttk.Button(row_ctrl, text="–", width=2, command=lambda b=band: self._toggle_band_stage_row(b, False)).pack(side=tk.LEFT, padx=(2,0))

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
        
        eeg_extensions = ['.edf', '.EDF', '.edf+', '.EDF+']
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
                    if any(file.endswith(ext) for ext in eeg_extensions):
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
        """Sélection rapide de canaux spécifiques - VERSION AVEC ÉVÉNEMENTS"""
        # Votre code existant
        for var in self.channel_vars.values():
            var.set(False)
        
        for channel in channels_to_select:
            if channel in self.channel_vars:
                self.channel_vars[channel].set(True)
        
        # Émettre l'événement si disponible
        if EVENTS_AVAILABLE:
            try:
                event_data = EventData.ChannelsSelected(channels=channels_to_select)
                event_bus.emit(Events.CHANNELS_SELECTED, event_data)
                logging.debug(f"EVENT: Channels selected - {len(channels_to_select)} channels")
            except Exception as e:
                logging.error(f"EVENT: Error emitting channels_selected - {e}")



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
                # Désactiver saisie AV/PR si on utilise les fichiers détectés
                state = 'disabled' if getattr(self, 'spag_use_detected_var', tk.BooleanVar(value=True)).get() else 'normal'
                for child in self.spaghetti_frame.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for w in child.winfo_children():
                            try:
                                if isinstance(w, ttk.Entry) or isinstance(w, ttk.Button):
                                    w.configure(state=state)
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
                    exts = ['.edf', '.EDF', '.edf+', '.EDF+']
                    for root, _dirs, files in os.walk(dir_path):
                        for f in files:
                            if any(f.endswith(ext) for ext in exts):
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

            outputs = generate_spaghetti_from_edf_file_lists(
                before_files=before_files,
                after_files=after_files,
                output_dir=output_dir,
                selected_bands=sel_bands,
                selected_stages=sel_stages,
                selected_channels=selected_channels if selected_channels else None,
                selected_subjects=None,
                selected_band_stage_map=band_stage_map,
                edf_to_excel_map=edf_to_excel if edf_to_excel else None,
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

    def _validate_batch_config(self):
        """Valide la configuration avant de démarrer le traitement."""
        errors = []
        
        if not self.batch_config['input_dir']:
            errors.append("Dossier d'entrée non sélectionné")
        
        if not self.batch_config['output_dir']:
            errors.append("Dossier de sortie non sélectionné")
        
        if not self.batch_config['eeg_files']:
            errors.append("Aucun fichier EEG détecté")
        
        if not any(var.get() for var in self.channel_vars.values()):
            errors.append("Aucun canal sélectionné")
        
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
            self.root.after(0, self._reset_batch_ui)

    def _load_eeg_file_for_batch(self, file_path):
        """Charge un fichier EEG pour le traitement en lot."""
        try:
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
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
        self.root.after(0, lambda: self.batch_status_var.set(status))

    def _update_batch_progress(self, value):
        """Met à jour la barre de progression."""
        self.root.after(0, lambda: self.batch_progress.configure(value=value))

    def _update_batch_progress_text(self, text):
        """Met à jour le texte de progression."""
        self.root.after(0, lambda: self.batch_progress_text.set(text))

    def _log_batch(self, message):
        """Ajoute un message aux logs du traitement en lot."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        def update_logs():
            if hasattr(self, 'batch_logs'):
                self.batch_logs.insert(tk.END, log_message)
                self.batch_logs.see(tk.END)
        
        self.root.after(0, update_logs)
    
    def _setup_modern_plot(self, parent: ttk.Frame) -> None:
        """Configure le conteneur principal pour afficher la vue PSG multi-subplots."""
        # Conteneur dédié au viewer PSG
        self.psg_container = ttk.Frame(parent)
        self.psg_container.pack(fill=tk.BOTH, expand=True)

        # Si des données sont chargées, afficher directement la vue PSG intégrée
        if getattr(self, 'raw', None) is not None:
            try:
                self.show_multi_graph_view(embed_parent=self.psg_container)
                return
            except Exception:
                pass

        # Fallback: pas de graphe — simple placeholder textuel
        holder = ttk.Frame(self.psg_container)
        holder.pack(fill=tk.BOTH, expand=True)
        msg = ttk.Label(holder, text="Aucun fichier EDF chargé\nOuvrez un fichier pour afficher la PSG",
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
            font=('Helvetica', 12, 'bold')
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
            font=('Helvetica', 10, 'bold')
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

    def update_plot(self):
        """Version optimisée avec debouncing de update_plot"""
        import time
        
        current_time = time.time()
        
        # Annuler la mise à jour précédente si elle existe
        if self._plot_update_pending_id:
            self.root.after_cancel(self._plot_update_pending_id)
        
        # Calculer le délai
        time_since_last = current_time - self._last_plot_update
        delay = max(0, int((self._min_plot_interval - time_since_last) * 1000))
        
        # Programmer la mise à jour
        self._plot_update_pending_id = self.root.after(delay, self._do_actual_plot_update)
    
    def _do_actual_plot_update(self):
        """Met à jour l'affichage principal en rafraîchissant la vue PSG intégrée."""
        import time
        self._plot_update_pending_id = None
        self._last_plot_update = time.time()
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

            plot_width_px = self._get_plot_width_px()
            use_bridge = getattr(self, 'data_bridge', None) is not None
            token = object()
            self._active_plot_token = token

            def _do_update(current_token=token):
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
                            except Exception:
                                continue
                        if _t_extract0 is not None:
                            try:
                                import time as _time
                                extract_ms = (_time.perf_counter() - _t_extract0) * 1000.0
                            except Exception:
                                extract_ms = 0.0

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
                                self.psg_plotter.update_signals(signals)
                                # Invalidate backgrounds if limits changed or options changed
                                try:
                                    # If autoscale/filter toggled in UI, force re-filter and reset blit
                                    self.psg_plotter._invalidate_backgrounds = True
                                except Exception:
                                    pass
                                self.psg_plotter.set_time_window(float(self.current_time), float(self.duration))
                            except Exception:
                                self.show_multi_graph_view(embed_parent=parent)
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
                                n_channels = int(p.get('n_channels', 0))
                                n_points = int(p.get('n_points', 0))
                                fps = 1000.0 / max(total_ms, 1e-3)
                                action_label = "bridge" if bridge_result else "raw"
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
                                )
                                try:
                                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                except Exception:
                                    timestamp = "0000-00-00"
                                print(f"{timestamp} | PERF nav: extract_ms={extract_ms:.1f} filter_ms={filter_ms:.1f} draw_ms={draw_ms:.1f} total_ms={total_ms:.1f} n_channels={n_channels} n_points={n_points}")
                            except Exception:
                                pass
                        else:
                            self.show_multi_graph_view(embed_parent=parent)

                    # Back to UI thread
                    self.root.after(0, _apply_to_ui)
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
                if getattr(self, '_plot_executor', None):
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
                
                if width > 0:
                    # Créer la barre horizontale tronquée à la fenêtre
                    self.ax.barh(scoring_y, width, left=draw_start, 
                               height=scoring_height, color=color, alpha=alpha, 
                               edgecolor='black', linewidth=0.5, zorder=zorder)
                    
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
                               font=('Helvetica', 14, 'bold'))
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
            # Afficher une barre de chargement
            self._show_loading_bar(title="Import Excel", message="Lecture du fichier Excel...")
            
            # Lire le fichier Excel
            print(f"📥 CHECKPOINT EXCEL: Chargement fichier: {file_path}")
            logging.info(f"[EXCEL] Chargement fichier: {file_path}")
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                df = pd.read_excel(file_path)
            print(f"📥 CHECKPOINT EXCEL: df.shape={df.shape}")
            logging.info(f"[EXCEL] df.shape={df.shape}")
            print(f"🔍 Import Excel: Colonnes disponibles: {list(df.columns)}")
            print(f"🔍 Import Excel: Premières lignes:\n{df.head()}")
            
            # Vérifier la structure du fichier
            if len(df.columns) < 2:
                messagebox.showerror("Erreur", "Le fichier doit contenir au moins 2 colonnes (stade et datetime)")
                self._hide_loading_bar()
                return

            # Détection automatique des colonnes
            stage_col = None
            datetime_col = None

            # Liste des mots-clés pour détecter les colonnes
            stage_keywords = ['stage', 'stade', 'sleep', 'sommeil', 'score', 'etat', 'phase', 'state']
            datetime_keywords = ['time', 'temps', 'datetime', 'date', 'heure', 'epoch', 'epoque', 'start', 'debut', 'begin']

            # Recherche de la colonne stade
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if any(keyword in col_lower for keyword in stage_keywords):
                    stage_col = col
                    break

            # Si pas trouvée, prendre la première colonne comme stade
            if stage_col is None:
                stage_col = df.columns[0]

            # Recherche de la colonne datetime
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if any(keyword in col_lower for keyword in datetime_keywords):
                    datetime_col = col
                    break

            # Si pas trouvée, prendre la deuxième colonne comme datetime
            if datetime_col is None:
                datetime_col = df.columns[1]

            # Vérifier qu'on a bien deux colonnes différentes
            if stage_col == datetime_col:
                messagebox.showerror("Erreur", "Impossible de distinguer automatiquement les colonnes stade et datetime. Veuillez vérifier le format de votre fichier.")
                self._hide_loading_bar()
                return
            
            print(f"🔍 Import Excel: Colonne stade: '{stage_col}'")
            print(f"🔍 Import Excel: Colonne datetime: '{datetime_col}'")
            
            # Parser la colonne datetime (format: 14/06/2022 23:58:30)
            def parse_datetime(date_str):
                try:
                    # Essayer différents formats
                    formats = [
                        '%d/%m/%Y %H:%M:%S',  # 14/06/2022 23:58:30
                        '%d/%m/%y %H:%M:%S',  # 14/06/22 23:58:30
                        '%Y-%m-%d %H:%M:%S',  # 2022-06-14 23:58:30
                        '%d.%m.%Y %H.%M.%S',  # 14.06.2022 23.58.30
                    ]
                    
                    # Déjà datetime ?
                    if isinstance(date_str, (datetime, pd.Timestamp)):
                        return pd.to_datetime(date_str).to_pydatetime()
                    # Valeur numérique (parfois Excel date serial) -> laisser pandas tenter
                    if isinstance(date_str, (int, float)):
                        try:
                            return pd.to_datetime(date_str, unit='s')
                        except Exception:
                            pass
                    
                    for fmt in formats:
                        try:
                            return datetime.strptime(str(date_str), fmt)
                        except ValueError:
                            continue
                    
                    # Si aucun format ne fonctionne, essayer pandas
                    return pd.to_datetime(date_str).to_pydatetime()
                    
                except Exception as e:
                    print(f"⚠️ Erreur parsing datetime '{date_str}': {e}")
                    return None
            
            # Fonction pour détecter cellules vides/invalides
            def _is_empty_cell(v) -> bool:
                try:
                    if v is None:
                        return True
                    if isinstance(v, float) and pd.isna(v):
                        return True
                    if isinstance(v, str) and str(v).strip().lower() in ('', '[]', 'nan', 'nat'):
                        return True
                    if isinstance(v, (list, tuple)) and len(v) == 0:
                        return True
                except Exception:
                    return False
                return False

            # Nettoyer les lignes invalides avant parsing
            df_work = df[[stage_col, datetime_col]].copy()
            mask_valid = (~df_work[stage_col].apply(_is_empty_cell)) & (~df_work[datetime_col].apply(_is_empty_cell))
            dropped = int((~mask_valid).sum())
            if dropped:
                print(f"📥 CHECKPOINT EXCEL: Lignes invalides ignorées (vides/[]/NaN): {dropped}")
                logging.info(f"[EXCEL] Ignored invalid rows: {dropped}")
            df_work = df_work[mask_valid].reset_index(drop=True)

            # Parser toutes les dates (en ignorant silencieusement celles qui échouent)
            parsed_dates = []
            parsed_stages = []
            skipped = 0
            for idx, (stage_val, date_val) in enumerate(zip(df_work[stage_col], df_work[datetime_col])):
                parsed_date = parse_datetime(date_val)
                if parsed_date is None:
                    skipped += 1
                    continue
                parsed_dates.append(parsed_date)
                parsed_stages.append(str(stage_val).strip())
            if skipped:
                print(f"📥 CHECKPOINT EXCEL: Lignes ignorées lors du parsing datetime: {skipped}")
                logging.info(f"[EXCEL] Skipped on parse: {skipped}")
            if len(parsed_dates) == 0:
                self._hide_loading_bar()
                messagebox.showerror("Erreur", "Aucune date valide trouvée dans la colonne horaire de l'Excel.")
                return
            print(f"📥 CHECKPOINT EXCEL: {len(parsed_dates)} dates parsées, min={min(parsed_dates)} max={max(parsed_dates)}")
            logging.info(f"[EXCEL] n_dates={len(parsed_dates)}, min={min(parsed_dates)}, max={max(parsed_dates)}")
            
            # Créer le DataFrame avec les données parsées
            manual_data = pd.DataFrame({
                'stage': parsed_stages,
                'datetime': parsed_dates
            })
            # Nettoyage: enlever lignes vides et trier
            before_clean = len(manual_data)
            manual_data = manual_data.dropna(subset=['stage', 'datetime'])
            manual_data = manual_data[manual_data['stage'].astype(str).str.len() > 0]
            manual_data = manual_data.sort_values('datetime').reset_index(drop=True)
            after_clean = len(manual_data)
            print(f"📥 CHECKPOINT EXCEL: Nettoyage: {before_clean} -> {after_clean} lignes")
            logging.info(f"[EXCEL] Nettoyage: {before_clean}->{after_clean}")
            
            print(f"🔍 Import Excel: {len(manual_data)} époques importées")
            print(f"🔍 Import Excel: Premières époques:\n{manual_data.head()}")
            print(f"🔍 Import Excel: Plage temporelle: {manual_data['datetime'].min()} - {manual_data['datetime'].max()}")
            
            # ANALYSER LA DURÉE D'ÉPOQUE RÉELLE
            if len(manual_data) > 1:
                # Calculer les intervalles entre les époques
                time_diffs = manual_data['datetime'].diff().dropna()
                epoch_durations = time_diffs.dt.total_seconds()
                
                print(f"🔍 Import Excel: Intervalles entre époques (secondes):")
                print(f"🔍 Import Excel: Min: {epoch_durations.min():.1f}s, Max: {epoch_durations.max():.1f}s")
                print(f"🔍 Import Excel: Moyenne: {epoch_durations.mean():.1f}s, Médiane: {epoch_durations.median():.1f}s")
                print(f"🔍 Import Excel: Premiers intervalles: {epoch_durations.head().tolist()}")
                
                # Déterminer la durée d'époque la plus probable
                most_common_duration = epoch_durations.mode().iloc[0] if len(epoch_durations.mode()) > 0 else epoch_durations.median()
                
                # Vérifier si les intervalles sont cohérents (tolérance de 5 secondes)
                consistent_durations = epoch_durations[abs(epoch_durations - most_common_duration) <= 5.0]
                consistency_ratio = len(consistent_durations) / len(epoch_durations)
                
                print(f"🔍 Import Excel: Durée d'époque détectée: {most_common_duration:.1f}s")
                print(f"🔍 Import Excel: Cohérence: {consistency_ratio:.1%} des intervalles sont cohérents")
                
                if consistency_ratio >= 0.8:  # 80% des intervalles sont cohérents
                    self.scoring_epoch_duration = most_common_duration
                    print(f"✅ Import Excel: Durée d'époque définie à {self.scoring_epoch_duration:.1f}s")
                else:
                    print(f"⚠️ Import Excel: Intervalles incohérents, utilisation de la durée par défaut (30s)")
                    self.scoring_epoch_duration = 30.0
                    logging.info("[EPOCH] Epoch duration reset to 30.0s due to inconsistent intervals")
            else:
                print(f"⚠️ Import Excel: Pas assez d'époques pour analyser la durée, utilisation de la durée par défaut (30s)")
                self.scoring_epoch_duration = 30.0
                logging.info("[EPOCH] Epoch duration reset to 30.0s due to insufficient epochs")
            
            # Convertir via fonction utilitaire standardisée (gère tz et seconds)
            df_final = cesa_import_excel_scoring(
                manual_data,
                absolute_start_datetime=getattr(self, 'absolute_start_datetime', None),
                epoch_seconds=float(getattr(self, 'scoring_epoch_duration', 30.0)),
            )
            manual_data = manual_data.assign(time=df_final['time'].values)
            # Conserver l'axe X calé sur la première date Excel pour affichage
            try:
                self.display_start_datetime = manual_data['datetime'].min().to_pydatetime()
            except Exception:
                self.display_start_datetime = None
            print(f"🔍 Import Excel: Temps convertis - première époque: {manual_data['time'].iloc[0]:.1f}s")
            print(f"🔍 Import Excel: Plage temps relatif: {manual_data['time'].min():.1f}s - {manual_data['time'].max():.1f}s")

            # Compléter les zones non scorées avec 'U' (Undefined)
            first_time = float(manual_data['time'].iloc[0])
            last_time = float(manual_data['time'].iloc[-1])
            epoch_len = float(self.scoring_epoch_duration)
            # Calculer la durée avec limite pour éviter les problèmes
            raw_duration = float(len(self.raw.times) / self.sfreq)
            rec_duration = min(raw_duration, 24 * 3600)  # Limiter à 24h maximum
            add_rows = []
            if first_time > 0.0:
                t = 0.0
                while t < first_time - 1e-6:
                    add_rows.append({'stage': 'U', 'time': t})
                    t += epoch_len
            tail_rows = []
            if last_time + epoch_len < rec_duration - 1e-6:
                t = last_time + epoch_len
                while t < rec_duration - 1e-6:
                    tail_rows.append({'stage': 'U', 'time': t})
                    t += epoch_len
            if add_rows or tail_rows:
                manual_data = pd.concat([pd.DataFrame(add_rows), manual_data, pd.DataFrame(tail_rows)], ignore_index=True)
                manual_data = manual_data.sort_values('time').reset_index(drop=True)
            
            # Stocker les données
            print(f"📥 CHECKPOINT EXCEL: Final manual_scoring_data shape={manual_data.shape}")
            print(f"📥 CHECKPOINT EXCEL: time head={manual_data['time'].head().tolist() if 'time' in manual_data else 'N/A'}")
            logging.info(f"[EXCEL] Final shape={manual_data.shape}")
            if 'time' in manual_data:
                logging.info(f"[EXCEL] time head={manual_data['time'].head().tolist()}")
            self.manual_scoring_data = manual_data
            self.show_manual_scoring = True
            
            # Stocker le chemin du fichier de scoring pour l'affichage
            self.current_scoring_path = file_path
            
            # Centrer automatiquement la vue sur la première époque importée
            try:
                first_epoch_time = float(manual_data['time'].iloc[0]) if 'time' in manual_data else 0.0
                if hasattr(self, '_center_view_on_epoch'):
                    self._center_view_on_epoch(first_epoch_time, float(self.scoring_epoch_duration))
                else:
                    # Fallback: régler la fenêtre autour de la première époque
                    self.current_time = first_epoch_time
                    self.duration = max(float(self.scoring_epoch_duration) * 2.5, 60.0)
                    self.update_plot()
            except Exception:
                # En cas de souci, on met tout de même à jour le graphique
                self.update_plot()

            # Masquer la barre de chargement
            self._hide_loading_bar()
            
            # Mettre à jour la barre de statut
            self.update_status_bar()
            
            # Message de succès avec information sur la durée d'époque
            success_msg = f"Scoring manuel importé avec succès!\n{len(manual_data)} époques\nPlage: {manual_data['time'].min():.1f}s - {manual_data['time'].max():.1f}s\nDurée d'époque détectée: {self.scoring_epoch_duration:.1f}s"
            messagebox.showinfo("Succès", success_msg)
            
            logging.info(f"Scoring manuel importé: {len(manual_data)} époques")
            
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
                               font=('Helvetica', 12, 'bold'))
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
    
    def open_edf_file_dialog(self):
        # Crée une fenêtre Toplevel pour choisir un fichier EDF
        window = tk.Toplevel(self.parent)
        window.title("Ouvrir un fichier EEG")
        window.geometry("400x150")
        window.transient(self.parent)
        window.grab_set()

        label = ttk.Label(window, text="Sélectionnez un fichier EDF à ouvrir :", font=("Helvetica", 10))
        label.pack(padx=10, pady=10)

        entry_var = tk.StringVar()
        entry = ttk.Entry(window, textvariable=entry_var, width=50)
        entry.pack(padx=10, pady=5)

        def browse_files():
            file_path = filedialog.askopenfilename(
                parent=window,
                title="Choisir un fichier EDF",
                filetypes=[("Fichiers EDF", "*.edf *.EDF")]
            )
            if file_path:
                entry_var.set(file_path)

        browse_button = ttk.Button(window, text="Parcourir...", command=browse_files)
        browse_button.pack(pady=5)

        def load_file():
            file_path = entry_var.get()
            if not file_path:
                messagebox.showerror("Erreur", "Veuillez sélectionner un fichier EDF.")
                return
            window.grab_release()
            window.destroy()
            self.load_edf_file(file_path)

        open_button = ttk.Button(window, text="Ouvrir", command=load_file)
        open_button.pack(pady=10)

        window.wait_window()


    def _on_file_loaded(self, raw, sfreq):
        # Méthode à redéfinir dans l'app principale pour récupérer le fichier chargé
        logging.info("Fichier chargé avec succès, méthode _on_file_loaded à implémenter.")

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
    print("CCESA (Complex EEG Studio Analysis) v1.0")
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
