# CESA/menu_builder.py
import tkinter as tk
import logging

# --- Classe ScrollableMenu réutilisable ---
class ScrollableMenu:
    def __init__(self, parent, max_items=20, **menu_config):
        self.parent = parent
        self.max_items = max_items
        self.menu_config = menu_config
        self.main_menu = tk.Menu(parent, **menu_config)
        self.current_submenu = None
        self.item_count = 0
        self.submenu_count = 0

    def add_command(self, **kwargs):
        if self.item_count >= self.max_items:
            if self.current_submenu is None:
                self.submenu_count += 1
                submenu_label = "📋 Plus d'options..." if self.submenu_count == 1 else f"📋 Plus d'options... ({self.submenu_count})"
                self.current_submenu = tk.Menu(self.main_menu, **self.menu_config)
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

# --- Classe MenuBuilder --
class MenuBuilder:
    def __init__(self, parent_app):
        self.app = parent_app
        self.root = parent_app.root

    def create_modern_menu(self):
        try:
            menubar = tk.Menu(self.root, bg='#f8f9fa', fg='#212529')
            self.root.config(menu=menubar)

            menu_config = {
                'tearoff': 0,
                'bg': '#f8f9fa',
                'fg': '#212529',
                'font': ('Helvetica', 9),
                'activebackground': '#e9ecef',
                'activeforeground': '#212529'
            }

            def fallback_func(label):
                return lambda: print(f"[FALLBACK: {label}] La fonction pour ce menu n'est pas encore disponible.")

            # --- Menu Fichier ---
            file_menu = ScrollableMenu(menubar, **menu_config)
            menubar.add_cascade(label="📁 Fichier", menu=file_menu.main_menu)

            # Ouvrir EDF
            file_menu.add_command(label="📂 Ouvrir EDF", command=self.app.open_file_dialog, accelerator="Ctrl+O")

            file_menu.add_separator()

            # Exporter Données
            file_menu.add_command(
                label="💾 Exporter Données",
                command=getattr(self.app, "_export_data", fallback_func("Exporter Données")),
                accelerator="Ctrl+S"
            )
            # Exporter segment EDF
            file_menu.add_command(
                label="✂️ Exporter segment EDF...",
                command=getattr(self.app, "_export_edf_segment", fallback_func("Exporter segment EDF"))
            )
            # Exporter rapport
            file_menu.add_command(
                label="📊 Exporter Rapport",
                command=getattr(self.app, "_export_report", fallback_func("Exporter Rapport"))
            )
            file_menu.add_separator()
            # Préférences
            file_menu.add_command(
                label="⚙️ Préférences",
                command=getattr(self.app, "_show_preferences", fallback_func("Préférences"))
            )
            file_menu.add_separator()
            # Automatisation FFT en lot
            file_menu.add_command(
                label="🤖 Automatisation FFT en Lot",
                command=getattr(self.app, "_show_batch_fft_automation", fallback_func("Automatisation FFT en Lot")),
                accelerator="Ctrl+B"
            )
            file_menu.add_separator()
            # Quitter
            file_menu.add_command(
                label="❌ Quitter",
                command=getattr(self.app, "_quit_application", fallback_func("Quitter")),
                accelerator="Ctrl+Q"
            )

            # ---- Pour test : active cette ligne ----
            logging.info("Menu Fichier robuste créé avec fallback OK.")

            # --- Menu Affichage ---
            view_menu = ScrollableMenu(menubar, **menu_config)
            menubar.add_cascade(label="👁️ Affichage", menu=view_menu.main_menu)

            def accel(label, key):
                return {"label": label, "accelerator": key}

            # Sélectionner Canaux
            view_menu.add_command(
                label="📋 Sélectionner Canaux",
                command=getattr(self.app, "_show_channel_selector", fallback_func("Sélectionner Canaux")),
                accelerator="Ctrl+1"
            )
            view_menu.add_separator()

            # Activer Autoscale
            view_menu.add_command(
                label="📏 Activer Autoscale",
                command=getattr(self.app, "_toggle_autoscale", fallback_func("Autoscale")),
                accelerator="Ctrl+A"
            )
            # Configuration Filtres
            view_menu.add_command(
                label="🔧 Configuration Filtres",
                command=getattr(self.app, "show_filter_config", fallback_func("Configuration Filtres")),
                accelerator="Ctrl+F"
            )
            view_menu.add_separator()

            # Vue Multi-Graphiques
            view_menu.add_command(
                label="📊 Vue Multi-Graphiques",
                command=getattr(self.app, "show_multi_graph_view", fallback_func("Vue Multi-Graphiques")),
                accelerator="Ctrl+M"
            )
            view_menu.add_separator()

            # Thème sombre
            view_menu.add_command(
                label="🎨 Thème Sombre",
                command=getattr(self.app, "_toggle_dark_theme", fallback_func("Thème Sombre"))
            )
            view_menu.add_separator()

            # Bascule panneau commandes
            view_menu.add_command(
                label="📋 Bascule Panneau Commandes",
                command=getattr(self.app, "_toggle_control_panel", fallback_func("Bascule Panneau Commandes")),
                accelerator="F2"
            )
            # Actualiser
            view_menu.add_command(
                label="🔄 Actualiser",
                command=getattr(self.app, "_refresh_plot", fallback_func("Actualiser")),
                accelerator="F5"
            )

            # --- Menu Analyse (scrollable, max_items=8) ---
            analysis_menu = ScrollableMenu(menubar, max_items=8, **menu_config)
            menubar.add_cascade(label="📊 Analyse", menu=analysis_menu.main_menu)

            analysis_menu.add_command(
                label="📈 Statistiques",
                command=getattr(self.app, "_show_channel_stats", fallback_func("Statistiques"))
            )
            analysis_menu.add_command(
                label="🔍 Diagnostic",
                command=getattr(self.app, "_show_diagnostics", fallback_func("Diagnostic"))
            )
            analysis_menu.add_separator()
            analysis_menu.add_command(
                label="📉 Analyse Spectrale",
                command=getattr(self.app, "_show_spectral_analysis", fallback_func("Analyse Spectrale"))
            )
            analysis_menu.add_command(
                label="📊 PSD par stade (FFT – Analyse_spectrale)",
                command=getattr(self.app, "_show_stage_psd_fft", fallback_func("PSD par stade"))
            )
            analysis_menu.add_command(
                label="🌈 Spectrogramme ondelettes (avant/après)",
                command=getattr(self.app, "_show_wavelet_spectrogram_before_after", fallback_func("Spectrogramme ondelettes"))
            )
            analysis_menu.add_command(
                label="🧮 Entropie Renormée (Issartel)",
                command=getattr(self.app, "_show_renormalized_entropy", fallback_func("Entropie Renormée"))
            )
            analysis_menu.add_command(
                label="🧮 Analyse périodes (SleepEEGpy)",
                command=getattr(self.app, "_analyze_sleep_periods", fallback_func("Analyse périodes"))
            )
            analysis_menu.add_command(
                label="🌊 Analyse Temporelle",
                command=getattr(self.app, "_show_temporal_analysis", fallback_func("Analyse Temporelle"))
            )
            analysis_menu.add_separator()
            analysis_menu.add_command(
                label="📈 Graphiques Spaghetti (EDF…)",
                command=getattr(self.app, "_generate_spaghetti_from_edf", fallback_func("Graphiques Spaghetti"))
            )
            analysis_menu.add_command(
                label="🤖 Automatisation FFT en Lot",
                command=getattr(self.app, "_show_batch_fft_automation", fallback_func("Automatisation FFT Batch"))
            )
            analysis_menu.add_command(
                label="📊 Analyse Avancée",
                command=getattr(self.app, "_show_advanced_analysis", fallback_func("Analyse Avancée"))
            )
            analysis_menu.add_command(
                label="🔬 Analyse Micro-états",
                command=getattr(self.app, "_show_microstates_analysis", fallback_func("Analyse Micro-états"))
            )
            analysis_menu.add_command(
                label="🧠 Connectivité Cérébrale",
                command=getattr(self.app, "_show_connectivity_analysis", fallback_func("Connectivité Cérébrale"))
            )
            analysis_menu.add_command(
                label="⚡ Détection d'Artefacts",
                command=getattr(self.app, "_show_artifact_detection", fallback_func("Détection d'Artefacts"))
            )
            analysis_menu.add_command(
                label="🎯 Analyse de Sources",
                command=getattr(self.app, "_show_source_analysis", fallback_func("Analyse de Sources"))
            )
            
            # --- Menu Scoring de Sommeil ---
            sleep_menu = ScrollableMenu(menubar, **menu_config)
            menubar.add_cascade(label="🛏️ Sommeil", menu=sleep_menu.main_menu)

            sleep_menu.add_command(
                label="⚙️ Scoring automatique (YASA)",
                command=getattr(self.app, "_run_yasa_scoring", fallback_func("Scoring YASA")),
                accelerator="Ctrl+Y"
            )
            sleep_menu.add_separator()
            sleep_menu.add_command(
                label="📥 Importer Scoring (Excel/EDF)",
                command=getattr(self.app, "_open_scoring_import_hub", fallback_func("Importer Scoring")),
                accelerator="Ctrl+Shift+M"
            )
            sleep_menu.add_command(
                label="🔀 Comparer Auto vs Manuel",
                command=getattr(self.app, "_compare_scoring", fallback_func("Comparer Auto/Manuel")),
                accelerator="Ctrl+C"
            )
            sleep_menu.add_command(
                label="💾 Sauvegarder Scoring (CSV)",
                command=getattr(self.app, "_save_active_scoring", fallback_func("Sauvegarder Scoring"))
            )
            sleep_menu.add_command(
                label="📈 Informations Scoring",
                command=getattr(self.app, "_show_sleep_scoring_info", fallback_func("Informations Scoring"))
            )
            sleep_menu.add_command(
                label="✍️ Scorer manuellement (éditeur)",
                command=getattr(self.app, "_open_manual_scoring_editor", fallback_func("Scoring Manuel")),
                accelerator="Ctrl+Shift+S"
            )
            sleep_menu.add_command(
                label="⚙️ Ajuster Durée Époque",
                command=getattr(self.app, "_adjust_epoch_duration", fallback_func("Ajuster Durée Époque"))
            )

            # --- Menu Outils ---
            tools_menu = ScrollableMenu(menubar, **menu_config)
            menubar.add_cascade(label="🛠️ Outils", menu=tools_menu.main_menu)

            tools_menu.add_command(
                label="🎯 Marqueurs",
                command=getattr(self.app, "_show_markers", fallback_func("Marqueurs"))
            )
            tools_menu.add_command(
                label="📏 Mesures",
                command=getattr(self.app, "_show_measurements", fallback_func("Mesures"))
            )
            tools_menu.add_command(
                label="⏩ Aller au temps...",
                command=getattr(self.app, "_open_goto_time_dialog", fallback_func("Aller au temps"))
            )
            tools_menu.add_separator()
            tools_menu.add_command(
                label="🔧 Configuration Avancée",
                command=getattr(self.app, "_show_advanced_config", fallback_func("Configuration Avancée"))
            )
            # --- Menu Aide ---
            help_menu = ScrollableMenu(menubar, **menu_config)
            menubar.add_cascade(label="❓ Aide", menu=help_menu.main_menu)

            help_menu.add_command(
                label="📖 Assistant de bienvenue",
                command=getattr(self.app, "_show_welcome_assistant", fallback_func("Assistant de bienvenue")),
                accelerator="F1"
            )
            help_menu.add_command(
                label="🔍 Explorateur de fonctionnalités",
                command=getattr(self.app, "_show_feature_explorer", fallback_func("Explorateur de fonctionnalités"))
            )
            help_menu.add_command(
                label="📚 Guide de référence complet",
                command=getattr(self.app, "_open_reference_guide", fallback_func("Guide de référence"))
            )
            help_menu.add_command(
                label="⌨️ Raccourcis Clavier",
                command=getattr(self.app, "_show_shortcuts", fallback_func("Raccourcis Clavier"))
            )
            help_menu.add_separator()
            help_menu.add_command(
                label="🔧 Diagnostic système",
                command=getattr(self.app, "_run_diagnostic", fallback_func("Diagnostic système"))
            )
            help_menu.add_command(
                label="📞 Support technique",
                command=getattr(self.app, "_open_support", fallback_func("Support technique"))
            )
            help_menu.add_separator()
            help_menu.add_command(
                label="🐛 Signaler un Bug",
                command=getattr(self.app, "_report_bug", fallback_func("Signaler un Bug"))
            )
            help_menu.add_command(
                label="💡 Suggestions",
                command=getattr(self.app, "_suggest_feature", fallback_func("Suggestions"))
            )
            help_menu.add_separator()
            help_menu.add_command(
                label="ℹ️ À propos",
                command=getattr(self.app, "_show_about", fallback_func("À propos"))
            )



        except Exception as e:
            print(f"🔥 DEBUG: Erreur création menu: {e}")
            logging.error(f"MENU: Erreur création menu - {e}")
            raise

        