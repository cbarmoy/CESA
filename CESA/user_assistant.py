"""
CESA (EEG Studio Analysis) v0.0beta1.0 - Assistant Utilisateur
======================================================

Module d'assistance utilisateur pour CESA v0.0beta1.0.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Ce module fournit des assistants interactifs, des tooltips détaillés,
et un système d'aide contextuelle pour faciliter l'utilisation d'CESA.

Fonctionnalités principales:
- Assistant de première utilisation
- Tooltips contextuels détaillés
- Guide interactif des fonctionnalités
- Diagnostic automatique des problèmes
- Suggestions d'utilisation intelligentes

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.0
Date: 2025-01-27
Licence: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Callable
import webbrowser
from pathlib import Path

class UserAssistant:
    """Assistant utilisateur pour CESA v0.0beta1.0."""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.tooltips = {}
        self.first_time_user = True
        self.user_preferences = {
            'show_tooltips': True,
            'show_welcome': True,
            'auto_suggestions': True,
            'help_level': 'beginner'  # beginner, intermediate, advanced
        }
        
        # Charger les préférences utilisateur
        self._load_user_preferences()
    
    def _load_user_preferences(self):
        """Charge les préférences utilisateur depuis un fichier."""
        try:
            prefs_file = Path.home() / '.esa_preferences'
            if prefs_file.exists():
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key in self.user_preferences:
                                if value.lower() in ('true', 'false'):
                                    self.user_preferences[key] = value.lower() == 'true'
                                else:
                                    self.user_preferences[key] = value
        except Exception:
            pass  # Utiliser les valeurs par défaut
    
    def _save_user_preferences(self):
        """Sauvegarde les préférences utilisateur."""
        try:
            prefs_file = Path.home() / '.esa_preferences'
            with open(prefs_file, 'w', encoding='utf-8') as f:
                for key, value in self.user_preferences.items():
                    f.write(f"{key}={value}\n")
        except Exception:
            pass
    
    def show_welcome_assistant(self):
        """Affiche l'assistant de bienvenue détaillé et personnalisé pour les nouveaux utilisateurs."""
        if not self.user_preferences.get('show_welcome', True):
            return
        
        welcome_window = tk.Toplevel(self.parent_app.root)
        welcome_window.title("🎉 Bienvenue dans CESA v0.0beta1.0 - Assistant Personnalisé")
        welcome_window.geometry("900x700")
        welcome_window.transient(self.parent_app.root)
        welcome_window.grab_set()
        
        # Centrer la fenêtre
        welcome_window.update_idletasks()
        x = (welcome_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (welcome_window.winfo_screenheight() // 2) - (700 // 2)
        welcome_window.geometry(f"900x700+{x}+{y}")
        
        # Frame principal avec scrollbar
        main_frame = ttk.Frame(welcome_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Créer un canvas avec scrollbar
        canvas_frame = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_frame.yview)
        scrollable_frame = ttk.Frame(canvas_frame)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )
        
        canvas_frame.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas et scrollbar
        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Titre de bienvenue
        title_label = ttk.Label(scrollable_frame, 
                               text="🎉 Bienvenue dans CESA v0.0beta1.0 !",
                               font=('Segoe UI', 20, 'bold'))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(scrollable_frame,
                                 text="Assistant Personnalisé - Guide Complet des Fonctionnalités",
                                 font=('Segoe UI', 12, 'italic'))
        subtitle_label.pack(pady=(0, 20))
        
        # Informations détaillées sur CESA
        esa_info_frame = ttk.LabelFrame(scrollable_frame, text="🧠 À Propos d'CESA v0.0beta1.0", padding=15)
        esa_info_frame.pack(fill=tk.X, pady=(0, 15))
        
        esa_info_text = """
CESA (EEG Studio Analysis) v0.0beta1.0 est une application professionnelle développée spécifiquement pour l'Unité 
Neuropsychologie du Stress (IRBA). Cette version révolutionnaire intègre des méthodes d'analyse avancées 
et une interface utilisateur moderne pour l'analyse complète des données EEG.

🎯 OBJECTIFS PRINCIPAUX :
• Analyse professionnelle des signaux EEG multi-canal
• Métriques de complexité cérébrale (entropie renormée)
• Détection automatique d'artefacts et de patterns
• Classification des stades de sommeil et de conscience
• Interface intuitive pour chercheurs et cliniciens

🔬 DOMAINES D'APPLICATION :
• Recherche en neurosciences et neuropsychologie
• Clinique du sommeil et des troubles neurologiques
• Études sur les états de conscience et l'anesthésie
• Analyse de la connectivité cérébrale
• Monitoring en temps réel des signaux EEG
        """
        
        esa_info_label = ttk.Label(esa_info_frame, text=esa_info_text.strip(), 
                                  font=('Segoe UI', 10), justify='left')
        esa_info_label.pack(anchor='w')
        
        # Nouvelles fonctionnalités v0.0beta1.0
        features_frame = ttk.LabelFrame(scrollable_frame, text="🚀 Nouvelles Fonctionnalités v0.0beta1.0", padding=15)
        features_frame.pack(fill=tk.X, pady=(0, 15))
        
        features_text = """
🧮 ENTROPIE RENORMÉE (ISSTARTEL) - NOUVEAUTÉ MAJEURE :
   • Méthode révolutionnaire basée sur les travaux de Jean-Pierre Issartel (2007)
   • Quantification de la complexité cérébrale par renormalisation spectrale
   • 4 kernels de renormée : Identity, Powerlaw, Logarithmic, Adaptive
   • Applications : états de conscience, pathologies neurologiques, recherche

🔬 ANALYSES AVANCÉES INTÉGRÉES :
   • Cohérence inter-canal et connectivité fonctionnelle
   • Corrélations temporelles et spatiales
   • Clustering des micro-états EEG
   • Analyse de sources (MNE, sLORETA, dSPM)

⚡ AUTOMATISATION INTELLIGENTE :
   • Détection automatique d'artefacts (EMG, EOG, ECG, mouvement)
   • Classification automatique des stades de sommeil (YASA)
   • Filtrage adaptatif et amplification automatique
   • Synchronisation avec scoring manuel

🎨 INTERFACE MODERNE ET INTUITIVE :
   • Thème sombre/clair adaptatif
   • Tooltips contextuels détaillés
   • Assistant de première utilisation
   • Export multi-format (CSV, PNG, PDF)
        """
        
        features_label = ttk.Label(features_frame, text=features_text.strip(), 
                                 font=('Segoe UI', 10), justify='left')
        features_label.pack(anchor='w')
        
        # Guide d'utilisation personnalisé
        guide_frame = ttk.LabelFrame(scrollable_frame, text="📋 Guide d'Utilisation Personnalisé", padding=15)
        guide_frame.pack(fill=tk.X, pady=(0, 15))
        
        guide_text = """
🎯 ÉTAPES RECOMMANDÉES POUR DÉBUTER :

1️⃣ CHARGEMENT DES DONNÉES :
   • Menu Fichier → Charger EDF → Sélectionnez votre fichier EEG
   • CESA détecte automatiquement le format (EDF/EDF+)
   • Les canaux sont listés et amplifiés automatiquement
   • Conseils : Utilisez des fichiers de qualité, vérifiez la fréquence d'échantillonnage

2️⃣ SÉLECTION ET CONFIGURATION :
   • Menu Affichage → Sélectionner Canaux → Choisissez vos canaux d'intérêt
   • Menu Affichage → Activer Filtre → Configurez les bandes fréquentielles
   • Recommandations : EEG 0.5-30 Hz, EOG 0.1-15 Hz, EMG 10-100 Hz

3️⃣ ANALYSES DISPONIBLES :
   • Menu Analyse → Entropie Renormée (Issartel) → NOUVELLE FONCTIONNALITÉ
   • Menu Analyse → Analyse Spectrale → FFT/Welch pour l'analyse fréquentielle
   • Menu Analyse → Détection d'Artefacts → Nettoyage automatique des signaux
   • Menu Analyse → Statistiques → Métriques de base des signaux

4️⃣ SCORING ET CLASSIFICATION :
   • Menu Scoring → Scoring Automatique → Classification YASA des stades
   • Menu Scoring → Scoring Manuel → Classification personnalisée
   • Import Excel → Chargement de scoring externe

5️⃣ EXPORT ET RAPPORTS :
   • Boutons Export dans chaque analyse → Sauvegarde des résultats
   • Formats supportés : CSV (données), PNG/PDF (graphiques)
   • Génération automatique de rapports complets
        """
        
        guide_label = ttk.Label(guide_frame, text=guide_text.strip(), 
                               font=('Segoe UI', 10), justify='left')
        guide_label.pack(anchor='w')
        
        # Options de démarrage personnalisées
        options_frame = ttk.LabelFrame(scrollable_frame, text="🚀 Comment Souhaitez-Vous Commencer ?", 
                                     padding=15)
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Boutons d'options avec descriptions
        buttons_frame = ttk.Frame(options_frame)
        buttons_frame.pack(fill=tk.X)
        
        def start_guided_tour():
            welcome_window.destroy()
            self.show_guided_tour()
        
        def load_sample_data():
            welcome_window.destroy()
            self._suggest_load_sample_data()
        
        def explore_features():
            welcome_window.destroy()
            self.show_feature_explorer()
        
        def open_reference():
            welcome_window.destroy()
            self._open_reference_guide()
        
        # Boutons avec descriptions
        tour_frame = ttk.Frame(buttons_frame)
        tour_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(tour_frame, text="🎯 Visite Guidée Interactive", 
                  command=start_guided_tour).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(tour_frame, text="Découvrez CESA étape par étape avec un guide interactif", 
                 font=('Segoe UI', 9), foreground='gray').pack(side=tk.LEFT)
        
        sample_frame = ttk.Frame(buttons_frame)
        sample_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(sample_frame, text="📁 Charger Données d'Exemple", 
                  command=load_sample_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(sample_frame, text="Testez CESA avec des données EEG pré-configurées", 
                 font=('Segoe UI', 9), foreground='gray').pack(side=tk.LEFT)
        
        explore_frame = ttk.Frame(buttons_frame)
        explore_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(explore_frame, text="🔍 Explorateur de Fonctionnalités", 
                  command=explore_features).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(explore_frame, text="Explorez toutes les fonctionnalités disponibles", 
                 font=('Segoe UI', 9), foreground='gray').pack(side=tk.LEFT)
        
        reference_frame = ttk.Frame(buttons_frame)
        reference_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(reference_frame, text="📚 Guide de Référence Complet", 
                  command=open_reference).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(reference_frame, text="Accédez à la documentation complète et aux références", 
                 font=('Segoe UI', 9), foreground='gray').pack(side=tk.LEFT)
        
        # Options utilisateur avancées
        prefs_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ Préférences Personnalisées", padding=15)
        prefs_frame.pack(fill=tk.X, pady=(0, 15))
        
        show_welcome_var = tk.BooleanVar(value=self.user_preferences['show_welcome'])
        show_tooltips_var = tk.BooleanVar(value=self.user_preferences['show_tooltips'])
        auto_suggestions_var = tk.BooleanVar(value=self.user_preferences['auto_suggestions'])
        
        ttk.Checkbutton(prefs_frame, text="Afficher cet assistant au démarrage", 
                       variable=show_welcome_var).pack(anchor='w', pady=(0, 5))
        ttk.Checkbutton(prefs_frame, text="Afficher les tooltips d'aide contextuelle", 
                       variable=show_tooltips_var).pack(anchor='w', pady=(0, 5))
        ttk.Checkbutton(prefs_frame, text="Activer les suggestions automatiques", 
                       variable=auto_suggestions_var).pack(anchor='w')
        
        # Informations de support
        support_frame = ttk.LabelFrame(scrollable_frame, text="📞 Support et Ressources", padding=15)
        support_frame.pack(fill=tk.X, pady=(0, 15))
        
        support_text = """
📚 DOCUMENTATION DISPONIBLE :
• README.md : Guide général et installation
• Guide de Référence : Documentation complète des fonctionnalités
• Assistant Intégré : Aide contextuelle dans l'application

🆘 SUPPORT TECHNIQUE :
• Email : come1.barmoy@supbiotech.fr
• Unité Neuropsychologie du Stress (IRBA)
• Diagnostic automatique intégré

🔬 RÉFÉRENCES SCIENTIFIQUES :
• Issartel, J.-P. (2007). Renormalized entropy and complexity measures
• Documentation MNE-Python pour l'analyse EEG
• Standards internationaux EEG (10-20, 10-10)
        """
        
        support_label = ttk.Label(support_frame, text=support_text.strip(), 
                                 font=('Segoe UI', 10), justify='left')
        support_label.pack(anchor='w')
        
        # Boutons de contrôle
        control_frame = ttk.Frame(scrollable_frame)
        control_frame.pack(fill=tk.X, pady=(20, 0))
        
        def close_welcome():
            # Sauvegarder les préférences
            self.user_preferences['show_welcome'] = show_welcome_var.get()
            self.user_preferences['show_tooltips'] = show_tooltips_var.get()
            self.user_preferences['auto_suggestions'] = auto_suggestions_var.get()
            self._save_user_preferences()
            welcome_window.destroy()
        
        ttk.Button(control_frame, text="🚀 Commencer avec CESA", 
                  command=close_welcome).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(control_frame, text="Passer", 
                  command=close_welcome).pack(side=tk.RIGHT)
    
    def show_guided_tour(self):
        """Affiche une visite guidée interactive d'CESA."""
        tour_window = tk.Toplevel(self.parent_app.root)
        tour_window.title("🎯 Visite Guidée CESA v0.0beta1.0")
        tour_window.geometry("700x600")
        tour_window.transient(self.parent_app.root)
        tour_window.grab_set()
        
        # Centrer la fenêtre
        tour_window.update_idletasks()
        x = (tour_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (tour_window.winfo_screenheight() // 2) - (600 // 2)
        tour_window.geometry(f"700x600+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(tour_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="🎯 Visite Guidée CESA v0.0beta1.0", 
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Étapes de la visite
        self.tour_steps = [
            {
                'title': '📁 Charger des données EEG',
                'description': 'Commencez par charger un fichier EDF+ contenant vos données EEG.',
                'action': 'Menu Fichier → Charger EDF+',
                'tip': 'CESA supporte les formats EDF et EDF+. Les fichiers sont automatiquement détectés et amplifiés.'
            },
            {
                'title': '👁️ Sélectionner les canaux',
                'description': 'Choisissez les canaux EEG que vous souhaitez analyser.',
                'action': 'Menu Affichage → Sélectionner Canaux',
                'tip': 'Vous pouvez sélectionner plusieurs canaux pour des analyses multi-canal.'
            },
            {
                'title': '🔧 Configurer les filtres',
                'description': 'Appliquez des filtres pour nettoyer vos signaux.',
                'action': 'Menu Affichage → Activer Filtre',
                'tip': 'Les filtres Butterworth sont recommandés pour l\'EEG (0.5-30 Hz).'
            },
            {
                'title': '📊 Analyser les données',
                'description': 'Explorez les différentes analyses disponibles.',
                'action': 'Menu Analyse → [Fonctionnalité souhaitée]',
                'tip': 'CESA propose de nombreuses analyses : spectrale, entropie renormée, cohérence, etc.'
            },
            {
                'title': '🧮 Entropie Renormée (Nouveau!)',
                'description': 'Découvrez la nouvelle métrique de complexité basée sur Issartel.',
                'action': 'Menu Analyse → Entropie Renormée (Issartel)',
                'tip': 'Cette méthode quantifie la complexité des signaux EEG de manière robuste.'
            },
            {
                'title': '📁 Exporter les résultats',
                'description': 'Sauvegardez vos analyses pour utilisation externe.',
                'action': 'Boutons Export dans chaque analyse',
                'tip': 'Les résultats peuvent être exportés en CSV, PNG, PDF pour publication.'
            }
        ]
        
        self.current_step = 0
        
        # Zone de contenu
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Informations de l'étape
        self.step_frame = ttk.LabelFrame(content_frame, text="Étape actuelle", padding=10)
        self.step_frame.pack(fill=tk.BOTH, expand=True)
        
        self.step_title = ttk.Label(self.step_frame, font=('Segoe UI', 12, 'bold'))
        self.step_title.pack(anchor='w', pady=(0, 10))
        
        self.step_description = ttk.Label(self.step_frame, font=('Segoe UI', 10), 
                                        wraplength=600, justify='left')
        self.step_description.pack(anchor='w', pady=(0, 10))
        
        self.step_action = ttk.Label(self.step_frame, font=('Segoe UI', 10, 'italic'), 
                                    foreground='blue')
        self.step_action.pack(anchor='w', pady=(0, 10))
        
        self.step_tip = ttk.Label(self.step_frame, font=('Segoe UI', 9), 
                                 foreground='green', wraplength=600, justify='left')
        self.step_tip.pack(anchor='w')
        
        # Barre de progression
        progress_frame = ttk.Frame(content_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=len(self.tour_steps))
        self.progress_bar.pack(fill=tk.X)
        
        self.progress_label = ttk.Label(progress_frame, text="Étape 1 sur 6")
        self.progress_label.pack(pady=(5, 0))
        
        # Boutons de navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X)
        
        def prev_step():
            if self.current_step > 0:
                self.current_step -= 1
                self._update_tour_step()
        
        def next_step():
            if self.current_step < len(self.tour_steps) - 1:
                self.current_step += 1
                self._update_tour_step()
            else:
                tour_window.destroy()
                messagebox.showinfo("Visite terminée", 
                                  "🎉 Félicitations ! Vous avez terminé la visite guidée.\n\n"
                                  "Vous êtes maintenant prêt à utiliser CESA v0.0beta1.0 !")
        
        def skip_tour():
            tour_window.destroy()
        
        ttk.Button(nav_frame, text="← Précédent", 
                  command=prev_step).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Suivant →", 
                  command=next_step).pack(side=tk.RIGHT)
        ttk.Button(nav_frame, text="Passer la visite", 
                  command=skip_tour).pack(side=tk.RIGHT, padx=(0, 10))
        
        # Initialiser la première étape
        self._update_tour_step()
    
    def _update_tour_step(self):
        """Met à jour l'affichage de l'étape actuelle."""
        if 0 <= self.current_step < len(self.tour_steps):
            step = self.tour_steps[self.current_step]
            
            self.step_title.config(text=step['title'])
            self.step_description.config(text=step['description'])
            self.step_action.config(text=f"💡 Action : {step['action']}")
            self.step_tip.config(text=f"💡 Conseil : {step['tip']}")
            
            # Mettre à jour la barre de progression
            self.progress_var.set(self.current_step + 1)
            self.progress_label.config(text=f"Étape {self.current_step + 1} sur {len(self.tour_steps)}")
    
    def show_feature_explorer(self):
        """Affiche un explorateur interactif approfondi des fonctionnalités."""
        explorer_window = tk.Toplevel(self.parent_app.root)
        explorer_window.title("🔍 Explorateur de Fonctionnalités CESA v0.0beta1.0 - Guide Complet")
        explorer_window.geometry("1200x900")
        explorer_window.transient(self.parent_app.root)
        explorer_window.grab_set()
        
        # Centrer la fenêtre
        explorer_window.update_idletasks()
        x = (explorer_window.winfo_screenwidth() // 2) - (1200 // 2)
        y = (explorer_window.winfo_screenheight() // 2) - (900 // 2)
        explorer_window.geometry(f"1200x900+{x}+{y}")
        
        # Frame principal avec scrollbar
        main_frame = ttk.Frame(explorer_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Créer un canvas avec scrollbar
        canvas_frame = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_frame.yview)
        scrollable_frame = ttk.Frame(canvas_frame)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )
        
        canvas_frame.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        
        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Titre principal
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="🔍 Explorateur de Fonctionnalités CESA v0.0beta1.0", 
                               font=('Segoe UI', 20, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, 
                                  text="Guide complet et personnalisé pour chaque fonctionnalité", 
                                  font=('Segoe UI', 12, 'italic'))
        subtitle_label.pack(pady=(5, 0))
        
        # Frame avec arbre et détails
        content_frame = ttk.Frame(scrollable_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Arbre des fonctionnalités (plus large)
        tree_frame = ttk.LabelFrame(content_frame, text="📋 Fonctionnalités Disponibles", padding=15)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        self.features_tree = ttk.Treeview(tree_frame, show='tree')
        self.features_tree.pack(fill=tk.BOTH, expand=True)
        
        # Détails de la fonctionnalité (plus large)
        details_frame = ttk.LabelFrame(content_frame, text="📖 Détails et Documentation", padding=15)
        details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Retour au widget Text avec scrollbar
        text_frame = ttk.Frame(details_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # Garder une référence au parent pour pouvoir recréer le widget
        self._text_frame = text_frame

        self.feature_details = tk.Text(text_frame, wrap=tk.WORD, height=25,
                                      font=('Segoe UI', 11), state='disabled')

        # Scrollbar pour le texte
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.feature_details.yview)
        self.feature_details.configure(yscrollcommand=scrollbar.set)

        self.feature_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Insérer le texte initial
        self.feature_details.config(state='normal')
        self.feature_details.insert(1.0, "Sélectionnez une fonctionnalité pour voir ses détails")
        self.feature_details.config(state='disabled')
        
        # Charger les fonctionnalités approfondies
        self._load_detailed_features()
        
        # Lier seulement le clic pour éviter les appels multiples
        self.features_tree.bind('<Button-1>', self._on_feature_click)
        
        # Boutons de contrôle améliorés
        control_frame = ttk.Frame(scrollable_frame)
        control_frame.pack(fill=tk.X, pady=(20, 0))
        
        def try_feature():
            selection = self.features_tree.selection()
            if selection:
                feature_id = self.features_tree.item(selection[0])['text']
                self._try_feature(feature_id)
        
        def show_tutorial():
            selection = self.features_tree.selection()
            if selection:
                feature_id = self.features_tree.item(selection[0])['text']
                self._show_feature_tutorial(feature_id)
        
        ttk.Button(control_frame, text="🚀 Essayer cette fonctionnalité", 
                  command=try_feature).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="📚 Tutoriel détaillé", 
                  command=show_tutorial).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="📚 Guide de référence complet", 
                  command=self._open_reference_guide).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Fermer", 
                  command=explorer_window.destroy).pack(side=tk.RIGHT)
    
    def _load_detailed_features(self):
        """Charge la liste détaillée des fonctionnalités avec descriptions approfondies."""
        features = {
            '📁 Gestion des Fichiers': {
                '📂 Charger fichier EDF': 'Importation et validation des fichiers EEG',
                '📤 Exporter données': 'Export multi-format (CSV, Excel, JSON)',
                '📋 Générer rapport': 'Rapport automatique complet des analyses',
                '💾 Sauvegarder configuration': 'Sauvegarde des paramètres personnalisés'
            },
            '👁️ Visualisation et Affichage': {
                '🎛️ Sélectionner Canaux': 'Interface avancée de sélection des canaux EEG',
                '📏 Autoscale Intelligent': 'Ajustement automatique optimisé des échelles',
                '🔧 Filtres Avancés': 'Système de filtrage multi-bandes et adaptatif',
                '🌙 Thème Sombre/Clair': 'Interface adaptative avec thèmes personnalisés',
                '📊 Zoom et Navigation': 'Navigation temporelle précise avec raccourcis',
                '🎨 Personnalisation Graphique': 'Couleurs, styles et annotations personnalisées'
            },
            '📊 Analyses Avancées': {
                '📈 Statistiques Descriptives': 'Métriques complètes des signaux EEG',
                '🌊 Analyse Spectrale (FFT/Welch)': 'Analyse fréquentielle haute résolution',
                '😴 PSD par Stades de Sommeil': 'Analyse spectrale contextuelle par stades',
                '🧮 Entropie Renormée (Issartel)': 'Métrique de complexité révolutionnaire',
                '⏱️ Analyse Temporelle': 'Caractéristiques temporelles et variabilité',
                '🔗 Cohérence Inter-Canal': 'Connectivité fonctionnelle entre régions',
                '📊 Corrélations Temporelles': 'Relations temporelles entre canaux',
                '🧠 Micro-états EEG': 'Clustering et topographies des micro-états',
                '⚠️ Détection Artefacts': 'Détection automatique et correction',
                '🎯 Localisation Sources': 'Reconstruction des sources cérébrales'
            },
            '😴 Classification du Sommeil': {
                '🤖 Scoring Automatique (YASA)': 'Classification automatique des stades',
                '✋ Scoring Manuel Interactif': 'Interface de classification manuelle',
                '⚖️ Comparaison Scoring': 'Validation automatique vs manuel',
                '📊 Métriques de Performance': 'Précision, rappel, F1-score par stade',
                '📈 Visualisation Hypnogramme': 'Affichage temporel des stades'
            },
            '🔬 Outils Scientifiques': {
                '📚 Documentation Intégrée': 'Guide complet avec références scientifiques',
                '🧪 Assistant de Première Utilisation': 'Tutoriel interactif personnalisé',
                '🔍 Explorateur de Fonctionnalités': 'Guide détaillé de chaque fonctionnalité',
                '🔧 Diagnostic Système': 'Vérification automatique de l\'installation',
                '📞 Support Technique': 'Aide et contact avec l\'équipe de développement'
            }
        }
        
        for category, functions in features.items():
            category_id = self.features_tree.insert('', 'end', text=category, open=True)
            for function, description in functions.items():
                self.features_tree.insert(category_id, 'end', text=function)
    
    def _on_feature_click(self, event):
        """Gère le clic sur une fonctionnalité pour forcer la mise à jour."""
        # Éviter les appels multiples avec un flag simple
        if hasattr(self, '_processing_click') and self._processing_click:
            return
        
        self._processing_click = True
        
        # Sélectionner l'élément cliqué
        item = self.features_tree.identify('item', event.x, event.y)
        if item:
            self.features_tree.selection_set(item)
            feature_name = self.features_tree.item(item)['text']
            self._show_detailed_feature_info(feature_name)
        
        # Réinitialiser le flag après un court délai
        self.parent_app.root.after(100, lambda: setattr(self, '_processing_click', False))
    
    def _show_detailed_feature_info(self, feature_name):
        """Affiche les détails approfondis d'une fonctionnalité avec références scientifiques."""
        # Détruire le widget Text actuel et sa scrollbar
        self.feature_details.destroy()
        for child in self._text_frame.winfo_children():
            child.destroy()

        # Créer un nouveau widget Text
        self.feature_details = tk.Text(self._text_frame, wrap=tk.WORD, height=25,
                                      font=('Segoe UI', 11), state='disabled')

        # Recréer la scrollbar
        scrollbar = ttk.Scrollbar(self._text_frame, orient="vertical", command=self.feature_details.yview)
        self.feature_details.configure(yscrollcommand=scrollbar.set)

        # Repacker les widgets
        self.feature_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Dictionnaire complet avec toutes les références scientifiques
        detailed_info = {
            # Descriptions des catégories
            '📁 Gestion des Fichiers': """
🎯 CATÉGORIE : GESTION DES FICHIERS
═══════════════════════════════════════════════════════════════════════════════

📋 DESCRIPTION DE LA CATÉGORIE
Cette catégorie regroupe toutes les fonctionnalités liées à la gestion des fichiers EEG, de l'importation à l'exportation des données et résultats d'analyse.

📂 FONCTIONNALITÉS INCLUSES
• 📂 Charger fichier EDF : Importation et validation des fichiers EEG
• 📤 Exporter données : Export multi-format (CSV, Excel, JSON, TXT, EDF)
• 📋 Générer rapport : Rapports automatiques complets des analyses
• 💾 Sauvegarder configuration : Sauvegarde des paramètres personnalisés

🔬 RÉFÉRENCES TECHNIQUES
• Format EDF/EDF+ : Kemp, B. et al. (2003) - Standard européen pour l'échange de données physiologiques
• MNE-Python : Plateforme complète d'analyse EEG/MEG
• Pandas : Manipulation et analyse de données tabulaires
• OpenPyXL : Bibliothèque Python pour les fichiers Excel

💡 APPLICATIONS CLINIQUES
• Gestion de bases de données EEG multi-patients
• Partage de données entre équipes de recherche
• Archivage réglementaire des enregistrements
• Intégration dans des workflows cliniques existants

🚀 COMMENT UTILISER
Sélectionnez une fonctionnalité spécifique dans cette catégorie pour accéder à ses détails complets.
            """,

            '👁️ Visualisation et Affichage': """
🎯 CATÉGORIE : VISUALISATION ET AFFICHAGE
═══════════════════════════════════════════════════════════════════════════════

📋 DESCRIPTION DE LA CATÉGORIE
Cette catégorie regroupe toutes les fonctionnalités liées à la visualisation des signaux EEG, de la sélection des canaux à la personnalisation graphique.

🎛️ FONCTIONNALITÉS INCLUSES
• 🎛️ Sélectionner Canaux : Interface avancée de sélection des canaux EEG
• 📏 Autoscale Intelligent : Ajustement automatique optimisé des échelles
• 🔧 Filtres Avancés : Système de filtrage multi-bandes et adaptatif
• 🌙 Thème Sombre/Clair : Interface adaptative avec thèmes personnalisés
• 📊 Zoom et Navigation : Navigation temporelle précise avec raccourcis
• 🎨 Personnalisation Graphique : Couleurs, styles et annotations personnalisées

🔬 RÉFÉRENCES TECHNIQUES
• Matplotlib : Bibliothèque de visualisation 2D pour Python
• Tkinter : Interface graphique standard de Python
• Standards EEG : American Clinical Neurophysiology Society guidelines
• Ergonomie interface : Nielsen (1994) - Principes d'utilisabilité

💡 APPLICATIONS CLINIQUES
• Examen visuel rapide des enregistrements EEG
• Détection d'anomalies et d'artefacts
• Présentation des résultats aux équipes médicales
• Formation et enseignement de l'analyse EEG

🚀 COMMENT UTILISER
Sélectionnez une fonctionnalité spécifique dans cette catégorie pour accéder à ses détails complets.
            """,

            '📊 Analyses Avancées': """
🎯 CATÉGORIE : ANALYSES AVANCÉES
═══════════════════════════════════════════════════════════════════════════════

📋 DESCRIPTION DE LA CATÉGORIE
Cette catégorie regroupe toutes les méthodes d'analyse avancée des signaux EEG, des statistiques descriptives aux analyses de connectivité.

🔬 FONCTIONNALITÉS INCLUSES
• 📈 Statistiques Descriptives : Métriques complètes des signaux EEG
• 🌊 Analyse Spectrale (FFT/Welch) : Analyse fréquentielle haute résolution
• 😴 PSD par Stades de Sommeil : Analyse spectrale contextuelle par stades
• 🧮 Entropie Renormée (Issartel) : Métrique de complexité révolutionnaire
• ⏱️ Analyse Temporelle : Caractéristiques temporelles et variabilité
• 🔗 Cohérence Inter-Canal : Connectivité fonctionnelle entre régions
• 📊 Corrélations Temporelles : Relations temporelles entre canaux
• 🧠 Micro-états EEG : Clustering et topographies des micro-états
• ⚠️ Détection Artefacts : Détection automatique et correction
• 🎯 Localisation Sources : Reconstruction des sources cérébrales

🔬 RÉFÉRENCES SCIENTIFIQUES
• Analyse spectrale EEG : Klimesch (1999) - Oscillations alpha et thêta
• Entropie renormalisée : Issartel (2007) - Mesures de complexité EEG
• Cohérence EEG : Nunez et al. (1997) - Synchronisation neuronale
• Micro-états EEG : Lehmann et al. (1987) - États fonctionnels cérébraux

💡 APPLICATIONS CLINIQUES
• Diagnostic de pathologies neurologiques
• Monitoring per-opératoire et anesthésique
• Recherche en neurosciences cognitives
• Évaluation de l'efficacité thérapeutique

🚀 COMMENT UTILISER
Sélectionnez une fonctionnalité spécifique dans cette catégorie pour accéder à ses détails complets.
            """,

            '😴 Classification du Sommeil': """
🎯 CATÉGORIE : CLASSIFICATION DU SOMMEIL
═══════════════════════════════════════════════════════════════════════════════

📋 DESCRIPTION DE LA CATÉGORIE
Cette catégorie regroupe toutes les fonctionnalités liées à l'analyse et la classification des stades de sommeil à partir des signaux EEG.

😴 FONCTIONNALITÉS INCLUSES
• 🤖 Scoring Automatique (YASA) : Classification automatique des stades
• ✋ Scoring Manuel Interactif : Interface de classification manuelle
• ⚖️ Comparaison Scoring : Validation automatique vs manuel
• 📊 Métriques de Performance : Précision, rappel, F1-score par stade
• 📈 Visualisation Hypnogramme : Affichage temporel des stades

🔬 RÉFÉRENCES SCIENTIFIQUES
• Standards AASM : Berry et al. (2017) - Règles de classification du sommeil
• YASA : Vallat & Walker (2019) - Outil automatique haute performance
• Hypnogramme : Rechtschaffen & Kales (1968) - Représentation temporelle
• Validation : Silber et al. (2007) - Méthodes de validation

💡 APPLICATIONS CLINIQUES
• Diagnostic des troubles du sommeil
• Évaluation de l'architecture du sommeil
• Suivi de l'efficacité des traitements
• Recherche en somnologie

🚀 COMMENT UTILISER
Sélectionnez une fonctionnalité spécifique dans cette catégorie pour accéder à ses détails complets.
            """,

            '🔬 Outils Scientifiques': """
🎯 CATÉGORIE : OUTILS SCIENTIFIQUES
═══════════════════════════════════════════════════════════════════════════════

📋 DESCRIPTION DE LA CATÉGORIE
Cette catégorie regroupe tous les outils d'aide à l'utilisation, de documentation et de support technique pour CESA.

🛠️ FONCTIONNALITÉS INCLUSES
• 📚 Documentation Intégrée : Guide complet avec références scientifiques
• 🧪 Assistant de Première Utilisation : Tutoriel interactif personnalisé
• 🔍 Explorateur de Fonctionnalités : Guide détaillé de chaque fonctionnalité
• 🔧 Diagnostic Système : Vérification automatique de l'installation
• 📞 Support Technique : Aide et contact avec l'équipe de développement

🔬 RÉFÉRENCES TECHNIQUES
• Design d'interface : Nielsen (1994) - Principes d'utilisabilité
• Onboarding utilisateur : Krug (2014) - Guide de l'utilisateur
• Documentation : Tufte (2001) - Principes de visualisation
• Support technique : Best practices ITIL

💡 AVANTAGES PÉDAGOGIQUES
• Courbe d'apprentissage réduite
• Formation personnalisée selon le niveau
• Références scientifiques intégrées
• Support technique réactif

🚀 COMMENT UTILISER
Sélectionnez une fonctionnalité spécifique dans cette catégorie pour accéder à ses détails complets.
            """,

            # Descriptions des fonctionnalités individuelles
            '📂 Charger fichier EDF': """
🎯 DESCRIPTION DÉTAILLÉE
Importation et validation complète des fichiers EEG au format EDF/EDF+ avec détection automatique des canaux et application des filtres optimaux.

📋 FONCTIONNALITÉS AVANCÉES
• Validation automatique du format EDF/EDF+
• Détection intelligente des canaux EEG, EOG, EMG, ECG
• Application automatique des filtres par défaut (0.3-35 Hz pour EEG)
• Création automatique des dérivations bipolaires
• Détection des annotations et segments d'artefacts
• Support des fichiers multi-canaux (jusqu'à 256 canaux)
• Gestion des métadonnées temporelles et de calibration

🔬 RÉFÉRENCES SCIENTIFIQUES COMPLÈTES
• Format EDF/EDF+ : Kemp, B. et al. "European data format 'plus' (EDF+), an EDF alike standard format for the exchange of physiological data." Clinical Neurophysiology 114.9 (2003): 1755-1761.
• Validation EDF : MNE-Python documentation - https://mne.tools/stable/
• Standards EEG : American Clinical Neurophysiology Society guidelines
• MNE-Python : https://github.com/mne-tools/mne-python
• SleepEEGpy : https://github.com/NirLab-TAU/sleepeegpy

💡 CONSEILS D'UTILISATION
1. Vérifiez que le fichier EDF est valide avant l'importation
2. Les canaux EEG sont automatiquement détectés et filtrés
3. Les dérivations bipolaires sont créées selon les standards internationaux
4. Utilisez les données d'exemple pour tester les fonctionnalités

🚀 COMMENT ESSAYER
Cliquez sur "Fichier" → "Charger fichier EDF" ou utilisez Ctrl+O
            """,
            
            '🧮 Entropie Renormée (Issartel)': """
🎯 DESCRIPTION DÉTAILLÉE
Métrique révolutionnaire de complexité basée sur la méthode d'Issartel (2007) pour quantifier la complexité des signaux EEG multi-canaux.

📋 FONCTIONNALITÉS AVANCÉES
• Calcul des moments généralisés d'ordre configurable
• Construction de la matrice de covariance des moments
• Renormalisation spectrale avec 4 kernels disponibles :
  - Identity: ψ(λ) = λ
  - Powerlaw: ψ(λ) = λ^γ (avec γ paramètre)
  - Logarithmic: ψ(λ) = ln(1 + λ/ε)
  - Adaptive: ψ(λ) = λ / median(λ)
• Calcul de l'entropie différentielle en nats et bits
• Fenêtrage glissant avec chevauchement configurable
• Visualisation temporelle de l'évolution de l'entropie

🔬 RÉFÉRENCES SCIENTIFIQUES COMPLÈTES
• Issartel, J.-P. (2007). "Renormalized entropy and complexity measures of electroencephalographic time series." Proceedings of the Royal Society A, 463(2087), 2647-2661. DOI: 10.1098/rspa.2007.1877
• Issartel, J.-P. (2011). "Renormalized entropy and complexity measures of electroencephalographic time series: Application to sleep analysis." Pure and Applied Geophysics, 168(12), 2209-2223. DOI: 10.1007/s00024-011-0381-4
• Méthode Welch : Welch, P.D. "The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms." IEEE Transactions on Audio and Electroacoustics 15.2 (1967): 70-73.
• Décomposition SVD : Golub, G.H. & Van Loan, C.F. "Matrix Computations" (4th ed.). Johns Hopkins University Press, 2013.

💡 APPLICATIONS CLINIQUES
• Analyse de l'état de conscience (éveil vs sommeil)
• Monitoring anesthésique en temps réel
• Détection de pathologies neurologiques
• Étude des micro-états EEG et transitions
• Recherche en neurosciences cognitives

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Entropie Renormée (Issartel)" ou utilisez le bouton dédié
            """,
            
            '🤖 Scoring Automatique (YASA)': """
🎯 DESCRIPTION DÉTAILLÉE
Classification automatique des stades de sommeil utilisant l'algorithme YASA (Yet Another Sleep Algorithm) basé sur l'apprentissage automatique.

📋 FONCTIONNALITÉS AVANCÉES
• Classification automatique en 5 stades : Éveil, N1, N2, N3, REM
• Utilisation de caractéristiques spectrales et temporelles
• Algorithme optimisé pour les enregistrements polysomnographiques
• Validation croisée et métriques de performance
• Export des résultats en format standard
• Comparaison avec scoring manuel

🔬 RÉFÉRENCES SCIENTIFIQUES COMPLÈTES
• YASA : Vallat, R. & Walker, M.P. "An open-source, high-performance tool for automated sleep staging." eLife 8 (2019): e70092. DOI: 10.7554/eLife.70092
• SleepEEGpy : Falach, R. et al. "SleepEEGpy: a Python-based software integration package to organize preprocessing, analysis, and visualization of sleep EEG data." Computers in Biology and Medicine 192 (2025): 110232. DOI: 10.1016/j.compbiomed.2025.110232
• Validation YASA : https://elifesciences.org/articles/70092
• Documentation YASA : https://yasa-sleep.org/generated/yasa.SleepStaging.html
• FAQ YASA : https://yasa-sleep.org/faq.html
• SleepChecker : https://github.com/nabilalibou/SleepChecker
• Preprint référencé : https://doi.org/10.1101/2023.12.17.572046

💡 PERFORMANCES ATTENDUES
• Précision globale : 80-85% sur données polysomnographiques
• Kappa de Cohen : 0.75-0.85
• Meilleures performances sur N2 et REM
• Sensible à la qualité des données et montage

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Scoring Automatique" après avoir chargé un fichier EDF
            """,
            
            '🌊 Analyse Spectrale (FFT/Welch)': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse fréquentielle haute résolution utilisant les méthodes FFT et Welch pour caractériser le contenu spectral des signaux EEG.

📋 FONCTIONNALITÉS AVANCÉES
• Méthode Welch avec fenêtre de Hann et 50% de chevauchement
• Calcul de la densité spectrale de puissance (PSD)
• Bands de fréquence standard : Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
• Détection des pics spectraux et centroïde spectral
• Visualisation interactive des spectres
• Export des résultats en format numérique

🔬 RÉFÉRENCES SCIENTIFIQUES COMPLÈTES
• Méthode Welch : Welch, P.D. "The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms." IEEE Transactions on Audio and Electroacoustics 15.2 (1967): 70-73.
• SciPy Signal Processing : https://scipy.org/
• Analyse spectrale EEG : Klimesch, W. "EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis." Brain Research Reviews 29.2-3 (1999): 169-195.
• Bands de fréquence EEG : Niedermeyer, E. & da Silva, F.L. "Electroencephalography: basic principles, clinical applications, and related fields." Lippincott Williams & Wilkins, 2005.
• Source interne : Analyse_spectrale.py (adaptée dans spectral_analysis.py)

💡 INTERPRÉTATION CLINIQUE
• Delta : Sommeil profond, pathologie neurologique
• Theta : Sommeil léger, méditation, créativité
• Alpha : Éveil détendu, yeux fermés
• Beta : Éveil actif, attention, anxiété
• Gamma : Processus cognitifs complexes

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Analyse Spectrale" après sélection des canaux
            """,
            
            '📈 Statistiques Descriptives': """
🎯 DESCRIPTION DÉTAILLÉE
Calcul complet des métriques statistiques descriptives pour caractériser les propriétés des signaux EEG.

📋 MÉTRIQUES CALCULÉES
• Statistiques de base : moyenne, médiane, écart-type, variance
• Métriques de forme : skewness, kurtosis
• Amplitude : min, max, amplitude crête-à-crête
• Variabilité : coefficient de variation, RMS
• Distribution : percentiles, quartiles
• Qualité du signal : rapport signal/bruit, artefacts

🔬 RÉFÉRENCES SCIENTIFIQUES
• Statistiques EEG : Nunez, P.L. & Srinivasan, R. "Electric fields of the brain: the neurophysics of EEG." Oxford University Press, 2006.
• Métriques de qualité : Delorme, A. & Makeig, S. "EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics." Journal of Neuroscience Methods 134.1 (2004): 9-21.
• Pandas : https://pandas.pydata.org/
• NumPy : https://numpy.org/
• scikit-learn : https://scikit-learn.org/stable/

💡 UTILISATION CLINIQUE
• Validation de la qualité des enregistrements
• Détection d'artefacts et de dérives
• Comparaison entre conditions expérimentales
• Contrôle qualité des analyses

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Statistiques" pour obtenir un rapport complet
            """,
            
            '🔧 Diagnostic Système': """
🎯 DESCRIPTION DÉTAILLÉE
Vérification automatique complète de l'installation et des dépendances pour assurer le bon fonctionnement d'CESA.

📋 VÉRIFICATIONS EFFECTUÉES
• Version de Python et compatibilité
• Installation des packages requis : numpy, pandas, matplotlib, scipy, mne, yasa
• Vérification des modules Excel : xlrd, openpyxl
• Test des fonctionnalités graphiques (Tkinter)
• Validation des chemins et permissions
• Test de performance et mémoire disponible

🔬 RÉFÉRENCES TECHNIQUES
• Python : https://www.python.org/
• MNE-Python : https://mne.tools/stable/
• YASA : https://yasa-sleep.org/
• SciPy : https://scipy.org/
• Matplotlib : https://matplotlib.org/
• Pandas : https://pandas.pydata.org/

💡 RÉSOLUTION DE PROBLÈMES
• Installation automatique des dépendances manquantes
• Suggestions de mise à jour des packages
• Diagnostic des conflits de versions
• Guide de résolution des erreurs courantes

🚀 COMMENT ESSAYER
Cliquez sur "Aide" → "Diagnostic système" pour vérifier votre installation
            """,
            
            '📤 Exporter données': """
🎯 DESCRIPTION DÉTAILLÉE
Export multi-format des données EEG analysées et des résultats d'analyse pour utilisation externe.

📋 FORMATS D'EXPORT DISPONIBLES
• CSV : Données tabulaires avec séparateur virgule
• Excel : Feuilles de calcul avec formatage avancé
• JSON : Données structurées pour intégration web
• TXT : Format texte simple pour compatibilité
• EDF : Réexport au format EDF standard

🔬 RÉFÉRENCES TECHNIQUES
• Pandas : https://pandas.pydata.org/
• OpenPyXL : https://openpyxl.readthedocs.io/
• NumPy : https://numpy.org/
• MNE-Python : https://mne.tools/stable/

💡 UTILISATION CLINIQUE
• Partage de données entre équipes
• Intégration dans d'autres logiciels d'analyse
• Archivage des résultats d'analyse
• Génération de rapports personnalisés

🚀 COMMENT ESSAYER
Cliquez sur "Fichier" → "Exporter données" après avoir effectué des analyses
            """,
            
            '💾 Sauvegarder configuration': """
🎯 DESCRIPTION DÉTAILLÉE
Sauvegarde des paramètres personnalisés et configurations utilisateur pour une utilisation optimale d'CESA.

📋 CONFIGURATIONS SAUVEGARDÉES
• Paramètres d'affichage (zoom, canaux, couleurs)
• Configurations d'analyse (filtres, fenêtres, kernels)
• Préférences utilisateur (thèmes, raccourcis)
• Paramètres de scoring automatique
• Configurations d'export et rapports

🔬 RÉFÉRENCES TECHNIQUES
• JSON : https://www.json.org/
• Pickle : https://docs.python.org/3/library/pickle.html
• Configuration management : Python best practices
• User preferences : Nielsen (1994)

💡 AVANTAGES DE LA SAUVEGARDE
• Persistance des paramètres entre sessions
• Configuration rapide pour différents protocoles
• Partage de configurations entre utilisateurs
• Restauration en cas de problème

🚀 COMMENT ESSAYER
Cliquez sur "Fichier" → "Sauvegarder configuration" après avoir configuré vos préférences
            """,
            
            '📋 Générer rapport': """
🎯 DESCRIPTION DÉTAILLÉE
Génération automatique de rapports complets incluant toutes les analyses effectuées et leurs résultats.

📋 CONTENU DU RAPPORT
• Résumé des données chargées (canaux, durée, fréquence)
• Résultats des analyses spectrales
• Métriques d'entropie renormée
• Classification des stades de sommeil
• Statistiques descriptives
• Graphiques et visualisations
• Recommandations cliniques

🔬 RÉFÉRENCES TECHNIQUES
• Matplotlib : https://matplotlib.org/
• ReportLab : https://www.reportlab.com/
• Pandas : https://pandas.pydata.org/
• Jinja2 : https://jinja.palletsprojects.com/

💡 APPLICATIONS PROFESSIONNELLES
• Rapports cliniques standardisés
• Documentation de recherche
• Présentation des résultats
• Archivage des analyses

🚀 COMMENT ESSAYER
Cliquez sur "Fichier" → "Générer rapport" après avoir complété vos analyses
            """,
            
            '🎛️ Sélectionner Canaux': """
🎯 DESCRIPTION DÉTAILLÉE
Interface avancée de sélection et configuration des canaux EEG pour l'analyse et la visualisation.

📋 FONCTIONNALITÉS AVANCÉES
• Sélection multiple avec prévisualisation
• Détection automatique des types de canaux (EEG, EOG, EMG, ECG)
• Filtrage par type de signal
• Configuration des dérivations bipolaires
• Sauvegarde des configurations personnalisées
• Validation de la qualité des canaux

🔬 RÉFÉRENCES SCIENTIFIQUES
• Standards EEG : American Clinical Neurophysiology Society guidelines
• Montages EEG : 10-20 system international standards
• MNE-Python : https://mne.tools/stable/
• Dérivations bipolaires : Niedermeyer & da Silva (2005)

💡 CONSEILS D'UTILISATION
• Sélectionnez 2-8 canaux EEG de bonne qualité
• Évitez les canaux avec beaucoup d'artefacts
• Utilisez les dérivations recommandées (C3-M2, C4-M1)
• Vérifiez la cohérence des amplitudes

🚀 COMMENT ESSAYER
Cliquez sur "Affichage" → "Sélectionner Canaux" ou utilisez Ctrl+1
            """,
            
            '📏 Autoscale Intelligent': """
🎯 DESCRIPTION DÉTAILLÉE
Système d'ajustement automatique optimisé des échelles d'affichage pour une visualisation optimale des signaux EEG.

📋 ALGORITHMES D'AJUSTEMENT
• Calcul automatique des amplitudes min/max
• Détection des artefacts et exclusion du calcul
• Ajustement adaptatif par canal
• Préservation des proportions relatives
• Optimisation pour différents types de signaux

🔬 RÉFÉRENCES TECHNIQUES
• NumPy : https://numpy.org/
• SciPy : https://scipy.org/
• Matplotlib : https://matplotlib.org/
• Détection d'artefacts : Delorme & Makeig (2004)

💡 AVANTAGES CLINIQUES
• Visualisation optimale sans manipulation manuelle
• Détection automatique des anomalies
• Économie de temps lors de l'examen
• Standardisation de l'affichage

🚀 COMMENT ESSAYER
Activez l'autoscale dans le menu "Affichage" ou utilisez le raccourci dédié
            """,
            
            '🔧 Filtres Avancés': """
🎯 DESCRIPTION DÉTAILLÉE
Système de filtrage multi-bandes et adaptatif pour le prétraitement optimal des signaux EEG.

📋 TYPES DE FILTRES DISPONIBLES
• Filtre passe-bas : Élimination des hautes fréquences
• Filtre passe-haut : Suppression des basses fréquences
• Filtre passe-bande : Sélection de bandes spécifiques
• Filtre notch : Élimination des interférences (50/60 Hz)
• Filtre adaptatif : Ajustement automatique selon le signal

🔬 RÉFÉRENCES SCIENTIFIQUES
• Filtrage EEG : Niedermeyer & da Silva (2005)
• SciPy Signal Processing : https://scipy.org/
• MNE-Python : https://mne.tools/stable/
• Méthodes de filtrage : Oppenheim & Schafer (2010)

💡 PARAMÈTRES RECOMMANDÉS
• EEG : 0.3-35 Hz (bande passante standard)
• EOG : 0.3-15 Hz (réduction des artefacts)
• EMG : 10-100 Hz (activité musculaire)
• ECG : 0.3-70 Hz (signal cardiaque)

🚀 COMMENT ESSAYER
Cliquez sur "Affichage" → "Activer Filtre" ou utilisez Ctrl+F
            """,
            
            '🌙 Thème Sombre/Clair': """
🎯 DESCRIPTION DÉTAILLÉE
Interface adaptative avec thèmes personnalisés pour optimiser l'expérience utilisateur selon les préférences et conditions d'utilisation.

📋 THÈMES DISPONIBLES
• Thème Clair : Interface traditionnelle avec fond blanc
• Thème Sombre : Interface moderne avec fond sombre
• Thème Automatique : Adaptation selon l'heure du jour
• Thème Personnalisé : Configuration des couleurs utilisateur

🔬 RÉFÉRENCES TECHNIQUES
• Tkinter : https://docs.python.org/3/library/tkinter.html
• ttk : https://docs.python.org/3/library/tkinter.ttk.html
• Ergonomie logicielle : Nielsen (1994)

💡 AVANTAGES ERGONOMIQUES
• Réduction de la fatigue oculaire
• Adaptation aux conditions d'éclairage
• Personnalisation selon les préférences
• Amélioration de la productivité

🚀 COMMENT ESSAYER
Utilisez le raccourci C ou le menu "Affichage" → "Thème"
            """,
            
            '📊 Zoom et Navigation': """
🎯 DESCRIPTION DÉTAILLÉE
Navigation temporelle précise avec raccourcis clavier pour une exploration efficace des données EEG.

📋 FONCTIONNALITÉS DE NAVIGATION
• Zoom avant/arrière avec molette de souris
• Navigation par époques (Z/Q/S/D)
• Déplacement temporel précis
• Marquage de positions d'intérêt
• Synchronisation multi-canaux

🔬 RACCOURCIS CLAVIER
• Z/Q : Époques précédentes
• S/D : Époques suivantes
• Molette : Zoom avant/arrière
• Flèches : Déplacement fin
• Espace : Pause/reprise

💡 CONSEILS D'UTILISATION
• Utilisez les raccourcis pour une navigation rapide
• Marquez les périodes d'intérêt
• Synchronisez l'affichage des canaux
• Optimisez le zoom selon l'analyse

🚀 COMMENT ESSAYER
Utilisez les raccourcis clavier Z/Q/S/D ou la molette de souris
            """,
            
            '🎨 Personnalisation Graphique': """
🎯 DESCRIPTION DÉTAILLÉE
Configuration des couleurs, styles et annotations personnalisées pour adapter l'affichage aux besoins spécifiques.

📋 OPTIONS DE PERSONNALISATION
• Couleurs des canaux EEG personnalisables
• Styles de lignes (solide, pointillé, etc.)
• Annotations temporelles
• Légendes personnalisées
• Sauvegarde des configurations

🔬 RÉFÉRENCES TECHNIQUES
• Matplotlib : https://matplotlib.org/
• Tkinter : https://docs.python.org/3/library/tkinter.html
• Design d'interface : Tufte (2001)

💡 APPLICATIONS CLINIQUES
• Adaptation aux protocoles spécifiques
• Amélioration de la lisibilité
• Standardisation des rapports
• Facilité d'interprétation

🚀 COMMENT ESSAYER
Accédez aux options via le menu "Affichage" → "Personnalisation"
            """,
            
            '😴 PSD par Stades de Sommeil': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse spectrale contextuelle par stades de sommeil pour caractériser les changements neurophysiologiques.

📋 ANALYSES DISPONIBLES
• Densité spectrale par stade (Éveil, N1, N2, N3, REM)
• Comparaison inter-stades
• Évolution temporelle des bandes spectrales
• Statistiques comparatives
• Visualisation des différences

🔬 RÉFÉRENCES SCIENTIFIQUES
• Analyse spectrale sommeil : Klimesch (1999)
• Bands EEG sommeil : Rechtschaffen & Kales (1968)
• YASA : Vallat & Walker (2019)
• SleepEEGpy : Falach et al. (2025)

💡 INTERPRÉTATION CLINIQUE
• Delta : Sommeil profond (N3)
• Theta : Sommeil léger (N1)
• Alpha : Éveil détendu
• Beta : Éveil actif
• Spindles : Sommeil N2

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "PSD par stade" après avoir chargé des données avec scoring
            """,
            
            '⏱️ Analyse Temporelle': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse des caractéristiques temporelles et de la variabilité des signaux EEG pour détecter les changements neurophysiologiques.

📋 MÉTRIQUES TEMPORELLES
• Variabilité temporelle (coefficient de variation)
• Détection des changements de régime
• Analyse des transitions
• Corrélations temporelles
• Détection d'événements

🔬 RÉFÉRENCES SCIENTIFIQUES
• Analyse temporelle EEG : Nunez & Srinivasan (2006)
• Variabilité temporelle : Stam (2005)
• Détection d'événements : Delorme & Makeig (2004)
• SciPy : https://scipy.org/

💡 APPLICATIONS CLINIQUES
• Détection des micro-éveils
• Analyse des transitions de sommeil
• Monitoring de l'anesthésie
• Détection d'épilepsie

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Analyse Temporelle" après sélection des canaux
            """,
            
            '🔗 Cohérence Inter-Canal': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse de connectivité fonctionnelle entre régions cérébrales pour étudier la synchronisation neuronale.

📋 MÉTRIQUES DE COHÉRENCE
• Cohérence spectrale (magnitude et phase)
• Cohérence partielle
• Cohérence imaginaire
• Analyse par bandes de fréquence
• Visualisation des réseaux

🔬 RÉFÉRENCES SCIENTIFIQUES
• Cohérence EEG : Nunez et al. (1997)
• Connectivité fonctionnelle : Stam (2005)
• MNE-Python : https://mne.tools/stable/
• SciPy : https://scipy.org/

💡 INTERPRÉTATION CLINIQUE
• Cohérence élevée : Synchronisation importante
• Cohérence faible : Découplage fonctionnel
• Variations pathologiques : Troubles neurologiques
• Applications : Recherche cognitive

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Cohérence" avec au moins 2 canaux sélectionnés
            """,
            
            '📊 Corrélations Temporelles': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse des relations temporelles entre canaux EEG pour identifier les patterns de synchronisation.

📋 ANALYSES DE CORRÉLATION
• Corrélations croisées temporelles
• Corrélations avec décalage temporel
• Matrices de corrélation
• Analyse de causalité
• Détection de patterns

🔬 RÉFÉRENCES SCIENTIFIQUES
• Corrélations EEG : Nunez & Srinivasan (2006)
• Analyse causale : Granger (1969)
• Pandas : https://pandas.pydata.org/
• NumPy : https://numpy.org/

💡 APPLICATIONS CLINIQUES
• Étude des réseaux neuronaux
• Détection de pathologies
• Recherche cognitive
• Monitoring anesthésique

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Corrélation" avec plusieurs canaux sélectionnés
            """,
            
            '🧠 Micro-états EEG': """
🎯 DESCRIPTION DÉTAILLÉE
Clustering et analyse des topographies des micro-états EEG pour identifier les patterns d'activité cérébrale.

📋 ANALYSES DES MICRO-ÉTATS
• Clustering des topographies
• Identification des classes de micro-états
• Analyse des transitions
• Durée et fréquence des états
• Cartographie des réseaux

🔬 RÉFÉRENCES SCIENTIFIQUES
• Micro-états EEG : Lehmann et al. (1987)
• Clustering : Pascual-Marqui et al. (1995)
• scikit-learn : https://scikit-learn.org/stable/
• NumPy : https://numpy.org/

💡 INTERPRÉTATION CLINIQUES
• Micro-états A : Visuel
• Micro-états B : Auditif
• Micro-états C : Attention
• Micro-états D : Réseau par défaut

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Micro-états" avec données multi-canaux
            """,
            
            '⚠️ Détection Artefacts': """
🎯 DESCRIPTION DÉTAILLÉE
Détection automatique et correction des artefacts dans les signaux EEG pour améliorer la qualité des analyses.

📋 TYPES D'ARTEFACTS DÉTECTÉS
• Artefacts musculaires (EMG)
• Artefacts oculaires (EOG)
• Artefacts cardiaques (ECG)
• Artefacts de mouvement
• Artefacts d'électrode

🔬 RÉFÉRENCES SCIENTIFIQUES
• Détection d'artefacts : Delorme & Makeig (2004)
• Correction automatique : Winkler et al. (2014)
• MNE-Python : https://mne.tools/stable/
• SciPy : https://scipy.org/

💡 MÉTHODES DE CORRECTION
• Rejet automatique des époques
• Interpolation des segments
• Filtrage adaptatif
• Marquage pour analyse

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Détection Artefacts" après chargement des données
            """,
            
            '🎯 Localisation Sources': """
🎯 DESCRIPTION DÉTAILLÉE
Reconstruction des sources cérébrales à partir des signaux EEG de surface pour localiser l'activité neuronale.

📋 MÉTHODES DE LOCALISATION
• Décomposition en sources indépendantes (ICA)
• Localisation dipolaire
• Reconstruction distribuée
• Analyse des composantes principales (PCA)
• Cartographie des sources

🔬 RÉFÉRENCES SCIENTIFIQUES
• Localisation sources : Baillet et al. (2001)
• ICA : Makeig et al. (1996)
• MNE-Python : https://mne.tools/stable/
• NumPy : https://numpy.org/

💡 APPLICATIONS CLINIQUES
• Localisation épileptogène
• Cartographie fonctionnelle
• Recherche cognitive
• Neurofeedback

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Localisation Sources" avec données multi-canaux
            """,
            
            '✋ Scoring Manuel Interactif': """
🎯 DESCRIPTION DÉTAILLÉE
Interface de classification manuelle des stades de sommeil avec outils d'aide à la décision et validation.

📋 FONCTIONNALITÉS INTERACTIVES
• Classification époque par époque
• Aide à la décision avec caractéristiques spectrales
• Validation croisée avec scoring automatique
• Outils de correction et révision
• Sauvegarde des modifications

🔬 RÉFÉRENCES SCIENTIFIQUES
• Standards AASM : Berry et al. (2017)
• Scoring manuel : Rechtschaffen & Kales (1968)
• Interface utilisateur : Nielsen (1994)
• Validation : Silber et al. (2007)

💡 CONSEILS D'UTILISATION
• Utilisez les caractéristiques spectrales comme aide
• Vérifiez la cohérence des transitions
• Marquez les périodes d'artefacts
• Validez avec le scoring automatique

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Scoring Manuel" après chargement des données
            """,
            
            '⚖️ Comparaison Scoring': """
🎯 DESCRIPTION DÉTAILLÉE
Validation automatique vs manuel avec métriques de performance détaillées et visualisation des différences.

📋 MÉTRIQUES DE PERFORMANCE
• Précision globale et par stade
• Rappel et spécificité
• F1-score et Kappa de Cohen
• Matrice de confusion
• Analyse des erreurs

🔬 RÉFÉRENCES SCIENTIFIQUES
• Métriques validation : scikit-learn documentation
• Kappa de Cohen : Cohen (1960)
• Validation scoring : Silber et al. (2007)
• YASA : Vallat & Walker (2019)

💡 INTERPRÉTATION DES RÉSULTATS
• Kappa > 0.8 : Excellent accord
• Kappa 0.6-0.8 : Bon accord
• Kappa < 0.6 : Accord modéré
• Analyse des désaccords pour amélioration

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Comparer Scoring" avec données automatiques et manuelles
            """,
            
            '📊 Métriques de Performance': """
🎯 DESCRIPTION DÉTAILLÉE
Calcul détaillé des métriques de performance par stade pour évaluer la qualité du scoring automatique.

📋 MÉTRIQUES CALCULÉES
• Précision, rappel, spécificité par stade
• F1-score et F-beta score
• Kappa de Cohen et accord inter-juges
• Temps de calcul et efficacité
• Analyse des erreurs systématiques

🔬 RÉFÉRENCES SCIENTIFIQUES
• Métriques classification : scikit-learn documentation
• Validation scoring : Silber et al. (2007)
• Kappa de Cohen : Cohen (1960)
• Performance YASA : Vallat & Walker (2019)

💡 INTERPRÉTATION CLINIQUE
• Précision élevée : Peu de faux positifs
• Rappel élevé : Peu de faux négatifs
• F1-score : Compromis optimal
• Kappa : Accord au-delà du hasard

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Métriques Performance" après comparaison
            """,
            
            '📈 Visualisation Hypnogramme': """
🎯 DESCRIPTION DÉTAILLÉE
Affichage temporel des stades de sommeil avec outils d'analyse et de navigation avancés.

📋 FONCTIONNALITÉS DE VISUALISATION
• Hypnogramme temporel complet
• Zoom et navigation temporelle
• Statistiques de sommeil automatiques
• Export des visualisations
• Comparaison multi-sujets

🔬 RÉFÉRENCES SCIENTIFIQUES
• Hypnogramme : Rechtschaffen & Kales (1968)
• Visualisation : Tufte (2001)
• Matplotlib : https://matplotlib.org/
• Statistiques sommeil : Silber et al. (2007)

💡 MÉTRIQUES DE SOMMEIL
• Temps total de sommeil (TTS)
• Efficacité du sommeil
• Latence d'endormissement
• Répartition des stades

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Visualisation Hypnogramme" après scoring
            """,
            
            '📚 Documentation Intégrée': """
🎯 DESCRIPTION DÉTAILLÉE
Guide complet avec références scientifiques pour une utilisation professionnelle d'CESA.

📋 CONTENU DE LA DOCUMENTATION
• Guide d'utilisation complet
• Références scientifiques détaillées
• Tutoriels étape par étape
• FAQ et résolution de problèmes
• Exemples d'analyses

🔬 RÉFÉRENCES SCIENTIFIQUES COMPLÈTES
• Toutes les références du fichier REFERENCES.txt
• DOI et URLs des publications
• Documentation des bibliothèques utilisées
• Standards internationaux EEG

💡 UTILISATION PROFESSIONNELLE
• Formation des utilisateurs
• Validation scientifique
• Documentation de recherche
• Support technique

🚀 COMMENT ESSAYER
Cliquez sur "Aide" → "Documentation complète" ou F1
            """,
            
            '🧪 Assistant de Première Utilisation': """
🎯 DESCRIPTION DÉTAILLÉE
Tutoriel interactif personnalisé pour guider les nouveaux utilisateurs dans la découverte d'CESA.

📋 CONTENU DU TUTORIEL
• Introduction à CESA v0.0beta1.0
• Guide de première utilisation
• Démonstration des fonctionnalités principales
• Conseils d'utilisation personnalisés
• Options de configuration

🔬 RÉFÉRENCES PÉDAGOGIQUES
• Design d'interface : Nielsen (1994)
• Onboarding utilisateur : Krug (2014)
• Ergonomie logicielle : Tufte (2001)
• UX Design : Norman (2013)

💡 AVANTAGES PÉDAGOGIQUES
• Apprentissage progressif
• Personnalisation selon le niveau
• Réduction de la courbe d'apprentissage
• Amélioration de l'adoption

🚀 COMMENT ESSAYER
Cliquez sur "Aide" → "Assistant de première utilisation" ou au premier lancement
            """,
            
            '📞 Support Technique': """
🎯 DESCRIPTION DÉTAILLÉE
Aide et contact avec l'équipe de développement pour résoudre les problèmes et améliorer CESA.

📋 SERVICES DE SUPPORT
• Résolution de problèmes techniques
• Aide à l'utilisation avancée
• Suggestions d'amélioration
• Formation personnalisée
• Support par email

🔬 CONTACT ET RESSOURCES
• Email : come1.barmoy@supbiotech.fr
• Documentation : README.md complet
• Diagnostic automatique intégré
• Communauté utilisateurs

💡 TYPES DE SUPPORT
• Support technique gratuit
• Formation sur site (sur demande)
• Développement de fonctionnalités personnalisées
• Intégration dans des workflows existants

🚀 COMMENT ESSAYER
Cliquez sur "Aide" → "Support technique" pour accéder aux informations de contact
            """,

            # Ajouter les descriptions manquantes identifiées
            '📈 Statistiques Descriptives': """
🎯 DESCRIPTION DÉTAILLÉE
Calcul complet des métriques statistiques descriptives pour caractériser les propriétés des signaux EEG.

📋 MÉTRIQUES CALCULÉES
• Statistiques de base : moyenne, médiane, écart-type, variance
• Métriques de forme : skewness, kurtosis
• Amplitude : min, max, amplitude crête-à-crête
• Variabilité : coefficient de variation, RMS
• Distribution : percentiles, quartiles
• Qualité du signal : rapport signal/bruit, artefacts

🔬 RÉFÉRENCES SCIENTIFIQUES
• Statistiques EEG : Nunez, P.L. & Srinivasan, R. "Electric fields of the brain: the neurophysics of EEG." Oxford University Press, 2006.
• Métriques de qualité : Delorme, A. & Makeig, S. "EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics." Journal of Neuroscience Methods 134.1 (2004): 9-21.
• Pandas : https://pandas.pydata.org/
• NumPy : https://numpy.org/
• scikit-learn : https://scikit-learn.org/stable/

💡 UTILISATION CLINIQUE
• Validation de la qualité des enregistrements
• Détection d'artefacts et de dérives
• Comparaison entre conditions expérimentales
• Contrôle qualité des analyses

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Statistiques" pour obtenir un rapport complet
            """,

            '🌊 Analyse Spectrale (FFT/Welch)': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse fréquentielle haute résolution utilisant les méthodes FFT et Welch pour caractériser le contenu spectral des signaux EEG.

📋 FONCTIONNALITÉS AVANCÉES
• Méthode Welch avec fenêtre de Hann et 50% de chevauchement
• Calcul de la densité spectrale de puissance (PSD)
• Bands de fréquence standard : Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
• Détection des pics spectraux et centroïde spectral
• Visualisation interactive des spectres
• Export des résultats en format numérique

🔬 RÉFÉRENCES SCIENTIFIQUES COMPLÈTES
• Méthode Welch : Welch, P.D. "The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms." IEEE Transactions on Audio and Electroacoustics 15.2 (1967): 70-73.
• SciPy Signal Processing : https://scipy.org/
• Analyse spectrale EEG : Klimesch, W. "EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis." Brain Research Reviews 29.2-3 (1999): 169-195.
• Bands de fréquence EEG : Niedermeyer, E. & da Silva, F.L. "Electroencephalography: basic principles, clinical applications, and related fields." Lippincott Williams & Wilkins, 2005.
• Source interne : Analyse_spectrale.py (adaptée dans spectral_analysis.py)

💡 INTERPRÉTATION CLINIQUE
• Delta : Sommeil profond, pathologie neurologique
• Theta : Sommeil léger, méditation, créativité
• Alpha : Éveil détendu, yeux fermés
• Beta : Éveil actif, attention, anxiété
• Gamma : Processus cognitifs complexes

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Analyse Spectrale" après sélection des canaux
            """,

            '😴 PSD par Stades de Sommeil': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse spectrale contextuelle par stades de sommeil pour caractériser les changements neurophysiologiques.

📋 ANALYSES DISPONIBLES
• Densité spectrale par stade (Éveil, N1, N2, N3, REM)
• Comparaison inter-stades
• Évolution temporelle des bandes spectrales
• Statistiques comparatives
• Visualisation des différences

🔬 RÉFÉRENCES SCIENTIFIQUES
• Analyse spectrale sommeil : Klimesch (1999)
• Bands EEG sommeil : Rechtschaffen & Kales (1968)
• YASA : Vallat & Walker (2019)
• SleepEEGpy : Falach et al. (2025)

💡 INTERPRÉTATION CLINIQUE
• Delta : Sommeil profond (N3)
• Theta : Sommeil léger (N1)
• Alpha : Éveil détendu
• Beta : Éveil actif
• Spindles : Sommeil N2

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "PSD par stade" après avoir chargé des données avec scoring
            """,

            '⏱️ Analyse Temporelle': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse des caractéristiques temporelles et de la variabilité des signaux EEG pour détecter les changements neurophysiologiques.

📋 MÉTRIQUES TEMPORELLES
• Variabilité temporelle (coefficient de variation)
• Détection des changements de régime
• Analyse des transitions
• Corrélations temporelles
• Détection d'événements

🔬 RÉFÉRENCES SCIENTIFIQUES
• Analyse temporelle EEG : Nunez & Srinivasan (2006)
• Variabilité temporelle : Stam (2005)
• Détection d'événements : Delorme & Makeig (2004)
• SciPy : https://scipy.org/

💡 APPLICATIONS CLINIQUES
• Détection des micro-éveils
• Analyse des transitions de sommeil
• Monitoring de l'anesthésie
• Détection d'épilepsie

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Analyse Temporelle" après sélection des canaux
            """,

            '🔗 Cohérence Inter-Canal': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse de connectivité fonctionnelle entre régions cérébrales pour étudier la synchronisation neuronale.

📋 MÉTRIQUES DE COHÉRENCE
• Cohérence spectrale (magnitude et phase)
• Cohérence partielle
• Cohérence imaginaire
• Analyse par bandes de fréquence
• Visualisation des réseaux

🔬 RÉFÉRENCES SCIENTIFIQUES
• Cohérence EEG : Nunez et al. (1997)
• Connectivité fonctionnelle : Stam (2005)
• MNE-Python : https://mne.tools/stable/
• SciPy : https://scipy.org/

💡 INTERPRÉTATION CLINIQUE
• Cohérence élevée : Synchronisation importante
• Cohérence faible : Découplage fonctionnel
• Variations pathologiques : Troubles neurologiques
• Applications : Recherche cognitive

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Cohérence" avec au moins 2 canaux sélectionnés
            """,

            '📊 Corrélations Temporelles': """
🎯 DESCRIPTION DÉTAILLÉE
Analyse des relations temporelles entre canaux EEG pour identifier les patterns de synchronisation.

📋 ANALYSES DE CORRÉLATION
• Corrélations croisées temporelles
• Corrélations avec décalage temporel
• Matrices de corrélation
• Analyse de causalité
• Détection de patterns

🔬 RÉFÉRENCES SCIENTIFIQUES
• Corrélations EEG : Nunez & Srinivasan (2006)
• Analyse causale : Granger (1969)
• Pandas : https://pandas.pydata.org/
• NumPy : https://numpy.org/

💡 APPLICATIONS CLINIQUES
• Étude des réseaux neuronaux
• Détection de pathologies
• Recherche cognitive
• Monitoring anesthésique

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Corrélation" avec plusieurs canaux sélectionnés
            """,

            '🧠 Micro-états EEG': """
🎯 DESCRIPTION DÉTAILLÉE
Clustering et analyse des topographies des micro-états EEG pour identifier les patterns d'activité cérébrale.

📋 ANALYSES DES MICRO-ÉTATS
• Clustering des topographies
• Identification des classes de micro-états
• Analyse des transitions
• Durée et fréquence des états
• Cartographie des réseaux

🔬 RÉFÉRENCES SCIENTIFIQUES
• Micro-états EEG : Lehmann et al. (1987)
• Clustering : Pascual-Marqui et al. (1995)
• scikit-learn : https://scikit-learn.org/stable/
• NumPy : https://numpy.org/

💡 INTERPRÉTATION CLINIQUES
• Micro-états A : Visuel
• Micro-états B : Auditif
• Micro-états C : Attention
• Micro-états D : Réseau par défaut

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Micro-états" avec données multi-canaux
            """,

            '⚠️ Détection Artefacts': """
🎯 DESCRIPTION DÉTAILLÉE
Détection automatique et correction des artefacts dans les signaux EEG pour améliorer la qualité des analyses.

📋 TYPES D'ARTEFACTS DÉTECTÉS
• Artefacts musculaires (EMG)
• Artefacts oculaires (EOG)
• Artefacts cardiaques (ECG)
• Artefacts de mouvement
• Artefacts d'électrode

🔬 RÉFÉRENCES SCIENTIFIQUES
• Détection d'artefacts : Delorme & Makeig (2004)
• Correction automatique : Winkler et al. (2014)
• MNE-Python : https://mne.tools/stable/
• SciPy : https://scipy.org/

💡 MÉTHODES DE CORRECTION
• Rejet automatique des époques
• Interpolation des segments
• Filtrage adaptatif
• Marquage pour analyse

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Détection Artefacts" après chargement des données
            """,

            '🎯 Localisation Sources': """
🎯 DESCRIPTION DÉTAILLÉE
Reconstruction des sources cérébrales à partir des signaux EEG de surface pour localiser l'activité neuronale.

📋 MÉTHODES DE LOCALISATION
• Décomposition en sources indépendantes (ICA)
• Localisation dipolaire
• Reconstruction distribuée
• Analyse des composantes principales (PCA)
• Cartographie des sources

🔬 RÉFÉRENCES SCIENTIFIQUES
• Localisation sources : Baillet et al. (2001)
• ICA : Makeig et al. (1996)
• MNE-Python : https://mne.tools/stable/
• NumPy : https://numpy.org/

💡 APPLICATIONS CLINIQUES
• Localisation épileptogène
• Cartographie fonctionnelle
• Recherche cognitive
• Neurofeedback

🚀 COMMENT ESSAYER
Cliquez sur "Analyse" → "Localisation Sources" avec données multi-canaux
            """,

            '✋ Scoring Manuel Interactif': """
🎯 DESCRIPTION DÉTAILLÉE
Interface de classification manuelle des stades de sommeil avec outils d'aide à la décision et validation.

📋 FONCTIONNALITÉS INTERACTIVES
• Classification époque par époque
• Aide à la décision avec caractéristiques spectrales
• Validation croisée avec scoring automatique
• Outils de correction et révision
• Sauvegarde des modifications

🔬 RÉFÉRENCES SCIENTIFIQUES
• Standards AASM : Berry et al. (2017)
• Scoring manuel : Rechtschaffen & Kales (1968)
• Interface utilisateur : Nielsen (1994)
• Validation : Silber et al. (2007)

💡 CONSEILS D'UTILISATION
• Utilisez les caractéristiques spectrales comme aide
• Vérifiez la cohérence des transitions
• Marquez les périodes d'artefacts
• Validez avec le scoring automatique

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Scoring Manuel" après chargement des données
            """,

            '⚖️ Comparaison Scoring': """
🎯 DESCRIPTION DÉTAILLÉE
Validation automatique vs manuel avec métriques de performance détaillées et visualisation des différences.

📋 MÉTRIQUES DE PERFORMANCE
• Précision globale et par stade
• Rappel et spécificité
• F1-score et Kappa de Cohen
• Matrice de confusion
• Analyse des erreurs

🔬 RÉFÉRENCES SCIENTIFIQUES
• Métriques validation : scikit-learn documentation
• Kappa de Cohen : Cohen (1960)
• Validation scoring : Silber et al. (2007)
• YASA : Vallat & Walker (2019)

💡 INTERPRÉTATION DES RÉSULTATS
• Kappa > 0.8 : Excellent accord
• Kappa 0.6-0.8 : Bon accord
• Kappa < 0.6 : Accord modéré
• Analyse des désaccords pour amélioration

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Comparer Scoring" avec données automatiques et manuelles
            """,

            '📊 Métriques de Performance': """
🎯 DESCRIPTION DÉTAILLÉE
Calcul détaillé des métriques de performance par stade pour évaluer la qualité du scoring automatique.

📋 MÉTRIQUES CALCULÉES
• Précision, rappel, spécificité par stade
• F1-score et F-beta score
• Kappa de Cohen et accord inter-juges
• Temps de calcul et efficacité
• Analyse des erreurs systématiques

🔬 RÉFÉRENCES SCIENTIFIQUES
• Métriques classification : scikit-learn documentation
• Validation scoring : Silber et al. (2007)
• Kappa de Cohen : Cohen (1960)
• Performance YASA : Vallat & Walker (2019)

💡 INTERPRÉTATION CLINIQUE
• Précision élevée : Peu de faux positifs
• Rappel élevé : Peu de faux négatifs
• F1-score : Compromis optimal
• Kappa : Accord au-delà du hasard

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Métriques Performance" après comparaison
            """,

            '📈 Visualisation Hypnogramme': """
🎯 DESCRIPTION DÉTAILLÉE
Affichage temporel des stades de sommeil avec outils d'analyse et de navigation avancés.

📋 FONCTIONNALITÉS DE VISUALISATION
• Hypnogramme temporel complet
• Zoom et navigation temporelle
• Statistiques de sommeil automatiques
• Export des visualisations
• Comparaison multi-sujets

🔬 RÉFÉRENCES SCIENTIFIQUES
• Hypnogramme : Rechtschaffen & Kales (1968)
• Visualisation : Tufte (2001)
• Matplotlib : https://matplotlib.org/
• Statistiques sommeil : Silber et al. (2007)

💡 MÉTRIQUES DE SOMMEIL
• Temps total de sommeil (TTS)
• Efficacité du sommeil
• Latence d'endormissement
• Répartition des stades

🚀 COMMENT ESSAYER
Cliquez sur "Scoring" → "Visualisation Hypnogramme" après scoring
            """,

            '📚 Documentation Intégrée': """
🎯 DESCRIPTION DÉTAILLÉE
Guide complet avec références scientifiques pour une utilisation professionnelle d'CESA.

📋 CONTENU DE LA DOCUMENTATION
• Guide d'utilisation complet
• Références scientifiques détaillées
• Tutoriels étape par étape
• FAQ et résolution de problèmes
• Exemples d'analyses

🔬 RÉFÉRENCES SCIENTIFIQUES COMPLÈTES
• Toutes les références du fichier REFERENCES.txt
• DOI et URLs des publications
• Documentation des bibliothèques utilisées
• Standards internationaux EEG

💡 UTILISATION PROFESSIONNELLE
• Formation des utilisateurs
• Validation scientifique
• Documentation de recherche
• Support technique

🚀 COMMENT ESSAYER
Cliquez sur "Aide" → "Documentation complète" ou F1
            """,

            '🧪 Assistant de Première Utilisation': """
🎯 DESCRIPTION DÉTAILLÉE
Tutoriel interactif personnalisé pour guider les nouveaux utilisateurs dans la découverte d'CESA.

📋 CONTENU DU TUTORIEL
• Introduction à CESA v0.0beta1.0
• Guide de première utilisation
• Démonstration des fonctionnalités principales
• Conseils d'utilisation personnalisés
• Options de configuration

🔬 RÉFÉRENCES PÉDAGOGIQUES
• Design d'interface : Nielsen (1994)
• Onboarding utilisateur : Krug (2014)
• Ergonomie logicielle : Tufte (2001)
• UX Design : Norman (2013)

💡 AVANTAGES PÉDAGOGIQUES
• Apprentissage progressif
• Personnalisation selon le niveau
• Réduction de la courbe d'apprentissage
• Amélioration de l'adoption

🚀 COMMENT ESSAYER
Cliquez sur "Aide" → "Assistant de première utilisation" ou au premier lancement
            """,

            '🔍 Explorateur de Fonctionnalités': """
🎯 DESCRIPTION DÉTAILLÉE
Explorateur interactif approfondi des fonctionnalités avec descriptions détaillées et références scientifiques.

📋 FONCTIONNALITÉS DE L'EXPLORATEUR
• Navigation par catégories (Fichiers, Visualisation, Analyses, etc.)
• Descriptions détaillées pour chaque fonctionnalité
• Références scientifiques complètes
• Tutoriels étape par étape
• Liens directs vers les fonctionnalités
• Interface de recherche et filtrage

🔬 RÉFÉRENCES TECHNIQUES
• Tkinter Treeview : https://docs.python.org/3/library/tkinter.ttk.html
• Interface utilisateur : Nielsen (1994)
• Design d'information : Tufte (2001)
• Base de données de références : REFERENCES.txt

💡 AVANTAGES PÉDAGOGIQUES
• Découverte progressive des fonctionnalités
• Compréhension approfondie des méthodes
• Accès rapide aux références scientifiques
• Formation personnalisée selon les besoins

🚀 COMMENT ESSAYER
Cliquez sur "Aide" → "Explorateur de fonctionnalités" pour découvrir toutes les capacités d'CESA
            """
        }
        
        # Afficher les informations détaillées
        if feature_name in detailed_info:
            content = detailed_info[feature_name]
            print(f"🔍 DEBUG: Insertion de {len(content)} caractères pour '{feature_name}'")
            self.feature_details.config(state='normal')
            self.feature_details.insert(tk.END, content)
            self.feature_details.config(state='disabled')
        else:
            # Fallback pour les fonctionnalités non définies
            fallback_content = f"""🎯 {feature_name}
═══════════════════════════════════════════════════════════════════════════════

📋 DESCRIPTION
Cette fonctionnalité fait partie d'CESA v0.0beta1.0 et contribue à l'analyse complète des signaux EEG.

🔬 RÉFÉRENCES SCIENTIFIQUES GÉNÉRALES
• MNE-Python : https://mne.tools/stable/
• YASA : https://yasa-sleep.org/
• SleepEEGpy : https://github.com/NirLab-TAU/sleepeegpy
• SciPy : https://scipy.org/
• Matplotlib : https://matplotlib.org/
• Pandas : https://pandas.pydata.org/
• scikit-learn : https://scikit-learn.org/stable/

💡 UTILISATION
Consultez le guide de référence complet pour des informations détaillées sur cette fonctionnalité.

🚀 COMMENT ESSAYER
Utilisez le bouton "🚀 Essayer cette fonctionnalité" pour lancer directement cette fonctionnalité."""
            print(f"🔍 DEBUG: Insertion fallback de {len(fallback_content)} caractères pour '{feature_name}'")
            self.feature_details.config(state='normal')
            self.feature_details.insert(tk.END, fallback_content)
            self.feature_details.config(state='disabled')

        # Vérifier que le contenu a été inséré
        final_content = self.feature_details.get(1.0, tk.END).strip()
        print(f"✅ DEBUG: Contenu final inséré - {len(final_content)} caractères")

        # Forcer la mise à jour et scroll vers le haut
        self.feature_details.update()
        self.feature_details.see(1.0)
    
    def _show_feature_tutorial(self, feature_name):
        """Affiche un tutoriel détaillé pour une fonctionnalité spécifique."""
        tutorial_window = tk.Toplevel(self.parent_app.root)
        tutorial_window.title(f"📚 Tutoriel - {feature_name}")
        tutorial_window.geometry("1000x800")
        tutorial_window.transient(self.parent_app.root)
        tutorial_window.grab_set()
        
        # Centrer la fenêtre
        tutorial_window.update_idletasks()
        x = (tutorial_window.winfo_screenwidth() // 2) - (1000 // 2)
        y = (tutorial_window.winfo_screenheight() // 2) - (800 // 2)
        tutorial_window.geometry(f"1000x800+{x}+{y}")
        
        # Frame principal avec scrollbar
        main_frame = ttk.Frame(tutorial_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas_frame = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_frame.yview)
        scrollable_frame = ttk.Frame(canvas_frame)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )
        
        canvas_frame.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        
        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Titre
        title_label = ttk.Label(scrollable_frame, text=f"📚 Tutoriel Détaillé - {feature_name}", 
                               font=('Segoe UI', 18, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Contenu du tutoriel
        tutorial_content = self._get_tutorial_content(feature_name)
        
        tutorial_text = tk.Text(scrollable_frame, wrap=tk.WORD, font=('Segoe UI', 11))
        tutorial_text.pack(fill=tk.BOTH, expand=True)
        tutorial_text.insert(tk.END, tutorial_content)
        tutorial_text.config(state='disabled')
        
        # Bouton fermer
        ttk.Button(scrollable_frame, text="Fermer", 
                  command=tutorial_window.destroy).pack(pady=(20, 0))
    
    def _get_tutorial_content(self, feature_name):
        """Retourne le contenu du tutoriel pour une fonctionnalité."""
        tutorials = {
            '🧮 Entropie Renormée (Issartel)': """
🎯 TUTORIEL COMPLET - ENTROPIE RENORMÉE (ISSTARTEL)
═══════════════════════════════════════════════════════════════════════════════

📋 ÉTAPE 1 : PRÉPARATION DES DONNÉES
─────────────────────────────────────
1. Chargez un fichier EDF avec des données EEG de qualité
2. Vérifiez que les canaux EEG sont bien détectés
3. Assurez-vous que la fréquence d'échantillonnage est ≥ 100 Hz
4. Les données doivent contenir au moins 2 canaux EEG

📋 ÉTAPE 2 : SÉLECTION DES CANAUX
─────────────────────────────────
1. Cliquez sur "Sélectionner Canaux" dans le menu Affichage
2. Sélectionnez 2-8 canaux EEG de bonne qualité
3. Évitez les canaux avec beaucoup d'artefacts
4. Les canaux recommandés : C3-M2, C4-M1, F3-M2, F4-M1

📋 ÉTAPE 3 : CONFIGURATION DES PARAMÈTRES
─────────────────────────────────────────
1. Menu Analyse → Entropie Renormée (Issartel)
2. Configurez les paramètres :
   • Longueur de fenêtre : 4-8 secondes (recommandé : 6s)
   • Chevauchement : 50% (recommandé)
   • Ordre du moment : 2.0 (RMS, robuste)
   • Kernel : Powerlaw avec γ=0.5 (recommandé)

📋 ÉTAPE 4 : LANCEMENT DU CALCUL
─────────────────────────────────
1. Cliquez sur "Calculer Entropie"
2. Attendez la fin du calcul (peut prendre quelques minutes)
3. Les résultats s'affichent automatiquement

📋 ÉTAPE 5 : INTERPRÉTATION DES RÉSULTATS
─────────────────────────────────────────
• Entropie élevée (> 5 nats) : Signal très complexe
• Entropie modérée (2-5 nats) : Signal moyennement complexe
• Entropie faible (< 2 nats) : Signal régulier

💡 CONSEILS AVANCÉS
───────────────────
• Utilisez des fenêtres de 6 secondes pour un bon compromis
• Le kernel Powerlaw avec γ=0.5 est optimal pour la plupart des cas
• Comparez les résultats entre différents stades de sommeil
• Vérifiez la cohérence des résultats avec d'autres métriques

🔬 APPLICATIONS CLINIQUES
─────────────────────────
• Monitoring anesthésique
• Analyse des états de conscience
• Détection de pathologies neurologiques
• Recherche en neurosciences cognitives
            """,

            '🤖 Scoring Automatique (YASA)': """
🎯 TUTORIEL COMPLET - SCORING AUTOMATIQUE (YASA)
═══════════════════════════════════════════════════════════════════════════════

📋 ÉTAPE 1 : PRÉPARATION DES DONNÉES
─────────────────────────────────────
1. Chargez un fichier EDF avec données polysomnographiques
2. Assurez-vous d'avoir au minimum : EEG, EOG, EMG
3. La durée d'enregistrement doit être ≥ 2 heures
4. Vérifiez la qualité des signaux (peu d'artefacts)

📋 ÉTAPE 2 : CONFIGURATION YASA
────────────────────────────────
1. Menu Scoring → Scoring Automatique
2. Sélectionnez les canaux requis :
   • EEG : C3-M2, C4-M1 (ou équivalents)
   • EOG : E1-M2, E2-M1 (ou équivalents)
   • EMG : Chin EMG (ou équivalent)
3. Configurez les paramètres :
   • Fréquence d'échantillonnage : détectée automatiquement
   • Durée d'époque : 30 secondes (standard)

📋 ÉTAPE 3 : LANCEMENT DU SCORING
──────────────────────────────────
1. Cliquez sur "Lancer Scoring Automatique"
2. Attendez la fin du calcul (peut prendre 5-15 minutes)
3. Les résultats s'affichent dans l'hypnogramme

📋 ÉTAPE 4 : VALIDATION DES RÉSULTATS
─────────────────────────────────────
1. Examinez l'hypnogramme généré
2. Vérifiez la cohérence des transitions
3. Identifiez les périodes problématiques
4. Comparez avec un scoring manuel si disponible

📋 ÉTAPE 5 : EXPORT ET ANALYSE
───────────────────────────────
1. Exportez les résultats en Excel/CSV
2. Analysez les métriques de performance
3. Calculez les statistiques de sommeil
4. Générez un rapport automatique

💡 CONSEILS POUR DE MEILLEURES PERFORMANCES
────────────────────────────────────────────
• Utilisez des données de haute qualité
• Évitez les périodes avec beaucoup d'artefacts
• Vérifiez la calibration des électrodes
• Les performances sont meilleures sur N2 et REM

🔬 INTERPRÉTATION CLINIQUE
───────────────────────────
• Précision attendue : 80-85%
• Kappa de Cohen : 0.75-0.85
• Meilleures performances sur données polysomnographiques
• Sensible à la qualité des données et au montage
            """
        }
        
        return tutorials.get(feature_name, f"""
🎯 TUTORIEL - {feature_name}
═══════════════════════════════════════════════════════════════════════════════

📋 GUIDE D'UTILISATION
─────────────────────
1. Chargez vos données EEG dans CESA
2. Sélectionnez les canaux appropriés
3. Accédez à cette fonctionnalité via le menu correspondant
4. Configurez les paramètres selon vos besoins
5. Lancez l'analyse et examinez les résultats

💡 CONSEILS GÉNÉRAUX
────────────────────
• Consultez la documentation détaillée pour plus d'informations
• Utilisez les données d'exemple pour vous familiariser
• Vérifiez la qualité de vos données avant l'analyse
• Sauvegardez vos configurations personnalisées

🔬 RÉFÉRENCES
─────────────
Consultez le guide de référence complet pour les références scientifiques détaillées.
        """)
    
    def _try_feature(self, feature_name):
        """Lance directement une fonctionnalité depuis l'explorateur."""
        # Mapping des fonctionnalités vers les méthodes de l'application
        feature_mapping = {
            '📂 Charger fichier EDF': '_load_file',
            '🧮 Entropie Renormée (Issartel)': '_show_renormalized_entropy',
            '🤖 Scoring Automatique (YASA)': '_run_auto_scoring',
            '🌊 Analyse Spectrale (FFT/Welch)': '_show_spectral_analysis',
            '📈 Statistiques Descriptives': '_show_statistics',
            '🔧 Diagnostic Système': '_run_diagnostic'
        }
        
        method_name = feature_mapping.get(feature_name)
        if method_name and hasattr(self.parent_app, method_name):
            try:
                method = getattr(self.parent_app, method_name)
                method()
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de lancer {feature_name}:\n\n{str(e)}")
        else:
            messagebox.showinfo("Information", f"Fonctionnalité {feature_name} en cours de développement.\n\nConsultez le tutoriel détaillé pour plus d'informations.")
    
    def _suggest_load_sample_data(self):
        """Suggère de charger des données d'exemple."""
        samples_dir = Path("samples")
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.edf"))
            if sample_files:
                sample_file = sample_files[0]
                result = messagebox.askyesno("Données d'exemple", 
                                           f"Voulez-vous charger le fichier d'exemple :\n{sample_file.name} ?")
                if result:
                    try:
                        self.parent_app._load_file(str(sample_file))
                        messagebox.showinfo("Succès", 
                                          "Données d'exemple chargées avec succès !\n\n"
                                          "Vous pouvez maintenant explorer les fonctionnalités d'CESA.")
                    except Exception as e:
                        messagebox.showerror("Erreur", f"Erreur lors du chargement :\n{e}")
            else:
                messagebox.showinfo("Information", 
                                  "Aucun fichier d'exemple trouvé dans le dossier 'samples'.\n\n"
                                  "Vous pouvez charger vos propres fichiers EDF+ via le menu Fichier.")
        else:
            messagebox.showinfo("Information", 
                              "Dossier 'samples' non trouvé.\n\n"
                              "Vous pouvez charger vos propres fichiers EDF+ via le menu Fichier.")
    
    def create_tooltip(self, widget, text, delay=500):
        """Crée un tooltip détaillé pour un widget."""
        if not self.user_preferences.get('show_tooltips', True):
            return
        
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            # Style du tooltip
            tooltip.configure(bg='#ffffe0', relief='solid', borderwidth=1)
            
            label = tk.Label(tooltip, text=text, justify='left', 
                           background='#ffffe0', relief='solid', borderwidth=1,
                           font=('Segoe UI', 9), wraplength=300, padx=5, pady=5)
            label.pack()
            
            widget.tooltip = tooltip
            
            # Auto-fermeture après 5 secondes
            tooltip.after(5000, lambda: hide_tooltip(event))
        
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip') and widget.tooltip:
                widget.tooltip.destroy()
                widget.tooltip = None
        
        widget.bind('<Enter>', lambda e: widget.after(delay, lambda: show_tooltip(e)))
        widget.bind('<Leave>', hide_tooltip)
        widget.bind('<Button-1>', hide_tooltip)
    
    def show_help_menu(self):
        """Affiche le menu d'aide contextuel."""
        help_window = tk.Toplevel(self.parent_app.root)
        help_window.title("❓ Aide CESA v0.0beta1.0")
        help_window.geometry("600x500")
        help_window.transient(self.parent_app.root)
        help_window.grab_set()
        
        # Centrer la fenêtre
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (help_window.winfo_screenheight() // 2) - (500 // 2)
        help_window.geometry(f"600x500+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(help_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="❓ Aide CESA v0.0beta1.0", 
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Options d'aide
        help_options = [
            ("🎯 Visite guidée", "Découvrez CESA étape par étape", self.show_guided_tour),
            ("🔍 Explorateur de fonctionnalités", "Explorez toutes les fonctionnalités", self.show_feature_explorer),
            ("📚 Documentation complète", "Consultez le README.md", lambda: self._open_documentation()),
            ("🧮 Guide entropie renormée", "Documentation de l'entropie renormée", lambda: self._open_entropy_docs()),
            ("📞 Support technique", "Contactez le support", lambda: self._open_support()),
            ("🔧 Diagnostic système", "Vérifiez votre installation", lambda: self._run_diagnostic())
        ]
        
        for title, description, command in help_options:
            option_frame = ttk.Frame(main_frame)
            option_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(option_frame, text=title, font=('Segoe UI', 12, 'bold')).pack(anchor='w')
            ttk.Label(option_frame, text=description, font=('Segoe UI', 10)).pack(anchor='w')
            ttk.Button(option_frame, text="Ouvrir", command=command).pack(anchor='e')
        
        # Bouton fermer
        ttk.Button(main_frame, text="Fermer", 
                  command=help_window.destroy).pack(pady=(20, 0))
    
    def _open_documentation(self):
        """Ouvre la documentation principale."""
        try:
            readme_path = Path("README.md")
            if readme_path.exists():
                webbrowser.open(f"file://{readme_path.absolute()}")
            else:
                messagebox.showinfo("Documentation", 
                                  "Fichier README.md non trouvé.\n\n"
                                  "Consultez la documentation en ligne ou contactez le support.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ouverture de la documentation :\n{e}")
    
    def _open_entropy_docs(self):
        """Ouvre la documentation de l'entropie renormée."""
        try:
            entropy_docs_path = Path("ENTROPY_INTEGRATION.md")
            if entropy_docs_path.exists():
                webbrowser.open(f"file://{entropy_docs_path.absolute()}")
            else:
                messagebox.showinfo("Documentation", 
                                  "Fichier ENTROPY_INTEGRATION.md non trouvé.\n\n"
                                  "Consultez la documentation en ligne ou contactez le support.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ouverture de la documentation :\n{e}")
    
    def _open_support(self):
        """Ouvre les informations de support."""
        support_info = """
📞 SUPPORT CESA v0.0beta1.0

🆘 EN CAS DE PROBLÈME :

1️⃣ CONSULTEZ LA DOCUMENTATION :
   • README.md : Guide général
   • ENTROPY_INTEGRATION.md : Entropie renormée
   • GUIDE_INSTALLATION_NOOB.md : Installation

2️⃣ UTILISEZ LES OUTILS INTÉGRÉS :
   • Assistant de première utilisation
   • Diagnostic automatique
   • Tooltips contextuels

3️⃣ CONTACTEZ LE SUPPORT :
   • Email : come1.barmoy@supbiotech.fr
   • GitHub : cbarmoy
   • Unité Neuropsychologie du Stress (IRBA)

💡 CONSEILS :
• Décrivez précisément votre problème
• Incluez les messages d'erreur
• Précisez votre configuration système
• Joignez des captures d'écran si utile

🎯 DÉVELOPPEMENT :
CESA v0.0beta1.0 est développé pour l'Unité Neuropsychologie du Stress (IRBA)
Auteur : Côme Barmoy
Version : 0.0beta1.0
Licence : MIT
        """
        
        messagebox.showinfo("Support CESA v0.0beta1.0", support_info)
    
    def _open_reference_guide(self):
        """Ouvre le guide de référence complet."""
        reference_window = tk.Toplevel(self.parent_app.root)
        reference_window.title("📚 Guide de Référence CESA v0.0beta1.0")
        reference_window.geometry("1000x800")
        reference_window.transient(self.parent_app.root)
        reference_window.grab_set()
        
        # Centrer la fenêtre
        reference_window.update_idletasks()
        x = (reference_window.winfo_screenwidth() // 2) - (1000 // 2)
        y = (reference_window.winfo_screenheight() // 2) - (800 // 2)
        reference_window.geometry(f"1000x800+{x}+{y}")
        
        # Frame principal avec scrollbar
        main_frame = ttk.Frame(reference_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Créer un canvas avec scrollbar
        canvas_frame = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_frame.yview)
        scrollable_frame = ttk.Frame(canvas_frame)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )
        
        canvas_frame.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas et scrollbar
        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Titre
        title_label = ttk.Label(scrollable_frame, text="📚 Guide de Référence CESA v0.0beta1.0", 
                               font=('Segoe UI', 18, 'bold'))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(scrollable_frame,
                                 text="Documentation Complète et Références Scientifiques",
                                 font=('Segoe UI', 12, 'italic'))
        subtitle_label.pack(pady=(0, 20))
        
        # Section Documentation
        doc_frame = ttk.LabelFrame(scrollable_frame, text="📖 Documentation Disponible", padding=15)
        doc_frame.pack(fill=tk.X, pady=(0, 15))
        
        doc_text = """
📋 FICHIERS DE DOCUMENTATION :

1️⃣ README.md - Guide Principal :
   • Installation et configuration d'CESA v0.0beta1.0
   • Guide d'utilisation général
   • Résolution des problèmes courants
   • Informations sur les nouvelles fonctionnalités

2️⃣ Guide de Référence Intégré :
   • Documentation complète de toutes les fonctionnalités
   • Exemples d'utilisation détaillés
   • Paramètres recommandés pour chaque analyse
   • Conseils d'interprétation des résultats

3️⃣ Assistant Intégré :
   • Aide contextuelle dans l'application
   • Tooltips détaillés pour chaque fonctionnalité
   • Diagnostic automatique des problèmes
   • Support technique intégré

4️⃣ Documentation Scientifique :
   • Références bibliographiques complètes
   • Méthodes d'analyse utilisées
   • Standards EEG internationaux
   • Validation des algorithmes
        """
        
        doc_label = ttk.Label(doc_frame, text=doc_text.strip(), 
                             font=('Segoe UI', 10), justify='left')
        doc_label.pack(anchor='w')
        
        # Section Références Scientifiques
        ref_frame = ttk.LabelFrame(scrollable_frame, text="🔬 Références Scientifiques", padding=15)
        ref_frame.pack(fill=tk.X, pady=(0, 15))
        
        ref_text = """
📚 RÉFÉRENCES PRINCIPALES :

🧮 ENTROPIE RENORMÉE (ISSTARTEL) :
• Issartel, J.-P. (2007). "Renormalized entropy and complexity measures of electroencephalographic time series." 
  Proceedings of the Royal Society A, 463(2087), 2647-2661. DOI: 10.1098/rspa.2007.1877

• Issartel, J.-P. (2011). "Renormalized entropy and complexity measures of electroencephalographic time series: 
  Application to sleep analysis." Pure and Applied Geophysics, 168(12), 2209-2223. 
  DOI: 10.1007/s00024-011-0381-4

🔬 ANALYSE EEG ET NEUROSCIENCES :
• Niedermeyer, E., & da Silva, F. L. (2005). Electroencephalography: basic principles, clinical applications, 
  and related fields. Lippincott Williams & Wilkins.

• Nunez, P. L., & Srinivasan, R. (2006). Electric fields of the brain: the neurophysics of EEG. Oxford University Press.

• Michel, C. M., & Murray, M. M. (2012). Towards the utilization of EEG as a brain imaging tool. 
  NeuroImage, 61(2), 371-385.

⚡ DÉTECTION D'ARTEFACTS :
• Mognon, A., Jovicich, J., Bruzzone, L., & Buiatti, M. (2011). ADJUST: An automatic EEG artifact detector 
  based on the joint use of spatial and temporal features. Psychophysiology, 48(2), 229-240.

• Winkler, I., Haufe, S., & Tangermann, M. (2011). Automatic classification of artifactual ICA-components 
  for artifact removal in EEG signals. Behavioral and Brain Functions, 7(1), 30.

😴 SCORING DE SOMMEIL :
• Rosenberg, R. S., & Van Hout, S. (2013). The American Academy of Sleep Medicine inter-scorer reliability 
  program: sleep stage scoring. Journal of Clinical Sleep Medicine, 9(1), 81-87.

• Lajnef, T., Chaibi, S., Ruby, P., Aguera, P. E., Eichenlaub, J. B., Samet, M., ... & Jerbi, K. (2015). 
  Learning machines and sleeping brains: automatic sleep stage classification using decision-tree multi-class 
  support vector machines. Journal of Neuroscience Methods, 250, 94-105.
        """
        
        ref_label = ttk.Label(ref_frame, text=ref_text.strip(), 
                             font=('Segoe UI', 10), justify='left')
        ref_label.pack(anchor='w')
        
        # Section Standards et Conventions
        standards_frame = ttk.LabelFrame(scrollable_frame, text="📏 Standards et Conventions", padding=15)
        standards_frame.pack(fill=tk.X, pady=(0, 15))
        
        standards_text = """
🎯 STANDARDS EEG INTERNATIONAUX :

📍 SYSTÈME 10-20 :
• Standard international pour le placement des électrodes EEG
• Définit les positions Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, etc.
• Utilisé pour la localisation anatomique des signaux

📍 SYSTÈME 10-10 :
• Extension du système 10-20 avec plus d'électrodes
• Permet une résolution spatiale plus fine
• Standard pour les études de haute densité

📊 BANDES FRÉQUENTIELLES EEG :
• Delta : 0.5-4 Hz (sommeil profond, coma)
• Theta : 4-8 Hz (somnolence, méditation, mémoire)
• Alpha : 8-12 Hz (relaxation, yeux fermés)
• Beta : 12-30 Hz (activité cognitive, attention)
• Gamma : 30-45 Hz (traitement d'information, conscience)

🔧 PARAMÈTRES RECOMMANDÉS :
• Fréquence d'échantillonnage : ≥ 200 Hz pour EEG clinique
• Filtrage : 0.5-30 Hz pour EEG standard
• Impédance des électrodes : < 5 kΩ
• Durée d'enregistrement : selon protocole (minimum 20 min)
        """
        
        standards_label = ttk.Label(standards_frame, text=standards_text.strip(), 
                                   font=('Segoe UI', 10), justify='left')
        standards_label.pack(anchor='w')
        
        # Section Support Technique
        support_frame = ttk.LabelFrame(scrollable_frame, text="🆘 Support Technique", padding=15)
        support_frame.pack(fill=tk.X, pady=(0, 15))
        
        support_text = """
📞 CONTACT ET SUPPORT :

👨‍💻 DÉVELOPPEMENT :
• Auteur : Côme Barmoy
• Unité Neuropsychologie du Stress (IRBA)
• Email : come1.barmoy@supbiotech.fr
• Version : CESA v0.0beta1.0.0
• Licence : MIT

🔧 DIAGNOSTIC AUTOMATIQUE :
• Menu Aide → Diagnostic Système
• Vérification automatique des dépendances
• Test des fonctionnalités principales
• Rapport détaillé des problèmes

📚 RESSOURCES ADDITIONNELLES :
• Documentation MNE-Python : https://mne.tools/
• Standards EEG : American Clinical Neurophysiology Society
• Guidelines AASM : American Academy of Sleep Medicine
• IRBA : Institut de Recherche Biomédicale des Armées

💡 CONSEILS DE DÉPANNAGE :
• Vérifiez que Python 3.8+ est installé
• Installez toutes les dépendances via requirements.txt
• Utilisez le diagnostic automatique intégré
• Consultez les logs d'erreur pour plus de détails
• Contactez le support avec des informations précises
        """
        
        support_label = ttk.Label(support_frame, text=support_text.strip(), 
                                 font=('Segoe UI', 10), justify='left')
        support_label.pack(anchor='w')
        
        # Boutons d'action
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, pady=(20, 0))
        
        def open_readme():
            try:
                readme_path = Path("README.md")
                if readme_path.exists():
                    webbrowser.open(f"file://{readme_path.absolute()}")
                else:
                    messagebox.showinfo("Information", "Fichier README.md non trouvé.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'ouverture :\n{e}")
        
        def run_diagnostic():
            reference_window.destroy()
            self._run_diagnostic()
        
        def open_support():
            reference_window.destroy()
            self._open_support()
        
        ttk.Button(action_frame, text="📖 Ouvrir README.md", 
                  command=open_readme).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="🔧 Diagnostic Système", 
                  command=run_diagnostic).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="📞 Support Technique", 
                  command=open_support).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="Fermer", 
                  command=reference_window.destroy).pack(side=tk.RIGHT)
    
    def _run_diagnostic(self):
        """Lance un diagnostic automatique du système."""
        diagnostic_window = tk.Toplevel(self.parent_app.root)
        diagnostic_window.title("🔧 Diagnostic Système CESA v0.0beta1.0")
        diagnostic_window.geometry("600x500")
        diagnostic_window.transient(self.parent_app.root)
        diagnostic_window.grab_set()
        
        # Frame principal
        main_frame = ttk.Frame(diagnostic_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="🔧 Diagnostic Système", 
                               font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Zone de résultats
        results_frame = ttk.LabelFrame(main_frame, text="Résultats du diagnostic", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        results_text = tk.Text(results_frame, wrap=tk.WORD, font=('Consolas', 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_text.yview)
        results_text.configure(yscrollcommand=results_scrollbar.set)
        
        results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Lancer le diagnostic
        def run_diagnostic():
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, "🔍 Diagnostic en cours...\n\n")
            diagnostic_window.update()
            
            try:
                # Diagnostic Python
                results_text.insert(tk.END, "=== DIAGNOSTIC PYTHON ===\n")
                import sys
                results_text.insert(tk.END, f"Version Python: {sys.version}\n")
                results_text.insert(tk.END, f"Exécutable: {sys.executable}\n")
                results_text.insert(tk.END, f"Plateforme: {sys.platform}\n\n")
                
                # Diagnostic des dépendances
                results_text.insert(tk.END, "=== DIAGNOSTIC DÉPENDANCES ===\n")
                dependencies = ['mne', 'numpy', 'matplotlib', 'scipy', 'pandas', 'yasa', 'PIL']
                
                for dep in dependencies:
                    try:
                        if dep == 'PIL':
                            import PIL
                            version = PIL.__version__
                        else:
                            module = __import__(dep)
                            version = getattr(module, '__version__', 'Inconnue')
                        results_text.insert(tk.END, f"✅ {dep}: {version}\n")
                    except ImportError:
                        results_text.insert(tk.END, f"❌ {dep}: Non installé\n")
                
                results_text.insert(tk.END, "\n")
                
                # Diagnostic des fichiers CESA
                results_text.insert(tk.END, "=== DIAGNOSTIC FICHIERS CESA ===\n")
                esa_files = ['eeg_studio_fixed.py', 'run.py', 'esa/entropy.py', 'requirements.txt']
                
                for file in esa_files:
                    if Path(file).exists():
                        results_text.insert(tk.END, f"✅ {file}: Présent\n")
                    else:
                        results_text.insert(tk.END, f"❌ {file}: Manquant\n")
                
                results_text.insert(tk.END, "\n")
                
                # Test de l'entropie renormée
                results_text.insert(tk.END, "=== TEST ENTROPIE RENORMÉE ===\n")
                try:
                    from esa.entropy import compute_renormalized_entropy, RenormalizedEntropyConfig
                    import numpy as np
                    
                    # Test rapide
                    data = np.array([np.random.randn(100), np.random.randn(100)])
                    config = RenormalizedEntropyConfig(window_length=1.0, overlap=0.5)
                    result = compute_renormalized_entropy(data, 100.0, ["C3", "C4"], config)
                    
                    results_text.insert(tk.END, f"✅ Entropie renormée: {result.entropy_nats:.6f} nats\n")
                    results_text.insert(tk.END, "✅ Module d'entropie fonctionnel\n")
                    
                except Exception as e:
                    results_text.insert(tk.END, f"❌ Erreur entropie renormée: {e}\n")
                
                results_text.insert(tk.END, "\n🎉 Diagnostic terminé!\n")
                
            except Exception as e:
                results_text.insert(tk.END, f"❌ Erreur lors du diagnostic: {e}\n")
        
        # Boutons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, text="🔍 Lancer le diagnostic", 
                  command=run_diagnostic).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Fermer", 
                  command=diagnostic_window.destroy).pack(side=tk.RIGHT)
        
        # Lancer automatiquement le diagnostic
        diagnostic_window.after(100, run_diagnostic)

# Fonction utilitaire pour créer des tooltips
def create_detailed_tooltip(widget, text, delay=500):
    """Crée un tooltip détaillé pour un widget."""
    def show_tooltip(event):
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        label = tk.Label(tooltip, text=text, justify='left', 
                       background='#ffffe0', relief='solid', borderwidth=1,
                       font=('Segoe UI', 9), wraplength=300, padx=5, pady=5)
        label.pack()
        
        widget.tooltip = tooltip
        
        # Auto-fermeture après 5 secondes
        tooltip.after(5000, lambda: hide_tooltip(event))
    
    def hide_tooltip(event):
        if hasattr(widget, 'tooltip') and widget.tooltip:
            widget.tooltip.destroy()
            widget.tooltip = None
    
    widget.bind('<Enter>', lambda e: widget.after(delay, lambda: show_tooltip(e)))
    widget.bind('<Leave>', hide_tooltip)
    widget.bind('<Button-1>', hide_tooltip)


