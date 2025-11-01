"""
Interface de sélection des canaux pour la construction de la pyramide multiscale.

Permet à l'utilisateur de choisir quels canaux inclure avant la construction.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Dict, Set


class ChannelSelector:
    """Dialogue de sélection des canaux pour la pyramide multiscale."""
    
    def __init__(self, parent: tk.Tk, available_channels: List[str], preselected_channels: List[str] | None = None):
        """
        Initialise le sélecteur de canaux.
        
        Args:
            parent: Fenêtre parent Tkinter
            available_channels: Liste de tous les canaux disponibles
            preselected_channels: Canaux pré-sélectionnés (optionnel)
        """
        self.parent = parent
        self.available_channels = sorted(available_channels)
        self.preselected_channels = preselected_channels or []
        self.selected_channels: List[str] = []
        self.cancelled = False
        
        # Détection automatique des types de canaux
        self.channel_types = self._detect_channel_types()
    
    def _detect_channel_types(self) -> Dict[str, List[str]]:
        """Détecte automatiquement les types de canaux (EEG, EOG, EMG, ECG, etc.)."""
        types: Dict[str, List[str]] = {
            'EEG': [],
            'EOG': [],
            'EMG': [],
            'ECG': [],
            'Respiration': [],
            'SpO2': [],
            'Autres': []
        }
        
        for ch in self.available_channels:
            ch_lower = ch.lower()
            
            # EEG (électrodes crâniennes)
            if any(eeg in ch_lower for eeg in ['f3', 'f4', 'c3', 'c4', 'o1', 'o2', 'p3', 'p4', 't3', 't4', 'fp1', 'fp2', 'fz', 'cz', 'pz', '-m1', '-m2', '-a1', '-a2']):
                types['EEG'].append(ch)
            # EOG (mouvements oculaires)
            elif any(eog in ch_lower for eog in ['e1', 'e2', 'eog', 'loc', 'roc', 'eye']):
                types['EOG'].append(ch)
            # EMG (activité musculaire)
            elif any(emg in ch_lower for emg in ['leg', 'chin', 'emg', 'jambe', 'menton']):
                types['EMG'].append(ch)
            # ECG (activité cardiaque)
            elif any(ecg in ch_lower for ecg in ['ecg', 'heart', 'cardiac', 'coeur', 'rate']):
                types['ECG'].append(ch)
            # Respiration
            elif any(resp in ch_lower for resp in ['flow', 'airflow', 'nasal', 'abdomen', 'chest', 'thorax', 'resp', 'snor']):
                types['Respiration'].append(ch)
            # SpO2 et saturation
            elif any(spo in ch_lower for spo in ['spo2', 'sat', 'pulse', 'oxygen']):
                types['SpO2'].append(ch)
            else:
                types['Autres'].append(ch)
        
        # Supprimer les catégories vides
        return {k: v for k, v in types.items() if v}
    
    def show_dialog(self) -> Optional[List[str]]:
        """
        Affiche le dialogue de sélection et retourne la liste des canaux sélectionnés.
        
        Returns:
            Liste des canaux sélectionnés, ou None si annulé
        """
        dialog = tk.Toplevel(self.parent)
        dialog.title("🔧 Sélection des Canaux")
        dialog.geometry("750x700")
        dialog.resizable(True, True)
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Centrer la fenêtre avec une hauteur suffisante
        dialog.update_idletasks()
        width = 750
        height = 700  # Augmenté de 600 à 700 pour avoir l'espace pour les boutons
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre et description
        title_label = ttk.Label(
            main_frame,
            text="🔧 Sélection des Canaux pour la Pyramide",
            font=('Segoe UI', 14, 'bold')
        )
        title_label.pack(pady=(0, 5))
        
        desc_label = ttk.Label(
            main_frame,
            text="Choisissez les canaux à inclure dans le fichier de navigation rapide.\n"
                 "Plus vous sélectionnez de canaux, plus la construction sera longue.",
            font=('Segoe UI', 9),
            foreground='gray'
        )
        desc_label.pack(pady=(0, 15))
        
        # Frame pour les statistiques
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        total_label = ttk.Label(stats_frame, text=f"📊 Total disponible : {len(self.available_channels)} canaux")
        total_label.pack(side=tk.LEFT)
        
        selected_count_var = tk.StringVar(value=f"✅ Sélectionnés : 0 canaux")
        selected_label = ttk.Label(stats_frame, textvariable=selected_count_var, foreground='green')
        selected_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Frame pour les boutons de sélection rapide
        quick_frame = ttk.LabelFrame(main_frame, text="⚡ Sélection Rapide", padding="10")
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        button_row = ttk.Frame(quick_frame)
        button_row.pack(fill=tk.X)
        
        # Variables pour stocker les checkbuttons
        checkbuttons: Dict[str, tk.BooleanVar] = {}
        
        # Label pour le temps estimé (sera utilisé par update_count)
        time_label = ttk.Label(main_frame, text="", font=('Segoe UI', 9, 'italic'))
        
        # Définir update_count AVANT toutes les fonctions qui l'utilisent
        def update_count():
            count = sum(1 for var in checkbuttons.values() if var.get())
            selected_count_var.set(f"✅ Sélectionnés : {count} canaux")
            
            # Estimation du temps
            if count > 0:
                estimated_time = count * 15 / 90  # Base : 15 min pour 90 canaux
                time_label.config(
                    text=f"⏱️  Temps estimé de construction : ~{estimated_time:.1f} minutes",
                    foreground='orange' if estimated_time > 5 else 'green'
                )
            else:
                time_label.config(text="⚠️  Aucun canal sélectionné", foreground='red')
        
        def select_all():
            for var in checkbuttons.values():
                var.set(True)
            update_count()
        
        def select_none():
            for var in checkbuttons.values():
                var.set(False)
            update_count()
        
        def select_preselected():
            for ch, var in checkbuttons.items():
                var.set(ch in self.preselected_channels)
            update_count()
        
        def select_type(channel_type: str):
            channels_of_type = self.channel_types.get(channel_type, [])
            for ch, var in checkbuttons.items():
                if ch in channels_of_type:
                    var.set(True)
            update_count()
        
        ttk.Button(button_row, text="✅ Tout sélectionner", command=select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row, text="❌ Tout désélectionner", command=select_none).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row, text="⭐ Pré-sélection UI", command=select_preselected).pack(side=tk.LEFT, padx=(0, 5))
        
        # Boutons par type
        type_row = ttk.Frame(quick_frame)
        type_row.pack(fill=tk.X, pady=(5, 0))
        
        for channel_type in self.channel_types.keys():
            ttk.Button(
                type_row,
                text=f"{channel_type} ({len(self.channel_types[channel_type])})",
                command=lambda t=channel_type: select_type(t)
            ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Frame scrollable pour la liste des canaux
        list_frame = ttk.LabelFrame(main_frame, text="📋 Liste des Canaux", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas + Scrollbar
        canvas = tk.Canvas(list_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Créer les checkbuttons par type
        for channel_type, channels in self.channel_types.items():
            if not channels:
                continue
            
            # Titre du groupe
            type_label = ttk.Label(
                scrollable_frame,
                text=f"━━━ {channel_type} ({len(channels)} canaux) ━━━",
                font=('Segoe UI', 10, 'bold'),
                foreground='navy'
            )
            type_label.pack(anchor=tk.W, pady=(10, 5))
            
            # Checkbuttons pour chaque canal
            for ch in channels:
                var = tk.BooleanVar(value=(ch in self.preselected_channels))
                checkbuttons[ch] = var
                
                cb = ttk.Checkbutton(
                    scrollable_frame,
                    text=ch,
                    variable=var,
                    command=update_count
                )
                cb.pack(anchor=tk.W, padx=(20, 0), pady=1)
        
        # Placer le label du temps estimé (déjà créé plus haut)
        time_label.pack(pady=(5, 10))
        
        # Frame pour les boutons d'action
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        def confirm():
            selected = [ch for ch, var in checkbuttons.items() if var.get()]
            
            if not selected:
                messagebox.showwarning(
                    "Aucun canal sélectionné",
                    "Veuillez sélectionner au moins un canal pour construire la pyramide.",
                    parent=dialog
                )
                return
            
            self.selected_channels = selected
            self.cancelled = False
            dialog.destroy()
        
        def cancel():
            self.cancelled = True
            dialog.destroy()
        
        ttk.Button(action_frame, text="❌ Annuler", command=cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(action_frame, text="✅ Confirmer", command=confirm).pack(side=tk.RIGHT)
        
        # Initialiser le compteur
        update_count()
        
        # Attendre la fermeture du dialogue
        dialog.wait_window()
        
        if self.cancelled:
            return None
        
        return self.selected_channels
    
    def get_selection(self) -> Optional[List[str]]:
        """
        Méthode publique pour obtenir la sélection.
        
        Returns:
            Liste des canaux sélectionnés, ou None si annulé
        """
        return self.show_dialog()

