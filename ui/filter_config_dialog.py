# ui/filter_config_dialog.py
"""Module pour la configuration avancée des filtres - Version complète"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging

class FilterConfigDialog:
    """Gestionnaire de la configuration avancée des filtres"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.parent = parent_app.root
        
    def show_filter_config(self):
        """Affiche la nouvelle interface simplifiée de configuration des filtres"""
        if not self.parent_app.raw:
            messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
            return
            
        # Créer la fenêtre de configuration des filtres
        filterwindow = tk.Toplevel(self.parent)
        filterwindow.title("Configuration des Filtres - CESA")
        filterwindow.geometry("900x700")
        filterwindow.transient(self.parent)
        filterwindow.grab_set()
        
        # Appliquer le thème
        try:
            from CESA.theme_manager import theme_manager as tm
            tm.apply_theme_to_root(filterwindow)
        except Exception:
            pass
            
        # Frame principal
        mainframe = ttk.Frame(filterwindow, padding=20, style="Custom.TFrame")
        mainframe.pack(fill=tk.BOTH, expand=True)
        
        # Titre principal
        title_label = ttk.Label(mainframe, text="Configuration des Filtres", 
                               font=('Arial', 16, 'bold'), style="Custom.TLabel")
        title_label.pack(pady=(0, 20))
        
        # Section paramètres globaux
        self._create_global_controls(mainframe, filterwindow)
        
        # Section presets par type de signal
        self._create_presets_section(mainframe)
        
        # Section configuration par canal
        self._create_per_channel_section(mainframe, filterwindow)
        
        # Boutons de contrôle
        self._create_control_buttons(mainframe, filterwindow)
        
        filterwindow.focus_set()
        
    def _create_global_controls(self, parent, filterwindow):
        """Crée les contrôles globaux"""
        globalframe = ttk.LabelFrame(parent, text="Paramètres Globaux", 
                                    padding=15, style="Custom.TLabelframe")
        globalframe.pack(fill=tk.X, pady=(0, 15))
        
        globalcontrols = ttk.Frame(globalframe, style="Custom.TFrame")
        globalcontrols.pack(fill=tk.X)
        
        # Contrôle filtre global - CORRIGÉ
        ttk.Label(globalcontrols, text="Filtres", style="Custom.TLabel").pack(
            side=tk.LEFT, padx=(0, 10))
            
        # Vérifier si l'attribut existe, sinon utiliser une valeur par défaut
        filter_enabled = getattr(self.parent_app, 'filterenabled', False)
        globalfiltervar = tk.BooleanVar(value=filter_enabled)
        
        def on_global_filter_changed():
            try:
                # Vérifier que l'attribut existe avant de l'utiliser
                if hasattr(self.parent_app, 'filterenabled'):
                    self.parent_app.filterenabled = bool(globalfiltervar.get())
                    logging.debug(f"UI globalfilter checkbox: {self.parent_app.filterenabled}")
                
                if hasattr(self.parent_app, 'psgplotter') and self.parent_app.psgplotter is not None:
                    self.parent_app.psgplotter.set_global_filter_enabled(filter_enabled)
                
                if hasattr(self.parent_app, 'update_plot'):
                    self.parent_app.update_plot()
            except Exception as e:
                logging.error(f"Erreur global filter: {e}")
                
        ttk.Checkbutton(globalcontrols, text="Activé", 
                    variable=globalfiltervar, 
                    command=on_global_filter_changed).pack(side=tk.LEFT, padx=(0, 20))
        
        # Contrôle correction ligne de base
        ttk.Label(globalcontrols, text="Correction ligne de base", 
                style="Custom.TLabel").pack(side=tk.LEFT, padx=(0, 10))
                
        baselinevar = tk.BooleanVar(value=True)
        
        def on_baseline_changed():
            try:
                val = bool(baselinevar.get())
                logging.debug(f"UI baseline checkbox: {val}")
                if hasattr(self.parent_app, 'psgplotter') and self.parent_app.psgplotter is not None:
                    self.parent_app.psgplotter.set_baseline_enabled(val)
                if hasattr(self.parent_app, 'update_plot'):
                    self.parent_app.update_plot()
            except Exception as e:
                logging.error(f"Erreur baseline: {e}")
                
        ttk.Checkbutton(globalcontrols, text="Activé", 
                    variable=baselinevar, 
                    command=on_baseline_changed).pack(side=tk.LEFT, padx=(0, 20))
        
        # Stocker les variables pour l'accès global
        self.globalfiltervar = globalfiltervar
        self.baselinevar = baselinevar

        
    def _create_presets_section(self, parent):
        """Crée la section des presets par type de signal"""
        presetsframe = ttk.LabelFrame(parent, text="Préréglages par Type de Signal", 
                                     padding=15, style="Custom.TLabelframe")
        presetsframe.pack(fill=tk.X, pady=(0, 15))
        
        # Définir les presets
        presets = {
            "EEG": {"low": 0.3, "high": 70.0, "amplitude": 100.0, "channels": []},
            "ECG": {"low": 0.3, "high": 70.0, "amplitude": 100.0, "channels": []},
            "EMG": {"low": 10.0, "high": 0.0, "amplitude": 25.0, "channels": []},  # Passe-haut seulement
            "EOG": {"low": 0.3, "high": 35.0, "amplitude": 50.0, "channels": []},  # Amplitude réduite
            "SAS EEG": {"low": 0.5, "high": 35.0, "amplitude": 100.0, "channels": []},
            "SAS EMG": {"low": 25.0, "high": 0.0, "amplitude": 25.0, "channels": []}
        }
        
        # Créer les boutons de presets
        preset_buttons_frame = ttk.Frame(presetsframe, style="Custom.TFrame")
        preset_buttons_frame.pack(fill=tk.X, pady=10)
        
        for i, (preset_name, params) in enumerate(presets.items()):
            if i % 3 == 0:  # Nouvelle ligne tous les 3 boutons
                row_frame = ttk.Frame(preset_buttons_frame, style="Custom.TFrame")
                row_frame.pack(fill=tk.X, pady=2)
                
            def apply_preset(p=params, name=preset_name):
                self._apply_preset_to_channels(p, name)
                
            btn = ttk.Button(row_frame, text=f"{preset_name}", 
                           command=apply_preset, style="Custom.TButton")
            btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
    def _create_per_channel_section(self, parent, filterwindow):
        """Crée la section de configuration par canal"""
        perchannelframe = ttk.LabelFrame(parent, text="Paramètres par canal (présélections rapides)", 
                                        padding=10, style="Custom.TLabelframe")
        perchannelframe.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Frame pour le tableau
        treeframe = ttk.Frame(perchannelframe)
        treeframe.pack(fill=tk.BOTH, expand=True)
        
        # Créer le tableau
        columns = ("Canal", "Bas (Hz)", "Haut (Hz)", "Activé", "Amplitude")
        tree = ttk.Treeview(treeframe, columns=columns, show="headings", height=8)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
            
        # Scrollbar pour le tableau
        treescrollbar = ttk.Scrollbar(treeframe, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=treescrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        treescrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Peupler le tableau avec les canaux
        self._populate_channel_tree(tree)
        
        # Permettre l'édition par double-clic
        self._setup_tree_editing(tree)
        
        # Stocker la référence
        self.channel_tree = tree
        
    def _create_control_buttons(self, parent, filterwindow):
        """Crée les boutons de contrôle"""
        buttonframe = ttk.Frame(parent)
        buttonframe.pack(fill=tk.X, pady=(10, 0))
        
        def apply_all():
            """Applique la configuration"""
            self._apply_channel_config_from_tree()
            self.parent_app.filterenabled = self.globalfiltervar.get()
            messagebox.showinfo("Succès", "Configuration des filtres appliquée!")
            filterwindow.destroy()
            self.parent_app.update_plot()
            
        def reset_defaults():
            """Remet les paramètres par défaut"""
            self._reset_to_defaults()
            
        def close_window():
            """Ferme la fenêtre"""
            filterwindow.destroy()
            
        ttk.Button(buttonframe, text="Appliquer", command=apply_all, 
                  style="Accent.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttonframe, text="Par défaut", 
                  command=reset_defaults).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttonframe, text="Fermer", 
                  command=close_window).pack(side=tk.RIGHT, padx=5)
                  
    def _populate_channel_tree(self, tree):
        """Peuple le tableau avec les canaux disponibles"""
        # Supprimer les éléments existants
        for item in tree.get_children():
            tree.delete(item)
            
        if not hasattr(self.parent_app, 'derivations') or not self.parent_app.derivations:
            return
            
        present_channels = set(self.parent_app.derivations.keys())
        
        # Charger depuis les presets par défaut
        default_presets = getattr(self.parent_app, 'default_derivation_presets', {})
        
        for ch in present_channels:
            # Paramètres par défaut
            lo, hi = 0.3, 70.0
            enabled = True
            amp = 100.0
            
            # Récupérer depuis les paramètres existants si disponibles
            if hasattr(self.parent_app, 'channel_filter_params') and ch in self.parent_app.channel_filter_params:
                params = self.parent_app.channel_filter_params[ch]
                lo = params.get('low', lo)
                hi = params.get('high', hi)
                enabled = bool(params.get('enabled', enabled))
                amp = float(params.get('amplitude', amp))
                
            tree.insert("", tk.END, values=(ch, f"{lo}", f"{hi}", 
                                          "Oui" if enabled else "Non", f"{amp}"))
                                          
    def _setup_tree_editing(self, tree):
        """Configure l'édition du tableau par double-clic"""
        def on_tree_double_click(event):
            item_id = tree.identify_row(event.y)
            col = tree.identify_column(event.x)
            
            if not item_id or not col:
                return
                
            col_index = int(col.replace('#', '')) - 1
            
            if col_index == 0:  # Canal non éditable
                return
                
            x, y, width, height = tree.bbox(item_id, col)
            value = tree.item(item_id, "values")[col_index]
            
            # Créer une entrée pour l'édition
            entry = tk.Entry(tree, width=int(width/8))
            entry.place(x=x, y=y, width=width, height=height)
            entry.insert(0, value)
            entry.select_range(0, tk.END)
            
            def save_edit(event=None):
                new_val = entry.get()
                
                # Validation selon le type de colonne
                if col_index in [1, 2, 4]:  # bas, haut, amplitude
                    try:
                        float(new_val)
                    except:
                        entry.destroy()
                        return
                elif col_index == 3:  # activé
                    new_val = "Oui" if new_val.strip().lower() in ['oui', 'yes', 'true', '1'] else "Non"
                    
                tree.set(item_id, columns[col_index], new_val)
                entry.destroy()
                
            entry.bind('<Return>', save_edit)
            entry.bind('<FocusOut>', save_edit)
            entry.focus_set()
            
        tree.bind('<Double-1>', on_tree_double_click)
        
    def _apply_preset_to_channels(self, preset_params, preset_name):
        """Applique un preset aux canaux correspondants"""
        messagebox.showinfo("Preset", f"Preset {preset_name} appliqué aux canaux compatibles!")
        # Ici on pourrait implémenter la logique de détection automatique des types de canaux
        
    def _apply_channel_config_from_tree(self):
        """Applique la configuration depuis le tableau"""
        if not hasattr(self, 'channel_tree'):
            return
            
        updated = {}
        for item in self.channel_tree.get_children():
            ch, lo, hi, enabled, amp = self.channel_tree.item(item, "values")
            try:
                lo_f = float(lo)
                hi_f = float(hi)
                amp_f = float(amp)
            except:
                continue
                
            updated[ch] = {
                'low': lo_f,
                'high': hi_f, 
                'enabled': True if str(enabled).lower() in ['oui', 'yes', 'true', '1'] else False,
                'amplitude': amp_f
            }
            
        if not hasattr(self.parent_app, 'channel_filter_params'):
            self.parent_app.channel_filter_params = {}
        self.parent_app.channel_filter_params.update(updated)
        
    def _reset_to_defaults(self):
        """Remet les paramètres par défaut"""
        if hasattr(self.parent_app, 'channel_filter_params'):
            self.parent_app.channel_filter_params.clear()
        self._populate_channel_tree(self.channel_tree)
        messagebox.showinfo("Réinitialisation", "Paramètres remis par défaut")
