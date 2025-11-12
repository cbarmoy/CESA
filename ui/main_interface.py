# ui/main_interface.py
"""Interface principale modulaire - Complète les modules UI existants"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Any
import logging

# Import des modules ui existants
from ui.shortcuts_dialog import ShortcutsDialog
from ui.filter_config_dialog import FilterConfigDialog  
from ui.report_dialog import ReportDialog

class MainInterfaceManager:
    """Gestionnaire principal de l'interface - Coordonne les modules ui/ existants"""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.root = parent_app.root
        
        # Références vers les modules ui existants
        self.shortcuts_dialog = ShortcutsDialog(parent_app)
        self.filter_config_dialog = FilterConfigDialog(parent_app)
        self.report_dialog = ReportDialog(parent_app)
    
    def create_main_layout(self):
        """Crée la mise en page principale avec les optimisations"""
        # Configuration moderne
        self._setup_modern_theme()
        
        # Conteneur principal
        self.main_container = ttk.Frame(self.root, style="Modern.TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Zone graphique (utilise psg_plot.py existant)
        self.plot_container = ttk.Frame(self.main_container)
        self.plot_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Panneau de contrôle optimisé
        self.control_container = ttk.Frame(self.main_container, width=350)
        self.control_container.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_container.pack_propagate(False)
        
        self._create_optimized_controls()
    
    def _setup_modern_theme(self):
        """Configure le thème moderne (complète theme_manager.py)"""
        try:
            from CESA.theme_manager import theme_manager as tm
            tm.apply_theme_to_root(self.root)
            
            # Styles additionnels pour les modules ui
            style = ttk.Style()
            style.configure('UI.Modern.TFrame', 
                          background='#f8f9fa', 
                          relief='flat')
            style.configure('UI.Modern.TButton',
                          background='#007bff',
                          foreground='white',
                          font=('Helvetica', 9, 'bold'))
                          
        except Exception as e:
            logging.warning(f"Theme setup failed: {e}")
    
    def _create_optimized_controls(self):
        """Crée les contrôles optimisés avec les modules existants"""
        
        # Navigation temporelle avec événements
        nav_frame = ttk.LabelFrame(self.control_container, 
                                  text="🕒 Navigation", 
                                  padding=15,
                                  style="Modern.TLabelframe")
        nav_frame.pack(fill=tk.X, pady=5)
        
        # Temps actuel avec validation
        time_container = ttk.Frame(nav_frame)
        time_container.pack(fill=tk.X)
        
        ttk.Label(time_container, text="Temps (s):", 
                 style="Modern.TLabel").pack(side=tk.LEFT)
        
        self.time_var = tk.StringVar(value=str(self.app.current_time))
        time_entry = ttk.Entry(time_container, 
                              textvariable=self.time_var, 
                              width=12,
                              font=('Consolas', 10))
        time_entry.pack(side=tk.RIGHT)
        time_entry.bind('<Return>', self._on_time_change)
        
        # Boutons de navigation avec icônes
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill=tk.X, pady=10)
        
        ttk.Button(nav_buttons, text="⏮️ -30s", 
                  command=self._nav_previous_epoch,
                  style="Modern.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(nav_buttons, text="⏭️ +30s", 
                  command=self._nav_next_epoch,
                  style="Modern.TButton").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # Sélection de canaux (utilise channel_selector.py existant)
        channels_frame = ttk.LabelFrame(self.control_container, 
                                      text="📊 Canaux", 
                                      padding=15)
        channels_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(channels_frame, 
                  text="🔧 Sélectionner Canaux",
                  command=self._show_channel_selector,
                  style="Modern.TButton").pack(fill=tk.X, pady=2)
        
        # Label pour afficher les canaux sélectionnés
        self.selected_channels_label = ttk.Label(channels_frame, 
                                               text="Aucun canal sélectionné",
                                               style="Modern.TLabel",
                                               font=('Helvetica', 8))
        self.selected_channels_label.pack(anchor=tk.W, pady=5)
        
        # Configuration des filtres (utilise le module existant)
        filter_frame = ttk.LabelFrame(self.control_container, 
                                    text="🔍 Filtres", 
                                    padding=15)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(filter_frame,
                  text="⚙️ Configuration Filtres",
                  command=self.filter_config_dialog.show_filter_config,
                  style="Modern.TButton").pack(fill=tk.X, pady=2)
        
        # Actions rapides  
        actions_frame = ttk.LabelFrame(self.control_container, 
                                     text="🚀 Actions", 
                                     padding=15)
        actions_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(actions_frame,
                  text="⌨️ Raccourcis Clavier",
                  command=self.shortcuts_dialog.show_shortcuts,
                  style="Modern.TButton").pack(fill=tk.X, pady=2)
        
        ttk.Button(actions_frame,
                  text="📋 Générer Rapport",
                  command=self.report_dialog.generate_report,
                  style="Modern.TButton").pack(fill=tk.X, pady=2)
    
    def _nav_previous_epoch(self):
        """Navigation époque précédente avec événements"""
        try:
            from CESA.event_system import event_bus, Events, EventData
            
            # Navigation
            old_time = self.app.current_time
            self.app.current_time = max(0, self.app.current_time - 30.0)
            self.time_var.set(str(self.app.current_time))
            
            # Émettre événement
            event_data = EventData.TimeChanged(
                current_time=self.app.current_time,
                duration=getattr(self.app, 'duration', 30.0)
            )
            event_bus.emit(Events.TIME_CHANGED, event_data, throttle=0.05)
            
            # Mise à jour
            self.app.update_plot()
            logging.debug(f"Navigation: {old_time} → {self.app.current_time}")
            
        except Exception as e:
            logging.error(f"Navigation error: {e}")
    
    def _nav_next_epoch(self):
        """Navigation époque suivante avec événements"""
        try:
            from CESA.event_system import event_bus, Events, EventData
            
            # Calculer le temps max
            if hasattr(self.app, 'raw') and self.app.raw:
                max_time = len(self.app.raw.times) / self.app.sfreq - getattr(self.app, 'duration', 30.0)
            else:
                max_time = 1000.0
            
            # Navigation
            old_time = self.app.current_time
            self.app.current_time = min(max_time, self.app.current_time + 30.0)
            self.time_var.set(str(self.app.current_time))
            
            # Émettre événement
            event_data = EventData.TimeChanged(
                current_time=self.app.current_time,
                duration=getattr(self.app, 'duration', 30.0)
            )
            event_bus.emit(Events.TIME_CHANGED, event_data, throttle=0.05)
            
            # Mise à jour
            self.app.update_plot()
            logging.debug(f"Navigation: {old_time} → {self.app.current_time}")
            
        except Exception as e:
            logging.error(f"Navigation error: {e}")
    
    def _on_time_change(self, event):
        """Changement de temps via entry"""
        try:
            new_time = float(self.time_var.get())
            self.app.current_time = max(0, new_time)
            
            # Émettre événement si disponible
            try:
                from CESA.event_system import event_bus, Events, EventData
                event_data = EventData.TimeChanged(
                    current_time=self.app.current_time,
                    duration=getattr(self.app, 'duration', 30.0)
                )
                event_bus.emit(Events.TIME_CHANGED, event_data, throttle=0.05)
            except:
                pass
                
            self.app.update_plot()
            
        except ValueError:
            self.time_var.set(str(self.app.current_time))
    
    def _show_channel_selector(self):
        """Affiche le sélecteur de canaux (utilise le fichier existant)"""
        if not hasattr(self.app, 'raw') or not self.app.raw:
            tk.messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier EDF")
            return
        
        try:
            # Utiliser le channel_selector.py existant
            if hasattr(self.app, 'show_channel_selector'):
                self.app.show_channel_selector()
            else:
                tk.messagebox.showinfo("Sélecteur", "Module de sélection en cours d'implémentation")
        except Exception as e:
            logging.error(f"Channel selector error: {e}")
    
    def update_selected_channels_display(self):
        """Met à jour l'affichage des canaux sélectionnés"""
        try:
            if hasattr(self.app, 'selected_channels') and self.app.selected_channels:
                count = len(self.app.selected_channels)
                if count <= 3:
                    channels_text = ", ".join(self.app.selected_channels)
                else:
                    channels_text = f"{', '.join(self.app.selected_channels[:3])}... (+{count-3})"
                
                self.selected_channels_label.config(
                    text=f"{count} canaux: {channels_text}")
            else:
                self.selected_channels_label.config(text="Aucun canal sélectionné")
        except Exception as e:
            logging.error(f"Update channels display error: {e}")
