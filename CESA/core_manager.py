# CESA/core_manager.py
"""Gestionnaire central qui coordonne tous les modules existants"""

from typing import Dict, List, Optional, Any
import logging
import tkinter as tk

# Import des modules existants
from ui.main_interface import MainInterfaceManager

from CESA.psg_plot import PSGPlotter

class CoreManager:
    """Coordonnateur central pour tous les modules CESA"""
    
    def __init__(self, parent_app):
        self.app = parent_app
        self.root = parent_app.root
        
        # Gestionnaires
        self.interface_manager = None
        self.plot_manager = None
        self.data_manager = None
    
    def initialize_all_modules(self):
        """Initialise tous les modules dans le bon ordre"""
        try:
            # 1. Interface principale (avec modules ui existants)
            self.interface_manager = MainInterfaceManager(self.app)
            
            # 2. Gestionnaire de graphiques (utilise psg_plot.py)
            self._initialize_plot_manager()
            
            # 3. Gestionnaire de données (utilise les fichiers existants)
            self._initialize_data_manager()
            
            logging.info("CORE: All modules initialized successfully")
            
        except Exception as e:
            logging.error(f"CORE: Module initialization failed: {e}")
            
    def _initialize_plot_manager(self):
        """Initialise le gestionnaire de graphiques avec psg_plot.py existant"""
        try:
            # Réutiliser psg_plot.py existant
            if hasattr(self.app, 'psg_plot'):
                self.plot_manager = self.app.psg_plot
            else:
                logging.warning("psg_plot.py not found, using fallback")
        except Exception as e:
            logging.error(f"Plot manager init error: {e}")
    
    def _initialize_data_manager(self):
        """Initialise le gestionnaire de données"""
        try:
            # Utiliser les modules existants pour la gestion des données
            self.data_manager = {
                'spectral': getattr(self.app, 'spectral_analysis', None),
                'user_assistant': getattr(self.app, 'user_assistant', None),
                'theme': getattr(self.app, 'theme_manager', None)
            }
        except Exception as e:
            logging.error(f"Data manager init error: {e}")
