"""
Data Bridge pour la navigation rapide CESA
Module de pont entre les données Zarr et l'interface
"""

import logging
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor

class DataBridge:
    """Pont de données pour la navigation rapide avec Zarr"""
    
    def __init__(self, provider, executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialise le pont de données
        
        Args:
            provider: Fournisseur de données pré-calculées (PrecomputedProvider)
            executor: Executeur pour les tâches asynchrones
        """
        self.provider = provider
        self.executor = executor or ThreadPoolExecutor(max_workers=2)
        logging.info("DataBridge initialisé avec provider et executor")
    
    def get_data(self, channel: str, start_time: float, duration: float, **kwargs) -> Any:
        """Récupère les données pour un canal donné"""
        try:
            # Essayer différentes méthodes du provider
            if hasattr(self.provider, 'get_channel_data'):
                return self.provider.get_channel_data(channel, start_time, duration, **kwargs)
            elif hasattr(self.provider, 'get_data'):
                return self.provider.get_data(channel, start_time, duration, **kwargs)
            elif hasattr(self.provider, 'fetch_data'):
                return self.provider.fetch_data(channel, start_time, duration, **kwargs)
            else:
                logging.warning(f"Provider {type(self.provider)} ne supporte aucune méthode get_data connue")
                return None
        except Exception as e:
            logging.error(f"Erreur récupération données {channel}: {e}")
            return None

    
    def get_channels(self) -> list:
        """Retourne la liste des canaux disponibles"""
        try:
            if hasattr(self.provider, 'channels'):
                return self.provider.channels
            elif hasattr(self.provider, 'get_channels'):
                return self.provider.get_channels()
            else:
                return []
        except Exception as e:
            logging.error(f"Erreur récupération canaux: {e}")
            return []
    
    def close(self):
        """Ferme le pont de données et libère les ressources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logging.info("DataBridge fermé proprement")
        except Exception as e:
            logging.error(f"Erreur fermeture DataBridge: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_signals_for_window(self, start_time: float, duration: float, channels: list = None, **kwargs) -> dict:
        """
        Récupère les signaux pour une fenêtre temporelle donnée
        Méthode appelée par l'interface graphique pour la navigation rapide
        
        Args:
            start_time: Temps de début (secondes)
            duration: Durée de la fenêtre (secondes)  
            channels: Liste des canaux à récupérer
            
        Returns:
            Dict avec les données des canaux {canal: données}
        """
        try:
            if not channels:
                channels = self.get_channels()
                
            signals = {}
            for channel in channels:
                try:
                    data = self.get_data(channel, start_time, duration, **kwargs)
                    if data is not None:
                        signals[channel] = data
                        
                except Exception as e:
                    logging.warning(f"Erreur récupération canal {channel}: {e}")
                    
            logging.debug(f"✅ get_signals_for_window: récupéré {len(signals)}/{len(channels)} signaux pour {start_time:.1f}s-{start_time+duration:.1f}s")
            return signals
            
        except Exception as e:
            logging.error(f"❌ Erreur get_signals_for_window: {e}")
            return {}

