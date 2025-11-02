# CESA/data_optimizer.py
"""Optimiseur de données pour CESA"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps

class DataOptimizer:
    """Optimiseur pour les données EEG"""
    
    def __init__(self):
        self.downsample_cache = {}
        self.filter_cache = {}
        
    def optimize_for_display(self, data: np.ndarray, target_points: int = 2000) -> np.ndarray:
        """Optimise les données pour l'affichage en réduisant les points"""
        if len(data) <= target_points:
            return data
            
        # Downsampling intelligent
        step = len(data) // target_points
        if step <= 1:
            return data
            
        # Conserver les pics et vallées importantes
        optimized = self._intelligent_downsample(data, step, target_points)
        logging.debug(f"DATA: Optimized from {len(data)} to {len(optimized)} points")
        return optimized
    
    def _intelligent_downsample(self, data: np.ndarray, step: int, target_points: int) -> np.ndarray:
        """Downsampling intelligent qui préserve les caractéristiques importantes"""
        try:
            # Méthode 1: Downsampling avec conservation des extrema locaux
            indices = []
            for i in range(0, len(data), step):
                end_idx = min(i + step, len(data))
                segment = data[i:end_idx]
                
                if len(segment) == 0:
                    continue
                    
                # Trouver min et max dans le segment
                local_min_idx = np.argmin(segment) + i
                local_max_idx = np.argmax(segment) + i
                
                # Ajouter les indices (sans doublons)
                for idx in sorted([local_min_idx, local_max_idx]):
                    if not indices or idx != indices[-1]:
                        indices.append(idx)
            
            # Si on a trop de points, prendre un échantillonnage uniforme
            if len(indices) > target_points:
                step_final = len(indices) // target_points
                indices = indices[::step_final]
            
            return data[indices] if indices else data[::step]
            
        except Exception as e:
            logging.error(f"DATA: Intelligent downsample failed: {e}")
            # Fallback: downsampling simple
            return data[::step]
    
    def prepare_channel_data(self, raw_data: Dict[str, np.ndarray], 
                           selected_channels: List[str], 
                           time_range: Tuple[float, float],
                           sfreq: float) -> Dict[str, np.ndarray]:
        """Prépare les données des canaux pour l'affichage"""
        start_time, end_time = time_range
        start_idx = int(start_time * sfreq)
        end_idx = int(end_time * sfreq)
        
        optimized_data = {}
        
        for channel in selected_channels:
            if channel not in raw_data:
                continue
                
            # Extraire la portion de données
            channel_data = raw_data[channel][start_idx:end_idx]
            
            # Optimiser pour l'affichage
            optimized_data[channel] = self.optimize_for_display(channel_data)
        
        return optimized_data
    
    def estimate_data_size(self, data: Any) -> int:
        """Estime la taille d'un objet de données"""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, (list, tuple)):
                return sum(self.estimate_data_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self.estimate_data_size(k) + self.estimate_data_size(v) 
                          for k, v in data.items())
            else:
                return sys.getsizeof(data)
        except:
            return 0
    
    def compress_for_cache(self, data: np.ndarray, compression_ratio: float = 0.8) -> np.ndarray:
        """Compresse les données pour le cache"""
        if len(data) == 0:
            return data
            
        target_length = int(len(data) * compression_ratio)
        if target_length >= len(data):
            return data
            
        # Compression par moyennage par blocs
        block_size = len(data) // target_length
        compressed = []
        
        for i in range(0, len(data), block_size):
            block = data[i:i+block_size]
            compressed.append(np.mean(block))
        
        return np.array(compressed[:target_length])

# Instance globale
data_optimizer = DataOptimizer()

# Décorateur pour l'optimisation automatique
def optimize_data(func):
    """Décorateur pour optimiser automatiquement les données"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Appeler la fonction originale
        result = func(*args, **kwargs)
        
        # Optimiser le résultat si c'est un array numpy
        if isinstance(result, np.ndarray) and len(result) > 5000:
            optimized = data_optimizer.optimize_for_display(result)
            logging.debug(f"DATA: Auto-optimized {func.__name__} output")
            return optimized
        elif isinstance(result, dict):
            # Optimiser chaque array dans le dictionnaire
            for key, value in result.items():
                if isinstance(value, np.ndarray) and len(value) > 5000:
                    result[key] = data_optimizer.optimize_for_display(value)
            
        return result
    return wrapper
