# CESA/memory_manager.py
"""Gestionnaire de mémoire intelligent pour CESA"""

import gc
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import weakref

class LRUCache:
    """Cache LRU (Least Recently Used) thread-safe"""
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._memory_usage = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Récupère un élément du cache"""
        with self._lock:
            if key in self._cache:
                # Déplacer vers la fin (plus récemment utilisé)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value['data']
            return None
    
    def put(self, key: str, data: Any, size_bytes: int = None) -> None:
        """Ajoute un élément au cache"""
        if size_bytes is None:
            size_bytes = sys.getsizeof(data)
            
        with self._lock:
            # Supprimer si déjà existant
            if key in self._cache:
                old_size = self._cache[key]['size']
                self._memory_usage -= old_size
                del self._cache[key]
            
            # Vérifier les limites
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage + size_bytes > self.max_memory_bytes):
                if not self._cache:
                    break
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self._memory_usage -= oldest_value['size']
                logging.debug(f"CACHE: Evicted {oldest_key} (size: {oldest_value['size']} bytes)")
            
            # Ajouter le nouvel élément
            self._cache[key] = {
                'data': data,
                'size': size_bytes,
                'timestamp': time.time()
            }
            self._memory_usage += size_bytes
            logging.debug(f"CACHE: Added {key} (size: {size_bytes} bytes, total: {self._memory_usage} bytes)")
    
    def clear(self) -> None:
        """Vide le cache"""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0
            logging.debug("CACHE: Cleared all entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': getattr(self, '_hit_rate', 0.0)
            }

class MemoryManager:
    """Gestionnaire de mémoire intelligent"""
    
    def __init__(self):
        self.plot_cache = LRUCache(max_size=50, max_memory_mb=256)
        self.data_cache = LRUCache(max_size=20, max_memory_mb=512)
        self.analysis_cache = LRUCache(max_size=30, max_memory_mb=128)
        self._weak_refs = weakref.WeakSet()
        self._last_cleanup = time.time()
        self._cleanup_interval = 30  # secondes
        
    def cache_plot_data(self, key: str, plot_data: Any) -> None:
        """Cache les données de tracé"""
        try:
            self.plot_cache.put(key, plot_data)
        except Exception as e:
            logging.error(f"MEMORY: Failed to cache plot data: {e}")
    
    def get_plot_data(self, key: str) -> Optional[Any]:
        """Récupère les données de tracé en cache"""
        return self.plot_cache.get(key)
    
    def cache_analysis_result(self, key: str, result: Any) -> None:
        """Cache les résultats d'analyse"""
        try:
            self.analysis_cache.put(key, result)
        except Exception as e:
            logging.error(f"MEMORY: Failed to cache analysis result: {e}")
    
    def get_analysis_result(self, key: str) -> Optional[Any]:
        """Récupère les résultats d'analyse en cache"""
        return self.analysis_cache.get(key)
    
    def register_large_object(self, obj: Any) -> None:
        """Enregistre un objet volumineux pour le suivi"""
        self._weak_refs.add(obj)
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force un nettoyage mémoire"""
        stats_before = self.get_memory_stats()
        
        # Vider les caches
        self.plot_cache.clear()
        self.analysis_cache.clear()
        
        # Forcer le garbage collection
        collected = gc.collect()
        
        stats_after = self.get_memory_stats()
        
        cleanup_stats = {
            'objects_collected': collected,
            'memory_before_mb': stats_before.get('memory_usage_mb', 0),
            'memory_after_mb': stats_after.get('memory_usage_mb', 0),
            'memory_freed_mb': stats_before.get('memory_usage_mb', 0) - stats_after.get('memory_usage_mb', 0)
        }
        
        logging.info(f"MEMORY: Cleanup completed - freed {cleanup_stats['memory_freed_mb']:.1f} MB")
        return cleanup_stats
    
    def auto_cleanup(self) -> None:
        """Nettoyage automatique périodique"""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                stats = self.get_memory_stats()
                if stats.get('memory_usage_mb', 0) > 1024:  # Plus de 1GB
                    logging.info("MEMORY: Auto cleanup triggered")
                    self.force_cleanup()
                self._last_cleanup = current_time
            except Exception as e:
                logging.error(f"MEMORY: Auto cleanup failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques mémoire"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'memory_usage_mb': memory_info.rss / (1024 * 1024),
                'memory_percent': process.memory_percent(),
                'plot_cache_stats': self.plot_cache.get_stats(),
                'data_cache_stats': self.data_cache.get_stats(),
                'analysis_cache_stats': self.analysis_cache.get_stats(),
                'weak_refs_count': len(self._weak_refs)
            }
        except ImportError:
            return {
                'plot_cache_stats': self.plot_cache.get_stats(),
                'data_cache_stats': self.data_cache.get_stats(),
                'analysis_cache_stats': self.analysis_cache.get_stats()
            }
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """Génère une clé de cache basée sur les paramètres"""
        key_parts = []
        for arg in args:
            if hasattr(arg, '__name__'):
                key_parts.append(arg.__name__)
            else:
                key_parts.append(str(arg)[:50])  # Limiter la longueur
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={str(v)[:30]}")
        
        return "_".join(key_parts)

# Instance globale
memory_manager = MemoryManager()

# Décorateur pour mettre en cache les résultats
def cached_analysis(cache_key_func=None):
    """Décorateur pour mettre en cache les résultats d'analyse"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Générer la clé de cache
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = memory_manager.generate_cache_key(func.__name__, *args[:3], **kwargs)
            
            # Vérifier le cache
            cached_result = memory_manager.get_analysis_result(cache_key)
            if cached_result is not None:
                logging.debug(f"CACHE: Hit for {func.__name__}")
                return cached_result
            
            # Calculer et mettre en cache
            result = func(*args, **kwargs)
            memory_manager.cache_analysis_result(cache_key, result)
            logging.debug(f"CACHE: Miss for {func.__name__}, cached result")
            
            # Nettoyage automatique
            memory_manager.auto_cleanup()
            
            return result
        return wrapper
    return decorator
