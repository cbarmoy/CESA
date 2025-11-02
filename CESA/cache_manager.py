# CESA/cache_manager.py
from functools import wraps
import hashlib
import pickle
import os
from pathlib import Path
import time

class SimpleCache:
    def __init__(self, cache_dir="cache", max_memory_items=50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.max_memory_items = max_memory_items
        
    def _get_cache_key(self, func_name, args, kwargs):
        """Génère une clé unique pour les paramètres"""
        content = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key):
        """Récupère depuis le cache mémoire d'abord, puis disque"""
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Mettre en cache mémoire
                    if len(self.memory_cache) < self.max_memory_items:
                        self.memory_cache[key] = data
                    return data
            except:
                pass
        return None
    
    def set(self, key, value):
        """Stocke en cache mémoire et disque"""
        # Cache mémoire
        if len(self.memory_cache) >= self.max_memory_items:
            # Supprimer le plus ancien
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
        
        # Cache disque
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except:
            pass

# Instance globale
cache = SimpleCache()

def cache_result(expire_hours=1):
    """Décorateur pour mettre en cache les résultats de fonctions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._get_cache_key(func.__name__, args, kwargs)
            
            # Vérifier le cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Calculer et mettre en cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator
