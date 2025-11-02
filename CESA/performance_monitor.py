# CESA/performance_monitor.py
"""Monitoring des performances pour CESA"""

import time
import logging
from functools import wraps
from typing import Dict, List
import threading

class PerformanceMonitor:
    """Moniteur de performance pour suivre les temps d'exécution"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def start_timer(self, name: str) -> None:
        """Démarre un timer"""
        with self._lock:
            self.active_timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Arrête un timer et enregistre la durée"""
        with self._lock:
            if name in self.active_timers:
                duration = time.time() - self.active_timers[name]
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append(duration)
                del self.active_timers[name]
                return duration
        return 0.0
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Retourne les statistiques pour une métrique"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
            
        times = self.metrics[name]
        return {
            'count': len(times),
            'total': sum(times),
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'last_10_avg': sum(times[-10:]) / min(len(times), 10)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Retourne toutes les statistiques"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def clear_metrics(self, name: str = None) -> None:
        """Efface les métriques"""
        with self._lock:
            if name:
                self.metrics.pop(name, None)
            else:
                self.metrics.clear()

# Instance globale
perf_monitor = PerformanceMonitor()

def measure_time(name: str = None):
    """Décorateur pour mesurer le temps d'exécution d'une fonction"""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            perf_monitor.start_timer(func_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = perf_monitor.stop_timer(func_name)
                if duration > 0.1:  # Log seulement si > 100ms
                    logging.debug(f"PERF: {func_name} took {duration:.3f}s")
        return wrapper
    return decorator

class MemoryMonitor:
    """Moniteur simple de mémoire"""
    
    @staticmethod
    def get_memory_usage():
        """Retourne l'utilisation mémoire actuelle"""
        try:
            import psutil
            process = psutil.Process()
            return {
                'rss': process.memory_info().rss / 1024 / 1024,  # MB
                'vms': process.memory_info().vms / 1024 / 1024,  # MB
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
