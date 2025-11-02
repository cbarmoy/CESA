# CESA/event_system.py
"""Système d'événements centralisé pour CESA - Découplage des composants"""

from collections import defaultdict
from typing import Callable, Any, Dict, List
import logging
import time

class EventBus:
    """Bus d'événements centralisé pour découpler les composants"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_stats: Dict[str, int] = defaultdict(int)
        self._last_emit_time: Dict[str, float] = {}
        self._throttle_intervals: Dict[str, float] = {}
        
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """S'abonne à un type d'événement"""
        self._subscribers[event_type].append(callback)
        logging.debug(f"EVENT: Subscription added for {event_type}")
        
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Se désabonne d'un type d'événement"""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logging.debug(f"EVENT: Subscription removed for {event_type}")
    
    def emit(self, event_type: str, data: Any = None, throttle: float = None) -> None:
        """Émet un événement vers tous les abonnés"""
        current_time = time.time()
        
        # Throttling pour éviter les émissions trop fréquentes
        if throttle and event_type in self._last_emit_time:
            time_since_last = current_time - self._last_emit_time[event_type]
            if time_since_last < throttle:
                return
        
        self._last_emit_time[event_type] = current_time
        self._event_stats[event_type] += 1
        
        # Notifier tous les abonnés
        for callback in self._subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                logging.error(f"EVENT: Error in callback for {event_type}: {e}")
    
    def clear_subscribers(self, event_type: str = None) -> None:
        """Supprime tous les abonnés (ou pour un type spécifique)"""
        if event_type:
            self._subscribers[event_type].clear()
        else:
            self._subscribers.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques d'utilisation des événements"""
        return dict(self._event_stats)

# Instance globale
event_bus = EventBus()

# Types d'événements définis
class Events:
    """Constantes pour les types d'événements"""
    
    # Événements de données
    DATA_LOADED = "data_loaded"
    DATA_UNLOADED = "data_unloaded"
    CHANNELS_SELECTED = "channels_selected"
    
    # Événements de navigation
    TIME_CHANGED = "time_changed"
    EPOCH_CHANGED = "epoch_changed"
    ZOOM_CHANGED = "zoom_changed"
    
    # Événements de filtrage
    FILTER_CHANGED = "filter_changed"
    FILTER_ENABLED = "filter_enabled"
    
    # Événements d'affichage
    PLOT_UPDATE_REQUESTED = "plot_update_requested"
    AMPLITUDE_CHANGED = "amplitude_changed"
    THEME_CHANGED = "theme_changed"
    
    # Événements de scoring
    SCORING_LOADED = "scoring_loaded"
    SCORING_CHANGED = "scoring_changed"
    
    # Événements système
    ERROR_OCCURRED = "error_occurred"
    STATUS_CHANGED = "status_changed"

class EventData:
    """Classes pour structurer les données d'événements"""
    
    class TimeChanged:
        def __init__(self, current_time: float, duration: float):
            self.current_time = current_time
            self.duration = duration
    
    class DataLoaded:
        def __init__(self, filename: str, channels: list, sfreq: float):
            self.filename = filename
            self.channels = channels
            self.sfreq = sfreq
    
    class FilterChanged:
        def __init__(self, enabled: bool, low: float, high: float):
            self.enabled = enabled
            self.low = low
            self.high = high
    
    class ChannelsSelected:
        def __init__(self, channels: list):
            self.channels = channels
    
    class Error:
        def __init__(self, message: str, exception: Exception = None):
            self.message = message
            self.exception = exception

# Décorateurs utilitaires
def emit_on_change(event_type: str):
    """Décorateur pour émettre automatiquement un événement quand une méthode est appelée"""
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            event_bus.emit(event_type, result)
            return result
        return wrapper
    return decorator

def throttle_events(interval: float):
    """Décorateur pour limiter la fréquence d'émission des événements"""
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            event_type = f"{method.__name__}_throttled"
            result = method(self, *args, **kwargs)
            event_bus.emit(event_type, result, throttle=interval)
            return result
        return wrapper
    return decorator
