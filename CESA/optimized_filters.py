# CESA/optimized_filters.py
from functools import lru_cache
import numpy as np
from typing import Optional, Tuple
from CESA.cache_manager import cache_result

# Cache des coefficients de filtres pour éviter de les recalculer
@lru_cache(maxsize=32)
def _get_filter_coefficients(sfreq: float, filter_order: int, low: Optional[float], high: Optional[float]) -> Tuple:
    """Cache les coefficients de filtre pour éviter les recalculs"""
    try:
        from scipy.signal import butter
        
        nyquist = float(sfreq) / 2.0
        if nyquist <= 0.0:
            return None, None
        
        low = 0.0 if low is None else float(low)
        high = 0.0 if high is None else float(high)
        
        if low > 0.0 and high > 0.0:
            # Band-pass
            if high >= nyquist:
                high = nyquist - 1e-6
            wn = (low / nyquist, high / nyquist)
            return butter(int(filter_order), wn, btype='bandpass')
        elif low > 0.0 and high <= 0.0:
            # High-pass
            wn = low / nyquist
            return butter(int(filter_order), wn, btype='highpass')
        elif low <= 0.0 and high > 0.0:
            # Low-pass
            wn = high / nyquist
            return butter(int(filter_order), wn, btype='lowpass')
        else:
            return None, None
    except Exception:
        return None, None

@cache_result(expire_hours=1)
def apply_filter_optimized(data: np.ndarray, sfreq: float, filter_order: int, 
                          low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
    """Version optimisée de apply_filter avec cache des coefficients"""
    try:
        from scipy.signal import filtfilt
        
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=float)
        
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Utiliser le cache des coefficients
        b, a = _get_filter_coefficients(sfreq, filter_order, low, high)
        
        if b is None or a is None:
            return data
        
        return filtfilt(b, a, data)
    except Exception:
        return data

# Traitement vectorisé multi-canaux
def apply_filter_batch(data_batch: np.ndarray, sfreq: float, filter_order: int,
                      low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
    """Applique un filtre à plusieurs canaux simultanément"""
    if data_batch.ndim == 1:
        return apply_filter_optimized(data_batch, sfreq, filter_order, low, high)
    
    # Traitement par lots pour économiser la mémoire
    result = np.zeros_like(data_batch)
    batch_size = min(4, data_batch.shape[0])  # Traiter 4 canaux maximum à la fois
    
    for i in range(0, data_batch.shape[0], batch_size):
        end_idx = min(i + batch_size, data_batch.shape[0])
        for j in range(i, end_idx):
            result[j] = apply_filter_optimized(data_batch[j], sfreq, filter_order, low, high)
    
    return result
