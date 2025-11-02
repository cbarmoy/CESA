# test_optimizations.py
"""Script de test pour toutes les optimisations CESA"""

import time
import sys
import logging

def test_all_optimizations():
    """Teste toutes les optimisations"""
    print("🚀 Test des optimisations CESA")
    print("=" * 50)
    
    # Test 1: Cache système
    try:
        from CESA.cache_manager import cache_result
        
        @cache_result(expire_hours=1)
        def test_cache_function(x):
            time.sleep(0.1)
            return x * 2
        
        start = time.time()
        result1 = test_cache_function(42)
        time1 = time.time() - start
        
        start = time.time()
        result2 = test_cache_function(42)
        time2 = time.time() - start
        
        cache_speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"✅ Cache système: {cache_speedup:.1f}x plus rapide")
        
    except ImportError:
        print("❌ Cache système non disponible")
    
    # Test 2: Système d'événements
    try:
        from CESA.event_system import event_bus, Events
        
        events_received = []
        def test_handler(data):
            events_received.append(data)
        
        event_bus.subscribe("test_event", test_handler)
        event_bus.emit("test_event", "test_data")
        
        if events_received:
            print("✅ Système d'événements fonctionnel")
        else:
            print("❌ Système d'événements non fonctionnel")
            
    except ImportError:
        print("❌ Système d'événements non disponible")
    
    # Test 3: Monitoring performance
    try:
        from CESA.performance_monitor import perf_monitor, measure_time
        
        @measure_time("test_function")
        def slow_function():
            time.sleep(0.05)
            return "done"
        
        slow_function()
        stats = perf_monitor.get_stats("test_function")
        
        if stats:
            print(f"✅ Monitoring performance: {stats['count']} mesures")
        else:
            print("❌ Monitoring performance non fonctionnel")
            
    except ImportError:
        print("❌ Monitoring performance non disponible")
    
    # Test 4: Gestionnaire mémoire
    try:
        from CESA.memory_manager import memory_manager
        
        # Test du cache
        test_data = list(range(1000))
        memory_manager.cache_plot_data("test_key", test_data)
        retrieved = memory_manager.get_plot_data("test_key")
        
        if retrieved == test_data:
            print("✅ Gestionnaire mémoire fonctionnel")
        else:
            print("❌ Gestionnaire mémoire non fonctionnel")
            
    except ImportError:
        print("❌ Gestionnaire mémoire non disponible")
    
    # Test 5: Optimiseur de données
    try:
        from CESA.data_optimizer import data_optimizer
        import numpy as np
        
        # Créer des données test
        large_data = np.random.randn(10000)
        optimized = data_optimizer.optimize_for_display(large_data, target_points=1000)
        
        reduction_ratio = len(optimized) / len(large_data)
        print(f"✅ Optimiseur de données: {reduction_ratio:.1%} des données conservées")
        
    except ImportError:
        print("❌ Optimiseur de données non disponible")
    
    print("=" * 50)
    print("✅ Test des optimisations terminé")

if __name__ == "__main__":
    test_all_optimizations()
