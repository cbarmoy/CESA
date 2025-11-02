# CESA/performance_dashboard.py
"""Dashboard de monitoring des performances CESA"""

import tkinter as tk
from tkinter import ttk
import time
import threading
from typing import Dict, Any
import logging

try:
    from CESA.event_system import event_bus
    from CESA.performance_monitor import perf_monitor
    from CESA.memory_manager import memory_manager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class PerformanceDashboard:
    """Dashboard pour monitorer les performances en temps réel"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.parent = parent_app.root
        self.dashboard_window = None
        self.update_thread = None
        self.running = False
        
    def show_dashboard(self):
        """Affiche le dashboard de performance"""
        if not MONITORING_AVAILABLE:
            tk.messagebox.showwarning("Attention", "Module de monitoring non disponible")
            return
            
        if self.dashboard_window and self.dashboard_window.winfo_exists():
            self.dashboard_window.lift()
            return
            
        self.dashboard_window = tk.Toplevel(self.parent)
        self.dashboard_window.title("CESA - Dashboard Performance")
        self.dashboard_window.geometry("800x600")
        
        # Créer l'interface
        self._create_dashboard_ui()
        
        # Démarrer le monitoring
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Gérer la fermeture
        self.dashboard_window.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_dashboard_ui(self):
        """Crée l'interface du dashboard"""
        # Frame principal avec onglets
        notebook = ttk.Notebook(self.dashboard_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Onglet Performance
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")
        self._create_performance_tab(perf_frame)
        
        # Onglet Mémoire
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="Mémoire")
        self._create_memory_tab(memory_frame)
        
        # Onglet Événements
        events_frame = ttk.Frame(notebook)
        notebook.add(events_frame, text="Événements")
        self._create_events_tab(events_frame)
        
        # Onglet Système
        system_frame = ttk.Frame(notebook)
        notebook.add(system_frame, text="Système")
        self._create_system_tab(system_frame)
    
    def _create_performance_tab(self, parent):
        """Crée l'onglet performance"""
        # Statistiques des fonctions
        ttk.Label(parent, text="Fonctions les plus lentes", font=('Arial', 12, 'bold')).pack(pady=(0,10))
        
        # Treeview pour les fonctions
        columns = ('Fonction', 'Calls', 'Total (s)', 'Moyenne (s)', 'Max (s)')
        self.perf_tree = ttk.Treeview(parent, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.perf_tree.heading(col, text=col)
            self.perf_tree.column(col, width=120, anchor='center')
        
        self.perf_tree.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        
        # Scrollbar
        perf_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.perf_tree.yview)
        self.perf_tree.configure(yscrollcommand=perf_scroll.set)
        perf_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Boutons de contrôle
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="Réinitialiser Stats", 
                  command=self._reset_performance_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Rafraîchir", 
                  command=self._update_performance_display).pack(side=tk.LEFT, padx=5)
    
    def _create_memory_tab(self, parent):
        """Crée l'onglet mémoire"""
        # Informations générales
        info_frame = ttk.LabelFrame(parent, text="Utilisation Mémoire")
        info_frame.pack(fill=tk.X, pady=(0,10))
        
        self.memory_info = ttk.Label(info_frame, text="Chargement...", font=('Courier', 10))
        self.memory_info.pack(padx=10, pady=10)
        
        # Statistiques des caches
        cache_frame = ttk.LabelFrame(parent, text="Caches")
        cache_frame.pack(fill=tk.BOTH, expand=True)
        
        self.cache_info = ttk.Label(cache_frame, text="Chargement...", font=('Courier', 10), justify=tk.LEFT)
        self.cache_info.pack(padx=10, pady=10, anchor='w')
        
        # Boutons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10,0))
        
        ttk.Button(button_frame, text="Forcer Nettoyage", 
                  command=self._force_cleanup).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Vider Caches", 
                  command=self._clear_caches).pack(side=tk.LEFT, padx=5)
    
    def _create_events_tab(self, parent):
        """Crée l'onglet événements"""
        ttk.Label(parent, text="Statistiques des Événements", 
                 font=('Arial', 12, 'bold')).pack(pady=(0,10))
        
        self.events_info = ttk.Label(parent, text="Chargement...", 
                                   font=('Courier', 10), justify=tk.LEFT)
        self.events_info.pack(padx=10, pady=10, anchor='w')
    
    def _create_system_tab(self, parent):
        """Crée l'onglet système"""
        ttk.Label(parent, text="Informations Système", 
                 font=('Arial', 12, 'bold')).pack(pady=(0,10))
        
        self.system_info = ttk.Label(parent, text="Chargement...", 
                                   font=('Courier', 10), justify=tk.LEFT)
        self.system_info.pack(padx=10, pady=10, anchor='w')
        
        # Bouton diagnostic
        ttk.Button(parent, text="Générer Rapport Diagnostic", 
                  command=self._generate_diagnostic).pack(pady=10)
    
    def _update_loop(self):
        """Boucle de mise à jour du dashboard"""
        while self.running and self.dashboard_window and self.dashboard_window.winfo_exists():
            try:
                self._update_all_displays()
                time.sleep(2)  # Mise à jour toutes les 2 secondes
            except tk.TclError:
                break
            except Exception as e:
                logging.error(f"DASHBOARD: Update error - {e}")
                break
    
    def _update_all_displays(self):
        """Met à jour tous les affichages"""
        if not self.dashboard_window or not self.dashboard_window.winfo_exists():
            return
            
        try:
            self.dashboard_window.after(0, self._update_performance_display)
            self.dashboard_window.after(0, self._update_memory_display)
            self.dashboard_window.after(0, self._update_events_display)
            self.dashboard_window.after(0, self._update_system_display)
        except:
            pass
    
    def _update_performance_display(self):
        """Met à jour l'affichage des performances"""
        try:
            # Vider la liste
            for item in self.perf_tree.get_children():
                self.perf_tree.delete(item)
            
            # Récupérer les stats
            stats = perf_monitor.get_all_stats()
            
            # Trier par temps total décroissant
            sorted_stats = sorted(stats.items(), 
                                key=lambda x: x[1].get('total', 0), 
                                reverse=True)
            
            # Afficher les 20 premiers
            for func_name, data in sorted_stats[:20]:
                self.perf_tree.insert('', 'end', values=(
                    func_name,
                    data.get('count', 0),
                    f"{data.get('total', 0):.3f}",
                    f"{data.get('avg', 0):.3f}",
                    f"{data.get('max', 0):.3f}"
                ))
        except Exception as e:
            logging.error(f"DASHBOARD: Performance update error - {e}")
    
    def _update_memory_display(self):
        """Met à jour l'affichage mémoire"""
        try:
            stats = memory_manager.get_memory_stats()
            
            info_text = f"""Utilisation Mémoire: {stats.get('memory_usage_mb', 0):.1f} MB
Pourcentage: {stats.get('memory_percent', 0):.1f}%
Objets faibles: {stats.get('weak_refs_count', 0)}
"""
            
            self.memory_info.config(text=info_text)
            
            # Informations des caches
            cache_text = ""
            for cache_name in ['plot_cache_stats', 'data_cache_stats', 'analysis_cache_stats']:
                if cache_name in stats:
                    cache_stats = stats[cache_name]
                    cache_text += f"""
{cache_name.replace('_stats', '').title()}:
  Taille: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}
  Mémoire: {cache_stats.get('memory_usage_mb', 0):.1f}/{cache_stats.get('max_memory_mb', 0):.1f} MB
  Taux de hit: {cache_stats.get('hit_rate', 0):.1%}
"""
            
            self.cache_info.config(text=cache_text)
        except Exception as e:
            logging.error(f"DASHBOARD: Memory update error - {e}")
    
    def _update_events_display(self):
        """Met à jour l'affichage des événements"""
        try:
            stats = event_bus.get_stats()
            
            events_text = "Événements émis:\n"
            for event_type, count in stats.items():
                events_text += f"  {event_type}: {count}\n"
            
            self.events_info.config(text=events_text)
        except Exception as e:
            logging.error(f"DASHBOARD: Events update error - {e}")
    
    def _update_system_display(self):
        """Met à jour l'affichage système"""
        try:
            import platform
            import sys
            
            system_text = f"""Système: {platform.system()} {platform.release()}
Python: {sys.version.split()[0]}
Threads actifs: {threading.active_count()}
Application: CESA v3.0
Uptime: {time.time() - getattr(self.parent_app, '_start_time', time.time()):.0f}s
"""
            
            # Ajouter des infos sur l'application si disponibles
            if hasattr(self.parent_app, 'raw') and self.parent_app.raw:
                system_text += f"""
Fichier chargé: Oui
Canaux: {len(self.parent_app.raw.ch_names)}
Fréquence: {self.parent_app.sfreq} Hz
"""
            else:
                system_text += "\nFichier chargé: Non"
            
            self.system_info.config(text=system_text)
        except Exception as e:
            logging.error(f"DASHBOARD: System update error - {e}")
    
    def _reset_performance_stats(self):
        """Remet à zéro les statistiques de performance"""
        perf_monitor.clear_metrics()
        self._update_performance_display()
    
    def _force_cleanup(self):
        """Force un nettoyage mémoire"""
        cleanup_stats = memory_manager.force_cleanup()
        tk.messagebox.showinfo("Nettoyage", 
                              f"Mémoire libérée: {cleanup_stats.get('memory_freed_mb', 0):.1f} MB")
    
    def _clear_caches(self):
        """Vide tous les caches"""
        memory_manager.plot_cache.clear()
        memory_manager.data_cache.clear()
        memory_manager.analysis_cache.clear()
        tk.messagebox.showinfo("Caches", "Tous les caches ont été vidés")
    
    def _generate_diagnostic(self):
        """Génère un rapport de diagnostic"""
        try:
            from ui.report_dialog import ReportDialog
            report_dialog = ReportDialog(self.parent_app)
            report_dialog.report_bug()
        except Exception as e:
            tk.messagebox.showerror("Erreur", f"Impossible de générer le diagnostic: {e}")
    
    def _on_close(self):
        """Fermeture du dashboard"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
        if self.dashboard_window:
            self.dashboard_window.destroy()