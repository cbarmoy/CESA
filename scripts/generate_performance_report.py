"""
Script de génération de rapport interactif pour le diagnostic de performance.

Crée une interface Tkinter pour visualiser les résultats d'analyse de performance.
"""

import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous d'avoir installé matplotlib, pandas et numpy")
    sys.exit(1)

try:
    from scripts.analyze_performance import load_telemetry_data, calculate_statistics, identify_bottlenecks
except ImportError:
    # Essayer depuis le répertoire racine
    try:
        import analyze_performance
        load_telemetry_data = analyze_performance.load_telemetry_data
        calculate_statistics = analyze_performance.calculate_statistics
        identify_bottlenecks = analyze_performance.identify_bottlenecks
    except Exception:
        print("Impossible d'importer analyze_performance")
        sys.exit(1)


class PerformanceReportWindow:
    """Fenêtre principale pour le rapport de diagnostic de performance."""
    
    def __init__(self, root, csv_path):
        self.root = root
        self.root.title("Diagnostic de Performance - Visualisation Signaux")
        self.root.geometry("1200x800")
        
        self.csv_path = Path(csv_path) if csv_path else None
        self.data = []
        
        # Créer l'interface
        self._create_ui()
        
        # Charger les données si un chemin est fourni
        if self.csv_path and self.csv_path.exists():
            self.load_data(self.csv_path)
    
    def _create_ui(self):
        """Crée l'interface utilisateur."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame supérieur pour les contrôles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Charger fichier CSV", command=self._browse_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Actualiser", command=self._refresh).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Générer rapport texte", command=self._generate_text_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Générer graphiques PDF", command=self._generate_pdf_plots).pack(side=tk.LEFT, padx=5)
        
        # Label d'état
        self.status_label = ttk.Label(control_frame, text="Aucun fichier chargé", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet 1: Résumé
        summary_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(summary_frame, text="Résumé")
        self._create_summary_tab(summary_frame)
        
        # Onglet 2: Goulots d'étranglement
        bottlenecks_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(bottlenecks_frame, text="Goulots d'étranglement")
        self._create_bottlenecks_tab(bottlenecks_frame)
        
        # Onglet 3: Graphiques
        plots_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(plots_frame, text="Graphiques")
        self._create_plots_tab(plots_frame)
        
        # Onglet 4: Recommandations
        recommendations_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(recommendations_frame, text="Recommandations")
        self._create_recommendations_tab(recommendations_frame)
    
    def _create_summary_tab(self, parent):
        """Crée l'onglet de résumé."""
        # Text widget pour afficher les statistiques
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.summary_text = tk.Text(text_frame, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.summary_text.insert(tk.END, "Chargement des données...\n")
        self.summary_text.config(state=tk.DISABLED)
    
    def _create_bottlenecks_tab(self, parent):
        """Crée l'onglet des goulots d'étranglement."""
        # Treeview pour afficher les opérations
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Operation", "TempsMoyen", "TempsMedian", "P95", "P99", "Echantillons")
        self.bottlenecks_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            self.bottlenecks_tree.heading(col, text=col.replace("_", " "))
            self.bottlenecks_tree.column(col, width=150, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.bottlenecks_tree.yview)
        self.bottlenecks_tree.configure(yscrollcommand=scrollbar.set)
        
        self.bottlenecks_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_plots_tab(self, parent):
        """Crée l'onglet des graphiques."""
        # Frame pour les graphiques
        self.plots_frame = ttk.Frame(parent)
        self.plots_frame.pack(fill=tk.BOTH, expand=True)
    
    def _create_recommendations_tab(self, parent):
        """Crée l'onglet des recommandations."""
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.recommendations_text = tk.Text(text_frame, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.recommendations_text.yview)
        self.recommendations_text.configure(yscrollcommand=scrollbar.set)
        
        self.recommendations_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.recommendations_text.insert(tk.END, "Chargement des données...\n")
        self.recommendations_text.config(state=tk.DISABLED)
    
    def _browse_csv(self):
        """Ouvre un dialogue pour sélectionner un fichier CSV."""
        filename = filedialog.askopenfilename(
            title="Sélectionner le fichier de télémetrie",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path = Path(filename)
            self.load_data(self.csv_path)
    
    def _refresh(self):
        """Rafraîchit les données."""
        if self.csv_path and self.csv_path.exists():
            self.load_data(self.csv_path)
        else:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
    
    def load_data(self, csv_path):
        """Charge les données depuis le fichier CSV."""
        try:
            self.csv_path = Path(csv_path)
            self.data = load_telemetry_data(self.csv_path)
            self.status_label.config(text=f"{len(self.data)} échantillons chargés", foreground="green")
            self._update_all_tabs()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")
            self.status_label.config(text="Erreur de chargement", foreground="red")
    
    def _update_all_tabs(self):
        """Met à jour tous les onglets avec les données."""
        self._update_summary_tab()
        self._update_bottlenecks_tab()
        self._update_plots_tab()
        self._update_recommendations_tab()
    
    def _update_summary_tab(self):
        """Met à jour l'onglet de résumé."""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        if not self.data:
            self.summary_text.insert(tk.END, "Aucune donnée disponible\n")
            self.summary_text.config(state=tk.DISABLED)
            return
        
        # Statistiques générales
        self.summary_text.insert(tk.END, "=" * 80 + "\n")
        self.summary_text.insert(tk.END, "RAPPORT DE DIAGNOSTIC DES PERFORMANCES\n")
        self.summary_text.insert(tk.END, "=" * 80 + "\n\n")
        
        self.summary_text.insert(tk.END, f"Nombre total d'échantillons: {len(self.data)}\n")
        if self.data:
            first_timestamp = self.data[0].get('timestamp', 'N/A')
            last_timestamp = self.data[-1].get('timestamp', 'N/A')
            self.summary_text.insert(tk.END, f"Période: {first_timestamp} à {last_timestamp}\n")
        self.summary_text.insert(tk.END, "\n")
        
        # Statistiques temps total
        total_stats = calculate_statistics(self.data, 'total_ms')
        if total_stats:
            self.summary_text.insert(tk.END, "Temps total par frame:\n")
            self.summary_text.insert(tk.END, f"  Moyenne: {total_stats['mean']:.2f} ms\n")
            self.summary_text.insert(tk.END, f"  Médiane: {total_stats['median']:.2f} ms\n")
            self.summary_text.insert(tk.END, f"  Min: {total_stats['min']:.2f} ms\n")
            self.summary_text.insert(tk.END, f"  Max: {total_stats['max']:.2f} ms\n")
            self.summary_text.insert(tk.END, f"  P95: {total_stats['p95']:.2f} ms\n")
            self.summary_text.insert(tk.END, f"  P99: {total_stats['p99']:.2f} ms\n\n")
        
        # FPS
        fps_stats = calculate_statistics(self.data, 'fps')
        if fps_stats:
            self.summary_text.insert(tk.END, "FPS (Images par seconde):\n")
            self.summary_text.insert(tk.END, f"  Moyenne: {fps_stats['mean']:.2f}\n")
            self.summary_text.insert(tk.END, f"  Médiane: {fps_stats['median']:.2f}\n")
            self.summary_text.insert(tk.END, f"  Min: {fps_stats['min']:.2f}\n")
            self.summary_text.insert(tk.END, f"  Max: {fps_stats['max']:.2f}\n\n")
        
        # Statistiques par opération
        operations = ['draw_ms', 'extract_ms', 'filter_ms', 'baseline_ms', 'decim_ms', 'render_ms',
                     'load_edf_ms', 'extract_channels_ms', 'convert_uv_ms', 'prepare_hypno_ms', 'create_plotter_ms']
        
        self.summary_text.insert(tk.END, "\nStatistiques par opération:\n")
        self.summary_text.insert(tk.END, "-" * 80 + "\n")
        
        for op in operations:
            stats = calculate_statistics(self.data, op)
            if stats and stats['count'] > 0:
                self.summary_text.insert(tk.END, f"\n{op}:\n")
                self.summary_text.insert(tk.END, f"  Échantillons: {stats['count']}\n")
                self.summary_text.insert(tk.END, f"  Moyenne: {stats['mean']:.2f} ms | Médiane: {stats['median']:.2f} ms\n")
                self.summary_text.insert(tk.END, f"  P95: {stats['p95']:.2f} ms | P99: {stats['p99']:.2f} ms\n")
        
        self.summary_text.config(state=tk.DISABLED)
    
    def _update_bottlenecks_tab(self):
        """Met à jour l'onglet des goulots d'étranglement."""
        # Effacer les anciennes entrées
        for item in self.bottlenecks_tree.get_children():
            self.bottlenecks_tree.delete(item)
        
        if not self.data:
            return
        
        bottlenecks = identify_bottlenecks(self.data)
        
        for i, (field, avg_time, time_str) in enumerate(bottlenecks[:20], 1):
            stats = calculate_statistics(self.data, field)
            if stats:
                self.bottlenecks_tree.insert("", tk.END, values=(
                    field,
                    f"{stats['mean']:.2f} ms",
                    f"{stats['median']:.2f} ms",
                    f"{stats['p95']:.2f} ms",
                    f"{stats['p99']:.2f} ms",
                    stats['count']
                ))
    
    def _update_plots_tab(self):
        """Met à jour l'onglet des graphiques."""
        # Effacer les anciens graphiques
        for widget in self.plots_frame.winfo_children():
            widget.destroy()
        
        if not self.data:
            label = ttk.Label(self.plots_frame, text="Aucune donnée disponible")
            label.pack()
            return
        
        try:
            # Créer une figure avec plusieurs sous-graphiques
            fig = Figure(figsize=(12, 8))
            
            # Graphique 1: Timeline des temps totaux
            ax1 = fig.add_subplot(2, 2, 1)
            total_times = []
            for r in self.data:
                val = r.get('total_ms')
                if val is not None:
                    try:
                        total_times.append(float(val))
                    except (ValueError, TypeError):
                        continue
            if total_times:
                ax1.plot(range(len(total_times)), total_times, alpha=0.6, linewidth=0.8)
                ax1.axhline(y=np.mean(total_times), color='r', linestyle='--', 
                           label=f'Moyenne: {np.mean(total_times):.1f} ms')
                ax1.set_xlabel('Échantillon')
                ax1.set_ylabel('Temps total (ms)')
                ax1.set_title('Timeline des temps de rendu')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
            
            # Graphique 2: Distribution des temps par opération
            ax2 = fig.add_subplot(2, 2, 2)
            time_fields = ['draw_ms', 'extract_ms', 'filter_ms', 'baseline_ms', 'decim_ms']
            available_fields = [f for f in time_fields if any(r.get(f) is not None for r in self.data)]
            
            if available_fields:
                data_to_plot = []
                labels = []
                for field in available_fields:
                    values = []
                    for r in self.data:
                        val = r.get(field)
                        if val is not None:
                            try:
                                values.append(float(val))
                            except (ValueError, TypeError):
                                continue
                    if values:
                        data_to_plot.append(values)
                        labels.append(field.replace('_ms', ''))
                
                if data_to_plot:
                    # Utiliser tick_labels pour matplotlib >= 3.9
                    try:
                        bp = ax2.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
                    except TypeError:
                        # Fallback pour les versions antérieures
                        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.7)
                    ax2.set_ylabel('Temps (ms)')
                    ax2.set_title('Distribution des temps par opération')
                    ax2.grid(True, alpha=0.3, axis='y')
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Graphique 3: FPS au fil du temps
            ax3 = fig.add_subplot(2, 2, 3)
            fps_values = []
            for r in self.data:
                val = r.get('fps')
                if val is not None:
                    try:
                        fps_values.append(float(val))
                    except (ValueError, TypeError):
                        continue
            if fps_values:
                ax3.plot(range(len(fps_values)), fps_values, alpha=0.6, linewidth=0.8, color='green')
                ax3.axhline(y=np.mean(fps_values), color='r', linestyle='--',
                           label=f'Moyenne: {np.mean(fps_values):.2f} FPS')
                ax3.set_xlabel('Échantillon')
                ax3.set_ylabel('FPS')
                ax3.set_title('FPS au fil du temps')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            
            # Graphique 4: Comparaison par mode d'accès
            ax4 = fig.add_subplot(2, 2, 4)
            actions = {}
            for record in self.data:
                action = record.get('action', 'unknown')
                if action not in actions:
                    actions[action] = []
                val = record.get('total_ms')
                if val is not None:
                    try:
                        actions[action].append(float(val))
                    except (ValueError, TypeError):
                        continue
            
            if actions:
                action_data = []
                action_labels = []
                for action, values in sorted(actions.items()):
                    if values:
                        action_data.append(values)
                        action_labels.append(action.upper())
                
                if action_data:
                    # Utiliser tick_labels pour matplotlib >= 3.9
                    try:
                        bp = ax4.boxplot(action_data, tick_labels=action_labels, patch_artist=True)
                    except TypeError:
                        # Fallback pour les versions antérieures
                        bp = ax4.boxplot(action_data, labels=action_labels, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightcoral')
                        patch.set_alpha(0.7)
                    ax4.set_ylabel('Temps total (ms)')
                    ax4.set_title('Comparaison par mode d\'accès')
                    ax4.grid(True, alpha=0.3, axis='y')
            
            fig.tight_layout()
            
            # Afficher dans Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Ajouter une toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.plots_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        except Exception as e:
            label = ttk.Label(self.plots_frame, text=f"Erreur lors de la génération des graphiques: {e}")
            label.pack()
    
    def _update_recommendations_tab(self):
        """Met à jour l'onglet des recommandations."""
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        
        if not self.data:
            self.recommendations_text.insert(tk.END, "Aucune donnée disponible\n")
            self.recommendations_text.config(state=tk.DISABLED)
            return
        
        bottlenecks = identify_bottlenecks(self.data)
        
        self.recommendations_text.insert(tk.END, "RECOMMANDATIONS D'OPTIMISATION\n")
        self.recommendations_text.insert(tk.END, "=" * 80 + "\n\n")
        
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            self.recommendations_text.insert(tk.END, f"1. Opération la plus lente: {top_bottleneck[0]}\n")
            self.recommendations_text.insert(tk.END, f"   Temps moyen: {top_bottleneck[2]}\n")
            self.recommendations_text.insert(tk.END, "   Action recommandée: Optimiser cette opération en priorité\n\n")
        
        # Comparer draw_ms vs total_ms
        draw_stats = calculate_statistics(self.data, 'draw_ms')
        total_stats = calculate_statistics(self.data, 'total_ms')
        if draw_stats and total_stats:
            draw_percent = (draw_stats['mean'] / total_stats['mean']) * 100 if total_stats['mean'] > 0 else 0
            self.recommendations_text.insert(tk.END, f"2. Le rendu (draw_ms) représente {draw_percent:.1f}% du temps total\n")
            if draw_percent > 50:
                self.recommendations_text.insert(tk.END, "   Action recommandée: Optimiser le rendu matplotlib (décimation, blitting)\n\n")
        
        self.recommendations_text.insert(tk.END, "3. Actions générales pour améliorer les performances:\n")
        self.recommendations_text.insert(tk.END, "   - Activer la décimation si ce n'est pas déjà fait\n")
        self.recommendations_text.insert(tk.END, "   - Utiliser le mode 'precomputed' (multiscale) pour de grands fichiers\n")
        self.recommendations_text.insert(tk.END, "   - Réduire le nombre de canaux affichés simultanément\n")
        self.recommendations_text.insert(tk.END, "   - Désactiver les filtres non essentiels\n")
        self.recommendations_text.insert(tk.END, "   - Vérifier que le blitting est activé dans PSGPlotter\n\n")
        
        # Recommandations spécifiques basées sur les données
        self.recommendations_text.insert(tk.END, "4. Recommandations spécifiques:\n")
        
        # Analyser chaque opération
        operations_to_check = [
            ('load_edf_ms', 'Chargement EDF'),
            ('extract_channels_ms', 'Extraction des canaux'),
            ('convert_uv_ms', 'Conversion µV'),
            ('prepare_hypno_ms', 'Préparation hypnogramme'),
            ('create_plotter_ms', 'Création PSGPlotter'),
            ('filter_ms', 'Filtrage'),
            ('baseline_ms', 'Correction baseline'),
            ('decim_ms', 'Décimation'),
            ('draw_ms', 'Rendu matplotlib'),
        ]
        
        for field, description in operations_to_check:
            stats = calculate_statistics(self.data, field)
            if stats and stats['mean'] > 100:  # Si moyenne > 100ms
                self.recommendations_text.insert(tk.END, f"   - {description}: {stats['mean']:.1f} ms en moyenne\n")
                self.recommendations_text.insert(tk.END, f"     → Optimiser cette opération (objectif: <50ms)\n")
        
        self.recommendations_text.config(state=tk.DISABLED)
    
    def _generate_text_report(self):
        """Génère un rapport texte."""
        if not self.data:
            messagebox.showwarning("Attention", "Aucune donnée chargée")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Enregistrer le rapport",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                from scripts.analyze_performance import generate_summary_report
                generate_summary_report(self.data, Path(filename))
                messagebox.showinfo("Succès", f"Rapport sauvegardé: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la génération: {e}")
    
    def _generate_pdf_plots(self):
        """Génère un PDF avec les graphiques."""
        if not self.data:
            messagebox.showwarning("Attention", "Aucune donnée chargée")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Enregistrer les graphiques",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            try:
                from scripts.analyze_performance import generate_plots
                generate_plots(self.data, Path(filename))
                messagebox.showinfo("Succès", f"Graphiques sauvegardés: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la génération: {e}")


def main():
    parser = argparse.ArgumentParser(description='Génère un rapport interactif de performance')
    parser.add_argument('--csv', type=str, default='logs/telemetry.csv',
                       help='Chemin vers le fichier CSV de télémetrie')
    
    args = parser.parse_args()
    
    root = tk.Tk()
    app = PerformanceReportWindow(root, args.csv)
    root.mainloop()


if __name__ == '__main__':
    main()

