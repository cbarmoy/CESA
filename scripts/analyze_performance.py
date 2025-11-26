"""
Script d'analyse des performances pour la visualisation des signaux.

Analyse les données de télémetrie collectées dans logs/telemetry.csv
et génère des statistiques détaillées avec graphiques.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_telemetry_data(csv_path: Path) -> List[Dict[str, float]]:
    """Charge les données de télémetrie depuis le fichier CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de télémetrie introuvable: {csv_path}")
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Convertir les valeurs numériques
                record = {}
                for key, value in row.items():
                    # Gérer les cas où value peut être None, une liste, ou une chaîne
                    if value is None:
                        record[key] = None
                        continue
                    
                    # Si c'est une liste, convertir en chaîne
                    if isinstance(value, list):
                        value = ', '.join(str(v) for v in value)
                    
                    # Convertir en chaîne si ce n'est pas déjà le cas
                    if not isinstance(value, str):
                        value = str(value)
                    
                    # Vérifier si la valeur est vide après strip
                    value_str = value.strip()
                    if value_str == '':
                        record[key] = None
                    else:
                        # Les champs de type chaîne doivent toujours rester des chaînes
                        if key in ['timestamp', 'action', 'dataset_id', 'channel', 'notes']:
                            # S'assurer que c'est bien une chaîne, même si ça ressemble à un nombre
                            record[key] = str(value_str)
                        elif key.endswith('_ms') or key in ['fps', 'level', 'chunks_read', 'bytes_read', 
                                                           'viewport_px', 'spp_screen', 'level_k', 
                                                           'cpu_pct', 'rss_mb', 'cache_hit', 'start_s', 'duration_s']:
                            # Champs numériques - essayer de convertir en float
                            try:
                                record[key] = float(value_str)
                            except (ValueError, TypeError):
                                # Si la conversion échoue (chaîne non numérique), mettre None
                                record[key] = None
                        else:
                            # Pour les autres champs, essayer de détecter le type
                            try:
                                # Essayer float d'abord
                                record[key] = float(value_str)
                            except (ValueError, TypeError):
                                # Sinon, garder comme chaîne
                                record[key] = str(value_str)
                data.append(record)
            except Exception as e:
                print(f"Erreur lors de la lecture d'une ligne: {e}")
                continue
    
    return data


def calculate_statistics(data: List[Dict], field: str) -> Dict[str, float]:
    """Calcule les statistiques pour un champ numérique."""
    values = []
    for r in data:
        val = r.get(field)
        if val is not None and val != "":
            try:
                # Essayer de convertir en float
                float_val = float(val)
                values.append(float_val)
            except (ValueError, TypeError):
                # Ignorer les valeurs non numériques
                continue
    
    if not values:
        return {}
    
    values = np.array(values, dtype=float)
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
    }


def identify_bottlenecks(data: List[Dict]) -> List[Tuple[str, float, str]]:
    """Identifie les opérations les plus lentes."""
    # Champs de temps à analyser (en millisecondes)
    time_fields = [
        'total_ms',
        'draw_ms',
        'extract_ms',
        'filter_ms',
        'baseline_ms',
        'decim_ms',
        'render_ms',
        'io_ms',
        'load_edf_ms',
        'extract_channels_ms',
        'convert_uv_ms',
        'prepare_hypno_ms',
        'create_plotter_ms',
        'prepare_data_ms',
    ]
    
    bottlenecks = []
    for field in time_fields:
        stats = calculate_statistics(data, field)
        if stats and stats['count'] > 0:
            # Utiliser la moyenne pondérée par le nombre d'occurrences
            avg_time = stats['mean']
            if avg_time > 0:
                bottlenecks.append((field, avg_time, f"{avg_time:.2f} ms"))
    
    # Trier par temps décroissant
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    return bottlenecks


def generate_summary_report(data: List[Dict], output_path: Path) -> None:
    """Génère un rapport texte avec les statistiques."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT DE DIAGNOSTIC DES PERFORMANCES - VISUALISATION SIGNAUX\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Nombre total d'échantillons: {len(data)}\n")
        f.write(f"Période analysée: {data[0].get('timestamp', 'N/A')} à {data[-1].get('timestamp', 'N/A')}\n\n")
        
        # Statistiques générales
        f.write("-" * 80 + "\n")
        f.write("STATISTIQUES GÉNÉRALES\n")
        f.write("-" * 80 + "\n\n")
        
        if data:
            total_stats = calculate_statistics(data, 'total_ms')
            if total_stats:
                f.write("Temps total par frame:\n")
                f.write(f"  Moyenne: {total_stats['mean']:.2f} ms\n")
                f.write(f"  Médiane: {total_stats['median']:.2f} ms\n")
                f.write(f"  Min: {total_stats['min']:.2f} ms\n")
                f.write(f"  Max: {total_stats['max']:.2f} ms\n")
                f.write(f"  P95: {total_stats['p95']:.2f} ms\n")
                f.write(f"  P99: {total_stats['p99']:.2f} ms\n\n")
            
            fps_stats = calculate_statistics(data, 'fps')
            if fps_stats:
                f.write("FPS (Images par seconde):\n")
                f.write(f"  Moyenne: {fps_stats['mean']:.2f}\n")
                f.write(f"  Médiane: {fps_stats['median']:.2f}\n")
                f.write(f"  Min: {fps_stats['min']:.2f}\n")
                f.write(f"  Max: {fps_stats['max']:.2f}\n\n")
        
        # Identification des goulots d'étranglement
        f.write("-" * 80 + "\n")
        f.write("GOULOTS D'ÉTRANGLEMENT (opérations les plus lentes)\n")
        f.write("-" * 80 + "\n\n")
        
        bottlenecks = identify_bottlenecks(data)
        for i, (field, avg_time, time_str) in enumerate(bottlenecks[:10], 1):
            stats = calculate_statistics(data, field)
            if stats:
                f.write(f"{i}. {field}: {time_str}\n")
                f.write(f"   Moyenne: {stats['mean']:.2f} ms | "
                       f"Médiane: {stats['median']:.2f} ms | "
                       f"P95: {stats['p95']:.2f} ms | "
                       f"P99: {stats['p99']:.2f} ms\n\n")
        
        # Statistiques détaillées par opération
        f.write("-" * 80 + "\n")
        f.write("STATISTIQUES DÉTAILLÉES PAR OPÉRATION\n")
        f.write("-" * 80 + "\n\n")
        
        operations = ['draw_ms', 'extract_ms', 'filter_ms', 'baseline_ms', 'decim_ms', 'render_ms']
        for op in operations:
            stats = calculate_statistics(data, op)
            if stats and stats['count'] > 0:
                f.write(f"{op}:\n")
                f.write(f"  Échantillons: {stats['count']}\n")
                f.write(f"  Moyenne: {stats['mean']:.2f} ms | "
                       f"Médiane: {stats['median']:.2f} ms | "
                       f"Std: {stats['std']:.2f} ms\n")
                f.write(f"  Min: {stats['min']:.2f} ms | "
                       f"Max: {stats['max']:.2f} ms\n")
                f.write(f"  P25: {stats['p25']:.2f} ms | "
                       f"P75: {stats['p75']:.2f} ms | "
                       f"P95: {stats['p95']:.2f} ms | "
                       f"P99: {stats['p99']:.2f} ms\n\n")
        
        # Analyse par action/mode
        f.write("-" * 80 + "\n")
        f.write("ANALYSE PAR MODE D'ACCÈS\n")
        f.write("-" * 80 + "\n\n")
        
        actions = {}
        for record in data:
            action = record.get('action', 'unknown')
            if action not in actions:
                actions[action] = []
            actions[action].append(record)
        
        for action, records in sorted(actions.items()):
            stats = calculate_statistics(records, 'total_ms')
            if stats:
                f.write(f"{action.upper()}:\n")
                f.write(f"  Nombre d'échantillons: {stats['count']}\n")
                f.write(f"  Temps moyen: {stats['mean']:.2f} ms\n")
                f.write(f"  Temps médian: {stats['median']:.2f} ms\n")
                f.write(f"  FPS moyen: {calculate_statistics(records, 'fps').get('mean', 0):.2f}\n\n")
        
        # Recommandations
        f.write("-" * 80 + "\n")
        f.write("RECOMMANDATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            f.write(f"1. Opération la plus lente: {top_bottleneck[0]} ({top_bottleneck[2]})\n")
            f.write("   Action recommandée: Optimiser cette opération en priorité\n\n")
        
        # Comparer draw_ms vs total_ms pour identifier les pertes
        draw_stats = calculate_statistics(data, 'draw_ms')
        total_stats = calculate_statistics(data, 'total_ms')
        if draw_stats and total_stats:
            draw_percent = (draw_stats['mean'] / total_stats['mean']) * 100 if total_stats['mean'] > 0 else 0
            f.write(f"2. Le rendu (draw_ms) représente {draw_percent:.1f}% du temps total\n")
            if draw_percent > 50:
                f.write("   Action recommandée: Optimiser le rendu matplotlib (décimation, blitting)\n\n")
        
        f.write("3. Pour améliorer les performances:\n")
        f.write("   - Activer la décimation si ce n'est pas déjà fait\n")
        f.write("   - Utiliser le mode 'precomputed' (multiscale) pour de grands fichiers\n")
        f.write("   - Réduire le nombre de canaux affichés simultanément\n")
        f.write("   - Désactiver les filtres non essentiels\n\n")


def generate_plots(data: List[Dict], output_path: Path) -> None:
    """Génère des graphiques de performance."""
    if not data:
        print("Aucune donnée à visualiser")
        return
    
    with PdfPages(output_path) as pdf:
        # Graphique 1: Timeline des temps totaux
        fig, ax = plt.subplots(figsize=(12, 6))
        timestamps = [r.get('timestamp', '') for r in data]
        total_times = [r.get('total_ms', 0) for r in data if r.get('total_ms') is not None]
        
        if total_times:
            ax.plot(range(len(total_times)), total_times, alpha=0.6, linewidth=0.8)
            ax.set_xlabel('Échantillon')
            ax.set_ylabel('Temps total (ms)')
            ax.set_title('Timeline des temps de rendu')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=np.mean(total_times), color='r', linestyle='--', label=f'Moyenne: {np.mean(total_times):.1f} ms')
            ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Graphique 2: Distribution des temps par opération
        time_fields = ['draw_ms', 'extract_ms', 'filter_ms', 'baseline_ms', 'decim_ms']
        available_fields = [f for f in time_fields if any(r.get(f) is not None for r in data)]
        
        if available_fields:
            fig, ax = plt.subplots(figsize=(12, 8))
            data_to_plot = []
            labels = []
            for field in available_fields:
                values = [r.get(field) for r in data if r.get(field) is not None]
                if values:
                    data_to_plot.append(values)
                    labels.append(field.replace('_ms', ''))
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                ax.set_ylabel('Temps (ms)')
                ax.set_title('Distribution des temps par opération')
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Graphique 3: FPS au fil du temps
        fig, ax = plt.subplots(figsize=(12, 6))
        fps_values = [r.get('fps', 0) for r in data if r.get('fps') is not None]
        
        if fps_values:
            ax.plot(range(len(fps_values)), fps_values, alpha=0.6, linewidth=0.8, color='green')
            ax.set_xlabel('Échantillon')
            ax.set_ylabel('FPS')
            ax.set_title('FPS au fil du temps')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=np.mean(fps_values), color='r', linestyle='--', label=f'Moyenne: {np.mean(fps_values):.2f} FPS')
            ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Graphique 4: Comparaison par mode d'accès
        actions = {}
        for record in data:
            action = record.get('action', 'unknown')
            if action not in actions:
                actions[action] = []
            if record.get('total_ms') is not None:
                actions[action].append(record['total_ms'])
        
        if actions:
            fig, ax = plt.subplots(figsize=(12, 6))
            action_data = []
            action_labels = []
            for action, values in sorted(actions.items()):
                if values:
                    action_data.append(values)
                    action_labels.append(action.upper())
            
            if action_data:
                bp = ax.boxplot(action_data, labels=action_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightcoral')
                    patch.set_alpha(0.7)
                ax.set_ylabel('Temps total (ms)')
                ax.set_title('Comparaison des performances par mode d\'accès')
                ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Graphique 5: Heatmap de corrélation entre les opérations
        numeric_fields = ['total_ms', 'draw_ms', 'extract_ms', 'filter_ms', 'baseline_ms', 'decim_ms', 'fps']
        available_numeric = [f for f in numeric_fields if any(r.get(f) is not None for r in data)]
        
        if len(available_numeric) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            # Créer une matrice de corrélation
            correlation_data = []
            for field in available_numeric:
                values = [r.get(field) for r in data if r.get(field) is not None]
                correlation_data.append(values)
            
            # Aligner les longueurs (prendre le minimum)
            min_len = min(len(v) for v in correlation_data if v)
            correlation_data = [v[:min_len] for v in correlation_data]
            
            if len(correlation_data) > 1 and all(len(v) == min_len for v in correlation_data):
                corr_matrix = np.corrcoef(correlation_data)
                im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(available_numeric)))
                ax.set_yticks(range(len(available_numeric)))
                ax.set_xticklabels([f.replace('_ms', '') for f in available_numeric], rotation=45, ha='right')
                ax.set_yticklabels([f.replace('_ms', '') for f in available_numeric])
                ax.set_title('Matrice de corrélation entre opérations')
                
                # Ajouter les valeurs de corrélation
                for i in range(len(available_numeric)):
                    for j in range(len(available_numeric)):
                        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=8)
                
                plt.colorbar(im, ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyse les données de performance de la visualisation')
    parser.add_argument('--csv', type=str, default='logs/telemetry.csv',
                       help='Chemin vers le fichier CSV de télémetrie')
    parser.add_argument('--output-dir', type=str, default='logs',
                       help='Répertoire de sortie pour les rapports')
    parser.add_argument('--report', type=str, default='performance_report.txt',
                       help='Nom du fichier de rapport texte')
    parser.add_argument('--plots', type=str, default='performance_plots.pdf',
                       help='Nom du fichier PDF avec les graphiques')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Chargement des données depuis {csv_path}...")
    data = load_telemetry_data(csv_path)
    print(f"  {len(data)} échantillons chargés")
    
    if not data:
        print("Aucune donnée à analyser")
        return
    
    # Générer le rapport texte
    report_path = output_dir / args.report
    print(f"Génération du rapport texte: {report_path}...")
    generate_summary_report(data, report_path)
    print("  Rapport généré avec succès")
    
    # Générer les graphiques
    plots_path = output_dir / args.plots
    print(f"Génération des graphiques: {plots_path}...")
    try:
        generate_plots(data, plots_path)
        print("  Graphiques générés avec succès")
    except Exception as e:
        print(f"  Erreur lors de la génération des graphiques: {e}")
    
    print("\nAnalyse terminée!")
    print(f"Rapport: {report_path}")
    print(f"Graphiques: {plots_path}")


if __name__ == '__main__':
    main()

