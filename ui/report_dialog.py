# ui/report_dialog.py
"""Module pour la génération de rapports et le diagnostic - Version complète"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import os
import platform
import sys
from datetime import datetime

class ReportDialog:
    """Gestionnaire des rapports et du diagnostic système"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.parent = parent_app.root
        
    def report_bug(self):
        """Signaler un bug - crée un fichier de rapport avec logs et checkpoints"""
        try:
            # Créer le nom de fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bugreport_filename = f"bug_report_{timestamp}.txt"
            
            # Générer le contenu du rapport
            report_content = self.generate_bug_report()
            
            # Créer le chemin complet
            fullpath = os.path.abspath(bugreport_filename)
            
            # Écrire le fichier
            with open(bugreport_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Demander si l'utilisateur veut ouvrir le fichier
            open_file = messagebox.askyesno(
                "Rapport de Bug Créé", 
                f"Rapport de bug créé avec succès !\n\n"
                f"Fichier : {bugreport_filename}\n"
                f"Emplacement : {fullpath}\n\n"
                f"Le fichier contient :\n"
                f"• Informations système\n"
                f"• État de l'application\n"
                f"• Logs et checkpoints récents\n"
                f"• Configuration actuelle\n"
                f"• Erreurs récentes\n\n"
                f"Voulez-vous ouvrir le fichier maintenant ?"
            )
            
            if open_file:
                try:
                    # Ouvrir le fichier avec l'application par défaut
                    import subprocess
                    system = platform.system()
                    if system == "Windows":
                        os.startfile(fullpath)
                    elif system == "Darwin":  # macOS
                        subprocess.run(["open", fullpath])
                    else:  # Linux
                        subprocess.run(["xdg-open", fullpath])
                except Exception as e:
                    print(f"❌ RAPPORT BUG: Impossible d'ouvrir le fichier: {e}")
                    messagebox.showwarning(
                        "Attention", 
                        f"Le fichier a été créé mais n'a pas pu être ouvert automatiquement.\n{fullpath}"
                    )
            
            messagebox.showinfo(
                "Instructions",
                "Veuillez joindre ce fichier à votre signalement de bug avec :\n\n"
                "• Une description détaillée du problème\n"
                "• Les étapes pour reproduire le bug\n"
                "• Le fichier EDF si applicable\n"
                "• Une capture d'écran si possible"
            )
            
            print(f"✅ RAPPORT BUG: Fichier créé - {fullpath}")
            logging.info(f"BUGREPORT: Bug report created {fullpath}")
            
        except Exception as e:
            error_msg = f"Erreur lors de la création du rapport de bug : {str(e)}"
            print(f"❌ RAPPORT BUG: {error_msg}")
            logging.error(f"BUGREPORT: Failed to create bug report: {e}")
            messagebox.showerror("Erreur", error_msg)
    
    def generate_bug_report(self):
        """Génère le contenu du rapport de bug avec toutes les informations pertinentes"""
        try:
            report_lines = []
            
            # En-tête du rapport
            report_lines.append("=" * 80)
            report_lines.append("EEG ANALYSIS STUDIO - RAPPORT DE BUG")
            report_lines.append("=" * 80)
            report_lines.append(f"Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Version de l'application : v3.0")
            report_lines.append("")
            
            # Informations système
            report_lines.append("INFORMATIONS SYSTÈME")
            report_lines.append("-" * 40)
            report_lines.append(f"Système d'exploitation : {platform.system()} {platform.release()}")
            report_lines.append(f"Architecture : {platform.architecture()[0]}")
            report_lines.append(f"Version Python : {sys.version}")
            report_lines.append(f"Répertoire de travail : {os.getcwd()}")
            report_lines.append("")
            
            # État de l'application
            report_lines.append("ÉTAT DE L'APPLICATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Fichier EDF chargé : {'Oui' if hasattr(self.parent_app, 'raw') and self.parent_app.raw is not None else 'Non'}")
            
            if hasattr(self.parent_app, 'raw') and self.parent_app.raw is not None:
                report_lines.append(f"  - Nom du fichier : {getattr(self.parent_app, 'current_file', 'Inconnu')}")
                report_lines.append(f"  - Fréquence d'échantillonnage : {self.parent_app.sfreq} Hz")
                report_lines.append(f"  - Nombre de canaux : {len(self.parent_app.raw.ch_names)}")
                report_lines.append(f"  - Durée totale : {len(self.parent_app.raw.times):.1f} secondes")
            
            report_lines.append(f"Temps actuel affiché : {getattr(self.parent_app, 'current_time', 0):.1f}s")
            report_lines.append(f"Durée d'affichage : {getattr(self.parent_app, 'duration', 10):.1f}s")
            report_lines.append(f"Canaux sélectionnés : {len(getattr(self.parent_app, 'selected_channels', []))}")
            
            if hasattr(self.parent_app, 'selected_channels'):
                report_lines.append(f"  - Liste : {', '.join(self.parent_app.selected_channels[:5])}{'...' if len(self.parent_app.selected_channels) > 5 else ''}")
            report_lines.append("")
            
            # Informations temporelles
            if hasattr(self.parent_app, 'absolute_start_datetime') and self.parent_app.absolute_start_datetime:
                report_lines.append("INFORMATIONS TEMPORELLES")
                report_lines.append("-" * 40)
                report_lines.append(f"Début enregistrement EDF : {self.parent_app.absolute_start_datetime}")
                if hasattr(self.parent_app, 'display_start_datetime') and self.parent_app.display_start_datetime:
                    report_lines.append(f"Base d'affichage Excel : {self.parent_app.display_start_datetime}")
                report_lines.append("")
            
            # Configuration du scoring
            report_lines.append("CONFIGURATION DU SCORING")
            report_lines.append("-" * 40)
            report_lines.append(f"Scoring automatique chargé : {'Oui' if hasattr(self.parent_app, 'sleep_scoring_data') and self.parent_app.sleep_scoring_data is not None else 'Non'}")
            report_lines.append(f"Scoring manuel chargé : {'Oui' if hasattr(self.parent_app, 'manual_scoring_data') and self.parent_app.manual_scoring_data is not None else 'Non'}")
            report_lines.append(f"Durée d'époque : {getattr(self.parent_app, 'scoring_epoch_duration', 30.0)}s")
            report_lines.append(f"Affichage scoring : {'Manuel' if getattr(self.parent_app, 'show_manual_scoring', True) else 'Automatique'}")
            report_lines.append("")
            
            # Configuration du filtre
            report_lines.append("CONFIGURATION DU FILTRE")
            report_lines.append("-" * 40)
            report_lines.append(f"Filtre activé : {'Oui' if getattr(self.parent_app, 'filter_enabled', False) else 'Non'}")
            report_lines.append(f"Filtre bas : {getattr(self.parent_app, 'filter_low', 0.5)} Hz")
            report_lines.append(f"Filtre haut : {getattr(self.parent_app, 'filter_high', 30.0)} Hz")
            report_lines.append(f"Type de filtre : {getattr(self.parent_app, 'filter_type', 'butterworth')}")
            report_lines.append(f"Ordre du filtre : {getattr(self.parent_app, 'filter_order', 4)}")
            report_lines.append("")
            
            # Logs récents
            report_lines.append("LOGS ET CHECKPOINTS RÉCENTS")
            report_lines.append("-" * 40)
            recent_logs = self.get_recent_logs()
            if recent_logs:
                report_lines.append("Dernières 20 lignes de log :")
                for log_line in recent_logs:
                    report_lines.append(f"  {log_line}")
            else:
                report_lines.append("Aucun log récent trouvé")
            
            # Checkpoints récents de la console
            recent_checkpoints = self.get_recent_checkpoints()
            if recent_checkpoints:
                report_lines.append("")
                report_lines.append("Derniers checkpoints de la console :")
                for checkpoint in recent_checkpoints:
                    report_lines.append(f"  {checkpoint}")
            report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Erreur lors de la génération du rapport : {str(e)}"
    
    def get_recent_logs(self):
        """Récupère les logs récents"""
        try:
            # Essayer de lire le fichier de log s'il existe
            log_files = ['eeg_studio.log', 'cesa.log', 'application.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        return lines[-20:] if len(lines) > 20 else lines
            return []
        except Exception:
            return []
    
    def get_recent_checkpoints(self):
        """Récupère les checkpoints récents de la console"""
        try:
            # Dans une vraie implémentation, on pourrait capturer les prints
            # Pour l'instant, retourner une liste vide
            return []
        except Exception:
            return []
    
    def export_data(self):
        """Exporte les données actuelles"""
        if not self.parent_app.raw or not getattr(self.parent_app, 'selected_channels', []):
            messagebox.showwarning("Attention", "Aucun fichier chargé ou canaux sélectionnés")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Exporter les données",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json"), ("TXT", "*.txt")]
        )
        
        if filepath:
            try:
                data_to_export = {}
                for channel in self.parent_app.selected_channels:
                    if hasattr(self.parent_app, 'derivations') and channel in self.parent_app.derivations:
                        data_to_export[channel] = self.parent_app.derivations[channel].tolist()
                
                if filepath.endswith('.json'):
                    import json
                    with open(filepath, 'w') as f:
                        json.dump(data_to_export, f, indent=2)
                elif filepath.endswith('.csv'):
                    import pandas as pd
                    df = pd.DataFrame(data_to_export)
                    df.to_csv(filepath, index=False)
                else:  # TXT
                    with open(filepath, 'w') as f:
                        f.write("Données EEG Exportées\n")
                        f.write("=" * 30 + "\n")
                        for channel, data in data_to_export.items():
                            f.write(f"Canal: {channel}\n")
                            f.write(f"Échantillons: {len(data)}\n")
                            f.write(f"Min: {min(data):.6f} V\n")
                            f.write(f"Max: {max(data):.6f} V\n")
                
                messagebox.showinfo("Succès", f"Données exportées vers {filepath}")
                logging.info(f"Données exportées vers {filepath}")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'export : {str(e)}")
                logging.error(f"Erreur export: {e}")
    
    def generate_report(self):
        """Génère un rapport complet"""
        if not self.parent_app.raw:
            messagebox.showwarning("Attention", "Aucun fichier chargé")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Exporter le rapport",
            defaultextension=".txt",
            filetypes=[("TXT", "*.txt"), ("JSON", "*.json")]
        )
        
        if filepath:
            try:
                report = self.generate_analysis_report()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Succès", f"Rapport généré : {filepath}")
                logging.info(f"Rapport généré : {filepath}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la génération : {str(e)}")
                logging.error(f"Erreur rapport: {e}")
    
    def generate_analysis_report(self):
        """Génère un rapport textuel des données"""
        if not self.parent_app.raw:
            return "Aucun fichier chargé"
            
        report = []
        report.append("CESA EEG Studio Analysis - Rapport d'Analyse")
        report.append("=" * 50)
        report.append(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Fichier : {getattr(self.parent_app.raw.info, 'filename', 'Inconnu')}")
        report.append(f"Canaux : {len(self.parent_app.raw.ch_names)}")
        report.append(f"Fréquence : {self.parent_app.sfreq} Hz")
        report.append(f"Durée : {len(self.parent_app.raw.times)/self.parent_app.sfreq:.1f}s")
        report.append("")
        
        if getattr(self.parent_app, 'selected_channels', []):
            report.append("Canaux Sélectionnés :")
            for i, channel in enumerate(self.parent_app.selected_channels, 1):
                report.append(f"{i}. {channel}")
            report.append("")
            
            report.append("Statistiques par Canal :")
            for channel in self.parent_app.selected_channels:
                if hasattr(self.parent_app, 'derivations') and channel in self.parent_app.derivations:
                    data = self.parent_app.derivations[channel]
                    report.append(f"{channel} :")
                    report.append(f"  Échantillons : {len(data)}")
                    report.append(f"  Min : {np.min(data):.6f} V")
                    report.append(f"  Max : {np.max(data):.6f} V")
                    report.append(f"  Moyenne : {np.mean(data):.6f} V")
                    report.append(f"  Écart-type : {np.std(data):.6f} V")
                    report.append(f"  RMS : {np.sqrt(np.mean(data**2)):.6f} V")
        
        return "\n".join(report)
