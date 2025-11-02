# ui/shortcuts_dialog.py
"""Module pour la gestion des raccourcis clavier - Version complète avec toutes les fonctionnalités"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging

class ShortcutsDialog:
    """Gestionnaire de la fenêtre des raccourcis - Version complète"""
    
    def __init__(self, parent):
        self.parent = parent
        
    def show_shortcuts(self):
        """Affiche les raccourcis clavier dans une interface moderne - Version complète"""
        try:
            print("🔍 CHECKPOINT SHORTCUTS: Affichage des raccourcis")
            logging.info("[SHORTCUTS] Displaying shortcuts")
            
            # Créer la fenêtre des raccourcis
            shortcuts_window = tk.Toplevel(self.parent)
            shortcuts_window.title("Raccourcis Clavier - EEG Analysis Studio")
            shortcuts_window.geometry("700x800")
            shortcuts_window.transient(self.parent)
            shortcuts_window.grab_set()
            
            # Frame principal avec scrollbar
            main_frame = ttk.Frame(shortcuts_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Titre
            title_label = ttk.Label(main_frame, text="⌨️ Raccourcis Clavier", 
                                  font=('Segoe UI', 18, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # Canvas et scrollbar pour le contenu
            canvas = tk.Canvas(main_frame)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Définir les catégories de raccourcis (exactement comme l'original)
            shortcuts_categories = {
                "📁 Fichier": [
                    ("Ctrl+O", "Ouvrir fichier EDF"),
                    ("Ctrl+S", "Exporter données"),
                    ("Ctrl+Shift+S", "Exporter segment EDF"),
                    ("Ctrl+Shift+R", "Générer rapport complet"),
                    ("Ctrl+Q", "Quitter l'application")
                ],
                "🧭 Navigation Temporelle": [
                    ("Z", "Époque précédente"),
                    ("Q", "Époque précédente (alternatif)"),
                    ("S", "Époque suivante"),
                    ("D", "Époque suivante (alternatif)"),
                    ("←", "Navigation temporelle (gauche)"),
                    ("→", "Navigation temporelle (droite)"),
                    ("Ctrl+←", "Navigation rapide (gauche)"),
                    ("Ctrl+→", "Navigation rapide (droite)"),
                    ("Home", "Aller au début"),
                    ("End", "Aller à la fin")
                ],
                "📊 Amplitude et Zoom": [
                    ("↑", "Augmenter amplitude"),
                    ("↓", "Diminuer amplitude"),
                    ("Ctrl+↑", "Augmenter amplitude (grand pas)"),
                    ("Ctrl+↓", "Diminuer amplitude (grand pas)"),
                    ("+", "Zoom avant"),
                    ("-", "Zoom arrière"),
                    ("Ctrl+0", "Réinitialiser zoom")
                ],
                "🎯 Scoring de Sommeil": [
                    ("Ctrl+Y", "Scoring automatique (YASA)"),
                    ("Ctrl+Shift+M", "Importer scoring manuel (Excel)"),
                    ("Ctrl+C", "Comparer scoring auto vs manuel"),
                    ("1", "Marquer comme Éveil (W)"),
                    ("2", "Marquer comme N1"),
                    ("3", "Marquer comme N2"),
                    ("4", "Marquer comme N3"),
                    ("5", "Marquer comme REM (R)")
                ],
                "⚙️ Interface et Affichage": [
                    ("Ctrl+A", "Activer/Désactiver autoscale"),
                    ("Ctrl+F", "Activer/Désactiver filtre"),
                    ("Ctrl+T", "Basculer thème sombre/clair"),
                    ("Ctrl+1", "Sélectionner canaux"),
                    ("Ctrl+P", "Basculer panneau commandes"),
                    ("F2", "Basculer panneau commandes (alternatif)"),
                    ("F5", "Actualiser graphique"),
                    ("Ctrl+R", "Actualiser graphique (alternatif)")
                ],
                "📈 Analyse et Outils": [
                    ("Ctrl+Shift+T", "Analyse temporelle"),
                    ("Ctrl+Shift+K", "Système de marqueurs"),
                    ("Ctrl+Shift+L", "Outils de mesure"),
                    ("Ctrl+Shift+F", "Configuration filtres avancée"),
                    ("Ctrl+Shift+P", "Analyse spectrale (PSD)"),
                    ("Ctrl+Shift+W", "Analyse TFR (Time-Frequency)")
                ],
                "🐛 Debug et Support": [
                    ("Ctrl+Shift+B", "Générer rapport de bug"),
                    ("Ctrl+Shift+D", "Afficher informations debug"),
                    ("Ctrl+Shift+I", "Informations système")
                ],
                "❓ Aide et Navigation": [
                    ("F1", "Assistant de bienvenue"),
                    ("F3", "Rechercher dans les données"),
                    ("F4", "Afficher statistiques canaux"),
                    ("Escape", "Fermer les dialogues"),
                    ("Ctrl+?", "Afficher cette aide (raccourcis)")
                ]
            }
            
            # Créer les sections pour chaque catégorie
            for category, shortcuts in shortcuts_categories.items():
                # Frame pour la catégorie
                category_frame = ttk.LabelFrame(scrollable_frame, text=category, padding=10)
                category_frame.pack(fill=tk.X, pady=(0, 15))
                
                # Créer un frame interne pour les raccourcis
                shortcuts_frame = ttk.Frame(category_frame)
                shortcuts_frame.pack(fill=tk.X)
                
                # Ajouter chaque raccourci
                for i, (shortcut, description) in enumerate(shortcuts):
                    # Frame pour chaque raccourci
                    shortcut_frame = ttk.Frame(shortcuts_frame)
                    shortcut_frame.pack(fill=tk.X, pady=2)
                    
                    # Raccourci (en gras et coloré)
                    shortcut_label = ttk.Label(shortcut_frame, text=shortcut, 
                                             font=('Consolas', 10, 'bold'))
                    shortcut_label.pack(side=tk.LEFT, padx=(0, 15))
                    
                    # Description
                    desc_label = ttk.Label(shortcut_frame, text=description, 
                                         font=('Segoe UI', 9))
                    desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Ajouter une section d'informations supplémentaires
            info_frame = ttk.LabelFrame(scrollable_frame, text="ℹ️ Informations", padding=10)
            info_frame.pack(fill=tk.X, pady=(10, 0))
            
            info_text = """• Les raccourcis ZQSD fonctionnent uniquement quand la fenêtre principale a le focus
• Les flèches directionnelles peuvent temporairement désactiver ZQSD (utilisez les raccourcis avec Ctrl pour réactiver)
• Certains raccourcis peuvent varier selon le contexte (dialogue ouvert, etc.)
• Utilisez Escape pour fermer la plupart des fenêtres et dialogues
• Les raccourcis sont également disponibles dans les menus contextuels"""
            
            info_label = ttk.Label(info_frame, text=info_text.strip(), 
                                 font=('Segoe UI', 8), justify=tk.LEFT)
            info_label.pack(anchor='w')
            
            # Boutons de contrôle
            button_frame = ttk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=(20, 0))
            
            def print_shortcuts():
                """Imprime les raccourcis dans la console."""
                try:
                    print("\n" + "="*60)
                    print("📋 RACCOURCIS CLAVIER - EEG ANALYSIS STUDIO")
                    print("="*60)
                    
                    for category, shortcuts in shortcuts_categories.items():
                        print(f"\n{category}")
                        print("-" * len(category))
                        for shortcut, description in shortcuts:
                            print(f"  {shortcut:<20} - {description}")
                    
                    print("\n" + "="*60)
                    print("✅ CHECKPOINT SHORTCUTS: Raccourcis imprimés dans la console")
                    logging.info("[SHORTCUTS] Shortcuts printed to console")
                    
                except Exception as e:
                    print(f"❌ CHECKPOINT SHORTCUTS: Erreur impression: {e}")
                    logging.error(f"[SHORTCUTS] Failed to print shortcuts: {e}")
            
            def copy_shortcuts():
                """Copie les raccourcis dans le presse-papiers."""
                try:
                    shortcuts_text = "RACCOURCIS CLAVIER - EEG ANALYSIS STUDIO\n"
                    shortcuts_text += "="*50 + "\n\n"
                    
                    for category, shortcuts in shortcuts_categories.items():
                        shortcuts_text += f"{category}\n"
                        shortcuts_text += "-" * len(category) + "\n"
                        for shortcut, description in shortcuts:
                            shortcuts_text += f"{shortcut:<20} - {description}\n"
                        shortcuts_text += "\n"
                    
                    # Copier dans le presse-papiers (Windows)
                    try:
                        shortcuts_window.clipboard_clear()
                        shortcuts_window.clipboard_append(shortcuts_text)
                        shortcuts_window.update()  # Nécessaire pour que la copie prenne effet
                        messagebox.showinfo("Succès", "Raccourcis copiés dans le presse-papiers !")
                        print("✅ CHECKPOINT SHORTCUTS: Raccourcis copiés dans le presse-papiers")
                        logging.info("[SHORTCUTS] Shortcuts copied to clipboard")
                    except Exception:
                        messagebox.showwarning("Attention", "Impossible de copier dans le presse-papiers")
                        
                except Exception as e:
                    print(f"❌ CHECKPOINT SHORTCUTS: Erreur copie: {e}")
                    logging.error(f"[SHORTCUTS] Failed to copy shortcuts: {e}")
                    messagebox.showerror("Erreur", f"Erreur lors de la copie : {str(e)}")
            
            # Boutons avec emojis
            ttk.Button(button_frame, text="📋 Imprimer dans la Console", 
                      command=print_shortcuts).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="📋 Copier dans le Presse-papiers", 
                      command=copy_shortcuts).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Fermer", 
                      command=shortcuts_window.destroy).pack(side=tk.RIGHT)
            
            # Configuration du canvas et scrollbar
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Lier la molette de la souris au scroll
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            def _unbind_mousewheel(event):
                canvas.unbind_all("<MouseWheel>")
            shortcuts_window.bind("<Destroy>", _unbind_mousewheel)
            
            # Raccourci pour fermer la fenêtre
            shortcuts_window.bind("<Escape>", lambda e: shortcuts_window.destroy())
            shortcuts_window.bind("<Control-?>", lambda e: shortcuts_window.destroy())
            
            # Focus sur la fenêtre
            shortcuts_window.focus_set()
            
            print("✅ CHECKPOINT SHORTCUTS: Interface raccourcis affichée")
            logging.info("[SHORTCUTS] Shortcuts interface displayed")
            
        except Exception as e:
            print(f"❌ CHECKPOINT SHORTCUTS: Erreur affichage raccourcis: {e}")
            logging.error(f"[SHORTCUTS] Failed to show shortcuts: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des raccourcis : {str(e)}")
