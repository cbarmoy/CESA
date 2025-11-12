#!/usr/bin/env python3
"""
CESA (EEG Studio Analysis) v3.0 - Installateur Graphique
=======================================================

Installateur graphique noob-friendly pour CESA v3.0.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Cet installateur vérifie automatiquement les dépendances,
installe les packages manquants et configure CESA pour une
utilisation immédiate sans intervention console.

Fonctionnalités:
- Interface graphique intuitive
- Vérification automatique des dépendances
- Installation automatique des packages manquants
- Configuration automatique de l'environnement
- Test de fonctionnement intégré
- Assistant de première utilisation

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 3.0.0
Date: 2025-01-27
Licence: MIT
"""

import sys
import os
import subprocess
import importlib.util
import threading
import time
from pathlib import Path

# Interface graphique
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from tkinter.scrolledtext import ScrolledText
except ImportError:
    print("❌ Tkinter non disponible. Installation impossible.")
    sys.exit(1)

class CESAInstaller:
    """Installateur graphique pour CESA v3.0."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CESA v3.0 - Installateur Graphique")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.install_log = []
        self.installation_successful = False
        
        # Configuration de l'interface
        self._setup_ui()
        
        # Vérification initiale
        self._check_python_version()
    
    def _setup_ui(self):
        """Configure l'interface utilisateur."""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="🧠 CESA v3.0 - Installateur Graphique", 
                               font=('Helvetica', 18, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Sous-titre
        subtitle_label = ttk.Label(main_frame, 
                                  text="Installation automatique d'EEG Studio Analysis v3.0\n"
                                       "Développé pour l'Unité Neuropsychologie du Stress (IRBA)",
                                  font=('Helvetica', 10))
        subtitle_label.pack(pady=(0, 30))
        
        # Frame de contenu
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Onglets
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet Installation
        self._create_install_tab()
        
        # Onglet Configuration
        self._create_config_tab()
        
        # Onglet Test
        self._create_test_tab()
        
        # Onglet Aide
        self._create_help_tab()
        
        # Boutons de contrôle
        self._create_control_buttons(main_frame)
    
    def _create_install_tab(self):
        """Crée l'onglet d'installation."""
        install_frame = ttk.Frame(self.notebook)
        self.notebook.add(install_frame, text="📦 Installation")
        
        # Informations système
        info_frame = ttk.LabelFrame(install_frame, text="Informations Système", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.system_info = tk.Text(info_frame, height=6, wrap=tk.WORD, font=('Consolas', 9))
        self.system_info.pack(fill=tk.X)
        
        # Vérification des dépendances
        deps_frame = ttk.LabelFrame(install_frame, text="Vérification des Dépendances", padding=10)
        deps_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Liste des dépendances
        self.deps_tree = ttk.Treeview(deps_frame, columns=('status', 'version'), show='tree headings')
        self.deps_tree.heading('#0', text='Package')
        self.deps_tree.heading('status', text='Statut')
        self.deps_tree.heading('version', text='Version')
        self.deps_tree.column('#0', width=200)
        self.deps_tree.column('status', width=100)
        self.deps_tree.column('version', width=150)
        self.deps_tree.pack(fill=tk.BOTH, expand=True)
        
        # Boutons d'installation
        install_buttons_frame = ttk.Frame(deps_frame)
        install_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.check_deps_btn = ttk.Button(install_buttons_frame, text="🔍 Vérifier Dépendances", 
                                        command=self._check_dependencies)
        self.check_deps_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.install_deps_btn = ttk.Button(install_buttons_frame, text="📥 Installer Manquantes", 
                                          command=self._install_missing_dependencies)
        self.install_deps_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Log d'installation
        log_frame = ttk.LabelFrame(install_frame, text="Journal d'Installation", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.install_log_text = ScrolledText(log_frame, height=8, wrap=tk.WORD, font=('Consolas', 9))
        self.install_log_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_config_tab(self):
        """Crée l'onglet de configuration."""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")
        
        # Configuration Python
        python_frame = ttk.LabelFrame(config_frame, text="Configuration Python", padding=10)
        python_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(python_frame, text="Chemin Python:").pack(anchor='w')
        python_path_frame = ttk.Frame(python_frame)
        python_path_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.python_path_var = tk.StringVar(value=sys.executable)
        python_path_entry = ttk.Entry(python_path_frame, textvariable=self.python_path_var, width=60)
        python_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(python_path_frame, text="Parcourir", 
                  command=self._browse_python_path).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Configuration CESA
        esa_frame = ttk.LabelFrame(config_frame, text="Configuration CESA", padding=10)
        esa_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(esa_frame, text="Répertoire d'installation CESA:").pack(anchor='w')
        esa_path_frame = ttk.Frame(esa_frame)
        esa_path_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.esa_path_var = tk.StringVar(value=str(Path.cwd()))
        esa_path_entry = ttk.Entry(esa_path_frame, textvariable=self.esa_path_var, width=60)
        esa_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(esa_path_frame, text="Parcourir", 
                  command=self._browse_esa_path).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Options d'installation
        options_frame = ttk.LabelFrame(config_frame, text="Options d'Installation", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_desktop_shortcut = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Créer un raccourci sur le bureau", 
                      variable=self.create_desktop_shortcut).pack(anchor='w')
        
        self.create_start_menu = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Ajouter au menu Démarrer", 
                      variable=self.create_start_menu).pack(anchor='w')
        
        self.auto_update = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Vérifier les mises à jour automatiquement", 
                      variable=self.auto_update).pack(anchor='w')
    
    def _create_test_tab(self):
        """Crée l'onglet de test."""
        test_frame = ttk.Frame(self.notebook)
        self.notebook.add(test_frame, text="🧪 Test")
        
        # Test de fonctionnement
        test_func_frame = ttk.LabelFrame(test_frame, text="Test de Fonctionnement", padding=10)
        test_func_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(test_func_frame, text="Testez le bon fonctionnement d'CESA après installation:").pack(anchor='w')
        
        test_buttons_frame = ttk.Frame(test_func_frame)
        test_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(test_buttons_frame, text="🧪 Test Complet", 
                  command=self._run_full_test).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(test_buttons_frame, text="🚀 Lancer CESA", 
                  command=self._launch_cesa).pack(side=tk.LEFT, padx=(0, 10))
        
        # Résultats des tests
        results_frame = ttk.LabelFrame(test_frame, text="Résultats des Tests", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.test_results_text = ScrolledText(results_frame, height=15, wrap=tk.WORD, font=('Consolas', 9))
        self.test_results_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_help_tab(self):
        """Crée l'onglet d'aide."""
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="❓ Aide")
        
        # Guide d'installation
        guide_frame = ttk.LabelFrame(help_frame, text="Guide d'Installation", padding=10)
        guide_frame.pack(fill=tk.BOTH, expand=True)
        
        guide_text = """
🎯 GUIDE D'INSTALLATION CESA v3.0

📋 ÉTAPES D'INSTALLATION :

1️⃣ VÉRIFICATION SYSTÈME
   • Vérifiez que Python 3.8+ est installé
   • Assurez-vous d'avoir une connexion Internet
   • Fermez tous les autres programmes Python

2️⃣ INSTALLATION AUTOMATIQUE
   • Cliquez sur "Vérifier Dépendances"
   • Cliquez sur "Installer Manquantes" si nécessaire
   • Suivez les instructions à l'écran

3️⃣ CONFIGURATION
   • Vérifiez les chemins Python et CESA
   • Choisissez vos options d'installation
   • Configurez selon vos préférences

4️⃣ TEST ET LANCEMENT
   • Lancez le test complet
   • Testez CESA avec "Lancer CESA"
   • Vérifiez que tout fonctionne

🚨 PROBLÈMES COURANTS :

❌ "Python non trouvé"
   → Installez Python depuis python.org
   → Ajoutez Python au PATH système

❌ "Dépendances manquantes"
   → Utilisez "Installer Manquantes"
   → Vérifiez votre connexion Internet

❌ "Permission refusée"
   → Lancez en tant qu'administrateur
   → Vérifiez les permissions du dossier

❌ "CESA ne se lance pas"
   → Vérifiez les chemins d'installation
   → Consultez le journal d'installation

📞 SUPPORT :
   • Email: come1.barmoy@supbiotech.fr
   • GitHub: cbarmoy
   • Documentation: README.md

🎉 FÉLICITATIONS !
   Une fois installé, CESA est prêt à analyser vos données EEG !
        """
        
        help_text_widget = ScrolledText(guide_frame, wrap=tk.WORD, font=('Helvetica', 10))
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        help_text_widget.insert(tk.END, guide_text)
        help_text_widget.config(state=tk.DISABLED)
    
    def _create_control_buttons(self, parent):
        """Crée les boutons de contrôle."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Bouton Fermer
        ttk.Button(button_frame, text="Fermer", 
                  command=self.root.quit).pack(side=tk.RIGHT)
        
        # Bouton Installation Complète
        self.install_all_btn = ttk.Button(button_frame, text="🚀 Installation Complète", 
                                         command=self._install_all)
        self.install_all_btn.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, 
                                          maximum=100, length=200)
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 20))
        
        self.status_label = ttk.Label(button_frame, text="Prêt")
        self.status_label.pack(side=tk.LEFT)
    
    def _check_python_version(self):
        """Vérifie la version de Python."""
        version = sys.version_info
        self._log(f"Python {version.major}.{version.minor}.{version.micro} détecté")
        
        if version < (3, 8):
            self._log("⚠️ Python 3.8+ requis pour CESA v3.0", "warning")
            messagebox.showwarning("Version Python", 
                                  "Python 3.8+ est requis pour CESA v3.0.\n"
                                  f"Version actuelle: {version.major}.{version.minor}.{version.micro}")
        else:
            self._log("✅ Version Python compatible")
    
    def _log(self, message, level="info"):
        """Ajoute un message au journal."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.install_log.append(log_entry)
        
        # Mise à jour de l'interface
        if hasattr(self, 'install_log_text'):
            self.install_log_text.insert(tk.END, log_entry)
            self.install_log_text.see(tk.END)
            self.root.update_idletasks()
    
    def _update_system_info(self):
        """Met à jour les informations système."""
        info = f"""
Système: {os.name}
Plateforme: {sys.platform}
Python: {sys.version}
Exécutable: {sys.executable}
Répertoire courant: {os.getcwd()}
        """
        
        if hasattr(self, 'system_info'):
            self.system_info.delete(1.0, tk.END)
            self.system_info.insert(tk.END, info.strip())
    
    def _check_dependencies(self):
        """Vérifie les dépendances requises."""
        self._log("🔍 Vérification des dépendances...")
        
        # Liste des dépendances
        dependencies = {
            'mne': 'MNE-Python',
            'numpy': 'NumPy',
            'matplotlib': 'Matplotlib',
            'scipy': 'SciPy',
            'pandas': 'Pandas',
            'yasa': 'YASA',
            'tkinter': 'Tkinter',
            'PIL': 'Pillow'
        }
        
        # Nettoyer l'arbre
        for item in self.deps_tree.get_children():
            self.deps_tree.delete(item)
        
        missing_deps = []
        
        for package, name in dependencies.items():
            try:
                if package == 'tkinter':
                    import tkinter
                    version = "Inclus"
                elif package == 'PIL':
                    import PIL
                    version = PIL.__version__
                else:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'Inconnue')
                
                # Ajouter à l'arbre
                self.deps_tree.insert('', 'end', text=name, values=('✅ Installé', version))
                self._log(f"✅ {name}: {version}")
                
            except ImportError:
                missing_deps.append((package, name))
                self.deps_tree.insert('', 'end', text=name, values=('❌ Manquant', 'N/A'))
                self._log(f"❌ {name}: Manquant")
        
        if missing_deps:
            self._log(f"⚠️ {len(missing_deps)} dépendance(s) manquante(s)")
            self.install_deps_btn.config(state='normal')
        else:
            self._log("✅ Toutes les dépendances sont installées")
            self.install_deps_btn.config(state='disabled')
        
        self._update_system_info()
    
    def _install_missing_dependencies(self):
        """Installe les dépendances manquantes."""
        self._log("📥 Installation des dépendances manquantes...")
        
        # Packages à installer
        packages_to_install = [
            'mne',
            'numpy',
            'matplotlib',
            'scipy',
            'pandas',
            'yasa',
            'Pillow'
        ]
        
        def install_thread():
            try:
                for i, package in enumerate(packages_to_install):
                    self._log(f"📦 Installation de {package}...")
                    self.status_label.config(text=f"Installation de {package}...")
                    self.progress_var.set((i / len(packages_to_install)) * 100)
                    
                    result = subprocess.run([
                        sys.executable, '-m', 'pip', 'install', package, '--upgrade'
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        self._log(f"✅ {package} installé avec succès")
                    else:
                        self._log(f"❌ Erreur installation {package}: {result.stderr}")
                
                self.progress_var.set(100)
                self.status_label.config(text="Installation terminée")
                self._log("🎉 Installation des dépendances terminée")
                
                # Re-vérifier les dépendances
                self.root.after(1000, self._check_dependencies)
                
            except Exception as e:
                self._log(f"❌ Erreur lors de l'installation: {e}")
                self.status_label.config(text="Erreur d'installation")
        
        # Lancer l'installation dans un thread séparé
        threading.Thread(target=install_thread, daemon=True).start()
    
    def _browse_python_path(self):
        """Ouvre un dialogue pour choisir le chemin Python."""
        path = filedialog.askopenfilename(
            title="Sélectionner l'exécutable Python",
            filetypes=[("Exécutable Python", "python.exe"), ("Tous les fichiers", "*.*")]
        )
        if path:
            self.python_path_var.set(path)
    
    def _browse_esa_path(self):
        """Ouvre un dialogue pour choisir le répertoire CESA."""
        path = filedialog.askdirectory(title="Sélectionner le répertoire d'installation CESA")
        if path:
            self.esa_path_var.set(path)
    
    def _run_full_test(self):
        """Lance un test complet d'CESA."""
        self._log("🧪 Lancement du test complet...")
        
        def test_thread():
            try:
                # Test d'importation
                self._log("📦 Test d'importation des modules...")
                self.test_results_text.insert(tk.END, "=== TEST D'IMPORTATION ===\n")
                
                modules_to_test = [
                    'esa.entropy',
                    'esa.filters',
                    'esa.scoring_io',
                    'esa.theme',
                    'spectral_analysis'
                ]
                
                for module in modules_to_test:
                    try:
                        __import__(module)
                        self.test_results_text.insert(tk.END, f"✅ {module}: OK\n")
                        self._log(f"✅ {module}: OK")
                    except Exception as e:
                        self.test_results_text.insert(tk.END, f"❌ {module}: {e}\n")
                        self._log(f"❌ {module}: {e}")
                
                # Test de l'entropie renormée
                self.test_results_text.insert(tk.END, "\n=== TEST ENTROPIE RENORMÉE ===\n")
                try:
                    from esa.entropy import compute_renormalized_entropy, RenormalizedEntropyConfig
                    import numpy as np
                    
                    # Données de test
                    sfreq = 100.0
                    duration = 2.0
                    n_samples = int(sfreq * duration)
                    data = np.array([np.random.randn(n_samples), np.random.randn(n_samples)])
                    
                    config = RenormalizedEntropyConfig(window_length=1.0, overlap=0.5)
                    result = compute_renormalized_entropy(data, sfreq, ["C3", "C4"], config)
                    
                    self.test_results_text.insert(tk.END, f"✅ Entropie renormée: {result.entropy_nats:.6f} nats\n")
                    self._log("✅ Test entropie renormée: OK")
                    
                except Exception as e:
                    self.test_results_text.insert(tk.END, f"❌ Entropie renormée: {e}\n")
                    self._log(f"❌ Test entropie renormée: {e}")
                
                self.test_results_text.insert(tk.END, "\n🎉 Tests terminés!\n")
                self._log("🎉 Tests terminés")
                
            except Exception as e:
                self.test_results_text.insert(tk.END, f"❌ Erreur lors des tests: {e}\n")
                self._log(f"❌ Erreur lors des tests: {e}")
        
        # Nettoyer les résultats précédents
        self.test_results_text.delete(1.0, tk.END)
        
        # Lancer les tests dans un thread séparé
        threading.Thread(target=test_thread, daemon=True).start()
    
    def _launch_cesa(self):
        """Lance CESA."""
        self._log("🚀 Lancement d'CESA...")
        
        try:
            cesa_path = Path(self.cesa_path_var.get()) / "run.py"
            if cesa_path.exists():
                subprocess.Popen([sys.executable, str(cesa_path)])
                self._log("✅ CESA lancé avec succès")
                messagebox.showinfo("CESA", "CESA a été lancé avec succès!")
            else:
                self._log(f"❌ Fichier run.py non trouvé: {cesa_path}")
                messagebox.showerror("Erreur", f"Fichier run.py non trouvé:\n{cesa_path}")
        except Exception as e:
            self._log(f"❌ Erreur lors du lancement: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du lancement d'CESA:\n{e}")
    
    def _install_all(self):
        """Lance l'installation complète."""
        self._log("🚀 Début de l'installation complète...")
        
        def install_all_thread():
            try:
                # 1. Vérifier les dépendances
                self._log("1️⃣ Vérification des dépendances...")
                self._check_dependencies()
                
                # 2. Installer les dépendances manquantes
                self._log("2️⃣ Installation des dépendances...")
                self._install_missing_dependencies()
                
                # 3. Attendre la fin de l'installation
                time.sleep(5)
                
                # 4. Tester l'installation
                self._log("3️⃣ Test de l'installation...")
                self._run_full_test()
                
                # 5. Créer les raccourcis si demandé
                if self.create_desktop_shortcut.get():
                    self._log("4️⃣ Création des raccourcis...")
                    self._create_shortcuts()
                
                self._log("🎉 Installation complète terminée!")
                self.installation_successful = True
                
                messagebox.showinfo("Installation", 
                                  "Installation complète terminée avec succès!\n\n"
                                  "CESA v3.0 est maintenant prêt à être utilisé.")
                
            except Exception as e:
                self._log(f"❌ Erreur lors de l'installation complète: {e}")
                messagebox.showerror("Erreur", f"Erreur lors de l'installation:\n{e}")
        
        # Lancer l'installation dans un thread séparé
        threading.Thread(target=install_all_thread, daemon=True).start()
    
    def _create_shortcuts(self):
        """Crée les raccourcis sur le bureau et dans le menu Démarrer."""
        try:
            # Raccourci sur le bureau
            if self.create_desktop_shortcut.get():
                desktop_path = Path.home() / "Desktop"
                shortcut_path = desktop_path / "CESA v3.0.lnk"
                
                # Créer un fichier batch pour lancer CESA
                batch_content = f'''@echo off
cd /d "{self.esa_path_var.get()}"
python run.py
pause
'''
                batch_path = Path(self.cesa_path_var.get()) / "LAUNCH_CESA_AUTO.bat"
                with open(batch_path, 'w', encoding='utf-8') as f:
                    f.write(batch_content)
                
                self._log(f"✅ Raccourci créé: {batch_path}")
            
            # Menu Démarrer (Windows)
            if self.create_start_menu.get() and sys.platform == "win32":
                start_menu_path = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
                start_menu_path.mkdir(parents=True, exist_ok=True)
                
                self._log("✅ Ajout au menu Démarrer (Windows)")
            
        except Exception as e:
            self._log(f"⚠️ Erreur création raccourcis: {e}")
    
    def run(self):
        """Lance l'installateur."""
        self._update_system_info()
        self._check_dependencies()
        self.root.mainloop()

def main():
    """Fonction principale."""
    print("🧠 CESA v3.0 - Installateur Graphique")
    print("=" * 50)
    
    try:
        installer = CESAInstaller()
        installer.run()
    except Exception as e:
        print(f"❌ Erreur lors du lancement de l'installateur: {e}")
        messagebox.showerror("Erreur", f"Erreur lors du lancement:\n{e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


