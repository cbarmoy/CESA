#!/usr/bin/env python3
"""
CESA (Complex EEG Studio Analysis) 0.0alpha3.0 - Main Launcher
=============================================================

Script de lancement principal pour l'application CESA.
Lance l'interface EEG Studio avec vérification des dépendances.

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0alpha3.0
Date: 2025-11-01
"""

import sys
import os
import importlib.util
import logging
import io

# Configuration de l'encodage UTF-8 pour Windows
# Permet l'affichage correct des emojis et caractères Unicode dans la console
if sys.platform == 'win32':
    try:
        # Forcer UTF-8 pour stdout et stderr
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # Si ça échoue, continuer avec l'encodage par défaut

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


def _find_logo_path():
    """Trouve le chemin du logo IRBA."""
    base = os.path.dirname(os.path.abspath(__file__))
    logo_dir = os.path.join(base, 'CESA', 'logo')
    candidates = ['logo_IRBA.jpg', 'logo_IRBA.png', 'logo.png', 'logo.jpg']
    for name in candidates:
        p = os.path.join(logo_dir, name)
        if os.path.exists(p):
            return p
    return None


def _find_logo_esa_path():
    """Trouve le chemin du logo CESA."""
    base = os.path.dirname(os.path.abspath(__file__))
    logo_dir = os.path.join(base, 'CESA', 'logo')
    candidates = ['logo_CESA.png', 'logo_CESA.jpg']
    for name in candidates:
        p = os.path.join(logo_dir, name)
        if os.path.exists(p):
            return p
    return None


def show_splash():
    """Affiche un écran de démarrage avec les logos."""
    try:
        import tkinter as tk
        try:
            from PIL import Image, ImageTk
        except Exception:
            Image = None
            ImageTk = None
            
        root = tk.Tk()
        # Icône du splash (Windows tâchebar) -> logo CESA
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            ico_path = os.path.join(base, 'CESA', 'logo', 'Icone_CESA.ico')
            if os.path.exists(ico_path):
                root.iconbitmap(ico_path)
        except Exception:
            pass
        root.overrideredirect(True)
        root.attributes('-topmost', True)
        
        try:
            root.configure(bg='black')
        except Exception:
            pass

        # Charger les logos
        logo_path = _find_logo_path()
        def _load_img(path, max_w, max_h):
            if not path:
                return None
            try:
                if Image and ImageTk:
                    im = Image.open(path)
                    im.thumbnail((max_w, max_h), Image.LANCZOS)
                    return ImageTk.PhotoImage(im)
                tmp = tk.PhotoImage(file=path)
                w, h = tmp.width(), tmp.height()
                scale = max(1, int(max(w / max_w, h / max_h)))
                return tmp.subsample(scale, scale) if scale > 1 else tmp
            except Exception:
                return None

        max_w, max_h = 320, 240
        img_main = _load_img(logo_path, max_w, max_h)
        img_esa = _load_img(_find_logo_esa_path(), max_w, max_h)

        if img_main is not None or img_esa is not None:
            frame = tk.Frame(root, bg='black', borderwidth=0, highlightthickness=0)
            frame._is_splash = True
            frame.pack()
            if img_main is not None:
                lbl1 = tk.Label(frame, image=img_main, bg='black', borderwidth=0, highlightthickness=0)
                lbl1._is_splash = True
                lbl1.image = img_main
                lbl1.pack(side=tk.LEFT, padx=8, pady=0)
            if img_esa is not None:
                lbl2 = tk.Label(frame, image=img_esa, bg='black', borderwidth=0, highlightthickness=0)
                lbl2._is_splash = True
                lbl2.image = img_esa
                lbl2.pack(side=tk.LEFT, padx=8, pady=0)
        else:
            lbl = tk.Label(root, text="CESA 0.0alpha3.0", fg='white', bg='black', font=('Helvetica', 12, 'bold'))
            lbl._is_splash = True
            lbl.pack(padx=10, pady=10)

        # Centrage
        root.update_idletasks()
        w = max(root.winfo_reqwidth(), 360)
        h = max(root.winfo_reqheight(), 120)
        x = root.winfo_screenwidth() // 2 - w // 2
        y = root.winfo_screenheight() // 2 - h // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.update()

        def _destroy():
            try:
                root.destroy()
            except Exception:
                pass
        return root, _destroy
    except Exception:
        return None, (lambda: None)


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    required_packages = {
        'mne': 'MNE-Python',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'yasa': 'YASA',
        'tkinter': 'Tkinter'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        if package == 'tkinter':
            try:
                import tkinter
            except ImportError:
                missing_packages.append(f"{name} (inclus avec Python)")
        else:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(name)
    
    if missing_packages:
        print("❌ Dépendances manquantes:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPour installer les dépendances:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Fonction principale de lancement."""
    print("CESA (Complex EEG Studio Analysis) 0.0alpha3.0")
    print("=" * 50)
    
    # Vérification des dépendances
    logging.info("Checking dependencies...")
    if not check_dependencies():
        logging.error("Cannot launch application - missing dependencies")
        return 1
    
    logging.info("All dependencies available")
    
    # Lancement de l'application
    try:
        logging.info("Launching CESA 0.0alpha3.0...")
        
        # Splash screen
        splash_root, close_splash = show_splash()
        
        def launch_main_app():
            """Construit l'application principale."""
            try:
                import tkinter as tk
                
                # S'assurer que le package CESA est dans le path
                parent_dir = os.path.dirname(os.path.abspath(__file__))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                # Import de l'application principale
                from CESA.eeg_studio_fixed import EEGAnalysisStudio
                
                root = splash_root
                
                # Retirer le mode splash
                try:
                    root.unbind_all("<Escape>")
                except Exception:
                    pass
                
                # Retirer les widgets du splash AVANT de construire l'interface
                try:
                    for child in list(root.winfo_children()):
                        if getattr(child, '_is_splash', False):
                            child.destroy()
                except Exception:
                    pass
                
                # Désactiver le mode splash window
                try:
                    root.overrideredirect(False)
                    root.attributes('-topmost', False)
                    root.update_idletasks()
                except Exception:
                    pass
                
                # Construire l'application
                logging.info("Building main UI...")
                _app = EEGAnalysisStudio(root)
                
                # Forcer la désactivation de topmost et donner le focus
                try:
                    root.attributes('-topmost', False)
                    root.lift()
                    root.focus_force()
                    root.update()
                except Exception:
                    pass
                
                logging.info("Application ready")
                
            except Exception as e:
                logging.error(f"Error launching main application: {e}")
                import traceback
                traceback.print_exc()
        
        # Programmer le lancement
        if splash_root:
            splash_root.after(10, launch_main_app)
            splash_root.mainloop()
        else:
            launch_main_app()
        
    except KeyboardInterrupt:
        logging.info("Application closed by user")
        return 0
    except Exception as e:
        logging.error(f"Error during launch: {e}")
        print("\n💡 Suggestions:")
        print("   - Vérifiez que tous les fichiers sont présents")
        print("   - Consultez le README.md pour plus d'informations")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

