#!/usr/bin/env python3
"""
CESA v3.0 - Script de lancement
===============================

Script de lancement pour CESA (Complex EEG Studio Analysis) v3.0.
Lance l'application avec vérification des dépendances.

Ce script est un wrapper qui appelle le lanceur principal run.py.

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 3.0.0
Date: 2025-10-26
"""

import os
import subprocess
import sys

def main():
    """Lance l'application CESA."""
    # Trouver le répertoire racine (2 niveaux au-dessus)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Chemin vers le lanceur principal
    run_py = os.path.join(root_dir, "run.py")
    
    if not os.path.exists(run_py):
        print("❌ Erreur: run.py introuvable")
        print(f"   Recherché dans: {run_py}")
        return 1
    
    # Lancer l'application
    print("🚀 Lancement de CESA v3.0...")
    try:
        subprocess.call([sys.executable, run_py])
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
