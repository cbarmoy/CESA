#!/usr/bin/env python3
"""
CESA 0.0beta1.1 - Release Preparation Script
=============================================

Script de préparation de la release CESA (EEG Studio Analysis) 0.0beta1.1.
Automatise les tâches de préparation pour la distribution.

Fonctionnalités:
- Vérification des dépendances
- Test d'importation des modules
- Validation de la syntaxe Python
- Génération du changelog
- Préparation de l'archive de release

Auteur: Côme Barmoy (IRBA)
Version: 0.0beta1.1
Date: 2025-11-01
"""

import os
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dir(path: str) -> None:
    """Crée un répertoire s'il n'existe pas."""
    os.makedirs(path, exist_ok=True)


def check_python_files():
    """Vérifie la syntaxe de tous les fichiers Python principaux."""
    print("\n🔍 Vérification de la syntaxe Python...")

    important_files = [
        os.path.join(ROOT, 'run.py'),
        os.path.join(ROOT, 'CESA', 'eeg_studio_fixed.py'),
        os.path.join(ROOT, 'CESA', 'spectral_analysis.py'),
        os.path.join(ROOT, 'CESA', 'filters.py'),
        os.path.join(ROOT, 'CESA', 'scoring_io.py'),
        os.path.join(ROOT, 'CESA', 'theme_manager.py'),
    ]

    success = True
    for py_file in important_files:
        if not os.path.exists(py_file):
            print(f"   ⚠️ {os.path.basename(py_file)} - Fichier introuvable")
            continue
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file, 'exec')
            print(f"   ✅ {os.path.basename(py_file)}")
        except SyntaxError as e:
            print(f"   ❌ {os.path.basename(py_file)} - Erreur de syntaxe: {e}")
            success = False

    return success


def check_imports():
    """Teste l'importation des modules principaux."""
    print("\n🔍 Test d'importation des modules...")

    # Ajouter le répertoire racine au PYTHONPATH temporairement
    sys.path.insert(0, ROOT)

    modules_to_test = [
        ('CESA.eeg_studio_fixed', 'EEG Studio Fixed'),
        ('CESA.spectral_analysis', 'Spectral Analysis'),
        ('CESA.filters', 'Filters'),
        ('CESA.scoring_io', 'Scoring I/O'),
        ('CESA.theme_manager', 'Theme Manager')
    ]

    success = True
    for module, description in modules_to_test:
        try:
            __import__(module)
            print(f"   ✅ {description}")
        except ImportError as e:
            print(f"   ❌ {description} - Échec d'import: {e}")
            success = False
        except Exception as e:
            print(f"   ⚠️ {description} - Erreur: {e}")

    # Restaurer le PYTHONPATH
    sys.path.pop(0)
    return success


def generate_changelog():
    """Génère un changelog pour la release."""
    print("\n📝 Génération du changelog...")

    try:
        # Essayer de récupérer les derniers commits Git
        git_info = ""
        try:
            result = subprocess.run(
                ['git', 'log', '--oneline', '-10'],
                capture_output=True, text=True, check=True, timeout=10,
                cwd=ROOT
            )
            git_info = f"\n### 📊 Derniers commits\n```\n{result.stdout}\n```"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            git_info = "\n### 📊 Historique\n- Publication 0.0beta1.1 - Première pré-release publique"

        changelog = f"""# CESA 0.0beta1.1 - Changelog
Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Version 0.0beta1.1 - Première pré-release

### 🎉 Contenu principal
- Consolidation de la documentation (`docs/`)
- Déplacement de `requirements.txt` à la racine du projet
- Ajout de la licence MIT et du fichier `.gitattributes`
- Initialisation Git (`main` + tag `v0.0beta1.1`)
- Archivage des notes historiques (`archives/md-legacy-*.zip`)

### ⚙️ Infrastructure existante
- Pyramide multirésolution Zarr (`cli.py build-pyramid`)
- Interface Tkinter (`run.py`, `CESA/eeg_studio_fixed.py`)
- Scripts Windows (`RUN.bat`, `INSTALL.bat`, `CESA/install_cesa.py`)
- Modules analytiques (scoring, entropy, spectral)

### 📋 Prérequis système
- Python 3.9+ (64 bits recommandé)
- Packages listés dans `requirements.txt`
- 8 Go de RAM conseillés pour l'analyse multicanale

### 🚀 Installation

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancement de l'application
python run.py
# ou sous Windows
RUN.bat
```

### 📖 Documentation
- `README.md` : guide d'utilisation
- `docs/` : Quickstart, installation, dépannage, méthodes statistiques
- `documentation/REFERENCES.txt` : références scientifiques
{git_info}

---
**CESA (Complex EEG Studio Analysis) 0.0beta1.1**  
Développé par Côme Barmoy - Unité Neuropsychologie du Stress (IRBA)
"""

        changelog_path = os.path.join(ROOT, 'CHANGELOG.md')
        with open(changelog_path, 'w', encoding='utf-8') as f:
            f.write(changelog)

        print(f"   ✅ Changelog généré: CHANGELOG.md")
        return True

    except Exception as e:
        print(f"   ⚠️ Erreur lors de la génération du changelog: {e}")
        return False


def prepare_release() -> bool:
    """Prépare la release complète avec toutes les validations."""
    print("🚀 CESA 0.0beta1.1 - Préparation de la release")
    print("=" * 60)

    # Tests de validation
    checks = [
        ("Syntaxe Python", check_python_files),
        ("Imports des modules", check_imports),
        ("Changelog", generate_changelog)
    ]

    all_success = True
    for check_name, check_func in checks:
        if not check_func():
            all_success = False
            print(f"   ⚠️ {check_name} - Échec")

    if all_success:
        print("\n✅ Release préparée avec succès!")
        print("\n📋 Instructions pour la distribution:")
        print("   1. Vérifier que tous les tests passent")
        print("   2. Tester l'application: python run.py")
        print("   3. Créer le tag Git: git tag -a v0.0beta1.1 -m 'Pre-release 0.0beta1.1'")
        print("   4. Publier sur le dépôt")
        return True
    else:
        print("\n❌ Échec de la préparation de la release")
        print("   Corrigez les erreurs avant de continuer")
        return False


def main() -> int:
    """Fonction principale - point d'entrée du script."""
    try:
        success = prepare_release()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Préparation interrompue par l'utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
