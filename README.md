# CESA – Complex EEG Studio Analysis (0.0alpha3.0)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)]()
[![Release](https://img.shields.io/badge/Release-0.0alpha3.0-purple.svg)]()

## Apercu

**CESA** est un studio complet pour la visualisation et l'analyse de signaux EEG/PSG (format EDF+). Cette pré-version `0.0alpha3.0` ouvre le dépôt public et pose les fondations documentaires et techniques pour les prochaines itérations.

Cette version se concentre sur :
- l'interface graphique Tkinter optimisée pour la lecture long-terme,
- la pyramide multirésolution (accès quasi instantané aux données EDF+),
- les workflows de scoring manuel/automatique et d'analyse spectrale,
- la préparation du pipeline de connectivité et d'entropie renormée.

> ⚠️ **Statut Alpha** : les fonctionnalités scientifiques majeures sont présentes mais la documentation est en cours de consolidation. Les retours sont bienvenus via GitHub.

## Fonctionnalites principales

- Visualisation multi-canaux avec synchronisation spectrogrammes / hypnogramme.
- Filtres Butterworth prédéfinis (delta, alpha, beta) et paramétrages libres.
- Scoring manuel (Excel/CSV) et import d'annotations EDF+ automatiques.
- Pré-calcul multirésolution Zarr pour navigation < 50 ms.
- Analyses spectrales stade-par-stade (FFT robuste) et tableaux synthétiques.
- Préparation des modules avancés (entropie renormée, connectivité, micro-états).

## Prerequis

- Windows 10/11 (support principal) ; macOS/Linux en mode manuel.
- Python 3.9 ou supérieur avec `pip`.
- Bibliothèques listées dans `requirements.txt` (MNE, NumPy, SciPy, Matplotlib, Pandas, YASA, Zarr...).
- Carte graphique standard (Matplotlib + Tkinter) et 8 Go de RAM recommandés.

## Installation rapide

### Option 1 – Installation automatique (Windows)
1. Cloner le dépôt ou télécharger l'archive GitHub.
2. Double-cliquer sur `INSTALL.bat` (installe Python si absent et toutes les dépendances).
3. Lancer `RUN.bat` pour démarrer l'interface.

### Option 2 – Installateur graphique
1. Lancer `python CESA/install_cesa.py`.
2. Suivre les étapes GUI (détection Python, installation dépendances, vérifications).
3. Terminer en lançant `RUN.bat` ou `python run.py`.

### Option 3 – Installation manuelle

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

## Lancement et scripts utiles

- `RUN.bat` : lance l'application avec l'environnement par défaut.
- `python run.py` : lanceur principal (Tkinter GUI).
- `python cli.py view --ms-path <dossier_pyramide>` : ouvre directement une pyramide multirésolution existante.
- `python cli.py build-pyramid --raw <edf> --out <cible>` : construit la pyramide min/max Zarr.
- `monitor_pyramid.ps1` : surveillance temps réel de la construction (Windows PowerShell).

## Documentation

La documentation fonctionnelle et technique est regroupée dans `docs/` :
- `docs/Quickstart.md` : démarrage rapide et parcours utilisateur.
- `docs/Guide_Navigation.md` : navigation avancée, raccourcis et bonnes pratiques.
- `docs/Installation.md` : détails sur les méthodes d'installation et scripts.
- `docs/Troubleshooting.md` : résolution de problèmes connus.
- `docs/Methodes_Statistiques.md` : annexes sur les indicateurs clés.
- `docs/Import_Excel.md` : préparation des fichiers de scoring.

Les notes historiques et analyses internes sont archivées dans `archives/` (voir section suivante) afin d'alléger la racine du dépôt.

## Structure du depot

```
CESA/
├── CESA/                    # Module principal (GUI, entropie, dialogs...)
├── core/                    # Infrastructure multirésolution et fournisseurs
├── docs/                    # Documentation utilisateur et technique
├── scripts/                 # Outils additionnels (monitoring, vérifications)
├── archives/                # Sauvegardes des anciens fichiers documentation
├── requirements.txt         # Dépendances Python pour l'application
├── run.py                   # Entrée principale Tkinter
├── cli.py                   # Interface ligne de commande (build/view)
├── INSTALL.bat              # Installateur Windows automatique
├── RUN.bat                  # Lanceur Windows
└── LICENSE                  # Licence MIT (Côme Barmoy, 2025)
```

## Developpement

1. Créer un environnement virtuel et installer les dépendances :
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
2. Lancer `python run.py` et charger un fichier EDF+ de test (`CESA/samples/`).
3. Utiliser `python cli.py build-pyramid` pour générer une pyramide lors des tests de performance.

Tests automatisés : non inclus dans cette pré-version. Des scripts de vérification seront ajoutés avant la version bêta.

## Feuille de route Alpha

- Consolidation des modules d'analyses avancées (connectivité, entropie, micro-états).
- Ajouts de tests automatisés et de benchmarks reproductibles.
- Internationalisation (FR/EN) de l'interface et de la documentation.
- Intégration continue (GitHub Actions) pour les builds Windows.

## Support

- Email : come1.barmoy@supbiotech.fr
- GitHub : [@cbarmoy](https://github.com/cbarmoy)
- Unité : Neuropsychologie du Stress – IRBA

Signaler un bug ou proposer une amélioration sur la page Issues du dépôt.

## Licence

Projet distribué sous licence [MIT](LICENSE). © 2025 Côme Barmoy.

---

*CESA – Première publication alpha pour la communauté EEG de l'IRBA*