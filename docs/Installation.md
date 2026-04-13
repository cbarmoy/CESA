# Installation et lancement

Ce document regroupe les scénarios d'installation et les scripts de lancement de CESA `0.0beta1.1`.

## 1. Pré-requis

- Windows 10 ou 11 (support principal).
- Python 3.9+ installé ou droits administrateur pour l'installer.
- Droits d'écriture dans le répertoire du projet (création de l'environnement, logs, stores `_ms`).

## 2. Méthodes d'installation

### 2.1 Installateur automatique Windows

```
INSTALL.bat
```

- Détecte Python (PATH + emplacements standards).
- Installe/actualise `pip`, crée un environnement virtuel si nécessaire.
- Installe toutes les dépendances de `requirements.txt`.
- Lance un test de démarrage rapide.

**Utilisation recommandée** pour les utilisateurs non techniques.

### 2.2 Installateur graphique (Python GUI)

```
python CESA/install_cesa.py
```

- Interface Tkinter guidée (détection de Python, installation modules, vérifications post-installation).
- Possibilité de renseigner un proxy ou un miroir PyPI.
- Génère un rapport récapitulatif (`logs/install_report.txt`).

### 2.3 Installation manuelle

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- Adaptée aux environnements gérés (conda, Poetry, etc.).
- Veiller à installer les modules Excel (`xlrd`, `openpyxl`) et Zarr (`zarr`, `numcodecs`).

## 3. Scripts de lancement

| Script            | Plateforme | Description                              |
|------------------|------------|------------------------------------------|
| `RUN.bat`        | Windows    | Détection automatique de Python puis exécution de `run.py`.
| `run.py`         | Tous       | Lanceur principal (splash screen, vérification dépendances).
| `cli.py`         | Tous       | Interface CLI (construction pyramide, visualisation directe).
| `monitor_pyramid.ps1` | Windows | Surveilance temps réel du dossier `_ms` (fichiers, taille).

### 3.1 Exemple de lancement Windows

```
Double-cliquez sur RUN.bat
```

Sortie typique :

```
CESA – Complex EEG Studio Analysis
==================================
[OK] Python trouvé (3.11.6)
[OK] Dépendances chargées
[LANCEMENT] Ouverture de CESA…
```

### 3.2 Exemple de lancement CLI multiplateforme

```bash
python run.py
# ou
python3 run.py
```

## 4. Résolution de problèmes d'installation

### Python introuvable

1. Installer Python 3.9+ depuis [python.org](https://www.python.org/downloads/).
2. Cocher « Add Python to PATH » durant l'installation.
3. Vérifier dans un nouveau terminal :
   ```powershell
   python --version
   where python
   ```
4. Relancer `RUN.bat` ou `INSTALL.bat`.

### Modules manquants

```powershell
python -m pip install -r requirements.txt
```

Pour un module spécifique :

```powershell
python -m pip install mne scipy matplotlib pandas yasa xlrd openpyxl zarr numcodecs
```

### Lancement silencieux / logs vides

- Lancer en mode non bufferisé : `python -u run.py`.
- Consulter `logs/cesa.log` (créé par `run.py` en cas d'erreur).
- Activer le mode debug : `python run.py --debug` (option disponible dans la CLI).

## 5. Bonnes pratiques

- Conserver `RUN.bat`, `INSTALL.bat` et `run.py` à la racine du projet.
- Mettre à jour `requirements.txt` après chaque ajout de dépendance.
- Archiver l'environnement (`.venv/`) en dehors du dépôt pour éviter les commits involontaires.
- Documenter les paramètres spécifiques (proxy, variables d'environnement) dans `docs/Troubleshooting.md`.

---

*CESA – installation & lancement (pré-release 0.0beta1.1)*

