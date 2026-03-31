# Dépannage

Cette fiche recense les problèmes fréquemment rencontrés lors de l'utilisation de CESA `0.0alpha4.0` et propose des solutions rapides.

## Python introuvable (Windows)

**Message** : `Python non trouvé`

1. Installer Python 3.9+ depuis [python.org](https://www.python.org/downloads/).
2. Cocher « Add Python to PATH » pendant l'installation.
3. Ouvrir un nouveau terminal puis vérifier :
   ```powershell
   python --version
   where python
   ```
4. Relancer `RUN.bat`. En cas d'environnement spécifique, renseigner le chemin complet dans le script :
   ```powershell
   C:\Users\Nom\AppData\Local\Programs\Python\Python311\python.exe run.py
   ```

## Modules manquants

**Symptôme** : erreurs `ModuleNotFoundError: mne` (ou `numpy`, `yasa`, etc.).

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour un module isolé :

```powershell
python -m pip install mne numpy scipy matplotlib pandas yasa zarr numcodecs xlrd openpyxl
```

## Import Excel : échec de lecture

Consultez `docs/Import_Excel.md` pour la structure attendue et les dépendances. Rappels rapides :

- Vérifier `xlrd` (pour `.xls`) et `openpyxl` (pour `.xlsx`).
- S'assurer que les colonnes contiennent au moins les stades et les horodatages.
- Tester avec l'exemple `CESA/samples/S043_AP_sleep scoring.xls`.

## Construction de pyramide très lente

**Contexte** : fichiers très denses (ex. 90 canaux × 8 h) → temps > 15 min.

### Pourquoi ?

- Nombre de canaux multiplié par 5 à 6 par rapport à un PSG standard (16 canaux).
- Volume du store `_ms` proportionnel au nombre de canaux (prévoir 20 % du volume EDF).

### Que faire ?

1. Laisser le processus se terminer (temps estimé ≈ temps standard × canaux/16).
2. Surveiller la progression avec `monitor_pyramid.ps1` (fichiers générés, taille disque).
3. Éviter d'interrompre le processus (risque de store incomplet).
4. Planifier la construction en dehors des heures d'analyse.

### Astuces techniques

- Lancer en mode non bufferisé pour afficher les checkpoints :
  ```powershell
  python -u cli.py build-pyramid --raw DATA/fichier.edf --out DATA/fichier_ms --progress
  ```
- Ajouter `flush=True` aux `print` de `core/multiscale.py` pour forcer l'affichage immédiat.
- Configurer un log dédié :
  ```python
  log_file = ms_path.parent / "pyramid_build.log"
  with open(log_file, "a", encoding="utf-8") as log:
      log.write(f"[{time.strftime('%H:%M:%S')}] CHECKPOINT {current_percent}%\n")
      log.flush()
  ```

## Store `_ms` absent ou corrompu

1. Supprimer le dossier `_ms` existant.
2. Relancer la commande de construction avec `--fresh` :
   ```powershell
   python cli.py build-pyramid --raw DATA/fichier.edf --out DATA/fichier_ms --fresh
   ```
3. Vérifier l'espace disque disponible (> 20 % du volume EDF).

## Latences élevées pendant la navigation

- Ouvrir `logs/telemetry.csv` pour mesurer les latences récentes.
- Vérifier que la navigation rapide est activée (barre de statut ⚡).
- Fermer les vues inutilisées (spectrogrammes multiples augmentent la charge).
- Mettre à jour les pilotes graphiques si l'affichage est lent.

## Support

- Email : come1.barmoy@supbiotech.fr
- GitHub : [Issues](https://github.com/cbarmoy/CESA/issues)
- Documentation complémentaire : `docs/Quickstart.md`, `docs/Guide_Navigation.md`, `docs/Import_Excel.md`

---

*CESA – dépannage (pré-release 0.0alpha4.0)*

