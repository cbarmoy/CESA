# Guide navigation rapide

La navigation rapide exploite une pyramide multirésolution (store Zarr) pour garantir une latence < 50 ms sur les enregistrements longs. Ce guide détaille son fonctionnement et les bonnes pratiques.

## Pourquoi utiliser la navigation rapide ?

- **Fluidité** : zooms et déplacements instantanés, même sur 24 h × 32 canaux.
- **Mémoire contrôlée** : seules les fenêtres visibles sont chargées en RAM.
- **Robustesse** : reprise automatique si la création du store est interrompue.

| Critère                 | Mode standard                 | Navigation rapide                   |
|-------------------------|--------------------------------|-------------------------------------|
| Préparation initiale    | Aucune                         | 5 à 30 min selon la taille          |
| Surcharge disque        | Fichier EDF                    | +10–20 % (dossier `_ms`)            |
| Vitesse de navigation   | Dépend de la taille du fichier | Constante (< 50 ms)                 |
| Cas d'usage privilégié  | Tests rapides, petits enregistrements | Analyses longues, revue experte |

## Cycle de vie du store `_ms`

```
patient_001.edf
patient_001_ms/
    levels/
        lvl1/
        lvl2/
        ...
    attrs.json
```

- `levels/lvl*` : niveaux de résolution, du plus fin au plus grossier.
- `attrs.json` : métadonnées (version, paramètres de génération).
- Recréation idempotente : CESA détecte si un store existe déjà et propose sa mise à jour.

## Création en interface graphique

1. Chargez l'EDF via `Fichier` → `Ouvrir EDF+`.
2. Sélectionnez **Navigation rapide (recommandé)**.
3. Laissez le champ du chemin vide pour créer un nouveau store.
4. Confirmez. Une fenêtre de progression s'affiche ; vous pouvez relancer plus tard si vous fermez CESA.

### Indicateurs de progression

- Fenêtre CLI (si lancée depuis un terminal) : affichage des checkpoints.
- Fichiers générés : un monitor système (`monitor_pyramid.ps1`) permet d'observer le nombre de fichiers/volume en temps réel.
- Logs : `logs/telemetry.csv` suit les latences, niveaux utilisés et FPS lors de la navigation.

## Création via la ligne de commande

```powershell
python cli.py build-pyramid --raw DATA/patient.edf --out DATA/patient_ms --chunk-seconds 20
```

Options utiles :

- `--levels "1,2,4,8,16,32"` : niveaux personnalisés.
- `--fresh` : force la régénération complète (supprime l'existant).
- `--progress` : active un affichage détaillé des checkpoints dans la console.

Le store généré peut être versionné ou partagé avec l'EDF pour éviter sa recréation.

## Résolution de problèmes

### Construction plus lente que prévu

- **Symptôme** : > 10 min pour un enregistrement de 8 h.
- **Facteur** : nombre de canaux élevé (ex. 90 canaux ≈ ×5,6 le volume standard).
- **Action** : laisser le processus aboutir ; suivre la progression avec `monitor_pyramid.ps1`.
- **Optimisation** : planifier la création en fin de journée, vérifier l'espace disque (prévoir 20 % supplémentaires).

### Checkpoints invisibles dans la console

- Lancer CESA en mode non bufferisé : `python -u run.py`.
- Ajouter `flush=True` aux `print` dans `core/multiscale.py` lors de la génération.
- Écrire un log dédié : voir exemple dans `docs/Troubleshooting.md`.

### Store corrompu ou incomplet

1. Supprimer le dossier `_ms` concerné.
2. Relancer `cli.py build-pyramid` avec `--fresh`.
3. Vérifier l'alimentation et l'espace disque (éviter les arrêts brutaux).

## Conseils pratiques

- Conserver un store par enregistrement et le partager avec les collaborateurs pour éviter une régénération.
- Vérifier régulièrement `logs/telemetry.csv` afin de suivre les performances (< 50 ms P95, ≥ 30 FPS recommandés).
- Prévoir une alerte si le nombre de canaux > 50 pour ajuster les temps de traitement.

---

*Documentation CESA – navigation rapide (pré-release 0.0beta1.1)*

