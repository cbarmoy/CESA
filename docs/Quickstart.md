# ⚡ Démarrage rapide

Ce guide résume les étapes clés pour charger un enregistrement EDF+ et activer la navigation multirésolution dans CESA `0.0beta1.0`.

## 1. Première ouverture d'un enregistrement

1. Lancez CESA (`RUN.bat` ou `python run.py`).
2. Menu `Fichier` → `Ouvrir EDF+` et choisissez votre fichier `.edf`.
3. Dans la boîte de dialogue **Mode de données**, laissez l'option **⚡ Navigation rapide (recommandé)** sélectionnée.
4. Laissez le champ « Fichier existant (optionnel) » vide puis cliquez sur **Continuer**.
5. Confirmez la création du store multirésolution lorsque la fenêtre de confirmation apparaît. La génération dure de 5 à 20 minutes selon la taille du fichier (voir tableau ci-dessous).

| Durée EEG | Canaux | Temps de création estimé |
|-----------|--------|--------------------------|
| 2 h       | 16     | 2 à 3 min                |
| 8 h       | 16     | 5 à 10 min               |
| 24 h      | 32     | 20 à 30 min              |

Le dossier optimisé (`*_ms/`) est créé à côté de votre fichier EDF d'origine et occupe environ 10 à 20 % de son volume.

## 2. Utilisations suivantes

Pour recharger le même enregistrement :

1. Ouvrez votre fichier EDF via `Fichier` → `Ouvrir EDF+`.
2. Conservez l'option **Navigation rapide**.
3. Laissez le champ du chemin vide : CESA détecte automatiquement le dossier `_ms` existant.
4. Cliquez sur **Continuer** → chargement instantané (< 50 ms).

## 3. Mode standard (si vous êtes pressé)

Si vous ne pouvez pas attendre la création du store :

1. Dans la boîte de dialogue, sélectionnez **📂 Mode standard**.
2. Cliquez sur **Continuer** pour charger immédiatement l'EDF (navigation moins fluide).
3. Vous pourrez créer la navigation rapide plus tard via le menu `Fichier` → `Créer navigation rapide` ou via la CLI (`python cli.py build-pyramid`).

## 4. Questions fréquentes

- **Dois-je créer le store à chaque fois ?** Non. Une seule création suffit ; CESA le réutilise automatiquement.
- **Puis-je supprimer le dossier `_ms` ?** Oui. Le mode standard reste disponible. Vous pourrez recréer le store à tout moment.
- **Où est stockée la pyramide ?** Dans un dossier `<nom_fichier>_ms/levels/` situé à côté du fichier EDF.
- **La création s'est arrêtée ?** Relancez la commande ; le processus reprend là où il s'était arrêté.

## 5. Suivi CLI (optionnel)

Pour les fichiers volumineux, vous pouvez lancer la création en ligne de commande :

```powershell
python cli.py build-pyramid --raw DATA/patient.edf --out DATA/patient_ms
```

Ajoutez `--fresh` pour forcer une reconstruction complète. Le script `monitor_pyramid.ps1` permet de suivre la progression (nombre de fichiers et taille disque).

---

*CESA – documentation interne, pré-release 0.0beta1.0*

