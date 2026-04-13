# Import de scoring Excel

CESA prend en charge l'import d'époques de sommeil depuis Excel (`.xls`, `.xlsx`) et CSV. Ce guide décrit la structure attendue, les dépendances et les solutions aux problèmes courants.

## Formats pris en charge

- `.xls` (Excel 97-2003) — nécessite `xlrd >= 2.0.0`.
- `.xlsx` (Excel 2007+) — nécessite `openpyxl >= 3.0.0`.
- `.csv` — via `pandas`.

## Structure minimale du fichier

| Colonne         | Description                               | Exemples de noms acceptés                |
|-----------------|-------------------------------------------|------------------------------------------|
| Stade de sommeil| Valeur de stade (W, N1, N2, N3, REM, U…)  | `Stage`, `Stade`, `Sleep`, `Sommeil`     |
| Horodatage      | Heure de début de l'époque (horodatage)   | `Time`, `Heure`, `Epoch`, `Début`, `Start`|

- Durée d'époque par défaut : 30 secondes (configurable dans l'UI).
- Les entêtes sont détectées sans tenir compte de la casse ou des accents.

## Exemple

```
| Sommeil | Heure de début       |
|---------|---------------------|
| Éveil   | 2024-02-01 23:58:30 |
| N1      | 2024-02-01 23:59:00 |
| N2      | 2024-02-01 23:59:30 |
| REM     | 2024-02-02 00:45:00 |
```

Un fichier d'exemple est fourni dans `CESA/samples/S043_AP_sleep scoring.xls`.

## Installation des dépendances

```powershell
python -m pip install -r requirements.txt
# ou
python -m pip install pandas xlrd openpyxl
```

Vérification rapide :

```powershell
python -c "import pandas, xlrd, openpyxl; print('ok')"
```

## Procédure dans CESA

1. Lancer l'application (`RUN.bat` ou `python run.py`).
2. Charger un fichier EDF+ (`Fichier` → `Ouvrir EDF+`).
3. Menu `Sommeil` → `Importer scoring Excel`.
4. Sélectionner le fichier (`.xls`, `.xlsx`, `.csv`).
5. Vérifier/ajuster le décalage temporel si nécessaire (ex. début d'enregistrement vs début de scoring).

Le scoring est immédiatement affiché dans :

- l'hypnogramme (bandes colorées),
- le module d'indicateurs (`Sommeil` → `Informations scoring`),
- le graphique camembert.

## Dépannage

| Problème                        | Solution rapide                                                        |
|---------------------------------|-------------------------------------------------------------------------|
| `Module xlrd manquant`          | `python -m pip install xlrd>=2.0.0`                                     |
| `Module openpyxl manquant`      | `python -m pip install openpyxl>=3.0.0`                                 |
| Erreur de lecture               | Vérifier le format (.xls/.xlsx), les entêtes et l'encodage UTF-8        |
| Colonnes non reconnues          | Renommer selon les exemples (`Stage`, `Sommeil`, `Time`, `Heure`, ...)   |
| Synchronisation incorrecte      | Ajuster le décalage horaire dans la boîte de dialogue d'import          |

## Visualisation et export

- Menu `Sommeil` → `Informations scoring` : statistiques TST, WASO, SOL, etc. (voir `docs/Methodes_Statistiques.md`).
- Export CSV : `Sommeil` → `Exporter indicateurs`.
- Copie dans le presse-papiers : `Sommeil` → `Copier rapport` (format texte prêt à coller dans Word/LibreOffice).

## Bonnes pratiques

- Préparer le fichier Excel avec des en-têtes explicites (`Sommeil`, `Heure de début`).
- Utiliser des horodatages ISO (`YYYY-MM-DD HH:MM:SS`) pour éviter les ambiguïtés de fuseau.
- Conserver une copie du scoring importé dans `archives/` pour traçabilité.
- Tester l'import avec un sous-ensemble avant de lancer une étude complète.

---

*CESA – import Excel (pré-release 0.0beta1.1)*

