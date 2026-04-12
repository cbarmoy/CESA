# Méthodes statistiques et indicateurs de sommeil

Ce document regroupe les algorithmes robustes utilisés par CESA pour les analyses « spaghetti » (différences AVANT/APRÈS) et les indicateurs de sommeil calculés dans l'interface.

## 1. Statistique de base

Toutes les méthodes reposent sur la médiane des différences :

```
X_avant → données de référence
X_après → données post-intervention

D_obs = median(X_après) - median(X_avant)

# Si données appariées :
d_i = x_après,i - x_avant,i
D_obs = median(d_i)
```

Implémentation : `core/bridge.py::_median_diff`.

## 2. Tests robustes disponibles

### 2.1 Test de permutation sur la médiane

- Cas apparié : permutation aléatoire des signes ±1 appliqués aux `d_i`.
- Cas non apparié : permutations entre les ensembles AVANT/APRÈS.
- Résultat : p-valeur ; décision (`augmentation`, `diminution`, `stagnation`) selon `alpha = 0.05`.

### 2.2 Intervalle de confiance bootstrap

- Resampling avec remise (`n_boot = 2000`).
- Décision selon la position du zéro dans l'IC 95 % (`CI_low`, `CI_high`).

### 2.3 Score Z robuste (MAD)

- Calcul : `Z = D_obs / (1.4826 × MAD)` où `MAD = median(|d_i - median(d_i)|)`.
- Seuil : `|Z| > 2.5` → effet significatif.

### 2.4 Consensus

- Chaque méthode produit ↑, ↓ ou =.
- Un consensus (≥ 2 méthodes concordantes) place un symbole ★ sur le point APRÈS.
- Cas limites (données insuffisantes, une seule époque) → décision forcée à `stagnation`.

## 3. Paramètres par défaut

```python
RNG_SEED = 42
N_PERM = 5000
N_BOOT = 2000
ALPHA = 0.05
ROBUST_Z_THRESHOLD = 2.5
```

Les valeurs sont configurables via la CLI et stockées dans `core/providers.py`.

## 4. Indicateurs de sommeil

Accessible via `Sommeil → Informations scoring`. Les calculs respectent les standards AASM et se basent sur les stades : W, N1, N2, N3, REM (et U pour les périodes non scorées).

| Indicateur              | Description                                                   | Formule                                    |
|-------------------------|---------------------------------------------------------------|--------------------------------------------|
| TIB (Time In Bed)       | Durée totale de l'enregistrement                              | `nb_epochs_total × durée_epoch`            |
| TST (Total Sleep Time)  | Temps total de sommeil (N1+N2+N3+REM)                         | `nb_epochs_sommeil × durée_epoch`          |
| Efficacité du sommeil   | Pourcentage de TIB passé à dormir                             | `(TST / TIB) × 100`                        |
| SOL (Sleep Onset Latency) | Temps avant la première époque de sommeil                   | `index_first_sleep × durée_epoch`          |
| Latence REM             | Temps avant la première époque REM                           | `index_first_REM × durée_epoch`            |
| WASO                    | Temps d'éveil après l'endormissement                          | `nb_epochs_W_post_onset × durée_epoch`     |
| Temps d'éveil total     | Somme des périodes d'éveil                                    | `nb_epochs_W × durée_epoch`                |

### Option « Inclure les périodes U »

- **Désactivée** (par défaut) : les U sont exclues des statistiques.
- **Activée** : les U sont traitées comme des non-sommeil → impact sur efficacité, latences, WASO.

### Visualisations

- **Camembert interactif** : répartition des stades (avec ou sans U).
- **Tableau détaillé** : nombre d'époques, pourcentage et durée par stade.
- **Rapport** : export CSV + copie texte formatée pour insertion dans des comptes rendus.

### Compatibilité

- Fonctionne avec les scorings manuels (Excel/CSV), EDF+ et le scoring automatique (YASA).
- Reconnaissance robuste des libellés (`Wake`, `Éveil`, `REM`, `R`, `U`, etc.).

## 5. Références et standards

- American Academy of Sleep Medicine (AASM) – Manual for the Scoring of Sleep and Associated Events.
- DSM-5 – Critères d'insomnie (efficacité du sommeil).
- Statistiques robustes : Hampel et al., 1986 (median, MAD) ; Good, 2005 (tests de permutation).

## 6. Entropie multiscale (MSE) – Nouveau

L'analyse MSE est accessible depuis `Analyse → 🔬 Entropie Multiscale (MSE)` ou via l'API `compute_multiscale_entropy*` dans `CESA/entropy.py`. Elle s'appuie sur la procédure de Costa et al. (2002) :

1. **Coarse-graining** : pour chaque échelle `τ`, le signal est moyenné par blocs (`y_j^(τ) = (1/τ) Σ_{i=(j-1)τ+1}^{jτ} x_i`).
2. **Sample Entropy (SampEn)** : on extrait tous les motifs de longueur `m` et `m+1`, on compte les paires dont la distance de Chebyshev est inférieure à `r × σ`, puis `SampEn = -ln(A/B)` avec `A` = matches(m+1) et `B` = matches(m).
3. **Profil multi-échelle** : on trace `SampEn(τ)` pour visualiser la complexité du signal à différentes résolutions temporelles.

### 6.1 Paramètres disponibles

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `m` | Dimension d'embedding (taille des motifs) | 2 |
| `r` | Tolérance relative (multipliée par l'écart-type du canal) | 0.2 |
| `scales` | Liste ou plage d'échelles `τ` | `1-20` |
| `max_samples` | Nombre maximum d'échantillons analysés (troncature sécurisée) | 200 000 |
| `max_pattern_length` | Nombre de points conservés par échelle pour SampEn | 5 000 |
| `return_intermediate` | Conserver les signaux coarse-grained | False |

### 6.2 Interprétation rapide

- **Profil décroissant** : le signal devient plus régulier à mesure que l'échelle augmente.
- **Profil stable ou croissant** : présence d'une variabilité persistante (complexité élevée).
- **SampEn → ∞** : absence de motifs répétés au seuil `r` (bruit ou instabilité marquée).

Des checkpoints détaillés sont inscrits dans la console (validation des entrées, coarse-graining, comptage des motifs) et repris dans la fenêtre GUI pour favoriser l'analyse pas-à-pas.

> ⚠️ **Stabilité mémoire**  
> Le comptage complet des motifs serait O(N²) et exploserait au-delà de quelques milliers de points.  
> CESA tronque donc automatiquement les séries :  
> • troncature globale via `max_samples` (GUI: champ *Max samples*)  
> • sous-échantillonnage pour SampEn via `max_pattern_length`.  
> Saisissez `all` dans l'interface si vous souhaitez traiter toute la série (opération coûteuse !).

## 7. Analyse CSV post-export (MSE & entropie renormée)

Pour comparer des cohortes (ex. AVANT/APRÈS MBSR) sans passer par l’interface graphique, exportez les fichiers `profiles.csv` / `stats.csv`, puis utilisez le script `tools/csv_analysis/analyse_csv.py`.

### 7.1 Chargement & filtrage

- `--input` : chemin vers le CSV (profil ou tests).  
- Filtres rapides : `--condition`, `--stage`, `--channel`, `--tau-min`, `--tau-max`, `--query`.  
- Le script détecte automatiquement les colonnes `subject`, `condition`, `value`, `stage`, `tau`. Les entrées incomplètes sont ignorées.

### 7.2 Tests intégrés

- `--wilcoxon` : Wilcoxon apparié sur les différences sujet par sujet.  
- `--permutation` : permutation de la médiane (`--n-perm`).  
- `--bootstrap` : intervalle de confiance 95 % (`--n-boot`).  
- `--robust-z` : score Z basé sur la MAD.  
- `--bh` : correction Benjamini–Hochberg sur les p-valeurs obtenues.

Les résultats sont affichés à l’écran ET sauvegardés via `--export-stats`.

### 7.3 Visualisation & exports

- `--plot spaghetti` : profils τ par sujet (style identique à la FFT en lot).  
- `--plot mean` : moyennes + IC par condition.  
- Exports PNG/PDF/SVG (`--output-fig`).  
- `--export-profiles` / `--export-filtered` : tables filtrées prêtes pour R/Python.

### 7.4 Performance

Le script détecte `os.cpu_count()` et répartit les tests sur plusieurs workers (`--max-workers` pour forcer une limite). Les mêmes garde-fous mémoire que dans l’UI (troncature, `max_pattern_length`) restent disponibles pour les recalculs MSE.

### 7.5 Références complémentaires

- Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). **Multiscale entropy analysis of complex physiologic time series**. *Physical Review Letters*, 89(6).

---

*CESA – méthodes statistiques & indicateurs (pré-release 0.0beta1.0)*

