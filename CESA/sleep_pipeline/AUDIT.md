# Audit du scoring sommeil CESA - Analyse critique

## 1. Architecture actuelle

### Modules impliques
| Module | Role | Lignes |
|--------|------|--------|
| `CESA/sleep_scorer.py` | Facade auto-scoring (YASA/U-Sleep/PFTSleep) | ~655 |
| `CESA/manual_scoring_service.py` | Import/normalisation scoring manuel (Excel/EDF) | ~227 |
| `CESA/scoring_io.py` | I/O bas niveau (Excel timestamps, EDF annotations) | ~156 |
| `CESA/eeg_studio_fixed.py` | UI monolithique (~18500 lignes), contient comparaison, FFT/stade, indicateurs | ~18500 |

### Flux de donnees actuel
```
Import Excel/EDF --> ManualScoringService --> DataFrame(time, stage)
                                                    |
Raw EDF --> SleepScorer(YASA|USleep|PFTSleep) --> DataFrame(time, stage)
                                                    |
                                        eeg_studio_fixed._compare_scoring()
                                                    |
                                        Accuracy / Kappa / Matrice confusion
```

## 2. Regles et heuristiques identifiees

### 2.1 Normalisation des labels (ManualScoringService.STAGE_MAP)
- Mapping exhaustif : W/Wake/Eveil -> W, N1-N4 -> N1-N3, REM/Paradoxal -> R
- N4 fusionne avec N3 (conforme AASM 2007+, l'ancien R&K separait N3/N4)
- Artefact/Movement/MT -> U (correct: pas un stade de sommeil)
- **Ecart AASM** : aucune distinction artefact vs mouvement vs non-score dans les metadonnees

### 2.2 Epoching
- Default 30s (conforme AASM)
- `infer_epoch_seconds()` : infere depuis les diffs medianes des timestamps (range 5-120s)
- **Ecart AASM** : accepte des durees d'epoch non-standard (20s, 60s) sans warning

### 2.3 Auto-scoring (SleepScorer)
- Delegation complete a YASA / U-Sleep / PFTSleep
- Aucune regle AASM explicite dans le code CESA
- YASA applique son propre modele interne (LightGBM) avec features EEG/EOG/EMG
- **Ecart AASM** : pas de transparence sur les criteres de decision (boite noire)

### 2.4 Comparaison auto vs manuel (eeg_studio_fixed._compare_scoring)
- Alignement temporel par epoch (floor(time/30))
- Merge inner sur les epochs communes
- Options : exclure U manuels, exclure epochs faible confiance
- Metriques : accuracy, Cohen's kappa, F1 macro, precision/recall/specificite par stade
- **Point fort** : implementation correcte de la matrice de confusion
- **Ecart** : pas de gestion du biais lie aux epochs de transition (3s rule AASM)

### 2.5 Indicateurs de sommeil
- TIB, TST, SE, SOL, WASO, latence REM : formules conformes AASM
- Option inclure/exclure U dans les calculs
- **Ecart** : SOL calcule comme index * epoch_duration (approximation si les epochs U du debut ne sont pas exclus)
- **Ecart** : pas de detection du "lights off" / "lights on" (essentiel en clinique)

### 2.6 Detection d'evenements
- **ABSENT** : aucune detection d'arousal, apnee, hypopnee, desaturation
- Les canaux respiratoires (abdomen, thorax, SpO2) sont references dans PFTSleep mais uniquement pour l'inference DL, pas pour du scoring d'evenements

## 3. Biais et simplifications

| Probleme | Impact | Priorite |
|----------|--------|----------|
| Scoring = boite noire (YASA/USleep/PFT) | Pas de tracabilite des decisions | Haute |
| Pas de features explicites calculees | Impossible de debugger/auditer un stade | Haute |
| Comparaison uniquement epoch-a-epoch | Ignore le contexte sequentiel | Moyenne |
| Normalisation labels dupliquee (3+ endroits) | Risque d'incoherence | Moyenne |
| UI monolithique (18K lignes) | Maintenance/testabilite tres faible | Haute |
| Pas de detection d'evenements respiratoires | Non conforme PSG complete | Haute |
| Pas de lissage post-scoring | Transitions impossibles (W->N3 direct) | Moyenne |
| Pas de tests unitaires | Regressions non detectees | Haute |

## 4. Conformite AASM - Matrice

| Critere AASM | Statut actuel | Action requise |
|-------------|---------------|----------------|
| Epochs 30 secondes | OK (default) | Valider/forcer dans le pipeline |
| 5 stades (W, N1, N2, N3, REM) | OK | - |
| Criteres EEG alpha/theta/delta | Delegue a YASA | Implementer en regles explicites |
| Fuseaux de sommeil (N2) | Delegue a YASA | Detecteur sigma explicite |
| Complexes K (N2) | Delegue a YASA | Detecteur optionnel |
| Ondes lentes >75uV (N3) | Delegue a YASA | Seuil delta explicite |
| Mouvements oculaires rapides (REM) | Delegue a YASA | Feature EOG explicite |
| Atonie musculaire (REM) | Delegue a YASA | Feature EMG explicite |
| Arousal scoring | ABSENT | Implementer detecteur |
| Apnee obstructive/centrale/mixte | ABSENT | Implementer si canaux dispo |
| Hypopnee (>=3% desat OU arousal) | ABSENT | Implementer si canaux dispo |
| AHI (Apnea-Hypopnea Index) | ABSENT | Calculer depuis evenements |

## 5. Recommandations

1. **Pipeline modulaire** : extraire toute la logique de scoring dans `CESA/sleep_pipeline/`
2. **Regles AASM explicites** : implementer des criteres EEG/EOG/EMG interpretables
3. **Features calculees** : puissance par bande, ratios, variance, entropie
4. **Lissage sequentiel** : HMM ou regles de transition pour eviter les sauts impossibles
5. **Detection evenements** : arousals + respiratoire quand les canaux sont disponibles
6. **Tests** : couvrir normalisation, epoching, metriques, regles
7. **Separation UI/logique** : l'UI ne fait qu'orchestrer, pas de calculs inline
