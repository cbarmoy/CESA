# Choix du modèle de scoring automatique

Ce guide accompagne la fenêtre **Sommeil → Scoring automatique…**. Il répond à trois questions : quel backend choisir, quels canaux sélectionner, et comment vérifier honnêtement la qualité du résultat.

## 1. Quel backend choisir ?

| Backend | Principe | Canaux | Prérequis | Accord typique | À choisir quand |
|---|---|---|---|---|---|
| **YASA** | Descripteurs par époque (spectre, entropies, contexte voisin) + gradient boosting pré-entraîné | 1 EEG requis, EOG et EMG recommandés | `pip install yasa`, pas de GPU | 80-88 %, kappa 0,70-0,80 | Choix par défaut : rapide, robuste, aucun poids à télécharger, confiance par époque |
| **PFTSleep** | Transformer sur la nuit entière, pré-entraîné en auto-supervisé sur de la PSG multicanal | EEG requis ; EOG, EMG, ECG recommandés ; SpO2 et sangles optionnelles | PyTorch + paquet `PFTSleep` + poids Hugging Face, et le fichier EDF d'origine sur le disque | souvent kappa > 0,80 si tous les canaux sont présents | Montage PSG complet et temps de calcul acceptable |
| **U-Sleep** | Réseau convolutif de segmentation dense (U-Net temporel) | 1 EEG requis, 1 EOG optionnel ; EMG non utilisé | `pip install usleep-api` + **token** [sleep.ai.ku.dk](https://sleep.ai.ku.dk) (12 h) | 80-88 % (modèle publié) | Second avis face à YASA, surtout sur un montage réduit (bandeau) |
| **Règles AASM** | Application directe des critères AASM (alpha/thêta/delta, fuseaux, complexes K, mouvements oculaires, tonus mentonnier) | 1 EEG requis, EOG et EMG fortement conseillés | aucune dépendance | 65-78 % | Aucun modèle installé, ou besoin d'un scoring explicable |
| **Règles AASM + HMM** | Règles AASM puis décodage de Viterbi | idem règles AASM | aucune dépendance | quelques points de mieux, hypnogramme moins fragmenté | À préférer systématiquement aux règles seules |
| **ML** | Descripteurs CESA + modèle entraîné par vous | idem règles AASM | un modèle entraîné (aucun n'est fourni) | peut dépasser YASA sur votre propre population | Études internes avec des nuits déjà scorées |
| **ML + HMM** | Modèle ML puis lissage HMM | idem ML | même modèle entraîné | meilleur que le ML seul | À préférer au ML seul |

Sans rien installer, seuls **Règles AASM** et **Règles AASM + HMM** fonctionnent d'emblée. YASA ne demande qu'un `pip install yasa`. U-Sleep passe par le service officiel : `pip install usleep-api`, un compte sur [sleep.ai.ku.dk](https://sleep.ai.ku.dk), et un token (valable 12 h) collé dans la fenêtre ou dans `USLEEP_API_TOKEN`. Les poids de l'article ne sont pas distribués ; un checkpoint local n'est utile que si vous en avez entraîné un. Les backends ML exigent un modèle entraîné. La fenêtre signale ces prérequis avant le lancement plutôt que d'échouer en cours de route.

U-Sleep **envoie un extrait anonymisé** (EEG + EOG choisis, pas les ~90 voies de la PSG) au Danemark. Ne lancez pas ce backend sur un enregistrement que vous n'avez pas le droit de transférer.

### Stratégie recommandée

1. Commencer par **YASA** : c'est le meilleur rapport qualité/effort et il fournit une confiance par époque, utile pour repérer les passages à revoir.
2. Ajouter **U-Sleep** comme second avis sur le même couple `F4-M1` + `E1-M2` : il n'a pas besoin d'EMG, ce qui correspond au bandeau. Les désaccords entre YASA et U-Sleep signalent les époques réellement ambiguës.
3. Ajouter **PFTSleep** seulement si le montage est complet (ECG, sangles).
4. Utiliser **Règles AASM + HMM** quand il faut justifier chaque décision plutôt que maximiser la performance.

## 2. Combien de canaux, et lesquels ?

### Les listes sont des listes de candidats, pas des entrées cumulées

Point le plus souvent mal compris : dans la fenêtre de configuration, les trois listes EEG / EOG / EMG définissent un **ordre de priorité**. Pour chaque modalité, le backend retient le **premier canal disponible** dans l'enregistrement, et ignore les autres. En sélectionner dix n'apporte aucune information supplémentaire : cela définit seulement des solutions de repli si le canal préféré est absent.

Conséquence pratique : mettez en tête de liste la dérivation que vous voulez réellement utiliser.

### Canaux recommandés par modalité

- **EEG (indispensable)** : une dérivation centrale référencée, `C4-M1` en premier choix, `C3-M2` en repli. Les dérivations frontales (`F4-M1`) conviennent aussi ; les occipitales (`O2-M1`) sont moins adaptées au sommeil profond. Un EEG non référencé (`C4` seul) fonctionne mais dégrade la détection des fuseaux et des complexes K.
- **EOG (fortement recommandé)** : `E1-M2` ou `E2-M1`. C'est ce canal qui permet de séparer le REM de l'éveil calme et de repérer les mouvements oculaires lents du N1. Sans EOG, attendez-vous à une chute nette de la sensibilité en REM et en N1.
- **EMG menton (recommandé)** : `Chin1-Chin2` ou `Chin1-Chin3`. Sur beaucoup de PSG françaises les trois électrodes du menton s'exportent `1`, `2` et `F`, avec les dérivations bipolaires `1-2`, `1-F`, `2-F`. CESA les reconnaît ; **utilisez `1-2` en priorité** (c'est la dérivation AASM). Les voies d'impédance (`1 Impédance`, `1-2 (Imp)`) ne sont pas de l'EMG : ne les sélectionnez pas. Sans EMG, les confusions REM/N1 et REM/éveil augmentent.

> **Un EMG de jambe n'est pas un EMG de scoring.** Les modèles attendent un EMG **mentonnier** : le REM se définit par l'atonie du menton, que le tibial antérieur ne mesure pas (les jambes bougent en REM et sont relâchées en éveil calme). Renseigner `Jambe droite`, `Left Leg` ou `Tib ant` dans la liste EMG fournit au modèle une entrée hors distribution et **dégrade** la détection du REM au lieu de l'améliorer. En l'absence de canal mentonnier, laissez la liste EMG vide : le modèle se rabat proprement sur l'EEG et l'EOG. Le canal de jambe reste utile pour la détection des mouvements périodiques des membres, ailleurs dans CESA.
>
> **Ne mettez pas non plus une voie d'une autre modalité dans la liste EMG.** Une référence mastoïdienne (`M1`, `M2`, `A1`), une électrode EEG (`C3`, `F4`), un EOG (`E1-M2`), l'ECG, un signal respiratoire ou la SpO2 ne contiennent aucun tonus musculaire. Fournis comme EMG, ils donnent au modèle une entrée dénuée de sens là où il attend l'atonie du menton — c'est pire que de ne rien fournir.
>
> CESA détecte ces deux cas : la fenêtre de configuration affiche un avertissement orange sous le canal retenu, propose un canal mentonnier s'il en existe un dans le montage, demande confirmation avant de lancer le calcul, et le rapport de comparaison le rappelle dans son interprétation.
- **ECG et canaux respiratoires** : utilisés uniquement par PFTSleep (ECG, SpO2, sangles thoracique et abdominale). Les canaux absents sont remplacés par un vecteur nul, ce qui dégrade la prédiction sans bloquer le calcul.

### Combien de canaux au total ?

| Montage disponible | Backend conseillé | Ce qu'il faut en attendre |
|---|---|---|
| 1 EEG seul | YASA, ou Règles AASM + HMM | Utilisable, mais N1 et REM peu fiables |
| 1 EEG + 1 EOG | YASA | Bon compromis, REM correctement identifié |
| 1 EEG + 1 EOG + 1 EMG | YASA ou Règles AASM + HMM | Configuration minimale conforme à l'esprit AASM |
| PSG complète (EEG, EOG, EMG, ECG, respiration) | PFTSleep, avec YASA en second avis | Meilleure performance attendue |

La fenêtre de configuration vérifie en direct les canaux de l'enregistrement chargé, affiche le canal effectivement retenu pour chaque modalité, et signale ceux qui manquent.

### Chaque algorithme a ses propres attentes

Les listes proposées **changent quand vous changez d'algorithme** : elles suivent ce que chaque modèle a réellement appris, pas un profil unique copié de YASA. La fenêtre affiche sous les listes la justification du profil, et une modalité qu'un backend n'exploite pas est grisée et étiquetée *non utilisée* au lieu d'être offerte inutilement.

| Backend | EEG | EOG | EMG | Autres |
|---|---|---|---|---|
| YASA | centrale d'abord (`C4-M1`), frontale et occipitale en repli | recommandé | mentonnier recommandé | âge et sexe exploités |
| U-Sleep | frontale d'abord (`F4-M1`), centrale en repli | recommandé | **non utilisé** | service officiel, token 12 h |
| PFTSleep | dérivations référencées uniquement (`C4-M1`, `C3-M2`, `FPZ-CZ`, `PZ-OZ`) | `E1-M2` | `Chin1-Chin2` | ECG recommandé, SpO2 et sangles optionnelles |
| Règles AASM, Règles + HMM | centrale (frontale si le sommeil lent est l'objet de l'étude) | recommandé | mentonnier recommandé | — |
| ML, ML + HMM | **celui de l'entraînement de votre modèle** | idem | idem | — |

Un canal absent n'est **pas** remplacé par un canal par défaut : une liste vide veut dire « pas de canal pour cette modalité ». C'est indispensable pour pouvoir mesurer l'effet réel de l'EOG ou de l'EMG — sans quoi la configuration « sans EOG » du balayage tournerait avec un EOG et donnerait le même résultat que la configuration qui en a un.

Symétriquement, les backends à règles ne traitent plus un canal absent comme une mesure. `emg_rms = 0` signifiait auparavant « aucun tonus musculaire » alors qu'il voulait dire « aucun EMG branché » : les deux règles de l'éveil exigeant un EMG élevé, **l'éveil devenait inatteignable** sur un montage sans EMG mentonnier, et le critère d'atonie du REM était trivialement satisfait. Sans EMG, l'éveil repose désormais sur l'EEG seul (rythme alpha avec peu d'activité lente), avec une confiance réduite puisqu'un des deux critères AASM n'a pas pu être vérifié.

### Mouvements oculaires : un critère relatif à la nuit

Le critère REM des règles comparait l'activité de l'EOG à un seuil **absolu**. Cette activité est la variance de la dérivée du signal : elle grandit avec le calibre de l'amplificateur et diminue quand la fréquence d'échantillonnage monte, si bien qu'aucune valeur fixe ne se transporte d'un enregistrement à l'autre.

L'activité est désormais exprimée **en multiples de la ligne de base de l'enregistrement**, prise comme la médiane sur toute la nuit (seuil : 2 fois). Comme les REM sont des événements brefs, cette variance sur 30 s bouge trop peu : un second critère, le **nombre de REM discrets** dans l'époque, est jugé de la même façon contre la médiane de la nuit. L'un ou l'autre suffit. Sans ligne de base — moins de 20 époques, EOG plat ou absent — les règles reviennent aux seuils absolus.

Sans EMG mentonnier, l'alpha ne peut plus trancher seul entre éveil et REM : les mouvements oculaires sont examinés **avant** la règle d'éveil. Un rythme alpha avec les yeux calmes reste de l'éveil ; le même alpha avec des REM devient du REM.

Ce recalibrage se vérifie sur vos données : relancez **Sommeil → Comparer des configurations de scoring** avec la comparaison inter-algorithmes et regardez la colonne *F1 R %* des lignes `aasm_rules` et `rules_hmm`.

Votre personnalisation est **mémorisée par algorithme** : les canaux choisis pour YASA ne sont pas réappliqués à un autre backend quand vous en changez, et les retrouver ne demande que de resélectionner l'algorithme. Une modalité qu'un backend ignore n'est pas transmise au calcul, ce qui évite un avertissement EMG sans objet sur U-Sleep.

## 3. Durée d'époque

30 s est le standard AASM. C'est la seule valeur cohérente pour YASA, U-Sleep et PFTSleep, qui ont été entraînés ainsi : le champ est donc verrouillé sur 30 s pour ces trois backends. Les backends à base de règles et ML acceptent une autre durée, mais toute comparaison avec un scoring humain à 30 s en devient plus approximative (les époques sont alors regroupées par vote majoritaire sur la grille la plus grossière).

## 4. Comparer le scoring automatique au scoring manuel

La comparaison (**Sommeil → Comparer auto vs manuel**) confronte deux références conservées séparément :

- `auto_scoring_data` : résultat du dernier scoring automatique ;
- `manual_scoring_data` : scoring humain, importé depuis un fichier ou saisi dans l'éditeur.

Ces deux stockages sont indépendants : l'import d'un scoring manuel n'écrase plus le résultat automatique, et l'ordre des opérations n'a plus d'importance.

### Procédure

1. Lancer le scoring automatique.
2. Fournir le scoring humain, au choix :
   - **Sommeil → Importer scoring** (Excel, CSV, JSON ou annotations EDF+) ;
   - **Sommeil → Éditeur de scoring manuel**, en laissant la cible sur « Scoring manuel (référence de comparaison) ».
3. Ouvrir **Comparer auto vs manuel**.

### Lire le résultat

L'onglet *Résumé* affiche la concordance globale, le kappa de Cohen, le F1 macro et un bloc **Diagnostic** (backend utilisé, source du scoring manuel, nombre d'époques de chaque côté, époques communes, exclusions). L'onglet *Métriques par stade* détaille sensibilité, précision, spécificité et F1, le scoring manuel servant de référence.

Repères d'interprétation du kappa : < 0,40 faible, 0,40-0,60 modéré, 0,60-0,80 bon, > 0,80 excellent. À titre de comparaison, l'accord entre deux experts humains plafonne généralement vers 0,80.

L'onglet *Paramètres du calcul* reprend les réglages du run automatique (backend, canal EEG/EOG/EMG **réellement** retenu, fréquences, seuil de confiance, âge et sexe, répartition des stades prédits, confiance moyenne), l'alignement des deux hypnogrammes, les principales confusions et une interprétation en clair. Le bouton **Exporter le rapport complet** reprend l'ensemble dans un CSV sectionné, suivi du détail époque par époque.

### Concordance attendue

| Backend | Concordance typique face à un consensus d'experts |
| --- | --- |
| YASA, U-Sleep | 80-88 % |
| PFTSleep | 78-87 % |
| ML, ML+HMM | 70-86 % |
| Règles AASM, Règles+HMM | 60-78 % |

Une concordance nettement inférieure à ces valeurs n'est pas une simple divergence de jugement : elle signale un problème de signal ou de réglage.

### Un stade jamais prédit

Si le modèle ne produit **aucune** époque d'un stade que le scoreur humain a bien annoté, CESA affiche un bandeau orange. Ce cas ne s'explique pas par un désaccord de frontière et se traite comme un défaut technique :

- **N3 absent** : le stade est défini par un critère d'amplitude (ondes lentes > 75 µV crête à crête). Une mise à l'échelle erronée (données en µV interprétées en V, ou l'inverse), une calibration incorrecte ou une dérivation EEG inadaptée (électrode frontale ou occipitale au lieu d'une centrale C3/C4) suffisent à l'empêcher. Vérifier l'amplitude affichée dans le visualiseur avant de conclure sur le modèle.
- **REM classé éveil** : sans canal EMG mentonnier, la distinction repose sur l'EEG et l'EOG seuls. Renseigner un EMG **mentonnier** si l'enregistrement en contient un ; un EMG de jambe n'y remédie pas et aggrave le problème.

Le rapport indique aussi la concordance obtenue en **fusionnant N2 et N3**. Un écart important avec la concordance brute signifie que le modèle lit correctement l'architecture de la nuit et ne diverge que sur le placement de cette frontière, ce qui est bien moins grave qu'une erreur de structure.

### Mesure du signal : le N3 est-il détectable ?

Pour éviter d'en rester aux hypothèses, CESA mesure l'amplitude du canal EEG **réellement utilisé** pendant le scoring et applique le critère AASM du N3 (ondes lentes 0,5-2 Hz de plus de 75 µV crête à crête sur au moins 20 % de l'époque). Trois lectures possibles, reportées dans l'onglet *Paramètres du calcul* :

- **amplitude crête à crête hors de la plage 20-1000 µV** : problème d'échelle ou de calibration. Un signal trop faible ne peut pas franchir le seuil de 75 µV, le N3 est mécaniquement indétectable et le modèle n'est pas en cause ;
- **aucune époque ne remplit le critère** alors que le scoreur humain a annoté du N3 : le signal analysé n'est pas celui sur lequel la lecture manuelle a été faite (dérivation, filtre passe-haut trop agressif, unités) ;
- **le critère est rempli sur une part notable des époques mais le modèle ne prédit presque pas de N3** : les ondes lentes sont bien là, la limite vient du modèle.

Dans ce dernier cas, la case **Corriger le N3** de la fenêtre YASA relève en N3 les époques predites N2 dont au moins 20 % sont des ondes lentes > 75 µV, mesurées de préférence sur `F4-M1`. L'éveil et le REM ne sont pas touchés. Sur une PSG, scorer YASA sur la **frontale** (sans EMG si le N3 est l'objectif) puis cocher cette option est le levier le plus efficace ; le menton `1-2` améliore le REM mais fait chuter le N3.

### Un résultat à 100 % est un signal d'alarme

Une concordance de 100 % ne se produit pas entre deux scoreurs indépendants. CESA affiche un bandeau rouge quand les deux hypnogrammes sont strictement identiques. Causes possibles :

- le scoring manuel a été initialisé depuis le scoring automatique et aucune époque n'a encore été corrigée ;
- le même fichier a été utilisé des deux côtés.

De même, une couverture commune faible (moins de 80 % des époques) déclenche un avertissement : les deux hypnogrammes ne se recouvrent probablement pas dans le temps, ou leurs durées d'époque diffèrent.

### Comparer plusieurs configurations plutôt que tâtonner

**Sommeil → Comparer des configurations de scoring** score plusieurs configurations d'affilée et mesure chacune face au **même** scoring manuel. Un scoring manuel de référence est donc requis.

**L'algorithme se choisit en haut de la fenêtre.** Les backends dont les prérequis manquent (paquet non installé, poids absents, modèle non entraîné) sont marqués *indisponible* et ne peuvent pas être sélectionnés ; le backend proposé par défaut est celui du scoring automatique s'il est utilisable, sinon le premier qui l'est. Une configuration dont le backend échoue à la vérification préalable n'est jamais lancée : elle est reportée directement comme échec avec la raison, sans consommer de temps de calcul.

La case **Ajouter une configuration par autre algorithme disponible** répond à l'autre question : non plus « quels canaux pour cet algorithme » mais « quel algorithme pour cet enregistrement ». Chaque algorithme y est testé avec les canaux que son article recommande, pas avec ceux d'un autre.

CESA propose ensuite les configurations pertinentes pour le montage chargé, en faisant varier **un élément à la fois** — chaque dérivation EEG disponible, puis l'effet de retirer l'EOG, puis chaque EOG, puis chaque EMG valide, puis l'effet des métadonnées âge/sexe. Le nombre de calculs reste ainsi proportionnel au nombre de canaux au lieu d'exploser en produit cartésien. Les canaux proposés sont **ceux du profil de l'algorithme choisi** : aucun EMG n'est testé pour U-Sleep qui ne le lit pas, et les métadonnées ne sont balayées que pour les backends qui les exploitent. Seuls les EMG utilisables comme entrée de scoring sont proposés : un EMG de jambe ou une mastoïde n'apparaissent jamais dans la liste.

La durée estimée est affichée avant le lancement, calculée à partir de la durée réellement mesurée lors du dernier scoring. Le balayage est interruptible et les configurations déjà testées restent affichées.

Le classement se fait par défaut sur le **kappa de Cohen**, car la concordance brute flatte une configuration qui se contente de prédire le stade majoritaire. Le tableau donne aussi le F1 du N3 et du REM, ce qui permet de préférer une configuration légèrement moins bonne globalement mais meilleure sur le stade qui vous intéresse ; le résumé le signale explicitement quand c'est le cas.

Votre scoring et vos réglages sont restaurés à la fin du balayage : rien n'est écrasé. Appliquer une configuration est une action explicite, qui met à jour l'algorithme, les listes de canaux et les métadonnées ; il reste ensuite à relancer le scoring automatique.

### Époques non scorées

Par défaut, les époques `U` (non scorées, artefact, mouvement) sont exclues des deux côtés : ce ne sont pas des stades de sommeil, et les inclure fait chuter artificiellement la concordance. La case à cocher en haut de la fenêtre permet de les réintégrer pour inspection.

## 5. Normalisation des libellés

Les deux hypnogrammes sont normalisés vers `W / N1 / N2 / N3 / R / U` avant comparaison : `REM` → `R`, `WAKE` / `Éveil` → `W`, codes numériques `0-4`, `artefact` / `mouvement` / `?` → `U`. Un scoring externe utilisant `REM` n'est donc plus compté comme un désaccord face à un backend qui produit `R`.

## Voir aussi

- [Import_Excel.md](Import_Excel.md) — format attendu pour un scoring manuel importé
- [Methodes_Statistiques.md](Methodes_Statistiques.md) — métriques et statistiques de sommeil
- [Troubleshooting.md](Troubleshooting.md) — dépendances manquantes
