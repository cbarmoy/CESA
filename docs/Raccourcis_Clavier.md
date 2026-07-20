# Raccourcis clavier CESA

Ce document est la reference unique des raccourcis. Le comportement depend du mode de lancement.

## Matrice par mode

| Raccourci | `run.py` (Qt seul) | `eeg_studio_fixed.py` (Tk) | Tk + viewer Qt |
|---|---|---|---|
| `Ctrl+O` | Ouvrir enregistrement | Ouvrir EDF | Selon la fenetre active |
| `Ctrl+Q` | Quitter | Quitter | Selon la fenetre active |
| `Ctrl+S` | - | Exporter donnees | Tk uniquement |
| `F1` | - | Guide utilisateur | Tk uniquement |
| `F5` | - | Rafraichir plot | Tk uniquement |
| `F8` | Inspection panel | - | Qt uniquement |
| `F9` | Dashboard panel | - | Qt uniquement |
| `Ctrl+Alt+F` | Config filtres Qt | Affiche en menu (Tk) | Qt prioritaire si focus Qt |
| `Ctrl+N` | Normalisation amplitudes | - | Qt uniquement |
| `Ctrl+A` | Autoscale (via navigation Qt) | Autoscale | Harmonise autoscale |
| `Ctrl+Shift+N` | Nouvelle annotation | - | Qt uniquement |
| `0..5` | Scoring W/N1/N2/N3/R/U | Scoring selon contexte Tk | Qt prioritaire si focus Qt |
| `Ctrl+Shift+A` | Toggle auto-avance scoring | Toggle auto-avance scoring | Disponible des deux cotes |
| `Ctrl+Shift+F` | FFT scoring epoch | FFT scoring epoch | Preferer Qt (ApplicationShortcut) |
| `Ctrl+Alt+X` | FFT scoring epoch (fallback) | FFT scoring epoch (fallback) | Preferer Qt (ApplicationShortcut) |
| `Ctrl+B` | Batch FFT | Batch FFT | Selon fenetre active |
| `Z/Q/S/D` | Navigation/epochs Qt | Navigation Tk | Depend du focus |
| `Left/Right` | Navigation Qt | Navigation Tk | Depend du focus |

## Notes importantes

- En mode hybride, la fenetre avec le focus recoit les touches.
- Les raccourcis critiques de scoring/FFT sont installes en `ApplicationShortcut` cote Qt pour etre plus robustes.
- Les entrees de menu Tk avec `accelerator=` affichent un raccourci, mais il faut aussi un `bind` pour qu'il fonctionne.
- Si un raccourci semble inactif, verifier d'abord: mode de lancement, focus, puis clavier (AZERTY/AltGr).
