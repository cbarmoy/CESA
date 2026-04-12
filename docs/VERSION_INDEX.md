# Index des versions CESA

> **Version actuelle : `0.0beta1.0`**
>
> Ce fichier recense **tous** les emplacements où le numéro de version apparaît dans le projet.
> Lors d'un bump de version, parcourir cette checklist pour garantir la cohérence.

---

## 1. Source de vérité

| Fichier | Ligne | Rôle |
|---|---|---|
| `CESA/__init__.py` | `__version__ = "..."` | **Version canonique** du package Python |

Toute mise à jour de version **doit commencer ici**.

---

## 2. Lanceurs & scripts principaux

| Fichier | Occurrences | Détail |
|---|---|---|
| `run.py` | 5 | Docstring (l. 3, 10), splash fallback (l. 203), print console (l. 264), logging (l. 277) |
| `RUN.bat` | 0 | Pas de version affichée (uniquement le nom) |
| `INSTALL.bat` | 12 | Header, echo messages, raccourci bureau (.lnk), lancement |
| `push_to_github.bat` | 7 | Header, echo, commit message, tag Git |

---

## 3. Modules Python (`CESA/`)

### 3.1 Interface principale

| Fichier | Occurrences | Détail |
|---|---|---|
| `eeg_studio_fixed.py` | ~17 | Docstring header (l. 3, 14, 17, 19), `root.title()` (l. 561), barre de statut `version_label` (l. 12905, 12945), support `messagebox` (l. 8241-8267), rapport de bug (l. 9366), fenêtre "À propos" (l. 12108, 12128), FFT batch (l. 13070), `print()` final (l. 18250), `logging.info` (l. 484) |

### 3.2 Modules fonctionnels

| Fichier | Occurrences | Détail |
|---|---|---|
| `filters.py` | 1 | Docstring `Version:` |
| `entropy.py` | 3 | Docstring header + `Version:` |
| `spectral_analysis.py` | 3 | Docstring header + `Version:` |
| `scoring_io.py` | 3 | Docstring header + `Version:` |
| `psg_plot.py` | 1 | Docstring header |
| `sleep_pipeline/__init__.py` | 1 | `__version__ = "..."` |

### 3.3 Interface & thèmes

| Fichier | Occurrences | Détail |
|---|---|---|
| `user_assistant.py` | ~30 | Docstring (l. 2, 5, 19, 31), titres de fenêtres (welcome, visite guidée, explorateur, aide, référence, diagnostic), labels, textes d'aide, support `messagebox` |
| `ui_dialogs.py` | 4 | Docstring header + `Version:` |
| `theme_manager.py` | 3 | Docstring header + `Version:` |
| `math_formulas_window.py` | 3 | Docstring header + `Version:` + `window.title()` |
| `install_cesa.py` | ~14 | Docstring, `root.title()`, labels, guide d'installation, raccourci .lnk, messages |

---

## 4. Documentation (`docs/`)

| Fichier | Occurrences | Détail |
|---|---|---|
| `Quickstart.md` | 2 | Intro + pied de page |
| `Installation.md` | 2 | Intro + pied de page |
| `Troubleshooting.md` | 2 | Intro + pied de page |
| `Methodes_Statistiques.md` | 1 | Pied de page |
| `Import_Excel.md` | 1 | Pied de page |
| `Guide_Navigation.md` | 1 | Pied de page |
| `CHANGELOG.md` | variable | Entrée de la version courante (ne pas modifier les entrées historiques) |
| `README.md` | ~3 | Badge version, section "What's new", arborescence |

---

## 5. Scripts secondaires (`documentation/scripts/`)

| Fichier | Occurrences | Détail |
|---|---|---|
| `prepare_release.py` | ~11 | Docstring, changelog généré, instructions tag Git |
| `run_eeg_studio.py` | 4 | Docstring header + `Version:` + print |
| `install.bat` | 4 | Header REM + echo + `Version:` |

---

## 6. Tests

| Fichier | Occurrences | Détail |
|---|---|---|
| `tests/test_filter_engine.py` | 5 | Assertions `__version__`, contenu de `run.py` et `CHANGELOG.md` |

---

## 7. Autres

| Fichier | Occurrences | Détail |
|---|---|---|
| `requirements.txt` | 1 | Commentaire header |

---

## 8. Fichiers à NE PAS modifier

| Fichier | Raison |
|---|---|
| `CHANGELOG.md` (entrées historiques) | Les anciennes entrées (`[0.0alpha4.0]`, etc.) documentent l'historique |
| `archives/` | Fichiers archivés, figés dans le temps |

---

## Procédure de bump de version

1. **Modifier `CESA/__init__.py`** → `__version__ = "X.Y.Z"`
2. **Rechercher** l'ancienne version dans tout le projet :
   ```powershell
   rg "ANCIENNE_VERSION" --glob "*.{py,bat,md}" .
   ```
3. **Remplacer** dans chaque fichier listé ci-dessus (sauf archives et historique CHANGELOG)
4. **Vérifier** qu'il ne reste rien :
   ```powershell
   rg "ANCIENNE_VERSION" --glob "*.{py,bat,md}" .
   ```
5. **Mettre à jour ce fichier** (`docs/VERSION_INDEX.md`) avec la nouvelle version courante
6. **Ajouter une entrée** dans `CHANGELOG.md`
