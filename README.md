# CESA

![Version](https://img.shields.io/badge/version-0.0beta2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Tests](https://img.shields.io/badge/tests-217%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**CESA** (Complex EEG Studio Analysis) is a Windows workstation for clinicians and researchers working with long-form EEG or polysomnography recordings. It sits on top of MNE-Python and bundles everything needed to inspect raw EDF+, run standard sleep scorings, compute advanced metrics (coherence, microstates, renormalised entropy) and export publication-ready plots.

## What's new in v0.0beta2.0

- **Ocular cleaning (non-destructive)** -- EOG→EEG regression in memory with mandatory before/after preview and restore; recording files are never modified (`Filtres → Nettoyage oculaire...`).
- **Native Qt analysis dialogs** -- artifact detection, stage PSD, wavelet spectrogram, entropy, temporal metrics, connectivity, microstates, sleep periods, scoring editor/comparison, PSG report, markers, EDF segment export, batch/multi-subject.
- **Spectrogram bars** -- night-long and window spectrograms in the Qt viewer.
- **Sleep metrics & batch engine** -- clinical sleep stats helpers and batch processing utilities.
- **Import profiles** -- PSG / PSG-HD import presets under `config/import_profiles/`.

See [CHANGELOG.md](CHANGELOG.md) for the full history.

## What you can do today

- **High-throughput review** -- load multi-night EDF+, navigate with keyboard shortcuts, annotate artefacts, and mix manual + YASA auto-scoring inside the same UI.
- **Signal conditioning** -- apply configurable band-pass/notch filters, baseline correction and montages on the fly via the composable filter dialog.
- **Visual analytics ("spaghetti")** -- generate multi-channel/multi-stage PSD spiders, topomaps and per-stage summaries with one click.
- **Advanced metrics** -- compute renormalised entropy directly in the UI, or run offline multiscale entropy (MSE) with full control over τ ranges, tolerances, pattern lengths and memory guards.
- **Exports & reporting** -- dump all intermediate results to CSV, render high-quality PNG/PDF/SVG plots, generate HTML filter reports, and feed the data into R/Python notebooks or the bundled CLI helper.

## Multiscale Entropy (MSE)

The current build ships an experimental MSE workflow based on Costa et al. (2002) and implemented in `CESA/entropy.py`. Launch it from `Analyse → 🔬 Entropie multiscale (MSE)`:

- **Coarse-graining inspector**: CESA logs every τ so you know exactly which file/channel/stage is being processed.
- **Per-channel SampEn**: amplitudes are normalised, padded and down-sampled (default 5 000 points) before computing `SampEn = -ln(A/B)` to avoid O(N²) memory explosions.
- **Offline mode**: all intermediate matrices can be exported for external analysis.
- **Safety rails**: only the first 200 000 samples of a stage are processed by default, with optional overrides.

## Offline CSV analysis helper

For large cohort work (e.g. AVANT vs APRÈS MBSR), use the standalone CLI in `tools/csv_analysis/`:

1. Export `profiles.csv` or `stats.csv` from the GUI.
2. Run `python tools/csv_analysis/analyse_csv.py --input path/to/profiles.csv`.
3. Filter by condition, stage or τ, and run Wilcoxon / permutation / bootstrap / robust-Z tests with optional Benjamini-Hochberg correction.
4. Produce quick plots (spaghetti, mean profiles, boxplots) and export filtered CSVs for statistical packages.

The CLI auto-detects the number of CPU cores and can run analyses in parallel (`--max-workers`).

## Getting started

1. Install Python ≥ 3.10, `pip install -r requirements.txt`.
2. Launch `python run.py`, point the UI to your EDF/PSG folder, and start exploring.
3. Check `docs/Methodes_Statistiques.md` for the exact formulas behind the sleep and MSE metrics.
4. For scripted pipelines, call the functions in `CESA/entropy.py` or use the CSV analysis CLI.
5. To generate a standalone filter report: `python -m CESA.report_generator --audit audit.json --output report.html`.

## Windows standalone (PyInstaller)

The frozen build is produced with `pyinstaller cesa.spec` and lands under **`dist/CESA/`** (one-folder layout: `CESA.exe` plus DLLs, Qt plugins, and usually an `_internal` tree).

**Rules for distribution**

- **Copy the entire `dist/CESA/` folder** (zip it, ship it, extract it as-is on the target PC). Copying **only** `CESA.exe` will fail with errors such as **Shiboken6 / libShiboken not found** or **DLL load failed**, because PySide6’s native libraries live next to the executable inside that folder.
- On the **target** machine, install the **Microsoft Visual C++ Redistributable (x64)** (v14 covers VS 2015–2022-era runtimes used by Qt/PySide6). Many developer PCs already have it; clean laptops often do not. **Download hub (French):** [visualstudio.microsoft.com/fr/downloads](https://visualstudio.microsoft.com/fr/downloads/) — expand **« Tous les téléchargements »** / **« Autres outils, infrastructure et redistribuables »**, then **« Microsoft Visual C++ v14 Redistributable »** and pick **x64**. Technical reference: [Latest supported VC++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).
- Prefer a short path without sync quirks; some antivirus or OneDrive policies can block or strip DLLs from the bundle.

`INSTALL.bat` (developer setup) includes checks for the VC++ runtime and, if `dist/CESA/CESA.exe` is present, for a coherent Shiboken6 DLL next to the PyInstaller output.

## Project structure

```
CESA-main/
├── CESA/                    # Core package
│   ├── __init__.py          # Version 0.0beta2.0
│   ├── filter_engine.py     # Composable filter engine
│   ├── filters.py           # Legacy wrapper (backward-compat)
│   ├── report_generator.py  # HTML report generator
│   ├── eeg_studio_fixed.py  # Main GUI
│   ├── psg_plot.py          # PSG signal renderer
│   ├── profile_schema.py    # User profiles + filter pipeline schema
│   └── sleep_pipeline/      # AASM scoring pipeline (ML/DL/HMM)
├── ui/
│   └── filter_dialog.py     # Professional filter config dialog (v4)
├── config/
│   └── filter_presets.json   # Built-in clinical presets
├── tests/                   # 217+ unit tests
├── scripts/                 # CLI utilities
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
├── run.py                   # Main launcher
├── CHANGELOG.md             # Release history
└── README.md                # This file
```
