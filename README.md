# CESA

![Version](https://img.shields.io/badge/version-0.0beta1.1-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Tests](https://img.shields.io/badge/tests-217%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**CESA** (Complex EEG Studio Analysis) is a Windows workstation for clinicians and researchers working with long-form EEG or polysomnography recordings. It sits on top of MNE-Python and bundles everything needed to inspect raw EDF+, run standard sleep scorings, compute advanced metrics (coherence, microstates, renormalised entropy) and export publication-ready plots.

## What's new in v0.0beta1.1

- **Composable filter engine** -- typed filter nodes (Bandpass, Highpass, Lowpass, Notch, Smoothing) with `FilterPipeline` chaining, JSON serialization, frequency-response visualisation.
- **Clinical filter presets** -- 11 built-in presets (EEG Standard PSG, AASM Scoring, EOG, EMG, ECG, Notch 50/60 Hz, etc.) + user presets with import/export.
- **Physiological range validation** -- AASM-based parameter ranges with visual warnings on every slider.
- **Audit logging** -- timestamped record of every filter configuration change, exportable to JSON.
- **Undo/Redo** -- 60-level stack with Ctrl+Z / Ctrl+Y shortcuts.
- **Adaptive filter suggestions** -- rule-based and spectral-analysis-driven preset recommendations (delta/alpha/beta detection, 50/60 Hz mains contamination).
- **Professional filter dialog** -- dark/light theme, channel search, batch apply, collapsible filter cards with color-coded tags, 3-panel preview (signal overlay + frequency response + PSD) with SNR indicator.
- **Favorite presets** -- star/unstar presets for quick access, persisted to JSON.
- **Channel annotations** -- per-channel text notes, visible in the dialog and included in reports.
- **HTML report generator** -- combined pipeline summary, audit log, embedded SVG charts, annotations, and adaptive suggestions in a single styled HTML document. CLI and UI export.
- **Mini-dashboard** -- popup summary of filtering state (active channels, warnings, SNR per channel, recent audit actions).
- **Sleep pipeline** v4 -- HMM with Viterbi decoding, SHAP explainability, Transformer/Attention DL models, clinical metrics (TST, SE, SOL, WASO), temporal context features.
- **217+ unit tests** with zero regressions.

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

## Project structure

```
CESA-main/
├── CESA/                    # Core package
│   ├── __init__.py          # Version 0.0beta1.1
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
