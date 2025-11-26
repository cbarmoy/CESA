# CESA

CESA (Complex EEG Studio Analysis) is a Windows workstation for clinicians and researchers working with long-form EEG or polysomnography recordings. It sits on top of MNE-Python and bundles everything needed to inspect raw EDF+, run standard sleep scorings, compute advanced metrics (coherence, microstates, renormalised entropy) and export publication-ready plots.

## What you can do today

- **High-throughput review** – load multi-night EDF+, navigate with keyboard shortcuts, annotate artefacts, and mix manual + YASA auto-scoring inside the same UI.
- **Signal conditioning** – apply configurable band-pass/notch filters, baseline correction and montages on the fly.
- **Visual analytics (“spaghetti”)** – generate multi-channel/multi-stage PSD spiders, topomaps and per-stage summaries with one click.
- **Advanced metrics** – compute renormalised entropy directly in the UI, or run offline multiscale entropy (MSE) with full control over τ ranges, tolerances, pattern lengths and memory guards.
- **Exports & reporting** – dump all intermediate results to CSV, render high-quality PNG/PDF/SVG plots, and feed the data into R/Python notebooks or the bundled CLI helper.

## Multiscale Entropy (MSE)

The current build ships an experimental MSE workflow based on Costa et al. (2002) and implemented in `CESA/entropy.py`. Launch it from `Analyse → 🔬 Entropie multiscale (MSE)`:

- **Coarse-graining inspector**: CESA logs every τ (`⏩`, `✅`, `⚖️` messages) so you know exactly which file/channel/stage is being processed.
- **Per-channel SampEn**: amplitudes are normalised, padded and down-sampled (default 5 000 points) before computing `SampEn = -ln(A/B)` to avoid O(N²) memory explosions.
- **Offline mode**: all intermediate matrices (coarse-grained signals, counts, weighted covariances) can be exported for external analysis.
- **Safety rails**: only the first 200 000 samples of a stage are processed by default, with optional `--max-samples` / `--max-pattern-length` overrides.

## Offline CSV analysis helper

For large cohort work (e.g. AVANT vs APRÈS MBSR), use the standalone CLI in `tools/csv_analysis/`:

1. Export `profiles.csv` or `stats.csv` from the GUI.
2. Run `python tools/csv_analysis/analyse_csv.py --input path/to/profiles.csv`.
3. Filter by condition, stage or τ, and run Wilcoxon / permutation / bootstrap / robust-Z tests with optional Benjamini–Hochberg correction.
4. Produce quick plots (spaghetti, mean profiles, boxplots) and export filtered CSVs for statistical packages.

The CLI auto-detects the number of CPU cores and can run analyses in parallel (`--max-workers`).

## Getting started

1. Install Python ≥ 3.10, `pip install -r requirements.txt`.
2. Launch `python run.py`, point the UI to your EDF/PSG folder, and start exploring.
3. Check `docs/Methodes_Statistiques.md` for the exact formulas behind the sleep and MSE metrics.
4. For scripted pipelines, call the functions in `CESA/entropy.py` or use the CSV analysis CLI.
