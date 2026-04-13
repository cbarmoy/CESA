# Changelog

All notable changes to CESA are documented in this file.

## [0.0beta1.1] - 2026-04-05

### Added
- **Composable filter engine** (`CESA/filter_engine.py`): typed filter nodes
  (Bandpass, Highpass, Lowpass, Notch, Smoothing) with `FilterPipeline` chaining,
  JSON serialization, and frequency-response computation.
- **Clinical filter presets** (`config/filter_presets.json`): 11 built-in presets
  (EEG Standard PSG, AASM Scoring, EOG, EMG, ECG, Notch 50/60 Hz, etc.).
- **User preset management**: separate user/built-in storage, import/export
  to standalone JSON files for sharing between users.
- **Physiological range validation**: AASM-based parameter ranges for EEG, EOG,
  EMG, ECG with visual warnings on every slider.
- **Audit logging** (`FilterAuditLog`): timestamped record of every filter
  configuration change, exportable to JSON.
- **Undo/Redo** (`UndoManager`): stack-based undo/redo with Ctrl+Z / Ctrl+Y
  shortcuts, bounded depth (60 levels).
- **Adaptive filter suggestions** (`AdaptiveFilterSuggester`): rule-based and
  spectral-analysis-driven preset recommendations with confidence scores;
  detects delta/alpha/beta dominance and 50/60 Hz mains contamination.
- **Professional filter dialog** (`ui/filter_dialog.py` v3): dark/light theme
  toggle, channel search, multi-select with group buttons (All EEG / EOG / EMG /
  ECG), batch apply/copy/reset, collapsible filter cards with color-coded tags,
  3-panel preview (signal overlay + frequency response + PSD) with SNR indicator,
  adaptive suggestions popup, status bar.
- **HTML report generator** (`CESA/report_generator.py`): combined pipeline,
  audit log, embedded SVG charts, channel annotations, and adaptive suggestions
  in a single styled HTML document.
- **Favorite presets**: star/unstar presets for quick access, persisted in user
  config.
- **Channel annotations**: per-channel text notes visible in the dialog and
  included in exported reports.
- **Sleep pipeline** enhancements: HMM with Viterbi decoding, advanced features
  (spindles, K-complexes, REM bursts, spectral entropy), temporal context,
  clinical metrics (TST, SE, SOL, WASO), SHAP explainability, Transformer/
  Attention DL models.
- **217 unit tests** covering filters, pipelines, presets, serialization,
  physiological warnings, audit log, undo/redo, adaptive suggestions,
  report generation, favorites, and annotations.

### Changed
- Unified version to **0.0beta1.1** across all modules (`CESA/__init__.py`,
  `sleep_pipeline/__init__.py`, `run.py`, `requirements.txt`).
- `CESA/filters.py` now delegates to `filter_engine.py` while preserving the
  original `apply_filter()` API for backward compatibility.
- `CESA/psg_plot.py` uses `FilterPipeline.apply()` with fallback to legacy.
- `CESA/profile_schema.py` extended with `channel_filter_pipelines` field and
  automatic migration from old `channel_filter_params`.
- Preview debounce reduced from 250 ms to **150 ms** for instant feedback.

### Fixed
- `ValueError` from SciPy IIR filters on very short or empty data: all filter
  `apply()` methods now handle gracefully with try/except + size checks.

## [3.0.0] - 2025-01-27

- Initial modular package release (filters, scoring_io, theme, entropy).

## [0.0alpha4.0] - 2025-11-01

- Original launcher and EEG Studio interface.
