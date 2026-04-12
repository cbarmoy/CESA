"""Experiment orchestrator tying datasets, CV, scoring, and figures.

The single entry point :func:`run_experiment` executes a full
reproducible benchmark and writes all artefacts to disk.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BenchmarkConfig, fix_all_seeds, log_environment, setup_output_dir
from .cross_validation import (
    AggregatedResults,
    FoldResult,
    aggregate_folds,
    run_fold,
    subject_kfold_split,
)
from .datasets import DatasetRecord, load_dataset
from .inter_scorer import InterScorerReport, compute_inter_scorer_report

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Complete results of a benchmark experiment."""

    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    environment: Dict[str, Any] = field(default_factory=dict)
    n_records: int = 0
    n_subjects: int = 0
    per_backend: Dict[str, AggregatedResults] = field(default_factory=dict)
    inter_scorer: Optional[InterScorerReport] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "environment": self.environment,
            "n_records": self.n_records,
            "n_subjects": self.n_subjects,
            "elapsed_seconds": self.elapsed_seconds,
            "output_dir": self.output_dir,
        }
        d["per_backend"] = {k: v.to_dict() for k, v in self.per_backend.items()}
        if self.inter_scorer:
            d["inter_scorer"] = self.inter_scorer.to_dict()
        if self.feature_importance:
            d["feature_importance"] = dict(
                sorted(self.feature_importance.items(), key=lambda x: -x[1])[:30]
            )
        return d


def run_experiment(config: BenchmarkConfig) -> ExperimentResult:
    """Execute a full reproducible benchmark experiment.

    Steps:
    1. Fix all random seeds
    2. Log environment metadata
    3. Load dataset
    4. Subject-level k-fold cross-validation
    5. For each fold x backend: preprocess, train, score, evaluate
    6. Aggregate results
    7. Compute inter-scorer agreement
    8. Generate all figures
    9. Generate paper report
    10. Export everything

    Returns the :class:`ExperimentResult`.
    """
    t0 = time.time()

    # 1. Seeds
    fix_all_seeds(config.random_seed)

    # 2. Environment
    env = log_environment()

    # 3. Output directory
    out_dir = setup_output_dir(config)

    result = ExperimentResult(
        config=config,
        environment=env,
        output_dir=str(out_dir),
    )

    # 4. Load dataset
    logger.info("Loading dataset: %s", config.dataset)
    try:
        records = load_dataset(
            config.dataset,
            data_path=config.data_path,
            n_subjects=config.n_subjects,
            epoch_duration_s=config.epoch_duration_s,
        )
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        result.elapsed_seconds = time.time() - t0
        return result

    result.n_records = len(records)
    result.n_subjects = len({r.subject_id for r in records})
    logger.info("Loaded %d records from %d subjects",
                result.n_records, result.n_subjects)

    if not records:
        logger.error("No records loaded -- aborting")
        result.elapsed_seconds = time.time() - t0
        return result

    # 5. Cross-validation per backend
    all_fold_results: Dict[str, List[FoldResult]] = {b: [] for b in config.backends}

    for fold_idx, (train, test) in enumerate(
        subject_kfold_split(records, config.n_folds, config.random_seed)
    ):
        logger.info("=== Fold %d/%d: %d train / %d test records ===",
                     fold_idx + 1, config.n_folds, len(train), len(test))

        for backend in config.backends:
            logger.info("  Backend: %s", backend)
            try:
                fr = run_fold(
                    fold_idx, train, test, backend,
                    ml_model_type=config.ml_model_type,
                    epoch_duration_s=config.epoch_duration_s,
                    target_sfreq=config.target_sfreq,
                    random_seed=config.random_seed,
                )
                all_fold_results[backend].append(fr)
                logger.info("    kappa=%.3f  acc=%.3f  F1=%.3f",
                            fr.cohen_kappa, fr.accuracy, fr.macro_f1)
            except Exception as exc:
                logger.error("  Fold %d backend %s failed: %s",
                             fold_idx, backend, exc)

    # 6. Aggregate
    for backend in config.backends:
        folds = all_fold_results[backend]
        if folds:
            agg = aggregate_folds(
                folds,
                bootstrap_n=config.bootstrap_n,
                ci_level=config.bootstrap_ci,
                random_seed=config.random_seed,
            )
            result.per_backend[backend] = agg
            logger.info("Backend %s: kappa=%.3f +/- %.3f",
                        backend, agg.kappa_mean, agg.kappa_std)

    # 7. Inter-scorer agreement (using last fold as example)
    _compute_inter_scorer(result, all_fold_results, config)

    # 8. Merge feature importance across folds
    _merge_feature_importance(result, all_fold_results)

    # 9. Generate figures
    _generate_figures(result, all_fold_results, config, out_dir)

    # 10. Export
    _export_results(result, out_dir)

    # 11. Paper report
    try:
        from .paper_report import generate_paper_report
        generate_paper_report(result, out_dir)
    except Exception as exc:
        logger.warning("Paper report generation failed: %s", exc)

    result.elapsed_seconds = time.time() - t0
    logger.info("Experiment complete in %.1f seconds. Output: %s",
                result.elapsed_seconds, out_dir)

    return result


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _compute_inter_scorer(
    result: ExperimentResult,
    all_fold_results: Dict[str, List[FoldResult]],
    config: BenchmarkConfig,
) -> None:
    """Build inter-scorer agreement from last fold's labels."""
    try:
        # Collect predictions from the last fold for each backend
        scorers: Dict[str, List[str]] = {}
        for backend, folds in all_fold_results.items():
            if folds and folds[-1].comparison_report:
                report = folds[-1].comparison_report
                pred_stages = [
                    ep.stage.value
                    for ep in report.metadata.get("pred_epochs", [])
                ] if "pred_epochs" in report.metadata else []
                if not pred_stages:
                    # Reconstruct from confusion data if available
                    continue
                scorers[backend] = pred_stages

        if len(scorers) >= 2:
            result.inter_scorer = compute_inter_scorer_report(
                scorers,
                epoch_duration_s=config.epoch_duration_s,
            )
    except Exception as exc:
        logger.warning("Inter-scorer computation failed: %s", exc)


def _merge_feature_importance(
    result: ExperimentResult,
    all_fold_results: Dict[str, List[FoldResult]],
) -> None:
    """Average feature importance across all ML folds."""
    all_imp: Dict[str, List[float]] = {}
    for folds in all_fold_results.values():
        for fr in folds:
            if fr.feature_importance:
                for k, v in fr.feature_importance.items():
                    all_imp.setdefault(k, []).append(v)

    if all_imp:
        result.feature_importance = {
            k: float(np.mean(v)) for k, v in all_imp.items()
        }


def _generate_figures(
    result: ExperimentResult,
    all_fold_results: Dict[str, List[FoldResult]],
    config: BenchmarkConfig,
    out_dir: Path,
) -> None:
    """Generate all publication figures."""
    fig_dir = out_dir / "figures"
    fmt = config.figure_format
    dpi = config.figure_dpi

    try:
        from . import figures as fig_mod

        # CV boxplot
        cv_data: Dict[str, Dict[str, List[float]]] = {}
        for backend, agg in result.per_backend.items():
            cv_data[backend] = {
                "accuracy": [f.accuracy for f in agg.fold_results],
                "kappa": [f.cohen_kappa for f in agg.fold_results],
                "f1": [f.macro_f1 for f in agg.fold_results],
            }
        if cv_data:
            fig_mod.plot_cv_boxplot(cv_data, output_path=fig_dir / "cv_boxplot", fmt=fmt, dpi=dpi)
            logger.info("Generated: cv_boxplot")

        # Confusion matrices (from last fold per backend)
        for backend, folds in all_fold_results.items():
            if folds and folds[-1].confusion_matrix is not None:
                fig_mod.plot_confusion_matrix(
                    folds[-1].confusion_matrix,
                    title=f"Confusion Matrix - {backend}",
                    output_path=fig_dir / f"confusion_{backend}",
                    fmt=fmt, dpi=dpi,
                )
                logger.info("Generated: confusion_%s", backend)

        # Feature importance
        if result.feature_importance:
            fig_mod.plot_feature_importance(
                result.feature_importance,
                output_path=fig_dir / "feature_importance",
                fmt=fmt, dpi=dpi,
            )
            logger.info("Generated: feature_importance")

        # Inter-scorer heatmap
        if result.inter_scorer:
            names, mat = result.inter_scorer.kappa_matrix_as_array()
            fig_mod.plot_kappa_heatmap(
                names, mat,
                output_path=fig_dir / "kappa_heatmap",
                fmt=fmt, dpi=dpi,
            )
            logger.info("Generated: kappa_heatmap")

        # Clinical metrics table
        clinical_data = {}
        for backend, agg in result.per_backend.items():
            if agg.clinical_metrics_mean:
                clinical_data[backend] = agg.clinical_metrics_mean
        if clinical_data:
            fig_mod.plot_clinical_metrics_table(
                clinical_data,
                output_path=fig_dir / "clinical_metrics",
                fmt=fmt, dpi=dpi,
            )
            logger.info("Generated: clinical_metrics")

    except Exception as exc:
        logger.warning("Figure generation failed: %s", exc)


def _export_results(result: ExperimentResult, out_dir: Path) -> None:
    """Export results as JSON and CSV."""
    # Main JSON
    (out_dir / "results.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    # Per-fold CSV
    rows = []
    for backend, agg in result.per_backend.items():
        for fr in agg.fold_results:
            rows.append({
                "backend": backend,
                "fold": fr.fold_index,
                "accuracy": fr.accuracy,
                "kappa": fr.cohen_kappa,
                "macro_f1": fr.macro_f1,
                "n_train_epochs": fr.n_train_epochs,
                "n_test_epochs": fr.n_test_epochs,
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "results_per_fold.csv", index=False)

    # Clinical metrics CSV
    cm_rows = []
    for backend, agg in result.per_backend.items():
        row = {"backend": backend}
        row.update(agg.clinical_metrics_mean)
        cm_rows.append(row)
    if cm_rows:
        pd.DataFrame(cm_rows).to_csv(out_dir / "clinical_metrics.csv", index=False)

    logger.info("Results exported to %s", out_dir)
