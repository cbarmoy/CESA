"""Subject-level k-fold cross-validation for sleep scoring benchmarks.

Ensures no data leakage by splitting at the subject level using
``sklearn.model_selection.GroupKFold``.  Provides fold execution and
result aggregation with bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from .datasets import DatasetRecord

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""

    fold_index: int
    backend: str
    n_train_subjects: int = 0
    n_test_subjects: int = 0
    n_train_epochs: int = 0
    n_test_epochs: int = 0
    accuracy: float = 0.0
    cohen_kappa: float = 0.0
    macro_f1: float = 0.0
    per_stage: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clinical_metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    comparison_report: Optional[Any] = None
    error_analysis: Optional[Any] = None
    feature_importance: Optional[Dict[str, float]] = None
    ml_model: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "fold_index": self.fold_index,
            "backend": self.backend,
            "n_train_subjects": self.n_train_subjects,
            "n_test_subjects": self.n_test_subjects,
            "n_train_epochs": self.n_train_epochs,
            "n_test_epochs": self.n_test_epochs,
            "accuracy": self.accuracy,
            "cohen_kappa": self.cohen_kappa,
            "macro_f1": self.macro_f1,
            "per_stage": self.per_stage,
            "clinical_metrics": self.clinical_metrics,
        }
        if self.confusion_matrix is not None:
            d["confusion_matrix"] = self.confusion_matrix.tolist()
        return d


@dataclass
class AggregatedResults:
    """Aggregated metrics across all folds for one backend."""

    backend: str
    n_folds: int = 0
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    kappa_mean: float = 0.0
    kappa_std: float = 0.0
    macro_f1_mean: float = 0.0
    macro_f1_std: float = 0.0
    accuracy_ci: Tuple[float, float] = (0.0, 0.0)
    kappa_ci: Tuple[float, float] = (0.0, 0.0)
    macro_f1_ci: Tuple[float, float] = (0.0, 0.0)
    per_stage_mean: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clinical_metrics_mean: Dict[str, float] = field(default_factory=dict)
    clinical_metrics_std: Dict[str, float] = field(default_factory=dict)
    fold_results: List[FoldResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "n_folds": self.n_folds,
            "accuracy": {"mean": self.accuracy_mean, "std": self.accuracy_std,
                         "ci_low": self.accuracy_ci[0], "ci_high": self.accuracy_ci[1]},
            "kappa": {"mean": self.kappa_mean, "std": self.kappa_std,
                      "ci_low": self.kappa_ci[0], "ci_high": self.kappa_ci[1]},
            "macro_f1": {"mean": self.macro_f1_mean, "std": self.macro_f1_std,
                         "ci_low": self.macro_f1_ci[0], "ci_high": self.macro_f1_ci[1]},
            "per_stage_mean": self.per_stage_mean,
            "clinical_metrics_mean": self.clinical_metrics_mean,
            "clinical_metrics_std": self.clinical_metrics_std,
            "folds": [fr.to_dict() for fr in self.fold_results],
        }


# ---------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------

def subject_kfold_split(
    records: List[DatasetRecord],
    n_folds: int = 5,
    random_seed: int = 42,
) -> Generator[Tuple[List[DatasetRecord], List[DatasetRecord]], None, None]:
    """Yield (train_records, test_records) for each fold.

    Splitting is done at the subject level -- all recordings from the
    same subject stay together in either train or test.
    """
    from sklearn.model_selection import GroupKFold

    subjects = np.array([r.subject_id for r in records])
    unique_subjects = np.unique(subjects)
    actual_folds = min(n_folds, len(unique_subjects))
    if actual_folds < n_folds:
        logger.warning("Only %d unique subjects, reducing folds from %d to %d",
                        len(unique_subjects), n_folds, actual_folds)

    gkf = GroupKFold(n_splits=actual_folds)
    indices = np.arange(len(records))

    for train_idx, test_idx in gkf.split(indices, groups=subjects):
        train = [records[i] for i in train_idx]
        test = [records[i] for i in test_idx]

        # Verify no leakage
        train_subjects = {r.subject_id for r in train}
        test_subjects = {r.subject_id for r in test}
        assert train_subjects.isdisjoint(test_subjects), "Subject leakage detected!"

        yield train, test


# ---------------------------------------------------------------------
# Fold execution
# ---------------------------------------------------------------------

def run_fold(
    fold_index: int,
    train_records: List[DatasetRecord],
    test_records: List[DatasetRecord],
    backend: str,
    *,
    ml_model_type: str = "hgb",
    epoch_duration_s: float = 30.0,
    target_sfreq: float = 100.0,
    random_seed: int = 42,
) -> FoldResult:
    """Execute one fold: preprocess, train (if ML), score, evaluate."""
    import mne
    from CESA.sleep_pipeline.contracts import ScoringResult, Epoch, StageLabel
    from CESA.sleep_pipeline.evaluation import compare, compute_clinical_metrics, error_analysis
    from CESA.sleep_pipeline.preprocessing import preprocess, PreprocessingConfig
    from CESA.sleep_pipeline.features import extract_all_features

    result = FoldResult(
        fold_index=fold_index,
        backend=backend,
        n_train_subjects=len({r.subject_id for r in train_records}),
        n_test_subjects=len({r.subject_id for r in test_records}),
    )

    config = PreprocessingConfig(
        target_sfreq=target_sfreq,
        epoch_duration_s=epoch_duration_s,
    )

    # -- Train phase (ML backends) --
    ml_model = None
    if backend in ("ml", "ml_hmm"):
        from CESA.sleep_pipeline.ml_scorer import train_model
        train_features: List[Dict] = []
        train_labels: List[str] = []

        for rec in train_records:
            try:
                raw = mne.io.read_raw_edf(rec.edf_path, preload=True, verbose=False)
                epoched = preprocess(raw, config)
                feats = extract_all_features(epoched)
                n = min(len(feats), len(rec.labels))
                train_features.extend(feats[:n])
                train_labels.extend(rec.labels[:n])
            except Exception as exc:
                logger.warning("Train preprocessing failed for %s: %s",
                               rec.subject_id, exc)

        result.n_train_epochs = len(train_features)
        if train_features and train_labels:
            try:
                ml_model = train_model(
                    train_features, train_labels,
                    model_type=ml_model_type,
                    random_state=random_seed,
                )
                result.ml_model = ml_model
            except Exception as exc:
                logger.error("Model training failed: %s", exc)
                return result

    # -- Test phase --
    all_ref_labels: List[str] = []
    all_pred_labels: List[str] = []
    test_features_all: List[Dict] = []

    for rec in test_records:
        try:
            raw = mne.io.read_raw_edf(rec.edf_path, preload=True, verbose=False)
            epoched = preprocess(raw, config)
            feats = extract_all_features(epoched)
        except Exception as exc:
            logger.warning("Test preprocessing failed for %s: %s",
                           rec.subject_id, exc)
            continue

        n = min(len(feats), len(rec.labels))
        ref_labels = rec.labels[:n]
        test_features_all.extend(feats[:n])
        all_ref_labels.extend(ref_labels)

        # Score
        try:
            predicted = _score_features(
                feats[:n], backend, ml_model, epoch_duration_s,
            )
            all_pred_labels.extend(predicted)
        except Exception as exc:
            logger.warning("Scoring failed for %s: %s", rec.subject_id, exc)
            all_pred_labels.extend(["U"] * n)

    result.n_test_epochs = len(all_ref_labels)

    if not all_ref_labels:
        return result

    # -- Build ScoringResults for evaluation --
    ref_result = _labels_to_scoring_result(all_ref_labels, epoch_duration_s, "reference")
    pred_result = _labels_to_scoring_result(all_pred_labels, epoch_duration_s, backend)

    try:
        report = compare(ref_result, pred_result)
        result.accuracy = report.accuracy
        result.cohen_kappa = report.cohen_kappa
        result.macro_f1 = report.macro_f1
        if report.confusion_matrix is not None:
            cm = report.confusion_matrix
            result.confusion_matrix = cm.values if hasattr(cm, 'values') else np.asarray(cm)
        result.per_stage = {
            stage: {"precision": sm.precision, "recall": sm.recall,
                    "f1": sm.f1, "support": sm.support}
            for stage, sm in report.per_stage.items()
        }
        result.comparison_report = report
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)

    try:
        clinical = compute_clinical_metrics(pred_result)
        result.clinical_metrics = clinical.to_dict()
    except Exception:
        pass

    try:
        ref_stages = [StageLabel.from_string(s) for s in all_ref_labels]
        pred_stages = [StageLabel.from_string(s) for s in all_pred_labels]
        result.error_analysis = error_analysis(ref_stages, pred_stages)
    except Exception:
        pass

    # Feature importance (only for ML)
    if ml_model is not None and test_features_all:
        try:
            from CESA.sleep_pipeline.ml_scorer import feature_importance
            result.feature_importance = feature_importance(
                ml_model, test_features_all, all_ref_labels,
                n_repeats=5, random_state=random_seed,
            )
        except Exception:
            pass

    return result


def _score_features(
    features: List[Dict],
    backend: str,
    ml_model: Optional[Any],
    epoch_duration_s: float,
) -> List[str]:
    """Score a list of feature dicts and return stage labels."""
    if backend == "aasm_rules":
        from CESA.sleep_pipeline.rules_aasm import score_rule_based
        result = score_rule_based(features, epoch_duration_s=epoch_duration_s)
        return result.stages

    if backend in ("ml", "ml_hmm"):
        from CESA.sleep_pipeline.ml_scorer import score_ml
        result = score_ml(
            features,
            epoch_duration_s=epoch_duration_s,
            model=ml_model,
            apply_smoothing=(backend == "ml"),
        )
        if backend == "ml_hmm":
            from CESA.sleep_pipeline.sequence_model import hmm_decode_scoring
            result = hmm_decode_scoring(result)
        return result.stages

    raise ValueError(f"Unknown backend: {backend}")


def _labels_to_scoring_result(
    labels: List[str],
    epoch_duration_s: float,
    backend: str,
) -> Any:
    """Convert a flat label list to a ScoringResult."""
    from CESA.sleep_pipeline.contracts import ScoringResult, Epoch, StageLabel

    epochs = []
    for i, s in enumerate(labels):
        epochs.append(Epoch(
            index=i,
            start_s=i * epoch_duration_s,
            duration_s=epoch_duration_s,
            stage=StageLabel.from_string(s),
        ))
    return ScoringResult(epochs=epochs, events=[], epoch_duration_s=epoch_duration_s,
                         backend=backend, metadata={})


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_folds(
    fold_results: List[FoldResult],
    *,
    bootstrap_n: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> AggregatedResults:
    """Aggregate per-fold results with bootstrap confidence intervals."""
    if not fold_results:
        return AggregatedResults(backend="unknown")

    backend = fold_results[0].backend
    accs = np.array([f.accuracy for f in fold_results])
    kappas = np.array([f.cohen_kappa for f in fold_results])
    f1s = np.array([f.macro_f1 for f in fold_results])

    agg = AggregatedResults(
        backend=backend,
        n_folds=len(fold_results),
        accuracy_mean=float(np.mean(accs)),
        accuracy_std=float(np.std(accs)),
        kappa_mean=float(np.mean(kappas)),
        kappa_std=float(np.std(kappas)),
        macro_f1_mean=float(np.mean(f1s)),
        macro_f1_std=float(np.std(f1s)),
        accuracy_ci=_bootstrap_ci(accs, bootstrap_n, ci_level, random_seed),
        kappa_ci=_bootstrap_ci(kappas, bootstrap_n, ci_level, random_seed),
        macro_f1_ci=_bootstrap_ci(f1s, bootstrap_n, ci_level, random_seed),
        fold_results=fold_results,
    )

    # Per-stage mean
    all_stages: set = set()
    for fr in fold_results:
        all_stages.update(fr.per_stage.keys())

    for stage in sorted(all_stages):
        metrics_for_stage: Dict[str, List[float]] = {}
        for fr in fold_results:
            if stage in fr.per_stage:
                for k, v in fr.per_stage[stage].items():
                    metrics_for_stage.setdefault(k, []).append(v)
        agg.per_stage_mean[stage] = {
            k: float(np.mean(v)) for k, v in metrics_for_stage.items()
        }

    # Clinical metrics mean/std
    cm_keys: set = set()
    for fr in fold_results:
        cm_keys.update(fr.clinical_metrics.keys())
    for k in sorted(cm_keys):
        vals = [fr.clinical_metrics.get(k, 0.0) for fr in fold_results]
        agg.clinical_metrics_mean[k] = float(np.mean(vals))
        agg.clinical_metrics_std[k] = float(np.std(vals))

    return agg


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) < 2:
        m = float(np.mean(values)) if len(values) > 0 else 0.0
        return (m, m)

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    n = len(values)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = (1 - ci_level) / 2
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return (lo, hi)
