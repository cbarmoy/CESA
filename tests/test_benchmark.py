"""Tests for the CESA benchmark / publication pipeline.

All tests run without network access or real dataset downloads by using
synthetic data and mocked loaders.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =====================================================================
# Config
# =====================================================================

from CESA.sleep_pipeline.benchmark.config import (
    BenchmarkConfig,
    fix_all_seeds,
    log_environment,
    setup_output_dir,
)


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.dataset == "sleep_edf"
        assert cfg.n_folds == 5
        assert cfg.random_seed == 42
        assert "aasm_rules" in cfg.backends

    def test_config_hash_deterministic(self):
        c1 = BenchmarkConfig(random_seed=42)
        c2 = BenchmarkConfig(random_seed=42)
        assert c1.config_hash == c2.config_hash

    def test_config_hash_differs(self):
        c1 = BenchmarkConfig(random_seed=42)
        c2 = BenchmarkConfig(random_seed=99)
        assert c1.config_hash != c2.config_hash

    def test_json_roundtrip(self, tmp_path):
        cfg = BenchmarkConfig(dataset="sleep_edf", n_folds=3, random_seed=123)
        path = str(tmp_path / "test_cfg.json")
        cfg.save(path)
        loaded = BenchmarkConfig.load(path)
        assert loaded.dataset == "sleep_edf"
        assert loaded.n_folds == 3
        assert loaded.random_seed == 123

    def test_yaml_roundtrip(self, tmp_path):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        cfg = BenchmarkConfig(dataset="mass", n_subjects=10)
        path = str(tmp_path / "test_cfg.yaml")
        cfg.save(path)
        loaded = BenchmarkConfig.load(path)
        assert loaded.dataset == "mass"
        assert loaded.n_subjects == 10

    def test_to_dict(self):
        cfg = BenchmarkConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "dataset" in d
        assert "backends" in d

    def test_unknown_fields_ignored(self, tmp_path):
        """Extra keys in config file are silently ignored."""
        path = tmp_path / "extra.json"
        data = BenchmarkConfig().to_dict()
        data["new_future_field"] = 999
        path.write_text(json.dumps(data))
        loaded = BenchmarkConfig.load(str(path))
        assert loaded.dataset == "sleep_edf"


class TestFixAllSeeds:
    def test_numpy_deterministic(self):
        fix_all_seeds(42)
        a = np.random.rand(5)
        fix_all_seeds(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_deterministic(self):
        import random
        fix_all_seeds(42)
        a = [random.random() for _ in range(5)]
        fix_all_seeds(42)
        b = [random.random() for _ in range(5)]
        assert a == b


class TestLogEnvironment:
    def test_has_required_keys(self):
        env = log_environment()
        assert "python_version" in env
        assert "platform" in env
        assert "numpy_version" in env
        assert "timestamp" in env


class TestSetupOutputDir:
    def test_creates_directories(self, tmp_path):
        cfg = BenchmarkConfig(output_dir=str(tmp_path / "results"))
        out = setup_output_dir(cfg)
        assert out.is_dir()
        assert (out / "figures").is_dir()
        assert (out / "environment.json").is_file()


# =====================================================================
# Datasets
# =====================================================================

from CESA.sleep_pipeline.benchmark.datasets import (
    DatasetRecord,
    _parse_sleepedf_hypnogram,
    _trim_trailing,
    load_dataset,
)


def _make_records(n_subjects: int = 6, n_epochs: int = 100) -> List[DatasetRecord]:
    """Create synthetic DatasetRecords for testing."""
    stages = ["W", "N1", "N2", "N3", "R"]
    records = []
    for i in range(n_subjects):
        labels = [stages[j % 5] for j in range(n_epochs)]
        records.append(DatasetRecord(
            subject_id=f"S{i:02d}",
            night=1,
            labels=labels,
            epoch_duration_s=30.0,
        ))
    return records


class TestDatasetRecord:
    def test_n_epochs(self):
        rec = DatasetRecord(subject_id="test", labels=["W", "N1", "N2"])
        assert rec.n_epochs == 3

    def test_empty(self):
        rec = DatasetRecord(subject_id="empty")
        assert rec.n_epochs == 0


class TestTrimTrailing:
    def test_trims_u(self):
        assert _trim_trailing(["W", "N1", "U", "U"]) == ["W", "N1"]

    def test_no_trailing_u(self):
        assert _trim_trailing(["W", "N1", "R"]) == ["W", "N1", "R"]

    def test_all_u(self):
        assert _trim_trailing(["U", "U"]) == []


class TestLoadDataset:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_mass_requires_data_path(self):
        with pytest.raises(ValueError, match="MASS requires data_path"):
            load_dataset("mass")


# =====================================================================
# Cross-Validation
# =====================================================================

from CESA.sleep_pipeline.benchmark.cross_validation import (
    FoldResult,
    AggregatedResults,
    subject_kfold_split,
    aggregate_folds,
    _bootstrap_ci,
)

_has_sklearn = True
try:
    import sklearn  # noqa: F401
except ImportError:
    _has_sklearn = False


@pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed")
class TestSubjectKFoldSplit:
    def test_no_leakage(self):
        records = _make_records(10)
        for train, test in subject_kfold_split(records, n_folds=5):
            train_subs = {r.subject_id for r in train}
            test_subs = {r.subject_id for r in test}
            assert train_subs.isdisjoint(test_subs), "Subject leakage!"

    def test_all_records_used(self):
        records = _make_records(10)
        all_test_ids = set()
        for _, test in subject_kfold_split(records, n_folds=5):
            for r in test:
                all_test_ids.add(r.subject_id)
        assert all_test_ids == {f"S{i:02d}" for i in range(10)}

    def test_reduces_folds_if_few_subjects(self):
        records = _make_records(3)
        folds = list(subject_kfold_split(records, n_folds=10))
        assert len(folds) == 3

    def test_yields_correct_fold_count(self):
        records = _make_records(10)
        folds = list(subject_kfold_split(records, n_folds=5))
        assert len(folds) == 5


class TestBootstrapCI:
    def test_single_value(self):
        lo, hi = _bootstrap_ci(np.array([0.8]), 100, 0.95, 42)
        assert lo == hi == 0.8

    def test_range_sensible(self):
        vals = np.array([0.7, 0.75, 0.8, 0.82, 0.85])
        lo, hi = _bootstrap_ci(vals, 1000, 0.95, 42)
        assert lo < hi
        assert lo >= 0.6
        assert hi <= 0.9

    def test_deterministic(self):
        vals = np.array([0.7, 0.8, 0.9])
        ci1 = _bootstrap_ci(vals, 500, 0.95, 42)
        ci2 = _bootstrap_ci(vals, 500, 0.95, 42)
        assert ci1 == ci2


class TestAggregateFolds:
    def test_basic_aggregation(self):
        folds = [
            FoldResult(fold_index=0, backend="ml", accuracy=0.80, cohen_kappa=0.70, macro_f1=0.75),
            FoldResult(fold_index=1, backend="ml", accuracy=0.85, cohen_kappa=0.75, macro_f1=0.80),
        ]
        agg = aggregate_folds(folds)
        assert agg.backend == "ml"
        assert agg.n_folds == 2
        assert abs(agg.accuracy_mean - 0.825) < 1e-6
        assert abs(agg.kappa_mean - 0.725) < 1e-6

    def test_empty_folds(self):
        agg = aggregate_folds([])
        assert agg.backend == "unknown"
        assert agg.n_folds == 0

    def test_per_stage_mean(self):
        folds = [
            FoldResult(fold_index=0, backend="ml",
                       per_stage={"W": {"f1": 0.8}, "N1": {"f1": 0.3}}),
            FoldResult(fold_index=1, backend="ml",
                       per_stage={"W": {"f1": 0.9}, "N1": {"f1": 0.4}}),
        ]
        agg = aggregate_folds(folds)
        assert abs(agg.per_stage_mean["W"]["f1"] - 0.85) < 1e-6
        assert abs(agg.per_stage_mean["N1"]["f1"] - 0.35) < 1e-6

    def test_clinical_metrics_aggregation(self):
        folds = [
            FoldResult(fold_index=0, backend="ml",
                       clinical_metrics={"total_sleep_time_min": 400}),
            FoldResult(fold_index=1, backend="ml",
                       clinical_metrics={"total_sleep_time_min": 420}),
        ]
        agg = aggregate_folds(folds)
        assert abs(agg.clinical_metrics_mean["total_sleep_time_min"] - 410) < 1e-6

    def test_to_dict(self):
        folds = [FoldResult(fold_index=0, backend="ml", accuracy=0.8,
                            cohen_kappa=0.7, macro_f1=0.75)]
        agg = aggregate_folds(folds)
        d = agg.to_dict()
        assert "accuracy" in d
        assert "mean" in d["accuracy"]
        assert "folds" in d


class TestFoldResult:
    def test_to_dict(self):
        fr = FoldResult(fold_index=0, backend="rules", accuracy=0.75,
                        cohen_kappa=0.65, macro_f1=0.70)
        d = fr.to_dict()
        assert d["accuracy"] == 0.75
        assert d["backend"] == "rules"

    def test_confusion_matrix_serialisation(self):
        fr = FoldResult(fold_index=0, backend="ml",
                        confusion_matrix=np.eye(5, dtype=int))
        d = fr.to_dict()
        assert d["confusion_matrix"] == np.eye(5, dtype=int).tolist()


# =====================================================================
# Inter-Scorer
# =====================================================================

from CESA.sleep_pipeline.benchmark.inter_scorer import (
    _cohens_kappa,
    _accuracy,
    _fleiss_kappa,
    compute_pairwise_kappa,
    disagreement_timeline,
    critical_confusion_analysis,
    compute_inter_scorer_report,
    STAGES,
)


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels = ["W", "N1", "N2", "N3", "R"] * 20
        k = _cohens_kappa(labels, labels, STAGES)
        assert abs(k - 1.0) < 1e-10

    def test_no_agreement(self):
        a = ["W"] * 100
        b = ["R"] * 100
        k = _cohens_kappa(a, b, STAGES)
        assert k <= 0

    def test_moderate_agreement(self):
        np.random.seed(42)
        a = np.random.choice(STAGES, 200).tolist()
        b = a.copy()
        for i in range(40):
            b[i] = np.random.choice(STAGES)
        k = _cohens_kappa(a, b, STAGES)
        assert 0.3 < k < 1.0

    def test_empty(self):
        assert _cohens_kappa([], [], STAGES) == 0.0


class TestAccuracy:
    def test_perfect(self):
        a = ["W", "N1", "N2"]
        assert _accuracy(a, a) == 1.0

    def test_half(self):
        a = ["W", "N1", "N2", "N3"]
        b = ["W", "N1", "R", "W"]
        assert _accuracy(a, b) == 0.5


class TestFleissKappa:
    def test_perfect_agreement(self):
        scorers = {
            "A": ["W", "N1", "N2", "N3", "R"] * 10,
            "B": ["W", "N1", "N2", "N3", "R"] * 10,
            "C": ["W", "N1", "N2", "N3", "R"] * 10,
        }
        k = _fleiss_kappa(scorers, STAGES)
        assert abs(k - 1.0) < 1e-6

    def test_single_rater(self):
        assert _fleiss_kappa({"A": ["W"]}, STAGES) == 0.0


class TestDisagreementTimeline:
    def test_identifies_disagreements(self):
        a = ["W", "W", "N1", "N2"]
        b = ["W", "N1", "N1", "N2"]
        events = disagreement_timeline(a, b, epoch_duration_s=30.0)
        assert len(events) == 1
        assert events[0].epoch_index == 1
        assert events[0].time_s == 30.0

    def test_no_disagreements(self):
        a = ["W", "N1"]
        events = disagreement_timeline(a, a)
        assert len(events) == 0


class TestCriticalConfusionAnalysis:
    def test_detects_n1_wake_confusion(self):
        ref = ["N1", "N1", "N1", "W", "W"]
        pred = ["W", "N1", "N1", "W", "W"]
        results = critical_confusion_analysis(ref, pred)
        n1_w = next(c for c in results if c.true_stage == "N1" and c.pred_stage == "W")
        assert n1_w.count == 1
        assert abs(n1_w.rate - 1 / 3) < 1e-6

    def test_no_confusions(self):
        ref = ["W", "N1", "N2"]
        results = critical_confusion_analysis(ref, ref)
        assert all(c.count == 0 for c in results)


class TestInterScorerReport:
    def test_full_report(self):
        scorers = {
            "human": ["W", "N1", "N2", "N3", "R"] * 20,
            "cesa_ml": ["W", "N1", "N2", "N3", "R"] * 20,
            "cesa_hmm": ["W", "N1", "N2", "N3", "R"] * 20,
        }
        report = compute_inter_scorer_report(scorers)
        assert len(report.scorer_names) == 3
        assert report.fleiss_kappa is not None
        assert abs(report.fleiss_kappa - 1.0) < 1e-6
        assert report.disagreement_rate == 0.0

    def test_to_dict(self):
        scorers = {
            "a": ["W", "N1"],
            "b": ["W", "N2"],
        }
        report = compute_inter_scorer_report(scorers)
        d = report.to_dict()
        assert "pairwise_kappa" in d
        assert "disagreement_rate" in d

    def test_kappa_matrix(self):
        scorers = {
            "a": ["W", "N1", "N2"] * 10,
            "b": ["W", "N1", "N2"] * 10,
        }
        report = compute_inter_scorer_report(scorers)
        names, mat = report.kappa_matrix_as_array()
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 1.0  # self-agreement
        assert mat[0, 1] == mat[1, 0]


# =====================================================================
# Figures
# =====================================================================

from CESA.sleep_pipeline.benchmark.figures import (
    plot_hypnogram_comparison,
    plot_confusion_matrix,
    plot_stage_distribution,
    plot_error_timeline,
    plot_cv_boxplot,
    plot_kappa_heatmap,
    plot_feature_importance,
    plot_clinical_metrics_table,
    plot_ml_probabilities,
    plot_eeg_excerpts,
    plot_psd_by_stage,
)


class TestFigures:
    """Smoke tests: each figure function should return a Figure and
    save files without error."""

    @pytest.fixture(autouse=True)
    def _close_figs(self):
        import matplotlib
        matplotlib.use("Agg")
        yield
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_hypnogram_comparison(self, tmp_path):
        ref = (["W"] * 10 + ["N1"] * 10 + ["N2"] * 20 + ["N3"] * 20 + ["R"] * 10)
        preds = {"ml": ref[::-1], "hmm": ref}
        fig = plot_hypnogram_comparison(ref, preds, output_path=tmp_path / "hyp", fmt="png")
        assert fig is not None
        assert (tmp_path / "hyp.png").is_file()

    def test_confusion_matrix(self, tmp_path):
        cm = np.array([[80, 5, 3, 1, 1],
                       [10, 30, 5, 0, 5],
                       [2, 3, 85, 5, 0],
                       [0, 0, 10, 70, 0],
                       [5, 8, 0, 0, 60]])
        fig = plot_confusion_matrix(cm, output_path=tmp_path / "cm", fmt="png")
        assert fig is not None
        assert (tmp_path / "cm.png").is_file()

    def test_stage_distribution(self, tmp_path):
        ref = ["W"] * 20 + ["N2"] * 40 + ["R"] * 10
        preds = {"ml": ["W"] * 15 + ["N2"] * 35 + ["N1"] * 10 + ["R"] * 10}
        fig = plot_stage_distribution(ref, preds, output_path=tmp_path / "dist", fmt="png")
        assert fig is not None

    def test_error_timeline(self, tmp_path):
        ref = ["W", "N1", "N2", "N3", "R"] * 20
        pred = ["W", "N1", "N2", "N3", "N1"] * 20
        fig = plot_error_timeline(ref, pred, output_path=tmp_path / "err", fmt="png")
        assert fig is not None

    def test_cv_boxplot(self, tmp_path):
        data = {
            "rules": {"accuracy": [0.7, 0.72], "kappa": [0.6, 0.62], "f1": [0.65, 0.67]},
            "ml": {"accuracy": [0.8, 0.82], "kappa": [0.75, 0.77], "f1": [0.78, 0.80]},
        }
        fig = plot_cv_boxplot(data, output_path=tmp_path / "cv", fmt="png")
        assert fig is not None

    def test_kappa_heatmap(self, tmp_path):
        names = ["human", "ml", "hmm"]
        mat = np.array([[1.0, 0.8, 0.82], [0.8, 1.0, 0.9], [0.82, 0.9, 1.0]])
        fig = plot_kappa_heatmap(names, mat, output_path=tmp_path / "kh", fmt="png")
        assert fig is not None

    def test_feature_importance(self, tmp_path):
        imp = {f"feat_{i}": np.random.rand() for i in range(25)}
        fig = plot_feature_importance(imp, top_n=10, output_path=tmp_path / "fi", fmt="png")
        assert fig is not None

    def test_clinical_metrics_table(self, tmp_path):
        data = {
            "Reference": {"total_sleep_time_min": 420, "sleep_efficiency_pct": 85},
            "ML": {"total_sleep_time_min": 415, "sleep_efficiency_pct": 83},
        }
        fig = plot_clinical_metrics_table(data, output_path=tmp_path / "cm_tbl", fmt="png")
        assert fig is not None

    def test_ml_probabilities(self, tmp_path):
        n = 50
        probs = np.random.dirichlet([1, 1, 1, 1, 1], size=n)
        ref = ["W"] * 10 + ["N1"] * 10 + ["N2"] * 10 + ["N3"] * 10 + ["R"] * 10
        fig = plot_ml_probabilities(probs, ref, output_path=tmp_path / "mlp", fmt="png")
        assert fig is not None

    def test_eeg_excerpts(self, tmp_path):
        signal = np.random.randn(30000)
        epochs_info = [
            {"epoch_index": 0, "true_stage": "W", "pred_stage": "W",
             "label": "Correct W", "epoch_duration_s": 30.0},
            {"epoch_index": 5, "true_stage": "N2", "pred_stage": "N1",
             "label": "N2 -> N1", "epoch_duration_s": 30.0},
        ]
        fig = plot_eeg_excerpts(signal, 100.0, epochs_info,
                                output_path=tmp_path / "eeg", fmt="png")
        assert fig is not None

    def test_psd_by_stage(self, tmp_path):
        freqs = np.linspace(0, 50, 100)
        psds = {
            "W": (freqs, np.random.rand(20, 100) + 1),
            "N3": (freqs, np.random.rand(20, 100) * 5 + 1),
        }
        fig = plot_psd_by_stage(psds, output_path=tmp_path / "psd", fmt="png")
        assert fig is not None


# =====================================================================
# Experiment Result
# =====================================================================

from CESA.sleep_pipeline.benchmark.experiment import ExperimentResult


class TestExperimentResult:
    def test_to_dict(self):
        r = ExperimentResult(
            n_records=10,
            n_subjects=5,
            elapsed_seconds=42.0,
        )
        d = r.to_dict()
        assert d["n_records"] == 10
        assert d["n_subjects"] == 5
        assert "config" in d
        assert "per_backend" in d

    def test_with_backends(self):
        agg = AggregatedResults(
            backend="ml", n_folds=3,
            accuracy_mean=0.82, kappa_mean=0.75,
        )
        r = ExperimentResult(per_backend={"ml": agg})
        d = r.to_dict()
        assert "ml" in d["per_backend"]


# =====================================================================
# Paper Report
# =====================================================================

from CESA.sleep_pipeline.benchmark.paper_report import generate_paper_report


class TestPaperReport:
    def test_generates_markdown(self, tmp_path):
        agg = AggregatedResults(
            backend="ml", n_folds=3,
            accuracy_mean=0.82, accuracy_std=0.02,
            kappa_mean=0.75, kappa_std=0.03,
            macro_f1_mean=0.78, macro_f1_std=0.02,
            accuracy_ci=(0.79, 0.85),
            kappa_ci=(0.71, 0.79),
            per_stage_mean={"W": {"f1": 0.9, "precision": 0.88, "recall": 0.92}},
            clinical_metrics_mean={"total_sleep_time_min": 400},
            clinical_metrics_std={"total_sleep_time_min": 20},
        )
        r = ExperimentResult(
            n_records=20, n_subjects=10,
            per_backend={"ml": agg},
        )
        path = generate_paper_report(r, tmp_path)
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert "## Abstract" in content
        assert "## 2. Methods" in content
        assert "## 3. Results" in content
        assert "## 4. Discussion" in content
        assert "kappa" in content.lower()

    def test_handles_empty_result(self, tmp_path):
        r = ExperimentResult()
        path = generate_paper_report(r, tmp_path)
        assert path.is_file()


# =====================================================================
# CLI (scripts/run_benchmark.py)
# =====================================================================

class TestCLI:
    def test_parser_builds(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.run_benchmark import build_parser
        parser = build_parser()
        args = parser.parse_args(["--config", "test.yaml"])
        assert args.config == "test.yaml"

    def test_parser_overrides(self):
        from scripts.run_benchmark import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--config", "test.yaml",
            "--n-subjects", "5",
            "--n-folds", "3",
            "--backends", "ml", "ml_hmm",
            "--seed", "99",
        ])
        assert args.n_subjects == 5
        assert args.n_folds == 3
        assert args.backends == ["ml", "ml_hmm"]
        assert args.seed == 99
