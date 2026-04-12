"""CESA benchmark subpackage for reproducible scientific validation.

Provides dataset loaders, subject-level cross-validation, inter-scorer
agreement analysis, publication-quality figure generation, and a full
experiment orchestrator producing paper-ready results.
"""

__all__ = [
    "BenchmarkConfig",
    "DatasetRecord",
    "ExperimentResult",
    "run_experiment",
]
