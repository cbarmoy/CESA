#!/usr/bin/env python
"""CLI entry point for the CESA benchmark experiment pipeline.

Usage
-----
    python scripts/run_benchmark.py --config config/benchmark_sleepedf.yaml
    python scripts/run_benchmark.py --config config/benchmark_mass.yaml --n-subjects 20
    python scripts/run_benchmark.py --config config/benchmark_sleepedf.yaml --backends ml ml_hmm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``CESA`` is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from CESA.sleep_pipeline.benchmark.config import BenchmarkConfig
from CESA.sleep_pipeline.benchmark.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a CESA sleep-staging benchmark experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML or JSON configuration file.",
    )
    p.add_argument(
        "--data-path", type=str, default=None,
        help="Override data_path in the config (e.g. path to MASS data).",
    )
    p.add_argument(
        "--n-subjects", type=int, default=None,
        help="Override n_subjects (-1 = all).",
    )
    p.add_argument(
        "--n-folds", type=int, default=None,
        help="Override n_folds.",
    )
    p.add_argument(
        "--backends", nargs="+", default=None,
        help="Override backends (e.g. --backends ml ml_hmm).",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = BenchmarkConfig.load(args.config)

    # Apply CLI overrides
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.n_subjects is not None:
        config.n_subjects = args.n_subjects
    if args.n_folds is not None:
        config.n_folds = args.n_folds
    if args.backends is not None:
        config.backends = args.backends
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.random_seed = args.seed

    logging.getLogger(__name__).info("Config: %s", config)

    # Run
    result = run_experiment(config)

    # Summary
    print("\n" + "=" * 60)
    print("CESA BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset:    {config.dataset}")
    print(f"Subjects:   {result.n_subjects}")
    print(f"Records:    {result.n_records}")
    print(f"Folds:      {config.n_folds}")
    print(f"Duration:   {result.elapsed_seconds:.1f}s")
    print("-" * 60)

    for bk, agg in result.per_backend.items():
        print(f"\n  {bk}:")
        print(f"    Accuracy:  {agg.accuracy_mean:.3f} ± {agg.accuracy_std:.3f} "
              f"[{agg.accuracy_ci[0]:.3f}, {agg.accuracy_ci[1]:.3f}]")
        print(f"    Kappa:     {agg.kappa_mean:.3f} ± {agg.kappa_std:.3f} "
              f"[{agg.kappa_ci[0]:.3f}, {agg.kappa_ci[1]:.3f}]")
        print(f"    Macro-F1:  {agg.macro_f1_mean:.3f} ± {agg.macro_f1_std:.3f}")

    print(f"\nOutput: {result.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
