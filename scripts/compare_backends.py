#!/usr/bin/env python3
"""Compare scoring backends: AASM rules vs ML vs ML+HMM.

Runs all available backends on the same recording and produces a
comparative report showing the incremental improvement of each layer.

Usage
-----
::

    python scripts/compare_backends.py \\
        --edf data/subject.edf \\
        --reference data/subject_scoring.csv \\
        --ml-model models/sleep_ml.joblib \\
        --output reports/comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_reference(path: str) -> "ScoringResult":
    from CESA.sleep_pipeline.contracts import ScoringResult
    df = pd.read_csv(path)
    return ScoringResult.from_dataframe(df, backend="reference")


def run_comparison(
    edf_path: str,
    reference_path: str,
    ml_model_path: str | None = None,
    epoch_duration_s: float = 30.0,
    target_sfreq: float = 100.0,
    output_path: str | None = None,
) -> Dict[str, Any]:
    """Run all backends and compare against a reference.

    Returns a structured comparison dict.
    """
    import mne
    from CESA.sleep_pipeline.preprocessing import PreprocessingConfig, preprocess
    from CESA.sleep_pipeline.features import extract_all_features
    from CESA.sleep_pipeline.rules_aasm import score_rule_based
    from CESA.sleep_pipeline.evaluation import compare, compute_clinical_metrics
    from CESA.sleep_pipeline.sequence_model import SleepHMM, hmm_decode_scoring

    reference = _load_reference(reference_path)

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    config = PreprocessingConfig(
        target_sfreq=target_sfreq,
        epoch_duration_s=epoch_duration_s,
    )
    epoched = preprocess(raw, config)
    feature_list = extract_all_features(epoched)

    results: Dict[str, Any] = {"backends": {}}

    # --- Backend 1: AASM rules ---
    logger.info("Running AASM rules backend...")
    rules_result = score_rule_based(feature_list, epoch_duration_s=epoch_duration_s)
    rules_report = compare(reference, rules_result)
    results["backends"]["aasm_rules"] = {
        "accuracy": rules_report.accuracy,
        "cohen_kappa": rules_report.cohen_kappa,
        "macro_f1": rules_report.macro_f1,
        "n_epochs": rules_report.n_epochs,
        "clinical": compute_clinical_metrics(rules_result).to_dict(),
    }
    logger.info("  AASM rules: acc=%.3f, kappa=%.3f, F1=%.3f",
                rules_report.accuracy, rules_report.cohen_kappa, rules_report.macro_f1)

    # --- Backend 2: ML (if model available) ---
    if ml_model_path and Path(ml_model_path).exists():
        logger.info("Running ML backend...")
        from CESA.sleep_pipeline.ml_scorer import score_ml
        ml_result = score_ml(
            feature_list,
            epoch_duration_s=epoch_duration_s,
            model_path=ml_model_path,
            apply_smoothing=False,
        )
        ml_report = compare(reference, ml_result)
        results["backends"]["ml"] = {
            "accuracy": ml_report.accuracy,
            "cohen_kappa": ml_report.cohen_kappa,
            "macro_f1": ml_report.macro_f1,
            "n_epochs": ml_report.n_epochs,
            "clinical": compute_clinical_metrics(ml_result).to_dict(),
        }
        logger.info("  ML: acc=%.3f, kappa=%.3f, F1=%.3f",
                    ml_report.accuracy, ml_report.cohen_kappa, ml_report.macro_f1)

        # --- Backend 3: ML + HMM ---
        logger.info("Running ML + HMM backend...")
        hmm = SleepHMM()
        ml_hmm_result = hmm_decode_scoring(ml_result, hmm=hmm)
        ml_hmm_report = compare(reference, ml_hmm_result)
        results["backends"]["ml_hmm"] = {
            "accuracy": ml_hmm_report.accuracy,
            "cohen_kappa": ml_hmm_report.cohen_kappa,
            "macro_f1": ml_hmm_report.macro_f1,
            "n_epochs": ml_hmm_report.n_epochs,
            "clinical": compute_clinical_metrics(ml_hmm_result).to_dict(),
        }
        logger.info("  ML+HMM: acc=%.3f, kappa=%.3f, F1=%.3f",
                    ml_hmm_report.accuracy, ml_hmm_report.cohen_kappa, ml_hmm_report.macro_f1)

        # --- Delta improvements ---
        results["deltas"] = {
            "rules_to_ml": {
                "kappa": ml_report.cohen_kappa - rules_report.cohen_kappa,
                "accuracy": ml_report.accuracy - rules_report.accuracy,
                "macro_f1": ml_report.macro_f1 - rules_report.macro_f1,
            },
            "ml_to_ml_hmm": {
                "kappa": ml_hmm_report.cohen_kappa - ml_report.cohen_kappa,
                "accuracy": ml_hmm_report.accuracy - ml_report.accuracy,
                "macro_f1": ml_hmm_report.macro_f1 - ml_report.macro_f1,
            },
        }
    else:
        logger.warning("No ML model provided -- skipping ML and ML+HMM backends.")

    # --- AASM rules + HMM ---
    logger.info("Running AASM rules + HMM backend...")
    hmm_rules = SleepHMM()
    rules_hmm_result = hmm_decode_scoring(rules_result, hmm=hmm_rules)
    rules_hmm_report = compare(reference, rules_hmm_result)
    results["backends"]["rules_hmm"] = {
        "accuracy": rules_hmm_report.accuracy,
        "cohen_kappa": rules_hmm_report.cohen_kappa,
        "macro_f1": rules_hmm_report.macro_f1,
        "n_epochs": rules_hmm_report.n_epochs,
    }
    logger.info("  Rules+HMM: acc=%.3f, kappa=%.3f, F1=%.3f",
                rules_hmm_report.accuracy, rules_hmm_report.cohen_kappa, rules_hmm_report.macro_f1)

    # Reference clinical metrics
    results["reference_clinical"] = compute_clinical_metrics(reference).to_dict()

    # Pretty-print summary
    print("\n" + "=" * 70)
    print("        BACKEND COMPARISON REPORT")
    print("=" * 70)
    for name, data in results["backends"].items():
        print(f"  {name:<15} | acc={data['accuracy']:.3f} | kappa={data['cohen_kappa']:.3f} | F1={data['macro_f1']:.3f}")
    if "deltas" in results:
        print("-" * 70)
        for step, d in results["deltas"].items():
            print(f"  {step:<20} | delta_kappa={d['kappa']:+.3f} | delta_acc={d['accuracy']:+.3f}")
    print("=" * 70)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Report saved to %s", output_path)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare scoring backends")
    parser.add_argument("--edf", required=True, help="EDF recording")
    parser.add_argument("--reference", required=True, help="Reference scoring CSV")
    parser.add_argument("--ml-model", default=None, help="Trained ML model (joblib)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--epoch-duration", type=float, default=30.0)
    args = parser.parse_args()

    run_comparison(
        edf_path=args.edf,
        reference_path=args.reference,
        ml_model_path=args.ml_model,
        epoch_duration_s=args.epoch_duration,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
