#!/usr/bin/env python3
"""Generate a comparison report between two scoring files.

Usage
-----
::

    python scripts/generate_comparison_report.py \\
        --reference scoring_manual.csv \\
        --predicted scoring_auto.csv \\
        --output report.json

Both CSV files must contain ``time`` and ``stage`` columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CESA.sleep_pipeline.contracts import ScoringResult
from CESA.sleep_pipeline.evaluation import compare

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_csv(path: str) -> ScoringResult:
    df = pd.read_csv(path)
    if "time" not in df.columns or "stage" not in df.columns:
        raise ValueError(f"File {path} must have 'time' and 'stage' columns")
    return ScoringResult.from_dataframe(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two sleep scoring files")
    parser.add_argument("--reference", required=True, help="Reference scoring CSV")
    parser.add_argument("--predicted", required=True, help="Predicted scoring CSV")
    parser.add_argument("--output", default=None, help="Output JSON report path (optional)")
    parser.add_argument("--include-u", action="store_true", help="Include U epochs in comparison")
    args = parser.parse_args()

    ref = _load_csv(args.reference)
    pred = _load_csv(args.predicted)

    report = compare(ref, pred, exclude_u=not args.include_u)

    print("\n" + "=" * 60)
    print("   SLEEP SCORING COMPARISON REPORT")
    print("=" * 60)
    print(report.summary_text())
    print("=" * 60)

    if report.confusion_matrix is not None:
        print("\nConfusion Matrix (rows=reference, cols=predicted):")
        print(report.confusion_matrix.to_string())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Report saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
