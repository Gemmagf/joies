"""Run the offline eval framework and write a report the dashboard can read.

Usage:
    python scripts/run_eval.py              # runs retrieval + groundedness smoke
    python scripts/run_eval.py --quiet      # no console output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.config import get_settings  # noqa: E402
from maison_concierge.eval.runner import run_full_eval, write_eval_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline eval framework.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = run_full_eval()

    output_path = get_settings().data_dir / "eval" / "report.json"
    write_eval_report(report, output_path)

    if args.quiet:
        return

    c = report.catalog_retrieval
    h = report.heritage_retrieval
    print(f"Wrote eval report → {output_path}")
    print()
    print(f"Catalog retrieval (n={c.n_queries}):")
    print(f"  recall@1 = {c.recall_at_1:.2f}")
    print(f"  recall@3 = {c.recall_at_3:.2f}")
    print(f"  recall@5 = {c.recall_at_5:.2f}")
    print(f"  MRR      = {c.mrr:.2f}")
    print()
    print(f"Heritage retrieval (n={h.n_queries}):")
    print(f"  recall@1 = {h.recall_at_1:.2f}")
    print(f"  recall@3 = {h.recall_at_3:.2f}")
    print(f"  recall@5 = {h.recall_at_5:.2f}")
    print(f"  MRR      = {h.mrr:.2f}")
    print()
    print("Groundedness smoke test:")
    for sample in report.groundedness_samples:
        status = "PASS" if sample.is_grounded else f"FAIL ({sample.hallucination_count} issues)"
        print(f"  {status}: {sample.reply[:80]}…")
    if report.calibration is not None:
        print()
        print(
            f"Calibration: ECE={report.calibration.expected_calibration_error:.3f}  "
            f"Brier={report.calibration.brier_score:.3f}  N={report.calibration.n_total}"
        )


if __name__ == "__main__":
    main()
