"""Run the hospitality eval harness and print / write the report.

Usage:
    python scripts/run_hospitality_eval.py [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maison_concierge.eval.hospitality_runner import run_harness, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print("[eval] running hospitality harness…", flush=True)
    report = run_harness()
    json_path, md_path = write_report(report, out_dir=args.out)

    print(f"\nCases          : {report.n_cases}")
    print(f"Task success   : {report.task_success:.1%} ({report.n_passed}/{report.n_cases})")
    print(f"Routing        : {report.routing_accuracy:.1%}")
    print(f"Groundedness   : {report.groundedness:.1%}")
    print(f"Guardrail      : {report.guardrail_catch:.1%}")
    print(f"\nReport: {md_path}")
    print(f"JSON:   {json_path}")

    fails = [r for r in report.results if not r.passed]
    if fails:
        print("\nFailures:")
        for r in fails:
            print(f"  {r.case_id} ({r.category}, {r.actual_intent})")
            for f in r.failures:
                print(f"    - {f}")

    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
