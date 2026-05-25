"""Run the conversation simulator and write a report the dashboard can read.

Usage:
    python scripts/run_simulator.py                # 100 turns, demo mode
    python scripts/run_simulator.py --n 200        # bigger run
    python scripts/run_simulator.py --seed 7       # reproducible alternate seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.eval.simulator import (  # noqa: E402
    default_report_path,
    run_simulation,
    write_simulation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test the orchestrator with synthetic conversations.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running {args.n} synthetic conversations through the orchestrator…")
    report, turns = run_simulation(n_turns=args.n, seed=args.seed)
    path = default_report_path()
    write_simulation_report(report, turns, path)

    print(f"Wrote report → {path}")
    print()
    print("HEADLINE")
    print(f"  n turns                       {report.n_turns}")
    print(f"  intent accuracy               {report.intent_accuracy:.2%}")
    print(f"  groundedness rate             {report.grounded_rate:.2%}")
    print(f"  escalation rate               {report.escalation_rate:.2%}")
    print(f"  avg quality score (0-1)       {report.avg_quality_score:.3f}")
    print(f"  avg intent confidence         {report.avg_intent_confidence:.2f}")
    print()
    print("PER INTENT")
    for intent, m in sorted(report.per_intent.items()):
        print(
            f"  {intent:<20} n={m['n']:>3}  "
            f"acc={m['intent_accuracy']:.2%}  grounded={m['grounded_rate']:.2%}  "
            f"quality={m['avg_quality']:.2f}"
        )


if __name__ == "__main__":
    main()
