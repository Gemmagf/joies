"""Generate synthetic calibration pairs so the dashboard has a reliability diagram.

These are the (confidence, correct) pairs the intent classifier would produce on the
golden set. For the demo we synthesise a slightly miscalibrated distribution — that's
realistic for an out-of-the-box LLM classifier and makes the chart interesting.

Usage:
    python scripts/seed_eval_calibration.py [--n 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.config import get_settings  # noqa: E402


def _draw_pair(rng: random.Random) -> tuple[float, bool]:
    bin_idx = rng.randint(0, 9)
    bin_lower = bin_idx / 10
    confidence = round(rng.uniform(bin_lower, bin_lower + 0.1), 3)
    # Slight overconfidence at high bins (typical of LLM intent classifiers)
    true_accuracy = max(0.0, min(1.0, confidence - 0.08 if confidence > 0.75 else confidence + 0.02))
    correct = rng.random() < true_accuracy
    return confidence, correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed calibration pairs.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pairs = [_draw_pair(rng) for _ in range(args.n)]

    path = get_settings().data_dir / "eval" / "calibration_pairs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for conf, correct in pairs:
            f.write(json.dumps({"confidence": conf, "correct": correct}) + "\n")

    print(f"Wrote {len(pairs)} pairs → {path}")


if __name__ == "__main__":
    main()
