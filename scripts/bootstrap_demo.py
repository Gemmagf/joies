"""Bootstrap the hospitality demo — idempotent, safe to run at cold start.

Streamlit Cloud clones the repo and runs the entry point; the analytics
model artifacts and the property-KB Chroma index are not committed (both
are regeneratable and would bloat the repo). This script fills them in
on first boot and no-ops on subsequent boots.

Steps:
  1. Train analytics artifacts if `data/models/churn_lgbm.txt` is missing
     (or `--force`).
  2. Generate synthetic personas if the JSON is missing.
  3. Warm the Property KB Chroma index (creates it if needed).

Run locally:
    python scripts/bootstrap_demo.py
    python scripts/bootstrap_demo.py --force   # re-do everything
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _train_analytics_if_needed(*, force: bool) -> bool:
    from maison_concierge.config import get_settings

    marker = get_settings().models_dir / "churn_lgbm.txt"
    if marker.exists() and not force:
        print(f"[bootstrap] analytics artifacts present at {marker} — skipping")
        return False
    print("[bootstrap] training analytics artifacts…", flush=True)
    from maison_concierge.analytics import (
        fit_clv_bander,
        fit_guest_segmenter,
        load_bookings,
        save_all_artifacts,
        split_chronological,
        train_churn_model,
    )

    bookings = load_bookings()
    X_train, y_train, X_test, y_test, cutoff = split_chronological(bookings)
    churn = train_churn_model(X_train, y_train, X_test, y_test, cutoff_date=cutoff)
    clv = fit_clv_bander(bookings.df.iloc[: len(X_train)])
    seg = fit_guest_segmenter(bookings.df.iloc[: len(X_train)])
    save_all_artifacts(churn=churn, clv=clv, segmenter=seg)
    print(f"[bootstrap] churn AUROC={churn.report.auroc:.3f}")
    return True


def _generate_personas_if_needed(*, force: bool) -> bool:
    from maison_concierge.hospitality.personas import personas_path

    if personas_path().exists() and not force:
        print(f"[bootstrap] personas present at {personas_path()} — skipping")
        return False
    print("[bootstrap] generating synthetic personas…", flush=True)
    import runpy

    runpy.run_path(
        str(REPO_ROOT / "scripts" / "generate_personas.py"),
        run_name="__main__",
    )
    return True


def _warm_kb(*, force: bool) -> int:
    print("[bootstrap] indexing property KB…", flush=True)
    from maison_concierge.hospitality.kb import PropertyKB

    kb = PropertyKB()
    n = kb.index(force=force)
    print(f"[bootstrap] KB has {n} pages")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate everything.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    _train_analytics_if_needed(force=args.force)
    _generate_personas_if_needed(force=args.force)
    _warm_kb(force=args.force)
    print(f"[bootstrap] done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
