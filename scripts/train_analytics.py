"""Train churn / CLV / segmentation models and write artifacts + metrics.

Run from repo root:

    python scripts/train_analytics.py

Writes:
    data/models/churn_lgbm.txt
    data/models/clv_bander.joblib
    data/models/guest_segmenter.joblib
    data/metrics/analytics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maison_concierge.analytics import (  # noqa: E402
    fit_clv_bander,
    fit_guest_segmenter,
    load_bookings,
    save_all_artifacts,
    split_chronological,
    train_churn_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Chronological test-set fraction (default: 0.2).",
    )
    parser.add_argument(
        "--bookings",
        type=Path,
        default=None,
        help="Override path to hotel_bookings.csv[.gz]. Defaults to BOOKINGS_PATH.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("[analytics] loading bookings...", flush=True)
    bookings = load_bookings(args.bookings)
    print(
        f"[analytics] loaded {bookings.n_rows:,} rows, "
        f"overall cancellation rate = {bookings.cancellation_rate:.3f}",
        flush=True,
    )

    print("[analytics] chronological split + churn (LightGBM)...", flush=True)
    X_train, y_train, X_test, y_test, cutoff = split_chronological(
        bookings, test_fraction=args.test_fraction
    )
    churn = train_churn_model(
        X_train, y_train, X_test, y_test, cutoff_date=cutoff
    )
    print(
        f"[analytics] churn AUROC={churn.report.auroc:.3f} "
        f"AP={churn.report.average_precision:.3f} "
        f"Brier={churn.report.brier:.3f} "
        f"best_iter={churn.report.best_iteration}",
        flush=True,
    )
    print("[analytics] top churn drivers:", flush=True)
    for name, gain in churn.top_drivers(5):
        print(f"    {name:32s} {gain:.3f}", flush=True)

    print("[analytics] fitting CLV bander on training split...", flush=True)
    clv = fit_clv_bander(bookings.df.iloc[: len(X_train)])
    print(
        f"[analytics] CLV cutoffs (25/50/75) = "
        f"{[round(c, 1) for c in clv.cutoffs]}",
        flush=True,
    )

    print("[analytics] fitting guest segmenter (KMeans)...", flush=True)
    segmenter = fit_guest_segmenter(bookings.df.iloc[: len(X_train)])
    print(
        f"[analytics] segments: {json.dumps(segmenter.report.sizes)}",
        flush=True,
    )

    metrics_path = save_all_artifacts(churn=churn, clv=clv, segmenter=segmenter)
    print(
        f"[analytics] wrote {metrics_path} in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
