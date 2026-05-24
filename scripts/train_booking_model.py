"""Generate synthetic (conversation, booking) training data and fit the booking model.

The synthetic labelling is rule-based with structured noise: VIP and collector
segments book more often, longer conversations book more often, low intent confidence
hurts. The labelling is hidden from the model — only feature columns are exposed.

Usage:
    python scripts/train_booking_model.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.config import get_settings  # noqa: E402
from maison_concierge.models import ClientIntent, ClientSegment  # noqa: E402
from maison_concierge.segmentation.booking_model import (  # noqa: E402
    save_booking_model,
    train_booking_model,
)
from maison_concierge.segmentation.profiles import load_profiles  # noqa: E402

_SEGMENT_PRIOR = {
    ClientSegment.PROSPECT: 0.08,
    ClientSegment.ESTABLISHED: 0.25,
    ClientSegment.VIP: 0.55,
    ClientSegment.COLLECTOR: 0.72,
}

_INTENT_PRIOR = {
    ClientIntent.APPOINTMENT: 0.55,
    ClientIntent.CELEBRATION: 0.40,
    ClientIntent.INVESTMENT_PIECE: 0.45,
    ClientIntent.GIFT: 0.25,
    ClientIntent.PRICE_INQUIRY: 0.20,
    ClientIntent.BROWSE: 0.10,
    ClientIntent.HERITAGE_INQUIRY: 0.05,
    ClientIntent.UNKNOWN: 0.02,
}


def _booking_probability(
    segment: ClientSegment,
    intent: ClientIntent,
    n_turns: int,
    intent_confidence: float,
    days_since_last_appointment: int,
) -> float:
    base = (_SEGMENT_PRIOR[segment] + _INTENT_PRIOR[intent]) / 2
    turn_lift = min(0.15, 0.025 * max(0, n_turns - 2))
    conf_lift = (intent_confidence - 0.5) * 0.2
    recency_lift = -0.08 if days_since_last_appointment < 30 else 0.0
    return float(np.clip(base + turn_lift + conf_lift + recency_lift, 0.02, 0.97))


def _generate_dataset(n: int, *, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    profiles = load_profiles()
    if not profiles:
        raise SystemExit("No client profiles found. Run scripts/seed_client_profiles.py first.")

    intents = list(_INTENT_PRIOR.keys())
    rows = []
    for _ in range(n):
        profile = rng.choice(profiles)
        intent = rng.choices(intents, weights=[_INTENT_PRIOR[i] + 0.1 for i in intents])[0]
        n_turns = rng.randint(1, 12)
        intent_confidence = round(np_rng.beta(8, 2), 3)
        days_since_last_appt = (
            profile.days_since_last_appointment() or rng.randint(180, 720)
        )
        prob = _booking_probability(
            profile.segment, intent, n_turns, intent_confidence, days_since_last_appt
        )
        booked = int(np_rng.random() < prob)
        rows.append(
            {
                "segment": profile.segment.value,
                "intent": intent.value,
                "tenure_years": profile.tenure_years,
                "lifetime_spend_chf": profile.lifetime_spend_chf,
                "preferred_language": profile.preferred_language,
                "days_since_last_appointment": days_since_last_appt,
                "n_turns": n_turns,
                "intent_confidence": intent_confidence,
                "booked": booked,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    settings = get_settings()
    df = _generate_dataset(n=1_500)
    print(f"Synthetic training set: n={len(df)}, base rate={df['booked'].mean():.2%}")

    model = train_booking_model(df)
    r = model.report
    print()
    print("MODEL PERFORMANCE")
    print("-----------------")
    print(f"  n_train             : {r.n_train}")
    print(f"  n_test              : {r.n_test}")
    print(f"  accuracy            : {r.accuracy:.3f}")
    print(f"  ROC AUC             : {r.roc_auc:.3f}")
    print(f"  Average precision   : {r.average_precision:.3f}")
    print()
    print("PER-SEGMENT")
    print("-----------")
    for seg in sorted(r.per_segment_precision):
        p = r.per_segment_precision[seg]
        rec = r.per_segment_recall.get(seg, 0.0)
        print(f"  {seg:>12}  precision={p:.2f}  recall={rec:.2f}")
    print()
    print("TOP FEATURE COEFFICIENTS")
    print("------------------------")
    top = sorted(r.feature_importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    for name, coef in top:
        sign = "+" if coef >= 0 else "-"
        print(f"  {sign} {abs(coef):.3f}  {name}")
    print()
    print("CONFUSION MATRIX (rows=true, cols=pred)")
    print("                pred_no  pred_yes")
    print(f"  true_no     {r.confusion_matrix[0][0]:>8} {r.confusion_matrix[0][1]:>9}")
    print(f"  true_yes    {r.confusion_matrix[1][0]:>8} {r.confusion_matrix[1][1]:>9}")

    model_path = settings.data_dir / "clients" / "booking_model.joblib"
    report_path = settings.data_dir / "clients" / "booking_model_report.json"
    save_booking_model(model, model_path=model_path, report_path=report_path)
    print()
    print(f"Saved pipeline → {model_path}")
    print(f"Saved report   → {report_path}")


if __name__ == "__main__":
    main()
