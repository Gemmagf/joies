from datetime import UTC, datetime, timedelta

import pandas as pd

from maison_concierge.models import ClientProfile, ClientSegment, Collection
from maison_concierge.segmentation.booking_model import (
    ALL_FEATURES,
    train_booking_model,
)
from maison_concierge.segmentation.segment import segment_context_for_prompt


def _make_profile(segment: ClientSegment, spend: float) -> ClientProfile:
    return ClientProfile(
        id="CL-TEST-0001",
        display_name="Test Client",
        segment=segment,
        tenure_years=4.0,
        lifetime_spend_chf=spend,
        collections_owned=[Collection.ALHAMBRA, Collection.FRIVOLE],
        preferred_language="en",
        city="Geneva",
        last_appointment_at=datetime.now(UTC) - timedelta(days=45),
    )


def test_segment_context_mentions_segment_and_collections():
    p = _make_profile(ClientSegment.VIP, 200_000)
    block = segment_context_for_prompt(p, locale="en")
    assert "segment: vip" in block
    assert "alhambra" in block
    assert "guidance:" in block
    assert "</segment_context>" in block


def test_segment_context_supports_french():
    p = _make_profile(ClientSegment.COLLECTOR, 800_000)
    block_fr = segment_context_for_prompt(p, locale="fr")
    assert "Stratégie" in block_fr or "patrimoine" in block_fr


def test_days_since_last_appointment_is_positive():
    p = _make_profile(ClientSegment.ESTABLISHED, 30_000)
    days = p.days_since_last_appointment(now=datetime.now(UTC))
    assert days is not None and 40 <= days <= 50


def _synthetic_training_frame(n: int = 200) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    segments = rng.choice(["prospect", "established", "vip", "collector"], size=n)
    intents = rng.choice(
        ["gift", "celebration", "appointment", "browse", "heritage_inquiry"], size=n
    )
    n_turns = rng.integers(1, 12, size=n)
    confidence = rng.beta(8, 2, size=n)
    spend = rng.uniform(0, 500_000, size=n)
    tenure = rng.uniform(0, 25, size=n)
    days_since = rng.integers(5, 720, size=n)

    booking_logits = (
        (segments == "vip") * 0.8
        + (segments == "collector") * 1.2
        + (intents == "appointment") * 0.7
        + 0.1 * n_turns
        - 1.0
        + rng.normal(0, 0.3, size=n)
    )
    booked = (booking_logits > 0).astype(int)
    return pd.DataFrame(
        {
            "segment": segments,
            "intent": intents,
            "tenure_years": tenure,
            "lifetime_spend_chf": spend,
            "preferred_language": "en",
            "days_since_last_appointment": days_since,
            "n_turns": n_turns,
            "intent_confidence": confidence,
            "booked": booked,
        }
    )


def test_booking_model_trains_and_predicts_above_chance():
    df = _synthetic_training_frame(n=400)
    model = train_booking_model(df)
    assert model.report.roc_auc >= 0.6
    sample = df.iloc[:10].copy()
    import numpy as np
    sample["log_lifetime_spend_chf"] = np.log1p(sample["lifetime_spend_chf"])
    probs = model.predict_proba(sample[ALL_FEATURES])
    assert probs.shape == (10,)
    assert ((probs >= 0) & (probs <= 1)).all()
