"""Profile agent + end-to-end orchestrator."""

from __future__ import annotations

import pytest

from maison_concierge.analytics import load_all_artifacts
from maison_concierge.hospitality import (
    HospitalityIntent,
    HospitalityOrchestrator,
    load_personas,
)
from maison_concierge.hospitality.profile import ProfileAgent


@pytest.fixture(scope="module")
def profile_agent() -> ProfileAgent:
    artifacts = load_all_artifacts()
    if artifacts is None:
        pytest.skip("Analytics artifacts missing — run scripts/train_analytics.py.")
    return ProfileAgent(artifacts=artifacts)


@pytest.fixture(scope="module")
def personas():
    p = load_personas()
    if not p:
        pytest.skip("No personas — run scripts/generate_personas.py.")
    return p


def test_profile_snapshot_shape(profile_agent: ProfileAgent, personas) -> None:
    snap = profile_agent.snapshot(personas[0].booking, persona_id=personas[0].id)
    assert 0.0 <= snap.p_cancel <= 1.0
    assert snap.churn_band in {"low", "medium", "high"}
    assert snap.value_band in {"Bronze", "Silver", "Gold", "Platinum"}
    assert snap.segment
    assert snap.top_drivers
    assert snap.notes


def test_snapshots_vary_across_personas(profile_agent: ProfileAgent, personas) -> None:
    p_cancels = [
        profile_agent.snapshot(p.booking, persona_id=p.id).p_cancel for p in personas
    ]
    assert min(p_cancels) < max(p_cancels), "all snapshots identical — bug in features?"


def test_orchestrator_routes_spa_to_knowledge(personas) -> None:
    orch = HospitalityOrchestrator()
    result = orch.run(
        conversation_id="t1",
        user_message="What time does the spa open?",
        persona_id=personas[0].id,
    )
    assert result["intent"] == HospitalityIntent.KNOWLEDGE
    assert result["kb_hits"], "knowledge agent should have hits"
    assert result["kb_hits"][0].page.id == "kb-spa-wellness"
    assert "spa" in result["assistant_reply"].lower()
    assert "kb-spa-wellness" in result["citations"]


def test_orchestrator_routes_profile_query_and_calls_analytics(personas) -> None:
    orch = HospitalityOrchestrator()
    result = orch.run(
        conversation_id="t2",
        user_message="Am I likely to cancel my booking?",
        persona_id=personas[0].id,
    )
    assert result["intent"] == HospitalityIntent.PROFILE
    assert result["profile_snapshot"] is not None
    assert 0.0 <= result["profile_snapshot"].p_cancel <= 1.0
    assert not result["kb_hits"], "knowledge should not fire on profile intent"


def test_orchestrator_escalates_on_complaint(personas) -> None:
    orch = HospitalityOrchestrator()
    result = orch.run(
        conversation_id="t3",
        user_message="This is unacceptable, I want to complain to a manager.",
        persona_id=personas[0].id,
    )
    assert result["intent"] == HospitalityIntent.ESCALATE
    assert result["escalate"] is True
    assert "human colleague" in result["assistant_reply"].lower()


def test_knowledge_scopes_to_persona_property(personas) -> None:
    lisboa_persona = next(p for p in personas if "Lisboa" in p.property)
    orch = HospitalityOrchestrator()
    result = orch.run(
        conversation_id="t4",
        user_message="what time is check-in and dress code at the restaurant?",
        persona_id=lisboa_persona.id,
    )
    trace = result["trace"]
    assert trace["knowledge"]["property_scope"] == "lisboa"
    # None of the returned hits should be algarve-exclusive.
    for h in result["kb_hits"]:
        assert h.page.property_scope in {"lisboa", "both"}
