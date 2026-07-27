"""LangGraph orchestrator for the hospitality concierge.

Graph shape:

    classify_intent ─┬─► profile ──┐
                     ├─► knowledge ┼─► compose ─► end
                     └─► (booking / recommendation stubs)

The classify node produces the routing flags. Downstream nodes each return
either their result payload or a `skipped` trace entry, so the compose node
sees a uniform state to work from.

Booking and Recommendation are stubbed in Phase 2 (they set a "coming soon"
route flag and the composer replies as such). Phase 4 promotes them to real
nodes with their own tools.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from .composer import compose_templated
from .intent import HospitalityIntent, classify_intent_rule_based
from .kb import PropertyKB
from .personas import Persona, persona_by_id
from .profile import ProfileAgent
from .state import HospitalityState


class HospitalityOrchestrator:
    def __init__(
        self,
        profile: ProfileAgent | None = None,
        kb: PropertyKB | None = None,
    ) -> None:
        self._profile = profile
        self._kb = kb or PropertyKB()
        self._settings = get_settings()
        self._graph = self._build_graph()

    def _get_profile_agent(self) -> ProfileAgent:
        if self._profile is None:
            self._profile = ProfileAgent()
        return self._profile

    def _node_classify(self, state: HospitalityState) -> dict:
        text = state.get("user_message", "")
        result = classify_intent_rule_based(text)
        intent = result.intent
        confidence = result.confidence

        escalate = result.escalate or (
            intent != HospitalityIntent.UNKNOWN
            and confidence < self._settings.app_escalation_threshold
        )
        route_profile = intent == HospitalityIntent.PROFILE and not escalate
        route_knowledge = intent == HospitalityIntent.KNOWLEDGE and not escalate
        route_booking = intent == HospitalityIntent.BOOKING and not escalate
        route_recommendation = (
            intent == HospitalityIntent.RECOMMENDATION and not escalate
        )

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "route_profile": route_profile,
            "route_knowledge": route_knowledge,
            "route_booking": route_booking,
            "route_recommendation": route_recommendation,
            "escalate": escalate,
            "escalation_reason": result.reason,
            "trace": {
                "intent": {
                    "mode": "rule-based",
                    "detected": intent.value,
                    "confidence": round(confidence, 3),
                    "matched_terms": result.matched_terms,
                    "escalate": escalate,
                }
            },
        }

    def _node_profile(self, state: HospitalityState) -> dict:
        if not state.get("route_profile"):
            return {"profile_snapshot": None, "trace": {"profile": {"skipped": True}}}
        persona_id = state.get("persona_id")
        persona: Persona | None = persona_by_id(persona_id) if persona_id else None
        if persona is None:
            return {
                "profile_snapshot": None,
                "trace": {
                    "profile": {
                        "skipped": True,
                        "reason": "no persona selected",
                    }
                },
            }
        start = time.perf_counter()
        snapshot = self._get_profile_agent().snapshot(
            persona.booking, persona_id=persona.id
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "profile_snapshot": snapshot,
            "trace": {
                "profile": {
                    "persona_id": persona.id,
                    "p_cancel": round(snapshot.p_cancel, 4),
                    "churn_band": snapshot.churn_band,
                    "value_band": snapshot.value_band,
                    "segment": snapshot.segment,
                    "latency_ms": latency_ms,
                }
            },
        }

    def _node_knowledge(self, state: HospitalityState) -> dict:
        if not state.get("route_knowledge"):
            return {"kb_hits": [], "trace": {"knowledge": {"skipped": True}}}
        persona_id = state.get("persona_id")
        persona = persona_by_id(persona_id) if persona_id else None
        scope: str | None = None
        if persona is not None:
            if "Lisboa" in persona.property:
                scope = "lisboa"
            elif "Algarve" in persona.property:
                scope = "algarve"
        start = time.perf_counter()
        hits = self._kb.search(state["user_message"], k=3, property_scope=scope)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "kb_hits": hits,
            "trace": {
                "knowledge": {
                    "retriever": "dense (Chroma)",
                    "n_hits": len(hits),
                    "top_ids": [h.page.id for h in hits],
                    "property_scope": scope,
                    "latency_ms": latency_ms,
                }
            },
        }

    def _node_compose(self, state: HospitalityState) -> dict:
        intent = state.get("intent") or HospitalityIntent.UNKNOWN
        persona_id = state.get("persona_id")
        persona = persona_by_id(persona_id) if persona_id else None
        start = time.perf_counter()
        output = compose_templated(
            intent=intent,
            profile=state.get("profile_snapshot"),
            kb_hits=state.get("kb_hits", []),
            persona=persona,
            escalate=bool(state.get("escalate")),
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "assistant_reply": output.reply,
            "citations": output.citations,
            "trace": {
                "compose": {
                    "mode": output.mode,
                    "intent": intent.value,
                    "reply_length_chars": len(output.reply),
                    "n_citations": len(output.citations),
                    "latency_ms": latency_ms,
                }
            },
        }

    def _build_graph(self):
        graph = StateGraph(HospitalityState)
        graph.add_node("classify", self._node_classify)
        graph.add_node("profile", self._node_profile)
        graph.add_node("knowledge", self._node_knowledge)
        graph.add_node("compose", self._node_compose)

        graph.add_edge(START, "classify")
        graph.add_edge("classify", "profile")
        graph.add_edge("classify", "knowledge")
        graph.add_edge("profile", "compose")
        graph.add_edge("knowledge", "compose")
        graph.add_edge("compose", END)
        return graph.compile()

    def run(
        self,
        *,
        conversation_id: str,
        user_message: str,
        persona_id: str | None = None,
        locale: str = "en",
    ) -> HospitalityState:
        initial: HospitalityState = {
            "conversation_id": conversation_id,
            "user_message": user_message,
            "persona_id": persona_id,
            "locale": locale,
        }
        result = self._graph.invoke(initial)
        return result  # type: ignore[return-value]


def run_hospitality_turn(
    *,
    conversation_id: str,
    user_message: str,
    persona_id: str | None = None,
    locale: str = "en",
) -> HospitalityState:
    return HospitalityOrchestrator().run(
        conversation_id=conversation_id,
        user_message=user_message,
        persona_id=persona_id,
        locale=locale,
    )
