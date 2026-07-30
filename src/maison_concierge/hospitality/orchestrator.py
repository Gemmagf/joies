"""LangGraph orchestrator for the hospitality concierge.

Graph shape:

    classify ─┬─► profile ────┐
              ├─► knowledge ──┤
              ├─► booking ────┼─► compose ──► guardrail ──► end
              └─► recommend ──┘

The classify node produces routing flags. Each specialist node returns
either its result payload or a `skipped` trace entry, so compose sees a
uniform state. Guardrail runs after compose and may rewrite the reply
(confidence override, PII scrub, citation subset, escalation content).
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from .booking import BookingAgent
from .composer import compose_templated
from .guardrail import apply_guardrails
from .intent import HospitalityIntent, classify_intent_rule_based
from .kb import PropertyKB
from .personas import Persona, persona_by_id
from .profile import ProfileAgent
from .recommend import RecommendationAgent
from .state import HospitalityState


class HospitalityOrchestrator:
    def __init__(
        self,
        profile: ProfileAgent | None = None,
        kb: PropertyKB | None = None,
        booking: BookingAgent | None = None,
        recommend: RecommendationAgent | None = None,
    ) -> None:
        self._profile = profile
        self._kb = kb or PropertyKB()
        self._booking = booking or BookingAgent()
        self._recommend = recommend or RecommendationAgent()
        self._settings = get_settings()
        self._graph = self._build_graph()

    def _get_profile_agent(self) -> ProfileAgent:
        if self._profile is None:
            self._profile = ProfileAgent()
        return self._profile

    # ── nodes ────────────────────────────────────────────────────────
    def _node_classify(self, state: HospitalityState) -> dict:
        text = state.get("user_message", "")
        result = classify_intent_rule_based(text)
        intent = result.intent
        confidence = result.confidence

        escalate = result.escalate
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
        persona = self._resolve_persona(state)
        if persona is None:
            return {
                "profile_snapshot": None,
                "trace": {"profile": {"skipped": True, "reason": "no persona selected"}},
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
        persona = self._resolve_persona(state)
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

    def _node_booking(self, state: HospitalityState) -> dict:
        if not state.get("route_booking"):
            return {"booking_result": None, "trace": {"booking": {"skipped": True}}}
        persona = self._resolve_persona(state)
        if persona is None:
            return {
                "booking_result": None,
                "trace": {"booking": {"skipped": True, "reason": "no persona selected"}},
            }
        start = time.perf_counter()
        result = self._booking.handle(persona, state["user_message"])
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "booking_result": result,
            "trace": {
                "booking": {
                    "action": result.action.value,
                    "ok": result.ok,
                    "requires_human": result.requires_human,
                    "booking_ref": result.booking_ref,
                    "refund_eur": result.refund_eur,
                    "latency_ms": latency_ms,
                }
            },
        }

    def _node_recommend(self, state: HospitalityState) -> dict:
        if not state.get("route_recommendation"):
            return {
                "recommendation": None,
                "trace": {"recommend": {"skipped": True}},
            }
        persona = self._resolve_persona(state)
        if persona is None:
            return {
                "recommendation": None,
                "trace": {"recommend": {"skipped": True, "reason": "no persona selected"}},
            }
        try:
            snap = self._get_profile_agent().snapshot(
                persona.booking, persona_id=persona.id
            )
        except Exception as exc:
            return {
                "recommendation": None,
                "trace": {"recommend": {"skipped": True, "reason": f"profile failed: {exc}"}},
            }
        start = time.perf_counter()
        rec = self._recommend.rank(
            persona,
            segment=snap.segment,
            value_band=snap.value_band,
            top_n=3,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "recommendation": rec,
            "trace": {
                "recommend": {
                    "n_ranked": len(rec.ranked),
                    "signals": rec.signals_used,
                    "top_ids": [r.experience.id for r in rec.ranked],
                    "latency_ms": latency_ms,
                }
            },
        }

    def _node_compose(self, state: HospitalityState) -> dict:
        intent = state.get("intent") or HospitalityIntent.UNKNOWN
        persona = self._resolve_persona(state)
        start = time.perf_counter()
        output = compose_templated(
            intent=intent,
            profile=state.get("profile_snapshot"),
            kb_hits=state.get("kb_hits", []),
            booking_result=state.get("booking_result"),
            recommendation=state.get("recommendation"),
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

    def _node_guardrail(self, state: HospitalityState) -> dict:
        intent = state.get("intent") or HospitalityIntent.UNKNOWN
        confidence = float(state.get("intent_confidence", 0.0))
        reply = state.get("assistant_reply", "") or ""
        citations = list(state.get("citations", []) or [])
        kb_hit_ids = [h.page.id for h in state.get("kb_hits", []) or []]

        start = time.perf_counter()
        new_reply, new_citations, report = apply_guardrails(
            reply=reply,
            citations=citations,
            intent=intent,
            intent_confidence=confidence,
            kb_hit_ids=kb_hit_ids,
            escalation_threshold=self._settings.app_escalation_threshold,
            force_escalation=bool(state.get("escalate")),
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "assistant_reply": new_reply,
            "citations": new_citations,
            "guardrail_report": report,
            "trace": {
                "guardrail": {
                    "passed": report.passed,
                    "reply_rewritten": report.reply_rewritten,
                    "checks": report.checks,
                    "n_redactions": len(report.redactions),
                    "latency_ms": latency_ms,
                }
            },
        }

    def _resolve_persona(self, state: HospitalityState) -> Persona | None:
        persona_id = state.get("persona_id")
        if not persona_id:
            return None
        return persona_by_id(persona_id)

    # ── graph ────────────────────────────────────────────────────────
    def _build_graph(self):
        graph = StateGraph(HospitalityState)
        graph.add_node("classify", self._node_classify)
        graph.add_node("profile", self._node_profile)
        graph.add_node("knowledge", self._node_knowledge)
        graph.add_node("booking", self._node_booking)
        graph.add_node("recommend", self._node_recommend)
        graph.add_node("compose", self._node_compose)
        graph.add_node("guardrail", self._node_guardrail)

        graph.add_edge(START, "classify")
        for downstream in ("profile", "knowledge", "booking", "recommend"):
            graph.add_edge("classify", downstream)
            graph.add_edge(downstream, "compose")
        graph.add_edge("compose", "guardrail")
        graph.add_edge("guardrail", END)
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
        return self._graph.invoke(initial)  # type: ignore[return-value]


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
