"""LangGraph state machine orchestrating intent → retrieval → composition.

Graph shape:

    classify_intent ─┬─► catalog ──┐
                     ├─► heritage ─┼─► compose ─► end
                     └─► visual ───┘
                     (any subset, run in parallel)

Escalation short-circuits straight to compose so the apology is centralised.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..eval.groundedness import check_groundedness
from ..i18n import detect_locale, t
from ..models import ClientIntent, EscalationReason
from ..observability import get_recorder
from ..retrieval import CatalogRAG, HeritageRAG, HybridCatalogRAG, VisualSearch
from .composer import compose_reply
from .demo_composer import compose_reply_templated
from .demo_intent import classify_intent_rule_based
from .intent import classify_intent
from .state import OrchestratorState


class Orchestrator:
    def __init__(
        self,
        catalog: CatalogRAG | None = None,
        heritage: HeritageRAG | None = None,
        visual: VisualSearch | None = None,
    ) -> None:
        # Default the catalog to the hybrid (BM25 + dense, RRF) retriever — it produced
        # +33% recall@1 / +27% recall@5 on the golden set; see the README.
        self._catalog = catalog or HybridCatalogRAG()
        self._heritage = heritage or HeritageRAG()
        self._visual = visual or VisualSearch()
        self._recorder = get_recorder()
        self._settings = get_settings()
        self._graph = self._build_graph()

    @property
    def demo_mode(self) -> bool:
        return self._settings.use_demo_mode

    def _node_classify(self, state: OrchestratorState) -> dict:
        text = state.get("user_message", "")
        has_image = state.get("image_bytes") is not None
        locale = state.get("locale") or detect_locale(text, self._settings.app_locale_default)
        if self.demo_mode:
            result = classify_intent_rule_based(text, default_locale=locale, has_image=has_image)
        else:
            result = classify_intent(text, default_locale=locale, has_image=has_image)

        intent = ClientIntent(result.intent)
        self._recorder.record_intent(state["conversation_id"], intent, result.confidence)

        escalate = (
            result.escalate
            or result.confidence < self._settings.app_escalation_threshold
        )
        escalation_reason = (
            EscalationReason.EXPLICIT_REQUEST
            if result.escalate
            else EscalationReason.LOW_CONFIDENCE
            if escalate
            else None
        )
        if escalation_reason is not None:
            self._recorder.record_escalation(state["conversation_id"], escalation_reason)

        return {
            "locale": result.locale,
            "intent": intent,
            "intent_confidence": result.confidence,
            "route_catalog": result.needs_catalog and not escalate,
            "route_heritage": result.needs_heritage and not escalate,
            "route_visual": (result.needs_visual_search or has_image) and not escalate,
            "escalate": escalate,
            "escalation_reason": escalation_reason,
            "trace": {
                "intent": {
                    "mode": "demo" if self.demo_mode else "claude",
                    "detected": intent.value,
                    "confidence": round(result.confidence, 3),
                    "locale": result.locale,
                    "escalate": escalate,
                    "escalation_reason": escalation_reason.value if escalation_reason else None,
                },
            },
        }

    def _node_catalog(self, state: OrchestratorState) -> dict:
        if not state.get("route_catalog"):
            return {"catalog_hits": [], "trace": {"catalog": {"skipped": True}}}
        self._catalog.index()
        start = time.perf_counter()
        hits = self._catalog.search(state["user_message"], k=4)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        trace_payload: dict = {"n_hits": len(hits), "latency_ms": latency_ms}
        if isinstance(self._catalog, HybridCatalogRAG):
            trace_payload.update(self._catalog.last_diagnostics)
            trace_payload["retriever"] = "hybrid (BM25 + dense, RRF)"
        else:
            trace_payload["retriever"] = "dense only"
            trace_payload["fused_top"] = [h.piece.id for h in hits]
        return {"catalog_hits": hits, "trace": {"catalog": trace_payload}}

    def _node_heritage(self, state: OrchestratorState) -> dict:
        if not state.get("route_heritage"):
            return {"heritage_hits": [], "trace": {"heritage": {"skipped": True}}}
        self._heritage.index()
        start = time.perf_counter()
        hits = self._heritage.search(state["user_message"], k=3, locale=state.get("locale", "en"))
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "heritage_hits": hits,
            "trace": {
                "heritage": {
                    "retriever": "dense (Chroma)",
                    "n_hits": len(hits),
                    "top_ids": [h.document.id for h in hits],
                    "latency_ms": latency_ms,
                }
            },
        }

    def _node_visual(self, state: OrchestratorState) -> dict:
        if not state.get("route_visual") or state.get("image_bytes") is None:
            return {"visual_hits": [], "trace": {"visual": {"skipped": True}}}
        hits = self._visual.search_by_image(state["image_bytes"], k=4)
        return {
            "visual_hits": hits,
            "trace": {
                "visual": {
                    "retriever": "CLIP" if not (hits and hits[0].degraded) else "text fallback",
                    "n_hits": len(hits),
                }
            },
        }

    def _node_compose(self, state: OrchestratorState) -> dict:
        locale = state.get("locale", "en")
        intent = state.get("intent") or ClientIntent.UNKNOWN
        if state.get("escalate"):
            return {
                "assistant_reply": t("chat.escalation", locale),
                "citations": [],
                "trace": {
                    "compose": {
                        "mode": "escalation",
                        "template": "chat.escalation",
                    }
                },
            }
        start = time.perf_counter()
        if self.demo_mode:
            reply, citations = compose_reply_templated(
                intent=intent,
                locale=locale,
                catalog_hits=state.get("catalog_hits", []),
                heritage_hits=state.get("heritage_hits", []),
                visual_hits=state.get("visual_hits", []),
            )
            compose_mode = "demo"
        else:
            reply, citations = compose_reply(
                user_message=state["user_message"],
                locale=locale,
                catalog_hits=state.get("catalog_hits", []),
                heritage_hits=state.get("heritage_hits", []),
                visual_hits=state.get("visual_hits", []),
            )
            compose_mode = "claude (sonnet-4-6, adaptive thinking, prompt-cached)"
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        groundedness = check_groundedness(
            reply,
            catalog_evidence=[h.piece for h in state.get("catalog_hits", [])],
            heritage_evidence=[h.document for h in state.get("heritage_hits", [])],
        )
        finding_counts: dict[str, int] = {}
        for f in groundedness.findings:
            finding_counts[f.kind] = finding_counts.get(f.kind, 0) + 1

        return {
            "assistant_reply": reply,
            "citations": citations,
            "trace": {
                "compose": {
                    "mode": compose_mode,
                    "intent": intent.value,
                    "latency_ms": latency_ms,
                    "reply_length_chars": len(reply),
                    "n_citations": len(citations),
                },
                "groundedness": {
                    "is_grounded": groundedness.is_grounded,
                    "hallucination_count": groundedness.hallucination_count,
                    "findings_by_kind": finding_counts,
                },
            },
        }

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("classify", self._node_classify)
        graph.add_node("catalog", self._node_catalog)
        graph.add_node("heritage", self._node_heritage)
        graph.add_node("visual", self._node_visual)
        graph.add_node("compose", self._node_compose)

        graph.add_edge(START, "classify")
        graph.add_edge("classify", "catalog")
        graph.add_edge("classify", "heritage")
        graph.add_edge("classify", "visual")
        graph.add_edge("catalog", "compose")
        graph.add_edge("heritage", "compose")
        graph.add_edge("visual", "compose")
        graph.add_edge("compose", END)
        return graph.compile()

    def run(
        self,
        *,
        conversation_id: str,
        user_message: str,
        locale: str | None = None,
        image_bytes: bytes | None = None,
    ) -> OrchestratorState:
        initial: OrchestratorState = {
            "conversation_id": conversation_id,
            "user_message": user_message,
            "image_bytes": image_bytes,
        }
        if locale:
            initial["locale"] = locale  # type: ignore[typeddict-item]
        result = self._graph.invoke(initial)
        return result  # type: ignore[return-value]


def run_turn(
    *,
    conversation_id: str,
    user_message: str,
    locale: str | None = None,
    image_bytes: bytes | None = None,
) -> OrchestratorState:
    return Orchestrator().run(
        conversation_id=conversation_id,
        user_message=user_message,
        locale=locale,
        image_bytes=image_bytes,
    )
