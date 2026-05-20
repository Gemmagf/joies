"""End-to-end orchestrator tests with the Anthropic client + retrieval layer stubbed.

These tests prove the LangGraph state machine wires up correctly and that the
intent / route / escalate / compose branches all reach the right outcomes — without
calling the live API or loading sentence-transformers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from maison_concierge.agents import Orchestrator
from maison_concierge.agents.intent import IntentResult
from maison_concierge.models import ClientIntent
from maison_concierge.retrieval import CatalogSearchResult, HeritageSearchResult


class _StubMessages:
    def __init__(self, intent_result: IntentResult, reply_text: str) -> None:
        self._intent = intent_result
        self._reply = reply_text
        self.parse_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return SimpleNamespace(parsed_output=self._intent)

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._reply)]
        )


class _StubAnthropic:
    def __init__(self, intent_result: IntentResult, reply_text: str) -> None:
        self.messages = _StubMessages(intent_result, reply_text)


class _StubCatalogRAG:
    def __init__(self, hits: list[CatalogSearchResult] | None = None) -> None:
        self._hits = hits or []
        self.search_calls: list[str] = []

    def index(self, *, force: bool = False) -> int:
        return len(self._hits)

    def search(self, query: str, **_: Any) -> list[CatalogSearchResult]:
        self.search_calls.append(query)
        return self._hits


class _StubHeritageRAG:
    def __init__(self, hits: list[HeritageSearchResult] | None = None) -> None:
        self._hits = hits or []
        self.search_calls: list[str] = []

    def index(self, *, force: bool = False) -> int:
        return len(self._hits)

    def search(self, query: str, **_: Any) -> list[HeritageSearchResult]:
        self.search_calls.append(query)
        return self._hits


class _StubVisualSearch:
    enabled = False

    def search_by_image(self, *_: Any, **__: Any) -> list[Any]:
        return []


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("APP_ESCALATION_THRESHOLD", "0.65")
    from maison_concierge.config import get_settings
    from maison_concierge.observability.metrics import get_recorder

    get_settings.cache_clear()
    get_recorder.cache_clear()
    yield
    get_settings.cache_clear()
    get_recorder.cache_clear()


def _inject_stub_client(monkeypatch, stub: _StubAnthropic) -> None:
    monkeypatch.setattr(
        "maison_concierge.agents.intent.get_anthropic_client", lambda: stub
    )
    monkeypatch.setattr(
        "maison_concierge.agents.composer.get_anthropic_client", lambda: stub
    )


def test_orchestrator_happy_path(monkeypatch, isolated_settings):
    intent = IntentResult(
        intent="celebration",
        confidence=0.92,
        locale="en",
        needs_catalog=True,
        needs_heritage=False,
        needs_visual_search=False,
        escalate=False,
    )
    reply = "An anniversary calls for something quietly memorable."
    stub = _StubAnthropic(intent, reply)
    _inject_stub_client(monkeypatch, stub)

    catalog = _StubCatalogRAG()
    heritage = _StubHeritageRAG()
    orchestrator = Orchestrator(
        catalog=catalog, heritage=heritage, visual=_StubVisualSearch()
    )

    state = orchestrator.run(
        conversation_id="conv-happy",
        user_message="I'm looking for a wedding anniversary gift around CHF 5000.",
    )

    assert state["intent"] is ClientIntent.CELEBRATION
    assert state["intent_confidence"] == 0.92
    assert state["locale"] == "en"
    assert state["assistant_reply"] == reply
    assert state["escalate"] is False
    assert state["escalation_reason"] is None

    assert len(stub.messages.parse_calls) == 1
    assert len(stub.messages.create_calls) == 1
    assert catalog.search_calls == [
        "I'm looking for a wedding anniversary gift around CHF 5000."
    ]
    assert heritage.search_calls == []


def test_orchestrator_low_confidence_triggers_escalation(monkeypatch, isolated_settings):
    intent = IntentResult(
        intent="unknown",
        confidence=0.3,
        locale="en",
        needs_catalog=False,
        needs_heritage=False,
        needs_visual_search=False,
        escalate=False,
    )
    stub = _StubAnthropic(intent, "should-not-appear")
    _inject_stub_client(monkeypatch, stub)

    orchestrator = Orchestrator(
        catalog=_StubCatalogRAG(),
        heritage=_StubHeritageRAG(),
        visual=_StubVisualSearch(),
    )
    state = orchestrator.run(
        conversation_id="conv-escalate",
        user_message="My grandmother's necklace — could you tell me what it's worth today?",
    )

    assert state["escalate"] is True
    assert state["escalation_reason"] is not None
    assert "advisor" in state["assistant_reply"].lower()
    assert "should-not-appear" not in state["assistant_reply"]
    assert len(stub.messages.create_calls) == 0


def test_orchestrator_heritage_route_runs_heritage_search(monkeypatch, isolated_settings):
    intent = IntentResult(
        intent="heritage_inquiry",
        confidence=0.94,
        locale="fr",
        needs_catalog=False,
        needs_heritage=True,
        needs_visual_search=False,
        escalate=False,
    )
    stub = _StubAnthropic(intent, "L'Alhambra apparaît en 1968 [HER-001-ALHAMBRA-1968].")
    _inject_stub_client(monkeypatch, stub)

    catalog = _StubCatalogRAG()
    heritage = _StubHeritageRAG()
    orchestrator = Orchestrator(
        catalog=catalog, heritage=heritage, visual=_StubVisualSearch()
    )
    state = orchestrator.run(
        conversation_id="conv-heritage",
        user_message="D'où vient le motif Alhambra ?",
    )

    assert state["locale"] == "fr"
    assert state["intent"] is ClientIntent.HERITAGE_INQUIRY
    assert heritage.search_calls == ["D'où vient le motif Alhambra ?"]
    assert catalog.search_calls == []
    assert "[HER-001-ALHAMBRA-1968]" in state["assistant_reply"]


def test_orchestrator_records_metrics(monkeypatch, isolated_settings):
    intent = IntentResult(
        intent="gift",
        confidence=0.85,
        locale="en",
        needs_catalog=True,
        needs_heritage=False,
        needs_visual_search=False,
        escalate=False,
    )
    stub = _StubAnthropic(intent, "A graceful gift.")
    _inject_stub_client(monkeypatch, stub)

    orchestrator = Orchestrator(
        catalog=_StubCatalogRAG(),
        heritage=_StubHeritageRAG(),
        visual=_StubVisualSearch(),
    )
    orchestrator.run(conversation_id="conv-metrics", user_message="Looking for a gift.")

    from maison_concierge.observability.metrics import get_recorder

    snapshot = get_recorder().snapshot()
    assert snapshot.intents.get("gift") == 1
