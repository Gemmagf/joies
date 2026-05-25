"""Test that the orchestrator emits a populated trace in demo mode."""

from __future__ import annotations

from maison_concierge.agents import Orchestrator
from maison_concierge.config import get_settings


def test_trace_populated_for_celebration_query(monkeypatch, tmp_path):
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    from maison_concierge.observability.metrics import get_recorder
    get_recorder.cache_clear()

    orc = Orchestrator()
    assert orc.demo_mode
    state = orc.run(
        conversation_id="trace-1",
        user_message="I'm looking for an anniversary gift around CHF 5000",
    )
    trace = state.get("trace") or {}
    assert "intent" in trace
    assert "catalog" in trace
    assert "compose" in trace
    assert "groundedness" in trace
    assert trace["intent"]["detected"] == "celebration"
    assert "fused_top" in trace["catalog"]


def test_trace_includes_heritage_section_for_heritage_query(monkeypatch, tmp_path):
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    from maison_concierge.observability.metrics import get_recorder
    get_recorder.cache_clear()

    orc = Orchestrator()
    state = orc.run(
        conversation_id="trace-2",
        user_message="Tell me about the history of the Alhambra collection",
    )
    trace = state.get("trace") or {}
    assert trace["intent"]["detected"] == "heritage_inquiry"
    assert "heritage" in trace
    assert not trace["heritage"].get("skipped")
    assert "top_ids" in trace["heritage"]
