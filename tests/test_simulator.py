"""Tests for the conversation simulator."""

from __future__ import annotations

from maison_concierge.eval.simulator import run_simulation


def test_simulator_runs_and_aggregates():
    report, turns = run_simulation(n_turns=20, seed=1)
    assert report.n_turns == 20
    assert len(turns) == 20
    assert 0.0 <= report.intent_accuracy <= 1.0
    assert 0.0 <= report.grounded_rate <= 1.0
    assert 0.0 <= report.avg_quality_score <= 1.0


def test_simulator_per_intent_breakdown_non_empty():
    report, _ = run_simulation(n_turns=30, seed=2)
    assert len(report.per_intent) > 0
    for metrics in report.per_intent.values():
        assert metrics["n"] > 0
        assert 0.0 <= metrics["intent_accuracy"] <= 1.0
        assert 0.0 <= metrics["grounded_rate"] <= 1.0


def test_simulator_worst_turns_sorted_ascending():
    report, _ = run_simulation(n_turns=20, seed=3)
    if len(report.worst_turns) >= 2:
        scores = [t.quality_score for t in report.worst_turns]
        assert scores == sorted(scores)
