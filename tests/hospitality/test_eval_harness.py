"""Eval harness — must stay above the bar in CI.

This wraps `run_harness()` so CI catches regressions before the numbers ship
to the README. Bars are calibrated to what the current templated composer
demonstrably reaches (100% on all four metrics on the 39-case set); we keep a
small safety margin so an accidental one-case slip doesn't fail the build.
"""

from __future__ import annotations

from maison_concierge.eval.hospitality_runner import load_cases, run_harness


def test_case_set_is_at_expected_size() -> None:
    cases = load_cases()
    assert 35 <= len(cases) <= 45, f"expected ~40 cases, got {len(cases)}"


def test_case_set_covers_every_category() -> None:
    cases = load_cases()
    categories = {c.category for c in cases}
    for required in (
        "knowledge",
        "profile",
        "booking",
        "recommendation",
        "escalation",
        "guardrail_pii",
        "guardrail_out_of_scope",
        "guardrail_no_invention",
        "unknown",
    ):
        assert required in categories, f"case set missing category {required!r}"


def test_harness_meets_ci_bar() -> None:
    report = run_harness()
    # Routing and guardrails must be at 100% — every miss is a real behavioural
    # regression, not a metric wobble. Groundedness and task success have a
    # small margin (one case failing is a warning, not a build break).
    assert report.routing_accuracy >= 1.00, (
        f"routing regressed to {report.routing_accuracy:.1%}"
    )
    assert report.guardrail_catch >= 1.00, (
        f"guardrail catch regressed to {report.guardrail_catch:.1%}"
    )
    assert report.groundedness >= 0.95, (
        f"groundedness regressed to {report.groundedness:.1%}"
    )
    assert report.task_success >= 0.95, (
        f"task success regressed to {report.task_success:.1%}"
    )
