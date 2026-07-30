"""Hospitality eval harness — the numbers that make the demo credible.

Loads a YAML case set, runs each case through the orchestrator, and evaluates
must / must-not assertions. Reports four metrics:

- routing_accuracy — expected_intent matches actual_intent
- groundedness    — for cases with must_cite_any_of, at least one KB id fires
- guardrail_catch — for cases with must_escalate=true, escalation actually
                    happens AND the reply carries no forbidden content
- task_success    — every assertion on a case passes

Per-category breakdowns are surfaced too, since a suite that averages 90% but
misses every guardrail case is worse than one that averages 88% and gets
every guardrail right.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import get_settings
from ..hospitality import HospitalityIntent, HospitalityOrchestrator


@dataclass(slots=True)
class EvalCase:
    id: str
    category: str
    user_message: str
    expected_intent: str
    persona_id: str | None = None
    must_escalate: bool = False
    must_cite_any_of: list[str] = field(default_factory=list)
    must_not_cite: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        return cls(
            id=d["id"],
            category=d["category"],
            user_message=d["user_message"],
            expected_intent=d["expected_intent"],
            persona_id=d.get("persona_id"),
            must_escalate=bool(d.get("must_escalate", False)),
            must_cite_any_of=list(d.get("must_cite_any_of", []) or []),
            must_not_cite=list(d.get("must_not_cite", []) or []),
            must_contain=list(d.get("must_contain", []) or []),
            must_not_contain=list(d.get("must_not_contain", []) or []),
        )


@dataclass(slots=True)
class CaseResult:
    case_id: str
    category: str
    passed: bool
    failures: list[str]
    actual_intent: str
    actual_citations: list[str]
    actual_reply_excerpt: str
    latency_ms: float
    routing_ok: bool
    groundedness_applicable: bool
    groundedness_ok: bool
    guardrail_applicable: bool
    guardrail_ok: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HarnessReport:
    n_cases: int
    n_passed: int
    task_success: float
    routing_accuracy: float
    groundedness: float          # cases with must_cite_any_of where the citation fired
    guardrail_catch: float       # cases with must_escalate where escalation + no forbidden content
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[CaseResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "n_passed": self.n_passed,
            "task_success": round(self.task_success, 4),
            "routing_accuracy": round(self.routing_accuracy, 4),
            "groundedness": round(self.groundedness, 4),
            "guardrail_catch": round(self.guardrail_catch, 4),
            "by_category": self.by_category,
            "results": [r.as_dict() for r in self.results],
        }


def load_cases(path: Path | None = None) -> list[EvalCase]:
    src = path or (get_settings().data_dir / "eval" / "hospitality_cases.yaml")
    payload = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    return [EvalCase.from_dict(d) for d in payload["cases"]]


def _evaluate(case: EvalCase, state: dict) -> CaseResult:
    intent_obj = state.get("intent")
    actual_intent = (
        intent_obj.value if isinstance(intent_obj, HospitalityIntent) else str(intent_obj)
    )
    reply = state.get("assistant_reply", "") or ""
    citations = list(state.get("citations", []) or [])
    lower_reply = reply.lower()

    failures: list[str] = []
    routing_ok = actual_intent == case.expected_intent
    if not routing_ok:
        failures.append(f"intent {actual_intent!r} != expected {case.expected_intent!r}")

    groundedness_applicable = bool(case.must_cite_any_of)
    groundedness_ok = True
    if groundedness_applicable:
        groundedness_ok = any(c in citations for c in case.must_cite_any_of)
        if not groundedness_ok:
            failures.append(
                f"none of {case.must_cite_any_of} in citations {citations!r}"
            )

    for forbidden in case.must_not_cite:
        if forbidden in citations:
            failures.append(f"forbidden citation {forbidden!r} appeared")

    for needle in case.must_contain:
        if needle.lower() not in lower_reply:
            failures.append(f"reply missing required substring {needle!r}")

    for forbidden in case.must_not_contain:
        if forbidden.lower() in lower_reply:
            failures.append(f"reply contains forbidden substring {forbidden!r}")

    guardrail_applicable = case.must_escalate
    guardrail_ok = True
    if guardrail_applicable:
        escalated = bool(state.get("escalate"))
        no_bad_content = all(f.startswith("intent ") or "forbidden" not in f for f in failures)
        guardrail_ok = escalated and no_bad_content
        if not escalated:
            failures.append("expected escalation but escalate flag was False")

    return CaseResult(
        case_id=case.id,
        category=case.category,
        passed=not failures,
        failures=failures,
        actual_intent=actual_intent,
        actual_citations=citations,
        actual_reply_excerpt=reply[:240],
        latency_ms=0.0,  # filled by caller
        routing_ok=routing_ok,
        groundedness_applicable=groundedness_applicable,
        groundedness_ok=groundedness_ok,
        guardrail_applicable=guardrail_applicable,
        guardrail_ok=guardrail_ok,
    )


def run_harness(
    cases: list[EvalCase] | None = None,
    orchestrator: HospitalityOrchestrator | None = None,
) -> HarnessReport:
    cases = cases or load_cases()
    orch = orchestrator or HospitalityOrchestrator()

    results: list[CaseResult] = []
    for case in cases:
        t0 = time.perf_counter()
        state = orch.run(
            conversation_id=f"eval-{case.id}",
            user_message=case.user_message,
            persona_id=case.persona_id,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        result = _evaluate(case, dict(state))
        result.latency_ms = latency_ms
        results.append(result)

    n_cases = len(results)
    n_passed = sum(1 for r in results if r.passed)
    routing_correct = sum(1 for r in results if r.routing_ok)
    groundedness_applicable = [r for r in results if r.groundedness_applicable]
    guardrail_applicable = [r for r in results if r.guardrail_applicable]

    by_category: dict[str, dict[str, float]] = {}
    for r in results:
        bucket = by_category.setdefault(r.category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if r.passed:
            bucket["passed"] += 1
    for stats in by_category.values():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 4)

    return HarnessReport(
        n_cases=n_cases,
        n_passed=n_passed,
        task_success=n_passed / n_cases if n_cases else 0.0,
        routing_accuracy=routing_correct / n_cases if n_cases else 0.0,
        groundedness=(
            sum(1 for r in groundedness_applicable if r.groundedness_ok)
            / len(groundedness_applicable)
            if groundedness_applicable
            else 1.0
        ),
        guardrail_catch=(
            sum(1 for r in guardrail_applicable if r.guardrail_ok)
            / len(guardrail_applicable)
            if guardrail_applicable
            else 1.0
        ),
        by_category=by_category,
        results=results,
    )


def write_report(report: HarnessReport, *, out_dir: Path | None = None) -> tuple[Path, Path]:
    out = out_dir or (get_settings().data_dir / "eval")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "hospitality_report.json"
    md_path = out / "hospitality_report.md"

    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Hospitality eval — latest run",
        "",
        f"- **Cases:** {report.n_cases}",
        f"- **Task success:** {report.task_success:.1%}  "
        f"({report.n_passed}/{report.n_cases})",
        f"- **Routing accuracy:** {report.routing_accuracy:.1%}",
        f"- **Groundedness (citation fired when required):** {report.groundedness:.1%}",
        f"- **Guardrail catch-rate:** {report.guardrail_catch:.1%}",
        "",
        "## By category",
        "",
        "| Category | Passed | Total | Rate |",
        "|---|---:|---:|---:|",
    ]
    for cat, stats in sorted(report.by_category.items()):
        lines.append(
            f"| {cat} | {int(stats['passed'])} | {int(stats['total'])} | "
            f"{stats['pass_rate']:.0%} |"
        )
    lines.append("")
    fails = [r for r in report.results if not r.passed]
    if fails:
        lines.append("## Failures")
        lines.append("")
        for r in fails:
            lines.append(f"- **{r.case_id}** ({r.category}, {r.actual_intent})")
            for f in r.failures:
                lines.append(f"  - {f}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
