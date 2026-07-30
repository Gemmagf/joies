"""Guardrail / Responsible-AI layer — explicit LangGraph node.

Runs after compose. Applies four checks:

1. **Confidence override.** If the classifier's intent_confidence dropped
   below the escalation threshold on a non-UNKNOWN intent, the reply is
   replaced with the standard escalation message.
2. **PII leak scrub.** Redacts email addresses and long digit runs from
   the assistant reply. The templated composer never produces these, but
   the LLM composer (added later) might; the check is here so the wire is
   already in place.
3. **KB citation subset.** For knowledge answers, every declared citation
   must correspond to a KB page that actually surfaced in this turn.
4. **Refusal on unsafe request.** If the intent was escalate, the reply
   must not contain forbidden numeric or personal data patterns.

Each check produces an entry in `trace.guardrail`. The reply is only
rewritten when a check fires — the common case is a passthrough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .intent import HospitalityIntent

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")     # cards, phones without separators
_CARD_LIKE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_ESCALATION_REPLY = (
    "I'm going to bring this to a human colleague right away. A member of "
    "the Concierge d'Honneur will follow up within a few minutes. If it's "
    "urgent, please call the property directly at +351 21 000 1234."
)

_FORBIDDEN_IN_ESCALATION = ("credit card number", "cvv", "cvc", "iban")


@dataclass(slots=True)
class GuardrailReport:
    passed: bool = True
    reply_rewritten: bool = False
    checks: dict[str, str] = field(default_factory=dict)  # check_name -> "ok"|"fired"|"n/a"
    redactions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reply_rewritten": self.reply_rewritten,
            "checks": self.checks,
            "redactions": self.redactions,
        }


def _scrub(text: str, redactions: list[str]) -> str:
    def _repl(pattern: re.Pattern[str], mask: str, s: str) -> str:
        def sub(m: re.Match[str]) -> str:
            redactions.append(m.group(0))
            return mask
        return pattern.sub(sub, s)

    text = _repl(_EMAIL_RE, "[email redacted]", text)
    text = _repl(_CARD_LIKE_RE, "[card redacted]", text)
    text = _repl(_LONG_DIGITS_RE, "[digits redacted]", text)
    return text


def apply_guardrails(
    *,
    reply: str,
    citations: list[str],
    intent: HospitalityIntent,
    intent_confidence: float,
    kb_hit_ids: list[str],
    escalation_threshold: float,
    force_escalation: bool = False,
) -> tuple[str, list[str], GuardrailReport]:
    report = GuardrailReport()

    # 1. Confidence override — a real low-confidence classification should
    #    hand off rather than answer.
    if (
        force_escalation
        or (
            intent not in {HospitalityIntent.UNKNOWN, HospitalityIntent.ESCALATE}
            and intent_confidence < escalation_threshold
        )
    ):
        report.checks["confidence_override"] = "fired"
        report.reply_rewritten = True
        return _ESCALATION_REPLY, [], report
    report.checks["confidence_override"] = "ok"

    # 2. PII scrub — passthrough unless something matches.
    scrubbed = _scrub(reply, report.redactions)
    if scrubbed != reply:
        report.checks["pii_scrub"] = "fired"
        report.reply_rewritten = True
        reply = scrubbed
    else:
        report.checks["pii_scrub"] = "ok"

    # 3. Citation subset — every cited page must have been retrieved.
    if intent == HospitalityIntent.KNOWLEDGE and citations:
        stray = [c for c in citations if c not in kb_hit_ids]
        if stray:
            report.checks["citation_subset"] = "fired"
            report.passed = False
            citations = [c for c in citations if c in kb_hit_ids]
            report.redactions.append(f"stray_citations={stray}")
        else:
            report.checks["citation_subset"] = "ok"
    else:
        report.checks["citation_subset"] = "n/a"

    # 4. Escalation-content check — the escalation reply must not carry
    #    forbidden strings.
    if intent == HospitalityIntent.ESCALATE:
        lowered = reply.lower()
        hits = [t for t in _FORBIDDEN_IN_ESCALATION if t in lowered]
        if hits:
            report.checks["escalation_content"] = "fired"
            report.reply_rewritten = True
            report.passed = False
            return _ESCALATION_REPLY, [], report
        report.checks["escalation_content"] = "ok"
    else:
        report.checks["escalation_content"] = "n/a"

    return reply, citations, report
