"""Groundedness checks: every piece reference and price in the reply must be supported.

Hallucination in a luxury context is operationally catastrophic — a fabricated reference
or a wrong price reaches a client who acts on it. This module runs deterministic post-hoc
checks over (reply, retrieved evidence) pairs without needing an LLM judge.

Three categories of finding:
  • unsupported_piece — a VCA-XXX reference in the reply not present in evidence + not in
    the underlying catalog
  • unsupported_price — a CHF price tied to a piece reference whose catalog price differs
  • unsupported_heritage — a [HER-XXX] citation not present in retrieved heritage hits
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..data_loader import heritage_by_id, piece_by_id
from ..models import CatalogPiece, HeritageDocument

FindingKind = Literal[
    "unsupported_piece",
    "unsupported_price",
    "unsupported_heritage",
    "missing_citation",
]

_PIECE_REF = re.compile(r"\bVCA-[A-Z0-9-]+\b")
_HERITAGE_REF = re.compile(r"\[HER-[A-Z0-9-]+\]")
_PRICE_NEAR_REF = re.compile(
    r"(VCA-[A-Z0-9-]+)\s*(?:[—–-].*?)?\(?CHF\s*([\d,]+(?:\.\d+)?)\)?",  # noqa: RUF001
    re.IGNORECASE,
)


@dataclass(slots=True)
class GroundednessFinding:
    kind: FindingKind
    detail: str
    span: str


@dataclass(slots=True)
class GroundednessReport:
    reply: str
    findings: list[GroundednessFinding] = field(default_factory=list)
    is_grounded: bool = True

    @property
    def hallucination_count(self) -> int:
        return sum(1 for f in self.findings if f.kind != "missing_citation")


def _heritage_ids_from_reply(reply: str) -> list[str]:
    return [m.group(0).strip("[]") for m in _HERITAGE_REF.finditer(reply)]


def _piece_refs_from_reply(reply: str) -> list[str]:
    return [m.group(0) for m in _PIECE_REF.finditer(reply)]


def _price_claims(reply: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for m in _PRICE_NEAR_REF.finditer(reply):
        ref = m.group(1)
        amount_str = m.group(2).replace(",", "")
        try:
            out.append((ref, float(amount_str)))
        except ValueError:
            continue
    return out


def check_groundedness(
    reply: str,
    *,
    catalog_evidence: list[CatalogPiece] | None = None,
    heritage_evidence: list[HeritageDocument] | None = None,
    price_tolerance_chf: float = 1.0,
) -> GroundednessReport:
    report = GroundednessReport(reply=reply)
    evidence_piece_ids = {p.id for p in (catalog_evidence or [])}
    evidence_heritage_ids = {d.id for d in (heritage_evidence or [])}

    for ref in _piece_refs_from_reply(reply):
        if ref in evidence_piece_ids:
            continue
        catalog_piece = piece_by_id(ref)
        if catalog_piece is None:
            report.findings.append(
                GroundednessFinding(
                    kind="unsupported_piece",
                    detail=f"Reference {ref} does not exist in the catalog.",
                    span=ref,
                )
            )
        else:
            report.findings.append(
                GroundednessFinding(
                    kind="missing_citation",
                    detail=(
                        f"Reference {ref} exists in catalog but was not in retrieval evidence."
                    ),
                    span=ref,
                )
            )

    for ref, claimed_price in _price_claims(reply):
        catalog_piece = piece_by_id(ref)
        if catalog_piece is None:
            continue
        if abs(catalog_piece.price_chf - claimed_price) > price_tolerance_chf:
            report.findings.append(
                GroundednessFinding(
                    kind="unsupported_price",
                    detail=(
                        f"Stated CHF {claimed_price:,.0f} for {ref} disagrees with catalog "
                        f"value CHF {catalog_piece.price_chf:,.0f}."
                    ),
                    span=f"{ref} CHF {claimed_price:,.0f}",
                )
            )

    for hid in _heritage_ids_from_reply(reply):
        if hid in evidence_heritage_ids:
            continue
        doc = heritage_by_id(hid)
        if doc is None:
            report.findings.append(
                GroundednessFinding(
                    kind="unsupported_heritage",
                    detail=f"Heritage citation [{hid}] does not exist in the archive.",
                    span=f"[{hid}]",
                )
            )
        else:
            report.findings.append(
                GroundednessFinding(
                    kind="missing_citation",
                    detail=f"Heritage citation [{hid}] is valid but not in retrieval evidence.",
                    span=f"[{hid}]",
                )
            )

    report.is_grounded = report.hallucination_count == 0
    return report
