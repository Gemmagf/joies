"""Retrieval-side evaluation: recall@k, MRR, hit rate per intent.

These metrics quantify whether the catalog / heritage RAG actually surfaces the right
documents for a labelled query. They are the foundation for any downstream claim about
the system's quality — if recall@3 < 0.7 on a given intent, no system prompt can recover.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from ..models import ClientIntent
from ..retrieval import CatalogRAG, HeritageRAG, HybridCatalogRAG
from .golden import GoldenItem, load_golden_set


@dataclass(slots=True)
class PerQueryResult:
    item_id: str
    intent: ClientIntent
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    hit_at_1: bool
    retrieved_ids: list[str]
    expected_ids: list[str]
    target: str  # "catalog" or "heritage"


@dataclass(slots=True)
class RetrievalMetrics:
    target: str
    n_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    per_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query: list[PerQueryResult] = field(default_factory=list)


def _recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    relevant_in_top_k = sum(1 for x in retrieved[:k] if x in expected)
    return relevant_in_top_k / len(expected)


def _mrr(retrieved: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def _aggregate(per_query: list[PerQueryResult]) -> dict[str, float]:
    if not per_query:
        return {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0}
    return {
        "recall_at_1": mean(r.recall_at_1 for r in per_query),
        "recall_at_3": mean(r.recall_at_3 for r in per_query),
        "recall_at_5": mean(r.recall_at_5 for r in per_query),
        "mrr": mean(r.mrr for r in per_query),
    }


def evaluate_retrieval(
    *,
    target: str = "catalog",
    catalog: CatalogRAG | None = None,
    heritage: HeritageRAG | None = None,
    hybrid: HybridCatalogRAG | None = None,
    items: list[GoldenItem] | None = None,
    k: int = 5,
) -> RetrievalMetrics:
    items = items or load_golden_set()
    if target == "catalog":
        rag = catalog or CatalogRAG()
        rag.index()
        relevant_items = [it for it in items if it.expected_piece_ids_top_k]
        get_expected = lambda it: it.expected_piece_ids_top_k  # noqa: E731
    elif target == "catalog_hybrid":
        rag = hybrid or HybridCatalogRAG()
        rag.index()
        relevant_items = [it for it in items if it.expected_piece_ids_top_k]
        get_expected = lambda it: it.expected_piece_ids_top_k  # noqa: E731
    elif target == "heritage":
        rag = heritage or HeritageRAG()
        rag.index()
        relevant_items = [it for it in items if it.expected_heritage_ids]
        get_expected = lambda it: it.expected_heritage_ids  # noqa: E731
    else:
        raise ValueError(f"Unknown target: {target}")

    per_query: list[PerQueryResult] = []
    for it in relevant_items:
        expected = get_expected(it)
        if target in ("catalog", "catalog_hybrid"):
            hits = rag.search(it.text, k=k)
            retrieved_ids = [h.piece.id for h in hits]
        else:
            hits = rag.search(it.text, k=k, locale=it.locale)
            retrieved_ids = [h.document.id for h in hits]
        per_query.append(
            PerQueryResult(
                item_id=it.id,
                intent=it.expected_intent,
                recall_at_1=_recall_at_k(retrieved_ids, expected, 1),
                recall_at_3=_recall_at_k(retrieved_ids, expected, 3),
                recall_at_5=_recall_at_k(retrieved_ids, expected, 5),
                mrr=_mrr(retrieved_ids, expected),
                hit_at_1=any(rid in expected for rid in retrieved_ids[:1]),
                retrieved_ids=retrieved_ids,
                expected_ids=expected,
                target=target,
            )
        )

    by_intent: dict[ClientIntent, list[PerQueryResult]] = defaultdict(list)
    for r in per_query:
        by_intent[r.intent].append(r)
    per_intent = {intent.value: _aggregate(group) for intent, group in by_intent.items()}

    agg = _aggregate(per_query)
    return RetrievalMetrics(
        target=target,
        n_queries=len(per_query),
        recall_at_1=agg["recall_at_1"],
        recall_at_3=agg["recall_at_3"],
        recall_at_5=agg["recall_at_5"],
        mrr=agg["mrr"],
        per_intent=per_intent,
        per_query=per_query,
    )
