"""Build a small entity co-occurrence graph from the heritage corpus.

Entities are extracted with rule-based patterns (collections from the catalog enum,
4-digit years, hand-curated craftsmanship and place vocabularies). Nodes are entities
plus the documents themselves; edges connect every entity to the documents it appears
in, and entities that co-occur in the same document to each other (weighted).

This is deliberately not an ML extraction (e.g. spaCy NER) — for a 10-document corpus,
rule-based extraction is both more accurate and more legible. The visualisation lives
in the dashboard's "Heritage graph" tab and renders with pyvis.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

import networkx as nx

from ..data_loader import load_heritage
from ..models import Collection

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

_TECHNIQUES = {
    "Mystery Setting": ["mystery setting", "serti mystérieux", "serti mysterieux"],
    "Granulation": ["granulation", "granulés", "granules"],
    "Mother-of-pearl craft": ["mother-of-pearl", "nacre"],
    "Mosaic inspiration": ["mosaic", "mosaïque", "mosaique"],
    "Stone selection": ["stone selection", "sélection des pierres", "gemologists", "gemmologues"],
}

_PLACES = {
    "Geneva": ["geneva", "genève", "geneve"],
    "Paris": ["paris", "place vendôme", "place vendome"],
    "Granada": ["granada", "grenade"],
    "New York": ["new york", "broadway"],
    "Mogok": ["mogok"],
    "Muzo": ["muzo"],
}


def _find_substrings(text: str, vocabulary: dict[str, list[str]]) -> set[str]:
    text_lower = text.lower()
    found: set[str] = set()
    for canonical, surface_forms in vocabulary.items():
        if any(form in text_lower for form in surface_forms):
            found.add(canonical)
    return found


def _extract_entities(doc) -> dict[str, set[str]]:
    text = f"{doc.title.en} {doc.title.fr} {doc.body.en} {doc.body.fr}"

    collections: set[str] = set()
    text_lower = text.lower()
    for c in Collection:
        if c.value.replace("_", " ") in text_lower or c.value in text_lower:
            collections.add(c.value)
    if doc.collection is not None:
        collections.add(doc.collection.value)

    years: set[str] = {m.group(0) for m in _YEAR_RE.finditer(text)}
    techniques: set[str] = _find_substrings(text, _TECHNIQUES)
    places: set[str] = _find_substrings(text, _PLACES)

    return {
        "collection": collections,
        "year": years,
        "technique": techniques,
        "place": places,
    }


@lru_cache(maxsize=1)
def build_graph() -> nx.Graph:
    g = nx.Graph()
    documents = load_heritage()

    entity_to_docs: dict[tuple[str, str], list[str]] = defaultdict(list)

    for doc in documents:
        doc_label = doc.title.en
        g.add_node(doc.id, kind="document", label=doc_label)
        ents = _extract_entities(doc)
        all_entities_in_doc: list[tuple[str, str]] = []
        for kind, values in ents.items():
            for value in values:
                node_id = f"{kind}:{value}"
                if node_id not in g:
                    g.add_node(node_id, kind=kind, label=value)
                g.add_edge(doc.id, node_id, weight=1.0, source=doc.id)
                entity_to_docs[(kind, value)].append(doc.id)
                all_entities_in_doc.append((kind, value))

        # Add co-occurrence edges (entity-entity) within the same document
        for i, a in enumerate(all_entities_in_doc):
            for b in all_entities_in_doc[i + 1 :]:
                if a == b:
                    continue
                a_id = f"{a[0]}:{a[1]}"
                b_id = f"{b[0]}:{b[1]}"
                if g.has_edge(a_id, b_id):
                    g[a_id][b_id]["weight"] += 1.0
                else:
                    g.add_edge(a_id, b_id, weight=1.0, source=doc.id)

    return g
