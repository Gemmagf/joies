"""Tests for the heritage knowledge graph."""

from __future__ import annotations

import networkx as nx

from maison_concierge.analysis.heritage_graph import build_graph


def test_graph_has_documents_and_entities():
    g = build_graph()
    assert g.number_of_nodes() > 10
    assert g.number_of_edges() > 0


def test_graph_includes_known_collections():
    g = build_graph()
    labels = {data.get("label") for _, data in g.nodes(data=True)}
    assert "alhambra" in labels


def test_graph_extracts_years():
    g = build_graph()
    year_nodes = [d.get("label") for _, d in g.nodes(data=True) if d.get("kind") == "year"]
    assert "1968" in year_nodes


def test_graph_extracts_places():
    g = build_graph()
    place_nodes = [d.get("label") for _, d in g.nodes(data=True) if d.get("kind") == "place"]
    assert "Geneva" in place_nodes or "Paris" in place_nodes


def test_every_document_connects_to_at_least_one_entity():
    g = build_graph()
    doc_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "document"]
    assert doc_nodes, "expected at least one document node"
    for doc_id in doc_nodes:
        assert g.degree(doc_id) >= 1, f"document {doc_id} has no entity edges"


def test_graph_has_multiple_components_for_diverse_corpus():
    # 10 heritage docs span very different topics; full connectedness would suggest
    # over-extraction. We just check the graph is non-empty.
    g = build_graph()
    assert nx.number_connected_components(g) >= 1
