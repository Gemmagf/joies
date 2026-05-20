from maison_concierge.data_loader import (
    heritage_by_id,
    load_catalog,
    load_heritage,
    piece_by_id,
)


def test_catalog_loads():
    pieces = load_catalog()
    assert len(pieces) >= 20
    assert all(p.id.startswith("VCA-") for p in pieces)


def test_catalog_lookup_known_id():
    piece = piece_by_id("VCA-ALH-VIN-MOP-PEN-001")
    assert piece is not None
    assert piece.collection.value == "alhambra"


def test_heritage_loads():
    docs = load_heritage()
    assert len(docs) >= 8
    assert all(d.id.startswith("HER-") for d in docs)


def test_heritage_lookup_known_id():
    doc = heritage_by_id("HER-001-ALHAMBRA-1968")
    assert doc is not None
    assert doc.year == 1968
