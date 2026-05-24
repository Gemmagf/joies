from maison_concierge.data_loader import load_catalog, load_heritage
from maison_concierge.eval.groundedness import check_groundedness


def test_valid_reply_is_grounded():
    piece = load_catalog()[0]
    heritage = load_heritage()[0]
    reply = (
        f"{piece.name.en} — {piece.id} (CHF {piece.price_chf:,.0f}) is iconic [{heritage.id}]."
    )
    report = check_groundedness(reply, catalog_evidence=[piece], heritage_evidence=[heritage])
    assert report.is_grounded
    assert report.hallucination_count == 0


def test_fabricated_piece_is_flagged():
    reply = "The Mythical Alhambra — VCA-FAKE-999 (CHF 12,345) is exclusive."
    report = check_groundedness(reply)
    kinds = {f.kind for f in report.findings}
    assert "unsupported_piece" in kinds
    assert not report.is_grounded


def test_wrong_price_is_flagged():
    piece = load_catalog()[0]
    wrong_price = piece.price_chf + 1_000
    reply = f"{piece.name.en} — {piece.id} (CHF {wrong_price:,.0f})."
    report = check_groundedness(reply, catalog_evidence=[piece])
    assert any(f.kind == "unsupported_price" for f in report.findings)


def test_fabricated_heritage_is_flagged():
    reply = "The maison's tradition is centuries old [HER-NOPE-1234]."
    report = check_groundedness(reply)
    assert any(f.kind == "unsupported_heritage" for f in report.findings)


def test_valid_id_not_in_evidence_is_missing_citation_not_hallucination():
    piece = load_catalog()[0]
    reply = f"{piece.id} is fine. (CHF {piece.price_chf:,.0f})"
    report = check_groundedness(reply, catalog_evidence=[])
    kinds = {f.kind for f in report.findings}
    assert "missing_citation" in kinds
    assert "unsupported_piece" not in kinds
    assert report.is_grounded  # missing_citation is not a hallucination
