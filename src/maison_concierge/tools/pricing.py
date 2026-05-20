"""Pricing lookup tool — returns CHF list price and a discreet availability hint."""

from __future__ import annotations

from typing import TypedDict

from ..data_loader import piece_by_id


class PricingResponse(TypedDict):
    piece_id: str
    price_chf: float
    available: bool
    note: str


def pricing_lookup(piece_id: str) -> PricingResponse | None:
    piece = piece_by_id(piece_id)
    if piece is None:
        return None
    available = not piece.is_high_jewelry
    note = (
        "Available in boutique on request."
        if available
        else "High jewelry — viewing arranged privately by an advisor."
    )
    return PricingResponse(
        piece_id=piece.id,
        price_chf=piece.price_chf,
        available=available,
        note=note,
    )
