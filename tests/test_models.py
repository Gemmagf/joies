from datetime import datetime, timedelta, timezone

import pytest

from maison_concierge.models import (
    AppointmentRequest,
    CatalogPiece,
    ChatMessage,
    ClientIntent,
    Collection,
    ConversationState,
)


def test_catalog_piece_localised_name():
    piece = CatalogPiece.model_validate(
        {
            "id": "VCA-TST-001",
            "collection": "alhambra",
            "category": "ring",
            "name": {"en": "Test ring", "fr": "Bague test"},
            "description": {"en": "EN desc", "fr": "FR desc"},
            "material": "yellow_gold",
            "price_chf": 1234.0,
            "year_introduced": 2020,
        }
    )
    assert piece.display_name("en") == "Test ring"
    assert piece.display_name("fr") == "Bague test"
    assert piece.collection is Collection.ALHAMBRA


def test_catalog_piece_rejects_bad_id():
    with pytest.raises(ValueError):
        CatalogPiece.model_validate(
            {
                "id": "bad-id",
                "collection": "alhambra",
                "category": "ring",
                "name": {"en": "x", "fr": "x"},
                "description": {"en": "x", "fr": "x"},
                "material": "yellow_gold",
                "price_chf": 1.0,
                "year_introduced": 2020,
            }
        )


def test_conversation_last_user_message():
    state = ConversationState(conversation_id="c1")
    state.messages.append(ChatMessage(role="user", content="hi"))
    state.messages.append(ChatMessage(role="assistant", content="bonjour"))
    state.messages.append(ChatMessage(role="user", content="je cherche"))
    last = state.last_user_message()
    assert last is not None
    assert last.content == "je cherche"


def test_appointment_request_defaults():
    req = AppointmentRequest(
        client_name="Mme Dubois",
        preferred_at=datetime.now(timezone.utc) + timedelta(days=2),
    )
    assert req.boutique.startswith("Geneva")
    assert req.channel == "boutique"
    assert req.locale == "en"


def test_client_intent_enum_string():
    assert ClientIntent.GIFT.value == "gift"
