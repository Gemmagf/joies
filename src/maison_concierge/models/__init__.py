"""Pydantic domain models for catalog, conversation and observability."""

from .catalog import CatalogPiece, Collection, HeritageDocument
from .conversation import (
    AgentTrace,
    ChatMessage,
    Citation,
    ClientIntent,
    ConversationState,
    EscalationReason,
    Role,
)
from .tools import Appointment, AppointmentRequest

__all__ = [
    "AgentTrace",
    "Appointment",
    "AppointmentRequest",
    "CatalogPiece",
    "ChatMessage",
    "Citation",
    "ClientIntent",
    "Collection",
    "ConversationState",
    "EscalationReason",
    "HeritageDocument",
    "Role",
]
