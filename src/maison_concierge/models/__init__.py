"""Pydantic domain models for catalog, clients, conversation and observability."""

from .catalog import CatalogPiece, Collection, HeritageDocument
from .client import ClientProfile, ClientSegment
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
    "ClientProfile",
    "ClientSegment",
    "Collection",
    "ConversationState",
    "EscalationReason",
    "HeritageDocument",
    "Role",
]
