"""Conversation state, intents and observability records."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "system", "tool"]


class ClientIntent(StrEnum):
    BROWSE = "browse"
    GIFT = "gift"
    INVESTMENT_PIECE = "investment_piece"
    CELEBRATION = "celebration"
    HERITAGE_INQUIRY = "heritage_inquiry"
    APPOINTMENT = "appointment"
    PRICE_INQUIRY = "price_inquiry"
    UNKNOWN = "unknown"


class EscalationReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    EXPLICIT_REQUEST = "explicit_request"
    HIGH_VALUE_LEAD = "high_value_lead"
    OUT_OF_SCOPE = "out_of_scope"
    NEGATIVE_SENTIMENT = "negative_sentiment"


class Citation(BaseModel):
    document_id: str
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: Role
    content: str
    locale: Literal["en", "fr"] = "en"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    citations: list[Citation] = Field(default_factory=list)


class AgentTrace(BaseModel):
    agent: str
    started_at: datetime
    finished_at: datetime
    input_summary: str
    output_summary: str
    confidence: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class ConversationState(BaseModel):
    conversation_id: str
    client_id: str | None = None
    locale: Literal["en", "fr"] = "en"
    messages: list[ChatMessage] = Field(default_factory=list)
    detected_intent: ClientIntent = ClientIntent.UNKNOWN
    intent_confidence: float = 0.0
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    traces: list[AgentTrace] = Field(default_factory=list)

    def last_user_message(self) -> ChatMessage | None:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None
