"""Pluggable conversation store with Redis and in-memory backends.

The in-memory backend lets unit tests, local notebooks and Streamlit demos run without
Redis. Redis is used when REDIS_URL points at a reachable instance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, runtime_checkable

import redis
from redis.exceptions import RedisError

from ..config import get_settings
from ..models import ConversationState

_TTL_SECONDS = 60 * 60 * 24 * 7


@runtime_checkable
class ConversationStore(Protocol):
    def load(self, conversation_id: str) -> ConversationState | None: ...
    def save(self, state: ConversationState) -> None: ...
    def list_ids(self) -> list[str]: ...


class _InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, ConversationState] = {}

    def load(self, conversation_id: str) -> ConversationState | None:
        return self._data.get(conversation_id)

    def save(self, state: ConversationState) -> None:
        self._data[state.conversation_id] = state

    def list_ids(self) -> list[str]:
        return list(self._data)


class _RedisStore:
    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    def _key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}"

    def load(self, conversation_id: str) -> ConversationState | None:
        raw = self._client.get(self._key(conversation_id))
        if raw is None:
            return None
        return ConversationState.model_validate_json(raw)

    def save(self, state: ConversationState) -> None:
        self._client.setex(
            self._key(state.conversation_id),
            _TTL_SECONDS,
            state.model_dump_json(),
        )

    def list_ids(self) -> list[str]:
        return [k.decode("utf-8").removeprefix("conv:") for k in self._client.keys("conv:*")]


@lru_cache(maxsize=1)
def get_store() -> ConversationStore:
    settings = get_settings()
    try:
        client = redis.from_url(settings.redis_url, socket_connect_timeout=0.5)
        client.ping()
        return _RedisStore(client)
    except (RedisError, OSError):
        return _InMemoryStore()
