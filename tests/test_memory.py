from maison_concierge.memory.conversation import _InMemoryStore
from maison_concierge.models import ChatMessage, ConversationState


def test_in_memory_round_trip():
    store = _InMemoryStore()
    state = ConversationState(conversation_id="abc")
    state.messages.append(ChatMessage(role="user", content="bonjour", locale="fr"))
    store.save(state)

    loaded = store.load("abc")
    assert loaded is not None
    assert loaded.conversation_id == "abc"
    assert loaded.messages[0].content == "bonjour"
    assert "abc" in store.list_ids()


def test_missing_returns_none():
    store = _InMemoryStore()
    assert store.load("missing") is None
