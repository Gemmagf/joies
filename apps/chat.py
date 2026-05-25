"""Streamlit chat surface for the Maison Concierge.

Run locally:
    streamlit run apps/chat.py
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import streamlit as st

from maison_concierge.agents import Orchestrator
from maison_concierge.config import get_settings
from maison_concierge.i18n import t
from maison_concierge.memory import get_store
from maison_concierge.models import ChatMessage, ConversationState
from maison_concierge.observability import Event, get_recorder

st.set_page_config(
    page_title="Maison Concierge",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main { background-color: #faf8f4; }
    .stApp { font-family: "Garamond", "Cormorant Garamond", serif; }
    h1, h2, h3 { font-family: "Garamond", "Cormorant Garamond", serif; letter-spacing: 0.04em; }
    .stChatMessage { border-radius: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _orchestrator() -> Orchestrator:
    return Orchestrator()


def _ensure_state() -> ConversationState:
    if "conversation" not in st.session_state:
        conv_id = str(uuid.uuid4())
        state = ConversationState(conversation_id=conv_id)
        st.session_state["conversation"] = state
        get_recorder().record(Event(type="conversation_started", conversation_id=conv_id))
    return cast(ConversationState, st.session_state["conversation"])


def _render_message(msg: ChatMessage) -> None:
    avatar = "◆" if msg.role == "assistant" else "•"
    with st.chat_message(msg.role, avatar=avatar):
        st.markdown(msg.content)
        if msg.citations:
            with st.expander("Sources" if msg.locale == "en" else "Patrimoine"):
                for c in msg.citations:
                    st.caption(f"[{c.document_id}] — {c.snippet}")


def main() -> None:
    settings = get_settings()
    state = _ensure_state()

    locale = state.locale
    st.title(t("app.title", locale))
    st.caption(t("app.tagline", locale))

    if _orchestrator().demo_mode:
        st.info(
            "**Demo mode** — replies are produced by a rule-based intent classifier and "
            "templated EN/FR composer over the real retrieval layer (BM25 + dense + RRF). "
            "Add `ANTHROPIC_API_KEY` to `.env` to switch to live Claude Sonnet 4.6 generation.  \n"
            "*In a real production deployment, this would step up to Claude Opus 4.7 (or another "
            "frontier model) hosted on AWS Bedrock / Google Vertex AI, with EU data residency, "
            "fine-tuned vision over the maison's archive, and multi-language support beyond EN/FR. "
            "See the README — Scaling beyond the demo.*"
            if locale == "en"
            else "**Mode démo** — les réponses sont produites par un classificateur d'intention à règles "
            "et un composeur EN/FR templaté, au-dessus de la couche de recherche réelle "
            "(BM25 + dense + RRF). Ajoutez `ANTHROPIC_API_KEY` à `.env` pour activer Claude Sonnet 4.6.  \n"
            "*En production réelle, le système basculerait sur Claude Opus 4.7 (ou un modèle "
            "équivalent) hébergé sur AWS Bedrock / Google Vertex AI, avec résidence des données en "
            "Europe, vision affinée sur les archives de la maison, et support de langues au-delà "
            "de EN/FR. Voir le README — Scaling beyond the demo.*",
            icon="◆",
        )

    with st.sidebar:
        st.markdown("### Settings")
        new_locale = st.selectbox(
            "Locale",
            options=["en", "fr"],
            index=0 if locale == "en" else 1,
            format_func=lambda x: "English" if x == "en" else "Français",
        )
        if new_locale != locale:
            state.locale = new_locale  # type: ignore[assignment]
        uploaded = st.file_uploader(
            t("chat.upload_image", state.locale),
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )
        st.divider()
        st.caption(f"Model: {settings.claude_model}")
        st.caption(f"Conversation: {state.conversation_id[:8]}…")

    for message in state.messages:
        _render_message(message)

    user_input = st.chat_input(t("chat.placeholder", state.locale))
    if not user_input:
        return

    image_bytes = uploaded.getvalue() if uploaded is not None else None

    user_msg = ChatMessage(role="user", content=user_input, locale=state.locale)
    state.messages.append(user_msg)
    _render_message(user_msg)

    with st.chat_message("assistant", avatar="◆"):
        with st.spinner("…"):
            result = _orchestrator().run(
                conversation_id=state.conversation_id,
                user_message=user_input,
                locale=state.locale,
                image_bytes=image_bytes,
            )
        reply_text = result.get("assistant_reply", "")
        citations = result.get("citations", [])
        st.markdown(reply_text)

        catalog_hits = result.get("catalog_hits") or []
        if catalog_hits:
            with st.expander("Catalog matches"):
                for hit in catalog_hits:
                    name = hit.piece.display_name(state.locale)
                    st.markdown(
                        f"**{name}** — `{hit.piece.id}` · CHF {hit.piece.price_chf:,.0f} · "
                        f"score {hit.score:.2f}"
                    )

        visual_hits = result.get("visual_hits") or []
        if visual_hits:
            label = "Visual matches" if not visual_hits[0].degraded else "Catalog (text fallback)"
            with st.expander(label):
                for hit in visual_hits:
                    st.markdown(
                        f"**{hit.piece.display_name(state.locale)}** — `{hit.piece.id}` · "
                        f"CHF {hit.piece.price_chf:,.0f}"
                    )

        trace = result.get("trace") or {}
        if trace:
            with st.expander("Why this reply (debug trace)"):
                intent_info = trace.get("intent", {})
                catalog_info = trace.get("catalog", {})
                heritage_info = trace.get("heritage", {})
                compose_info = trace.get("compose", {})
                groundedness_info = trace.get("groundedness", {})

                if intent_info:
                    st.markdown(
                        f"**Intent** · `{intent_info.get('detected', '?')}` "
                        f"· confidence `{intent_info.get('confidence', 0):.2f}` "
                        f"· locale `{intent_info.get('locale', '?')}` "
                        f"· mode `{intent_info.get('mode', '?')}` "
                        f"{'· **escalated**' if intent_info.get('escalate') else ''}"
                    )
                if catalog_info and not catalog_info.get("skipped"):
                    st.markdown(f"**Catalog retrieval** · {catalog_info.get('retriever', '?')}")
                    cols_t = st.columns(3)
                    cols_t[0].caption("BM25 top-5")
                    for pid in catalog_info.get("bm25_top", []):
                        cols_t[0].markdown(f"- `{pid}`")
                    cols_t[1].caption("Dense top-5")
                    for pid in catalog_info.get("dense_top", []):
                        cols_t[1].markdown(f"- `{pid}`")
                    cols_t[2].caption("RRF-fused top-5")
                    for pid in catalog_info.get("fused_top", []):
                        cols_t[2].markdown(f"- `{pid}`")
                    st.caption(f"latency: {catalog_info.get('latency_ms', 0)} ms")
                if heritage_info and not heritage_info.get("skipped"):
                    st.markdown(
                        f"**Heritage retrieval** · {heritage_info.get('retriever', '?')} · "
                        f"{heritage_info.get('n_hits', 0)} hits · "
                        f"{heritage_info.get('latency_ms', 0)} ms"
                    )
                    if heritage_info.get("top_ids"):
                        st.code("\n".join(heritage_info["top_ids"]), language="text")
                if compose_info:
                    st.markdown(
                        f"**Composer** · {compose_info.get('mode', '?')} "
                        f"· template intent `{compose_info.get('intent', '?')}` "
                        f"· {compose_info.get('reply_length_chars', 0)} chars "
                        f"· {compose_info.get('n_citations', 0)} citations "
                        f"· {compose_info.get('latency_ms', 0)} ms"
                    )
                if groundedness_info:
                    is_grounded = groundedness_info.get("is_grounded", True)
                    badge = "PASS" if is_grounded else f"FAIL ({groundedness_info.get('hallucination_count', 0)})"
                    st.markdown(
                        f"**Groundedness** · {badge}"
                        + (
                            f" · findings: {groundedness_info.get('findings_by_kind', {})}"
                            if groundedness_info.get("findings_by_kind")
                            else ""
                        )
                    )

    assistant_msg = ChatMessage(
        role="assistant",
        content=reply_text,
        locale=state.locale,
        citations=list(citations),
        timestamp=datetime.now(UTC),
    )
    state.messages.append(assistant_msg)

    if result.get("escalate"):
        state.escalated = True
        state.escalation_reason = result.get("escalation_reason")

    get_store().save(state)


if __name__ == "__main__":
    main()
