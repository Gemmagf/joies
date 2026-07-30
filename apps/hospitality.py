"""Hospitality concierge — Streamlit demo.

Run locally:
    streamlit run apps/hospitality.py

Sidebar: guest persona (real distributions, invented identity).
Main: chat, with a per-message trace panel showing which agent fired.
"""

from __future__ import annotations

import streamlit as st

from maison_concierge.hospitality import (
    HospitalityOrchestrator,
    load_personas,
    persona_by_id,
)
from maison_concierge.hospitality.profile import ProfileAgent

st.set_page_config(
    page_title="Maison — Hospitality Concierge",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #faf8f4; }
    .stApp, .stMarkdown, .stChatMessage {
        font-family: "Garamond", "Cormorant Garamond", "EB Garamond", serif;
    }
    h1, h2, h3 { font-family: "Garamond", "Cormorant Garamond", serif;
                 letter-spacing: 0.04em; font-weight: 500; }
    .agent-chip {
        display: inline-block; padding: 2px 8px; margin-right: 6px;
        border: 1px solid #6c5b3f; border-radius: 12px; font-size: 12px;
        color: #6c5b3f; background: #f6f0e5;
    }
    .agent-chip.fired { background: #6c5b3f; color: #faf8f4; }
    .metric-tile {
        padding: 8px 12px; border: 1px solid #d9cdb3; border-radius: 4px;
        background: #fdfbf7; margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> HospitalityOrchestrator:
    orch = HospitalityOrchestrator()
    orch._kb.index()  # warm the retriever so first user turn isn't slow
    return orch


@st.cache_resource(show_spinner=False)
def get_profile_agent() -> ProfileAgent:
    return ProfileAgent()


@st.cache_data(show_spinner=False)
def get_personas():
    return load_personas()


def _persona_snapshot_card(persona_id: str) -> None:
    persona = persona_by_id(persona_id)
    if persona is None:
        st.warning("Persona not found.")
        return
    with st.container():
        st.markdown(f"### {persona.display_name}")
        st.markdown(
            f"*{persona.property}* · booking `{persona.booking_ref}`  \n"
            f"{persona.nationality} · party of {persona.party_size}"
        )
        st.markdown(persona.stay_summary)
        st.caption(
            "Synthetic identity; booking features drawn from real distributions in "
            "the Antonio et al 2019 hotel-bookings dataset."
        )

        st.markdown("---")
        st.markdown("**Profile snapshot**")
        try:
            profile_agent = get_profile_agent()
            snap = profile_agent.snapshot(persona.booking, persona_id=persona.id)
        except Exception as exc:
            st.error(f"Profile artifacts not loaded: {exc}")
            st.caption("Run `python scripts/train_analytics.py` from the repo root.")
            return

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cancellation risk", f"{snap.p_cancel:.0%}", snap.churn_band)
        col_b.metric("Value band", snap.value_band, f"€{snap.revenue_estimate:.0f}")
        col_c.metric("Segment", snap.segment)
        with st.expander("Top model drivers"):
            for name, share in snap.top_drivers[:5]:
                st.markdown(f"- `{name}` — {share:.0%}")
        with st.expander("Model notes on this booking"):
            for n in snap.notes:
                st.markdown(f"- {n}")


def _agent_chips(trace: dict) -> str:
    """Render one chip per agent, dimming those that did not fire this turn."""
    fired = {
        "Intent": True,
        "Profile": bool(trace.get("profile", {}).get("persona_id")),
        "Knowledge": bool(trace.get("knowledge", {}).get("n_hits", 0)),
        "Booking": bool(trace.get("booking", {}).get("action")),
        "Recommend": bool(trace.get("recommend", {}).get("n_ranked", 0)),
        "Compose": bool(trace.get("compose")),
        "Guardrail": bool(trace.get("guardrail")),
    }
    parts = [
        f'<span class="agent-chip{" fired" if v else ""}">{k}</span>'
        for k, v in fired.items()
    ]
    return "".join(parts)


def _trace_panel(trace: dict) -> None:
    st.markdown("**Which agents fired**", help="Nodes traversed by the LangGraph orchestrator this turn.")
    st.markdown(_agent_chips(trace), unsafe_allow_html=True)

    intent_info = trace.get("intent", {})
    if intent_info:
        st.markdown(
            f"**Intent** — `{intent_info.get('detected')}` "
            f"(conf {intent_info.get('confidence')}, {intent_info.get('mode')})"
        )
        if intent_info.get("matched_terms"):
            st.caption("Matched: " + ", ".join(intent_info["matched_terms"]))

    kn = trace.get("knowledge", {})
    if kn and not kn.get("skipped"):
        st.markdown(
            f"**Knowledge (RAG)** — {kn['n_hits']} hits in {kn['latency_ms']}ms "
            f"(scope: {kn.get('property_scope') or 'both'})"
        )
        for kb_id in kn.get("top_ids", []):
            st.caption(f"↳ {kb_id}")

    pr = trace.get("profile", {})
    if pr and not pr.get("skipped"):
        st.markdown(
            f"**Profile & Analytics** — p_cancel={pr['p_cancel']}, "
            f"band {pr['churn_band']}, value {pr['value_band']}, "
            f"segment {pr['segment']} ({pr['latency_ms']}ms)"
        )

    bk = trace.get("booking", {})
    if bk and not bk.get("skipped"):
        parts = [f"**Booking** — {bk['action']}"]
        if bk.get("refund_eur") is not None:
            parts.append(f"refund €{bk['refund_eur']:.0f}")
        if bk.get("requires_human"):
            parts.append("requires human")
        parts.append(f"({bk['latency_ms']}ms)")
        st.markdown(", ".join(parts))
        st.caption(f"↳ booking_ref {bk.get('booking_ref')}")

    rc = trace.get("recommend", {})
    if rc and not rc.get("skipped"):
        st.markdown(
            f"**Recommendation** — {rc['n_ranked']} experiences ranked "
            f"({rc['latency_ms']}ms)"
        )
        st.caption("signals: " + ", ".join(rc.get("signals", [])))
        for eid in rc.get("top_ids", []):
            st.caption(f"↳ {eid}")

    cp = trace.get("compose", {})
    if cp:
        st.markdown(
            f"**Compose** — {cp['mode']}, {cp['reply_length_chars']} chars, "
            f"{cp['n_citations']} citations ({cp['latency_ms']}ms)"
        )

    gr = trace.get("guardrail", {})
    if gr:
        badge = "✓ passed" if gr.get("passed") else "⚠ intervened"
        st.markdown(f"**Guardrail** — {badge} ({gr['latency_ms']}ms)")
        checks = gr.get("checks", {})
        for name, status in checks.items():
            icon = {"ok": "✓", "fired": "⚠", "n/a": "·"}.get(status, "?")
            st.caption(f"{icon} {name}: {status}")


def main() -> None:
    st.title("Maison — Hospitality Concierge")
    st.caption(
        "A multi-agent AI concierge for a fictional luxury maison, grounded in real "
        "hotel-booking data and a hand-written property knowledge base."
    )

    personas = get_personas()
    if not personas:
        st.error(
            "No personas found. Run `python scripts/generate_personas.py` from the "
            "repo root to bootstrap the demo."
        )
        return

    with st.sidebar:
        st.markdown("### Guest")
        options = {f"{p.display_name} — {p.property}": p.id for p in personas}
        choice = st.selectbox(
            "Choose a persona", options=list(options.keys()), index=0
        )
        persona_id = options[choice]
        st.session_state.persona_id = persona_id
        _persona_snapshot_card(persona_id)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.session_state.get("last_persona") != persona_id:
        st.session_state.messages = []
        st.session_state.last_persona = persona_id

    chat_col, trace_col = st.columns([2, 1], gap="large")

    with chat_col:
        st.markdown("#### Conversation")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Ask about your stay, your profile, or the property…")
        if user_msg:
            st.session_state.messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            orch = get_orchestrator()
            with st.spinner("The concierge is thinking…"):
                result = orch.run(
                    conversation_id=st.session_state.get("conversation_id", "demo-1"),
                    user_message=user_msg,
                    persona_id=persona_id,
                )
            reply = result.get("assistant_reply") or "…"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "trace": result.get("trace", {})}
            )

    with trace_col:
        st.markdown("#### Agent trace")
        last_asst = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
            None,
        )
        if last_asst and last_asst.get("trace"):
            _trace_panel(last_asst["trace"])
        else:
            st.caption("Send a message to see the routing and retrieval trace.")


if __name__ == "__main__":
    main()
