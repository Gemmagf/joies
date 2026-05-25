"""Standalone Streamlit chat surface.

If you want the chat + dashboard together in one app, run `streamlit run apps/dashboard.py`
instead — that's the canonical entry point for deployment.

Run locally:
    streamlit run apps/chat.py
"""

from __future__ import annotations

import streamlit as st

from maison_concierge.ui import render_chat_panel

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


def main() -> None:
    render_chat_panel(with_header=True)


if __name__ == "__main__":
    main()
