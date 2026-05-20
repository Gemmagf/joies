"""Stakeholder dashboard — non-technical view of conversation quality and demand signals.

Run locally:
    streamlit run apps/dashboard.py --server.port 8502
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from maison_concierge.i18n import t
from maison_concierge.observability import get_recorder

st.set_page_config(
    page_title="Maison Concierge — Dashboard",
    page_icon="◆",
    layout="wide",
)


def main() -> None:
    locale = st.sidebar.selectbox("Locale", ["en", "fr"], index=0)
    st.title(t("dashboard.title", locale))

    recorder = get_recorder()
    snapshot = recorder.snapshot()
    events = recorder.load_events()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversations", snapshot.total_conversations)
    col2.metric(
        "Escalation rate",
        f"{snapshot.escalation_rate:.0%}" if snapshot.total_conversations else "—",
    )
    col3.metric("Appointments booked", snapshot.appointments_booked)
    col4.metric(
        "Avg rating",
        f"{snapshot.avg_rating:.1f}/5" if snapshot.avg_rating is not None else "—",
    )

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader(t("dashboard.intents", locale))
        if snapshot.intents:
            intents_df = (
                pd.DataFrame(
                    {"intent": list(snapshot.intents), "count": list(snapshot.intents.values())}
                )
                .sort_values("count", ascending=False)
                .set_index("intent")
            )
            st.bar_chart(intents_df)
        else:
            st.caption("No intents recorded yet.")

    with right:
        st.subheader(t("dashboard.escalations", locale))
        if snapshot.escalation_reasons:
            esc_df = (
                pd.DataFrame(
                    {
                        "reason": list(snapshot.escalation_reasons),
                        "count": list(snapshot.escalation_reasons.values()),
                    }
                )
                .sort_values("count", ascending=False)
                .set_index("reason")
            )
            st.bar_chart(esc_df)
        else:
            st.caption("No escalations recorded yet.")

    st.divider()
    st.subheader("HNW leads by tier")
    if snapshot.leads_by_tier:
        st.dataframe(
            pd.DataFrame.from_dict(snapshot.leads_by_tier, orient="index", columns=["count"])
        )
    else:
        st.caption("No leads flagged yet.")

    st.subheader("Recent events")
    if events:
        df = pd.DataFrame(
            [
                {
                    "ts": e.timestamp.isoformat(timespec="seconds"),
                    "type": e.type,
                    "conversation": e.conversation_id[:8],
                    "payload": e.payload,
                }
                for e in events[-50:][::-1]
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("Event log is empty. Start a chat to populate it.")


if __name__ == "__main__":
    main()
