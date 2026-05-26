"""Scenario gallery + playback panel.

Loads pre-computed scenarios (deterministic snapshots of orchestrator output) and
plays them back like a video: messages appear one by one with a typing pause, the
trace panel unfolds after the assistant reply. No live orchestrator call — clicks
are instant, the demo is identical every time.

Re-run `scripts/precompute_scenarios.py` to refresh the snapshots when the
orchestrator's behaviour changes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from ..config import get_settings

_SCENARIOS_PATH = "data/scenarios/scenarios_played.json"
_DEFAULT_DELAY_SEC = 1.6  # pause between user message and assistant reply


@st.cache_data
def _load_scenarios() -> dict | None:
    path: Path = get_settings().data_dir / "scenarios" / "scenarios_played.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _render_user_message(text: str) -> None:
    with st.chat_message("user", avatar="•"):
        st.markdown(text)


def _render_assistant_message(turn: dict) -> None:
    with st.chat_message("assistant", avatar="◆"):
        st.markdown(turn.get("text", ""))

        catalog_hits = turn.get("catalog_hits") or []
        if catalog_hits:
            with st.expander("Catalog matches"):
                for hit in catalog_hits:
                    name = hit.get("name_en") if turn.get("locale", "en") == "en" else hit.get("name_fr")
                    st.markdown(
                        f"**{name}** · `{hit['id']}` · CHF {hit['price_chf']:,.0f} · score {hit['score']:.2f}"
                    )

        heritage_hits = turn.get("heritage_hits") or []
        if heritage_hits:
            with st.expander("Heritage matches"):
                for hit in heritage_hits:
                    title = hit.get("title_en") if turn.get("locale", "en") == "en" else hit.get("title_fr")
                    st.markdown(f"**{title}** · `{hit['id']}` · score {hit['score']:.2f}")
                    st.caption(hit.get("snippet", ""))

        citations = turn.get("citations") or []
        if citations:
            with st.expander("Sources"):
                for c in citations:
                    st.caption(f"[{c['document_id']}] - {c['snippet']}")

        trace = turn.get("trace") or {}
        if trace:
            _render_trace(trace)


def _render_trace(trace: dict) -> None:
    intent_info = trace.get("intent", {})
    catalog_info = trace.get("catalog", {})
    heritage_info = trace.get("heritage", {})
    compose_info = trace.get("compose", {})
    groundedness_info = trace.get("groundedness", {})

    with st.expander("Why this reply (debug trace)"):
        if intent_info:
            esc = " · **escalated**" if intent_info.get("escalate") else ""
            st.markdown(
                f"**Intent** · `{intent_info.get('detected', '?')}` "
                f"· confidence `{intent_info.get('confidence', 0):.2f}` "
                f"· locale `{intent_info.get('locale', '?')}` "
                f"· mode `{intent_info.get('mode', '?')}`{esc}"
            )
        if catalog_info and not catalog_info.get("skipped"):
            st.markdown(f"**Catalog retrieval** · {catalog_info.get('retriever', '?')}")
            cols = st.columns(3)
            cols[0].caption("BM25 top-5")
            for pid in catalog_info.get("bm25_top", []):
                cols[0].markdown(f"- `{pid}`")
            cols[1].caption("Dense top-5")
            for pid in catalog_info.get("dense_top", []):
                cols[1].markdown(f"- `{pid}`")
            cols[2].caption("RRF-fused top-5")
            for pid in catalog_info.get("fused_top", []):
                cols[2].markdown(f"- `{pid}`")
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
            findings = groundedness_info.get("findings_by_kind", {})
            findings_str = f" · findings: {findings}" if findings else ""
            st.markdown(f"**Groundedness** · {badge}{findings_str}")


def _play_scenario(scenario: dict, *, delay_sec: float) -> None:
    """Render the full scenario with a brief pause between turns to feel video-like."""
    turns = scenario.get("turns", [])
    for i, turn in enumerate(turns):
        if turn["role"] == "user":
            _render_user_message(turn["text"])
        else:
            if delay_sec > 0:
                with st.spinner("..."):
                    time.sleep(delay_sec)
            _render_assistant_message(turn)
        # Small pause between rendered turns so the user actually sees the sequence.
        if delay_sec > 0 and i < len(turns) - 1 and turns[i]["role"] == "assistant":
            time.sleep(delay_sec * 0.4)


def render_scenarios_panel() -> None:
    payload = _load_scenarios()
    if payload is None:
        st.warning(
            "No pre-computed scenarios found. Run "
            "`python scripts/precompute_scenarios.py` to generate them."
        )
        return

    scenarios = payload.get("scenarios", [])
    st.caption(
        f"{len(scenarios)} curated scenarios · precomputed at "
        f"`{payload.get('precomputed_at', '?')}` · click any scenario to replay it like a short video."
    )

    selected_id = st.session_state.get("scenario_selected_id")

    if not selected_id:
        st.markdown("### Choose a scenario")
        cols_per_row = 2
        for row_start in range(0, len(scenarios), cols_per_row):
            row = scenarios[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, scenario in zip(cols, row, strict=False):
                with col, st.container(border=True):
                    st.markdown(f"**{scenario['title']}**")
                    st.caption(" · ".join(scenario.get("tags", [])))
                    st.write(scenario.get("description", ""))
                    if st.button("▶ Play", key=f"play_{scenario['id']}", use_container_width=True):
                        st.session_state["scenario_selected_id"] = scenario["id"]
                        st.session_state["scenario_delay_sec"] = _DEFAULT_DELAY_SEC
                        st.rerun()
        return

    # Detail view
    scenario = next((s for s in scenarios if s["id"] == selected_id), None)
    if scenario is None:
        st.session_state.pop("scenario_selected_id", None)
        st.rerun()
        return

    header_cols = st.columns([3, 1, 1])
    header_cols[0].markdown(f"### {scenario['title']}")
    speed = header_cols[1].selectbox(
        "Speed",
        ["1x", "2x", "Instant"],
        index=0,
        key=f"speed_{selected_id}",
        label_visibility="collapsed",
    )
    if header_cols[2].button("← Back to gallery", use_container_width=True):
        st.session_state.pop("scenario_selected_id", None)
        st.rerun()

    st.caption(" · ".join(scenario.get("tags", [])))
    st.write(scenario.get("description", ""))
    st.divider()

    delay_map = {"1x": _DEFAULT_DELAY_SEC, "2x": _DEFAULT_DELAY_SEC / 2, "Instant": 0.0}
    delay = delay_map[speed]

    play_cols = st.columns([1, 1, 6])
    play_clicked = play_cols[0].button("▶ Play", key=f"play_btn_{selected_id}", use_container_width=True)
    replay_clicked = play_cols[1].button("↻ Replay", key=f"replay_btn_{selected_id}", use_container_width=True)

    if play_clicked or replay_clicked:
        _play_scenario(scenario, delay_sec=delay)
    else:
        # Default: render the full conversation instantly so the page isn't empty
        for turn in scenario.get("turns", []):
            if turn["role"] == "user":
                _render_user_message(turn["text"])
            else:
                _render_assistant_message(turn)
        st.caption(
            "_Showing full conversation. Click Play to watch it animate, or change Speed._"
        )
