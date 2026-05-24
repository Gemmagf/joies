"""Stakeholder dashboard for the Maison Concierge.

Four tabs:
  • Overview    — operational metrics (conversations, escalation rate, intents, leads)
  • Quality     — RAG retrieval metrics, groundedness, intent calibration
  • Business    — ROI model with assumption sliders and sensitivity matrix
  • Segments    — client segmentation, booking-probability model, lead scorer

Run locally:
    streamlit run apps/dashboard.py --server.port 8502
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from maison_concierge.analysis import (
    DEFAULT_ASSUMPTIONS,
    Assumptions,
    compute_business_case,
    sensitivity_matrix,
)
from maison_concierge.config import get_settings
from maison_concierge.observability import get_recorder
from maison_concierge.segmentation import load_profiles
from maison_concierge.segmentation.booking_model import load_booking_model

st.set_page_config(
    page_title="Maison Concierge — Dashboard",
    page_icon="◆",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #faf8f4; }
    .stApp h1, .stApp h2, .stApp h3 {
        font-family: "Garamond", "Cormorant Garamond", serif;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _eval_report() -> dict | None:
    path = get_settings().data_dir / "eval" / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _booking_model_paths() -> tuple[Path, Path]:
    base = get_settings().data_dir / "clients"
    return base / "booking_model.joblib", base / "booking_model_report.json"


# =========================================================================
# Overview
# =========================================================================


def render_overview() -> None:
    recorder = get_recorder()
    snapshot = recorder.snapshot()
    events = recorder.load_events()

    cols = st.columns(4)
    cols[0].metric("Conversations", snapshot.total_conversations)
    cols[1].metric(
        "Escalation rate",
        f"{snapshot.escalation_rate:.0%}" if snapshot.total_conversations else "—",
    )
    cols[2].metric("Appointments booked", snapshot.appointments_booked)
    cols[3].metric(
        "Avg rating",
        f"{snapshot.avg_rating:.1f}/5" if snapshot.avg_rating is not None else "—",
    )

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Top client intents")
        if snapshot.intents:
            df = pd.DataFrame(
                {"intent": list(snapshot.intents), "count": list(snapshot.intents.values())}
            ).sort_values("count", ascending=False)
            chart = (
                alt.Chart(df)
                .mark_bar(color="#8a6f3f")
                .encode(
                    x=alt.X("count:Q", title="Conversations"),
                    y=alt.Y("intent:N", sort="-x", title=None),
                    tooltip=["intent", "count"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No intents recorded yet.")
    with right:
        st.subheader("Escalation reasons")
        if snapshot.escalation_reasons:
            df = pd.DataFrame(
                {
                    "reason": list(snapshot.escalation_reasons),
                    "count": list(snapshot.escalation_reasons.values()),
                }
            )
            chart = (
                alt.Chart(df)
                .mark_bar(color="#3f2e1e")
                .encode(
                    x=alt.X("count:Q", title="Escalations"),
                    y=alt.Y("reason:N", sort="-x", title=None),
                    tooltip=["reason", "count"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No escalations recorded yet.")

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


# =========================================================================
# Quality — eval framework readout
# =========================================================================


def render_quality() -> None:
    report = _eval_report()
    if report is None:
        st.warning(
            "No eval report found. Run `python scripts/run_eval.py` to generate "
            "`data/eval/report.json`."
        )
        return

    st.caption(f"Eval run at `{report['timestamp']}`")

    st.subheader("Retrieval quality")
    cols = st.columns(2)
    for col, target_key in zip(cols, ["catalog_retrieval", "heritage_retrieval"], strict=True):
        m = report[target_key]
        with col:
            st.markdown(f"**{target_key.replace('_', ' ').title()}**  (n = {m['n_queries']})")
            metric_cols = st.columns(4)
            metric_cols[0].metric("recall@1", f"{m['recall_at_1']:.2f}")
            metric_cols[1].metric("recall@3", f"{m['recall_at_3']:.2f}")
            metric_cols[2].metric("recall@5", f"{m['recall_at_5']:.2f}")
            metric_cols[3].metric("MRR", f"{m['mrr']:.2f}")
            if m["per_intent"]:
                rows = [
                    {"intent": intent, **per_metrics}
                    for intent, per_metrics in m["per_intent"].items()
                ]
                df = pd.DataFrame(rows).set_index("intent")
                st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

    st.divider()
    st.subheader("Groundedness")
    samples = report.get("groundedness_samples", [])
    if samples:
        total = len(samples)
        grounded = sum(1 for s in samples if s["is_grounded"])
        st.metric(
            "Groundedness rate (smoke test)",
            f"{grounded}/{total}",
            help="Replies whose piece references, prices, and heritage citations all resolve to evidence.",
        )
        for sample in samples:
            badge = "PASS" if sample["is_grounded"] else "FAIL"
            with st.expander(f"{badge} — {sample['reply'][:80]}…", expanded=not sample["is_grounded"]):
                st.code(sample["reply"], language="text")
                if sample["findings"]:
                    st.dataframe(
                        pd.DataFrame(sample["findings"]),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.divider()
    st.subheader("Intent classifier calibration")
    calib = report.get("calibration")
    if not calib:
        st.caption(
            "No calibration data. Run `python scripts/seed_eval_calibration.py` to seed."
        )
        return
    cols = st.columns(3)
    cols[0].metric("ECE", f"{calib['ece']:.3f}", help="Expected Calibration Error — lower is better.")
    cols[1].metric("Brier score", f"{calib['brier']:.3f}")
    cols[2].metric("N predictions", calib["n_total"])

    bins = pd.DataFrame(calib["bins"])
    bins = bins[bins["n"] > 0]
    bins["midpoint"] = (bins["lower"] + bins["upper"]) / 2
    reliability = (
        alt.Chart(bins)
        .mark_line(point=True, color="#8a6f3f")
        .encode(
            x=alt.X("midpoint:Q", title="Predicted confidence", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("accuracy:Q", title="Empirical accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["midpoint", "accuracy", "n"],
        )
    )
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color="#888")
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart((diag + reliability).properties(height=320), use_container_width=True)
    st.caption(
        "Reliability diagram. Dots above the diagonal → underconfident; below → overconfident."
    )


# =========================================================================
# Business case
# =========================================================================


def render_business_case() -> None:
    st.markdown(
        "Quantified ROI model with every assumption named and editable. All currency in CHF."
    )

    with st.sidebar:
        st.markdown("### Business assumptions")
        monthly_vol = st.slider(
            "Monthly conversations", min_value=200, max_value=10_000, value=2_500, step=100
        )
        sa_cost = st.slider(
            "SA fully-loaded cost (CHF/year)",
            min_value=70_000,
            max_value=180_000,
            value=int(DEFAULT_ASSUMPTIONS.sa_cost_chf_per_year),
            step=5_000,
        )
        sa_minutes = st.slider(
            "Avg SA minutes per inquiry",
            min_value=3.0,
            max_value=25.0,
            value=float(DEFAULT_ASSUMPTIONS.sa_minutes_per_inquiry),
            step=0.5,
        )
        cache_hit = st.slider(
            "Prompt-cache hit rate",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_ASSUMPTIONS.cache_hit_rate),
            step=0.05,
        )
        custom_deflection = st.slider(
            "Override blended deflection (0 = use intent mix)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Leave at 0 to compute from the intent mix.",
        )

    assumptions = Assumptions(
        monthly_conversations=monthly_vol,
        sa_cost_chf_per_year=sa_cost,
        sa_minutes_per_inquiry=sa_minutes,
        cache_hit_rate=cache_hit,
    )
    case = compute_business_case(
        assumptions,
        deflection_rate_override=custom_deflection if custom_deflection > 0 else None,
    )

    headline = st.columns(4)
    headline[0].metric("Annual net saving", f"CHF {case.annual_net_saving_chf:,.0f}")
    headline[1].metric("Monthly net saving", f"CHF {case.monthly_net_saving_chf:,.0f}")
    headline[2].metric("Blended deflection", f"{case.blended_deflection_rate:.1%}")
    headline[3].metric("Break-even deflection", f"{case.break_even_deflection_rate:.2%}")

    st.divider()
    st.subheader("Unit economics")
    unit = pd.DataFrame(
        {
            "metric": [
                "SA effective cost",
                "SA cost per handled inquiry",
                "API cost per conversation",
                "Margin per deflected conversation",
                "SA hours saved per month",
                "Deflected conversations per month",
            ],
            "value": [
                f"CHF {assumptions.sa_hourly_cost_chf:,.2f} / hour",
                f"CHF {case.sa_cost_per_inquiry_chf:,.2f}",
                f"CHF {case.api_cost_per_conversation_chf:,.4f}",
                f"CHF {case.sa_cost_per_inquiry_chf - case.api_cost_per_conversation_chf:,.2f}",
                f"{case.sa_hours_saved_per_month:,.0f}",
                f"{case.deflected_conversations_per_month:,}",
            ],
        }
    )
    st.dataframe(unit, use_container_width=True, hide_index=True)

    st.subheader("Sensitivity — annual net saving by volume x deflection")
    grid = sensitivity_matrix(assumptions=assumptions)
    pivot = pd.DataFrame(
        [
            {
                "volume": c.monthly_conversations,
                "deflection": f"{c.deflection_rate:.0%}",
                "annual_chf": c.annual_net_saving_chf,
            }
            for c in grid
        ]
    )
    heatmap = (
        alt.Chart(pivot)
        .mark_rect()
        .encode(
            x=alt.X("deflection:O", title="Deflection rate"),
            y=alt.Y("volume:O", sort=alt.SortField("volume", order="descending"), title="Monthly conversations"),
            color=alt.Color(
                "annual_chf:Q",
                title="Annual saving (CHF)",
                scale=alt.Scale(scheme="goldorange"),
            ),
            tooltip=["volume", "deflection", alt.Tooltip("annual_chf:Q", format=",.0f")],
        )
        .properties(height=320)
    )
    st.altair_chart(heatmap, use_container_width=True)


# =========================================================================
# Segments
# =========================================================================


def render_segments() -> None:
    profiles = load_profiles()
    if not profiles:
        st.warning(
            "No client profiles found. Run `python scripts/seed_client_profiles.py`."
        )
        return

    df = pd.DataFrame(
        [
            {
                "id": p.id,
                "name": p.display_name,
                "segment": p.segment.value,
                "tenure_years": p.tenure_years,
                "lifetime_spend_chf": p.lifetime_spend_chf,
                "collections_owned": len(p.collections_owned),
                "preferred_language": p.preferred_language,
                "city": p.city,
            }
            for p in profiles
        ]
    )

    cols = st.columns(4)
    cols[0].metric("Clients", len(df))
    cols[1].metric("Avg lifetime spend", f"CHF {df['lifetime_spend_chf'].mean():,.0f}")
    cols[2].metric("VIP + Collector share", f"{(df['segment'].isin(['vip', 'collector'])).mean():.0%}")
    cols[3].metric("Avg tenure", f"{df['tenure_years'].mean():.1f} y")

    st.divider()
    st.subheader("Distribution by segment")
    segment_df = (
        df.groupby("segment")
        .agg(
            clients=("id", "count"),
            avg_spend=("lifetime_spend_chf", "mean"),
            total_spend=("lifetime_spend_chf", "sum"),
        )
        .reset_index()
    )

    left, right = st.columns(2)
    with left:
        chart = (
            alt.Chart(segment_df)
            .mark_bar(color="#8a6f3f")
            .encode(
                x=alt.X("segment:N", sort="-y"),
                y=alt.Y("clients:Q"),
                tooltip=["segment", "clients", alt.Tooltip("avg_spend:Q", format=",.0f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        chart = (
            alt.Chart(segment_df)
            .mark_bar(color="#3f2e1e")
            .encode(
                x=alt.X("segment:N", sort="-y"),
                y=alt.Y("total_spend:Q", title="Total lifetime spend (CHF)"),
                tooltip=["segment", alt.Tooltip("total_spend:Q", format=",.0f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Lifetime spend distribution")
    hist = (
        alt.Chart(df)
        .mark_bar(color="#8a6f3f")
        .encode(
            x=alt.X("lifetime_spend_chf:Q", bin=alt.Bin(maxbins=24), title="Lifetime spend (CHF)"),
            y=alt.Y("count():Q", title="Clients"),
            color=alt.Color("segment:N"),
            tooltip=["segment", alt.Tooltip("count():Q")],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, use_container_width=True)

    st.divider()
    st.subheader("Booking-probability model")
    model_path, report_path = _booking_model_paths()
    if not (model_path.exists() and report_path.exists()):
        st.warning(
            "Train the model first: `python scripts/train_booking_model.py`."
        )
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    perf = st.columns(4)
    perf[0].metric("ROC AUC", f"{report['roc_auc']:.3f}")
    perf[1].metric("Avg precision", f"{report['average_precision']:.3f}")
    perf[2].metric("Accuracy", f"{report['accuracy']:.3f}")
    perf[3].metric("N (train/test)", f"{report['n_train']}/{report['n_test']}")

    coef_left, coef_right = st.columns(2)
    with coef_left:
        st.markdown("**Per-segment precision / recall**")
        seg_metrics = pd.DataFrame(
            {
                "segment": list(report["per_segment_precision"].keys()),
                "precision": list(report["per_segment_precision"].values()),
                "recall": [
                    report["per_segment_recall"].get(seg, 0.0)
                    for seg in report["per_segment_precision"]
                ],
            }
        ).set_index("segment")
        st.dataframe(seg_metrics.style.format("{:.2f}"), use_container_width=True)
    with coef_right:
        st.markdown("**Top feature coefficients**")
        importance = pd.DataFrame(
            sorted(report["feature_importance"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:10],
            columns=["feature", "coefficient"],
        )
        chart = (
            alt.Chart(importance)
            .mark_bar()
            .encode(
                x="coefficient:Q",
                y=alt.Y("feature:N", sort="-x"),
                color=alt.condition(
                    alt.datum.coefficient > 0,
                    alt.value("#8a6f3f"),
                    alt.value("#3f2e1e"),
                ),
                tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Lead scorer — what-if")
    cols = st.columns(3)
    segment_choice = cols[0].selectbox("Segment", ["prospect", "established", "vip", "collector"], index=2)
    intent_choice = cols[1].selectbox(
        "Intent",
        ["gift", "celebration", "investment_piece", "appointment", "browse", "heritage_inquiry", "price_inquiry"],
        index=2,
    )
    n_turns = cols[2].slider("Conversation turns", 1, 12, 5)
    cols2 = st.columns(3)
    tenure = cols2[0].slider("Tenure (years)", 0.0, 30.0, 8.0)
    spend = cols2[1].slider("Lifetime spend (CHF)", 0, 2_000_000, 75_000, step=5_000)
    days_since_appt = cols2[2].slider("Days since last appointment", 0, 720, 90)
    intent_conf = st.slider("Intent confidence", 0.0, 1.0, 0.85)

    model = load_booking_model(model_path=model_path, report_path=report_path)
    if model is not None:
        scenario = pd.DataFrame(
            [
                {
                    "segment": segment_choice,
                    "intent": intent_choice,
                    "tenure_years": float(tenure),
                    "lifetime_spend_chf": float(spend),
                    "log_lifetime_spend_chf": float(np.log1p(spend)),
                    "preferred_language": "en",
                    "days_since_last_appointment": int(days_since_appt),
                    "n_turns": int(n_turns),
                    "intent_confidence": float(intent_conf),
                }
            ]
        )
        proba = float(model.predict_proba(scenario)[0])
        st.metric("Estimated booking probability", f"{proba:.1%}")
        if proba >= 0.5:
            st.success("Above 50% — SA should prioritise this lead.")
        else:
            st.info("Below 50% — keep nurturing, do not escalate yet.")


# =========================================================================
# App entry
# =========================================================================


def main() -> None:
    st.title("Maison Concierge — Stakeholder dashboard")
    st.caption(
        "Operational metrics, quality evaluation, ROI model and lead-scoring intelligence "
        "for the multi-agent client advisor."
    )

    overview_tab, quality_tab, business_tab, segments_tab = st.tabs(
        ["Overview", "Quality", "Business case", "Segments"]
    )
    with overview_tab:
        render_overview()
    with quality_tab:
        render_quality()
    with business_tab:
        render_business_case()
    with segments_tab:
        render_segments()


if __name__ == "__main__":
    main()
