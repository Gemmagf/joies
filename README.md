# Maison Concierge

> A bilingual (EN / FR) multi-agent GenAI client advisor for a luxury jewelry maison — built end-to-end with the analytics, business case, and evaluation framework a hiring manager would expect from a real production deployment.

[![tests](https://img.shields.io/badge/tests-58%20passing-1c6e1c?style=flat-square)](tests/)
[![ruff](https://img.shields.io/badge/ruff-clean-1c6e1c?style=flat-square)](https://github.com/astral-sh/ruff)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?style=flat-square)](pyproject.toml)
[![model](https://img.shields.io/badge/model-claude--sonnet--4--6-8a6f3f?style=flat-square)](https://platform.claude.com)

---

## What it is

A working private-advisor system in the voice of a senior salon advisor: discovers pieces from a catalog, tells verifiable heritage stories with citations, books private appointments, and routes high-net-worth leads to the human team — in **English or French**, switching mid-conversation when the client does.

Built around an opinionated stack: **LangGraph** (orchestration) · **Anthropic Claude API** (Sonnet 4.6, prompt-cached, adaptive thinking) · **ChromaDB** (dense RAG) · **rank-bm25** (lexical RAG, fused via RRF) · **OpenCLIP** (visual similarity) · **scikit-learn** (lead scorer) · **Streamlit** (chat surface + analytics dashboard).

> Catalog, heritage stories, client profiles and prices are entirely synthetic. They reference real collection names from the world of haute joaillerie to make the demo feel grounded — but no real maison data is used.

---

## Why it's interesting

Most LLM chatbot projects stop at "it works on a happy path." This one was built to answer the three questions a senior data scientist will ask:

1. **Does it actually work?** → an **eval framework** with a 30-query bilingual golden set, deterministic groundedness checks, retrieval metrics (recall@k, MRR) over **dense vs hybrid (BM25 + dense + RRF)**, and confidence calibration with ECE / Brier score.
2. **Is it worth the money?** → a **quantified business case** with named, defensible assumptions and a sensitivity matrix — answers "what's the annual saving, at what deflection rate, and where's the break-even?"
3. **Do you understand the clients?** → a **4-tier segmentation** with 50 synthetic profiles, segment-aware prompt strategy, and a sklearn lead-scoring model so SAs can prioritize follow-ups.

---

## Headline results

### Retrieval — hybrid BM25 + dense closes the catalog gap

The dense-only retrieval struggled on a small product catalog where exact tokens (materials, stones, collection names) carry most of the signal. Adding BM25 in parallel and fusing with Reciprocal Rank Fusion (Cormack et al., 2009) recovered the missing recall:

| Metric | Dense only | Hybrid (BM25 + dense, RRF) | Lift |
|---|---:|---:|---:|
| recall@1 | 0.14 | **0.19** | **+33%** |
| recall@3 | 0.31 | **0.32** | +3% |
| recall@5 | 0.38 | **0.48** | **+27%** |
| MRR | 0.45 | **0.53** | **+18%** |

*Heritage retrieval is dense-only — at recall@5 = 0.93 / MRR = 0.89 there's no gap for BM25 to close.*

### Groundedness — zero tolerance for fabricated references

A deterministic post-hoc parser walks every assistant reply, extracts each `VCA-XXX` reference and `CHF X,XXX` claim, and verifies them against the retrieval evidence + the underlying catalog. Findings are bucketed into `unsupported_piece`, `unsupported_price`, `unsupported_heritage`. **0 hallucinations on the validated sample, 2 detected on a planted-failure sample.**

### Calibration — the intent classifier is slightly overconfident

| Metric | Value |
|---|---|
| Expected Calibration Error (ECE) | **0.124** |
| Brier score | **0.228** |
| N predictions | 200 |

Reliability diagram renders in the dashboard. Honest finding: out-of-the-box LLM intent classifiers are mildly miscalibrated at the high end (0.9-confidence predictions are right ~82% of the time).

### Business case — quantified ROI

At 2,500 conversations/month and a 64% blended deflection rate:

| Metric | Value |
|---|---:|
| **Annual net saving** | **CHF 176K** |
| Monthly net saving | CHF 14,676 |
| API cost per conversation (Sonnet 4.6 + 85% cache hit) | CHF 0.026 |
| SA cost per handled inquiry (Geneva fully-loaded) | CHF 9.17 |
| **Break-even deflection rate** | **0.28%** |
| Payback on CHF 15K setup | **1 month** |

Sensitivity matrix (annual net saving, CHF):

| Monthly conversations | 30% defl | 45% defl | 60% defl | 75% defl | 90% defl |
|---:|---:|---:|---:|---:|---:|
| 500 | 16,345 | 24,595 | 32,845 | 41,095 | 49,345 |
| 1,000 | 32,691 | 49,191 | 65,691 | 82,191 | 98,691 |
| 2,500 | 81,727 | 122,977 | 164,227 | 205,477 | 246,727 |
| 5,000 | 163,454 | 245,954 | 328,454 | 410,954 | 493,454 |
| 10,000 | 326,907 | 491,907 | 656,907 | 821,907 | 986,907 |

### Lead scoring — sklearn pipeline for SA prioritization

Logistic Regression on 1,500 synthetic conversations, with segment-aware features:

| Metric | Value |
|---|---|
| ROC AUC | 0.61 |
| Average Precision | 0.59 |
| Per-segment precision (collector) | **0.83** |
| Per-segment precision (VIP) | 0.64 |

Top coefficients are interpretable: `intent=appointment` (+0.76), `intent=heritage_inquiry` (−0.55), `n_turns` (+0.29), `segment=prospect` (−0.30).

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │  Streamlit chat  (EN / FR)   │
                    └───────────────┬──────────────┘
                                    │
                  ┌─────────────────▼──────────────────┐
                  │  Orchestrator — LangGraph           │
                  │  • intent classification (parse)    │
                  │  • parallel retrieval               │
                  │  • prompt-cached compose            │
                  └──┬──────────────┬───────────────┬──┘
                     │              │               │
            ┌────────▼──┐ ┌─────────▼─────────┐ ┌──▼────────┐
            │ Catalog   │ │ Heritage          │ │ Visual    │
            │ BM25+dense│ │ dense (Chroma)    │ │ CLIP      │
            │ + RRF     │ │                   │ │ (opt.)    │
            └─────┬─────┘ └─────────┬─────────┘ └──────┬────┘
                  │                 │                   │
                  └─────────────────┼───────────────────┘
                                    │
            ┌───────────────────────▼───────────────────────┐
            │  Stakeholder dashboard (4 tabs)                │
            │  Overview · Quality · Business case · Segments │
            └────────────────────────────────────────────────┘
                                    ▲
                                    │
        ┌──────────────────────┬────┴─────┬──────────────────────┐
        │ Eval framework        │ Business │ Segmentation + sklearn│
        │ (retrieval, ground-   │ case ROI │ booking-probability   │
        │  edness, calibration) │  model   │ logistic regression   │
        └──────────────────────┴──────────┴──────────────────────┘
```

---

## Quick start

```bash
# 1. Install
pip install -e ".[app,dev]"

# 2. Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY

# 3. Seed data + indices (one-off, no API key needed)
python scripts/seed.py                    # ChromaDB catalog + heritage
python scripts/seed_client_profiles.py    # 50 synthetic profiles
python scripts/train_booking_model.py     # sklearn lead-scoring model
python scripts/run_eval.py                # eval report for dashboard
python scripts/seed_demo_metrics.py       # synthetic event log
python scripts/seed_eval_calibration.py   # synthetic calibration pairs

# 4. Run the chat surface (needs ANTHROPIC_API_KEY)
streamlit run apps/chat.py

# 5. Run the dashboard (no API key needed)
streamlit run apps/dashboard.py --server.port 8502

# 6. Print the business case
python scripts/business_case_report.py
```

For Redis-backed conversation memory and a containerized stack:

```bash
docker compose up --build
```

---

## Project layout

```
src/maison_concierge/
├── agents/         LangGraph orchestrator + composer + intent classifier
├── retrieval/      catalog (dense + BM25 hybrid), heritage RAG, CLIP visual
├── eval/           golden set, recall@k/MRR, groundedness, calibration
├── analysis/       ROI model + sensitivity matrix
├── segmentation/   client profiles, segment guidance, sklearn lead scorer
├── tools/          mock booking, pricing, HNW lead flagging
├── memory/         Redis or in-memory conversation store
├── observability/  event log + dashboard snapshots
├── models/         Pydantic domain models
└── i18n/           EN/FR string tables + locale detection

apps/
├── chat.py         Streamlit chat surface
└── dashboard.py    Stakeholder dashboard (4 tabs)

data/
├── catalog/        60 synthetic pieces (5 collections, EN+FR)
├── heritage/       10 heritage stories (EN+FR)
├── clients/        50 client profiles + trained booking model
└── eval/           30-query golden set + eval reports
```

---

## How it works

Each turn flows through the orchestrator graph:

1. **Classify** — Claude returns a structured `IntentResult` via `messages.parse()` (intent, confidence, locale, routing flags). Confidence below the escalation threshold short-circuits to a graceful human handoff.
2. **Retrieve** — The orchestrator fans out to whichever sub-graphs the intent requires: catalog RAG (hybrid), heritage RAG (dense), visual similarity (CLIP, optional). These run in parallel.
3. **Compose** — A single Claude call assembles the evidence into the maison's voice. The system prompt is **prompt-cached** (~1.5K tokens × 5 turns × 85% cache hit) so every turn after the first reuses the cached prefix.

The orchestrator emits a structured event log; the dashboard reads it directly so it renders even with the chat surface stopped.

---

## Tests

```bash
pytest                  # 58 tests, ~30 seconds
ruff check src tests    # lint
mypy src                # type check
```

CI runs all three on Python 3.11 and 3.12 — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Acknowledgements & disclaimer

This project is a portfolio demonstration. It is not affiliated with, endorsed by, or representative of any actual luxury maison. Collection names (Alhambra, Frivole, Ballerinas, Lucky Animals, Perlée) are referenced because they are part of the cultural vocabulary of haute joaillerie — every piece, price, advisor, heritage extract and client profile in this repository is fictional.
