# Maison Concierge

A multi-agent GenAI client advisor for luxury maisons — discover pieces, share heritage stories, and book appointments in English or French.

Built around a small, opinionated stack: **LangGraph** for orchestration, the **Anthropic Claude API** (Sonnet 4.6) for conversation, **ChromaDB** for catalog + heritage retrieval, **OpenCLIP** for visual similarity, and **Streamlit** for the chat surface and stakeholder dashboard.

> The catalog and heritage data are entirely synthetic. They reference real collection names from the world of haute joaillerie to make the demo feel grounded, but no real pricing, references, or archival material is used.

## Architecture

```
                    ┌──────────────────────────┐
                    │  Streamlit chat (EN/FR)  │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  Orchestrator — LangGraph    │
                  │  • intent classification     │
                  │  • parallel retrieval        │
                  │  • reply composition         │
                  └──┬─────────┬──────────┬──────┘
                     │         │          │
            ┌────────▼──┐ ┌────▼─────┐ ┌──▼────────┐
            │ Catalog   │ │ Heritage │ │ Visual    │
            │ (Chroma)  │ │ (Chroma) │ │ (OpenCLIP)│
            └───────────┘ └──────────┘ └───────────┘
                                 │
                ┌────────────────▼─────────────────┐
                │  Stakeholder dashboard            │
                │  conversation quality • intents   │
                │  escalation reasons • lead tiers  │
                └───────────────────────────────────┘
```

## Quick start

```bash
# 1. Install
pip install -e ".[app,dev]"

# 2. Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY

# 3. Build the vector stores (one-off)
python scripts/seed.py

# 4. Run the chat UI
streamlit run apps/chat.py

# 5. (Optional) Run the dashboard in a second terminal
streamlit run apps/dashboard.py --server.port 8502
```

For Redis-backed conversation memory and a containerised stack:

```bash
docker compose up --build
```

## Configuration

All runtime configuration is read from environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required to call the Claude API |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Any current Claude model ID |
| `CHROMA_DIR` | `.chroma` | Persistent vector store path |
| `REDIS_URL` | `redis://localhost:6379/0` | Falls back to in-memory if unreachable |
| `APP_LOCALE_DEFAULT` | `en` | `en` or `fr` |
| `APP_ESCALATION_THRESHOLD` | `0.65` | Intent-confidence floor before escalating |
| `CLIP_ENABLED` | `false` | Enable OpenCLIP visual search (heavy deps) |

## Project layout

```
maison_concierge/
├── agents/         LangGraph orchestrator + composer + intent classifier
├── retrieval/      ChromaDB-backed catalog and heritage RAG, plus CLIP visual search
├── tools/          Mock booking, pricing lookup, HNW lead flagging
├── memory/         Redis-or-in-memory conversation store
├── observability/  Event log and aggregated dashboard snapshot
├── models/         Pydantic domain models (catalog, conversation, tools)
└── i18n/           EN/FR string tables and lightweight locale detection
apps/
├── chat.py         Streamlit chat surface
└── dashboard.py    Stakeholder dashboard
data/
├── catalog/        Synthetic Alhambra / Frivole / Ballerinas / Lucky Animals catalog
└── heritage/       Synthetic heritage stories (EN + FR)
```

## How it works

Each turn flows through the orchestrator graph:

1. **Classify** — Claude returns a structured `IntentResult` (intent, confidence, locale, routing flags). Confidence below the escalation threshold short-circuits to an apologetic handoff.
2. **Retrieve** — The orchestrator fans out to whichever sub-graphs the intent requires: catalog RAG, heritage RAG, visual similarity. These run in parallel.
3. **Compose** — A single Claude call assembles the evidence into the maison's voice. The system prompt is prompt-cached so every turn after the first reuses ~1 KB of cached prefix.

The orchestrator records intents, escalations, tool calls, and appointments to a JSONL event log that the dashboard reads directly. The chat surface and the dashboard are independent processes — either runs without the other.

## Tests

```bash
pytest                  # unit tests (no API key required)
ruff check src tests    # lint
mypy src                # type check
```

CI runs all three on Python 3.11 and 3.12 (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Acknowledgements & disclaimer

This project is a portfolio demonstration. It is not affiliated with, endorsed by, or representative of any actual luxury maison. Collection names (Alhambra, Frivole, Ballerinas, Lucky Animals, Perlée) are referenced because they are part of the cultural vocabulary of haute joaillerie — but every piece, price, advisor, and heritage extract in this repository is fictional.
