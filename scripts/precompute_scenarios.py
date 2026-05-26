"""Run every scenario through the orchestrator (offline) and freeze the responses.

The Scenarios tab in the dashboard plays back these pre-computed outputs so the demo
is deterministic and instant — no orchestrator call on every Play click.

Re-run this whenever the orchestrator behaviour changes (new intent rules, new
templates, retriever tuning).

Usage:
    python scripts/precompute_scenarios.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.agents import Orchestrator  # noqa: E402
from maison_concierge.config import get_settings  # noqa: E402


def _serialise_hit(hit) -> dict:
    if hasattr(hit, "piece"):
        return {
            "id": hit.piece.id,
            "name_en": hit.piece.name.en,
            "name_fr": hit.piece.name.fr,
            "price_chf": hit.piece.price_chf,
            "collection": hit.piece.collection.value,
            "score": round(hit.score, 3),
        }
    if hasattr(hit, "document"):
        return {
            "id": hit.document.id,
            "title_en": hit.document.title.en,
            "title_fr": hit.document.title.fr,
            "score": round(hit.score, 3),
            "snippet": hit.snippet,
        }
    return {"unknown": str(hit)}


def main() -> None:
    settings = get_settings()
    scenarios_path = settings.data_dir / "scenarios" / "scenarios.json"
    output_path = settings.data_dir / "scenarios" / "scenarios_played.json"

    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    print(f"Running {len(scenarios)} scenarios through the orchestrator (demo mode)…")

    orchestrator = Orchestrator()
    played = []

    for scenario in scenarios:
        print(f"  ▶ {scenario['id']} ({scenario['locale']})")
        turns = []
        conv_id = f"scenario-{scenario['id']}"
        for user_text in scenario["user_messages"]:
            state = orchestrator.run(
                conversation_id=conv_id,
                user_message=user_text,
                locale=scenario["locale"],
            )
            turns.append({"role": "user", "text": user_text})
            turns.append(
                {
                    "role": "assistant",
                    "text": state.get("assistant_reply", ""),
                    "intent": state["intent"].value if state.get("intent") else "unknown",
                    "confidence": round(float(state.get("intent_confidence", 0.0)), 3),
                    "locale": state.get("locale", scenario["locale"]),
                    "escalated": bool(state.get("escalate")),
                    "catalog_hits": [_serialise_hit(h) for h in state.get("catalog_hits", [])],
                    "heritage_hits": [_serialise_hit(h) for h in state.get("heritage_hits", [])],
                    "citations": [
                        {"document_id": c.document_id, "snippet": c.snippet, "relevance": round(c.relevance, 3)}
                        for c in state.get("citations", [])
                    ],
                    "trace": state.get("trace", {}),
                }
            )
        played.append({**scenario, "turns": turns})

    payload = {
        "precomputed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_scenarios": len(played),
        "scenarios": played,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Wrote {len(played)} pre-computed scenarios → {output_path}")


if __name__ == "__main__":
    main()
