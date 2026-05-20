from pathlib import Path

from maison_concierge.models import ClientIntent, EscalationReason
from maison_concierge.observability import Event, MetricsRecorder


def test_recorder_snapshot(tmp_path: Path):
    recorder = MetricsRecorder(tmp_path / "events.jsonl")
    recorder.record(Event(type="conversation_started", conversation_id="c1"))
    recorder.record(Event(type="conversation_started", conversation_id="c2"))
    recorder.record_intent("c1", ClientIntent.GIFT, 0.9)
    recorder.record_intent("c2", ClientIntent.HERITAGE_INQUIRY, 0.8)
    recorder.record_escalation("c1", EscalationReason.LOW_CONFIDENCE)
    recorder.record_rating("c2", 5)

    snapshot = recorder.snapshot()
    assert snapshot.total_conversations == 2
    assert snapshot.escalation_rate == 0.5
    assert snapshot.intents["gift"] == 1
    assert snapshot.escalation_reasons["low_confidence"] == 1
    assert snapshot.avg_rating == 5.0
