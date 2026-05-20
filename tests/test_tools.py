from datetime import UTC, datetime, timedelta

from maison_concierge.models import AppointmentRequest
from maison_concierge.tools import book_appointment, flag_high_value_lead, pricing_lookup


def test_book_appointment_writes_record(tmp_path, monkeypatch):
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    from maison_concierge.config import get_settings

    get_settings.cache_clear()
    # Redirect data dir to tmp_path so the .jsonl write is isolated
    settings = get_settings()
    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: tmp_path))

    request = AppointmentRequest(
        client_name="Mme Berthier",
        preferred_at=datetime.now(UTC) + timedelta(days=1),
    )
    appointment = book_appointment(request)
    assert appointment.reference.startswith("VCA-APT-")
    assert appointment.advisor_assigned

    path = tmp_path / "appointments.jsonl"
    assert path.exists()


def test_pricing_lookup_known_piece():
    response = pricing_lookup("VCA-ALH-VIN-MOP-PEN-001")
    assert response is not None
    assert response["available"] is True
    assert response["price_chf"] > 0


def test_pricing_lookup_unknown():
    assert pricing_lookup("VCA-NOPE") is None


def test_flag_high_value_lead_tiers(tmp_path, monkeypatch):
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics"))
    from maison_concierge.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: tmp_path))

    hot = flag_high_value_lead(
        "c1",
        "Looking for an investment piece, private viewing at residence, archive of high jewelry",
    )
    assert hot.tier == "hot"

    watch = flag_high_value_lead("c2", "just browsing")
    assert watch.tier == "watch"
