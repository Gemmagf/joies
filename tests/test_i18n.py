from maison_concierge.i18n import detect_locale, t


def test_detect_french():
    assert detect_locale("Bonjour, je cherche un cadeau d'anniversaire") == "fr"


def test_detect_english():
    assert detect_locale("Hello, I am looking for an anniversary gift") == "en"


def test_detect_falls_back_to_default():
    assert detect_locale("hmm", default="fr") == "fr"


def test_strings_have_fallback():
    assert t("nonexistent.key") == "nonexistent.key"
    assert t("app.title", "fr") == "Maison Concierge"
