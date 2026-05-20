"""Bilingual EN/FR string tables and locale detection."""

from .detect import detect_locale
from .strings import STRINGS, t

__all__ = ["STRINGS", "detect_locale", "t"]
