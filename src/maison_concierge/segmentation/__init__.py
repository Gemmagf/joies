"""Client segmentation: profile lookup, prompt personalization, lead scoring."""

from .booking_model import (
    BookingModel,
    BookingModelReport,
    load_booking_model,
    train_booking_model,
)
from .profiles import load_profiles, profile_by_id
from .segment import SEGMENT_GUIDANCE, segment_context_for_prompt

__all__ = [
    "SEGMENT_GUIDANCE",
    "BookingModel",
    "BookingModelReport",
    "load_booking_model",
    "load_profiles",
    "profile_by_id",
    "segment_context_for_prompt",
    "train_booking_model",
]
