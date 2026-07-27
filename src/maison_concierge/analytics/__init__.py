"""Analytics spine — the models the Profile & Analytics agent actually calls.

Nothing in here is LLM-driven. These are trained on the real Antonio et al 2019
hotel bookings dataset (see `data/bookings/README.md`). The Profile agent is
allowed to *report* what these return; it is not allowed to invent risk numbers.
"""

from .churn import ChurnModel, ChurnReport, train_churn_model
from .clv import CLVBander, CLVReport, fit_clv_bander
from .dataset import (
    HotelBookings,
    load_bookings,
    prepare_features,
    split_chronological,
)
from .segmentation import GuestSegmenter, SegmentationReport, fit_guest_segmenter
from .store import load_all_artifacts, save_all_artifacts

__all__ = [
    "CLVBander",
    "CLVReport",
    "ChurnModel",
    "ChurnReport",
    "GuestSegmenter",
    "HotelBookings",
    "SegmentationReport",
    "fit_clv_bander",
    "fit_guest_segmenter",
    "load_all_artifacts",
    "load_bookings",
    "prepare_features",
    "save_all_artifacts",
    "split_chronological",
    "train_churn_model",
]
