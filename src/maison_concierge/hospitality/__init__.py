"""Hospitality-domain orchestration layer.

Kept as a self-contained subpackage so the pivot from the jewelry-maison
scaffolding to the hospitality maison is a directory-level change rather than
a mass rewrite. The old modules stay callable until Phase 5 deletes them.
"""

from .booking import BookingAction, BookingActionResult, BookingAgent
from .guardrail import GuardrailReport, apply_guardrails
from .intent import HospitalityIntent, classify_intent_rule_based
from .kb import KBSearchResult, PropertyKB
from .orchestrator import HospitalityOrchestrator, run_hospitality_turn
from .personas import Persona, load_personas, persona_by_id
from .profile import ProfileAgent, ProfileSnapshot
from .recommend import Experience, ExperienceRanked, RecommendationAgent, RecommendationResult
from .state import HospitalityState

__all__ = [
    "BookingAction",
    "BookingActionResult",
    "BookingAgent",
    "Experience",
    "ExperienceRanked",
    "GuardrailReport",
    "HospitalityIntent",
    "HospitalityOrchestrator",
    "HospitalityState",
    "KBSearchResult",
    "Persona",
    "ProfileAgent",
    "ProfileSnapshot",
    "PropertyKB",
    "RecommendationAgent",
    "RecommendationResult",
    "apply_guardrails",
    "classify_intent_rule_based",
    "load_personas",
    "persona_by_id",
    "run_hospitality_turn",
]
