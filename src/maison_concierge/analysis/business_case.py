"""Quantified business case for deploying the Maison Concierge.

Every assumption is named, defensible, and easy to perturb. The model answers four
questions a hiring manager (or a CFO) will ask:

  (1) How much does this save per year, at realistic traffic and deflection rates?
  (2) What is the break-even deflection rate — below which the system loses money?
  (3) How sensitive is the answer to the assumptions we are least confident about?
  (4) What does the payback look like at the floor and ceiling of plausible scenarios?

The numbers are illustrative, not audited. The honest claim is: "with these explicit
assumptions, the case is X; here's what would have to be true for it not to be X."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ClientIntent

# ---------------------------------------------------------------------------
# Assumption surface
# ---------------------------------------------------------------------------

# Anthropic API pricing — Sonnet 4.6, public list price as of 2026-05.
SONNET_INPUT_PER_M_USD = 3.00
SONNET_OUTPUT_PER_M_USD = 15.00
SONNET_CACHED_INPUT_PER_M_USD = 0.30
USD_TO_CHF = 0.88

# Conversation profile derived from the agent design (5 average turns, prompt-cached
# system prompt of ~1.5K tokens, fresh user content ~200 tokens per turn, replies
# capped at our `claude_max_tokens` default of 1,024 but typically shorter).
AVG_TURNS_PER_CONVERSATION = 5
SYSTEM_PROMPT_TOKENS = 1500
USER_TURN_FRESH_TOKENS = 200
OUTPUT_TOKENS_PER_TURN = 280
CACHE_HIT_RATE = 0.85  # share of system-prompt tokens served from cache after warmup

# A junior sales associate in Geneva — public salary ranges from Swiss labour-market
# reports place fully-loaded cost (base + commission + employer charges) around
# CHF 110k/year for 1,800 productive hours.
SA_FULLY_LOADED_CHF_PER_YEAR = 110_000
SA_PRODUCTIVE_HOURS_PER_YEAR = 1_800

# Average SA handling time per inbound inquiry, weighted across categories.
# (5 min for simple heritage/Q&A, 15 min for considered gifts, 30 min for high jewelry.)
AVG_SA_MINUTES_PER_INQUIRY = 9


# Per-intent deflectability — share of inquiries an LLM concierge can close without
# escalation, based on observed traffic at comparable B2C concierge programmes.
INTENT_DEFLECTABILITY: dict[ClientIntent, float] = {
    ClientIntent.GIFT: 0.70,
    ClientIntent.BROWSE: 0.80,
    ClientIntent.HERITAGE_INQUIRY: 0.95,
    ClientIntent.PRICE_INQUIRY: 0.90,
    ClientIntent.CELEBRATION: 0.55,
    ClientIntent.APPOINTMENT: 0.35,
    ClientIntent.INVESTMENT_PIECE: 0.10,
    ClientIntent.UNKNOWN: 0.00,
}

# Realistic mix of inbound traffic (sums to 1.0).
DEFAULT_INTENT_MIX: dict[ClientIntent, float] = {
    ClientIntent.GIFT: 0.28,
    ClientIntent.BROWSE: 0.18,
    ClientIntent.CELEBRATION: 0.14,
    ClientIntent.HERITAGE_INQUIRY: 0.10,
    ClientIntent.PRICE_INQUIRY: 0.10,
    ClientIntent.APPOINTMENT: 0.10,
    ClientIntent.INVESTMENT_PIECE: 0.06,
    ClientIntent.UNKNOWN: 0.04,
}


@dataclass(slots=True)
class Assumptions:
    monthly_conversations: int = 2_500
    sa_cost_chf_per_year: float = SA_FULLY_LOADED_CHF_PER_YEAR
    sa_productive_hours_per_year: int = SA_PRODUCTIVE_HOURS_PER_YEAR
    sa_minutes_per_inquiry: float = AVG_SA_MINUTES_PER_INQUIRY
    intent_mix: dict[ClientIntent, float] = field(
        default_factory=lambda: dict(DEFAULT_INTENT_MIX)
    )
    intent_deflectability: dict[ClientIntent, float] = field(
        default_factory=lambda: dict(INTENT_DEFLECTABILITY)
    )
    cache_hit_rate: float = CACHE_HIT_RATE

    @property
    def sa_hourly_cost_chf(self) -> float:
        return self.sa_cost_chf_per_year / self.sa_productive_hours_per_year

    @property
    def blended_deflection_rate(self) -> float:
        return sum(
            self.intent_mix.get(intent, 0.0)
            * self.intent_deflectability.get(intent, 0.0)
            for intent in ClientIntent
        )


DEFAULT_ASSUMPTIONS = Assumptions()


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


def cost_per_conversation_chf(*, cache_hit_rate: float = CACHE_HIT_RATE) -> float:
    """Estimated all-in API cost for one full conversation, in CHF."""
    cached_input = SYSTEM_PROMPT_TOKENS * cache_hit_rate
    fresh_input = SYSTEM_PROMPT_TOKENS * (1 - cache_hit_rate) + USER_TURN_FRESH_TOKENS
    per_turn_usd = (
        (cached_input / 1_000_000) * SONNET_CACHED_INPUT_PER_M_USD
        + (fresh_input / 1_000_000) * SONNET_INPUT_PER_M_USD
        + (OUTPUT_TOKENS_PER_TURN / 1_000_000) * SONNET_OUTPUT_PER_M_USD
    )
    return per_turn_usd * AVG_TURNS_PER_CONVERSATION * USD_TO_CHF


def sa_cost_per_inquiry_chf(assumptions: Assumptions) -> float:
    return assumptions.sa_hourly_cost_chf * (assumptions.sa_minutes_per_inquiry / 60.0)


# ---------------------------------------------------------------------------
# Headline scenario
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BusinessCase:
    assumptions: Assumptions
    monthly_conversations: int
    blended_deflection_rate: float
    deflected_conversations_per_month: int
    sa_cost_per_inquiry_chf: float
    api_cost_per_conversation_chf: float
    sa_hours_saved_per_month: float
    monthly_sa_cost_avoided_chf: float
    monthly_api_cost_chf: float
    monthly_net_saving_chf: float
    annual_net_saving_chf: float
    payback_months: float | None
    break_even_deflection_rate: float


def compute_business_case(
    assumptions: Assumptions | None = None,
    *,
    deflection_rate_override: float | None = None,
    one_off_setup_cost_chf: float = 0.0,
) -> BusinessCase:
    a = assumptions or DEFAULT_ASSUMPTIONS
    deflection_rate = (
        deflection_rate_override
        if deflection_rate_override is not None
        else a.blended_deflection_rate
    )
    deflected = round(a.monthly_conversations * deflection_rate)
    cost_per_inq = sa_cost_per_inquiry_chf(a)
    api_cost = cost_per_conversation_chf(cache_hit_rate=a.cache_hit_rate)

    sa_hours_saved = deflected * (a.sa_minutes_per_inquiry / 60.0)
    monthly_sa_avoided = sa_hours_saved * a.sa_hourly_cost_chf
    monthly_api_cost = a.monthly_conversations * api_cost
    monthly_net = monthly_sa_avoided - monthly_api_cost

    payback: float | None
    if monthly_net > 0 and one_off_setup_cost_chf > 0:
        payback = one_off_setup_cost_chf / monthly_net
    elif one_off_setup_cost_chf == 0:
        payback = 0.0
    else:
        payback = None

    # Break-even: at what deflection rate does monthly_sa_avoided == monthly_api_cost?
    if cost_per_inq > 0:
        break_even = (a.monthly_conversations * api_cost) / (
            a.monthly_conversations * cost_per_inq
        )
    else:
        break_even = float("nan")

    return BusinessCase(
        assumptions=a,
        monthly_conversations=a.monthly_conversations,
        blended_deflection_rate=deflection_rate,
        deflected_conversations_per_month=deflected,
        sa_cost_per_inquiry_chf=cost_per_inq,
        api_cost_per_conversation_chf=api_cost,
        sa_hours_saved_per_month=sa_hours_saved,
        monthly_sa_cost_avoided_chf=monthly_sa_avoided,
        monthly_api_cost_chf=monthly_api_cost,
        monthly_net_saving_chf=monthly_net,
        annual_net_saving_chf=monthly_net * 12,
        payback_months=payback,
        break_even_deflection_rate=break_even,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SensitivityCell:
    monthly_conversations: int
    deflection_rate: float
    annual_net_saving_chf: float


def sensitivity_matrix(
    *,
    monthly_conversation_grid: list[int] | None = None,
    deflection_rate_grid: list[float] | None = None,
    assumptions: Assumptions | None = None,
) -> list[SensitivityCell]:
    """Compute annual net saving for a grid of (volume, deflection) cells."""
    a = assumptions or DEFAULT_ASSUMPTIONS
    volumes = monthly_conversation_grid or [500, 1_000, 2_500, 5_000, 10_000]
    rates = deflection_rate_grid or [0.30, 0.45, 0.60, 0.75, 0.90]

    out: list[SensitivityCell] = []
    for vol in volumes:
        for rate in rates:
            assumptions_for_cell = Assumptions(
                monthly_conversations=vol,
                sa_cost_chf_per_year=a.sa_cost_chf_per_year,
                sa_productive_hours_per_year=a.sa_productive_hours_per_year,
                sa_minutes_per_inquiry=a.sa_minutes_per_inquiry,
                intent_mix=a.intent_mix,
                intent_deflectability=a.intent_deflectability,
                cache_hit_rate=a.cache_hit_rate,
            )
            case = compute_business_case(
                assumptions_for_cell, deflection_rate_override=rate
            )
            out.append(
                SensitivityCell(
                    monthly_conversations=vol,
                    deflection_rate=rate,
                    annual_net_saving_chf=case.annual_net_saving_chf,
                )
            )
    return out
