from maison_concierge.analysis import (
    DEFAULT_ASSUMPTIONS,
    Assumptions,
    compute_business_case,
    sensitivity_matrix,
)
from maison_concierge.analysis.business_case import cost_per_conversation_chf


def test_default_case_produces_positive_annual_saving():
    case = compute_business_case()
    assert case.annual_net_saving_chf > 0
    assert 0.0 < case.blended_deflection_rate < 1.0


def test_api_cost_per_conversation_is_small():
    cost = cost_per_conversation_chf()
    # Sonnet 4.6 with prompt caching should be well under CHF 0.10 per conversation
    assert 0.0 < cost < 0.10


def test_higher_volume_increases_saving_proportionally():
    low = compute_business_case(Assumptions(monthly_conversations=500))
    high = compute_business_case(Assumptions(monthly_conversations=5_000))
    # 10x volume → roughly 10x net saving (within 5% because cost terms are linear)
    ratio = high.annual_net_saving_chf / low.annual_net_saving_chf
    assert 9.5 < ratio < 10.5


def test_low_deflection_can_make_net_saving_smaller():
    case_low = compute_business_case(deflection_rate_override=0.05)
    case_high = compute_business_case(deflection_rate_override=0.85)
    assert case_high.annual_net_saving_chf > case_low.annual_net_saving_chf


def test_sensitivity_matrix_returns_full_grid():
    grid = sensitivity_matrix(
        monthly_conversation_grid=[1_000, 5_000],
        deflection_rate_grid=[0.30, 0.80],
    )
    assert len(grid) == 4
    cells = {(c.monthly_conversations, c.deflection_rate): c for c in grid}
    assert cells[(5_000, 0.80)].annual_net_saving_chf > cells[(1_000, 0.30)].annual_net_saving_chf


def test_assumptions_hourly_cost_property():
    a = Assumptions()
    assert a.sa_hourly_cost_chf == DEFAULT_ASSUMPTIONS.sa_cost_chf_per_year / DEFAULT_ASSUMPTIONS.sa_productive_hours_per_year
