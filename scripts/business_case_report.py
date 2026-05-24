"""Print the headline ROI summary and a small sensitivity table.

Usage:
    python scripts/business_case_report.py
    python scripts/business_case_report.py --volume 5000 --deflection 0.70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maison_concierge.analysis import (  # noqa: E402
    DEFAULT_ASSUMPTIONS,
    Assumptions,
    compute_business_case,
    sensitivity_matrix,
)


def _format_chf(amount: float) -> str:
    return f"CHF {amount:>12,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Maison Concierge business case.")
    parser.add_argument("--volume", type=int, help="Monthly conversations.")
    parser.add_argument("--deflection", type=float, help="Override blended deflection rate (0-1).")
    parser.add_argument("--setup-cost", type=float, default=15_000.0, help="One-off implementation cost (CHF).")
    args = parser.parse_args()

    assumptions = Assumptions(
        monthly_conversations=args.volume or DEFAULT_ASSUMPTIONS.monthly_conversations,
    )
    case = compute_business_case(
        assumptions,
        deflection_rate_override=args.deflection,
        one_off_setup_cost_chf=args.setup_cost,
    )

    print("ASSUMPTIONS")
    print("-----------")
    print(f"  Monthly conversations               {case.monthly_conversations:>12,}")
    print(f"  SA fully-loaded cost                {_format_chf(assumptions.sa_cost_chf_per_year)} / year")
    print(f"  SA productive hours                 {assumptions.sa_productive_hours_per_year:>12,} / year")
    print(f"  SA effective cost                   {_format_chf(assumptions.sa_hourly_cost_chf)} / hour")
    print(f"  Avg SA handling time                {assumptions.sa_minutes_per_inquiry:>12.1f} minutes / inquiry")
    print(f"  Blended deflection rate             {case.blended_deflection_rate:>12.1%}")
    print(f"  Cache hit rate (prompt caching)     {assumptions.cache_hit_rate:>12.0%}")
    print()
    print("UNIT ECONOMICS")
    print("--------------")
    print(f"  SA cost per handled inquiry         {_format_chf(case.sa_cost_per_inquiry_chf)}")
    print(f"  API cost per conversation           {_format_chf(case.api_cost_per_conversation_chf)}")
    print(f"  Margin per deflected conversation   {_format_chf(case.sa_cost_per_inquiry_chf - case.api_cost_per_conversation_chf)}")
    print()
    print("MONTHLY FLOW")
    print("------------")
    print(f"  Deflected conversations             {case.deflected_conversations_per_month:>12,}")
    print(f"  SA hours saved                      {case.sa_hours_saved_per_month:>12.0f}")
    print(f"  SA cost avoided                     {_format_chf(case.monthly_sa_cost_avoided_chf)}")
    print(f"  API cost incurred                   {_format_chf(case.monthly_api_cost_chf)}")
    print(f"  Monthly net saving                  {_format_chf(case.monthly_net_saving_chf)}")
    print()
    print("HEADLINE")
    print("--------")
    print(f"  Annual net saving                   {_format_chf(case.annual_net_saving_chf)}")
    print(f"  Break-even deflection rate          {case.break_even_deflection_rate:>12.2%}")
    if case.payback_months is not None and case.payback_months > 0:
        print(f"  Payback on CHF {args.setup_cost:,.0f} setup        {case.payback_months:>12.1f} months")
    print()
    print("SENSITIVITY  (annual net saving, CHF)")
    print("-------------------------------------")
    grid = sensitivity_matrix()
    deflection_rates = sorted({c.deflection_rate for c in grid})
    volumes = sorted({c.monthly_conversations for c in grid})
    header = "vol\\defl".rjust(10) + "".join(f"{r:>10.0%}" for r in deflection_rates)
    print(header)
    cell_lookup = {(c.monthly_conversations, c.deflection_rate): c for c in grid}
    for vol in volumes:
        row = f"{vol:>10,}"
        for r in deflection_rates:
            row += f"{cell_lookup[(vol, r)].annual_net_saving_chf:>10,.0f}"
        print(row)


if __name__ == "__main__":
    main()
