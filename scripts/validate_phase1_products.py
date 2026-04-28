"""Small Phase 1 validation script.

Run from the repository root after copying the files:
    python scripts/validate_phase1_products.py
"""

from __future__ import annotations

from src.products.barrier_option import BarrierOption
from src.products.coupon_bond import CouponBond
from src.products.option_strategies import CallSpread
from src.products.vanilla_option import VanillaOption


def main() -> None:
    bond = CouponBond("CB-DEMO", 1_000.0, 2.0, 0.05, 2)
    assert bond.get_cash_flows()[-1][1] == 1_025.0

    strategy = CallSpread("CS-DEMO", 1.0, 100.0, 120.0)
    assert strategy.payoff({"spot": 130.0}) == 20.0

    path = {"path": [100.0, 88.0, 112.0]}
    ko = BarrierOption("KO", "call", 100.0, 1.0, 90.0, "KO", barrier_direction="down")
    ki = BarrierOption("KI", "call", 100.0, 1.0, 90.0, "KI", barrier_direction="down")
    vanilla = VanillaOption("V", "call", 100.0, 1.0)
    assert ko.payoff(path) + ki.payoff(path) == vanilla.payoff({"spot": 112.0})

    print("Phase 1 products validation: OK")


if __name__ == "__main__":
    main()
