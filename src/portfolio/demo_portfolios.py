"""Build financially meaningful demo portfolios from a neutral inventory.

The raw inventory stays product-centric. This module creates a portfolio allocation
layer for the dashboard/application:
- 4 mixed portfolios,
- at least 2 product families per portfolio,
- deterministic allocation,
- optional reuse of the same instrument across portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioLegRule:
    """One desired product family in a demo portfolio."""

    family: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PortfolioTemplate:
    """Definition of one demo portfolio."""

    name: str
    strategy: str
    legs: tuple[PortfolioLegRule, ...]


DEFAULT_PORTFOLIO_TEMPLATES: tuple[PortfolioTemplate, ...] = (
    PortfolioTemplate(
        name="PF1_Defensive_Income",
        strategy="Defensive income: rates exposure plus capital protection / limited equity upside.",
        legs=(
            PortfolioLegRule("rates", 1.00),
            PortfolioLegRule("capital_protected", 0.70),
            PortfolioLegRule("option_strategy", 0.40),
        ),
    ),
    PortfolioTemplate(
        name="PF2_Equity_Yield",
        strategy="Equity yield: coupon-oriented products with some rates hedge.",
        legs=(
            PortfolioLegRule("structured_yield", 1.00),
            PortfolioLegRule("autocall", 0.80),
            PortfolioLegRule("rates", 0.50),
        ),
    ),
    PortfolioTemplate(
        name="PF3_Convexity_Protection",
        strategy="Convexity and protection: optionality plus rates component.",
        legs=(
            PortfolioLegRule("vanilla_option", 1.00),
            PortfolioLegRule("barrier", 0.70),
            PortfolioLegRule("rates", 0.40),
        ),
    ),
    PortfolioTemplate(
        name="PF4_Balanced_Multi_Product",
        strategy="Balanced multi-product book used as dashboard benchmark.",
        legs=(
            PortfolioLegRule("rates", 0.70),
            PortfolioLegRule("option_strategy", 0.70),
            PortfolioLegRule("autocall", 0.50),
            PortfolioLegRule("capital_protected", 0.50),
        ),
    ),
)


FAMILY_FALLBACK_ORDER: tuple[str, ...] = (
    "rates",
    "capital_protected",
    "structured_yield",
    "autocall",
    "option_strategy",
    "barrier",
    "vanilla_option",
    "other",
)


def create_demo_mixed_portfolios(
    inventory: pd.DataFrame,
    *,
    templates: Iterable[PortfolioTemplate] = DEFAULT_PORTFOLIO_TEMPLATES,
    allow_reuse: bool = True,
    suffix_product_id_by_portfolio: bool = True,
) -> pd.DataFrame:
    """Create 4 financially meaningful demo portfolios from normalized inventory.

    Parameters
    ----------
    inventory:
        Output of combine_inventory_sheets(...), before build_pricing_inventory(...).
    templates:
        Portfolio definitions.
    allow_reuse:
        If True, the same source instrument can appear in multiple portfolios.
        This is useful for demo portfolios.
    suffix_product_id_by_portfolio:
        If True, product_id becomes portfolio-specific while source_product_id
        keeps the original instrument ID. This avoids grouping collisions,
        especially for autocalls.

    Returns
    -------
    pd.DataFrame
        Expanded inventory with portfolio, product_family, allocation_weight,
        source_product_id and portfolio_strategy columns.
    """
    if inventory.empty:
        raise ValueError("Cannot create demo portfolios from an empty inventory.")

    base = inventory.copy()
    _ensure_source_product_id(base)
    _normalize_mutable_identifier_columns(base)

    positions = _build_atomic_positions(base)
    if not positions:
        raise ValueError("No atomic positions available to create demo portfolios.")

    by_family = _positions_by_family(positions)
    available_families = {family for family, items in by_family.items() if items}

    if len(available_families) < 2:
        raise ValueError(
            "At least two product families are required to create mixed portfolios. "
            f"Available families: {sorted(available_families)}"
        )

    cursors: dict[str, int] = {family: 0 for family in FAMILY_FALLBACK_ORDER}
    used_position_ids: set[str] = set()
    output_blocks: list[pd.DataFrame] = []

    for template in templates:
        selected_families: set[str] = set()

        for leg in template.legs:
            position = _select_position(
                requested_family=leg.family,
                by_family=by_family,
                cursors=cursors,
                used_position_ids=used_position_ids,
                allow_reuse=allow_reuse,
            )

            block = position.frame.copy()
            source_product_id = str(block["source_product_id"].iloc[0])
            portfolio_product_id = _portfolio_product_id(template.name, source_product_id)

            block["portfolio"] = template.name
            block["portfolio_strategy"] = template.strategy
            block["product_family"] = position.family
            block["allocation_weight"] = float(leg.weight)
            block["instrument_id"] = source_product_id
            block["position_id"] = portfolio_product_id

            if suffix_product_id_by_portfolio:
                block["product_id"] = portfolio_product_id

            _scale_position(block, leg.weight)

            selected_families.add(position.family)
            used_position_ids.add(position.position_key)
            output_blocks.append(block)

        if len(selected_families) < 2:
            raise ValueError(
                f"Portfolio {template.name} has fewer than two product families: "
                f"{sorted(selected_families)}"
            )

    result = pd.concat(output_blocks, ignore_index=True, sort=False)

    # Nice column ordering for later debugging/dashboard.
    preferred = [
        "portfolio",
        "portfolio_strategy",
        "position_id",
        "instrument_id",
        "source_product_id",
        "product_family",
        "allocation_weight",
        "source_sheet",
        "source_row",
        "valuation_date",
        "product_id",
        "product_type",
        "underlying",
        "currency",
        "rate_currency",
    ]
    ordered = [c for c in preferred if c in result.columns] + [
        c for c in result.columns if c not in preferred
    ]
    return result[ordered]


@dataclass(frozen=True, slots=True)
class _AtomicPosition:
    position_key: str
    family: str
    frame: pd.DataFrame


def _ensure_source_product_id(frame: pd.DataFrame) -> None:
    """Create a stable source_product_id if product_id is missing."""
    if "product_id" not in frame.columns:
        frame["product_id"] = pd.NA

    if "source_sheet" in frame.columns:
        source_sheet = frame["source_sheet"].astype("string").fillna("unknown")
    else:
        source_sheet = pd.Series("unknown", index=frame.index, dtype="string")

    row_fallback = pd.Series(frame.index, index=frame.index).astype("string")
    if "source_row" in frame.columns:
        source_row = frame["source_row"].astype("string").fillna(row_fallback)
    else:
        source_row = row_fallback

    fallback = (
        source_sheet
        + "-"
        + source_row
    )

    product_id = frame["product_id"].astype("string")
    frame["source_product_id"] = product_id.fillna(fallback)
    frame["source_product_id"] = frame["source_product_id"].replace({"": pd.NA}).fillna(fallback)
    frame["source_product_id"] = frame["source_product_id"].map(_clean_id)




def _normalize_mutable_identifier_columns(frame: pd.DataFrame) -> None:
    """Normalize identifier columns that may receive portfolio-level string IDs.

    Excel often loads columns such as product_id with pandas nullable integer
    dtype (Int64). Later, demo portfolio construction writes IDs like
    "PF1_Defensive_Income__IRS-1" into product_id. Pandas cannot safely
    cast those strings back to Int64, so the mutable identifier columns must be
    string/object typed before assignment.
    """
    for column in (
        "product_id",
        "source_product_id",
        "instrument_id",
        "position_id",
        "portfolio",
        "portfolio_strategy",
        "product_family",
    ):
        if column in frame.columns:
            frame[column] = frame[column].astype("string")

def _build_atomic_positions(frame: pd.DataFrame) -> list[_AtomicPosition]:
    """Build atomic positions.

    Autocalls are grouped by source_product_id because one autocall is represented
    by several observation-date rows. Other products are one row = one position.
    """
    positions: list[_AtomicPosition] = []

    source_sheet = frame.get("source_sheet", pd.Series(index=frame.index, dtype="object"))
    is_autocall = source_sheet.astype("string").str.lower().eq("autocalls")

    # Group autocalls so all observation dates stay together.
    for source_product_id, group in frame.loc[is_autocall].groupby("source_product_id", dropna=False):
        family = _classify_product_family(group.iloc[0])
        key = f"autocalls::{source_product_id}"
        positions.append(_AtomicPosition(position_key=key, family=family, frame=group.copy()))

    # Other products: each row is one independent position.
    non_autocalls = frame.loc[~is_autocall].copy()
    for idx, row in non_autocalls.iterrows():
        family = _classify_product_family(row)
        source_product_id = str(row.get("source_product_id", f"line-{idx}"))
        key = f"{row.get('source_sheet', 'unknown')}::{source_product_id}::{idx}"
        positions.append(_AtomicPosition(position_key=key, family=family, frame=non_autocalls.loc[[idx]].copy()))

    return positions


def _positions_by_family(positions: list[_AtomicPosition]) -> dict[str, list[_AtomicPosition]]:
    by_family: dict[str, list[_AtomicPosition]] = {family: [] for family in FAMILY_FALLBACK_ORDER}
    for position in positions:
        by_family.setdefault(position.family, []).append(position)
    return by_family


def _select_position(
    *,
    requested_family: str,
    by_family: dict[str, list[_AtomicPosition]],
    cursors: dict[str, int],
    used_position_ids: set[str],
    allow_reuse: bool,
) -> _AtomicPosition:
    candidate_families = (requested_family,) + tuple(
        family for family in FAMILY_FALLBACK_ORDER if family != requested_family
    )

    for family in candidate_families:
        candidates = by_family.get(family, [])
        if not candidates:
            continue

        start = cursors.get(family, 0)
        for offset in range(len(candidates)):
            idx = (start + offset) % len(candidates)
            candidate = candidates[idx]

            if allow_reuse or candidate.position_key not in used_position_ids:
                cursors[family] = (idx + 1) % len(candidates)
                return candidate

    raise ValueError(
        f"Could not select any position for requested family={requested_family!r}. "
        "Try allow_reuse=True or enrich the source inventory."
    )


def _classify_product_family(row: pd.Series) -> str:
    sheet = str(row.get("source_sheet", "")).strip().lower()
    product_type = str(row.get("product_type", "")).strip().lower()
    sspa_code = str(row.get("sspa_code", "")).strip()

    if sheet in {"swaps", "bonds", "rates"}:
        return "rates"

    if sheet == "autocalls":
        return "autocall"

    if sheet == "structured_notes":
        if "1220" in sspa_code or "reverse" in product_type or "convertible" in product_type:
            return "structured_yield"
        return "capital_protected"

    if sheet == "options":
        has_barrier_level = pd.notna(row.get("barrier_level", pd.NA))
        has_barrier_type = pd.notna(row.get("barrier_type", pd.NA))
        barrier_words = ("barrier", "knock", "down-and", "up-and", "down and", "up and")
        strategy_words = ("spread", "butterfly", "straddle")

        if has_barrier_level or has_barrier_type or any(word in product_type for word in barrier_words):
            return "barrier"

        if any(word in product_type for word in strategy_words):
            return "option_strategy"

        if "call" in product_type or "put" in product_type:
            return "vanilla_option"

    return "other"


def _scale_position(block: pd.DataFrame, weight: float) -> None:
    """Scale notional-like columns for portfolio allocation.

    The output portfolio can use non-integer allocation weights (0.7, 0.4, ...).
    Therefore quantity/notional columns are converted to float before writing
    scaled values. This avoids pandas Int64 safe-cast errors when the source
    Excel inventory contains integer quantities.
    """
    for column in ("quantity", "notional"):
        if column in block.columns:
            values = pd.to_numeric(block[column], errors="coerce").astype("float64")
            block[column] = values
            mask = values.notna()
            block.loc[mask, column] = values.loc[mask] * float(weight)


def _portfolio_product_id(portfolio: str, source_product_id: str) -> str:
    return f"{_clean_id(portfolio)}__{_clean_id(source_product_id)}"


def _clean_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace(" ", "_").replace("/", "_").replace("\\", "_")
    text = text.replace(":", "_").replace(";", "_")
    return text or "UNKNOWN"


__all__ = [
    "DEFAULT_PORTFOLIO_TEMPLATES",
    "PortfolioLegRule",
    "PortfolioTemplate",
    "create_demo_mixed_portfolios",
]
