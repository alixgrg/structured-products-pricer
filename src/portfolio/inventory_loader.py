"""Load, validate and normalize the portfolio inventory workbook.

This module handles the portfolio inventory layer:
- reading the Excel workbook,
- normalizing each sheet,
- mapping French column names to canonical English names,
- parsing dates, numbers and text fields,
- exporting normalized sheets into data/interim,
- exporting a compact summary into data/processed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.convention import DEFAULT_YEAR_BASIS, canonicalize_columns, strip_accents, to_snake_case
from src.io_utils import (
    as_path as _as_path,
    copy_if_needed as _copy_if_needed,
    normalize_datetime as _normalize_datetime,
    prefer_existing_raw_source as _prefer_existing_raw_source,
    require_columns as _require_columns,
    require_file as _require_file,
    to_numeric as _to_numeric,
)


# ---------------------------------------------------------------------------
# Sheet configuration
# ---------------------------------------------------------------------------
DEFAULT_PORTFOLIO = "default"

DEFAULT_CURRENCY_BY_SHEET = {
    "swaps": "EUR",
    "swap": "EUR",
    "bonds": "EUR",
    "bond": "EUR",
    "options": "USD",
    "option": "USD",
    "autocalls": "USD",
    "autocall": "USD",
    "structured_notes": "EUR",
    "notes_structurees": "EUR",
}

_SHEET_ALIASES = {
    "swap": "swaps",
    "swaps": "swaps",
    "options": "options",
    "option": "options",
    "autocall": "autocalls",
    "autocalls": "autocalls",
    "notes_structurees": "structured_notes",
    "note_structuree": "structured_notes",
    "structured_notes": "structured_notes",
}

_COLUMN_MAPS: dict[str, dict[str, str]] = {
    "swaps": {
        "portefeuille": "portfolio",
        "date_valorisation": "valuation_date",
        "devise": "currency",
        "maturite": "maturity_date",
        "frequence_fixe": "fixed_leg_frequency",
        "nominal": "notional",
        "taux_fixe": "fixed_rate",
        "taux_variable_1": "floating_rate_index_1",
        "taux_variable_2": "floating_rate_index_2",
    },
    "options": {
        "portefeuille": "portfolio",
        "date_valorisation": "valuation_date",
        "devise": "currency",
        "produit": "product_type",
        "quantite": "quantity",
        "sous_jacent": "underlying",
        "maturite": "maturity_date",
        "strike_1": "strike_1",
        "strike_2": "strike_2",
        "strike_3": "strike_3",
        "type_barriere": "barrier_type",
        "niveau_barriere": "barrier_level",
    },
    "autocalls": {
        "portefeuille": "portfolio",
        "date_valorisation": "valuation_date",
        "devise": "currency",
        "id_produit": "product_id",
        "date_observation": "observation_date",
        "niveau_de_rappel": "autocall_trigger_level",
        "date_reference": "reference_date",
        "coupon": "coupon_rate",
        "sous_jacent": "underlying",
    },
    "structured_notes": {
        "portefeuille": "portfolio",
        "date_valorisation": "valuation_date",
        "code_produit_sspa": "sspa_code",
        "quantite": "quantity",
        "sous_jacent": "underlying",
        "taux_de_participation": "participation_rate",
        "devise": "currency",
        "devise_taux": "rate_currency",
        "maturite": "maturity_date",
        "barriere_1": "barrier_1",
        "cap": "cap",
        "barriere_2": "barrier_2",
    },
}

_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "swaps": {
        "valuation_date",
        "currency",
        "maturity_date",
        "notional",
        "fixed_rate",
    },
    "options": {
        "valuation_date",
        "product_type",
        "quantity",
        "underlying",
        "maturity_date",
    },
    "autocalls": {
        "valuation_date",
        "product_id",
        "observation_date",
        "autocall_trigger_level",
        "coupon_rate",
        "underlying",
    },
    "structured_notes": {
        "valuation_date",
        "sspa_code",
        "quantity",
        "underlying",
        "participation_rate",
        "maturity_date",
    },
}

_DATE_COLUMNS = {
    "swaps": ("valuation_date", "maturity_date"),
    "options": ("valuation_date", "maturity_date"),
    "autocalls": ("valuation_date", "observation_date", "reference_date"),
    "structured_notes": ("valuation_date", "maturity_date"),
}

_NUMERIC_COLUMNS = {
    "swaps": ("notional", "fixed_rate"),
    "options": ("quantity", "strike_1", "strike_2", "strike_3", "barrier_level"),
    "autocalls": ("product_id", "autocall_trigger_level", "coupon_rate"),
    "structured_notes": (
        "sspa_code",
        "quantity",
        "participation_rate",
        "barrier_1",
        "cap",
        "barrier_2",
    ),
}

_TEXT_COLUMNS = {
    "swaps": (
        "portfolio",
        "currency",
        "fixed_leg_frequency",
        "floating_rate_index_1",
        "floating_rate_index_2",
    ),
    "options": ("portfolio", "currency", "product_type", "underlying", "barrier_type"),
    "autocalls": ("portfolio", "currency", "underlying"),
    "structured_notes": ("portfolio", "currency", "underlying", "rate_currency"),
}

_UPPERCASE_TEXT_COLUMNS = {
    "currency",
    "rate_currency",
    "underlying",
}


def _normalize_sheet_name(sheet_name: str) -> str:
    """Map free-form workbook sheet names to canonical sheet keys."""
    key = to_snake_case(strip_accents(sheet_name))
    return _SHEET_ALIASES.get(key, key)


def _standardize_text(series: pd.Series, uppercase: bool = False) -> pd.Series:
    """Normalize text fields."""
    normalized = series.astype("string").str.strip()
    normalized = normalized.replace({"": pd.NA})
    if uppercase:
        normalized = normalized.str.upper()
    return normalized


def _drop_empty_rows_and_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove fully empty rows and columns commonly found in Excel files."""
    cleaned = frame.copy()
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


def _add_maturity_years(normalized: pd.DataFrame) -> pd.DataFrame:
    """Add a time_to_maturity_years column if valuation and maturity dates exist."""
    if {"valuation_date", "maturity_date"}.issubset(normalized.columns):
        normalized["time_to_maturity_days"] = (
            normalized["maturity_date"] - normalized["valuation_date"]
        ).dt.days
        normalized["time_to_maturity_years"] = (
            normalized["time_to_maturity_days"] / DEFAULT_YEAR_BASIS
        )
    return normalized

def _default_currency_for_sheet(sheet_key: str) -> str:
    return DEFAULT_CURRENCY_BY_SHEET.get(str(sheet_key).strip().lower(), "EUR")


def _ensure_portfolio_and_currency(normalized: pd.DataFrame, sheet_key: str) -> pd.DataFrame:
    """Ensure portfolio/currency metadata exists and is clean.

    portfolio:
        Preserved when present, otherwise defaults to "default".

    currency:
        Preserved when present.
        If absent and rate_currency exists, currency inherits rate_currency.
        Otherwise a sheet-level default is used:
        - swaps/bonds/rates: EUR
        - options/autocalls: USD
        - structured notes: EUR unless explicit.
    """
    data = normalized.copy()

    if "portfolio" not in data.columns:
        data["portfolio"] = DEFAULT_PORTFOLIO

    data["portfolio"] = (
        data["portfolio"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .fillna(DEFAULT_PORTFOLIO)
    )

    if "currency" not in data.columns:
        if "rate_currency" in data.columns:
            data["currency"] = data["rate_currency"]
        else:
            data["currency"] = _default_currency_for_sheet(sheet_key)

    data["currency"] = (
        data["currency"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
        .fillna(_default_currency_for_sheet(sheet_key))
    )

    if "rate_currency" in data.columns:
        data["rate_currency"] = (
            data["rate_currency"]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
            .fillna(data["currency"])
        )

    return data


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def stage_inventory_source(
    config: ProjectConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy the external inventory workbook into the repository raw layer."""
    cfg = config or ProjectConfig.default()
    cfg.ensure_directories()

    return _copy_if_needed(
        _as_path(cfg.inventory_source),
        cfg.raw_inventory_path,
        overwrite,
        dataset_name="inventory source",
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_inventory_sheet(sheet_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize one sheet from the inventory workbook.

    The function is strict for known sheets and permissive for unknown sheets:
    - known sheets are validated against required columns,
    - unknown sheets are simply canonicalized and kept for traceability.
    """
    sheet_key = _normalize_sheet_name(sheet_name)

    normalized = _drop_empty_rows_and_columns(frame)
    normalized.columns = canonicalize_columns(normalized.columns)
    normalized = normalized.rename(columns=_COLUMN_MAPS.get(sheet_key, {}))

    # Insert traceability columns.
    normalized.insert(0, "source_sheet", sheet_key)
    normalized.insert(1, "source_row", range(1, len(normalized) + 1))

    # Validate only known sheets.
    if sheet_key in _REQUIRED_COLUMNS:
        _require_columns(
            normalized,
            _REQUIRED_COLUMNS[sheet_key],
            dataset_name=f"inventory sheet '{sheet_name}'",
        )

    # Date parsing.
    for column in _DATE_COLUMNS.get(sheet_key, ()):
        if column in normalized.columns:
            normalized[column] = _normalize_datetime(normalized[column])

    # Numeric parsing.
    for column in _NUMERIC_COLUMNS.get(sheet_key, ()):
        if column in normalized.columns:
            normalized[column] = _to_numeric(normalized[column])

    # Text parsing.
    for column in _TEXT_COLUMNS.get(sheet_key, ()):
        if column in normalized.columns:
            normalized[column] = _standardize_text(
                normalized[column],
                uppercase=column in _UPPERCASE_TEXT_COLUMNS,
            )
    normalized = _ensure_portfolio_and_currency(normalized, sheet_key)
    normalized = _add_maturity_years(normalized)

    sort_columns = [
        column
        for column in (
            "valuation_date",
            "maturity_date",
            "observation_date",
            "underlying",
            "product_type",
        )
        if column in normalized.columns
    ]

    if sort_columns:
        normalized = normalized.sort_values(by=sort_columns, ignore_index=True)
    else:
        normalized = normalized.reset_index(drop=True)

    return normalized


def load_inventory_workbook(
    source: str | Path | None = None,
    *,
    config: ProjectConfig | None = None,
    normalize: bool = True,
    keep_empty_sheets: bool = False,
) -> dict[str, pd.DataFrame]:
    """Read every sheet from the inventory workbook.

    Parameters
    ----------
    source:
        Optional explicit Excel path.
    config:
        Project configuration.
    normalize:
        Whether to normalize sheet names, columns and types.
    keep_empty_sheets:
        If False, empty sheets are ignored.
    """
    cfg = config or ProjectConfig.default()
    default_source = _prefer_existing_raw_source(cfg.raw_inventory_path, cfg.inventory_source)
    path = _as_path(source) if source is not None else _as_path(default_source)
    _require_file(path, "inventory")

    workbook = pd.read_excel(path, sheet_name=None)

    if not keep_empty_sheets:
        workbook = {
            sheet_name: frame
            for sheet_name, frame in workbook.items()
            if not _drop_empty_rows_and_columns(frame).empty
        }

    if not normalize:
        return workbook

    return {
        _normalize_sheet_name(sheet_name): normalize_inventory_sheet(sheet_name, frame)
        for sheet_name, frame in workbook.items()
    }


def combine_inventory_sheets(
    inventory_by_sheet: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine normalized inventory sheets into one wide dataset."""
    if not inventory_by_sheet:
        return pd.DataFrame()

    combined = pd.concat(
        inventory_by_sheet.values(),
        ignore_index=True,
        sort=False,
    )

    preferred_columns = [
        "source_sheet",
        "source_row",
        "portfolio",
        "valuation_date",
        "product_id",
        "product_type",
        "underlying",
        "currency",
        "rate_currency",
        "maturity_date",
        "time_to_maturity_days",
        "time_to_maturity_years",
        "quantity",
        "notional",
        "position_sign",
        "position_size",
        "strike_1",
        "strike_2",
        "strike_3",
        "barrier_type",
        "barrier_level",
        "participation_rate",
        "coupon_rate",
        "cap",
    ]

    ordered_columns = [
        column for column in preferred_columns if column in combined.columns
    ] + [
        column for column in combined.columns if column not in preferred_columns
    ]

    return combined[ordered_columns]


def inventory_dataset_summary(
    inventory_by_sheet: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize the normalized inventory workbook."""
    rows = []

    for sheet_name, frame in inventory_by_sheet.items():
        date_columns = [
            column
            for column in ("valuation_date", "maturity_date", "observation_date")
            if column in frame.columns
        ]

        min_date = min((frame[column].min() for column in date_columns), default=pd.NaT)
        max_date = max((frame[column].max() for column in date_columns), default=pd.NaT)

        rows.append(
            {
                "sheet_name": sheet_name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "min_date": min_date,
                "max_date": max_date,
                "missing_cells": int(frame.isna().sum().sum()),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sheet_name",
                "rows",
                "columns",
                "min_date",
                "max_date",
                "missing_cells",
            ]
        )

    return pd.DataFrame(rows).sort_values("sheet_name", ignore_index=True)


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def build_inventory_data_assets(
    config: ProjectConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Populate raw, interim and processed folders for the inventory workbook."""
    cfg = config or ProjectConfig.default()
    stage_inventory_source(cfg, overwrite=overwrite)

    # Read from raw layer after staging to make the pipeline reproducible.
    inventory = load_inventory_workbook(cfg.raw_inventory_path, config=cfg)
    combined = combine_inventory_sheets(inventory)
    summary = inventory_dataset_summary(inventory)

    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {"inventory_raw": cfg.raw_inventory_path}

    for sheet_name, frame in inventory.items():
        target = cfg.interim_inventory_path(sheet_name)
        frame.to_csv(target, index=False)
        output_paths[f"inventory_{sheet_name}_interim"] = target

    combined_path = cfg.interim_dir / "inventory_all_normalized.csv"
    combined.to_csv(combined_path, index=False)
    output_paths["inventory_all_interim"] = combined_path

    summary.to_csv(cfg.processed_inventory_summary_path, index=False)
    output_paths["inventory_summary"] = cfg.processed_inventory_summary_path

    return output_paths

# ---------------------------------------------------------------------------
# Pricing inventory view
# ---------------------------------------------------------------------------

def build_pricing_inventory(
    inventory_by_sheet: dict[str, pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    """Create a product-level inventory ready for pricing.

    ``combine_inventory_sheets`` remains the diagnostic wide table. This function
    creates a cleaner product-level view:

    - one row = one product;
    - accepts either the sheet dict from ``load_inventory_workbook`` or a
      combined dataframe containing ``source_sheet``;
    - autocalls are grouped by product_id;
    - product_type is inferred when missing;
    - basis swaps are identified when fixed_rate is missing but two floating
      indices are available;
    - position_sign is separated from positive product notional.
    """
    if isinstance(inventory_by_sheet, pd.DataFrame):
        if inventory_by_sheet.empty:
            return pd.DataFrame()
        if "source_sheet" not in inventory_by_sheet.columns:
            raise ValueError("inventory dataframe must contain a 'source_sheet' column.")
        inventory_by_sheet = {
            str(sheet_name): group.copy()
            for sheet_name, group in inventory_by_sheet.groupby("source_sheet", dropna=False)
        }

    if not inventory_by_sheet:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for sheet_name, frame in inventory_by_sheet.items():
        if frame is None or frame.empty:
            continue

        sheet_key = str(sheet_name).strip().lower()
        data = frame.copy()

        if sheet_key in {"autocalls", "autocall"}:
            frames.append(_build_pricing_autocalls(data))
            continue

        if sheet_key in {"swaps", "swap"}:
            data = data.copy()
            data["product_type"] = data.get("product_type", pd.Series(pd.NA, index=data.index))
            data["pricing_status_hint"] = "ok"

            is_basis = (
                data.get("fixed_rate", pd.Series(pd.NA, index=data.index)).isna()
                & data.get("floating_rate_index_1", pd.Series(pd.NA, index=data.index)).notna()
                & data.get("floating_rate_index_2", pd.Series(pd.NA, index=data.index)).notna()
            )

            data.loc[is_basis, "product_type"] = "Basis Swap"
            data.loc[~is_basis, "product_type"] = data.loc[~is_basis, "product_type"].fillna("Interest Rate Swap")

            frames.append(_add_position_columns(data))
            continue

        if sheet_key in {"structured_notes", "structured notes", "notes_structurees"}:
            data = data.copy()
            if "product_type" not in data.columns:
                data["product_type"] = pd.NA
            data["product_type"] = data.apply(
                lambda row: row["product_type"]
                if pd.notna(row.get("product_type"))
                else _infer_structured_note_product_type(row),
                axis=1,
            )
            frames.append(_add_position_columns(data))
            continue

        frames.append(_add_position_columns(data))

    pricing_inventory = combine_inventory_sheets(
        {f"pricing_{i}": frame for i, frame in enumerate(frames)}
    )
    return _ensure_pricing_inventory_metadata(pricing_inventory)


def _build_pricing_autocalls(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        return data

    group_key = "product_id" if "product_id" in data.columns else None
    groups = data.groupby(group_key, dropna=False, sort=False) if group_key else [(None, data)]
    rows = []

    for _, group in groups:
        group = group.sort_values("observation_date").copy()
        first = group.iloc[0].copy()

        valuation_date = pd.Timestamp(first["valuation_date"]).normalize()
        observation_dates = pd.to_datetime(group["observation_date"], errors="coerce")
        observation_times = (observation_dates - valuation_date).dt.days / DEFAULT_YEAR_BASIS

        first["product_type"] = "Autocall"
        first["maturity_date"] = observation_dates.max()
        first["time_to_maturity_days"] = int((first["maturity_date"] - valuation_date).days)
        first["time_to_maturity_years"] = float(first["time_to_maturity_days"] / DEFAULT_YEAR_BASIS)
        first["observation_dates"] = observation_times.astype(float).tolist()
        first["trigger_levels"] = group["autocall_trigger_level"].astype(float).tolist()

        # In the source workbook the coupon column often looks cumulative by
        # observation. Convert final cumulative coupon to an annualized coupon
        # for the simplified AutocallProduct.
        maturity = max(float(first["time_to_maturity_years"]), 1e-12)
        final_coupon = float(pd.to_numeric(group["coupon_rate"], errors="coerce").dropna().iloc[-1])
        first["coupon_rate"] = final_coupon / maturity

        if "barrier_protection" not in first or pd.isna(first.get("barrier_protection")):
            first["barrier_protection"] = 0.70

        if "booking_notional" in first and pd.notna(first.get("booking_notional")):
            first["notional"] = float(first["booking_notional"])
        elif "notional" not in first or pd.isna(first.get("notional")):
            first["notional"] = 100.0

        rows.append(first)

    return _add_position_columns(pd.DataFrame(rows))


def _infer_structured_note_product_type(row: pd.Series) -> str:
    cap = row.get("cap", pd.NA)
    barrier_1 = row.get("barrier_1", pd.NA)
    sspa_code = row.get("sspa_code", pd.NA)

    try:
        code = int(float(sspa_code))
    except Exception:
        code = None

    if pd.notna(cap):
        return "Capped Capital Protected Note"
    if pd.notna(barrier_1):
        return "Reverse Convertible"

    if code is not None:
        if 1100 <= code < 1200:
            return "Capital Protected Note"
        if 1200 <= code < 1400:
            return "Reverse Convertible"

    return "Capital Protected Note"


def _add_position_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    raw_size = None
    for column in ("quantity", "notional"):
        if column in data.columns:
            raw_size = pd.to_numeric(data[column], errors="coerce")
            break

    if raw_size is None:
        raw_size = pd.Series(1.0, index=data.index)

    raw_size = raw_size.fillna(1.0)
    sign = raw_size.apply(lambda x: -1.0 if float(x) < 0.0 else 1.0)
    abs_size = raw_size.abs()

    data["position_sign"] = sign.astype(float)
    data["position_size"] = abs_size.astype(float)

    if "quantity" in data.columns:
        data["quantity"] = abs_size
    elif "notional" in data.columns:
        data["notional"] = abs_size

    return data

def _ensure_pricing_inventory_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Final metadata pass for the product-level pricing inventory."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    data = frame.copy()

    if "portfolio" not in data.columns:
        data["portfolio"] = DEFAULT_PORTFOLIO

    data["portfolio"] = (
        data["portfolio"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .fillna(DEFAULT_PORTFOLIO)
    )

    if "currency" not in data.columns:
        data["currency"] = pd.NA

    source_sheet = (
        data["source_sheet"].astype("string").str.strip().str.lower()
        if "source_sheet" in data.columns
        else pd.Series("", index=data.index, dtype="string")
    )

    default_currency = source_sheet.map(_default_currency_for_sheet).fillna("EUR")

    if "rate_currency" in data.columns:
        data["currency"] = data["currency"].fillna(data["rate_currency"])

    data["currency"] = (
        data["currency"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
        .fillna(default_currency)
    )

    return data


__all__ = [
    "build_inventory_data_assets",
    "combine_inventory_sheets",
    "inventory_dataset_summary",
    "load_inventory_workbook",
    "normalize_inventory_sheet",
    "stage_inventory_source",
    "build_pricing_inventory",
]
