"""Load, validate and normalize market datasets.

This module handles the market data layer:
- rate curves from a parquet file,
- option quotes from a semicolon-separated CSV file,
- staging from external source files into data/raw,
- normalized exports into data/interim,
- compact summaries into data/processed.

Expected data flow:
external -> raw -> interim -> processed
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.convention import DEFAULT_YEAR_BASIS, canonicalize_columns, tenor_to_years
from src.io_utils import (
    as_path as _as_path,
    copy_if_needed as _copy_if_needed,
    normalize_datetime as _normalize_datetime,
    prefer_existing_raw_source as _prefer_existing_raw_source,
    require_columns as _require_columns,
    require_file as _require_file,
    to_numeric as _to_numeric,
)


def _normalize_unix_or_datetime(series: pd.Series) -> pd.Series:
    """Parse either unix timestamps in seconds or normal date-like values.

    The options file usually stores expiration / update dates as unix timestamps.
    This helper remains robust if a future file contains standard dates instead.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # If most non-null values look numeric, interpret them as unix seconds.
    non_null_numeric_ratio = numeric.notna().mean() if len(series) else 0.0
    if non_null_numeric_ratio >= 0.8:
        parsed = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)

    return parsed.dt.tz_localize(None)


def _to_boolean(series: pd.Series) -> pd.Series:
    """Parse booleans from Python booleans, 0/1, true/false, yes/no."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "vrai": True,
        "faux": False,
    }
    return normalized.map(mapping).astype("boolean")


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical snake_case columns."""
    normalized = frame.copy()
    normalized.columns = canonicalize_columns(normalized.columns)
    return normalized


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def stage_market_sources(
    config: ProjectConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Copy external market datasets into the repository raw layer."""
    cfg = config or ProjectConfig.default()
    cfg.ensure_directories()

    return {
        "rate_curves": _copy_if_needed(
            _as_path(cfg.rate_curves_source),
            cfg.raw_rate_curves_path,
            overwrite,
        ),
        "options": _copy_if_needed(
            _as_path(cfg.options_source),
            cfg.raw_options_path,
            overwrite,
        ),
    }


# ---------------------------------------------------------------------------
# Rate curves
# ---------------------------------------------------------------------------


_RATE_CURVE_COLUMN_ALIASES = {
    "maturity": "curve_tenor",
    "tenor": "curve_tenor",
    "curve_maturity": "curve_tenor",
    "date": "observation_date",
    "valuation_date": "observation_date",
    "as_of_date": "observation_date",
    "rate": "rate_percent",
    "zero_rate": "rate_percent",
    "yield": "rate_percent",
}

_RATE_CURVE_REQUIRED_BASE = {"country", "curve_tenor", "observation_date"}


def normalize_rate_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize the raw rate curve parquet file.

    Output schema:
    - country
    - curve_tenor
    - curve_tenor_years
    - observation_date
    - rate_percent
    - rate_decimal
    """
    normalized = _standardize_columns(frame)
    normalized = normalized.rename(columns=_RATE_CURVE_COLUMN_ALIASES)

    _require_columns(normalized, _RATE_CURVE_REQUIRED_BASE, "rate_curves")

    if "rate_percent" not in normalized.columns and "rate_decimal" not in normalized.columns:
        raise ValueError(
            "rate_curves must contain either 'rate_percent'/'rate' or 'rate_decimal'. "
            f"Available columns are: {sorted(normalized.columns)}"
        )

    normalized["country"] = normalized["country"].astype("string").str.strip()
    normalized["curve_tenor"] = (
        normalized["curve_tenor"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    normalized["curve_tenor_years"] = normalized["curve_tenor"].map(tenor_to_years)
    normalized["observation_date"] = _normalize_datetime(normalized["observation_date"])

    if "rate_percent" in normalized.columns:
        normalized["rate_percent"] = _to_numeric(normalized["rate_percent"])
        normalized["rate_decimal"] = normalized["rate_percent"] / 100.0
    else:
        normalized["rate_decimal"] = _to_numeric(normalized["rate_decimal"])
        normalized["rate_percent"] = normalized["rate_decimal"] * 100.0

    normalized = normalized.dropna(
        subset=[
            "country",
            "curve_tenor",
            "curve_tenor_years",
            "observation_date",
            "rate_decimal",
        ]
    )

    normalized = normalized.sort_values(
        by=["country", "observation_date", "curve_tenor_years", "curve_tenor"],
        ignore_index=True,
    )

    return normalized[
        [
            "country",
            "curve_tenor",
            "curve_tenor_years",
            "observation_date",
            "rate_percent",
            "rate_decimal",
        ]
    ]


def load_rate_curves(
    source: str | Path | None = None,
    *,
    config: ProjectConfig | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Read the raw rate curve dataset."""
    cfg = config or ProjectConfig.default()
    default_source = _prefer_existing_raw_source(cfg.raw_rate_curves_path, cfg.rate_curves_source)
    path = _as_path(source) if source is not None else _as_path(default_source)
    _require_file(path, "rate_curves")

    frame = pd.read_parquet(path)
    return normalize_rate_curves(frame) if normalize else frame


# ---------------------------------------------------------------------------
# Option quotes
# ---------------------------------------------------------------------------


_OPTION_COLUMN_ALIASES = {
    "option_symbol": "contract_symbol",
    "contract": "contract_symbol",
    "symbol": "contract_symbol",
    "side": "option_type",
    "type": "option_type",
    "date": "valuation_date",
    "as_of_date": "valuation_date",
    "valuation": "valuation_date",
    "iv": "implied_volatility",
    "implied_vol": "implied_volatility",
    "underlying_px": "underlying_price",
    "spot": "underlying_price",
    "expiration": "expiration",
    "expiry": "expiration",
    "expiry_date": "expiration_date",
    "expiration_date": "expiration_date",
    "dte": "time_to_maturity_days",
}

_OPTION_REQUIRED = {
    "contract_symbol",
    "underlying",
    "option_type",
    "valuation_date",
    "strike",
    "underlying_price",
}

_OPTION_NUMERIC_COLUMNS = [
    "strike",
    "bid",
    "mid",
    "ask",
    "last",
    "underlying_price",
    "implied_volatility",
    "intrinsic_value",
    "extrinsic_value",
    "bid_size",
    "ask_size",
    "open_interest",
    "volume",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "time_to_maturity_days",
]

_OPTION_OUTPUT_COLUMNS = [
    "contract_symbol",
    "ticker",
    "underlying",
    "option_type",
    "valuation_date",
    "expiration_date",
    "time_to_maturity_days",
    "time_to_maturity_years",
    "strike",
    "bid",
    "mid",
    "ask",
    "last",
    "underlying_price",
    "implied_volatility",
    "intrinsic_value",
    "extrinsic_value",
    "bid_size",
    "ask_size",
    "open_interest",
    "volume",
    "in_the_money",
    "first_traded_at",
    "last_updated_at",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
]


def _read_csv_with_fallback(path: Path, sep: str | None) -> pd.DataFrame:
    """Read CSV with a preferred separator, then infer if needed."""
    if sep is not None:
        return pd.read_csv(path, sep=sep)

    try:
        return pd.read_csv(path, sep=";")
    except pd.errors.ParserError:
        return pd.read_csv(path, sep=None, engine="python")


def normalize_option_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize the raw option chain CSV file.

    Output schema keeps the important market fields and optional Greeks if present.
    """
    normalized = _standardize_columns(frame)
    normalized = normalized.rename(columns=_OPTION_COLUMN_ALIASES)

    # If ticker is absent, use the underlying as ticker.
    if "ticker" not in normalized.columns and "underlying" in normalized.columns:
        normalized["ticker"] = normalized["underlying"]

    # Handle expiration either as unix timestamp or already parsed date.
    if "expiration_date" not in normalized.columns and "expiration" in normalized.columns:
        normalized["expiration_date"] = _normalize_unix_or_datetime(
            normalized["expiration"]
        ).dt.normalize()

    if "first_traded_at" not in normalized.columns and "first_traded" in normalized.columns:
        normalized["first_traded_at"] = _normalize_unix_or_datetime(
            normalized["first_traded"]
        )

    if "last_updated_at" not in normalized.columns and "updated" in normalized.columns:
        normalized["last_updated_at"] = _normalize_unix_or_datetime(
            normalized["updated"]
        )

    _require_columns(normalized, _OPTION_REQUIRED, "options")

    normalized["contract_symbol"] = normalized["contract_symbol"].astype("string").str.strip()
    normalized["ticker"] = normalized["ticker"].astype("string").str.strip().str.upper()
    normalized["underlying"] = normalized["underlying"].astype("string").str.strip().str.upper()
    normalized["option_type"] = (
        normalized["option_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    normalized["valuation_date"] = _normalize_datetime(normalized["valuation_date"])

    # Numeric conversion for all known numeric columns if present.
    for column in _OPTION_NUMERIC_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _to_numeric(normalized[column])

    # Mid price fallback if missing.
    if "mid" not in normalized.columns and {"bid", "ask"}.issubset(normalized.columns):
        normalized["mid"] = (normalized["bid"] + normalized["ask"]) / 2.0

    # Maturity fallback if dte is absent but dates are available.
    if "time_to_maturity_days" not in normalized.columns:
        if "expiration_date" in normalized.columns:
            normalized["time_to_maturity_days"] = (
                normalized["expiration_date"] - normalized["valuation_date"]
            ).dt.days
        else:
            normalized["time_to_maturity_days"] = pd.NA

    normalized["time_to_maturity_years"] = (
        _to_numeric(normalized["time_to_maturity_days"]) / DEFAULT_YEAR_BASIS
    )

    if "in_the_money" in normalized.columns:
        normalized["in_the_money"] = _to_boolean(normalized["in_the_money"])
    else:
        normalized["in_the_money"] = pd.Series(pd.NA, index=normalized.index, dtype="boolean")

    # Create optional output columns if they do not exist.
    for column in _OPTION_OUTPUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized.dropna(
        subset=[
            "contract_symbol",
            "underlying",
            "option_type",
            "valuation_date",
            "strike",
            "underlying_price",
        ]
    )

    normalized = normalized[
        normalized["option_type"].isin(["call", "put", "c", "p"])
    ].copy()
    normalized["option_type"] = normalized["option_type"].replace({"c": "call", "p": "put"})

    normalized = normalized.sort_values(
        by=["valuation_date", "underlying", "expiration_date", "option_type", "strike"],
        ignore_index=True,
    )

    return normalized[_OPTION_OUTPUT_COLUMNS]


def load_option_quotes(
    source: str | Path | None = None,
    *,
    config: ProjectConfig | None = None,
    normalize: bool = True,
    sep: str | None = ";",
) -> pd.DataFrame:
    """Read the raw option chain CSV file.

    Parameters
    ----------
    source:
        Optional explicit CSV path.
    config:
        Project configuration.
    normalize:
        Whether to return normalized columns.
    sep:
        CSV separator. The course file uses ';'. Use None to infer.
    """
    cfg = config or ProjectConfig.default()
    default_source = _prefer_existing_raw_source(cfg.raw_options_path, cfg.options_source)
    path = _as_path(source) if source is not None else _as_path(default_source)
    _require_file(path, "options")

    frame = _read_csv_with_fallback(path, sep=sep)
    return normalize_option_quotes(frame) if normalize else frame


# ---------------------------------------------------------------------------
# Summaries and build pipeline
# ---------------------------------------------------------------------------


def market_dataset_summary(
    rate_curves: pd.DataFrame,
    option_quotes: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact processed summary of the market datasets."""
    rows = []

    if not rate_curves.empty:
        rows.append(
            {
                "dataset": "rate_curves",
                "rows": len(rate_curves),
                "columns": len(rate_curves.columns),
                "min_date": rate_curves["observation_date"].min(),
                "max_date": rate_curves["observation_date"].max(),
                "entity_count": rate_curves["country"].nunique(),
                "missing_cells": int(rate_curves.isna().sum().sum()),
            }
        )
    else:
        rows.append(
            {
                "dataset": "rate_curves",
                "rows": 0,
                "columns": len(rate_curves.columns),
                "min_date": pd.NaT,
                "max_date": pd.NaT,
                "entity_count": 0,
                "missing_cells": 0,
            }
        )

    if not option_quotes.empty:
        rows.append(
            {
                "dataset": "options",
                "rows": len(option_quotes),
                "columns": len(option_quotes.columns),
                "min_date": option_quotes["valuation_date"].min(),
                "max_date": option_quotes["valuation_date"].max(),
                "entity_count": option_quotes["underlying"].nunique(),
                "missing_cells": int(option_quotes.isna().sum().sum()),
            }
        )
    else:
        rows.append(
            {
                "dataset": "options",
                "rows": 0,
                "columns": len(option_quotes.columns),
                "min_date": pd.NaT,
                "max_date": pd.NaT,
                "entity_count": 0,
                "missing_cells": 0,
            }
        )

    return pd.DataFrame(rows)


def build_market_data_assets(
    config: ProjectConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Populate raw, interim and processed folders for market datasets."""
    cfg = config or ProjectConfig.default()
    stage_market_sources(cfg, overwrite=overwrite)

    # Read from raw layer after staging to make the pipeline reproducible.
    rate_curves = load_rate_curves(cfg.raw_rate_curves_path, config=cfg)
    option_quotes = load_option_quotes(cfg.raw_options_path, config=cfg)
    summary = market_dataset_summary(rate_curves, option_quotes)

    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    rate_curves.to_csv(cfg.interim_rate_curves_path, index=False)
    option_quotes.to_csv(cfg.interim_options_path, index=False)
    summary.to_csv(cfg.processed_market_summary_path, index=False)

    return {
        "rate_curves_raw": cfg.raw_rate_curves_path,
        "options_raw": cfg.raw_options_path,
        "rate_curves_interim": cfg.interim_rate_curves_path,
        "options_interim": cfg.interim_options_path,
        "market_summary": cfg.processed_market_summary_path,
    }


__all__ = [
    "build_market_data_assets",
    "load_option_quotes",
    "load_rate_curves",
    "market_dataset_summary",
    "normalize_option_quotes",
    "normalize_rate_curves",
    "stage_market_sources",
]
