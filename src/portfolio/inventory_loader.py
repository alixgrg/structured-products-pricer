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
from shutil import copy2
from typing import Iterable

import pandas as pd

from src.config import ProjectConfig
from src.convention import DEFAULT_YEAR_BASIS, canonicalize_columns, strip_accents, to_snake_case


# ---------------------------------------------------------------------------
# Sheet configuration
# ---------------------------------------------------------------------------


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
        "date_valorisation": "valuation_date",
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
        "date_valorisation": "valuation_date",
        "id_produit": "product_id",
        "date_observation": "observation_date",
        "niveau_de_rappel": "autocall_trigger_level",
        "date_reference": "reference_date",
        "coupon": "coupon_rate",
        "sous_jacent": "underlying",
    },
    "structured_notes": {
        "date_valorisation": "valuation_date",
        "code_produit_sspa": "sspa_code",
        "quantite": "quantity",
        "sous_jacent": "underlying",
        "taux_de_participation": "participation_rate",
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
        "currency",
        "fixed_leg_frequency",
        "floating_rate_index_1",
        "floating_rate_index_2",
    ),
    "options": ("product_type", "underlying", "barrier_type"),
    "autocalls": ("underlying",),
    "structured_notes": ("underlying", "rate_currency"),
}

_UPPERCASE_TEXT_COLUMNS = {
    "currency",
    "rate_currency",
    "underlying",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_path(source: str | Path) -> Path:
    """Return a resolved Path without requiring the file to exist."""
    return Path(source).expanduser().resolve()


def _require_file(path: Path, dataset_name: str) -> None:
    """Raise a clear error if an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {path}. "
            "Check ProjectConfig paths or set STRUCT_PRICER_INVENTORY_SOURCE."
        )
    if not path.is_file():
        raise ValueError(f"{dataset_name} path is not a file: {path}")


def _require_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Ensure a dataframe contains expected columns."""
    required = set(required_columns)
    available = set(frame.columns)
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns are: {sorted(available)}"
        )


def _copy_if_needed(source: Path, target: Path, overwrite: bool) -> Path:
    """Copy a source file to a target path if needed."""
    _require_file(source, "inventory source")
    target.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not target.exists():
        copy2(source, target)

    return target


def _prefer_existing_raw_source(raw_path: Path, external_path: Path) -> Path:
    """Use the repository raw copy when present, otherwise fall back to course data."""
    return raw_path if raw_path.exists() else external_path


def _normalize_sheet_name(sheet_name: str) -> str:
    """Map free-form workbook sheet names to canonical sheet keys."""
    key = to_snake_case(strip_accents(sheet_name))
    return _SHEET_ALIASES.get(key, key)


def _normalize_datetime(series: pd.Series) -> pd.Series:
    """Parse a date-like series into timezone-naive normalized datetimes."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None).dt.normalize()


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values, accepting decimal commas and percentages."""
    as_string = series.astype("string").str.strip()

    is_percent = as_string.str.endswith("%", na=False)
    cleaned = (
        as_string.str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )

    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric.loc[is_percent] = numeric.loc[is_percent] / 100.0

    return numeric


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


__all__ = [
    "build_inventory_data_assets",
    "combine_inventory_sheets",
    "inventory_dataset_summary",
    "load_inventory_workbook",
    "normalize_inventory_sheet",
    "stage_inventory_source",
]
