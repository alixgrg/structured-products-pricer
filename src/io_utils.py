"""Shared IO and dataframe normalization helpers."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import Iterable

import pandas as pd


def as_path(source: str | Path) -> Path:
    """Return a resolved Path without requiring the file to exist."""
    return Path(source).expanduser().resolve()


def require_file(
    path: Path,
    dataset_name: str,
    *,
    hint: str = "Check ProjectConfig paths or set the relevant environment variable.",
) -> None:
    """Raise a clear error if an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {path}. {hint}")
    if not path.is_file():
        raise ValueError(f"{dataset_name} path is not a file: {path}")


def require_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Ensure a dataframe contains the expected columns."""
    required = set(required_columns)
    available = set(frame.columns)
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns are: {sorted(available)}"
        )


def copy_if_needed(
    source: Path,
    target: Path,
    overwrite: bool,
    *,
    dataset_name: str = "source",
) -> Path:
    """Copy a source file to a target path if needed."""
    require_file(source, dataset_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not target.exists():
        copy2(source, target)

    return target


def prefer_existing_raw_source(raw_path: Path, external_path: Path) -> Path:
    """Use the repository raw copy when present, otherwise fall back to course data."""
    return raw_path if raw_path.exists() else external_path


def normalize_datetime(series: pd.Series) -> pd.Series:
    """Parse a date-like series into timezone-naive normalized datetimes."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None).dt.normalize()


def to_numeric(series: pd.Series) -> pd.Series:
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

