"""Project-wide naming and data flow conventions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

DATA_FLOW_STAGES = ("external", "raw", "interim", "processed")
DEFAULT_YEAR_BASIS = 365.25

_NON_ALNUM_PATTERN = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL_CASE_PATTERN = re.compile(r"([a-z0-9])([A-Z])")
_TENOR_PATTERN = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[DWMY])\s*$", re.IGNORECASE)


def strip_accents(value: str) -> str:
    """Return an ASCII-only representation of a label."""
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def to_snake_case(value: str) -> str:
    """Convert a free-form label into snake_case."""
    normalized = strip_accents(value).strip()
    normalized = _CAMEL_CASE_PATTERN.sub(r"\1_\2", normalized)
    normalized = _NON_ALNUM_PATTERN.sub("_", normalized)
    normalized = normalized.strip("_")
    return normalized.lower()


def canonicalize_columns(columns: Iterable[str]) -> list[str]:
    """Normalize a collection of column names into snake_case."""
    return [to_snake_case(str(column)) for column in columns]


def tenor_to_years(tenor: str | float | int | None) -> float | None:
    """Convert tenor labels such as 6M or 5Y into year fractions."""
    if tenor is None:
        return None

    if isinstance(tenor, (int, float)):
        return float(tenor)

    match = _TENOR_PATTERN.match(str(tenor))
    if match is None:
        return None

    value = float(match.group("value"))
    unit = match.group("unit").upper()
    if unit == "D":
        return value / 365.25
    if unit == "W":
        return value * 7.0 / 365.25
    if unit == "M":
        return value / 12.0
    return value


@dataclass(frozen=True, slots=True)
class NamingConvention:
    """Canonical naming rules shared across the project."""

    package_style: str = "snake_case"
    class_style: str = "PascalCase"
    function_style: str = "snake_case"
    column_style: str = "snake_case"
    flow_stages: tuple[str, ...] = DATA_FLOW_STAGES
    year_basis: float = DEFAULT_YEAR_BASIS


PROJECT_CONVENTIONS = NamingConvention()

__all__ = [
    "DATA_FLOW_STAGES",
    "DEFAULT_YEAR_BASIS",
    "PROJECT_CONVENTIONS",
    "NamingConvention",
    "canonicalize_columns",
    "strip_accents",
    "tenor_to_years",
    "to_snake_case",
]
