"""Lightweight public facade for the structured-products pricer."""

from src.config import ProjectConfig
from src.convention import (
    DATA_FLOW_STAGES,
    DEFAULT_YEAR_BASIS,
    PROJECT_CONVENTIONS,
    NamingConvention,
    canonicalize_columns,
    strip_accents,
    tenor_to_years,
    to_snake_case,
)

__all__ = [
    "DATA_FLOW_STAGES",
    "DEFAULT_YEAR_BASIS",
    "PROJECT_CONVENTIONS",
    "NamingConvention",
    "ProjectConfig",
    "canonicalize_columns",
    "strip_accents",
    "tenor_to_years",
    "to_snake_case",
]
