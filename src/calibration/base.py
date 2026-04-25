"""Shared calibration primitives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CalibrationResult:
    """Small container for a calibration output."""

    model_name: str
    parameters: dict[str, float] = field(default_factory=dict)
    objective_value: float | None = None


__all__ = ["CalibrationResult"]
