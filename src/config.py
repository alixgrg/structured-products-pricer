"""Central project configuration and canonical paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Resolve canonical project paths and default external datasets."""

    project_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    notebooks_dir: Path
    reports_dir: Path
    tests_dir: Path
    rate_curves_source: Path
    options_source: Path
    inventory_source: Path

    @classmethod
    def default(cls, project_root: Path | None = None) -> "ProjectConfig":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        project_dir = root.parent
        course_dir = project_dir.parent
        data_dir = root / "data"

        return cls(
            project_root=root,
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            interim_dir=data_dir / "interim",
            processed_dir=data_dir / "processed",
            notebooks_dir=root / "notebooks",
            reports_dir=root / "reports",
            tests_dir=root / "tests",
            rate_curves_source=Path(
                os.getenv(
                    "STRUCT_PRICER_RATE_CURVES_SOURCE",
                    str(course_dir / "1.rate_curves.parquet"),
                )
            ),
            options_source=Path(
                os.getenv(
                    "STRUCT_PRICER_OPTIONS_SOURCE",
                    str(course_dir / "2.options.csv"),
                )
            ),
            inventory_source=Path(
                os.getenv(
                    "STRUCT_PRICER_INVENTORY_SOURCE",
                    str(project_dir / "Inventaire.xlsx"),
                )
            ),
        )

    def ensure_directories(self) -> None:
        """Create the project directories used by the data pipeline."""
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.notebooks_dir,
            self.reports_dir,
            self.tests_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def raw_rate_curves_path(self) -> Path:
        return self.raw_dir / "rate_curves.parquet"

    @property
    def raw_options_path(self) -> Path:
        return self.raw_dir / "options.csv"

    @property
    def raw_inventory_path(self) -> Path:
        return self.raw_dir / "inventory.xlsx"

    @property
    def interim_rate_curves_path(self) -> Path:
        return self.interim_dir / "rate_curves_normalized.csv"

    @property
    def interim_options_path(self) -> Path:
        return self.interim_dir / "options_normalized.csv"

    @property
    def processed_market_summary_path(self) -> Path:
        return self.processed_dir / "market_dataset_summary.csv"

    @property
    def processed_inventory_summary_path(self) -> Path:
        return self.processed_dir / "inventory_dataset_summary.csv"

    def interim_inventory_path(self, sheet_name: str) -> Path:
        from src.convention import to_snake_case

        return self.interim_dir / f"inventory_{to_snake_case(sheet_name)}.csv"


__all__ = ["ProjectConfig"]
