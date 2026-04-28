"""Central project configuration and canonical paths.

The project now assumes that the three market/input files used by the
application live inside ``data/raw`` of the project repository:

- ``data/raw/rate_curves.parquet``
- ``data/raw/options.csv``
- ``data/raw/inventory.xlsx``

Environment variables can still override those defaults when needed, but the
standard path for the notebook and Streamlit application is now ``data/raw``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Resolve canonical project paths and default external datasets.

    Design choice
    -------------
    ``*_source`` points to the effective source file used by the pipeline.
    By default, those files are the canonical raw files inside ``data/raw``.
    This avoids relying on files stored outside the project directory and makes
    the notebook, tests and Streamlit dashboard reproducible from the repository.
    """

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
        """Return the default project configuration.

        Parameters
        ----------
        project_root:
            Root of the repository. If omitted, it is inferred from this file.

        Notes
        -----
        The default source files are now the canonical files in ``data/raw``.
        External overrides remain possible with:

        - ``STRUCT_PRICER_RATE_CURVES_SOURCE``
        - ``STRUCT_PRICER_OPTIONS_SOURCE``
        - ``STRUCT_PRICER_INVENTORY_SOURCE``
        """
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        data_dir = root / "data"
        raw_dir = data_dir / "raw"

        return cls(
            project_root=root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            interim_dir=data_dir / "interim",
            processed_dir=data_dir / "processed",
            notebooks_dir=root / "notebooks",
            reports_dir=root / "reports",
            tests_dir=root / "tests",
            rate_curves_source=Path(
                os.getenv(
                    "STRUCT_PRICER_RATE_CURVES_SOURCE",
                    str(raw_dir / "rate_curves.parquet"),
                )
            ),
            options_source=Path(
                os.getenv(
                    "STRUCT_PRICER_OPTIONS_SOURCE",
                    str(raw_dir / "options.csv"),
                )
            ),
            inventory_source=Path(
                os.getenv(
                    "STRUCT_PRICER_INVENTORY_SOURCE",
                    str(raw_dir / "inventory.xlsx"),
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

    @property
    def dashboard_exports_dir(self) -> Path:
        return self.reports_dir / "dashboard_exports"

    def interim_inventory_path(self, sheet_name: str) -> Path:
        from src.convention import to_snake_case

        return self.interim_dir / f"inventory_{to_snake_case(sheet_name)}.csv"

    def raw_inputs(self) -> dict[str, Path]:
        """Return the canonical raw input files expected by the project."""
        return {
            "rate_curves": self.rate_curves_source,
            "options": self.options_source,
            "inventory": self.inventory_source,
        }

    def raw_input_status(self) -> dict[str, bool]:
        """Return whether each required raw input file is available."""
        return {name: path.exists() for name, path in self.raw_inputs().items()}

    def require_raw_inputs(self) -> None:
        """Raise a clear error if one of the raw input files is missing."""
        missing = [str(path) for name, path in self.raw_inputs().items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required raw input file(s): " + ", ".join(missing)
            )


__all__ = ["ProjectConfig"]
