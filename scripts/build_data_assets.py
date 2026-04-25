from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig
from src.market.loaders import build_market_data_assets
from src.portfolio.inventory_loader import build_inventory_data_assets

def main() -> None:
    cfg = ProjectConfig.default()
    cfg.ensure_directories()

    # On utilise les fichiers déjà présents dans data/raw comme sources.
    cfg = replace(
        cfg,
        rate_curves_source=cfg.raw_rate_curves_path,
        options_source=cfg.raw_options_path,
        inventory_source=cfg.raw_inventory_path,
    )

    outputs: dict[str, Path] = {}
    outputs.update(build_market_data_assets(cfg))
    outputs.update(build_inventory_data_assets(cfg))

    print("Generated data assets:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()