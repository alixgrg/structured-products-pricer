from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.market.loaders import load_option_quotes, load_rate_curves
from src.portfolio.inventory_loader import load_inventory_workbook


def test_option_loader_reads_semicolon_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "options.csv"
    csv_path.write_text(
        "\n".join(
            [
                "optionSymbol;underlying;expiration;side;strike;firstTraded;dte;updated;bid;bidSize;mid;ask;askSize;last;openInterest;volume;inTheMoney;intrinsicValue;extrinsicValue;underlyingPrice;iv;delta;gamma;theta;vega;ticker;date",
                "AAPL260630C00260000;AAPL;1782777600;call;260;1780000000;30;1780100000;10.0;5;10.5;11.0;5;10.3;100;10;False;0.0;10.5;255.0;0.24;;;;;AAPL;2026-05-31",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_option_quotes(csv_path)

    assert frame.columns.tolist()[:5] == [
        "contract_symbol",
        "ticker",
        "underlying",
        "option_type",
        "valuation_date",
    ]
    assert frame.loc[0, "strike"] == 260.0
    assert frame.loc[0, "time_to_maturity_days"] == 30
    assert str(frame.loc[0, "valuation_date"].date()) == "2026-05-31"


def test_inventory_loader_normalizes_all_sheets(tmp_path: Path) -> None:
    workbook_path = tmp_path / "inventory.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Date Valorisation": ["2026-02-27"],
                "Devise": ["EUR"],
                "Maturité": ["2026-12-31"],
                "Fréquence fixe": ["3M"],
                "Nominal": [1_000_000],
                "Taux fixe": [0.05],
                "Taux Variable 1": ["3M"],
                "Taux variable 2": [None],
            }
        ).to_excel(writer, sheet_name="Swap", index=False)
        pd.DataFrame(
            {
                "Date Valorisation": ["2026-02-27"],
                "Produit": ["Call Spread"],
                "Quantité": [100_000],
                "Sous jacent": ["AAPL"],
                "Maturité": ["2026-06-30"],
                "Strike 1": [260],
                "Strike 2": [280],
                "Strike 3": [None],
                "Type Barrière": [None],
                "Niveau Barrière": [None],
            }
        ).to_excel(writer, sheet_name="Options", index=False)

    inventory = load_inventory_workbook(workbook_path)

    assert set(inventory) == {"swaps", "options"}
    assert "valuation_date" in inventory["swaps"].columns
    assert inventory["options"].loc[0, "underlying"] == "AAPL"
    assert inventory["options"].loc[0, "strike_2"] == 280


def test_rate_curve_loader_reads_parquet(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    assert pyarrow is not None

    parquet_path = tmp_path / "rate_curves.parquet"
    pd.DataFrame(
        {
            "country": ["France", "France"],
            "maturity": ["1M", "1Y"],
            "date": ["2026-02-27", "2026-02-27"],
            "rate": [2.4, 2.8],
        }
    ).to_parquet(parquet_path, index=False)

    frame = load_rate_curves(parquet_path)

    assert frame.columns.tolist() == [
        "country",
        "curve_tenor",
        "curve_tenor_years",
        "observation_date",
        "rate_percent",
        "rate_decimal",
    ]
    assert frame.loc[0, "rate_decimal"] == pytest.approx(0.024)
    assert frame.loc[1, "curve_tenor_years"] == pytest.approx(1.0)
