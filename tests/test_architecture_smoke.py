from src.calibration.base import CalibrationResult
from src.config import ProjectConfig
from src.dashboard.config import DashboardConfig
from src.factory.registry import ProductFactoryRegistry
from src.market.market_data import MarketData
from src.portfolio.pricing_engine import PortfolioPricingConfig
from src.risk.report import RiskSnapshot


def test_project_config_builds_expected_directories() -> None:
    config = ProjectConfig.default()

    assert config.raw_dir.name == "raw"
    assert config.interim_dir.name == "interim"
    assert config.processed_dir.name == "processed"


def test_smoke_instantiation_of_foundation_objects() -> None:
    market_data = MarketData(spot=100.0, rate=0.02, volatility=0.25)
    calibration = CalibrationResult(model_name="black_scholes", parameters={"sigma": 0.25})
    dashboard = DashboardConfig()
    risk_snapshot = RiskSnapshot(product_id="CALL-001", price=12.5, metrics={"delta": 0.55})
    portfolio_config = PortfolioPricingConfig(default_spot=100.0, default_rate=0.03)

    registry = ProductFactoryRegistry()
    registry.register("dummy", lambda product_id: {"product_id": product_id})

    assert market_data.spot == 100.0
    assert calibration.parameters["sigma"] == 0.25
    assert dashboard.default_pages[0] == "overview"
    assert risk_snapshot.metrics["delta"] == 0.55
    assert portfolio_config.default_spot == 100.0
    assert registry.build("dummy", product_id="P-1") == {"product_id": "P-1"}
