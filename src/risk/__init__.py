"""Risk package exports."""

from importlib import import_module

_EXPORT_MODULES = {
    "RiskAggregator": "src.risk.aggregator",
    "NumericalGreeksConfig": "src.risk.numerical_greeks",
    "NumericalGreeksEngine": "src.risk.numerical_greeks",
    "PortfolioRiskReport": "src.risk.report",
    "PortfolioRiskSummary": "src.risk.report",
    "RiskSnapshot": "src.risk.report",
    "aggregate_greeks": "src.risk.report",
    "build_portfolio_risk_summary": "src.risk.report",
    "risk_pivot_table": "src.risk.report",
    "ShiftedVolSurface": "src.risk.stress_testing",
    "StressScenario": "src.risk.stress_testing",
    "StressTestResult": "src.risk.stress_testing",
    "StressTester": "src.risk.stress_testing",
}

__all__ = [
    "NumericalGreeksConfig",
    "NumericalGreeksEngine",
    "PortfolioRiskReport",
    "PortfolioRiskSummary",
    "RiskAggregator",
    "RiskSnapshot",
    "ShiftedVolSurface",
    "StressScenario",
    "StressTestResult",
    "StressTester",
    "aggregate_greeks",
    "build_portfolio_risk_summary",
    "risk_pivot_table",
]


def __getattr__(name: str):
    if name in _EXPORT_MODULES:
        module = import_module(_EXPORT_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
