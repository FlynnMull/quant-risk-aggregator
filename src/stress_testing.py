"""
src/stress_testing.py

Deterministic Macro Stress-Testing and Historical Scenario Analysis.

Unlike stochastic Monte Carlo simulaton - which samples diffusion paths across a
fitted empirical covariance manifold - this module performs deterministic full 
revaluation under severe, non-linear market dislocations.

Financial & Quantitative Rationale:
-----------------------------------
1. Breakdown of Normality & Correlation Snap:
    During systematic market crises empirical asset correlations rapidly converge
    towards 1.0, implied volatility smiles steepen and asset returns exhibit extreme
    fat-til behaviour that standard diffusion models underestimat.

2. Full Revaluaton vs. Taylor Approximations:
    Standard Delta-Gamma-Vega approximations fail significantly under large market 
    dislocations. This exact analytical Black-Scholes revaluation across all non-linear derivative
    contracts under the shocked market parameters.

3. Delta Drift & Dynamic Convexity:
    Measures the directional exposure shift caused by option gamma and vega re-weighting
    under stressed market states, revealing portfolio tail fragility and hedging requirements.

Factor Shock Vectorisation:
---------------------------
- Multi-Asset Spot Shocks
- Volatility Surface Upward Shifts
- Interest Rate Level Shifts

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from src.data_models import MarketData, Portfolio, SimulationConfig
from src.engine import BlackScholesEngine, RiskEngine

__all__ = [
    "StressScenario",
    "ScenarioResult",
    "StressTestReport",
    "StressTestingEngine",
    "MODERATE_EQUITY_SELLOFF",
    "SEVERE_MARKET_CRASH",
    "STAGFLATION_SHOCK",
    "TECH_DISLOCATION",
    "LEHMAN_2008_REPLAY",
    "COVID_2020_REPLAY",
    "FED_HIKE_2022_REPLAY",
    "PARAMETRIC_SCENARIOS",
    "HISTORICAL_SCENARIOS",
    "ALL_SCENARIOS",
]

SpotShock = Union[float, Mapping[str, float]]
_SPOT_PRICE_FLOOR = 1e-6

# ----------------------------------------------------------------------------
# Scenario Data Contracts
# ----------------------------------------------------------------------------

@dataclass(frozen = True)
class StressScenario:
    """
    Immutable specification of a joint macro factor shock.

    Attributes:
    -----------
    name : str
        Human-readable identifier for the scenario
    description : str
        Economic and financial narrative detailing the market dislocation
    spot_shock : float | Mapping[str, float]
        Relative underlying price shift. Can be a single float applied uniformly
        across all assets or a dictionary mapping specific tickers to shocks 
    vol_shock : float
        Absolute annualised volatility shift added to current volatility levels
    rate_shock : float
        Absolute shift in the risk-free rate in decimal form
    """

    name:str
    description: str
    spot_shock: SpotShock
    vol_shock: float = 0.0
    rate_shock: float = 0.0

    def __post_init__(self) -> None:
        """Validates numerical finiteness and structural consistency of scenario parameters"""
        if not self.name:
            raise ValueError("StressScenario.name must be a non-empty string")
        if not self.description:
            raise ValueError("StressScenario.description must be a non-empty string")
        if not np.isfinite(self.vol_shock):
            raise ValueError("vol_shock must be a finite number")
        if not np.isfinite(self.rate_shock):
            raise ValueError("rate_shock must be a finite number")

        if isinstance(self.spot_shock, Mapping):
            if not self.spot_shock:
                raise ValueError("spot_shock mapping must not be empty")
            if not all(np.isfinite(v) for v in self.spot_shock.values()):
                raise ValueError("all spot_shock mapping values must be finite")
        elif not np.isfinite(self.spot_shock):
            raise ValueError("spot_shock must be a finite number or a ticker mapping")

@dataclass(frozen = True)
class ScenarioResult:
    """
    Quantitative revaluation output of the portfolio under a single StressScenario.

    Attributes:
    -----------
    scenario_name : str
        Identifier of the evaluated scenario
    scenario_description : str
        Narrative context of the shock
    base_value : float
        Unstressed base portfolio net asset value
    stressed_value : float
        Full revaluation portfolio net asset value
    dollar_pn1 : float
        Dollar Profit/Loss
    pct_drawdown : float
        Percentage portfolio drawdown
    delta_drift : pd.Series
        Asset-level shift in Delta directional sensitivity
    """

    scenario_name: str
    scenario_description: str
    base_value: float
    stressed_value: float
    dollar_pn1: float
    pct_drawdown: float
    delta_drift: pd.Series

    def __post_init__(self) -> None:
        """Enforces contiguous float series conversion for delta drift"""
        object.__setattr__(self, "delta_drift", pd.Series(self.delta_drift).astype(float))

    @property
    def total_delta_drift(self) -> float:
        """Total portfolio-level Delta shift"""
        return float(self.delta_drift.sum())

@dataclass
class StressTestReport:
    """
    Aggregated container storing multi-scenaro stress results with tabular reporting
    """
    
    portfolio_name: str
    results: Sequence[ScenarioResult] = field(default_factory = tuple)

    def __post_init__(self) -> None:
        self.results = tuple(self.results)

    def to_frame(self) -> pd.DataFrame:
        """Serialises stress test metrics across all scenarios into a formatted DataFrame"""
        records = [
            {
                "Scenario": r.scenario_name,
                "Description": r.scenario_description,
                "Base Value": r.base_value,
                "Stressed Value": r.stressed_value,
                "Dollar PnL": r.dollar_pn1,
                "% Drawdown": r.pct_drawdown * 100.0,
                "Total Delta Drift": r.total_delta_drift,
            }
            for r in self.results
        ]
        return pd.DataFrame.from_records(records).set_index("Scenario")

    def worst_scenario(self) -> ScenarioResult | None:
        """Identifies the maximum loss scenario across the report"""
        if not self.results:
            return None
        return min(self.results, key = lambda r: r.dollar_pn1)

# ----------------------------------------------------------------------------
# Pre-Configured Instituational Stress Scenarios
# ----------------------------------------------------------------------------

# --- 1. Parametric Macro shocks -------------------------------------------------------

MODERATE_EQUITY_SELLOFF = StressScenario(
    name="Moderate Equity Sell-Off",
    description="Broad-based equity correction with a modest implied-vol pop.",
    spot_shock=-0.10,
    vol_shock=0.05,
    rate_shock=0.0,
)

SEVERE_MARKET_CRASH = StressScenario(
    name="Severe Market Crash",
    description="Sharp, broad equity crash with a volatility spike and emergency rate cuts.",
    spot_shock=-0.20,
    vol_shock=0.20,
    rate_shock=-0.0050,
)

STAGFLATION_SHOCK = StressScenario(
    name="Stagflation Shock",
    description="Equities sell off while sticky inflation forces the risk-free rate higher.",
    spot_shock=-0.15,
    vol_shock=0.10,
    rate_shock=0.0150,
)

TECH_DISLOCATION = StressScenario(
    name="Tech Dislocation",
    description="Concentrated mega-cap tech unwind (-25%) alongside a broader market "
                "drawdown (-10%) for all other names.",
    spot_shock={
        "AAPL": -0.25, "MSFT": -0.25, "NVDA": -0.25, "GOOGL": -0.25,
        "GOOG": -0.25, "AMZN": -0.25, "META": -0.25, "TSLA": -0.25,
        "DEFAULT": -0.10,
    },
    vol_shock=0.15,
    rate_shock=0.0,
)

PARAMETRIC_SCENARIOS: tuple[StressScenario, ...] = (
    MODERATE_EQUITY_SELLOFF,
    SEVERE_MARKET_CRASH,
    STAGFLATION_SHOCK,
    TECH_DISLOCATION,
)

# --- 2. Historical Crisis Replays -----------------------------------------------

LEHMAN_2008_REPLAY = StressScenario(
    name="2008 Lehman Collapse Replay",
    description="Global equity crash and credit freeze following the Lehman Brothers "
                "bankruptcy, with the Fed slashing rates in response.",
    spot_shock=-0.20,
    vol_shock=0.25,
    rate_shock=-0.0200,
)

COVID_2020_REPLAY = StressScenario(
    name="March 2020 Covid Liquidity Shock Replay",
    description="Fastest-ever bear market on record as COVID-19 lockdowns triggered a "
                "liquidity crunch and a historic VIX spike, met with emergency rate cuts.",
    spot_shock=-0.30,
    vol_shock=0.40,
    rate_shock=-0.0150,
)

FED_HIKE_2022_REPLAY = StressScenario(
    name="2022 Fed Rate Hike Sell-Off Replay",
    description="Sustained equity de-rating as the Fed raised rates aggressively to "
                "combat multi-decade-high inflation.",
    spot_shock=-0.20,
    vol_shock=0.10,
    rate_shock=0.0300,
)

HISTORICAL_SCENARIOS: tuple[StressScenario, ...] = (
    LEHMAN_2008_REPLAY,
    COVID_2020_REPLAY,
    FED_HIKE_2022_REPLAY,
)

ALL_SCENARIOS: tuple[StressScenario, ...] = PARAMETRIC_SCENARIOS + HISTORICAL_SCENARIOS

# ----------------------------------------------------------------------------
# Deterministic Stress Engine
# ----------------------------------------------------------------------------

@dataclass
class StressTestingEngine:
    """
    Deterministic full-revaluation stress-testing engine.

    Computational Workflow:
    -----------------------
    1. Base State Evaluation
        Calculates baseline net asset value and asset-level Delta vector
    2. Stressted Market Synthesis
        Constructs a new, fully validated 'MarketData' contract per scenario with shocked spots, vols and rates
    3. Non-Linear Revaluation
        Revalues linear equities and option contracts via analytical 'BlackScholesEngine' under the stressed state.
    4. Impact Metric Calculations
    """

    market_data: MarketData
    portfolio: Portfolio
    scenarios: Sequence[StressScenario]
    bs_engine: BlackScholesEngine = field(default_factory = BlackScholesEngine)

    def __post_init__(self) -> None:
        """Initialises baseline evaluation engine and validates input scenario sequences"""
        self.scenarios = tuple(self.scenarios)
        if not self.scenarios:
            raise ValueError("StressTestingEngine at least one StressScenario")

        self._proxy_config = SimulationConfig(
            n_paths = 2, time_horizon = 1.0 / 365.0, n_steps = 1,
            random_seed = 0, antithetic = True, confidence_levels = (0.05,),
        )

        self._base_engine = RiskEngine(
            market_data = self.market_data, portfolio = self.portfolio,
            config = self._proxy_config, bs_engine = self.bs_engine,
        )
    # ------------------------------------------------------------------
    # Vectorised Shock construction
    # ------------------------------------------------------------------

    def _resolve_spot_shock_vector(self, scenario: StressScenario) -> np.ndarray:
        """Projects a scalar or ticker-mapped spot shock into an aligned NumPy vector"""
        if not isinstance(scenario.spot_shock, Mapping):
            return np.full(self.market_data.n_assets, float(scenario.spot_shock))

        shock_map = dict(scenario.spot_shock)
        default_shock = float(shock_map.pop("DEFAULT", 0.0))
        shock_series = pd.Series(shock_map, dtype = float).reindex(
            self.market_data.asset_names, fill_value = default_shock
        )
        return shock_series.to_numpy()

    def apply_scenario(self, scenario: StressScenario) -> MarketData:
        """
        Generates a new, immutable 'MarketData' state reflecting the scenario's factor shocks.

        Boundary Clamping:
        - Spot prices are clamped from below to prevent degenrate or negative asset prices
        - Volatilities are clamped at 0
        """
        spot_shift = self._resolve_spot_shock_vector(scenario)
        stressed_spots = np.maximum(
            self.market_data.spot_prices * (1.0 + spot_shift), _SPOT_PRICE_FLOOR
        )
        stressed_vols = np.maximum(self.market_data.volatilities + scenario.vol_shock, 0.0)
        stressed_rate = self.market_data.risk_free_rate + scenario.rate_shock

        return MarketData(
            valuation_date=self.market_data.valuation_date,
            asset_names=self.market_data.asset_names,
            spot_prices=stressed_spots,
            volatilities=stressed_vols,
            correlation_matrix=self.market_data.correlation_matrix,
            risk_free_rate=stressed_rate,
            dividend_yields=self.market_data.dividend_yields,
        )

    # ------------------------------------------------------------------
    # Single-Scenario Revaluation
    # ------------------------------------------------------------------

    def _revalue_scenario(
        self, scenario: StressScenario, base_value: float, base_delta: pd.Series
    ) -> ScenarioResult:
        """Applies a stress scenario, revalues the portfolio and computes Delta drift"""
        stressed_market = self.apply_scenario(scenario)
        stressed_engine = RiskEngine(
            market_data = stressed_market, portfolio = self.portfolio,
            config = self._proxy_config, bs_engine = self.bs_engine,
        )

        stressed_value = stressed_engine.base_portfolio_value()
        stressed_delta = stressed_engine.greeks_snapshot().delta

        dollar_pn1 = stressed_value - base_value
        pct_drawdown = (dollar_pn1 / base_value) if base_value != 0.0 else 0.0
        delta_drift = stressed_delta.subtract(base_delta, fill_value = 0.0)

        return ScenarioResult(
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            base_value=base_value,
            stressed_value=stressed_value,
            dollar_pn1=dollar_pn1,
            pct_drawdown=pct_drawdown,
            delta_drift=delta_drift,
        )

    # ------------------------------------------------------------------
    # Public Execution Entry Point
    # ------------------------------------------------------------------

    def run(self) -> StressTestReport:
        """
        Executes stress revaluation across all configured scenarios.

        Returns:
        --------
        StressTestReport
            Container of all scenario results with tabular formatting and worst-case analytics
        """
        base_value = self._base_engine.base_portfolio_value()
        base_delta = self._base_engine.greeks_snapshot().delta

        results = [
            self._revalue_scenario(scenario, base_value, base_delta)
            for scenario in self.scenarios
        ]
        return StressTestReport(portfolio_name = self.portfolio.name, results = results)