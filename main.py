"""
main.py

End to End Orchestration Pipeline & Command-Line Demonstration

This script serves as the primary entry point and production demonstration of the 
Risk Management Aggregator. It links all architectural layers-live market
ingestion, portfolio structuring, Monte Carlo simulation, analytical Greeks computation,
determinsitc stress testing and visual reporting into an automated pipeline.

Execution Lifecycle:
--------------------
1. Live Market Calibration:
    Ingests 1-year historical daily prices and live risk-free rates for the active basket
    via 'MarketDataLoader', calibrating annualied volatilities, continuous dividend yields 
    and an empirical positive semi-definite correlation matrix.

2. Portfolio Construction:
    Instantiates the "Diversified Growth Book" - a realistic multi-asset trading book combining:
    - Core linear equities: Long exposure across large-cap tech and market beta
    - Bullish Call Overlays: Asymmetric upside participation on MSFT and NVDA
    - Protective Put Overlays: Tail-risk insurance on SPY mitigating systematic macro drawdowns.

3. Correlated Stochastic Risk Simulation:
    Executes a 10-day, 100,000-path correlated GBM simulation with antithetic, variance reduction
    calculating empirical VaR, ES and standard error bounds alongside exact Black-Scholes Greeks.

4. Macro Stress-Testing & Historical Scenario Replay:
    Evaluates determinstic full-revaluation portfolio drawdowns across parametric shocks and
    historical crisis replays 

5. Publication Quality Reporting:
    Outputs formatted terminal summaries and exports two visual dashboards
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from src.data_models import(
    EquityPosition,
    EuropeanOptionPosition,
    MarketData,
    OptionType,
    Portfolio,
    SimulationConfig
)

from src.market_data_loader import MarketDataLoader, MarketDataFetchError

from src.engine import EngineError, RiskEngine
from src.stress_testing import ALL_SCENARIOS, StressTestingEngine
from src.visualiser import plot_dashboard, plot_stress_test_results

# --------------------------------------------------------------------
# 1. Live Market Data Ingestion
# --------------------------------------------------------------------

def build_market_data_live(valuation_date: date) -> MarketData:
    """
    Calibrates an empirical MarketData state from live Yahoo Finance price feeds.

    Parameters:
    -----------
    valuation_date : date
        The effective reference pricing date
    
    Returns:
    --------
    MarketData 
        An immutable market snapshot with positive semi-definite correlation structure.
    """
    loader = MarketDataLoader(
        tickers = ("AAPL", "MSFT", "NVDA", "SPY"),
        lookback = "1y",
        risk_free_ticker = "^IRX",
    )
    return loader.build_market_data(valuation_date = valuation_date)

# ----------------------------------------------------------------------------
# 2. Portfolio Structuring
# ----------------------------------------------------------------------------

def build_portfolio(valuation_date: date) -> Portfolio:
    """
    Constructs the 'Diversified Growth Book" holding linear and non-linear assets/
    """
    positions = [
        # --- Core long equity book ---
        EquityPosition(asset_name="AAPL", quantity=150),
        EquityPosition(asset_name="MSFT", quantity=100),
        EquityPosition(asset_name="NVDA", quantity=80),
        EquityPosition(asset_name="SPY", quantity=100),

        # --- Bullish call overlays (upside participation) ---
        EuropeanOptionPosition(
            asset_name="MSFT", quantity=2, option_type=OptionType.CALL,
            strike=440.0, maturity_date=valuation_date + timedelta(days=45),
            contract_multiplier=100.0,
        ),
        EuropeanOptionPosition(
            asset_name="NVDA", quantity=1, option_type=OptionType.CALL,
            strike=150.0, maturity_date=valuation_date + timedelta(days=60),
            contract_multiplier=100.0,
        ),

        # --- Protective put hedge on the core SPY holding ---
        EuropeanOptionPosition(
            asset_name="SPY", quantity=1, option_type=OptionType.PUT,
            strike=540.0, maturity_date=valuation_date + timedelta(days=30),
            contract_multiplier=100.0,
        ),
    ]
    return Portfolio(name="Diversified Growth Book", positions=positions)

# ----------------------------------------------------------------------------
# 3. Simulation Configuration
# ----------------------------------------------------------------------------

def build_simulation_config() -> SimulationConfig:
    """
    Instantiates execution parameters for the Monte Carlo Risk Engine.

    Risk Parameters:
    ----------------
    - Path Count (M): 100,000 terminal trajectories
    - Time Horizon: 10 trading days expressed in annualised continuous time
    - Variance Reduction: Antithetic sampling enabled
    - Quantile Levels: alpha = 0.05 and alpha = 0.01 
    """
    return SimulationConfig(
        n_paths=100_000,
        time_horizon=10 / 365,  # 10-day risk horizon, expressed in years
        n_steps=1,               # single terminal-price jump over the horizon
        random_seed=42,
        antithetic=True,
        confidence_levels=(0.05, 0.01),
    )


# ----------------------------------------------------------------------------
# 4. Command-Line Orchestration Entry Point
# ----------------------------------------------------------------------------
def main() -> int:
    """Executes the full quantitative risk pipeline and renders reports"""
    valuation_date = date.today()

    try:
        market_data = build_market_data_live(valuation_date)   # was: build_market_data(valuation_date)
        portfolio = build_portfolio(valuation_date)
        config = build_simulation_config()
        engine = RiskEngine(market_data=market_data, portfolio=portfolio, config=config)

        base_value = engine.base_portfolio_value()
        results = engine.run()

        stress_engine = StressTestingEngine(
            market_data = market_data, portfolio = portfolio, scenarios = ALL_SCENARIOS
        )
        stress_report = stress_engine.run()

    except EngineError as exc:
        print(f"[Risk Engine Error] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[Validation Error] {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------------
    # Terminal Output: Monte Carlo Risk Report
    # ------------------------------------------------------------------------
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    print("=" * 72)
    print(f"  QUANTITATIVE RISK REPORT  |  {results.portfolio_name}")
    print(f"  Valuation Date: {results.valuation_date}")
    print("=" * 72)
    print(f"Base Portfolio Value : {base_value:,.2f}")
    print(f"Mean Simulated P&L   : {results.expected_pn1:,.2f}")
    print(f"P&L Std. Error       : {results.pn1_std_error:,.4f}")
    print(f"Simulated Paths      : {results.simulated_pn1.size:,}")
    print("-" * 72)
    print("Value-at-Risk / Expected Shortfall by Confidence Level:")
    print(results.summary_frame().to_string())
    print("-" * 72)
    print("Portfolio Greeks (aggregated by asset):")
    if results.greeks is not None:
        greeks_frame = pd.DataFrame(
            {
                "Delta": results.greeks.delta,
                "Gamma": results.greeks.gamma,
                "Vega": results.greeks.vega,
                "Theta": results.greeks.theta,
                "Rho": results.greeks.rho,
            }
        )
        print(greeks_frame.to_string())
        print("-" * 72)
        print("Portfolio Totals:")
        print(results.greeks.total().to_string())
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Visual Dashboard Export
    # ------------------------------------------------------------------------
    plot_dashboard(results, base_value=base_value, save_path="risk_dashboard.png")
    print("\nDashboard saved to: risk_dashboard.png")

    # ------------------------------------------------------------------------
    # Terminal Output: Macro Stress Testing Report
    # ------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  MACRO STRESS TEST REPORT  |  {stress_report.portfolio_name}")
    print("=" * 72)
    print(stress_report.to_frame().to_string())
    worst = stress_report.worst_scenario()
    if worst is not None:
        print("-" * 72)
        print(
            f"Worst Scenario: {worst.scenario_name}  "
            f"(Dollar PnL: {worst.dollar_pn1:,.2f}, Drawdown: {worst.pct_drawdown:.2%})"
        )
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Stress Testing Dashboard Export
    # ------------------------------------------------------------------------
    plot_stress_test_results(stress_report, save_path="stress_test_dashboard.png")
    print("\nStress test dashboard saved to: stress_test_dashboard.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())