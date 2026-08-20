"""
tests/test_engine.py

Phase 3: pytest suite for the Black-Scholes, Monte Carlo, and Risk
aggregation engines defined in src.engine.

Run with:  pytest -v tests/test_engine.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.data_models import (
    EquityPosition,
    EuropeanOptionPosition,
    MarketData,
    OptionType,
    Portfolio,
    SimulationConfig,
)
from src.engine import (
    BlackScholesEngine,
    MonteCarloSimulator,
    PortfolioMarketMismatchError,
    RiskEngine,
)

VALUATION_DATE = date(2026, 1, 2)

# ----------------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def bs_engine() -> BlackScholesEngine:
    return BlackScholesEngine()

@pytest.fixture
def market_data() -> MarketData:
    betas = np.array([0.85, 0.88, 0.80, 0.97])
    correlation = np.outer(betas, betas)
    np.fill_diagonal(correlation, 1.0)

    return MarketData(
        valuation_date=VALUATION_DATE,
        asset_names=("AAPL", "MSFT", "NVDA", "SPY"),
        spot_prices=np.array([225.0, 430.0, 135.0, 560.0]),
        volatilities=np.array([0.28, 0.25, 0.45, 0.16]),
        correlation_matrix=correlation,
        risk_free_rate=0.045,
        dividend_yields=np.array([0.005, 0.007, 0.0, 0.013]),
    )

@pytest.fixture
def mixed_portfolio() -> Portfolio:
    positions = [
        EquityPosition(asset_name="AAPL", quantity=150),
        EquityPosition(asset_name="SPY", quantity=100),
        EuropeanOptionPosition(
            asset_name="MSFT", quantity=2, option_type=OptionType.CALL,
            strike=440.0, maturity_date=VALUATION_DATE + timedelta(days=45),
            contract_multiplier=100.0,
        ),
        EuropeanOptionPosition(
            asset_name="SPY", quantity=1, option_type=OptionType.PUT,
            strike=540.0, maturity_date=VALUATION_DATE + timedelta(days=30),
            contract_multiplier=100.0,
        ),
    ]
    return Portfolio(name="Test Book", positions=positions)

@pytest.fixture
def sim_config() -> SimulationConfig:
    return SimulationConfig(
        n_paths = 20_000,
        time_horizon= 10 / 365,
        n_steps = 1,
        random_seed = 7,
        antithetic = True,
        confidence_levels = (0.05, 0.01)
    )

#----------------------------------------------------------------------------
# 1. Black-Scholes pricing: put-call parity and analytical edge cases
# ----------------------------------------------------------------------------

class TestBlackScholesEngine:

    def test_put_call_parity(self, bs_engine: BlackScholesEngine) -> None:
        """C - P = S*exp(-qT) - K*exp(-rT) must hold across a vectorised batch."""
        S = np.array([100.0, 250.0, 50.0])
        K = np.array([105.0, 240.0, 55.0])
        T = np.array([0.5, 1.0, 0.25])
        r = 0.03
        q = np.array([0.01, 0.00, 0.02])
        sigma = np.array([0.20, 0.35, 0.15])

        call_price = bs_engine.price(S, K, T, r, q, sigma, is_call=np.array([True, True, True]))
        put_price = bs_engine.price(S, K, T, r, q, sigma, is_call=np.array([False, False, False]))

        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        np.testing.assert_allclose(lhs, rhs, atol=1e-6)

    def test_expired_option_equals_intrinsic_value(self, bs_engine: BlackScholesEngine) -> None:
        """At T = 0 (expiry), price must collapse exactly to intrinsic value."""
        S = np.array([120.0, 80.0])
        K = np.array([100.0, 100.0])
        T = np.array([0.0, 0.0])
        r, q, sigma = 0.03, 0.0, 0.20
        is_call = np.array([True, False])

        prices = bs_engine.price(S, K, T, r, q, sigma, is_call)
        expected = np.array([20.0, 20.0])  # max(120-100,0)=20 ; max(100-80,0)=20

        np.testing.assert_allclose(prices, expected, atol=1e-8)

    def test_deep_itm_call_converges_to_forward_intrinsic(self, bs_engine: BlackScholesEngine) -> None:
        """A far in-the-money call has negligible time value and prices near
        its discounted forward intrinsic value: S*exp(-qT) - K*exp(-rT)."""
        S, K, T, r, q, sigma = 1000.0, 10.0, 1.0, 0.03, 0.0, 0.20
        price = bs_engine.price(
            np.array([S]), np.array([K]), np.array([T]), r,
            np.array([q]), np.array([sigma]), np.array([True]),
        )[0]
        forward_intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)

        assert price == pytest.approx(forward_intrinsic, rel=1e-3)


# ----------------------------------------------------------------------------
# 2. Monte Carlo Simulator: antithetic variates
# ----------------------------------------------------------------------------

class TestMonteCarloSimulator:

    def test_antithetic_output_shape(self, market_data: MarketData) -> None:
        config = SimulationConfig(n_paths=10_000, time_horizon=1 / 365, random_seed=1, antithetic=True)
        simulator = MonteCarloSimulator(market_data=market_data, config=config)
        S_T = simulator.simulate_terminal_prices()

        assert S_T.shape == (config.n_paths, market_data.n_assets)
        assert np.all(np.isfinite(S_T))
        assert np.all(S_T > 0.0)

    def test_antithetic_same_mean_single_draw(self, market_data: MarketData) -> None:
        """A single antithetic draw and a single independent draw of equal size
        should target the same population mean (within Monte Carlo tolerance)."""
        n_paths, seed = 20_000, 123

        antithetic_config = SimulationConfig(n_paths=n_paths, time_horizon=10 / 365, random_seed=seed, antithetic=True)
        independent_config = SimulationConfig(n_paths=n_paths, time_horizon=10 / 365, random_seed=seed, antithetic=False)

        S_anti = MonteCarloSimulator(market_data=market_data, config=antithetic_config).simulate_terminal_prices()
        S_indep = MonteCarloSimulator(market_data=market_data, config=independent_config).simulate_terminal_prices()

        assert S_anti.shape == S_indep.shape
        np.testing.assert_allclose(S_anti.mean(axis=0), S_indep.mean(axis=0), rtol=0.05)

    def test_antithetic_reduces_variance_of_mean_estimator(self, market_data: MarketData) -> None:
        """The variance-reduction property of antithetic variates is a statement
        about the sampling variance of the MEAN estimator (which benefits from
        the negative covariance induced between +Z / -Z pairs) -- not about the
        marginal variance of the pooled individual draws, which is statistically
        indistinguishable between the two schemes. We verify this correctly by
        repeating the experiment across many independent seeds and comparing the
        variance of the resulting sample means, which is the quantity antithetic
        sampling is actually designed to shrink.
        """
        n_paths, n_repetitions = 500, 300
        seeds = np.arange(n_repetitions)

        def repeated_sample_means(antithetic: bool) -> np.ndarray:
            means = np.empty(n_repetitions)
            for i, seed in enumerate(seeds):
                cfg = SimulationConfig(
                    n_paths=n_paths, time_horizon=10 / 365,
                    random_seed=int(seed), antithetic=antithetic,
                )
                S_T = MonteCarloSimulator(market_data=market_data, config=cfg).simulate_terminal_prices()
                means[i] = S_T.mean()
            return means

        antithetic_means = repeated_sample_means(antithetic=True)
        independent_means = repeated_sample_means(antithetic=False)

        assert antithetic_means.var(ddof=1) < independent_means.var(ddof=1)


# ----------------------------------------------------------------------------
# 3. Risk Engine: Expected Shortfall >= VaR, and portfolio/market validation
# ----------------------------------------------------------------------------

class TestRiskEngine:

    def test_expected_shortfall_at_least_var(
        self, market_data: MarketData, mixed_portfolio: Portfolio, sim_config: SimulationConfig
    ) -> None:
        engine = RiskEngine(market_data=market_data, portfolio=mixed_portfolio, config=sim_config)
        results = engine.run()

        for alpha in sim_config.confidence_levels:
            var_alpha = results.var_by_level[alpha]
            es_alpha = results.es_by_level[alpha]
            assert es_alpha >= var_alpha - 1e-8, (
                f"ES ({es_alpha}) must be >= VaR ({var_alpha}) at alpha={alpha}"
            )

    def test_mismatched_asset_raises(self, market_data: MarketData, sim_config: SimulationConfig) -> None:
        bad_portfolio = Portfolio(
            name="Broken Book",
            positions=[EquityPosition(asset_name="TSLA", quantity=10)],  # absent from market_data
        )
        with pytest.raises(PortfolioMarketMismatchError):
            RiskEngine(market_data=market_data, portfolio=bad_portfolio, config=sim_config)

    def test_run_produces_valid_results(
        self, market_data: MarketData, mixed_portfolio: Portfolio, sim_config: SimulationConfig
    ) -> None:
        engine = RiskEngine(market_data=market_data, portfolio=mixed_portfolio, config=sim_config)
        results = engine.run()

        assert results.simulated_pn1.shape == (sim_config.n_paths,)
        assert np.isfinite(results.expected_pn1)
        assert results.pn1_std_error > 0.0
        assert set(results.var_by_level) == set(sim_config.confidence_levels)
        assert results.greeks is not None