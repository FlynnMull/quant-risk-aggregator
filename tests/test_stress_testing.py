"""
tests/test_stress_testing.py

Phase 4: pytest suite for the deterministic macro stress-testing engine
defined in src.stress_testing.

Run with:  pytest -v tests/test_stress_testing.py
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
)
from src.engine import BlackScholesEngine
from src.stress_testing import (
    ALL_SCENARIOS,
    SEVERE_MARKET_CRASH,
    StressScenario,
    StressTestingEngine,
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
def long_equity_portfolio() -> Portfolio:
    """A purely linear, long-only equity book -- no options."""
    positions = [
        EquityPosition(asset_name="AAPL", quantity=150),
        EquityPosition(asset_name="MSFT", quantity=100),
        EquityPosition(asset_name="NVDA", quantity=80),
        EquityPosition(asset_name="SPY", quantity=100),
    ]
    return Portfolio(name="Long Equity Book", positions=positions)


@pytest.fixture
def spy_equity_only_portfolio() -> Portfolio:
    """Long SPY, unhedged."""
    return Portfolio(
        name="SPY Unhedged",
        positions=[EquityPosition(asset_name="SPY", quantity=100)],
    )


@pytest.fixture
def spy_hedged_portfolio() -> Portfolio:
    """The same long SPY position, plus a deep out-of-the-money protective put."""
    positions = [
        EquityPosition(asset_name="SPY", quantity=100),
        EuropeanOptionPosition(
            asset_name="SPY", quantity=1, option_type=OptionType.PUT,
            strike=420.0,  # deep OTM: base spot is 560.0
            maturity_date=VALUATION_DATE + timedelta(days=90),
            contract_multiplier=100.0,
        ),
    ]
    return Portfolio(name="SPY Hedged", positions=positions)


# ----------------------------------------------------------------------------
# 1. Severe spot shocks must hurt long equity portfolios
# ----------------------------------------------------------------------------


class TestSpotShockDirectionality:

    def test_severe_crash_yields_negative_pnl_for_long_equity(
        self, market_data: MarketData, long_equity_portfolio: Portfolio
    ) -> None:
        """A long-only equity book must lose money under a severe, broadly
        negative spot shock scenario."""
        engine = StressTestingEngine(
            market_data=market_data, portfolio=long_equity_portfolio,
            scenarios=[SEVERE_MARKET_CRASH],
        )
        report = engine.run()
        result = report.results[0]

        assert result.dollar_pn1 < 0.0
        assert result.pct_drawdown < 0.0
        # Stressed value must be strictly below the base value.
        assert result.stressed_value < result.base_value

    def test_positive_spot_shock_yields_positive_pnl_for_long_equity(
        self, market_data: MarketData, long_equity_portfolio: Portfolio
    ) -> None:
        """Sanity check on shock directionality: a positive spot shock must
        produce a gain for a long-only book (guards against a sign-flip bug)."""
        rally_scenario = StressScenario(
            name="Broad Rally", description="Uniform +10% spot rally.",
            spot_shock=0.10, vol_shock=-0.02, rate_shock=0.0,
        )
        engine = StressTestingEngine(
            market_data=market_data, portfolio=long_equity_portfolio,
            scenarios=[rally_scenario],
        )
        result = engine.run().results[0]

        assert result.dollar_pn1 > 0.0
        assert result.pct_drawdown > 0.0


# ----------------------------------------------------------------------------
# 2. Deep OTM protective puts must gain value and cushion drawdowns
# ----------------------------------------------------------------------------


class TestProtectivePutHedging:

    def test_deep_otm_put_gains_value_under_severe_crash(
        self, market_data: MarketData, bs_engine: BlackScholesEngine
    ) -> None:
        """A deep out-of-the-money put's analytical price must rise once spot
        falls and implied volatility spikes (both raise put value)."""
        put_position = EuropeanOptionPosition(
            asset_name="SPY", quantity=1, option_type=OptionType.PUT,
            strike=420.0, maturity_date=VALUATION_DATE + timedelta(days=90),
            contract_multiplier=100.0,
        )
        portfolio = Portfolio(name="Put Only", positions=[put_position])
        engine = StressTestingEngine(
            market_data=market_data, portfolio=portfolio, scenarios=[SEVERE_MARKET_CRASH],
        )

        base_price = engine._base_engine.base_portfolio_value()
        stressed_market = engine.apply_scenario(SEVERE_MARKET_CRASH)

        T = put_position.time_to_maturity(VALUATION_DATE)
        spy_idx = market_data.asset_names.index("SPY")

        base_put_price = bs_engine.price(
            S=market_data.spot_prices[spy_idx], K=put_position.strike, T=T,
            r=market_data.risk_free_rate, q=market_data.dividend_yields[spy_idx],
            sigma=market_data.volatilities[spy_idx], is_call=False,
        )
        stressed_put_price = bs_engine.price(
            S=stressed_market.spot_prices[spy_idx], K=put_position.strike, T=T,
            r=stressed_market.risk_free_rate, q=stressed_market.dividend_yields[spy_idx],
            sigma=stressed_market.volatilities[spy_idx], is_call=False,
        )

        assert stressed_put_price > base_put_price
        assert base_price == pytest.approx(float(base_put_price) * 100.0, rel=1e-6)

    def test_protective_put_cushions_drawdown_vs_unhedged(
        self,
        market_data: MarketData,
        spy_equity_only_portfolio: Portfolio,
        spy_hedged_portfolio: Portfolio,
    ) -> None:
        """Under a severe crash, the hedged book's % drawdown must be strictly
        less severe (less negative) than the unhedged book's."""
        unhedged_engine = StressTestingEngine(
            market_data=market_data, portfolio=spy_equity_only_portfolio,
            scenarios=[SEVERE_MARKET_CRASH],
        )
        hedged_engine = StressTestingEngine(
            market_data=market_data, portfolio=spy_hedged_portfolio,
            scenarios=[SEVERE_MARKET_CRASH],
        )

        unhedged_result = unhedged_engine.run().results[0]
        hedged_result = hedged_engine.run().results[0]

        # The hedge should soften both the dollar loss and the % drawdown.
        assert hedged_result.pct_drawdown > unhedged_result.pct_drawdown
        assert hedged_result.dollar_pn1 > unhedged_result.dollar_pn1

        # And the hedged book's aggregate Delta must have moved *less*
        # negative than the unhedged book's (the put's negative Delta grows
        # in magnitude as it moves further into the money, partially
        # offsetting the equity leg's unchanged Delta).
        assert hedged_result.total_delta_drift < 0.0


# ----------------------------------------------------------------------------
# 3. Stressed MarketData contracts remain strictly valid and positive
# ----------------------------------------------------------------------------


class TestStressedMarketDataValidity:

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
    def test_stressed_market_data_is_valid_and_positive(
        self, market_data: MarketData, long_equity_portfolio: Portfolio, scenario: StressScenario
    ) -> None:
        """Every built-in scenario must produce a MarketData snapshot that
        satisfies MarketData's own strict validation: positive spots,
        non-negative vols, finite rate, and a well-formed correlation matrix."""
        engine = StressTestingEngine(
            market_data=market_data, portfolio=long_equity_portfolio, scenarios=[scenario],
        )
        stressed_market = engine.apply_scenario(scenario)

        # MarketData.__post_init__ already raised if any of these failed --
        # these assertions simply make the guaranteed invariants explicit.
        assert np.all(stressed_market.spot_prices > 0.0)
        assert np.all(np.isfinite(stressed_market.spot_prices))
        assert np.all(stressed_market.volatilities >= 0.0)
        assert np.all(np.isfinite(stressed_market.volatilities))
        assert np.isfinite(stressed_market.risk_free_rate)
        assert stressed_market.asset_names == market_data.asset_names
        np.testing.assert_array_equal(
            stressed_market.correlation_matrix, market_data.correlation_matrix
        )

    def test_extreme_negative_spot_shock_still_floors_above_zero(
        self, market_data: MarketData, long_equity_portfolio: Portfolio
    ) -> None:
        """An (unrealistic) -100% spot shock must still floor at a strictly
        positive price rather than raising or producing a non-positive spot."""
        annihilation_scenario = StressScenario(
            name="Total Wipeout", description="Extreme -100% spot shock (edge case).",
            spot_shock=-1.0, vol_shock=0.0, rate_shock=0.0,
        )
        engine = StressTestingEngine(
            market_data=market_data, portfolio=long_equity_portfolio,
            scenarios=[annihilation_scenario],
        )
        stressed_market = engine.apply_scenario(annihilation_scenario)

        assert np.all(stressed_market.spot_prices > 0.0)

    def test_report_to_frame_has_expected_shape_and_columns(
        self, market_data: MarketData, long_equity_portfolio: Portfolio
    ) -> None:
        engine = StressTestingEngine(
            market_data=market_data, portfolio=long_equity_portfolio, scenarios=ALL_SCENARIOS,
        )
        frame = engine.run().to_frame()

        assert len(frame) == len(ALL_SCENARIOS)
        expected_columns = {
            "Description", "Base Value", "Stressed Value",
            "Dollar PnL", "% Drawdown", "Total Delta Drift",
        }
        assert expected_columns.issubset(set(frame.columns))
        assert frame.index.name == "Scenario"