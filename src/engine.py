"""
src/engine.py

Core Computational and Numerical Pricing Engines.

This module implements the three core computational engines 
that power the risk aggregator.

1. BlackScholesEngine:
    - Analytical closed-form pricing for European vanilla options
    under the Merton (1973) continous-dividend extension.
    - Closed-form analytical first and sencond order Greeks
    - Fully vectorised 2D revaluation grid support across arbitrary
    MxN Monte Carlo path geometries without explicit Python loops

2. MonteCarloSimulator:
    - Simulates multi-asset correlated GBM terminal paths
    - Implements Cholesky factorisation to project independent Gaussian
    draws into the empirical asset correlation structure
    - Enforces variance reduction via antithetic variate generation

3. RiskEngine:
    - Orchestrates the full risk workflow by mapping a heterogenous 'Portfolio'
    against 'MarketData' and 'MonteCarloSimulator'
    - Computes empirical tail-risk metrics and standard errors
    - Assembles an asset-level 'GreeksSnapshot' and returned a validated
    'RiskMetricResults' payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_models import (
    EuropeanOptionPosition,
    GreeksSnapshot,
    MarketData,
    OptionType,
    Portfolio,
    PositionType,
    RiskMetricResults,
    SimulationConfig
)

__all__ = [
    "EngineError",
    "PortfolioMarketMismatchError",
    "BlackScholesEngine",
    "MonteCarloSimulator",
    "RiskEngine"
]

# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------

class EngineError(Exception):
    """Base expception for all engine-layer failiures."""

class PortfolioMarketMismatchError(EngineError):
    """Raised when a portfolio references assets absent from the MarketData snapshot"""

#----------------------------------------------------------------------------
# 1. Analytical Option Pricing & Greeks Engine
# ----------------------------------------------------------------------------

@dataclass(frozen = True)
class BlackScholesEngine:
    r"""
    Vectorised Black-Scholes-Merton pricing and Greeks calculation engine.

    Mathematical Framework:
    -----------------------
    Under the Black-Scholes-Merton continuous dividend model, the underlying
    spot price S(t) follows the risk-neutral SDE:
    dS(t) = (r - q)S(t)dt + \sigma S(t)dW(t)

    The pricing formulas for European Calls (C) and Puts (P) with strike K,
    maturity T, risk-free rate r, continous dividend yield q and volatility \sigma are:
        C = S e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2)
        P = K e^{-rT} \Phi(-d_2) - S e^{-qT} \Phi(-d_1)
    where Phi(\cdot)$ is the standard normal cumulative distribution function (CDF), and:
        d_1 = \frac{\ln(S / K) + \left(r - q + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}
        d_2 = d_1 - \sigma \sqrt{T}

    Boundary Conditions:
    --------------------
    When T converges to 0, the engine transitions to intrinsic payoff:
        V_call(S, K) = max(S - K, 0)
        V_put(S, K) = max(K - S, 0)
    """

    _eps: float = field(default = 1e-12, repr = False)

    # ------------------------------------------------------------------
    # Internal Vectorised Helpers
    # ------------------------------------------------------------------

    def _d1_d2(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Computes the standard Black-Scholes d1 and d2 dimensionless quantities.

        Clamps T and \sigma below by _eps to avoid division-by-zero and numerical
        singularity errors near expiry or zero volatility.
        """
        T_safe = np.maximum(T, self._eps)
        sigma_safe = np.maximum(sigma, self._eps)
        sqrt_T = np.sqrt(T_safe)

        d1 = (
            np.log(S / K) + (r - q + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
        d2 = d1 - sigma_safe * sqrt_T
        return d1, d2

    @staticmethod
    def intrinsic_value(S: np.ndarray, K: np.ndarray, is_call: np.ndarray) -> np.ndarray:
        """
        Computes the terminal / intrinsic exercise payoff
        """
        call_payoff = np.maximum(S - K, 0.0)
        put_payoff = np.maximum(K - S, 0.0)
        return np.where(is_call, call_payoff, put_payoff)

    @staticmethod
    def _broadcast_inputs(
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float | np.ndarray,
        q: np.ndarray,
        sigma: np.ndarray,
        is_call: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Ensures all market and contract inputs are uniformly typed NumPy arrays"""
        return (
            np.asarray(S, dtype=float),
            np.asarray(K, dtype=float),
            np.asarray(T, dtype=float),
            np.asarray(r, dtype=float),
            np.asarray(q, dtype=float),
            np.asarray(sigma, dtype=float),
            np.asarray(is_call, dtype=bool),
        )

    # ------------------------------------------------------------------
    # Public pricing API
    # ------------------------------------------------------------------

    def price(
            self,
            S: np.ndarray,
            K: np.ndarray,
            T: np.ndarray,
            r: float | np.ndarray,
            q: np.ndarray,
            sigma: np.ndarray,
            is_call: np.ndarray,
    ) -> np.ndarray:
        """Evaluates analytical European option prices across vectorised arrays
        
        Parameters:
        -----------
        S : np.ndarray
            Underlying spot price array (1D vector or 2D grid)
        K : np.ndarray
            Strike price array
        T : np.ndarray
            Annualised time to maturity
        r : float | np.ndarray
            Annual continous risk-free discount rate
        q : np.ndarray
            Annual continous dividend yield
        sigma : np.ndarray
            Annualised volatility
        is_call : np.ndarray
            Boolean mask ('True for Call, 'False' for Put)

        Returns:
        --------
        np.ndarray
            Array of theoretical option prices 
        """
        S, K, T, r, q, sigma, is_call = self._broadcast_inputs(S, K, T, r, q, sigma, is_call)
        d1, d2 = self._d1_d2(S, K, T, r, q, sigma)

        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)

        call_price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        put_price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        analytical_price = np.where(is_call, call_price, put_price)

        expired = T <= 0.0
        return np.where(expired, self.intrinsic_value(S, K, is_call), analytical_price)

    def revalue_grid(
            self,
            S_grid: np.ndarray,
            K: np.ndarray,
            T: np.ndarray,
            r: float,
            q: np.ndarray,
            sigma: np.ndarray,
            is_call: np.ndarray,
    ) -> np.ndarray:
        """
    Revalues an option vector across a 2D matrix of simulated spot paths.

    Parameters:
    -----------
    S_grid : np.ndarray
        Shape (M, N) where M is the number of Monte Carlo paths.
    K, T, q, sigma, is_call : np.ndarray
        Shape (N,) arrays of contract specifications.

    Returns:
    --------
    np.ndarray
        Shape (M, N) revalued price grid.
        """
        return self.price(S_grid, K, T, r, q, sigma, is_call)

    def greeks(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float | np.ndarray,
        q: np.ndarray,
        sigma: np.ndarray,
        is_call: np.ndarray,
    ) -> dict[str, np.ndarray]:
        r"""
        Calculates analytical first and second order Greeks under continous dividends
        """

        S, K, T, r, q, sigma, is_call = self._broadcast_inputs(S, K, T, r, q, sigma, is_call)
        d1, d2 = self._d1_d2(S, K, T, r, q, sigma)

        T_safe = np.maximum(T, self._eps)
        sigma_safe = np.maximum(sigma, self._eps)
        sqrt_T = np.sqrt(T_safe)

        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        pdf_d1 = norm.pdf(d1)

        delta = np.where(is_call, disc_q * norm.cdf(d1), -disc_q * norm.cdf(-d1))
        gamma = disc_q * pdf_d1 / (S * sigma_safe * sqrt_T)
        vega = S * disc_q * pdf_d1 * sqrt_T

        theta_common = -(S * pdf_d1 * sigma_safe * disc_q) / (2 * sqrt_T)
        theta_call = theta_common - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        theta_put = theta_common + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        theta = np.where(is_call, theta_call, theta_put)

        rho_call = K * T * disc_r * norm.cdf(d2)
        rho_put = -K * T * disc_r * norm.cdf(-d2)
        rho = np.where(is_call, rho_call, rho_put)

        expired = T <= 0.0
        zeros = np.zeros_like(S, dtype = float)

        return {
            "delta": np.where(expired, zeros, delta),
            "gamma": np.where(expired, zeros, gamma),
            "vega": np.where(expired, zeros, vega),
            "theta": np.where(expired, zeros, theta),
            "rho": np.where(expired, zeros, rho),
        }

#----------------------------------------------------------------------------
# 2. Correlated Monte Carlo Simulation Engine
# ----------------------------------------------------------------------------

@dataclass(frozen = True)
class MonteCarloSimulator:
    r"""
    Multi-asset Correlated GMB Simulator

    Stochastic Differential Equation:
    ---------------------------------
    Under the risk-neutral pricing measure, an N-dimensional asset vector
    S(t) = [S_1(t), ... , S_N(t)] satisfies the coupled SDE system:
    dS_i(t) / S_i(t) = (r - q_i)dt + \sigma_i dW_i(t), i = 1, ... N
    where the Brownian motions have instananeous correlation

    Correlated Innovation Generation:
    ---------------------------------
    1. Compute lower-triangular Cholesky factorisation 
    2. Sample independent standard normal matrix
    3. Generate correlated shocks

    Variance Reduction (Antithetic Sampling):
    When 'config.antithetic=True', each random draw is paired with its negative self
    Because terminal payoff is monotonic in the shocks, the negative covariance between
    the Z and -Z substantially reduce the variance of the Monte Carlo mean estimator.
    """

    market_data: MarketData
    config: SimulationConfig

    def _cholesky_lower(self) -> np.ndarray:
        """Computes the lower triangular Cholesky factor"""
        return np.linalg.cholesky(self.market_data.correlation_matrix)

    def _draw_correlated_shocks(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draws standard normal vectors and projects them onto the empirical correlation manifold.

        Returns:
        --------
        np.ndarray
            Matrix of shape (M, N) containing correlated Gaussian innovations
        """
        n_draws = self.config.n_independent_draws
        n_assets = self.market_data.n_assets

        z_independent = rng.standard_normal(size = (n_draws, n_assets))

        if self.config.antithetic:
            z_full = np.vstack([z_independent, -z_independent])
        else:
            z_full = z_independent

        L = self._cholesky_lower()
        correlated_shocks = z_full @ L.T
        return correlated_shocks

    def simulate_terminal_prices(self) -> np.ndarray:
        """
        Executes terminal price simulation across all assets simultaneously

        Returns:
        --------
        np.ndarray
            Simulated terminal price matrix of shape (M, N)
        """
        rng = np.random.default_rng(self.config.random_seed)
        Z = self._draw_correlated_shocks(rng)

        T = self.config.time_horizon
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yields
        sigma = self.market_data.volatilities
        S0 = self.market_data.spot_prices

        drift = (r - q - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T) * Z

        S_T = S0 * np.exp(drift + diffusion)
        return S_T

#----------------------------------------------------------------------------
# 3. Portfolio Risk Aggregation Engine
# ----------------------------------------------------------------------------

@dataclass
class RiskEngine:
    """
    Orchestrates portfolio revaluation, sensitivity analysis and tail-risk aggregation
    """

    market_data: MarketData
    portfolio: Portfolio
    config: SimulationConfig
    bs_engine: BlackScholesEngine = field(default_factory = BlackScholesEngine)

    def __post_init__(self) -> None:
        """Validates that all portfolio holdings exist in the market data snapshot"""
        missing = set(self.portfolio.unique_asset_names) - set(self.market_data.asset_names)
        if missing:
            raise PortfolioMarketMismatchError(
                f"Portfolio '{self.portfolio.name}' references assets with no market data: {sorted(missing)}"
            )

        self.asset_index: Mapping[str, int] = {
            name: i for i, name in enumerate(self.market_data.asset_names)
        }

        self._frame: pd.DataFrame = self.portfolio.to_frame()

    #----------------------------------------------------------------------------
    # Vectorised Position Extraction Helpers
    # ----------------------------------------------------------------------------

    def _equity_frame(self) -> pd.DataFrame:
        """Filters cash equity holdings from the portfolio table"""
        return self._frame.loc[self._frame["position_type"] == PositionType.EQUITY.value]

    def _option_frame(self) -> pd.DataFrame:
        """Filters option holdings from the portfolio table"""
        return self._frame.loc[self._frame["position_type"] == PositionType.EUROPEAN_OPTION.value]

    def _option_time_to_maturity(self, option_frame: pd.DataFrame) -> np.ndarray:
        """Extracts remaining time-to-maturity for each option holding"""
        maturities = pd.to_datetime(option_frame["maturity_date"])
        valuation = pd.Timestamp(self.market_data.valuation_date)
        days = (maturities - valuation).dt.days.to_numpy()
        return np.maximum(days, 0) / 365.0

    def _option_static_params(
        self, option_frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """Extracts aligned parameter arrays for vectorised option pricing"""
        asset_idx = option_frame["asset_name"].map(self.asset_index).to_numpy()
        S0 = self.market_data.spot_prices[asset_idx]
        sigma = self.market_data.volatilities[asset_idx]
        q = self.market_data.dividend_yields[asset_idx]
        K = option_frame["strike"].to_numpy(dtype = float)
        is_call = (option_frame["option_type"] == OptionType.CALL.value).to_numpy()
        return asset_idx, S0, sigma, q, K, is_call

    #----------------------------------------------------------------------------
    # Portfolio Valuation
    # ----------------------------------------------------------------------------
    
    def base_portfolio_value(self) -> float:
        "Computes current mark-to-market net asset value of the total portfolio"
        value = 0.0

        eq = self._equity_frame()
        if not eq.empty:
            idx = eq["asset_name"].map(self.asset_index).to_numpy()
            spots = self.market_data.spot_prices[idx]
            value += float(np.sum(eq["quantity"].to_numpy(dtype = float) * spots))

        opt = self._option_frame()
        if not opt.empty:
            _, S0, sigma, q, K, is_call = self._option_static_params(opt)
            T = self._option_time_to_maturity(opt)
            prices = self.bs_engine.price(S0, K, T, self.market_data.risk_free_rate, q, sigma, is_call)
            weights = (opt["quantity"] * opt["contract_multiplier"]).to_numpy(dtype = float)
            value += float(np.sum(prices * weights))

        return value

    def revalue_portfolio(self, S_grid: np.ndarray) -> np.ndarray:
        """
        Revalues the full portfolio across all simulated terminal price paths

        Parameters:
        -----------
        S_grid : np.ndarray
            Simulated price matrix of shape (M, N) where M is paths, N is assets.

        Returns:
        --------
        np.ndarray
            Vector of terminal portfolio values V(T) of shape (M, N)
        """
        n_paths = S_grid.shape[0]
        total_value = np.zeros(n_paths, dtype=float)

        eq = self._equity_frame()
        if not eq.empty:
            idx = eq["asset_name"].map(self.asset_index).to_numpy()
            eq_paths = S_grid[:, idx]
            eq_quantities = eq["quantity"].to_numpy(dtype = float)
            total_value += eq_paths @ eq_quantities

        opt = self._option_frame()
        if not opt.empty:
            idx, _, sigma, q, K, is_call = self._option_static_params(opt)
            T0 = self._option_time_to_maturity(opt)
            T_horizon = np.maximum(T0 - self.config.time_horizon, 0.0)

            opt_paths = S_grid[:, idx]
            price_grid = self.bs_engine.revalue_grid(
                opt_paths, K, T_horizon, self.market_data.risk_free_rate, q, sigma, is_call
            )

            weights = (opt["quantity"] * opt["contract_multiplier"]).to_numpy(dtype = float)
            total_value += price_grid @ weights

        return total_value

    #----------------------------------------------------------------------------
    # Greeks Aggregation
    # ----------------------------------------------------------------------------

    def greeks_snapshot(self) -> GreeksSnapshot:
        """
        Aggregates first and second order risk sensitivities by underlying asset.

        Combines linear equity exposures with non linear option derivatives scaled
        by contract multipliers
        """
        assets = pd.Index(self.market_data.asset_names, name = "asset_name")
        greeks_frame = pd.DataFrame(
            0.0, index=assets, columns = ["delta", "gamma", "vega", "theta", "rho"]
        )

        eq = self._equity_frame()
        if not eq.empty:
            eq_delta = eq.groupby("asset_name")["quantity"].sum()
            greeks_frame.loc[eq_delta.index, "delta"] += eq_delta

        opt = self._option_frame()
        if not opt.empty:
            _, S0, sigma, q, K, is_call = self._option_static_params(opt)
            T = self._option_time_to_maturity(opt)
            weights = (opt["quantity"] * opt["contract_multiplier"]).to_numpy(dtype = float)

            raw_greeks = self.bs_engine.greeks(
                S0, K, T, self.market_data.risk_free_rate, q, sigma, is_call
            )

            for greek_name, values in raw_greeks.items():
                contribution = pd.Series(
                    values * weights, index = opt["asset_name"].to_numpy()
                ).groupby(level = 0).sum()
                greeks_frame.loc[contribution.index, greek_name] += contribution

        return GreeksSnapshot(
            delta = greeks_frame["delta"],
            gamma = greeks_frame["gamma"],
            vega = greeks_frame["vega"],
            theta = greeks_frame["theta"],
            rho = greeks_frame["rho"],
        )

    #----------------------------------------------------------------------------
    # Main Execution Pipeline
    # ----------------------------------------------------------------------------

    def run(self) -> RiskMetricResults:
        r"""
        Executes full Monte Carlo risk simulation and computes tail metrics

        Statistical Estimators:
        -----------------------
        1. Expected P&L
        2. Monte Carlo Standard Error of the Mean
        3. Tail Losses
            • Value at Risk
            • Expected Shortfall

        Returns:
        --------
        RiskMetricResults
            Validated container with simulation paths, VaR/ES tables and Greeks
        """


        V0 = self.base_portfolio_value()

        simulator = MonteCarloSimulator(market_data = self.market_data, config = self.config)
        S_T = simulator.simulate_terminal_prices()
        V_T = self.revalue_portfolio(S_T)

        pn1 = V_T - V0
        loss = -pn1

        var_by_level: dict[float, float] = {}
        es_by_level: dict[float, float] = {}
        for alpha in self.config.confidence_levels:
            var_alpha = float(np.quantile(loss, 1.0 - alpha))
            tail_losses = loss[loss >= var_alpha]
            es_alpha = float(tail_losses.mean()) if tail_losses.size > 0 else var_alpha
            var_by_level[alpha] = var_alpha
            es_by_level[alpha] = es_alpha

        expected_pn1 = float(np.mean(pn1))
        pn1_std_error = float(np.std(pn1, ddof = 1) / np.sqrt(pn1.size))

        greeks = self.greeks_snapshot()

        return RiskMetricResults(
            portfolio_name=self.portfolio.name,
            valuation_date=self.market_data.valuation_date,
            simulated_pn1=pn1,
            var_by_level=var_by_level,
            es_by_level=es_by_level,
            expected_pn1=expected_pn1,
            pn1_std_error=pn1_std_error,
            greeks=greeks,
        )