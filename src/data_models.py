"""
src/data_models.py

Core Architectural Contracts and Immutable Data Models.

This module establishes the foundational data contracts for the Quantitative
Risk Management Aggregator. By enforcing strict immutability (frozen dataclasses)
and comprehensive domain validation at initialision, it guarantees that:
    1. Downstream numerical engines (Monte Carlo, Black-Scholes, Stress Testing)
    receive clean, mathematically consistent and dimensionally aligned structures.
    2. In-memory state cannot be mutated side-effectually during multi-threaded
    or high iteration simulation passes.
    3. Mathematical invariants (such as matrix positive semi-definiteness and 
    coherent risk measure relationships) are verifed at system boundaries.

Module Hierarchy:
-----------------
- Enumerations: OptionType, PositionType (Type discriminators for vectorisation)
- Marker State: MarketData (Spot prices, volatilities, correlations, yields)
- Instruments: Position (ABC), EquityPosition, EuropeanOptionPosition
- Aggregation: Portfolio (Container with vectorised slice projections)
- Simulation: SimulationConfig (Discretisation & variance reduction parameters)
- Risk Outputs: GreeksSnapshot, RiskMetricResults (VaR, ES and Greeks containers)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------

class OptionType(str, Enum):
    """Exercise payoff discriminator for vanilla European derivatives.
    
    Used in Black-Scholes analytical formulas to branch between:
    - CALL: Payoff max(S_T - K, 0)
    - PUT: Payoff max(K - S_T, 0)
    """
    CALL = "CALL"
    PUT = "PUT"

class PositionType(str, Enum):
    """Discriminator enabling branchless boolean masking and vectorised filtering
    of homogenous sub-portfolios (e.g. isolating linear equities vs non-linear options)."""

    EQUITY = "EQUITY"
    EUROPEAN_OPTION = "EUROPEAN_OPTION"

# --------------------------------------------------------------------
# Market Data Contract
# --------------------------------------------------------------------    

@dataclass(frozen = True)
class MarketData:
    """
    Immutable snapshot of the multi-asset state at a specific valuation date.

    Financial & Mathematical Parameters:
    ------------------------------------
    valuation_date : date
        The reference pricing date
    asset_names : tuple[str, ...]
        Ordered tuple of N unique ticker symbols defining the asset coordinate system.
    spot_prices : np.ndarray
        Vector of current underlying prices
    volatilities : np.ndarray
        Vector of annualised historical or implied standard deviations of returns
    correlation_matrix : np.ndarray
        Symmetric positive semi-definite matrix. Governs the coupled Brownian motion
    risk_free_rate : float
        Continously compounded annual risk-free-rate r used in drift adjustments under the risk-neautral pricing measure.
    dividend_yields : np.ndarray | None
        Vector of annualised continous dividend yields. Defaults to a zero vector.
    """

    valuation_date: date
    asset_names: tuple[str, ...]
    spot_prices: np.ndarray
    volatilities: np.ndarray
    correlation_matrix: np.ndarray
    risk_free_rate: float
    dividend_yields: np.array | None = None

    def __post_init__(self) -> None:
        """
        Enforce immutability defensively and validate dimensional, algebraic and financial invariants.
        """
        n = len(self.asset_names)

        object.__setattr__(self, "asset_names", tuple(self.asset_names))
        object.__setattr__(
            self, "spot_prices", np.asarray(self.spot_prices, dtype = float).copy()
        )
        object.__setattr__(
            self, "volatilities", np.asarray(self.volatilities, dtype = float).copy()
        )
        object.__setattr__(
            self,
            "correlation_matrix",
            np.asarray(self.correlation_matrix, dtype = float).copy()
        )

        if self.dividend_yields is None:
            object.__setattr__(self, "dividend_yields", np.zeros(n, dtype = float))
        else:
            object.__setattr__(
                self,
                "dividend_yields",
                np.asarray(self.dividend_yields, dtype = float).copy()
            )

        # --------------------------------------------------------------------
        # Structural & Dimensional Integrity Checks
        # --------------------------------------------------------------------
        if n == 0:
            raise ValueError("MarketData requires at least one asset.")
        if len(set(self.asset_names)) != n:
            raise ValueError("asset_names must be unique.")
        if self.spot_prices.shape != (n, ):
            raise ValueError(f"spot_prices shape {self.spot_prices.shape} != ({n},)")
        if self.volatilities.shape != (n, ):
                    raise ValueError(f"volatilities shape {self.volatilities.shape} != ({n},)")
        if self.dividend_yields.shape != (n, ):
                    raise ValueError(f"dividend_yields shape {self.dividend_yields.shape} != ({n},)")
        if self.correlation_matrix.shape != (n, n):
                    raise ValueError(f"correlation_matrix shape {self.correlation_matrix.shape} != ({n}, {n})")

        # --------------------------------------------------------------------
        # Financial & Physical Boundary Conditions
        # --------------------------------------------------------------------
        if np.any(self.spot_prices <= 0):
            raise ValueError("spot_prices must be strictly positive")
        if np.any(self.volatilities < 0):
            raise ValueError("volatilities must be non-negative")
        if np.any(self.dividend_yields < 0):
            raise ValueError("dividend_yields must be non-negative.")

        # --------------------------------------------------------------------
        # Linear Algebra: Correlation Matrix Regularity & PSD Invariant
        # --------------------------------------------------------------------
        # Invariant 1: Symmetry
        if not np.allclose(self.correlation_matrix, self.correlation_matrix.T):
            raise ValueError("correlation_matrix must be symmetric")
        
        # Invariant 2: Unit Diagonal 
        if not np.allclose(np.diag(self.correlation_matrix), 1.0):
            raise ValueError("correlation_matrix diagonal must be all ones")

        # Invariant 3: Positive Semi-Definiteness
        try:
            np.linalg.cholesky(
                 self.correlation_matrix + np.eye(n) * 1e-12 # Jitter prevents numerical roundoff failures.
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "correlation matrix is not positive semi-definite - Cholesky decompositon failed"
            ) from exc

    @property
    def n_assets(self) -> int:
        """Number of distinct underlying assets in the market snapshot"""
        return len(self.asset_names)

    def spot_prices_of(self, asset_name: str) -> float:
        """Convencience lookup for a single asset's spot price"""
        idx = self.asset_names.index(asset_name)
        return float(self.spot_prices[idx])

# --------------------------------------------------------------------
# Positions Data Contracts
# --------------------------------------------------------------------  

@dataclass(frozen = True)
class Position(ABC):
    """
    Abstract Base Class defining the immutable contract for any tradeable financial position.

    Attributes:
    -----------
    asset_name : str
        The ticker of the underlying asset to which this position is exposed
    quantity : float
        The signed position size (quantity > 0: Long exposure; quantity < 0: Short exposure)
    """
    asset_name: str
    quantity: float

    def __post_init__(self) -> None:
        if not self.asset_name:
            raise ValueError("asset_name must be a non-empty string")
        if not np.isfinite(self.quantity):
            raise ValueError("quantity must be a finite number")

    @property
    @abstractmethod
    def position_type(self) -> PositionType:
        """Returns the categorical PositionType discriminator"""
        raise NotImplementedError

    @abstractmethod
    def notional_exposure(self, spot_price: float) -> float:
        """Calculates the dollar notional exposure given the current spot price"""
        raise NotImplementedError


@dataclass(frozen = True)
class EquityPosition(Position):
    """
    Linear (Delta - 1) Cash Equity Holding

    Payoff is linear in the underlying asset spot price
    """

    @property
    def position_type(self) -> PositionType:
        return PositionType.EQUITY

    def notional_exposure(self, spot_price: float) -> float:
        """Dollar notional is simply share * price"""
        return self.quantity * spot_price

@dataclass(frozen = True)
class EuropeanOptionPosition(Position):
    """
    Non-linear Vanilla European Option Position (Standard Black-Scholes-Merton contract)

    Attributes:
    -----------
    option_type : OptionType
        CALL or PUT
    strike : float
        Strike price K > 0
    maturity_date : date
        Expiry date of the contract
    contract_multipier : float
        Standard shares per contract
    """
    option_type: OptionType = OptionType.CALL
    strike: float = 0.0
    maturity_date: date = field(default_factory=date.today)
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strike <= 0:
            raise ValueError("strike must be strictly positive")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be strictly positive")

    @property
    def position_type(self) -> PositionType:
        return PositionType.EUROPEAN_OPTION

    def time_to_maturity(self, valuation_date: date) -> float:
        """
        Calculates annualised day-cout fracting using the Actual / 365 convention.
        If the option has expired we set it to 0
        """
        days = (self.maturity_date - valuation_date).days
        return max(days, 0) / 365

    def notional_exposure(self, spot_price: float) -> float:
        """
        Underlying dollar notional controlled by the options contract
        """
        return self.quantity * self.contract_multiplier * spot_price

# --------------------------------------------------------------------
# Portfolio:
# --------------------------------------------------------------------  

@dataclass
class Portfolio:
    """
    Heterogenous collection of financial positions representing a trading book.

    Provides high-performance filtering properties that extract homoegenous sub-arrays
    for vectorised batch evaluation in the numerical pricing engines.
    """
    name: str
    positions: Sequence[Position] = field(default_factory = tuple)

    def __post_init__(self) -> None:
        self.positions = tuple(self.positions)

    @property
    def equity_positions(self) -> tuple[EquityPosition, ...]:
        """Type-filtered tuple of all linear equity holdings"""
        return tuple(p for p in self.positions if isinstance(p, EquityPosition))

    @property
    def option_position(self) -> tuple[EuropeanOptionPosition, ...]:
        """Type-filtered tuple of all non-linear European option holdings"""
        return tuple(
            p for p in self.positions if isinstance(p, EuropeanOptionPosition)
        )

    @property
    def unique_asset_names(self) -> tuple[str, ...]:
        """Unique underlying asset tickers across all portfolio positions"""
        seen: dict[str, None] = {}
        for p in self.positions:
            seen.setdefault(p.asset_name, None)
        return tuple(seen.keys())

    def to_frame(self) -> pd.DataFrame:
        """Serialises the entire portfolio book into a clean DataFrame"""
        records = []
        for p in self.positions:
            base = {
                "asset_name": p.asset_name,
                "quantity": p.quantity,
                "position_type": p.position_type.value,
                "option_type": None,
                "strike": np.nan,
                "maturity_date": None,
                "contract_multiplier": np.nan,
            }
            if isinstance(p, EuropeanOptionPosition):
                base.update(
                    option_type = p.option_type.value,
                    strike = p.strike,
                    maturity_date = p.maturity_date,
                    contract_multiplier = p.contract_multiplier
                )
            records.append(base)
        return pd.DataFrame.from_records(records)

    def validate_against(self, market_date: MarketData) -> None:
        """
        Confirms that every underlying asset referenced in the portfolio exists
        in the provided MarketData snapshot.
        """
        missing = set(self.unique_asset_names) - set(market_date.asset_names)
        if missing:
            raise ValueError(
                f"Portfolio '{self.name}' references assets with no market data:"
                f"{sorted(missing)}"
            )

# --------------------------------------------------------------------
# Simulation Configuration:
# --------------------------------------------------------------------  

@dataclass(frozen = True)
class SimulationConfig:
    """
    Execution parameters governing the multi-asset Monte Carlo simulation engine.

    Variance Reduction Mechanics:
    -----------------------------
    Antithetic Variates:
        When 'antithetic = True', random shocks are generated in symmetric complementary
        pairs; Introducing negative covariance between path pairs which strictly reduces 
        the standard error of the Monte Carlo mean estimator:

    Paramters:
    ----------
    n_paths : int
        Total number of terminal simulated paths M (even if antithetic = True)
    time_horizon : float
        Simulation risk horizon expressed in annualised years.
    n_steps : int
        Number of discrete time incremements along the path (1 for European options)
    random_seed : int | None
        Seed for the NumPy PCG64 pseudo-random number generator to guarantee reproducibility.
    antithetic : bool
        Enables antithetic sampling for variance reduction.
    confidence_levels : tuple[float, ...]
        Tail risk quantiles in (0, 1)
    """

    n_paths: int
    time_horizon: float
    n_steps: int = 1
    random_seed: int | None = None
    antithetic: bool = True
    confidence_levels: tuple[float, ...] = (0.05, 0.01)

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be strictly positive")
        if self.antithetic and self.n_paths %2 != 0:
            raise ValueError(
                "n_paths must be even when antithetic sampling is enabled"
                "(paths are generated in +Z / -Z pairs)"
            )
        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be strictly positive (years)")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be strictly positive")
        if not self.confidence_levels:
            raise ValueError("confidence_levels must contain at least one alpha")
        if any(not (0.0 < a < 1.0) for a in self.confidence_levels):
            raise ValueError("Every confidence level (alpha) must lie in (0,1)")

    @property
    def dt(self) -> float:
        """Discrete time step incremement"""
        return self.time_horizon / self.n_steps

    @property
    def n_independent_draws(self) -> int:
        """Number of independent Gaussian random vectors drawn prior to antithetic pairing"""
        return self.n_paths // 2 if self.antithetic else self.n_paths


# --------------------------------------------------------------------
# Risk Metric Outputs:
# --------------------------------------------------------------------  

@dataclass(frozen = True)
class GreeksSnapshot:
    """
    First and second-order analytical risk sensitivities aggregated across positions.

    By the linearity of differentiation, portfolio Greeks are the sum of individual Greeks:
    - Delta: Directional exposure per underlying.
    - Gamma: Curvature / delta-drift.
    - Vega: Volatility sensitivity
    - Theta: Time decay per calendar day
    - Rho: Interest rate sensitivity
    """

    delta: pd.Series
    gamma: pd.Series
    vega: pd.Series
    theta: pd.Series
    rho: pd.Series

    def total(self) -> pd.Series:
        """Aggregates sensitivities across all underlying assets into a portfolio total"""
        return pd.Series(
            {
                "delta": self.delta.sum(),
                "gamma": self.gamma.sum(),
                "vega": self.vega.sum(),
                "theta": self.theta.sum(),
                "rho": self.rho.sum(),
            },
            name = "portfolio_total"
        )


@dataclass(frozen = True)
class RiskMetricResults:
    """
    Comprehensive output payload generated by the Monte Carlo Risk Engine.

    Financial & Statistical Invariants:
    -----------------------------------
    Loss Distribution:
        Let terminal portfolio P&L be V(T) - V(0)

    Value-at-Risk: 
        The (1-alpha) quantile of portfolio losses

    Expected Shortfall:
        The conditional expectation of losses exceeding the VaR threshold

    Coherence Guarantee:
        Because ES averages exclusively over the extreme tail beyond VaR.
    """

    portfolio_name: str
    valuation_date: date
    simulated_pn1: np.ndarray
    var_by_level: dict[float, float]
    es_by_level: dict[float, float]
    expected_pn1: float
    pn1_std_error: float
    greeks: GreeksSnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "simulated_pn1", np.asarray(self.simulated_pn1, dtype = float)
        )
        if self.simulated_pn1.ndim != 1 or self.simulated_pn1.size == 0:
            raise ValueError("simulated_pn1 must be a non-empty 1D array")
        if set(self.var_by_level.keys()) != set(self.es_by_level.keys()):
            raise ValueError(
                "var_by_level and es_by_level must share the same set of"
                "confidence levels (alpha keys)"
            )

        for alpha, var in self.var_by_level.items():
            es = self.es_by_level[alpha]
            if es < var - 1e-8:
                raise ValueError(
                    f"ES at alpha = {alpha} ({es:.6f}) is less than VaR"
                    f"({var:.6f}); ES must be >= VaR for a coherent risk"
                    f"measure."
                )

            
    def summary_frame(self) -> pd.DataFrame:
        """Returns a formatted tabular summary of tail-risk metrics across confidence levels."""
        levels = sorted(self.var_by_level.keys())
        return pd.DataFrame(
            {
                "alpha": levels,
                "confidence": [1 - a for a in levels],
                "VaR": [self.var_by_level[a] for a in levels],
                "ES": [self.es_by_level[a] for a in levels],
            }
        ).set_index("alpha")