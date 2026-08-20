"""
src/market_data_loader.py

Live Market Data Ingestion and Statistical Parameter Calibration

This module acts as an empirical bride between external, asynchrous market data
APIs and the strict, immutable 'MarketData' data contracts defined in 'src.data_models'
It allows the 'RiskEngine' to execute live without modifying downstream numerical 
or pricing modules.

Core Pipeline & Calibration Methodology:
----------------------------------------
1. Historical Time-Series Ingestion & Panel Alignment:
    - Fetches historical adjusted-close daily prices for a specified basket of tickers
    - Cleans and aligns the multi-asset price matrix across disjoint trading calendars
    using forward-fill, followed by complete-case truncation.

2. Continuous Return Formulation:
    - Calclates daily continously compounded returns
    - Log returns ensure temporal additivity and align with the GBM continuous diffusion 
    assumptions of the BSM framework

3. Annualised Volatility Scaling:
    - Computes sample standard deviation with Bessel's correction
    - Scales to annualised volatility using the square-root-of-time rule under the i.i.d
    assumption (252 trading days per calendar year)

4. Empirical Correlation & Spectral Matrix Repair:
    - Computes empirical Pearson correlation matrix
    - Asynchrous quotes, holidays and missing data can cause empirical correlation matrices
    to lose positive semi-definiteness
    - Implements spectral decomposition and eigenvalue clipping
    - Rescales the repaired matrix to enforce unit diagonal entries

5. Corporate Actions & Yield Resolution:
    - Aggregates trailing twelve-motnh cash distributions to calibrate continuous dividend 
    yields
    - Resolves risk-free discount rate r from short-dated Treasury yield proxies
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "The yfinance package is required for the data loader"
    ) from exc

from src.data_models import MarketData

__all__ = [
    "MarketDataFetchError",
    "MarketDataLoader"
]

# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------

class MarketDataFetchError(Exception):
    """
    Raised whenever live market data cannot be retrieved or calibrated
    into a usable state -- e.g. invalid/delisted tickers, an unreachable
    network, an empty price panel after cleaning, or a degenerate
    (all-NaN) statistical calibration.
    """

# ----------------------------------------------------------------------------
# Live Ingestion & Calibration Loader
# ----------------------------------------------------------------------------

@dataclass
class MarketDataLoader:
    """
    Calibrates multi-asset statistical parameters from live Yahoo Finance feeds

    Parameters:
    -----------
    tickers : Sequence[str]
        Collection of equity/ETF ticker symbols
    lookback : str
        Historical time window for volatility and correlation estimation
    risk_free_ticker : str | None
        Yahoo Finance ticker for the risk-free rate proxy
    fallback_risk_free_rate : float
        Annualised continous discount rate used if the live proxy is unreachable
    end_date : date | None
        Upper boundary date for historical window. Defaults to current date.
    min_eigenvalue : float
        Lower spectral bound floor > 0 applied during correlation matrix PSD repair
    """

    tickers: Sequence[str]
    lookback: str = "1y"
    risk_free_ticker: str | None = "^IRX"
    fallback_risk_free_rate: float = 0.045
    end_date: date | None = None
    min_eigenvalue: float = 1e-8

    def __post_init__(self) -> None:
        """Standardises ticker symbols"""
        cleaned = tuple(dict.fromkeys(t.strip().upper() for t in self.tickers))
        if not cleaned:
            raise MarketDataFetchError("At least one ticker must be supplied")
        object.__setattr__(self, "tickers", cleaned)

    # ------------------------------------------------------------------
    # Public Orchestration Entry Point
    # ------------------------------------------------------------------

    def build_market_data(self, valuation_date: date | None = None) -> MarketData:
        """
        Executes the end-to-end ingestion and statistical calibration pipeline.

        Returns:
        --------
        MarketData
            A validated, immutable snapshot containing live spot prices, annualised
            volatilities, repaired correlation matrix, risk-free rate and dividend yields.
        """
        prices = self._download_price_panel()
        log_returns = self._compute_log_returns(prices)

        spot_prices = self._extract_spot_prices(prices)
        volatilities = self._compute_annualised_volatility(log_returns)
        correlation_matrix = self._compute_correlation_matrix(log_returns)
        dividend_yields = self._estimate_dividend_yields(spot_prices)
        risk_free_rate = self._resolve_risk_free_rate()

        resolved_valuation_date = valuation_date or prices.index[-1].date()
        
        try:
            return MarketData(
                valuation_date=resolved_valuation_date,
                asset_names=self.tickers,
                spot_prices=spot_prices,
                volatilities=volatilities,
                correlation_matrix=correlation_matrix,
                risk_free_rate=risk_free_rate,
                dividend_yields=dividend_yields,
            )
        except ValueError as exc:
            raise MarketDataFetchError(
                f"Live-calibrated market data failed contract validaton: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Step 1: Download & Clean Historical Price Panel
    # ------------------------------------------------------------------

    def _download_price_panel(self) -> pd.DataFrame:
        """
        Downloads adjusted closing prices from yfinance and aligns trading timestamps

        Handling Multi-Exchange Asynchrony:
        - Different assets may trade on different exchances with distinct holiday schedules.
        - Forward-filling propagates the last know valid price during single-market closures
        - Strips initial lookback gaps to yield a rectangular matrix
        """
        end = self.end_date or date.today()

        try:
            raw = yf.download(
                tickers = list(self.tickers),
                period = self.lookback,
                end = pd.Timestamp(end) + pd.Timedelta(days = 1),
                auto_adjust = True,
                progress = False,
                threads = True,
                group_by = "column",
            )
        except Exception as exc: 
            raise MarketDataFetchError(
                f"Failed to download price data from yfinance for "
                f"{self.tickers}: {exc}"
            )
        
        if raw is None or raw.empty:
            raise MarketDataFetchError(
                f"yfinance returned no data for tickers {self.tickers}."
                "Check that the symbols are valid and the network is reachable"
            )

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                raise MarketDataFetchError(
                    "Unexpected yfinance response schema: no 'Close' field."
                )
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = list(self.tickers)

        missing_tickers = [t for t in self.tickers if t not in close.columns]
        if missing_tickers:
            raise MarketDataFetchError(
                f"No price data returned for ticker(s): {missing_tickers}"
                "They may be invalid or delisted"
            )
        close = close[list(self.tickers)]

        close = close.ffill()
        close = close.dropna(how = "any", axis = 0)

        if close.shape[0] < 30:
            raise MarketDataFetchError(
                f"Insufficient aligned history after cleaning "
                f"({close.shape[0]} rows) to calibrate volatitlies and "
                "correlations reliably. Try a longer lookback period"
            )
                
        return close

    # ------------------------------------------------------------------
    # Step 2: Vectorised Statistical Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Computes continously compounded daily returns
        """
        log_returns = np.log(prices / prices.shift(1)).dropna(how = "any", axis = 0)
        if log_returns.empty:
            raise MarketDataFetchError(
                "Log-return panel is empty after differencing; cannot calibrate"
            )
        return log_returns

    @staticmethod
    def _extract_spot_prices(prices: pd.DataFrame) -> np.ndarray:
        """Extracts the most recent valid adjusted closing price for each underlying asset"""
        return prices.iloc[-1].to_numpy(dtype = float)

    @staticmethod
    def _compute_annualised_volatility(log_returns: pd.DataFrame) -> np.ndarray:
        """Computes sample volatility scaled by the square root of annual trading days"""
        daily_std = log_returns.std(axis = 0, ddof = 1).to_numpy(dtype = float)
        return daily_std * np.sqrt(252.0)

    def _compute_correlation_matrix(self, log_returns: pd.DataFrame) -> np.ndarray:
        """Computes empircal Pearson correlation and guarantees positive semi-definiteness"""
        corr = log_returns.corr().to_numpy(dtype = float)
        return self._repair_correlation_matrix(corr, self.min_eigenvalue)

    @staticmethod
    def _repair_correlation_matrix(corr: np.ndarray, floor: float) -> np.ndarray:
        """
        Enforces Positive Semi-Definiteness on an empirical correlation matrix.

        Mathematical Rationale:
        -----------------------
        Due to asynchronous pricing, missing data interpolation or floating point rounding,
        an empirical correlation matrix may possess negative eigenvalues violating the 
        requirement for Cholesky factorisation

        Spectral Projection Algorithm:
        ------------------------------
        1. Perform eigendecomposition:
        2. Clip negative or degenerate eigenvalues to a strictly positive floor
        3. Reconstruct symmetric matrix:
        4. Normalise diagonals to restore exact unit vairnace 
        """
        eigvals, eigvecs = np.linalg.eigh(corr)

        if np.min(eigvals) >= floor:
            return (corr + corr.T) / 2.0

        clipped = np.maximum(eigvals, floor)
        repaired = eigvecs @ np.diag(clipped) @ eigvecs.T

        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)
        np.fill_diagonal(repaired, 1.0)

        return(repaired + repaired.T) / 2.0

    # ------------------------------------------------------------------
    # Step 3: Continuous Dividend Yield Estimation
    # ------------------------------------------------------------------

    def _estimate_dividend_yields(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Estimates annualised dividend yield q from trailing twelve months of cash distribution
        """
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days = 365))
        yields = np.zeros(len(self.tickers), dtype = float)

        for i, ticker in enumerate(self.tickers):
            spot = spot_prices[i]
            try:
                handle = yf.Ticker(ticker)
                dividends = handle.dividends
                if dividends is not None and not dividends.empty:
                    dividend_index = dividends.index
                    if dividend_index.tz is not None:
                        cutoff_local = cutoff.tz_localize(dividend_index.tz)
                    else:
                        cutoff_local = cutoff
                    trailing = dividends[dividend_index >= cutoff_local]
                    ttm_dividends = float(trailing.sum())
                    yields[i] = ttm_dividends / spot if spot > 0 else 0.0
                else:
                    info_yield = handle.info.get("dividendYield")
                    yields[i] = float(info_yield) if info_yield else 0.0
            except Exception as exc:
                warnings.warn(
                    f"Dividend yield estimation failed for '{ticker}': {exc}."
                    "Defaulting to 0.0.",
                    RuntimeWarning,
                    stacklevel = 2,
                )
                yields[i] = 0.0
        return yields

    # ------------------------------------------------------------------
    # Step 4: Risk-Free Rate Resolution
    # ------------------------------------------------------------------

    def _resolve_risk_free_rate(self) -> float:
        """
        Resolves current annualised risk-free discount rate from 13-week T-Bill index

        Quotes for '^IRX' represent percentage annualised discount rates.
        These are converted to decimal format
        """

        if self.risk_free_ticker is None:
            return self.fallback_risk_free_rate

        try:
            rf_history = yf.download(
                tickers = self.risk_free_ticker,
                period = "5d",
                auto_adjust = True,
                progress = False,
            )
            if rf_history is None or rf_history.empty:
                raise MarketDataFetchError(
                    f"No data returned for risk-free proxy "
                    f" '{self.risk_free_ticker}'."
                )
            close = rf_history["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            latest = float(close.dropna().iloc[-1])

            return latest / 100.0
        except Exception as exc:
            warnings.warn(
                f"Risk-free rate proxy '{self.risk_free_ticker}' fetch failed "
                f"({exc}); falling back to {self.fallback_risk_free_rate:.4%}.",
                RuntimeWarning,
                stacklevel = 2,
            )
            return self.fallback_risk_free_rate