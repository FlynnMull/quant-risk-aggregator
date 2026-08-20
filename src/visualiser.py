"""
src/visualizer.py 

Visual Reporting & Publication-Quality Analytics Dashboards.

This module provides the visualisation layer for the Quantitaitve Risk Management 
Aggregator, rendering two complementary, high-resolution diagnostic reports:

1. Monte Carlo Risk Dashboard:
- A 2x2 multi-panel layout capturing the stochastic simulation output
    • Panel 1: Empiral P&L Distribution with VaR and ES tail markers
    • Panel 2: Asset-Level Normalised Greek Profiles 
    • Panel 3: Law of Large Numbers Convergence with asymptotic error bands
    • Panel 4: Executive Risk Summary & Valuation Card.

2. Macro Stress Testing Dashboard:
- A dual panel horizontal waterfall chart illustrating determinstic portfolio drawdowns
and dollar P&L parameter factor shocks and historical crisis replays.

Design & Performance Standards:
-------------------------------
- Object-Oriented Matplotlib API exclusively
- Fully vectorised numerical transofrmations
- Clean separation of visual styling, color palettes and data contracts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.data_models import RiskMetricResults
from src.stress_testing import StressTestReport

__all__ = ["plot_dashboard", "plot_stress_test_results"]

# --------------------------------------------------------------------
# Shared colour palette (kept consistent across all four panels)
# --------------------------------------------------------------------

_COLOR_PNL = "#3B6E8F"
_COLOR_VAR = "#D98E04"
_COLOR_ES = "#B23A48"
_COLOR_GREEK_POS = "#3B6E8F"
_COLOR_GREEK_NEG = "#B23A48"
_COLOR_CONVERGENCE = "#2F5233"
_COLOR_BAND = "#2F5233"
_COLOR_GAIN = "#2F5233"
_COLOR_LOSS = "#B23A48"

# --------------------------------------------------------------------
# Panel 1: Simulated P&L distribution with VaR / ES annotations:
# --------------------------------------------------------------------

def _plot_pnl_distribution(ax: plt.Axes, results: RiskMetricResults) -> None:
    """
    Renders empirical P&L histogram with annotated tail-risk threshold.

    Quantitative Theory:
    --------------------
    - The distribution reflects the terminal portfolio returns across 
    M correlated Monte Carlo paths
    - Tail Loss: Portfolio loss the negation of the portfolio return
    - Value at Risk: The (1 - alpha) quantile of loss.
    - Expected Shortfall: Conditional expectation of tail losses exceeding VaR.
    """
    pn1 = results.simulated_pn1

    ax.hist(
        pn1, bins = 100, color = _COLOR_PNL, alpha = 0.75,
        edgecolor = "white", linewidth = 0.3
    )

    alphas = sorted(results.var_by_level.keys())
    line_styles = ["--", ":", "-."]
    for i, alpha in enumerate(alphas):
        var_x = -results.var_by_level[alpha]
        es_x = -results.es_by_level[alpha]
        style = line_styles[i % len(line_styles)]
        confidence_pct = int(round((1-alpha) * 100))

        ax.axvline(
            var_x, color = _COLOR_VAR, linestyle = style, linewidth = 1.7,
            label = f"VaR {confidence_pct}% = {var_x:,.0f}",
        )
        ax.axvline(
            es_x, color = _COLOR_ES, linestyle = style, linewidth = 1.7,
            label = f"ES {confidence_pct}% = {es_x:,.0f}",
        )

    ax.axvline(0.0, color = "black", linewidth = 0.8, alpha = 0.6)
    ax.set_title("Simulated P&L Distribution", fontsize = 11, fontweight = "bold")
    ax.set_xlabel("Portfolio P&L")
    ax.set_ylabel("Path Frequency")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.legend(
        fontsize=7, loc = "upper left", bbox_to_anchor = (1.0, 1.0),
        frameon = False, borderaxespad = 0.0,
    )
    ax.grid(alpha = 0.25, linewidth = 0.5)

# --------------------------------------------------------------------
# Panel 2: Asset-Level Normalised Greek Profile
# --------------------------------------------------------------------

def _plot_greeks(ax: plt.Axes, results: RiskMetricResults) -> None:
    """
    Rends grouped bar chart of analytical portfolio Greeks normalised across assets.

    Mathematical Normalisation:
    ---------------------------
    Different Greeks possess incompatible dimensional physical units:
    - Delta
    - Gamma
    - Vega
    - Theta
    - Rho

    To facilitate cross-asset visual comparison on a single axis, each Greek column
    is normalised by its maximum absolutw exposure across the portfolio
    """
    if results.greeks is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No Greeks available", ha = "center", va = "center")
        return

    greeks_df = pd.DataFrame(
        {
            "Delta": results.greeks.delta,
            "Gamma": results.greeks.gamma,
            "Vega": results.greeks.vega,
            "Theta": results.greeks.theta,
            "Rho": results.greeks.rho,
        }
    )

    max_abs = greeks_df.abs().max().replace(0.0, 1.0)
    normalised = greeks_df.div(max_abs, axis = 1)

    assets = normalised.index.to_numpy()
    n_assets = len(assets)
    n_greeks = normalised.shape[1]
    x_positions = np.arange(n_assets)
    bar_width = 0.8 / n_greeks

    for i, greek_name in enumerate(normalised.columns):
        values = normalised[greek_name].to_numpy()
        offsets = x_positions + (i - (n_greeks -1) / 2 ) * bar_width
        colors = np.where(values >= 0, _COLOR_GREEK_POS, _COLOR_GREEK_NEG)
        ax.bar(
            offsets, values, width = bar_width, color = colors,
            alpha = 0.5 + 0.1 * i, label = greek_name,
        )

    ax.axhline(0.0, color = "black", linewidth = 0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(assets, fontsize = 8)
    ax.set_ylabel("Normalised Exposure (per Greek, max |value| = 1)")
    ax.set_title("Portfolio Greeks by Underlying Asset", fontsize = 11, fontweight = "bold")
    ax.legend(
        fontsize = 7, ncol = 5, loc = "upper center",
        bbox_to_anchor = (0.5, -0.18), frameon = False,
    )
    ax.grid(alpha = 0.25, linewidth = 0.5, axis = "y")

# --------------------------------------------------------------------
# Panel 3: Monte Carlo Law of Large Numbers Convergence:
# --------------------------------------------------------------------

def _plot_convergence(ax: plt.Axes, results: RiskMetricResults) -> None:
    """
    Plots the running sample mean and asymptotic standard error confidence band

    Statistical Mechanics:
    ----------------------
    - Running Mean 
    - Running Variance via one-pass sum-of-squares accumulator
    - Running Standard Error of the Mean:
    - Confidence Envelope
    """
    pn1 = results.simulated_pn1
    n_paths = pn1.size
    draw_index = np.arange(1, n_paths + 1)

    cum_sum = np.cumsum(pn1)
    cum_sum_eq = np.cumsum(pn1 ** 2)
    running_mean = cum_sum / draw_index

    with np.errstate(invalid = "ignore", divide = "ignore"):
        running_var = (cum_sum_eq - (cum_sum ** 2) / draw_index) / np.maximum(draw_index - 1, 1)
    running_var = np.clip(running_var, a_min = 0.0, a_max = None)
    running_std_error = np.sqrt(running_var / draw_index)

    stride = max(n_paths // 2000, 1)
    plot_idx = np.arange(0, n_paths, stride)

    ax.plot(
        draw_index[plot_idx], running_mean[plot_idx],
        color = _COLOR_CONVERGENCE, linewidth = 1.3, label = "Cumulative Mean P&L",
    )
    ax.fill_between(
        draw_index[plot_idx],
        running_mean[plot_idx] - 2 * running_std_error[plot_idx],
        running_mean[plot_idx] + 2* running_std_error[plot_idx],
        color = _COLOR_BAND, alpha = 0.18, label = "+/- 2 Standard Errors",
    )
    ax.axhline(
        results.expected_pn1, color = "black", linestyle = "--", linewidth = 0.8,
        label = f"Final Mean = {results.expected_pn1:,.2f}",
    )

    ax.set_title("Monte Carlo Convergence", fontsize = 11, fontweight = "bold")
    ax.set_xlabel("Simulation Draws")
    ax.set_ylabel("Cumulative Mean P&L")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(fontsize=7, loc="best", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)


# ----------------------------------------------------------------------------
# Panel 4: Executive Risk Summary Scorecard
# ----------------------------------------------------------------------------

def _plot_summary_card(ax: plt.Axes, results: RiskMetricResults, base_value: float) -> None:
    """
    Renders an institutional risk metric scorecard with base NAV and tail quantiles.
    """
    ax.axis("off")
    summary = results.summary_frame()

    rows: list[tuple[str, str]] = [
        ("Portfolio", results.portfolio_name),
        ("Valuation Date", str(results.valuation_date)),
        ("Base Portfolio Value", f"{base_value:,.2f}"),
        ("Mean Simulated P&L", f"{results.expected_pn1:,.2f}"),
        ("P&L Std. Error", f"{results.pn1_std_error:,.4f}"),
        ("", ""),
    ]

    for alpha, row in summary.iterrows():
        confidence_pct = int(round(row["confidence"] * 100))
        rows.append((f"VaR {confidence_pct}%", f"{row['VaR']:,.2f}"))
        rows.append((f"ES {confidence_pct}%", f"{row['ES']:,.2f}"))

    ax.set_title("Risk Summary", fontsize=11, fontweight="bold", loc="left")

    y = 0.92
    line_height = 0.85 / max(len(rows), 1)
    for label, value in rows:
        if label == "" and value == "":
            y -= line_height * 0.4
            continue
        ax.text(0.02, y, label, fontsize = 9.5, ha = "left", va = "top", transform = ax.transAxes)
        ax.text(0.98, y, value, fontsize = 9.5, ha = "right", va = "top", transform = ax.transAxes)
        y -= line_height

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")

# ----------------------------------------------------------------------------
# Public Dashboard Entry Point
# ----------------------------------------------------------------------------

def plot_dashboard(
        results: RiskMetricResults,
        base_value: float, 
        save_path: str | Path | None = None
) -> plt.Figure:
    """
    Generates and optionally exports the 2x2 Quantitative Risk Dashboard.

    Parameters:
    -----------
    results : RiskMetricResults
        Simulated P&L paths, VaR/ES quantiles and Greeks from 'RiskEngine.run()'
    base_value : float
        Current unstressed portfolio Net Asset Value V(0)
    save_path : str | Path | None
        Optional file path to save the generated image

    Returns:
    --------
    plt.Figure
        The assembled Matplotlib figure object
    """

    style_name = "seaborn-v0_8-whitegrid"
    plt.style.use(style_name if style_name in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize = (14, 9.5))
    fig.suptitle(
        f"Quantitative Risk Dashboard  |  {results.portfolio_name}  |  {results.valuation_date}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    _plot_pnl_distribution(axes[0, 0], results)
    _plot_greeks(axes[0, 1], results)
    _plot_convergence(axes[1, 0], results)
    _plot_summary_card(axes[1, 1], results, base_value)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig

# ----------------------------------------------------------------------------
# Macro Stress-Testing Waterfall Visualiser
# ----------------------------------------------------------------------------

def plot_stress_test_results(
    report: StressTestReport,
    save_path: str | Path | None = None
) -> plt.Figure:
    """
    Rends a dual-panel horizontal watefall chart summarising deterministic stress tests.

    Layout:
    -------
    - Left Panel: Dollar P&L impact
    - Right Panel: Percentage portfolio drawdown
    - Color Coding: Green for positive scenario gains; Red for drawdowns
    - Dynamic Annotations: Exact metrics printed beside each bar
    """

    style_name = "seaborn-v0_8-whitegrid"
    plt.style.use(style_name if style_name in plt.style.available else "default")

    frame = report.to_frame().sort_values("Dollar PnL")
    scenario_labels = frame.index.to_numpy()
    dollar_pn1 = frame["Dollar PnL"].to_numpy(dtype = float)
    pct_drawdown = frame["% Drawdown"].to_numpy(dtype = float)
    y_positions = np.arange(len(scenario_labels))

    fig_height = max(4.0, 0.65 * len(scenario_labels) + 2.0)
    fig, (ax_pnl, ax_dd) = plt.subplots(1, 2, figsize=(14, fig_height))
    fig.suptitle(
        f"Stress Test Dashboard  |  {report.portfolio_name}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # --- Left panel: Dollar P&L -------------------------------------------
    pnl_colors = np.where(dollar_pn1 >= 0.0, _COLOR_GAIN, _COLOR_LOSS)
    pnl_bars = ax_pnl.barh(
        y_positions, dollar_pn1, color=pnl_colors, edgecolor="white",
        linewidth=0.6, height=0.65,
    )
    ax_pnl.set_yticks(y_positions)
    ax_pnl.set_yticklabels(scenario_labels, fontsize=9)
    ax_pnl.axvline(0.0, color="black", linewidth=0.9)
    ax_pnl.set_title("Portfolio P&L by Scenario ($)", fontsize=11, fontweight="bold")
    ax_pnl.set_xlabel("Dollar P&L")
    ax_pnl.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax_pnl.bar_label(
        pnl_bars, labels=[f"${v:,.0f}" for v in dollar_pn1],
        padding=4, fontsize=8, fontweight="bold",
    )
    ax_pnl.grid(alpha=0.25, linewidth=0.5, axis="x")
    ax_pnl.margins(x=0.15)

    # --- Right panel: % Drawdown --------------------------------------------
    dd_colors = np.where(pct_drawdown >= 0.0, _COLOR_GAIN, _COLOR_LOSS)
    dd_bars = ax_dd.barh(
        y_positions, pct_drawdown, color=dd_colors, edgecolor="white",
        linewidth=0.6, height=0.65,
    )
    ax_dd.set_yticks(y_positions)
    ax_dd.set_yticklabels(scenario_labels, fontsize=9)
    ax_dd.axvline(0.0, color="black", linewidth=0.9)
    ax_dd.set_title("Portfolio Drawdown by Scenario (%)", fontsize=11, fontweight="bold")
    ax_dd.set_xlabel("% Drawdown")
    ax_dd.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.1f}%"))
    ax_dd.bar_label(
        dd_bars, labels=[f"{v:,.1f}%" for v in pct_drawdown],
        padding=4, fontsize=8, fontweight="bold",
    )
    ax_dd.grid(alpha=0.25, linewidth=0.5, axis="x")
    ax_dd.margins(x=0.15)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig