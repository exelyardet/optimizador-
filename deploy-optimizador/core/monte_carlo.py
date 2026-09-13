"""Monte Carlo simulation module using correlated Geometric Brownian Motion (GBM)."""
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict


TRADING_DAYS = 252


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def simulate_gbm_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights: np.ndarray,
    n_sims: int,
    n_days: int,
    initial_capital: float,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict:
    """
    Vectorized multi-asset correlated Geometric Brownian Motion simulation.

    Uses a Cholesky decomposition of the annualized covariance matrix to
    generate correlated shocks across assets, then projects each asset's
    price path independently before applying the portfolio's dollar
    allocation (buy-and-hold, no rebalancing):

        S_t = S_0 * exp(cumsum((mu - 0.5 * sigma^2) * dt + correlated_shocks * sqrt(dt)))

    Only the aggregates needed for the fan chart and terminal KPIs are kept
    in the cache (percentiles per day, a sample of individual paths, and the
    terminal value distribution) to keep memory usage low for large runs.

    Args:
        mean_returns: Annualized mean returns per asset
        cov_matrix: Annualized covariance matrix
        weights: Portfolio weights (aligned with mean_returns.index)
        n_sims: Number of simulated trajectories
        n_days: Projection horizon in trading days
        initial_capital: Starting portfolio value
        confidence: Confidence level for the outer fan-chart band and the
            Monte Carlo VaR/CVaR KPIs (e.g. 0.95 -> band from P5 to P95)
        seed: RNG seed for reproducibility

    Returns:
        Dict with 'percentiles' (lower/p25/median/p75/upper day-by-day
        arrays of length n_days + 1), 'sample_paths', 'terminal_values',
        'var_mc', 'cvar_mc', 'prob_loss', 'median_max_drawdown',
        'median_terminal' and 'initial_capital'.
    """
    mu = mean_returns.values.astype(np.float64)
    sigma = np.sqrt(np.diag(cov_matrix.values)).astype(np.float64)
    L = np.linalg.cholesky(cov_matrix.values.astype(np.float64)).astype(np.float32)
    n_assets = len(mu)
    dt = 1.0 / TRADING_DAYS

    rng = np.random.default_rng(seed)

    z = rng.standard_normal((n_sims, n_days, n_assets)).astype(np.float32)
    correlated_shocks = z @ L.T
    del z

    drift = ((mu - 0.5 * sigma ** 2) * dt).astype(np.float32)
    daily_log_returns = drift + correlated_shocks * np.float32(np.sqrt(dt))
    del correlated_shocks

    cum_log_returns = np.cumsum(daily_log_returns, axis=1)
    del daily_log_returns

    asset_growth = np.exp(cum_log_returns)
    del cum_log_returns

    dollar_allocation = (initial_capital * np.asarray(weights, dtype=np.float64)).astype(np.float32)
    portfolio_paths = np.einsum('sda,a->sd', asset_growth, dollar_allocation)
    del asset_growth

    day_zero = np.full((n_sims, 1), initial_capital, dtype=np.float32)
    full_paths = np.concatenate([day_zero, portfolio_paths], axis=1)
    del portfolio_paths

    lower_pct = (1 - confidence) * 100
    upper_pct = confidence * 100
    pct_levels = [lower_pct, 25, 50, 75, upper_pct]
    pct_values = np.percentile(full_paths, pct_levels, axis=0)
    percentiles = {
        'lower': pct_values[0],
        'p25': pct_values[1],
        'median': pct_values[2],
        'p75': pct_values[3],
        'upper': pct_values[4],
    }

    n_sample = min(50, n_sims)
    sample_idx = rng.choice(n_sims, size=n_sample, replace=False)
    sample_paths = full_paths[sample_idx].copy()

    terminal_values = full_paths[:, -1].copy()

    running_max = np.maximum.accumulate(full_paths, axis=1)
    drawdown = full_paths / running_max - 1
    median_max_drawdown = float(np.median(drawdown.min(axis=1)))
    del full_paths, running_max, drawdown

    var_mc = float(np.percentile(terminal_values, lower_pct))
    tail_mask = terminal_values <= var_mc
    cvar_mc = float(terminal_values[tail_mask].mean()) if tail_mask.any() else var_mc
    prob_loss = float(np.mean(terminal_values < initial_capital))
    median_terminal = float(np.median(terminal_values))

    return {
        'percentiles': percentiles,
        'sample_paths': sample_paths,
        'terminal_values': terminal_values,
        'var_mc': var_mc,
        'cvar_mc': cvar_mc,
        'prob_loss': prob_loss,
        'median_max_drawdown': median_max_drawdown,
        'median_terminal': median_terminal,
        'initial_capital': initial_capital,
        'confidence': confidence,
    }
