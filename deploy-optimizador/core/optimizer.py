"""Portfolio optimization module using Markowitz mean-variance optimization."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    """Container for portfolio optimization results."""
    name: str
    weights: np.ndarray
    returns: float
    volatility: float
    sharpe: float
    assets: List[str]


class PortfolioOptimizer:
    """
    Portfolio optimizer using Markowitz mean-variance optimization.
    """

    def __init__(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        min_weight: Union[float, Dict[str, float], List[float], np.ndarray] = 0.0
    ):
        """
        Initialize the optimizer.

        Args:
            mean_returns: Annualized mean returns for each asset
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Annual risk-free rate (default 2%)
            min_weight: Minimum weight per asset. Accepts either:
                - a single float applied to every asset (default 0), or
                - a dict {ticker: min_weight} for per-asset minimums, or
                - a list/array aligned with mean_returns.index
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.rf = risk_free_rate
        self.num_assets = len(mean_returns)
        self.assets = list(mean_returns.index)

        self.min_weights = self._resolve_min_weights(min_weight)
        # kept for backwards compatibility / external inspection
        self.min_weight = self.min_weights

        if self.min_weights.sum() > 1.0 + 1e-9:
            raise ValueError(
                "La suma de los pesos minimos por activo supera el 100%. "
                "Reduzca alguno de los pesos minimos para que la cartera sea factible."
            )

        self.bounds = tuple((self.min_weights[i], 1) for i in range(self.num_assets))
        self.constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
        self.initial_weights = self._feasible_initial_weights()

    def _resolve_min_weights(
        self,
        min_weight: Union[float, Dict[str, float], List[float], np.ndarray]
    ) -> np.ndarray:
        """Normalize the min_weight argument into a per-asset numpy array."""
        if isinstance(min_weight, dict):
            return np.array([float(min_weight.get(a, 0.0)) for a in self.assets])
        if isinstance(min_weight, (list, tuple, np.ndarray, pd.Series)):
            arr = np.array(min_weight, dtype=float)
            if len(arr) != self.num_assets:
                raise ValueError(
                    "La lista de pesos minimos debe tener la misma cantidad de "
                    "elementos que activos."
                )
            return arr
        # single scalar applied uniformly to every asset
        return np.array([float(min_weight)] * self.num_assets)

    def _feasible_initial_weights(self) -> np.ndarray:
        """Build a starting point that already respects the per-asset minimums."""
        remaining = 1.0 - self.min_weights.sum()
        if remaining < 0:
            remaining = 0.0
        return self.min_weights + (remaining / self.num_assets)

    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio statistics.

        Args:
            weights: Portfolio weights

        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        w = np.array(weights)
        ret = np.dot(w, self.mean_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
        sharpe = (ret - self.rf) / vol if vol != 0 else 0
        return ret, vol, sharpe

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        return -self.portfolio_stats(weights)[2]

    def _min_volatility(self, weights: np.ndarray) -> float:
        """Portfolio volatility for minimization."""
        return self.portfolio_stats(weights)[1]

    def optimize_max_sharpe(self) -> PortfolioResult:
        """
        Find the portfolio with maximum Sharpe ratio.

        Returns:
            PortfolioResult with optimal weights
        """
        result = minimize(
            self._neg_sharpe,
            self.initial_weights,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints
        )
        ret, vol, sharpe = self.portfolio_stats(result.x)
        return PortfolioResult(
            name="Sharpe Optimo",
            weights=result.x,
            returns=ret,
            volatility=vol,
            sharpe=sharpe,
            assets=self.assets
        )

    def optimize_min_volatility(self) -> PortfolioResult:
        """
        Find the minimum volatility portfolio.

        Returns:
            PortfolioResult with optimal weights
        """
        result = minimize(
            self._min_volatility,
            self.initial_weights,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints
        )
        ret, vol, sharpe = self.portfolio_stats(result.x)
        return PortfolioResult(
            name="Minima Volatilidad",
            weights=result.x,
            returns=ret,
            volatility=vol,
            sharpe=sharpe,
            assets=self.assets
        )

    def optimize_target_return(self, target_return: float) -> Optional[PortfolioResult]:
        """
        Find minimum volatility portfolio for a target return.

        Args:
            target_return: Target annual return

        Returns:
            PortfolioResult or None if optimization fails
        """
        def target_constraint(weights):
            return np.dot(weights, self.mean_returns) - target_return

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': target_constraint}
        )

        result = minimize(
            self._min_volatility,
            self.initial_weights,
            method='SLSQP',
            bounds=self.bounds,
            constraints=constraints
        )

        if not result.success:
            return None

        ret, vol, sharpe = self.portfolio_stats(result.x)
        return PortfolioResult(
            name=f"Objetivo {target_return*100:.1f}%",
            weights=result.x,
            returns=ret,
            volatility=vol,
            sharpe=sharpe,
            assets=self.assets
        )

    def calculate_efficient_frontier(self, n_points: int = 100) -> Dict:
        """
        Calculate the efficient frontier.

        Args:
            n_points: Number of points on the frontier

        Returns:
            Dictionary with returns, volatilities, and weights
        """
        min_ret = min(self.mean_returns) * 0.95
        max_ret = max(self.mean_returns) * 1.05
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_volatility = []
        frontier_weights = []

        for target in target_returns:
            def target_constraint(weights):
                return np.dot(weights, self.mean_returns) - target

            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': target_constraint}
            )

            result = minimize(
                self._min_volatility,
                self.initial_weights,
                method='SLSQP',
                bounds=self.bounds,
                constraints=constraints
            )

            if result.success:
                frontier_volatility.append(result.fun)
                frontier_weights.append(result.x)
            else:
                frontier_volatility.append(np.nan)
                frontier_weights.append(None)

        return {
            'returns': target_returns,
            'volatility': np.array(frontier_volatility),
            'weights': frontier_weights
        }

    def generate_random_portfolios(self, n_portfolios: int = 10000) -> Dict:
        """
        Generate random portfolios for visualization.

        Args:
            n_portfolios: Number of random portfolios to generate

        Returns:
            Dictionary with returns, volatilities, and sharpe ratios
        """
        results = np.zeros((3, n_portfolios))
        valid_count = 0

        for i in range(n_portfolios * 2):
            if valid_count >= n_portfolios:
                break

            weights = np.random.dirichlet(np.ones(self.num_assets))

            if np.any(self.min_weights > 0) and np.any(weights < self.min_weights):
                continue

            ret, vol, sharpe = self.portfolio_stats(weights)
            results[0, valid_count] = ret
            results[1, valid_count] = vol
            results[2, valid_count] = sharpe
            valid_count += 1

        return {
            'returns': results[0, :valid_count],
            'volatility': results[1, :valid_count],
            'sharpe': results[2, :valid_count]
        }
