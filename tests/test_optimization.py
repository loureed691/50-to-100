"""
Tests for walk-forward optimization and Monte Carlo validation.
"""

import unittest

from backtest.data import generate_ohlcv
from backtest.engine import CostModel
from backtest.optimization import (
    MonteCarloResult,
    WalkForwardFold,
    monte_carlo_trades,
    walk_forward,
)


class TestWalkForward(unittest.TestCase):
    def _get_data(self, days=120, seed=42):
        return {"TEST-USDT": generate_ohlcv("TEST-USDT", days=days, seed=seed)}

    def test_returns_correct_number_of_folds(self):
        data = self._get_data()
        param_grid = {"rsi_oversold": [30.0, 35.0]}
        results = walk_forward(data, param_grid, CostModel(), n_folds=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], WalkForwardFold)

    def test_fold_has_params(self):
        data = self._get_data()
        param_grid = {"rsi_oversold": [35.0]}
        results = walk_forward(data, param_grid, CostModel(), n_folds=2)
        self.assertIn("rsi_oversold", results[0].best_params)


class TestMonteCarlo(unittest.TestCase):
    def test_with_trades(self):
        pnls = [1.0, -0.5, 2.0, -1.0, 0.5, -0.3, 1.5, -0.8]
        result = monte_carlo_trades(pnls, n_simulations=100, seed=42)
        self.assertIsInstance(result, MonteCarloResult)
        self.assertIsInstance(result.median_sharpe, float)
        self.assertIsInstance(result.p5_profit, float)

    def test_empty_trades(self):
        result = monte_carlo_trades([], n_simulations=10)
        self.assertAlmostEqual(result.median_sharpe, 0.0)
        self.assertAlmostEqual(result.median_profit, 0.0)

    def test_deterministic(self):
        pnls = [1.0, -0.5, 2.0, -1.0]
        r1 = monte_carlo_trades(pnls, n_simulations=50, seed=99)
        r2 = monte_carlo_trades(pnls, n_simulations=50, seed=99)
        self.assertAlmostEqual(r1.median_sharpe, r2.median_sharpe)
        self.assertAlmostEqual(r1.median_profit, r2.median_profit)

    def test_confidence_interval_ordering(self):
        pnls = [1.0, -0.5, 2.0, -1.0, 0.5, -0.3]
        result = monte_carlo_trades(pnls, n_simulations=200, seed=42)
        self.assertLessEqual(result.p5_profit, result.median_profit)
        self.assertLessEqual(result.median_profit, result.p95_profit)


if __name__ == "__main__":
    unittest.main()
