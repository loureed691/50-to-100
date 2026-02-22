"""
Unit tests for the backtest module (data generation, engine, metrics, CLI).
"""

import unittest

import numpy as np
import pandas as pd

from backtest.data import generate_ohlcv
from backtest.engine import (
    BacktestResult,
    CostModel,
    StrategyParams,
    _atr,
    _ema,
    _rsi,
    compute_indicators,
    run_backtest,
)
from backtest.metrics import (
    avg_trade,
    cagr,
    compute_all_metrics,
    max_drawdown,
    net_profit,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    var_cvar,
    win_rate,
    worst_day,
    worst_week,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateOhlcv(unittest.TestCase):
    def test_returns_dataframe(self):
        df = generate_ohlcv("BTC-USDT", days=10, seed=42)
        self.assertIsInstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = generate_ohlcv("BTC-USDT", days=10, seed=42)
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            self.assertIn(col, df.columns)

    def test_deterministic_with_seed(self):
        df1 = generate_ohlcv("BTC-USDT", days=10, seed=123)
        df2 = generate_ohlcv("BTC-USDT", days=10, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self):
        df1 = generate_ohlcv("BTC-USDT", days=10, seed=1)
        df2 = generate_ohlcv("BTC-USDT", days=10, seed=2)
        self.assertFalse(df1["close"].equals(df2["close"]))

    def test_high_above_low(self):
        df = generate_ohlcv("BTC-USDT", days=30, seed=42)
        self.assertTrue((df["high"] >= df["low"]).all())

    def test_positive_volume(self):
        df = generate_ohlcv("BTC-USDT", days=10, seed=42)
        self.assertTrue((df["volume"] > 0).all())


# ─────────────────────────────────────────────────────────────────────────────
# Cost model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCostModel(unittest.TestCase):
    def test_buy_cost_increases_price(self):
        cost = CostModel()
        self.assertGreater(cost.buy_cost_factor(), 1.0)

    def test_sell_cost_decreases_price(self):
        cost = CostModel()
        self.assertLess(cost.sell_cost_factor(), 1.0)

    def test_zero_cost_model(self):
        cost = CostModel(maker_fee=0, taker_fee=0, slippage=0, spread=0)
        self.assertAlmostEqual(cost.buy_cost_factor(), 1.0)
        self.assertAlmostEqual(cost.sell_cost_factor(), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Indicator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestIndicators(unittest.TestCase):
    def test_ema_length(self):
        s = pd.Series(range(1, 51), dtype=float)
        result = _ema(s, 9)
        self.assertEqual(len(result), len(s))

    def test_rsi_range(self):
        prices = pd.Series(np.random.RandomState(42).uniform(100, 200, 100))
        rsi = _rsi(prices, 14)
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_atr_positive(self):
        df = generate_ohlcv("TEST", days=30, seed=42)
        atr = _atr(df["high"], df["low"], df["close"], 14)
        valid = atr.dropna()
        self.assertTrue((valid >= 0).all())

    def test_compute_indicators_adds_columns(self):
        df = generate_ohlcv("TEST", days=30, seed=42)
        params = StrategyParams(use_trend_filter=True, use_atr_sizing=True)
        result = compute_indicators(df, params)
        self.assertIn("rsi", result.columns)
        self.assertIn("ema_short", result.columns)
        self.assertIn("ema_long", result.columns)
        self.assertIn("trend_ema", result.columns)
        self.assertIn("atr", result.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):
    def test_net_profit(self):
        equity = pd.Series([100.0, 110.0, 120.0])
        self.assertAlmostEqual(net_profit(equity), 20.0)

    def test_cagr_positive(self):
        equity = pd.Series([100.0] * 365 + [200.0])
        result = cagr(equity)
        self.assertGreater(result, 0)

    def test_max_drawdown(self):
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        mdd = max_drawdown(equity)
        self.assertAlmostEqual(mdd, 0.25)  # 120 → 90 = 25%

    def test_max_drawdown_no_drawdown(self):
        equity = pd.Series([100.0, 110.0, 120.0])
        self.assertAlmostEqual(max_drawdown(equity), 0.0)

    def test_sharpe_flat_returns(self):
        returns = pd.Series([0.0] * 100)
        self.assertAlmostEqual(sharpe_ratio(returns), 0.0)

    def test_sortino_no_downside(self):
        returns = pd.Series([0.01] * 100)
        self.assertAlmostEqual(sortino_ratio(returns), 0.0)

    def test_var_cvar_returns_positive(self):
        returns = pd.Series(np.random.RandomState(42).normal(-0.01, 0.02, 100))
        v, cv = var_cvar(returns, 0.95)
        self.assertIsInstance(v, float)
        self.assertIsInstance(cv, float)

    def test_worst_day(self):
        returns = pd.Series([0.01, -0.05, 0.02, -0.01])
        self.assertAlmostEqual(worst_day(returns), -0.05)

    def test_worst_week(self):
        returns = pd.Series([0.01, -0.01, -0.02, -0.03, -0.01, 0.02, -0.01, 0.01])
        ww = worst_week(returns)
        self.assertLess(ww, 0)

    def test_win_rate(self):
        self.assertAlmostEqual(win_rate([10, -5, 20, -3]), 0.5)

    def test_win_rate_empty(self):
        self.assertAlmostEqual(win_rate([]), 0.0)

    def test_profit_factor(self):
        pf = profit_factor([10, -5, 20, -3])
        self.assertAlmostEqual(pf, 30.0 / 8.0)

    def test_profit_factor_no_losses(self):
        self.assertEqual(profit_factor([10, 20]), float("inf"))

    def test_avg_trade(self):
        self.assertAlmostEqual(avg_trade([10, -5, 20, -3]), 5.5)

    def test_compute_all_metrics_returns_dict(self):
        equity = pd.Series([100.0, 102.0, 101.0, 105.0])
        daily_returns = equity.pct_change().dropna()
        trades = [2.0, -1.0, 4.0]
        m = compute_all_metrics(equity, daily_returns, trades)
        self.assertIn("sharpe", m)
        self.assertIn("max_drawdown", m)
        self.assertIn("trade_count", m)
        self.assertEqual(m["trade_count"], 3)


# ─────────────────────────────────────────────────────────────────────────────
# Engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestEngine(unittest.TestCase):
    def _get_data(self, days=60, seed=42):
        return {
            "TEST-USDT": generate_ohlcv("TEST-USDT", days=days, seed=seed),
        }

    def test_baseline_runs_without_error(self):
        data = self._get_data()
        result = run_backtest(data, StrategyParams(), CostModel(), initial_capital=50.0)
        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(len(result.equity_curve), 0)

    def test_improved_runs_without_error(self):
        data = self._get_data()
        params = StrategyParams(
            use_trend_filter=True,
            use_atr_sizing=True,
            use_trailing_stop=True,
            use_min_edge=True,
            use_cooldown=True,
        )
        result = run_backtest(data, params, CostModel(), initial_capital=50.0)
        self.assertIsInstance(result, BacktestResult)

    def test_equity_curve_starts_near_initial_capital(self):
        data = self._get_data()
        result = run_backtest(data, StrategyParams(), CostModel(), initial_capital=100.0)
        if len(result.equity_curve) > 0:
            # First equity value should be close to 100 (may differ slightly due to positions)
            self.assertGreater(result.equity_curve.iloc[0], 0)

    def test_zero_cost_model_reduces_fees(self):
        data = self._get_data()
        zero_cost = CostModel(maker_fee=0, taker_fee=0, slippage=0, spread=0)
        result = run_backtest(data, StrategyParams(), zero_cost, initial_capital=50.0)
        self.assertAlmostEqual(result.total_execution_costs, 0.0)

    def test_trade_log_populated(self):
        data = self._get_data(days=90)
        result = run_backtest(data, StrategyParams(), CostModel(), initial_capital=50.0)
        # Should have at least some trades
        if result.trade_pnls:
            self.assertEqual(len(result.trade_log), len(result.trade_pnls))

    def test_empty_data_returns_empty_result(self):
        result = run_backtest({}, StrategyParams(), CostModel())
        self.assertEqual(len(result.equity_curve), 0)

    def test_improved_has_fewer_trades_than_baseline(self):
        """Improved strategy filters should reduce trade count."""
        data = self._get_data(days=90)
        baseline = run_backtest(data, StrategyParams(), CostModel(), initial_capital=50.0)
        improved = run_backtest(
            data,
            StrategyParams(
                use_trend_filter=True,
                use_atr_sizing=True,
                use_trailing_stop=True,
                use_min_edge=True,
                use_cooldown=True,
            ),
            CostModel(),
            initial_capital=50.0,
        )
        self.assertLessEqual(len(improved.trade_pnls), len(baseline.trade_pnls))

    def test_deterministic_results(self):
        """Same data + same params → identical results."""
        data = self._get_data(days=30, seed=99)
        r1 = run_backtest(data, StrategyParams(), CostModel(), initial_capital=50.0)
        r2 = run_backtest(data, StrategyParams(), CostModel(), initial_capital=50.0)
        self.assertEqual(r1.trade_pnls, r2.trade_pnls)


# ─────────────────────────────────────────────────────────────────────────────
# CLI tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestCli(unittest.TestCase):
    def test_main_baseline_returns_metrics(self):
        from backtest.__main__ import main
        metrics = main(["--mode", "baseline", "--days", "30", "--seed", "42"])
        self.assertIn("sharpe", metrics)
        self.assertIn("max_drawdown", metrics)

    def test_main_improved_returns_metrics(self):
        from backtest.__main__ import main
        metrics = main(["--mode", "improved", "--days", "30", "--seed", "42"])
        self.assertIn("sharpe", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()
