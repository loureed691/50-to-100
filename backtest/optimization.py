"""
Walk-forward optimisation and Monte Carlo validation.

Provides conservative parameter optimisation with out-of-sample validation
to guard against overfitting.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, CostModel, StrategyParams, run_backtest
from backtest.metrics import compute_all_metrics

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward optimisation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardFold:
    """Result of a single walk-forward fold."""
    train_sharpe: float
    test_sharpe: float
    train_sortino: float
    test_sortino: float
    train_mdd: float
    test_mdd: float
    best_params: dict


def walk_forward(
    data: dict[str, pd.DataFrame],
    param_grid: dict[str, list],
    cost: CostModel,
    initial_capital: float = 50.0,
    n_folds: int = 3,
    train_ratio: float = 0.7,
) -> list[WalkForwardFold]:
    """Run walk-forward optimisation over the parameter grid.

    Splits data into n_folds sequential train/test windows. For each fold,
    optimises parameters on the train window by Sharpe, then evaluates on
    the test window.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Symbol → OHLCV DataFrame with a 'timestamp' column.
    param_grid : dict[str, list]
        Parameter name → list of values to search.
    cost : CostModel
        Transaction cost model.
    initial_capital : float
        Starting capital.
    n_folds : int
        Number of walk-forward windows.
    train_ratio : float
        Fraction of each window used for training.

    Returns
    -------
    list[WalkForwardFold]
    """
    # Determine total data length (use the shortest symbol)
    min_len = min(len(df) for df in data.values())
    fold_size = min_len // n_folds
    if fold_size < 100:
        log.warning("Fold size (%d) is very small; results may be unreliable", fold_size)

    results: list[WalkForwardFold] = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = min(start + fold_size, min_len)
        split = start + int((end - start) * train_ratio)

        train_data = {s: df.iloc[start:split].reset_index(drop=True) for s, df in data.items()}
        test_data = {s: df.iloc[split:end].reset_index(drop=True) for s, df in data.items()}

        # Grid search on train set
        best_sharpe = -np.inf
        best_params: dict = {}
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for combo in itertools.product(*param_values):
            kw = dict(zip(param_names, combo))
            params = StrategyParams(**kw)
            result = run_backtest(train_data, params, cost, initial_capital)
            if len(result.equity_curve) < 2:
                continue
            metrics = compute_all_metrics(
                result.equity_curve, result.daily_returns, result.trade_pnls,
            )
            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_params = kw

        # Evaluate best params on test set
        train_result = run_backtest(
            train_data, StrategyParams(**best_params), cost, initial_capital,
        )
        test_result = run_backtest(
            test_data, StrategyParams(**best_params), cost, initial_capital,
        )

        train_metrics = _safe_metrics(train_result)
        test_metrics = _safe_metrics(test_result)

        results.append(WalkForwardFold(
            train_sharpe=train_metrics.get("sharpe", 0.0),
            test_sharpe=test_metrics.get("sharpe", 0.0),
            train_sortino=train_metrics.get("sortino", 0.0),
            test_sortino=test_metrics.get("sortino", 0.0),
            train_mdd=train_metrics.get("max_drawdown", 0.0),
            test_mdd=test_metrics.get("max_drawdown", 0.0),
            best_params=best_params,
        ))

    return results


def _safe_metrics(result: BacktestResult) -> dict:
    if len(result.equity_curve) < 2:
        return {}
    return compute_all_metrics(
        result.equity_curve, result.daily_returns, result.trade_pnls,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    """Summary of Monte Carlo simulation."""
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_mdd: float
    p95_mdd: float
    median_profit: float
    p5_profit: float
    p95_profit: float


def monte_carlo_trades(
    trade_pnls: list[float],
    n_simulations: int = 1000,
    seed: int = 42,
    initial_capital: float = 50.0,
) -> MonteCarloResult:
    """Bootstrap resample trade P&Ls to estimate confidence intervals.

    Shuffles the order of trades to simulate different sequencing outcomes.

    Parameters
    ----------
    trade_pnls : list[float]
        List of trade P&L values.
    n_simulations : int
        Number of bootstrap samples.
    seed : int
        Random seed for reproducibility.
    initial_capital : float
        Starting capital for equity curve construction.

    Returns
    -------
    MonteCarloResult
    """
    if not trade_pnls:
        return MonteCarloResult(
            median_sharpe=0.0, p5_sharpe=0.0, p95_sharpe=0.0,
            median_mdd=0.0, p95_mdd=0.0,
            median_profit=0.0, p5_profit=0.0, p95_profit=0.0,
        )

    rng = np.random.RandomState(seed)
    pnls = np.array(trade_pnls)
    sharpes: list[float] = []
    mdds: list[float] = []
    profits: list[float] = []

    for _ in range(n_simulations):
        shuffled = rng.choice(pnls, size=len(pnls), replace=True)
        equity = initial_capital + np.cumsum(shuffled)
        equity_series = pd.Series(equity)

        final_profit = float(equity[-1] - initial_capital)
        profits.append(final_profit)

        # Max drawdown
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak
        mdd = float(-dd.min()) if dd.min() < 0 else 0.0
        mdds.append(mdd)

        # Sharpe approximation from trade returns
        if len(shuffled) > 1 and np.std(shuffled) > 0:
            sharpe = float(np.mean(shuffled) / np.std(shuffled) * np.sqrt(len(shuffled)))
        else:
            sharpe = 0.0
        sharpes.append(sharpe)

    return MonteCarloResult(
        median_sharpe=float(np.median(sharpes)),
        p5_sharpe=float(np.percentile(sharpes, 5)),
        p95_sharpe=float(np.percentile(sharpes, 95)),
        median_mdd=float(np.median(mdds)),
        p95_mdd=float(np.percentile(mdds, 95)),
        median_profit=float(np.median(profits)),
        p5_profit=float(np.percentile(profits, 5)),
        p95_profit=float(np.percentile(profits, 95)),
    )
