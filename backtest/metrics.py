"""
Risk and return metrics for backtest evaluation.

All metrics operate on a pandas Series of daily portfolio returns or an
equity curve (cumulative portfolio value over time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_profit(equity_curve: pd.Series) -> float:
    """Absolute profit: final equity minus initial equity."""
    return float(equity_curve.iloc[-1] - equity_curve.iloc[0])


def cagr(equity_curve: pd.Series, trading_days: int = 365) -> float:
    """Compound Annual Growth Rate."""
    total = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_days = len(equity_curve)
    if n_days <= 1 or total <= 0:
        return 0.0
    return float(total ** (trading_days / n_days) - 1)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown)."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(-dd.min()) if dd.min() < 0 else 0.0


def sharpe_ratio(
    daily_returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 365,
) -> float:
    """Annualised Sharpe ratio."""
    excess = daily_returns - risk_free_rate / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(
    daily_returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 365,
) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    excess = daily_returns - risk_free_rate / periods
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def var_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Value at Risk and Conditional VaR (Expected Shortfall) at given confidence.

    Returns
    -------
    (VaR, CVaR) : tuple[float, float]
        Both are positive values representing loss magnitude.
    """
    sorted_returns = daily_returns.sort_values()
    cutoff_idx = int(len(sorted_returns) * (1 - confidence))
    cutoff_idx = max(cutoff_idx, 1)
    tail = sorted_returns.iloc[:cutoff_idx]
    var = float(-tail.iloc[-1]) if len(tail) > 0 else 0.0
    cvar = float(-tail.mean()) if len(tail) > 0 else 0.0
    return var, cvar


def worst_day(daily_returns: pd.Series) -> float:
    """Worst single-day return (most negative)."""
    return float(daily_returns.min())


def worst_week(daily_returns: pd.Series) -> float:
    """Worst rolling 7-day return."""
    if len(daily_returns) < 7:
        return float(daily_returns.sum())
    rolling = daily_returns.rolling(7).sum()
    return float(rolling.min())


def win_rate(trade_pnls: list[float]) -> float:
    """Fraction of trades that are profitable."""
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def profit_factor(trade_pnls: list[float]) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_trade(trade_pnls: list[float]) -> float:
    """Average P&L per trade."""
    if not trade_pnls:
        return 0.0
    return sum(trade_pnls) / len(trade_pnls)


def compute_all_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trade_pnls: list[float],
) -> dict:
    """Compute the full suite of risk and return metrics.

    Returns a dict suitable for rendering in a report.
    """
    var_95, cvar_95 = var_cvar(daily_returns, 0.95)
    return {
        "net_profit": net_profit(equity_curve),
        "cagr": cagr(equity_curve),
        "sharpe": sharpe_ratio(daily_returns),
        "sortino": sortino_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "worst_day": worst_day(daily_returns),
        "worst_week": worst_week(daily_returns),
        "win_rate": win_rate(trade_pnls),
        "profit_factor": profit_factor(trade_pnls),
        "avg_trade": avg_trade(trade_pnls),
        "trade_count": len(trade_pnls),
    }
