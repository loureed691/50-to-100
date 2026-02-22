"""
CLI entry point for the backtest module.

Usage
-----
    python -m backtest --pairs BTC-USDT,ETH-USDT --days 180 --fees 0.001 --slippage 0.0005
    python -m backtest --mode improved --pairs BTC-USDT --days 180
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from backtest.data import generate_ohlcv
from backtest.engine import BacktestResult, CostModel, StrategyParams, run_backtest
from backtest.metrics import compute_all_metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproducible backtest for the KuCoin trading bot strategy.",
    )
    p.add_argument("--pairs", default="BTC-USDT,ETH-USDT,SOL-USDT",
                    help="Comma-separated trading pairs")
    p.add_argument("--days", type=int, default=180, help="Days of synthetic data")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--capital", type=float, default=50.0, help="Initial capital USDT")
    p.add_argument("--fees", type=float, default=0.001, help="Maker/taker fee fraction")
    p.add_argument("--slippage", type=float, default=0.0005, help="Slippage fraction")
    p.add_argument("--spread", type=float, default=0.0003, help="Half-spread fraction")
    p.add_argument("--mode", choices=["baseline", "improved"], default="baseline",
                    help="Strategy mode")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    return p.parse_args(argv)


def _format_report(metrics: dict, mode: str, result: BacktestResult) -> str:
    lines = [
        f"# Backtest Report — {mode.upper()}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Net Profit | {metrics['net_profit']:.4f} USDT |",
        f"| CAGR | {metrics['cagr']:.2%} |",
        f"| Sharpe Ratio | {metrics['sharpe']:.4f} |",
        f"| Sortino Ratio | {metrics['sortino']:.4f} |",
        f"| Max Drawdown | {metrics['max_drawdown']:.2%} |",
        f"| 95% VaR | {metrics['var_95']:.4f} |",
        f"| 95% CVaR | {metrics['cvar_95']:.4f} |",
        f"| Worst Day | {metrics['worst_day']:.4f} |",
        f"| Worst Week | {metrics['worst_week']:.4f} |",
        f"| Win Rate | {metrics['win_rate']:.2%} |",
        f"| Profit Factor | {metrics['profit_factor']:.4f} |",
        f"| Avg Trade | {metrics['avg_trade']:.4f} USDT |",
        f"| Trade Count | {metrics['trade_count']} |",
        f"| Total Execution Costs | {result.total_execution_costs:.4f} USDT |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    pairs = [p.strip() for p in args.pairs.split(",")]

    cost = CostModel(
        maker_fee=args.fees,
        taker_fee=args.fees,
        slippage=args.slippage,
        spread=args.spread,
    )

    # Generate synthetic data with per-pair seeds for reproducibility
    data: dict[str, pd.DataFrame] = {}
    for idx, pair in enumerate(pairs):
        base_prices = {"BTC-USDT": 40000.0, "ETH-USDT": 2500.0, "SOL-USDT": 100.0}
        bp = base_prices.get(pair, 100.0)
        data[pair] = generate_ohlcv(
            symbol=pair,
            days=args.days,
            seed=args.seed + idx,
            base_price=bp,
        )

    # Build strategy params
    if args.mode == "improved":
        params = StrategyParams(
            use_trend_filter=True,
            trend_ema_period=50,
            use_atr_sizing=True,
            risk_per_trade=0.01,
            use_trailing_stop=True,
            trailing_atr_mult=2.0,
            use_min_edge=True,
            min_edge_mult=1.5,
            use_cooldown=True,
            cooldown_bars=6,
            max_consecutive_losses=3,
        )
    else:
        params = StrategyParams()

    result = run_backtest(data, params, cost, initial_capital=args.capital)

    if len(result.equity_curve) < 2:
        print("Insufficient data for metrics computation.", file=sys.stderr)
        return {}

    metrics = compute_all_metrics(
        result.equity_curve,
        result.daily_returns,
        result.trade_pnls,
    )

    if args.json:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        report = _format_report(metrics, args.mode, result)
        print(report)

    return metrics


if __name__ == "__main__":
    main()
