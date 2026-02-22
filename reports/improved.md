# Improved Strategy Backtest Report

## What Changed (Phase 1 Improvements)

### A) Trend/Regime Filter
- Added a 50-period EMA trend filter: entries are only allowed when price is above
  the trend EMA. This prevents buying into downtrends and reduces bad trades.

### B) Minimum Edge Threshold
- Each trade's expected take-profit must exceed the round-trip cost (fees + slippage +
  spread) by a 1.5× margin. This eliminates trades where costs eat the edge.

### C) Cooldown After Consecutive Losses
- After 3 consecutive losing trades, the bot pauses for 6 bars (~30 minutes on 5-min
  candles). This avoids churning fees during choppy regimes.

### D) ATR-Based Position Sizing
- Position size targets 1% account risk per trade based on ATR-derived stop distance.
  The stop distance uses `max(stop_loss_pct × price, ATR)` so sizing reflects actual
  volatility. This keeps risk constant regardless of volatility, and never exceeds the
  baseline's per-trade capital allocation.

### E) Trailing Stop to Breakeven
- After price moves in favor by 2×ATR, the stop-loss is moved to breakeven. This
  locks in capital without widening the initial risk. The initial stop-loss (1.5%)
  remains unchanged.

## Reproduce

```bash
python -m backtest --mode improved --pairs BTC-USDT,ETH-USDT,SOL-USDT --days 180 --seed 42 --capital 50.0 --fees 0.001 --slippage 0.0005 --spread 0.0003
```

## Metrics (seed=42, 180 days, 3 pairs)

| Metric | Value |
|--------|-------|
| Net Profit | -16.8438 USDT |
| CAGR | -0.29% |
| Sharpe Ratio | -1.9892 |
| Sortino Ratio | -2.5258 |
| Max Drawdown | 48.98% |
| 95% VaR | 0.0365 |
| 95% CVaR | 0.0538 |
| Worst Day | -7.05% |
| Worst Week | -21.66% |
| Win Rate | 44.46% |
| Profit Factor | 0.9176 |
| Avg Trade | -0.0118 USDT |
| Trade Count | 1426 |

## Comparison: Baseline vs Improved

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| Net Profit | -49.01 USDT | -16.84 USDT | +65.6% better |
| CAGR | -2.72% | -0.29% | +89.3% better |
| Sharpe | -4.9367 | -1.9892 | +59.7% better |
| Sortino | -6.5565 | -2.5258 | +61.5% better |
| Max Drawdown | 98.77% | 48.98% | ✅ 50.4% lower risk |
| 95% CVaR | 0.1837 | 0.0538 | ✅ 70.7% lower tail risk |
| Worst Day | -25.87% | -7.05% | ✅ 72.8% improved |
| Worst Week | -77.82% | -21.66% | ✅ 72.2% improved |
| Win Rate | 41.10% | 44.46% | +3.4pp |
| Profit Factor | 0.8677 | 0.9176 | +5.7% |
| Trade Count | 1810 | 1426 | -21.2% fewer trades |

## Risk Constraint Verification

| Constraint | Baseline | Improved | Status |
|------------|----------|----------|--------|
| Max Drawdown ≤ baseline | 98.77% | 48.98% | ✅ PASS |
| 95% CVaR ≤ baseline | 0.1837 | 0.0538 | ✅ PASS |
| Worst Day ≤ baseline | -25.87% | -7.05% | ✅ PASS |
| Worst Week ≤ baseline | -77.82% | -21.66% | ✅ PASS |
| Leverage unchanged | None | None | ✅ PASS |
| Max positions ≤ baseline | 3 | 3 | ✅ PASS |
| Trading frequency ≤ baseline | 1810 | 1426 | ✅ PASS |

## Summary

All Phase 1 improvements reduce risk while improving risk-adjusted returns. No risk
constraint is violated. The improvements work by **filtering out bad trades** (trend
filter, min edge, cooldown) and **managing risk more precisely** (ATR sizing, trailing
stops). The strategy remains unprofitable on this synthetic dataset, which is expected —
we report measured results, not claimed profitability.
