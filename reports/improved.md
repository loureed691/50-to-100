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
  This keeps risk constant regardless of volatility, and never exceeds the baseline's
  per-trade capital allocation.

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
| Net Profit | -29.3381 USDT |
| CAGR | -0.62% |
| Sharpe Ratio | -0.0868 |
| Sortino Ratio | -0.0355 |
| Max Drawdown | 75.76% |
| 95% VaR | 0.0000 |
| 95% CVaR | 0.0057 |
| Worst Day | -4.32% |
| Worst Week | -4.73% |
| Win Rate | 44.50% |
| Profit Factor | 0.9225 |
| Avg Trade | -0.0206 USDT |
| Trade Count | 1427 |

## Comparison: Baseline vs Improved

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| Net Profit | -49.01 USDT | -29.34 USDT | +40.1% better |
| CAGR | -2.72% | -0.62% | +77.2% better |
| Sharpe | -0.2743 | -0.0868 | +68.3% better |
| Sortino | -0.1238 | -0.0355 | +71.3% better |
| Max Drawdown | 98.77% | 75.76% | ✅ 23.3% lower risk |
| 95% CVaR | 0.0105 | 0.0057 | ✅ 45.7% lower tail risk |
| Worst Day | -6.16% | -4.32% | ✅ 29.9% improved |
| Worst Week | -7.99% | -4.73% | ✅ 40.8% improved |
| Win Rate | 41.10% | 44.50% | +3.4pp |
| Profit Factor | 0.8677 | 0.9225 | +6.3% |
| Trade Count | 1810 | 1427 | -21.2% fewer trades |

## Risk Constraint Verification

| Constraint | Baseline | Improved | Status |
|------------|----------|----------|--------|
| Max Drawdown ≤ baseline | 98.77% | 75.76% | ✅ PASS |
| 95% CVaR ≤ baseline | 0.0105 | 0.0057 | ✅ PASS |
| Worst Day ≤ baseline | -6.16% | -4.32% | ✅ PASS |
| Worst Week ≤ baseline | -7.99% | -4.73% | ✅ PASS |
| Leverage unchanged | None | None | ✅ PASS |
| Max positions ≤ baseline | 3 | 3 | ✅ PASS |
| Trading frequency ≤ baseline | 1810 | 1427 | ✅ PASS |

## Summary

All Phase 1 improvements reduce risk while improving risk-adjusted returns. No risk
constraint is violated. The improvements work by **filtering out bad trades** (trend
filter, min edge, cooldown) and **managing risk more precisely** (ATR sizing, trailing
stops). The strategy remains unprofitable on this synthetic dataset, which is expected —
we report measured results, not claimed profitability.
