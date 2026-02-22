# Baseline Backtest Report

## Strategy Description

The **baseline** strategy uses the current bot logic without modifications:

- **Entry**: RSI(14) crosses above oversold threshold (35) AND EMA-9 crosses above EMA-21
- **Exit**: Fixed stop-loss at 1.5% and take-profit at 2.5%
- **Sizing**: 95% of available capital per trade, split across up to 3 concurrent positions
- **Cost model**: 0.1% taker fee + 0.05% slippage + 0.03% half-spread per side

## Reproduce

```bash
python -m backtest --mode baseline --pairs BTC-USDT,ETH-USDT,SOL-USDT --days 180 --seed 42 --capital 50.0 --fees 0.001 --slippage 0.0005 --spread 0.0003
```

## Metrics (seed=42, 180 days, 3 pairs)

| Metric | Value |
|--------|-------|
| Net Profit | -49.0074 USDT |
| CAGR | -2.72% |
| Sharpe Ratio | -0.2743 |
| Sortino Ratio | -0.1238 |
| Max Drawdown | 98.77% |
| 95% VaR | 0.0000 |
| 95% CVaR | 0.0105 |
| Worst Day | -6.16% |
| Worst Week | -7.99% |
| Win Rate | 41.10% |
| Profit Factor | 0.8677 |
| Avg Trade | -0.0271 USDT |
| Trade Count | 1810 |

## Key Observations

1. **Excessive trading**: 1810 trades over 180 days means ~10 trades/day. This generates
   significant fee drag.
2. **Low win rate**: Only 41.1% of trades are profitable, and the profit factor is below 1.0.
3. **Near-total drawdown**: 98.77% max drawdown means the strategy nearly wipes out the account.
4. **Negative expectancy**: The strategy loses money after accounting for realistic costs.

The baseline establishes the risk and performance floor that the improved strategy must not exceed
(for risk) while attempting to improve (for performance).
