# 50-to-100 — KuCoin Intra-Day Trading Bot

An aggressive intra-day scalping bot for [KuCoin](https://www.kucoin.com/) that
aims to compound a starting capital of **50 USDT** towards **10 000 USDT** in a
single trading day by riding high-momentum setups across multiple USDT pairs.

> ⚠️ **Risk warning** — Cryptocurrency trading involves a significant risk of
> capital loss.  This bot is provided for **educational purposes only**.  Never
> risk money you cannot afford to lose.  Past performance does not guarantee
> future results.

---

## What's working now

- [x] RSI(14) + EMA-9/21 crossover signal detection on 5-minute candles
- [x] Automated position sizing (configurable fraction of available balance)
- [x] Hard stop-loss and take-profit per position
- [x] Consecutive-loss circuit breaker with cool-down
- [x] Equity floor emergency halt
- [x] Paper mode — real market data, simulated orders, no real funds at risk
- [x] All configuration via environment variables (`.env` supported)
- [x] 137 unit tests (all passing, no network calls)
- [x] CI pipeline (GitHub Actions: lint + tests on every push)
- [x] Structured logging to stdout and file
- [x] Regime detection (ADX-based) — filters out ranging markets
- [x] Confidence scoring — 0-1 signal quality score for entry filtering
- [x] Volatility-based position sizing with strict risk budgets (max portfolio heat)
- [x] Slippage caps — reject fills exceeding configured deviation
- [x] Decimal precision for quantity rounding
- [x] Limit-order-first execution (falls back to market orders)
- [x] Baseline-vs-improved comparison report (`python -m backtest --mode compare`)
- [x] Walk-forward OOS optimization to prevent overfitting
- [x] Monte Carlo trade resampling for robustness validation

---

## Strategy

| Component | Detail |
|-----------|--------|
| Signal | RSI(14) recovery from oversold **+** EMA-9 crossing above EMA-21 |
| Confidence | Multi-factor 0-1 score (RSI strength, EMA separation, trend alignment, ADX, volume) |
| Regime filter | ADX-based — only trades in trending markets (ADX > 25) |
| Time frame | 5-minute candles |
| Position sizing | Volatility-based via ATR, capped by max portfolio heat (6% default) |
| Stop-loss | −1.5 % from entry |
| Take-profit | +2.5 % from entry |
| Execution | Limit-order first (falls back to market); slippage cap at 0.2% |
| Circuit-breaker | Auto-pauses after 5 consecutive losses (5-minute cool-down) |
| Pairs | BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX vs USDT (configurable) |

---

## Requirements

* Python 3.10+
* A KuCoin account with **API credentials** (API key, secret, passphrase) — required for live
  trading; optional when running in **paper mode** (market data endpoints are public)

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/loureed691/50-to-100.git
cd 50-to-100

# 2. Install dependencies
make install
# or: pip install -r requirements.txt

# 3. Configure (copy the example and edit)
cp .env.example .env
# Edit .env — set PAPER_MODE=true for simulation, or add real API keys for live trading

# 4. Run (paper mode — no real funds at risk)
make dev
# or: PAPER_MODE=true python bot.py

# 5. Run tests
make test
# or: python -m pytest tests/ -v
```

---

## Configuration

Copy `.env.example` to `.env` and adjust the values.  All parameters can also be set
via environment variables or by editing `config.py`.  **Never commit `.env`.**

| Variable | Default | Description |
|----------|---------|-------------|
| `KUCOIN_API_KEY` | — | KuCoin API key (required for live trading) |
| `KUCOIN_API_SECRET` | — | KuCoin API secret (required for live trading) |
| `KUCOIN_API_PASSPHRASE` | — | KuCoin API passphrase (required for live trading) |
| `PAPER_MODE` | `false` | Simulate orders without sending them to KuCoin (real market data, no real trades) |
| `INITIAL_CAPITAL` | `50.0` | Starting USDT amount (used as paper balance in paper mode) |
| `TARGET_CAPITAL_USDT` | `10000.0` | Profit target — bot stops when equity reaches this value |
| `TRADE_FRACTION` | `0.95` | Fraction of balance used per trade batch |
| `STOP_LOSS_PCT` | `0.015` | Stop-loss as a fraction of entry price (1.5 %) |
| `TAKE_PROFIT_PCT` | `0.025` | Take-profit as a fraction of entry price (2.5 %) |
| `TRADING_PAIRS` | BTC-USDT,ETH-USDT,… | Comma-separated list of KuCoin symbols to scan (see `.env.example` for full default list) |
| `KLINE_INTERVAL` | `5min` | Candle interval for signal generation |
| `POLL_INTERVAL_SECONDS` | `30` | Main loop frequency (seconds) |
| `MAX_OPEN_POSITIONS` | `3` | Maximum concurrent open positions |
| `EQUITY_FLOOR_USDT` | `10.0` | Emergency halt if equity drops below this |
| `MAX_CONSECUTIVE_LOSSES` | `5` | Trigger cool-down after N losses in a row |
| `COOLDOWN_SECONDS` | `300` | Cool-down duration in seconds |
| `MIN_CONFIDENCE` | `0.0` | Minimum signal confidence (0-1) to open a position (0 = disabled) |
| `MAX_PORTFOLIO_HEAT` | `0.06` | Max fraction of capital at risk across all open positions |
| `MAX_SLIPPAGE_PCT` | `0.002` | Reject entries when fill deviates more than 0.2 % from expected price |
| `USE_LIMIT_ORDERS` | `false` | Prefer limit orders over market orders (falls back to market) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FILE` | `trading_bot.log` | Path to the log file |

### Advanced configuration

Tunable indicators and limits — see `.env.example` for exact defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `RSI_PERIOD` | `14` | Look-back period (candles) for the RSI indicator |
| `RSI_OVERSOLD` | `35` | RSI threshold below which an asset is considered oversold |
| `RSI_OVERBOUGHT` | `65` | RSI threshold above which an asset is considered overbought |
| `EMA_SHORT` | `9` | Short EMA period used in the crossover signal |
| `EMA_LONG` | `21` | Long EMA period used in the crossover signal |
| `KLINE_LIMIT` | `100` | Number of historical candles fetched per indicator calculation |
| `MIN_TRADE_BALANCE_USDT` | `1.0` | Minimum USDT balance required before opening new trades |

---

## Commands

| Command | Description |
|---------|-------------|
| `make install` | Install runtime dependencies (`requirements.txt`) |
| `make install-dev` | Install runtime + dev dependencies (`requirements-dev.txt`, includes ruff & pytest) |
| `make dev` | Install dev deps and run bot in paper mode |
| `make test` | Install dev deps, run lint (ruff) + full unit test suite |
| `make verify` | Run lint + tests + baseline, improved, and compare backtests |
| `make lint` | Run ruff linter only |
| `make fmt` | Auto-fix safe lint issues and format code |

---

## Project structure

```
50-to-100/
├── bot.py           # Main bot: strategy, order management, main loop
├── config.py        # All configuration loaded from environment variables
├── backtest/        # Backtest & optimisation module
│   ├── __init__.py
│   ├── engine.py    # Simulation engine with regime detection, confidence scoring
│   ├── __main__.py  # CLI: --mode baseline|improved|compare
│   ├── data.py      # Synthetic OHLCV generation
│   ├── metrics.py   # Risk/return metrics (Sharpe, Sortino, MDD, CVaR, …)
│   └── optimization.py  # Walk-forward OOS & Monte Carlo
├── reports/
├── requirements.txt     # Runtime Python dependencies
├── requirements-dev.txt # Dev dependencies (ruff, pytest)
├── Makefile         # Developer shortcuts (install / dev / test / lint / verify)
├── .env.example     # Environment variable template (copy to .env)
├── CHANGELOG.md     # Release notes
├── SECURITY.md      # Vulnerability reporting and threat model
├── .github/
│   └── workflows/
│       └── ci.yml   # GitHub Actions CI (lint + tests)
└── tests/
    ├── test_bot.py          # Bot unit tests
    ├── test_integration.py  # Full cycle integration tests
    ├── test_backtest.py     # Backtest engine, metrics, CLI tests
    └── test_optimization.py # Walk-forward & Monte Carlo tests
```

---

## Backtesting

Run reproducible backtests using synthetic OHLCV data with realistic fee, slippage,
and spread modelling:

```bash
# Baseline strategy
python -m backtest --mode baseline --days 180 --seed 42

# Improved strategy (regime filter, confidence scoring, ATR sizing, trailing stops)
python -m backtest --mode improved --days 180 --seed 42

# Side-by-side comparison report (Sharpe, Sortino, MDD, CVaR, worst day/week, net profit)
python -m backtest --mode compare --days 180 --seed 42

# JSON output for programmatic use
python -m backtest --mode compare --days 180 --seed 42 --json
```

The comparison report outputs a table like:

| Metric | Baseline | Improved | Delta |
|--------|----------|----------|-------|
| Sharpe Ratio | -11.39 | -2.20 | +9.19 |
| Max Drawdown | 80.16% | 8.71% | -71.46% |
| Trade Count | 341 | 111 | -230 |

Walk-forward OOS optimisation and Monte Carlo resampling are available via
`backtest.optimization` for parameter robustness validation.

---

## CI

Every push and pull request runs the GitHub Actions CI pipeline:

1. Install dependencies
2. Lint with `ruff`
3. Run all unit tests with `pytest`

Status badge: [![CI](https://github.com/loureed691/50-to-100/actions/workflows/ci.yml/badge.svg)](https://github.com/loureed691/50-to-100/actions/workflows/ci.yml)

---

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability reporting process and threat model.

---

## Major changes

| Change | Reason |
|--------|--------|
| `.env.example` added | Documents all env vars; enables one-step configuration from a fresh clone |
| `Makefile` added | Single-command `make dev` / `make test` / `make lint` — removes friction for new contributors |
| `.github/workflows/ci.yml` added | Catches regressions automatically on every push across Python 3.10/3.11/3.12 |
| `SECURITY.md` added | Provides responsible disclosure process and documents the threat model |
| `CHANGELOG.md` added | Tracks notable changes following Keep a Changelog format |
| Lint errors fixed (`tests/test_bot.py`) | Resolves 10 `ruff` warnings (E402, F841, E741) so `make test` passes cleanly |

---

## Disclaimer

This software is provided "as is", without warranty of any kind.  The authors
are not responsible for any financial losses incurred through its use.
