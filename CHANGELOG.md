# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **Regime detection**: ADX-based market regime filter skips ranging/choppy markets in both bot and backtest engine.
- **Confidence scoring**: Multi-factor 0-1 signal quality score (RSI strength, EMA separation, trend alignment, ADX, volume).
- **Volatility-based sizing with risk budgets**: ATR-based position sizing capped by `MAX_PORTFOLIO_HEAT` (default 6%).
- **Slippage caps**: Configurable `MAX_SLIPPAGE_PCT` (default 0.2%) — warns on excessive fill deviation.
- **Decimal precision**: Quantity rounding uses Python `Decimal` for exact exchange-increment compliance.
- **Limit-order-first execution**: `USE_LIMIT_ORDERS` flag attempts limit orders before falling back to market orders.
- **Comparison report**: `python -m backtest --mode compare` outputs side-by-side baseline-vs-improved metrics.
- **New config parameters**: `MIN_CONFIDENCE`, `MAX_PORTFOLIO_HEAT`, `MAX_SLIPPAGE_PCT`, `USE_LIMIT_ORDERS`.
- **24 new unit tests** for regime detection, ADX indicator, confidence scoring, Decimal rounding, risk budget, slippage cap, and compare mode.
- `.env.example` — documented template for all configurable environment variables.
- `Makefile` — single-command developer experience: `make install`, `make dev`, `make test`, `make lint`.
- `.github/workflows/ci.yml` — GitHub Actions CI pipeline (lint + unit tests on every push/PR).
- `SECURITY.md` — vulnerability reporting guidance and threat model.
- `CHANGELOG.md` — this file, following Keep a Changelog format.

### Fixed
- Resolved 10 `ruff` lint warnings in `tests/test_bot.py` (E402, F841, E741).

### Changed
- README updated with "What's working now" checklist and consolidated setup/run/test commands.
