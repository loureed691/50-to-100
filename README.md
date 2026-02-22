# 50-to-100 — KuCoin Intra-Day Trading Bot

An aggressive intra-day scalping bot for [KuCoin](https://www.kucoin.com/) that
aims to compound a starting capital of **50 USDT** towards **10 000 USDT** in a
single trading day by riding high-momentum setups across multiple USDT pairs.

> ⚠️ **Risk warning** — Cryptocurrency trading involves a significant risk of
> capital loss.  This bot is provided for **educational purposes only**.  Never
> risk money you cannot afford to lose.  Past performance does not guarantee
> future results.

---

## Strategy

| Component | Detail |
|-----------|--------|
| Signal | RSI(14) recovery from oversold **+** EMA-9 crossing above EMA-21 |
| Time frame | 5-minute candles |
| Position sizing | 95 % of available balance spread across ≤ 3 concurrent positions |
| Stop-loss | −1.5 % from entry |
| Take-profit | +2.5 % from entry |
| Circuit-breaker | Auto-pauses after 5 consecutive losses (5-minute cool-down) |
| Pairs | BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX vs USDT (configurable) |

---

## Requirements

* Python 3.10+
* A KuCoin account with **API credentials** (API key, secret, passphrase)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/loureed691/50-to-100.git
cd 50-to-100

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export your KuCoin API credentials
export KUCOIN_API_KEY="your-api-key"
export KUCOIN_API_SECRET="your-api-secret"
export KUCOIN_API_PASSPHRASE="your-passphrase"

# 4. (Optional) Test against the KuCoin Sandbox first
export KUCOIN_SANDBOX=true

# 5. Start the bot
python bot.py
```

Alternatively, create a `.env` file in the project root (never commit it):

```
KUCOIN_API_KEY=your-api-key
KUCOIN_API_SECRET=your-api-secret
KUCOIN_API_PASSPHRASE=your-passphrase
KUCOIN_SANDBOX=false
```

---

## Configuration

All parameters can be tuned via environment variables or by editing `config.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `KUCOIN_API_KEY` | — | KuCoin API key (required) |
| `KUCOIN_API_SECRET` | — | KuCoin API secret (required) |
| `KUCOIN_API_PASSPHRASE` | — | KuCoin API passphrase (required) |
| `KUCOIN_SANDBOX` | `false` | Use sandbox environment |
| `PAPER_MODE` | `false` | Simulate orders without sending them to KuCoin (real market data, no real trades) |
| `INITIAL_CAPITAL` | `50.0` | Starting USDT amount |
| `TARGET_CAPITAL_USDT` | `10000.0` | Profit target (bot stops when reached) |
| `TRADE_FRACTION` | `0.95` | Fraction of balance used per trade batch |
| `STOP_LOSS_PCT` | `0.015` | Stop-loss as a fraction of entry price |
| `TAKE_PROFIT_PCT` | `0.025` | Take-profit as a fraction of entry price |
| `TRADING_PAIRS` | BTC,ETH,SOL,… | Comma-separated list of symbols |
| `KLINE_INTERVAL` | `5min` | Candle interval for signal generation |
| `POLL_INTERVAL_SECONDS` | `30` | Main loop frequency (seconds) |
| `MAX_OPEN_POSITIONS` | `3` | Maximum concurrent open positions |
| `EQUITY_FLOOR_USDT` | `10.0` | Emergency halt if equity drops below this |
| `MAX_CONSECUTIVE_LOSSES` | `5` | Trigger cool-down after N losses in a row |
| `COOLDOWN_SECONDS` | `300` | Cool-down duration in seconds |

---

## Project structure

```
50-to-100/
├── bot.py           # Main bot (strategy, order management, main loop)
├── config.py        # All configuration loaded from environment variables
├── requirements.txt # Python dependencies
├── tests/
│   └── test_bot.py  # Unit tests (no live API calls)
└── README.md
```

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Disclaimer

This software is provided "as is", without warranty of any kind.  The authors
are not responsible for any financial losses incurred through its use.
