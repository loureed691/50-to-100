"""
Configuration for KuCoin Trading Bot.

Set your API credentials via environment variables:
  KUCOIN_API_KEY      - Your KuCoin API key
  KUCOIN_API_SECRET   - Your KuCoin API secret
  KUCOIN_API_PASSPHRASE - Your KuCoin API passphrase

Or edit the values directly here (NOT recommended for production).
"""

import os

# ── KuCoin API Credentials ────────────────────────────────────────────────────
API_KEY = os.getenv("KUCOIN_API_KEY", "")
API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")

# ── Capital & Risk Settings ───────────────────────────────────────────────────
# Starting capital in USDT
INITIAL_CAPITAL_USDT = float(os.getenv("INITIAL_CAPITAL", "50.0"))

# Target profit in USDT (goal for the day)
TARGET_CAPITAL_USDT = float(os.getenv("TARGET_CAPITAL_USDT", "10000.0"))

# Maximum fraction of current capital to risk per trade (e.g. 0.95 = 95 %)
TRADE_FRACTION = float(os.getenv("TRADE_FRACTION", "0.95"))

# Stop-loss: exit a position when unrealised loss exceeds this fraction
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))   # 1.5 %

# Take-profit: close a position when unrealised gain exceeds this fraction
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.025"))  # 2.5 %

# ── Trading Pairs ─────────────────────────────────────────────────────────────
# List of symbols to scan for opportunities (KuCoin format: BASE-QUOTE)
TRADING_PAIRS = os.getenv(
    "TRADING_PAIRS",
    "BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT,XRP-USDT,DOGE-USDT,ADA-USDT,AVAX-USDT,"
    "DOT-USDT,POL-USDT,LINK-USDT,LTC-USDT,UNI-USDT,ATOM-USDT,TRX-USDT,"
    "NEAR-USDT,APT-USDT,OP-USDT,ARB-USDT,SUI-USDT,TON-USDT,SHIB-USDT,"
    "BCH-USDT,FIL-USDT,INJ-USDT",
).split(",")

# ── Strategy Parameters ───────────────────────────────────────────────────────
# RSI period
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

# RSI thresholds
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "35"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "65"))

# EMA short / long periods used for trend confirmation
EMA_SHORT = int(os.getenv("EMA_SHORT", "9"))
EMA_LONG = int(os.getenv("EMA_LONG", "21"))

# Minimum signal confidence (0-1) to open a position (0 = disabled)
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.0"))

# Maximum portfolio heat: total capital fraction at risk across all open positions
MAX_PORTFOLIO_HEAT = float(os.getenv("MAX_PORTFOLIO_HEAT", "0.06"))

# Maximum acceptable slippage as a fraction of entry price
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.002"))

# Prefer limit orders over market orders when supported
USE_LIMIT_ORDERS = os.getenv("USE_LIMIT_ORDERS", "false").lower() in ("true", "1", "yes")

# Candlestick interval to trade on (KuCoin: 1min, 3min, 5min, 15min, …)
KLINE_INTERVAL = os.getenv("KLINE_INTERVAL", "5min")

# Number of candles to fetch for indicator calculation
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "100"))

# How often (seconds) the bot checks for new signals
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))

# Maximum number of concurrent open positions
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

# Minimum available USDT balance required to open any new position
MIN_TRADE_BALANCE_USDT = float(os.getenv("MIN_TRADE_BALANCE_USDT", "1.0"))

# ── Safety Limits ─────────────────────────────────────────────────────────────
# If total equity drops below this USDT value, halt all trading
EQUITY_FLOOR_USDT = float(os.getenv("EQUITY_FLOOR_USDT", "10.0"))

# Maximum number of consecutive losing trades before pausing (cool-down)
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))

# Cool-down period (seconds) after hitting MAX_CONSECUTIVE_LOSSES
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "300"))

# ── Paper Trading ─────────────────────────────────────────────────────────────
# When enabled, the bot uses real market data but simulates order execution
# without placing any real orders.  API credentials are optional in this mode.
PAPER_MODE = os.getenv("PAPER_MODE", "false").lower() in ("true", "1", "yes")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")
