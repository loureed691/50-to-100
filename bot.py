"""
KuCoin Trading Bot
==================
Aggressive intra-day scalping bot that targets 50 USDT → 10 000 USDT in one
trading day by compounding gains on high-momentum setups.

Strategy overview
-----------------
* Scans a configurable list of USDT pairs every POLL_INTERVAL_SECONDS seconds.
* Uses RSI(14) + dual-EMA crossover (EMA-9 / EMA-21) on 5-minute candles to
  detect high-probability long entries.
* Positions are sized as TRADE_FRACTION of current available balance.
* Each position is managed with a hard stop-loss and take-profit.
* A consecutive-loss circuit-breaker pauses the bot to avoid drawdown spirals.

WARNING
-------
Cryptocurrency trading involves significant risk of capital loss.  This bot is
provided for educational purposes.  Past performance does not guarantee future
results.  Never risk money you cannot afford to lose.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── Optional dotenv support ───────────────────────────────────────────────────
# Load .env *before* importing config so env vars are available when config
# reads them at module level.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config

# ── KuCoin SDK ────────────────────────────────────────────────────────────────
try:
    from kucoin.client import Market, Trade, User
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "kucoin-python is not installed.  Run: pip install kucoin-python"
    ) from exc

# ── Logging setup ─────────────────────────────────────────────────────────────
# Ensure stdout/stderr use UTF-8 so Unicode characters (e.g. →) in log messages
# are encoded correctly on platforms that default to a narrow encoding (e.g.
# Windows cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass  # best-effort; logging will still work, just may fall back to replacement chars
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataframe(klines: list) -> pd.DataFrame:
    """Convert raw KuCoin kline list into a typed DataFrame."""
    df = pd.DataFrame(
        klines,
        columns=["timestamp", "open", "close", "high", "low", "volume", "amount"],
    )
    for col in ("open", "close", "high", "low", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    # Compute RS; suppress divide-by-zero warnings — handled explicitly below.
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # When there are gains but no losses over the lookback window, standard
    # RSI is 100 (pure uptrend).  Avoid NaN propagating into signal logic.
    no_loss_mask = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~no_loss_mask, 100.0)
    return rsi


# ─────────────────────────────────────────────────────────────────────────────
# Position tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    order_id: str
    entry_price: float
    quantity: float
    cost_usdt: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stop_loss: float = 0.0
    take_profit: float = 0.0

    def unrealised_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.quantity

    def unrealised_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────────────────────

class KuCoinBot:
    """Intra-day scalping bot for KuCoin."""

    def __init__(self) -> None:
        self._validate_config()

        kw = dict(
            key=config.API_KEY,
            secret=config.API_SECRET,
            passphrase=config.API_PASSPHRASE,
        )
        if config.SANDBOX:
            kw["url"] = "https://openapi-sandbox.kucoin.com"

        self.market_client = Market(**kw)
        self.trade_client = Trade(**kw)
        self.user_client = User(**kw)

        self.open_positions: dict[str, Position] = {}  # symbol → Position
        self.consecutive_losses: int = 0
        self.session_start: datetime = datetime.now(timezone.utc)
        self.total_trades: int = 0
        self.winning_trades: int = 0

        log.info("Bot initialised.  Target: %.2f USDT → %.2f USDT",
                 config.INITIAL_CAPITAL_USDT, config.TARGET_CAPITAL_USDT)

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_config() -> None:
        missing = [
            name
            for name, val in (
                ("KUCOIN_API_KEY", config.API_KEY),
                ("KUCOIN_API_SECRET", config.API_SECRET),
                ("KUCOIN_API_PASSPHRASE", config.API_PASSPHRASE),
            )
            if not val
        ]
        if missing:
            raise EnvironmentError(
                "Missing required environment variables: " + ", ".join(missing)
            )

    # ── Account helpers ───────────────────────────────────────────────────────

    def _usdt_balance(self) -> float:
        """Return available USDT in the trading account."""
        try:
            accounts = self.user_client.get_account_list(currency="USDT", account_type="trade")
            for acct in accounts:
                if acct.get("currency") == "USDT":
                    return float(acct.get("available", 0))
        except Exception as exc:
            log.error("Failed to fetch USDT balance: %s", exc)
        return 0.0

    def _total_equity_usdt(self) -> float:
        """Approximate total equity: USDT balance + open position market value."""
        equity = self._usdt_balance()
        for pos in self.open_positions.values():
            try:
                ticker = self.market_client.get_ticker(pos.symbol)
                price = float(ticker.get("price", pos.entry_price))
                equity += price * pos.quantity
            except Exception:
                equity += pos.cost_usdt
        return equity

    # ── Symbol info ───────────────────────────────────────────────────────────

    def _symbol_info(self, symbol: str) -> dict:
        """Fetch trading rules (lot size, min size, etc.) for a symbol."""
        try:
            symbols = self.market_client.get_symbol_list()
            for s in symbols:
                if s.get("symbol") == symbol:
                    return s
        except Exception as exc:
            log.warning("Could not fetch symbol info for %s: %s", symbol, exc)
        return {}

    def _round_qty(self, qty: float, symbol_info: dict) -> float:
        """Round quantity to the exchange's base increment."""
        increment = float(symbol_info.get("baseIncrement", "0.00000001"))
        if increment <= 0:
            return round(qty, 8)
        precision = max(0, -int(round(np.log10(increment))))
        return round(float(int(qty / increment)) * increment, precision)

    # ── Indicators ────────────────────────────────────────────────────────────

    def _fetch_indicators(self, symbol: str) -> Optional[dict]:
        """
        Fetch recent candles and compute RSI + EMA crossover signals.
        Returns a dict with indicator values or None on error.
        """
        try:
            klines = self.market_client.get_kline(
                symbol,
                config.KLINE_INTERVAL,
                limit=config.KLINE_LIMIT,
            )
            if not klines or len(klines) < config.RSI_PERIOD + 5:
                return None

            df = _build_dataframe(klines)
            close = df["close"]

            rsi = _rsi(close, config.RSI_PERIOD)
            ema_s = _ema(close, config.EMA_SHORT)
            ema_l = _ema(close, config.EMA_LONG)

            last = -1  # most recent completed candle
            return {
                "close": close.iloc[last],
                "rsi": rsi.iloc[last],
                "ema_short": ema_s.iloc[last],
                "ema_long": ema_l.iloc[last],
                "ema_short_prev": ema_s.iloc[last - 1],
                "ema_long_prev": ema_l.iloc[last - 1],
                "volume": df["volume"].iloc[last],
            }
        except Exception as exc:
            log.warning("Indicator fetch error for %s: %s", symbol, exc)
            return None

    def _is_buy_signal(self, ind: dict) -> bool:
        """
        Entry conditions:
          1. RSI crossed above oversold threshold (momentum turning up).
          2. EMA-9 crossed above EMA-21 (trend confirmation).
        """
        rsi_above_oversold = (
            ind["rsi"] > config.RSI_OVERSOLD
        )
        ema_cross = (
            ind["ema_short"] > ind["ema_long"]
            and ind["ema_short_prev"] <= ind["ema_long_prev"]
        )
        return rsi_above_oversold and ema_cross

    # ── Order management ─────────────────────────────────────────────────────

    def _place_market_buy(
        self, symbol: str, usdt_amount: float
    ) -> Optional[Position]:
        """Place a market buy order and return a Position if successful."""
        try:
            ticker = self.market_client.get_ticker(symbol)
            price = float(ticker.get("price", 0))
            if price <= 0:
                log.warning("Invalid price for %s", symbol)
                return None

            info = self._symbol_info(symbol)
            raw_qty = usdt_amount / price
            qty = self._round_qty(raw_qty, info)

            min_size = float(info.get("baseMinSize", 0))
            if qty < min_size:
                log.debug("%s: qty %.8f below minSize %.8f — skipping", symbol, qty, min_size)
                return None

            order = self.trade_client.create_market_order(
                symbol=symbol,
                side="buy",
                size=str(qty),
            )
            order_id = order.get("orderId", "unknown")
            stop = price * (1 - config.STOP_LOSS_PCT)
            take = price * (1 + config.TAKE_PROFIT_PCT)

            pos = Position(
                symbol=symbol,
                order_id=order_id,
                entry_price=price,
                quantity=qty,
                cost_usdt=usdt_amount,
                stop_loss=stop,
                take_profit=take,
            )
            log.info(
                "BUY  %-15s  qty=%.6f  price=%.4f  SL=%.4f  TP=%.4f",
                symbol, qty, price, stop, take,
            )
            return pos
        except Exception as exc:
            log.error("Buy order failed for %s: %s", symbol, exc)
            return None

    def _place_market_sell(self, pos: Position, reason: str = "signal") -> bool:
        """Close an open position with a market sell order."""
        try:
            order = self.trade_client.create_market_order(
                symbol=pos.symbol,
                side="sell",
                size=str(pos.quantity),
            )
            ticker = self.market_client.get_ticker(pos.symbol)
            exit_price = float(ticker.get("price", pos.entry_price))
            pnl = pos.unrealised_pnl(exit_price)
            pct = pos.unrealised_pct(exit_price) * 100

            log.info(
                "SELL %-15s  reason=%-8s  pnl=%+.4f USDT (%+.2f%%)",
                pos.symbol, reason, pnl, pct,
            )
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            return True
        except Exception as exc:
            log.error("Sell order failed for %s: %s", pos.symbol, exc)
            return False

    # ── Position management loop ─────────────────────────────────────────────

    def _manage_open_positions(self) -> None:
        """Check each open position against stop-loss and take-profit."""
        to_close: list[str] = []
        for symbol, pos in self.open_positions.items():
            try:
                ticker = self.market_client.get_ticker(symbol)
                price = float(ticker.get("price", pos.entry_price))
            except Exception:
                continue

            if price <= pos.stop_loss:
                if self._place_market_sell(pos, reason="stop-loss"):
                    to_close.append(symbol)
            elif price >= pos.take_profit:
                if self._place_market_sell(pos, reason="take-profit"):
                    to_close.append(symbol)

        for symbol in to_close:
            del self.open_positions[symbol]

    # ── Entry scanning ────────────────────────────────────────────────────────

    def _scan_for_entries(self) -> None:
        """Scan configured pairs for buy signals and open positions."""
        available_slots = config.MAX_OPEN_POSITIONS - len(self.open_positions)
        if available_slots <= 0:
            return

        balance = self._usdt_balance()
        if balance < config.MIN_TRADE_BALANCE_USDT:
            log.debug("Insufficient USDT balance (%.4f) — skipping entry scan", balance)
            return

        usdt_per_trade = (balance * config.TRADE_FRACTION) / available_slots

        for symbol in config.TRADING_PAIRS:
            if symbol in self.open_positions:
                continue
            if len(self.open_positions) >= config.MAX_OPEN_POSITIONS:
                break

            ind = self._fetch_indicators(symbol)
            if ind is None:
                continue

            log.debug(
                "%-15s  RSI=%.1f  EMA_s=%.4f  EMA_l=%.4f",
                symbol, ind["rsi"], ind["ema_short"], ind["ema_long"],
            )

            if self._is_buy_signal(ind):
                pos = self._place_market_buy(symbol, usdt_per_trade)
                if pos:
                    self.open_positions[symbol] = pos

    # ── Circuit breaker ───────────────────────────────────────────────────────

    def _check_circuit_breaker(self) -> bool:
        """Return True (and sleep) if the consecutive-loss limit is hit."""
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            log.warning(
                "Circuit breaker triggered (%d consecutive losses).  "
                "Cooling down for %d seconds.",
                self.consecutive_losses,
                config.COOLDOWN_SECONDS,
            )
            time.sleep(config.COOLDOWN_SECONDS)
            self.consecutive_losses = 0
            return True
        return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _log_stats(self, equity: float) -> None:
        elapsed = (datetime.now(timezone.utc) - self.session_start).total_seconds() / 3600
        win_rate = (
            self.winning_trades / self.total_trades * 100 if self.total_trades else 0
        )
        log.info(
            "── Stats ──  equity=%.2f USDT  trades=%d  win_rate=%.1f%%  "
            "elapsed=%.1fh  target=%.2f USDT",
            equity, self.total_trades, win_rate, elapsed, config.TARGET_CAPITAL_USDT,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the trading loop."""
        log.info("Starting trading session at %s", self.session_start.isoformat())

        while True:
            try:
                equity = self._total_equity_usdt()
                self._log_stats(equity)

                # ── Goal reached ──────────────────────────────────────────────
                if equity >= config.TARGET_CAPITAL_USDT:
                    log.info(
                        "🎯  TARGET REACHED!  Equity=%.2f USDT — stopping bot.",
                        equity,
                    )
                    self._close_all_positions("target-reached")
                    break

                # ── Equity floor ─────────────────────────────────────────────
                if equity < config.EQUITY_FLOOR_USDT:
                    log.error(
                        "❌  Equity (%.2f USDT) below floor (%.2f USDT) — halting.",
                        equity, config.EQUITY_FLOOR_USDT,
                    )
                    self._close_all_positions("equity-floor")
                    break

                # ── Circuit breaker ───────────────────────────────────────────
                self._check_circuit_breaker()

                # ── Manage open positions ─────────────────────────────────────
                self._manage_open_positions()

                # ── Look for new entries ──────────────────────────────────────
                self._scan_for_entries()

            except KeyboardInterrupt:
                log.info("Keyboard interrupt — closing all positions and exiting.")
                self._close_all_positions("manual-exit")
                break
            except Exception as exc:
                log.exception("Unexpected error in main loop: %s", exc)

            time.sleep(config.POLL_INTERVAL_SECONDS)

        self._log_stats(self._total_equity_usdt())
        log.info("Session ended.")

    def _close_all_positions(self, reason: str) -> None:
        """Market-sell every open position."""
        for symbol, pos in list(self.open_positions.items()):
            if self._place_market_sell(pos, reason=reason):
                del self.open_positions[symbol]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = KuCoinBot()
    bot.run()
