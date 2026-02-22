"""
Backtest engine — simulates the trading strategy on historical OHLCV data
with realistic fee, slippage, and spread modelling.

Supports both the *baseline* strategy (RSI + EMA crossover) and the
*improved* strategy (with trend filter, ATR sizing, trailing stops, and
minimum edge threshold).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CostModel:
    """Transaction cost assumptions."""
    maker_fee: float = 0.001    # 0.1%
    taker_fee: float = 0.001    # 0.1%
    slippage: float = 0.0005    # 0.05% market impact
    spread: float = 0.0003      # 0.03% half-spread

    def buy_cost_factor(self) -> float:
        """Multiplier on price when buying (pay more)."""
        return 1 + self.taker_fee + self.slippage + self.spread

    def sell_cost_factor(self) -> float:
        """Multiplier on price when selling (receive less)."""
        return 1 - self.taker_fee - self.slippage - self.spread


# ─────────────────────────────────────────────────────────────────────────────
# Strategy parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyParams:
    """Tunable strategy parameters for both baseline and improved modes."""
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    # EMA crossover
    ema_short: int = 9
    ema_long: int = 21
    # Stop / take-profit (fractions)
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.025
    # Position sizing
    trade_fraction: float = 0.95
    max_open_positions: int = 3
    # ── Improved strategy additions ──
    use_trend_filter: bool = False
    trend_ema_period: int = 50       # HTF trend EMA
    use_atr_sizing: bool = False
    atr_period: int = 14
    risk_per_trade: float = 0.01     # 1% account risk when using ATR sizing
    use_trailing_stop: bool = False
    trailing_atr_mult: float = 2.0   # move stop to breakeven after 2×ATR move
    use_min_edge: bool = False
    min_edge_mult: float = 1.5       # required edge = costs × this multiplier
    use_cooldown: bool = False
    cooldown_bars: int = 6           # bars to wait after consecutive losses
    max_consecutive_losses: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Internal position tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Position:
    symbol: str
    entry_price: float
    effective_entry: float  # price after costs
    quantity: float
    cost_usdt: float
    stop_loss: float
    take_profit: float
    bar_index: int = 0
    trailing_activated: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Indicator computation
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    no_loss_mask = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~no_loss_mask, 100.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Add all indicator columns to the dataframe."""
    close = df["close"]
    df = df.copy()
    df["rsi"] = _rsi(close, params.rsi_period)
    df["ema_short"] = _ema(close, params.ema_short)
    df["ema_long"] = _ema(close, params.ema_long)
    df["ema_short_prev"] = df["ema_short"].shift(1)
    df["ema_long_prev"] = df["ema_long"].shift(1)
    if params.use_trend_filter:
        df["trend_ema"] = _ema(close, params.trend_ema_period)
    if params.use_atr_sizing or params.use_trailing_stop:
        df["atr"] = _atr(df["high"], df["low"], close, params.atr_period)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for backtest output."""
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trade_pnls: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    total_fees: float = 0.0


def run_backtest(
    data: dict[str, pd.DataFrame],
    params: StrategyParams,
    cost: CostModel,
    initial_capital: float = 50.0,
) -> BacktestResult:
    """Run a full backtest across one or more symbols.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Symbol → OHLCV DataFrame.
    params : StrategyParams
        Strategy configuration.
    cost : CostModel
        Fee/slippage assumptions.
    initial_capital : float
        Starting portfolio value in USDT.

    Returns
    -------
    BacktestResult
    """
    # Pre-compute indicators for each symbol
    indicator_data: dict[str, pd.DataFrame] = {}
    min_len = None
    for sym, df in data.items():
        ind_df = compute_indicators(df, params)
        indicator_data[sym] = ind_df
        if min_len is None or len(ind_df) < min_len:
            min_len = len(ind_df)

    if min_len is None or min_len < 2:
        return BacktestResult()

    capital = initial_capital
    positions: dict[str, _Position] = {}
    equity_values: list[float] = []
    trade_pnls: list[float] = []
    trade_log: list[dict] = []
    total_fees: float = 0.0
    consecutive_losses = 0
    cooldown_remaining = 0

    # Iterate bar by bar (use closed candles only → signal on bar i, execute on bar i+1)
    for i in range(1, min_len):
        # ── Manage open positions ─────────────────────────────────────
        to_close: list[str] = []
        for sym, pos in positions.items():
            idf = indicator_data[sym]
            current_high = idf["high"].iloc[i]
            current_low = idf["low"].iloc[i]
            current_close = idf["close"].iloc[i]

            # Trailing stop: tighten stop to breakeven after favorable move
            if params.use_trailing_stop and "atr" in idf.columns and not pos.trailing_activated:
                atr_val = idf["atr"].iloc[i - 1]
                if atr_val > 0:
                    move = current_close - pos.entry_price
                    if move >= params.trailing_atr_mult * atr_val:
                        pos.stop_loss = max(pos.stop_loss, pos.entry_price)
                        pos.trailing_activated = True

            # Check stop-loss
            if current_low <= pos.stop_loss:
                exit_price = pos.stop_loss * cost.sell_cost_factor()
                pnl = (exit_price - pos.effective_entry) * pos.quantity
                fee = abs(exit_price * pos.quantity) * (cost.taker_fee + cost.slippage)
                total_fees += fee
                capital += pos.cost_usdt + pnl
                trade_pnls.append(pnl)
                trade_log.append({
                    "symbol": sym, "entry": pos.entry_price, "exit": pos.stop_loss,
                    "pnl": pnl, "reason": "stop-loss", "bar": i,
                })
                to_close.append(sym)
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                continue

            # Check take-profit
            if current_high >= pos.take_profit:
                exit_price = pos.take_profit * cost.sell_cost_factor()
                pnl = (exit_price - pos.effective_entry) * pos.quantity
                fee = abs(exit_price * pos.quantity) * (cost.taker_fee + cost.slippage)
                total_fees += fee
                capital += pos.cost_usdt + pnl
                trade_pnls.append(pnl)
                trade_log.append({
                    "symbol": sym, "entry": pos.entry_price, "exit": pos.take_profit,
                    "pnl": pnl, "reason": "take-profit", "bar": i,
                })
                to_close.append(sym)
                consecutive_losses = 0

        for sym in to_close:
            del positions[sym]

        # ── Cooldown logic ────────────────────────────────────────────
        if params.use_cooldown and cooldown_remaining > 0:
            cooldown_remaining -= 1
            # Record equity even during cooldown
            pos_value = sum(
                indicator_data[s]["close"].iloc[i] * p.quantity
                for s, p in positions.items()
            )
            equity_values.append(capital + pos_value)
            continue

        if params.use_cooldown and consecutive_losses >= params.max_consecutive_losses:
            cooldown_remaining = params.cooldown_bars
            consecutive_losses = 0

        # ── Scan for entries (signals on bar i-1, execute on bar i) ───
        available_slots = params.max_open_positions - len(positions)
        if available_slots > 0 and capital > 1.0:
            candidates: list[str] = []
            for sym in indicator_data:
                if sym in positions:
                    continue
                idf = indicator_data[sym]
                prev = i - 1  # signal bar (closed candle)

                rsi_val = idf["rsi"].iloc[prev]
                ema_s = idf["ema_short"].iloc[prev]
                ema_l = idf["ema_long"].iloc[prev]
                ema_s_prev = idf["ema_short_prev"].iloc[prev]
                ema_l_prev = idf["ema_long_prev"].iloc[prev]

                if pd.isna(rsi_val) or pd.isna(ema_s) or pd.isna(ema_l):
                    continue
                if pd.isna(ema_s_prev) or pd.isna(ema_l_prev):
                    continue

                # Core signal: RSI above oversold + EMA crossover
                rsi_ok = rsi_val > params.rsi_oversold
                ema_cross = ema_s > ema_l and ema_s_prev <= ema_l_prev
                if not (rsi_ok and ema_cross):
                    continue

                # Trend filter: price must be above long-term EMA
                if params.use_trend_filter and "trend_ema" in idf.columns:
                    trend_val = idf["trend_ema"].iloc[prev]
                    if pd.notna(trend_val) and idf["close"].iloc[prev] < trend_val:
                        continue

                # Minimum edge filter: expected move must exceed costs
                if params.use_min_edge:
                    round_trip_cost = cost.buy_cost_factor() - 1 + 1 - cost.sell_cost_factor()
                    if params.take_profit_pct < round_trip_cost * params.min_edge_mult:
                        continue

                candidates.append(sym)

            trade_count = min(available_slots, len(candidates))
            if trade_count > 0:
                usdt_per_trade = (capital * params.trade_fraction) / trade_count
                for sym in candidates[:trade_count]:
                    idf = indicator_data[sym]
                    entry_price = idf["open"].iloc[i]  # execute at next bar open
                    effective_entry = entry_price * cost.buy_cost_factor()

                    # ATR-based sizing
                    if params.use_atr_sizing and "atr" in idf.columns:
                        atr_val = idf["atr"].iloc[i - 1]
                        if atr_val > 0:
                            stop_dist = params.stop_loss_pct * entry_price
                            risk_amount = capital * params.risk_per_trade
                            qty_by_risk = risk_amount / stop_dist
                            qty_by_balance = usdt_per_trade / effective_entry
                            qty = min(qty_by_risk, qty_by_balance)
                        else:
                            qty = usdt_per_trade / effective_entry
                    else:
                        qty = usdt_per_trade / effective_entry

                    actual_cost = qty * effective_entry
                    if actual_cost > capital:
                        continue

                    fee = actual_cost * (cost.taker_fee + cost.slippage)
                    total_fees += fee
                    capital -= actual_cost

                    stop = entry_price * (1 - params.stop_loss_pct)
                    take = entry_price * (1 + params.take_profit_pct)

                    positions[sym] = _Position(
                        symbol=sym,
                        entry_price=entry_price,
                        effective_entry=effective_entry,
                        quantity=qty,
                        cost_usdt=actual_cost,
                        stop_loss=stop,
                        take_profit=take,
                        bar_index=i,
                    )

        # ── Record equity ─────────────────────────────────────────────
        pos_value = sum(
            indicator_data[s]["close"].iloc[i] * p.quantity
            for s, p in positions.items()
        )
        equity_values.append(capital + pos_value)

    # ── Close remaining positions at last bar close ───────────────────
    if positions:
        last_bar = min_len - 1
        for sym, pos in list(positions.items()):
            exit_price = indicator_data[sym]["close"].iloc[last_bar] * cost.sell_cost_factor()
            pnl = (exit_price - pos.effective_entry) * pos.quantity
            capital += pos.cost_usdt + pnl
            trade_pnls.append(pnl)
            trade_log.append({
                "symbol": sym, "entry": pos.entry_price,
                "exit": indicator_data[sym]["close"].iloc[last_bar],
                "pnl": pnl, "reason": "end-of-data", "bar": last_bar,
            })
        positions.clear()

    equity_series = pd.Series(equity_values, dtype=float)
    daily_rets = equity_series.pct_change().dropna()

    return BacktestResult(
        equity_curve=equity_series,
        daily_returns=daily_rets,
        trade_pnls=trade_pnls,
        trade_log=trade_log,
        total_fees=total_fees,
    )
