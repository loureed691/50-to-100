"""
Unit tests for the KuCoin trading bot (indicator helpers and bot logic).
No live API calls are made; all KuCoin clients are mocked.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Stub out the kucoin package so tests run without the real SDK ──────────────
kucoin_mod = types.ModuleType("kucoin")
kucoin_client_mod = types.ModuleType("kucoin.client")
kucoin_client_mod.Market = MagicMock
kucoin_client_mod.Trade = MagicMock
kucoin_client_mod.User = MagicMock
kucoin_mod.client = kucoin_client_mod
sys.modules["kucoin"] = kucoin_mod
sys.modules["kucoin.client"] = kucoin_client_mod

import numpy as np
import pandas as pd

# Patch config credentials so the bot doesn't raise on import
with patch.dict(
    "os.environ",
    {
        "KUCOIN_API_KEY": "test-key",
        "KUCOIN_API_SECRET": "test-secret",
        "KUCOIN_API_PASSPHRASE": "test-pass",
    },
):
    from bot import (
        KuCoinBot,
        Position,
        _build_dataframe,
        _ema,
        _rsi,
    )
    import config  # noqa: F401 – needed to verify defaults


# ─────────────────────────────────────────────────────────────────────────────
# Helper / indicator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDataframe(unittest.TestCase):
    def _sample_klines(self, n: int = 5) -> list:
        """Generate minimal synthetic kline rows."""
        return [
            [str(1_700_000_000 + i * 300), "100", "101", "102", "99", "1000", "100100"]
            for i in range(n)
        ]

    def test_returns_dataframe(self):
        df = _build_dataframe(self._sample_klines())
        self.assertIsInstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = _build_dataframe(self._sample_klines())
        for col in ("timestamp", "open", "close", "high", "low", "volume"):
            self.assertIn(col, df.columns)

    def test_numeric_types(self):
        df = _build_dataframe(self._sample_klines())
        self.assertTrue(pd.api.types.is_numeric_dtype(df["close"]))

    def test_sorted_by_timestamp(self):
        klines = self._sample_klines(10)
        import random
        random.shuffle(klines)
        df = _build_dataframe(klines)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)


class TestEma(unittest.TestCase):
    def test_ema_length(self):
        s = pd.Series(range(1, 21), dtype=float)
        result = _ema(s, 9)
        self.assertEqual(len(result), len(s))

    def test_ema_single_value(self):
        s = pd.Series([5.0] * 20)
        result = _ema(s, 9)
        # EMA of a flat series should converge to that value
        self.assertAlmostEqual(result.iloc[-1], 5.0, places=4)


class TestRsi(unittest.TestCase):
    def test_rsi_range(self):
        prices = pd.Series([float(x) for x in range(50, 80)] + [float(x) for x in range(79, 49, -1)])
        rsi = _rsi(prices, 14)
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_rsi_length(self):
        prices = pd.Series(np.random.uniform(100, 200, 50))
        rsi = _rsi(prices, 14)
        self.assertEqual(len(rsi), len(prices))

    def test_rsi_100_on_monotonic_increase(self):
        # A strictly increasing series has no losses → RSI should be 100,
        # not NaN (regression guard for the zero-division fix).
        prices = pd.Series([float(x) for x in range(100, 140)])
        rsi = _rsi(prices, 14)
        valid = rsi.dropna()
        self.assertTrue((valid == 100.0).all())


# ─────────────────────────────────────────────────────────────────────────────
# Position tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPosition(unittest.TestCase):
    def _make_position(self, entry: float = 100.0, qty: float = 1.0) -> Position:
        return Position(
            symbol="BTC-USDT",
            order_id="abc123",
            entry_price=entry,
            quantity=qty,
            cost_usdt=entry * qty,
            stop_loss=entry * 0.985,
            take_profit=entry * 1.025,
        )

    def test_unrealised_pnl_profit(self):
        pos = self._make_position(100.0, 10.0)
        self.assertAlmostEqual(pos.unrealised_pnl(110.0), 100.0)

    def test_unrealised_pnl_loss(self):
        pos = self._make_position(100.0, 10.0)
        self.assertAlmostEqual(pos.unrealised_pnl(90.0), -100.0)

    def test_unrealised_pct(self):
        pos = self._make_position(200.0, 5.0)
        self.assertAlmostEqual(pos.unrealised_pct(210.0), 0.05)

    def test_unrealised_pct_zero_entry(self):
        pos = self._make_position(0.0, 1.0)
        self.assertEqual(pos.unrealised_pct(100.0), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Bot logic tests  (all API calls mocked)
# ─────────────────────────────────────────────────────────────────────────────

def _make_bot() -> KuCoinBot:
    """Create a KuCoinBot with all SDK clients replaced by mocks."""
    with patch.dict(
        "os.environ",
        {
            "KUCOIN_API_KEY": "k",
            "KUCOIN_API_SECRET": "s",
            "KUCOIN_API_PASSPHRASE": "p",
        },
    ):
        bot = KuCoinBot()
    bot.market_client = MagicMock()
    bot.trade_client = MagicMock()
    bot.user_client = MagicMock()
    return bot


class TestBotRoundQty(unittest.TestCase):
    def setUp(self):
        self.bot = _make_bot()

    def test_round_qty_standard(self):
        info = {"baseIncrement": "0.001"}
        qty = self.bot._round_qty(1.2345678, info)
        self.assertAlmostEqual(qty, 1.234, places=3)

    def test_round_qty_no_info(self):
        qty = self.bot._round_qty(0.123456789, {})
        # Should return a float with at most 8 decimal places
        self.assertIsInstance(qty, float)

    def test_round_qty_zero_increment(self):
        info = {"baseIncrement": "0"}
        qty = self.bot._round_qty(1.123456789, info)
        self.assertIsInstance(qty, float)


class TestBotUsdtBalance(unittest.TestCase):
    def setUp(self):
        self.bot = _make_bot()

    def test_returns_available_balance(self):
        self.bot.user_client.get_account_list.return_value = [
            {"currency": "USDT", "available": "123.45"}
        ]
        self.assertAlmostEqual(self.bot._usdt_balance(), 123.45)

    def test_returns_zero_on_error(self):
        self.bot.user_client.get_account_list.side_effect = Exception("network error")
        self.assertEqual(self.bot._usdt_balance(), 0.0)

    def test_returns_zero_if_no_usdt_account(self):
        self.bot.user_client.get_account_list.return_value = [
            {"currency": "BTC", "available": "0.5"}
        ]
        self.assertEqual(self.bot._usdt_balance(), 0.0)


class TestBotBuySignal(unittest.TestCase):
    def setUp(self):
        self.bot = _make_bot()

    def _ind(self, rsi: float, ema_s: float, ema_l: float,
             ema_s_prev: float, ema_l_prev: float) -> dict:
        return {
            "close": 100.0,
            "rsi": rsi,
            "ema_short": ema_s,
            "ema_long": ema_l,
            "ema_short_prev": ema_s_prev,
            "ema_long_prev": ema_l_prev,
            "volume": 1000.0,
        }

    def test_valid_buy_signal(self):
        ind = self._ind(rsi=40, ema_s=101, ema_l=100, ema_s_prev=99, ema_l_prev=100)
        self.assertTrue(self.bot._is_buy_signal(ind))

    def test_no_signal_rsi_too_low(self):
        # RSI below oversold → condition not met (must be > RSI_OVERSOLD)
        ind = self._ind(rsi=20, ema_s=101, ema_l=100, ema_s_prev=99, ema_l_prev=100)
        self.assertFalse(self.bot._is_buy_signal(ind))

    def test_no_signal_rsi_at_boundary(self):
        # RSI exactly at oversold threshold → condition not met (strict >)
        ind = self._ind(
            rsi=config.RSI_OVERSOLD,
            ema_s=101, ema_l=100, ema_s_prev=99, ema_l_prev=100,
        )
        self.assertFalse(self.bot._is_buy_signal(ind))

    def test_no_signal_ema_no_cross(self):
        # EMA short was already above long → no fresh crossover
        ind = self._ind(rsi=40, ema_s=101, ema_l=100, ema_s_prev=101, ema_l_prev=100)
        self.assertFalse(self.bot._is_buy_signal(ind))


class TestBotCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.bot = _make_bot()

    def test_triggers_after_max_losses(self):
        import config as cfg
        self.bot.consecutive_losses = cfg.MAX_CONSECUTIVE_LOSSES
        with patch("time.sleep") as mock_sleep:
            triggered = self.bot._check_circuit_breaker()
        self.assertTrue(triggered)
        mock_sleep.assert_called_once_with(cfg.COOLDOWN_SECONDS)
        self.assertEqual(self.bot.consecutive_losses, 0)

    def test_does_not_trigger_below_max(self):
        self.bot.consecutive_losses = 0
        triggered = self.bot._check_circuit_breaker()
        self.assertFalse(triggered)


class TestBotPlaceMarketBuy(unittest.TestCase):
    def setUp(self):
        self.bot = _make_bot()

    def test_successful_buy_returns_position(self):
        self.bot.market_client.get_ticker.return_value = {"price": "200.0"}
        self.bot.market_client.get_symbol_list.return_value = [
            {
                "symbol": "ETH-USDT",
                "baseIncrement": "0.001",
                "baseMinSize": "0.001",
            }
        ]
        self.bot.trade_client.create_market_order.return_value = {"orderId": "order1"}

        pos = self.bot._place_market_buy("ETH-USDT", 100.0)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.symbol, "ETH-USDT")
        self.assertGreater(pos.quantity, 0)
        self.assertGreater(pos.stop_loss, 0)
        self.assertGreater(pos.take_profit, pos.entry_price)

    def test_returns_none_on_api_error(self):
        self.bot.market_client.get_ticker.side_effect = Exception("API down")
        pos = self.bot._place_market_buy("ETH-USDT", 100.0)
        self.assertIsNone(pos)

    def test_returns_none_on_zero_price(self):
        self.bot.market_client.get_ticker.return_value = {"price": "0"}
        self.bot.market_client.get_symbol_list.return_value = []
        pos = self.bot._place_market_buy("ETH-USDT", 100.0)
        self.assertIsNone(pos)


class TestBotManagePositions(unittest.TestCase):
    def setUp(self):
        self.bot = _make_bot()

    def _add_position(self, symbol: str, entry: float) -> Position:
        pos = Position(
            symbol=symbol,
            order_id="o1",
            entry_price=entry,
            quantity=1.0,
            cost_usdt=entry,
            stop_loss=entry * 0.985,
            take_profit=entry * 1.025,
        )
        self.bot.open_positions[symbol] = pos
        return pos

    def test_closes_on_stop_loss(self):
        pos = self._add_position("BTC-USDT", 40000.0)
        # Price dropped below stop-loss
        self.bot.market_client.get_ticker.return_value = {"price": str(pos.stop_loss - 1)}
        self.bot.trade_client.create_market_order.return_value = {"orderId": "o2"}
        self.bot._manage_open_positions()
        self.assertNotIn("BTC-USDT", self.bot.open_positions)

    def test_closes_on_take_profit(self):
        pos = self._add_position("BTC-USDT", 40000.0)
        # Price rose above take-profit
        self.bot.market_client.get_ticker.return_value = {"price": str(pos.take_profit + 1)}
        self.bot.trade_client.create_market_order.return_value = {"orderId": "o3"}
        self.bot._manage_open_positions()
        self.assertNotIn("BTC-USDT", self.bot.open_positions)

    def test_holds_within_range(self):
        pos = self._add_position("BTC-USDT", 40000.0)
        # Price within range
        self.bot.market_client.get_ticker.return_value = {"price": "40100.0"}
        self.bot._manage_open_positions()
        self.assertIn("BTC-USDT", self.bot.open_positions)


if __name__ == "__main__":
    unittest.main()
