"""
Integration tests with mocked exchange responses.

Simulates full bot cycles with controlled market data to verify end-to-end
behaviour without requiring a live exchange connection.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Stub out kucoin SDK ───────────────────────────────────────────────────────
kucoin_mod = types.ModuleType("kucoin")
kucoin_client_mod = types.ModuleType("kucoin.client")
kucoin_client_mod.Market = MagicMock
kucoin_client_mod.Trade = MagicMock
kucoin_client_mod.User = MagicMock
kucoin_mod.client = kucoin_client_mod
sys.modules.setdefault("kucoin", kucoin_mod)
sys.modules.setdefault("kucoin.client", kucoin_client_mod)

with patch.dict("os.environ", {
    "KUCOIN_API_KEY": "test-key",
    "KUCOIN_API_SECRET": "test-secret",
    "KUCOIN_API_PASSPHRASE": "test-pass",
}):
    from bot import KuCoinBot, Position  # noqa: E402


def _make_paper_bot() -> KuCoinBot:
    """Create a paper-mode bot with mocked SDK clients."""
    import config as cfg
    with patch.object(cfg, "PAPER_MODE", True), \
         patch.object(cfg, "API_KEY", ""), \
         patch.object(cfg, "API_SECRET", ""), \
         patch.object(cfg, "API_PASSPHRASE", ""):
        bot = KuCoinBot()
    bot.market_client = MagicMock()
    return bot


class TestFullBuySellCycle(unittest.TestCase):
    """Simulate a complete buy → hold → sell cycle in paper mode."""

    def test_buy_then_take_profit(self):
        import config as cfg
        bot = _make_paper_bot()
        bot.paper_balance = 100.0

        # Setup mocks for a successful buy
        bot.market_client.get_ticker.return_value = {"price": "50.0"}
        bot.market_client.get_symbol_list.return_value = [{
            "symbol": "TEST-USDT",
            "baseIncrement": "0.001",
            "baseMinSize": "0.001",
        }]

        with patch.object(cfg, "PAPER_MODE", True):
            pos = bot._place_market_buy("TEST-USDT", 50.0)
        self.assertIsNotNone(pos)
        bot.open_positions["TEST-USDT"] = pos

        # Price rises above take-profit
        tp_price = pos.take_profit + 1
        bot.market_client.get_ticker.return_value = {"price": str(tp_price)}

        with patch.object(cfg, "PAPER_MODE", True):
            bot._manage_open_positions()

        self.assertNotIn("TEST-USDT", bot.open_positions)
        self.assertEqual(bot.winning_trades, 1)
        self.assertEqual(bot.consecutive_losses, 0)

    def test_buy_then_stop_loss(self):
        import config as cfg
        bot = _make_paper_bot()
        bot.paper_balance = 100.0

        bot.market_client.get_ticker.return_value = {"price": "50.0"}
        bot.market_client.get_symbol_list.return_value = [{
            "symbol": "TEST-USDT",
            "baseIncrement": "0.001",
            "baseMinSize": "0.001",
        }]

        with patch.object(cfg, "PAPER_MODE", True):
            pos = bot._place_market_buy("TEST-USDT", 50.0)
        self.assertIsNotNone(pos)
        bot.open_positions["TEST-USDT"] = pos

        # Price drops below stop-loss
        sl_price = pos.stop_loss - 1
        bot.market_client.get_ticker.return_value = {"price": str(sl_price)}

        with patch.object(cfg, "PAPER_MODE", True):
            bot._manage_open_positions()

        self.assertNotIn("TEST-USDT", bot.open_positions)
        self.assertEqual(bot.consecutive_losses, 1)

    def test_circuit_breaker_resets_after_cooldown(self):
        import config as cfg
        bot = _make_paper_bot()
        bot.consecutive_losses = cfg.MAX_CONSECUTIVE_LOSSES

        with patch("time.sleep"):
            triggered = bot._check_circuit_breaker()

        self.assertTrue(triggered)
        self.assertEqual(bot.consecutive_losses, 0)


class TestMultiplePositions(unittest.TestCase):
    """Test managing multiple concurrent positions."""

    def test_max_positions_respected(self):
        import config as cfg
        bot = _make_paper_bot()

        # Fill up to max positions
        for i in range(cfg.MAX_OPEN_POSITIONS):
            sym = f"TEST{i}-USDT"
            bot.open_positions[sym] = Position(
                symbol=sym,
                order_id=f"o{i}",
                entry_price=100.0,
                quantity=0.1,
                cost_usdt=10.0,
                stop_loss=98.5,
                take_profit=102.5,
            )

        # Scan should not add more
        with patch.object(cfg, "TRADING_PAIRS", ["NEW-USDT"]), \
             patch.object(cfg, "MAX_OPEN_POSITIONS", cfg.MAX_OPEN_POSITIONS):
            bot._usdt_balance = MagicMock(return_value=100.0)
            bot._fetch_indicators = MagicMock(return_value={
                "rsi": 40, "ema_short": 2, "ema_long": 1,
                "ema_short_prev": 0.9, "ema_long_prev": 1.0, "volume": 1000,
            })
            bot._is_buy_signal = MagicMock(return_value=True)
            bot._scan_for_entries()

        self.assertEqual(len(bot.open_positions), cfg.MAX_OPEN_POSITIONS)

    def test_close_all_positions(self):
        import config as cfg
        bot = _make_paper_bot()
        bot.paper_balance = 50.0

        for i in range(3):
            sym = f"TEST{i}-USDT"
            bot.open_positions[sym] = Position(
                symbol=sym,
                order_id=f"o{i}",
                entry_price=100.0,
                quantity=0.1,
                cost_usdt=10.0,
                stop_loss=98.5,
                take_profit=102.5,
            )

        bot.market_client.get_ticker.return_value = {"price": "101.0"}

        with patch.object(cfg, "PAPER_MODE", True):
            bot._close_all_positions("test-close")

        self.assertEqual(len(bot.open_positions), 0)


class TestMockedExchangeResponses(unittest.TestCase):
    """Test bot resilience to various exchange response formats."""

    def test_handles_missing_price_field(self):
        bot = _make_paper_bot()
        bot.market_client.get_ticker.return_value = {}
        bot.market_client.get_symbol_list.return_value = []
        pos = bot._place_market_buy("TEST-USDT", 10.0)
        self.assertIsNone(pos)

    def test_handles_empty_klines(self):
        bot = _make_paper_bot()
        bot.market_client.get_kline.return_value = []
        result = bot._fetch_indicators("TEST-USDT")
        self.assertIsNone(result)

    def test_handles_symbol_info_error(self):
        bot = _make_paper_bot()
        bot.market_client.get_symbol_list.side_effect = Exception("timeout")
        info = bot._symbol_info("TEST-USDT")
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
