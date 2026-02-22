import importlib
import os
import unittest
from unittest.mock import patch

import config


class TestTradingPairsParsing(unittest.TestCase):
    def tearDown(self):
        importlib.reload(config)

    def test_trading_pairs_strips_whitespace_and_drops_empty_values(self):
        with patch.dict(
            os.environ,
            {"TRADING_PAIRS": " BTC-USDT, ,ETH-USDT,, SOL-USDT  "},
            clear=False,
        ):
            importlib.reload(config)
            self.assertEqual(config.TRADING_PAIRS, ["BTC-USDT", "ETH-USDT", "SOL-USDT"])
