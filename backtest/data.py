"""
Data utilities for backtesting.

Provides deterministic synthetic OHLCV generation for reproducible backtests
and helpers to load CSV data when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlcv(
    symbol: str,
    days: int = 180,
    interval_minutes: int = 5,
    seed: int = 42,
    base_price: float = 100.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for backtesting.

    Parameters
    ----------
    symbol : str
        Pair name (stored as metadata, not used for generation).
    days : int
        Number of calendar days to generate.
    interval_minutes : int
        Candle interval in minutes.
    seed : int
        Random seed for reproducibility.
    base_price : float
        Starting price of the asset.
    volatility : float
        Per-candle return standard deviation.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, open, high, low, close, volume
    """
    rng = np.random.RandomState(seed)
    n_candles = (days * 24 * 60) // interval_minutes

    # Generate log-returns with mean-reverting drift
    returns = rng.normal(0, volatility, n_candles)
    # Add slight trending regime shifts
    regime = np.sin(np.linspace(0, 4 * np.pi, n_candles)) * volatility * 0.3
    returns += regime

    prices = base_price * np.exp(np.cumsum(returns))

    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_candles,
        freq=f"{interval_minutes}min",
    )

    # Build OHLCV
    high_noise = np.abs(rng.normal(0, volatility * 0.5, n_candles))
    low_noise = np.abs(rng.normal(0, volatility * 0.5, n_candles))

    opens = np.roll(prices, 1)
    opens[0] = base_price
    highs = np.maximum(prices, opens) * (1 + high_noise)
    lows = np.minimum(prices, opens) * (1 - low_noise)
    volumes = rng.uniform(1000, 50000, n_candles)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })
    df.attrs["symbol"] = symbol
    return df


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Expected columns: timestamp, open, high, low, close, volume
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
