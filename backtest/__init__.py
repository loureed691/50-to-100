"""
Backtest module for the KuCoin trading bot.

Provides a reproducible simulation engine that models fees, slippage, and
spread to evaluate strategy performance with realistic cost assumptions.

Usage
-----
    python -m backtest --pairs BTC-USDT,ETH-USDT --days 180 --seed 42 --capital 1000
"""
