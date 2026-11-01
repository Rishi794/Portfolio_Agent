from __future__ import annotations

import datetime as dt

import pandas as pd

from src.main_v4 import V4AutonomousPortfolioReducer


def test_daily_as_of_includes_same_day_close_but_excludes_next_trade_date(config, tmp_path) -> None:
    reducer = V4AutonomousPortfolioReducer(
        config,
        tmp_path / "lake",
        tmp_path / "state",
        as_of=dt.date(2026, 10, 30),
        offline=True,
    )
    reducer.lake.write(
        "market_prices",
        pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "market": "US",
                    "price_source": "YAHOO_FINANCE",
                    "date": dt.date(2026, 10, 30),
                    "open": 99.0,
                    "high": 102.0,
                    "low": 98.0,
                    "close": 101.0,
                    "volume": 1000.0,
                    "adj_close": 101.0,
                    "dollar_volume": 101000.0,
                    "knowledge_timestamp": pd.Timestamp("2026-10-30 22:00:00", tz="UTC"),
                },
                {
                    "ticker": "AAA",
                    "market": "US",
                    "price_source": "YAHOO_FINANCE",
                    "date": dt.date(2026, 10, 31),
                    "open": 102.0,
                    "high": 103.0,
                    "low": 100.0,
                    "close": 102.0,
                    "volume": 1000.0,
                    "adj_close": 102.0,
                    "dollar_volume": 102000.0,
                    "knowledge_timestamp": pd.Timestamp("2026-10-31 22:00:00", tz="UTC"),
                },
            ]
        ),
    )

    prices = reducer.price_snapshot("US")

    assert len(prices) == 1
    assert prices.iloc[0]["date"] == pd.Timestamp("2026-10-30")
    assert prices.iloc[0]["close"] == 101.0
