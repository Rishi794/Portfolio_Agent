from __future__ import annotations

import pandas as pd

from src.storage.lakehouse import Lakehouse
from src.storage.price_integrity import clean_ohlc_rows, repair_invalid_market_prices
from src.storage.verify_parquet import verify_required_tables


def test_invalid_ohlc_bars_are_dropped_without_fabricating_prices():
    frame = pd.DataFrame(
        [
            {"ticker": "GOOD", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5},
            {"ticker": "ZERO_OPEN", "open": 0.0, "high": 11.0, "low": 9.0, "close": 10.5},
            {"ticker": "MISSING_OPEN", "open": None, "high": 11.0, "low": 9.0, "close": 10.5},
        ]
    )

    cleaned, removed = clean_ohlc_rows(frame)

    assert removed == 2
    assert cleaned["ticker"].tolist() == ["GOOD"]
    assert cleaned.loc[0, "open"] == 10.0


def test_repair_removes_legacy_invalid_rows_from_persisted_lakehouse(tmp_path):
    lake = Lakehouse(tmp_path / "lake")
    lake.write(
        "market_prices",
        pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "market": "US",
                    "date": f"2026-01-0{day}",
                    "open": open_price,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 100.0,
                    "adj_close": 10.5,
                    "dollar_volume": 1050.0,
                    "knowledge_timestamp": pd.Timestamp(f"2026-01-0{day}", tz="UTC"),
                }
                for ticker, day, open_price in (("GOOD", 1, 10.0), ("BAD", 2, 0.0))
            ]
        ),
    )

    assert repair_invalid_market_prices(lake) == 1
    stored = lake.read("market_prices")
    assert stored["ticker"].tolist() == ["GOOD"]
    assert "raw_ohlcv: invalid OHLC" not in " ".join(verify_required_tables(lake)["issues"])
