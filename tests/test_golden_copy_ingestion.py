from __future__ import annotations

import datetime as dt

import pandas as pd

from src.ingestion.golden_copy_ingestor import MultiSourceGoldenCopyIngestor


def test_yahoo_only_us_bar_does_not_fake_a_quorum(config, tmp_path) -> None:
    ingestor = MultiSourceGoldenCopyIngestor(config, "US", offline=True, cache_dir=tmp_path)
    day = dt.date(2026, 10, 30)
    ingestor._yahoo_frame = lambda _ticker: pd.DataFrame(  # type: ignore[method-assign]
        [
            {
                "date": day,
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        ]
    )
    ingestor._tiingo_close = lambda _ticker, _day: None  # type: ignore[method-assign]
    ingestor._stooq_close = lambda _ticker, _day: None  # type: ignore[method-assign]
    ingestor.india.fetch_yahoo_actions = lambda *_args, **_kwargs: pd.DataFrame()  # type: ignore[method-assign]

    rows, dropped = ingestor._us_rows(["AAPL"], day)

    assert rows == []
    assert dropped == 1
