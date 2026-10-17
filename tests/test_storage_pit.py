"""Point-in-Time lakehouse semantics: first-print wins, snapshots, integrity checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.storage.lakehouse import Lakehouse
from src.storage.verify_parquet import verify_required_tables


def _fundamental_row(beneish: float, filing: str) -> dict:
    return {
        "ticker": "AAPL",
        "market": "US",
        "fiscal_period": "2026-03-31",
        "filing_timestamp": pd.Timestamp(filing, tz="UTC"),
        "beneish_m_score": beneish,
        "piotroski_f_score": 7,
        "roic_wacc_spread": 0.05,
        "fcf_ev_yield": 0.03,
        "gics_sector": "Technology",
    }


def test_first_print_wins_for_pit_tables(lake: Lakehouse) -> None:
    lake.write("fundamentals_pit", pd.DataFrame([_fundamental_row(-3.0, "2026-05-01")]))
    # a restatement targeting the same primary key must NOT overwrite the original print
    lake.write("fundamentals_pit", pd.DataFrame([_fundamental_row(-1.0, "2026-05-01")]))
    stored = lake.read("fundamentals_pit")
    assert len(stored) == 1
    assert float(stored.iloc[0]["beneish_m_score"]) == pytest.approx(-3.0)


def test_latest_wins_when_pit_protection_is_disabled(lake: Lakehouse) -> None:
    lake.write("fundamentals_pit", pd.DataFrame([_fundamental_row(-3.0, "2026-05-01")]))
    lake.write("fundamentals_pit", pd.DataFrame([_fundamental_row(-1.0, "2026-05-01")]), enforce_pit=False)
    assert float(lake.read("fundamentals_pit").iloc[0]["beneish_m_score"]) == pytest.approx(-1.0)


def test_snapshot_at_never_sees_future_filings(lake: Lakehouse) -> None:
    lake.write("fundamentals_pit", pd.DataFrame([_fundamental_row(-3.0, "2026-02-01")]))
    lake.write("fundamentals_pit", pd.DataFrame([_fundamental_row(-1.5, "2026-05-01")]))
    assert len(lake.read("fundamentals_pit")) == 2
    early = lake.snapshot_at("fundamentals_pit", pd.Timestamp("2026-03-01"), entity=["ticker"])
    assert len(early) == 1
    assert float(early.iloc[0]["beneish_m_score"]) == pytest.approx(-3.0)
    latest = lake.snapshot_at("fundamentals_pit", pd.Timestamp("2026-06-01"), entity=["ticker"])
    assert len(latest) == 1
    assert float(latest.iloc[0]["beneish_m_score"]) == pytest.approx(-1.5)
    before_any = lake.snapshot_at("fundamentals_pit", pd.Timestamp("2026-01-01"), entity=["ticker"])
    assert before_any.empty


def test_upsert_is_idempotent(lake: Lakehouse) -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "X", "market": "US", "date": "2026-01-05", "open": 1.0, "high": 1.1, "low": 0.9,
             "close": 1.0, "volume": 10.0, "adj_close": 1.0, "dollar_volume": 10.0,
             "knowledge_timestamp": pd.Timestamp("2026-01-05", tz="UTC")},
        ]
    )
    lake.write("market_prices", frame)
    for _ in range(3):
        lake.write("market_prices", frame)
    assert len(lake.read("market_prices")) == 1
    assert lake.verify()["ok"]


def test_embedding_width_is_validated(lake: Lakehouse) -> None:
    good = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "published_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "domain": "GOV_POLICY",
                "embedding": [[0.1] * 384],
            }
        ]
    )
    lake.write("policy_geopol_events", good)
    assert lake.verify()["ok"]
    bad = pd.DataFrame(
        [
            {
                "event_id": "E2",
                "published_timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                "domain": "GOV_POLICY",
                "embedding": [[0.1] * 8],
            }
        ]
    )
    lake.write("policy_geopol_events", bad)
    report = lake.verify()
    assert not report["ok"]
    assert any("widths" in issue for issue in report["issues"])


def test_required_table_report_detects_missing_and_bad_prices(lake: Lakehouse) -> None:
    report = verify_required_tables(lake)
    assert not report["ok"]
    assert len(report["issues"]) == len(report["tables"])  # everything missing
    assert report["tables"]["raw_ohlcv"]["physical_table"] == "market_prices"

    frame = pd.DataFrame(
        [
            {"ticker": "X", "market": "US", "date": "2026-01-05", "open": 1.0, "high": 1.1, "low": 0.9,
             "close": -1.0, "volume": 10.0, "adj_close": 1.0, "dollar_volume": 10.0,
             "knowledge_timestamp": pd.Timestamp("2026-01-05", tz="UTC")},
        ]
    )
    lake.write("market_prices", frame)
    report = verify_required_tables(lake, require_non_empty=False)
    assert "non-positive" in " ".join(report["issues"])
    assert report["tables"]["raw_ohlcv"]["non_positive_prices"] == {"close": 1}


def test_query_and_asof_join_work(lake: Lakehouse) -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "X", "market": "US", "date": f"2026-01-0{day}", "open": 1.0, "high": 1.1,
             "low": 0.9, "close": float(day), "volume": 10.0, "adj_close": float(day),
             "dollar_volume": 10.0 * day, "knowledge_timestamp": pd.Timestamp(f"2026-01-0{day}", tz="UTC")}
            for day in (1, 2, 3)
        ]
    )
    lake.write("market_prices", frame)
    result = lake.query("SELECT ticker, close FROM market_prices ORDER BY close")
    assert result["close"].tolist() == [1.0, 2.0, 3.0]
    joined = lake.asof_join(
        "market_prices", "market_prices", left_on="date", right_on="date", by="ticker",
        select="l.ticker, l.date, r.close",
    )
    assert len(joined) == 3
