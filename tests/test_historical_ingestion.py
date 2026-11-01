from __future__ import annotations

import pandas as pd

from src.config import Config
from src.ingestion.fundamentals_edgar_indian_xbrl import (
    USEdgarFundamentalsParser,
    extract_pit_xbrl_quarterly_flow,
)
from src.ingestion.history_coverage import build_history_coverage_manifest, history_groups
from src.main_v4 import V4AutonomousPortfolioReducer
from src.storage.lakehouse import Lakehouse


def test_history_coverage_counts_distinct_market_sessions_and_classifies_cold_symbols():
    prices = pd.DataFrame(
        {
            "market": ["US", "US", "US", "INDIA"],
            "ticker": ["AAA", "AAA", "AAA", "AAA.NS"],
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
        }
    )
    manifest = build_history_coverage_manifest(prices)
    us = manifest["markets"]["US"]
    assert us["AAA"]["bar_count"] == 2
    assert us["AAA"]["first_trade_date"] == "2025-01-02"
    assert us["AAA"]["last_trade_date"] == "2025-01-03"

    cold, warm = history_groups(
        ["AAA", "BBB"], us, minimum_bars=2
    )
    assert cold == ["BBB"]
    assert warm == ["AAA"]
    assert history_groups(["AAA"], us, minimum_bars=2, force_full_history=True) == (["AAA"], [])


def test_sec_quarterly_flows_prefer_discrete_quarters_and_derive_q4_from_ytd():
    accessions = ["a1", "a2", "a3", "a4"]
    items = [
        {"start": "2023-01-01", "end": "2023-03-31", "filed": "2023-05-01", "form": "10-Q", "accn": accessions[0], "fy": 2023, "val": 100.0},
        {"start": "2023-01-01", "end": "2023-06-30", "filed": "2023-08-01", "form": "10-Q", "accn": accessions[1], "fy": 2023, "val": 220.0},
        {"start": "2023-04-01", "end": "2023-06-30", "filed": "2023-08-01", "form": "10-Q", "accn": accessions[1], "fy": 2023, "val": 120.0},
        {"start": "2023-01-01", "end": "2023-09-30", "filed": "2023-11-01", "form": "10-Q", "accn": accessions[2], "fy": 2023, "val": 350.0},
        {"start": "2023-07-01", "end": "2023-09-30", "filed": "2023-11-01", "form": "10-Q", "accn": accessions[2], "fy": 2023, "val": 130.0},
        {"start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-15", "form": "10-K", "accn": accessions[3], "fy": 2023, "val": 500.0},
    ]
    result = extract_pit_xbrl_quarterly_flow(
        {"Revenue": {"units": {"USD": items}}},
        ["Revenue"],
        {acc: pd.Timestamp(f"{filed}T16:00:00Z") for acc, filed in zip(accessions, ["2023-05-01", "2023-08-01", "2023-11-01", "2024-02-15"])},
    )
    assert result["val"].tolist() == [100.0, 120.0, 130.0, 150.0]


class _StubEngine:
    def beta_from_returns(self, stock_returns, market_returns):
        self.last_stock_dates = stock_returns.index
        self.last_market_dates = market_returns.index
        return 1.0


class _StubClient:
    def get_json(self, _url):
        return {"filings": {"recent": {}}}

    def get_text(self, _url):
        return ""


def test_sec_history_respects_as_of_and_uses_only_pre_disclosure_prices():
    engine = _StubEngine()
    parser = USEdgarFundamentalsParser(Config.load(), _StubClient(), engine)
    parser.company_facts = lambda _cik: {}
    parser.statement_history_frame = lambda _facts, _cik: pd.DataFrame(
        {
            "period_end": pd.to_datetime(["2022-12-31", "2023-03-31", "2023-06-30"]),
            "filing_timestamp": pd.to_datetime(
                ["2023-02-15T16:00:00Z", "2023-05-15T16:00:00Z", "2023-08-15T16:00:00Z"],
                utc=True,
            ),
            "revenue": [100.0, 110.0, 120.0],
            "shares_out": [10.0, 10.0, 10.0],
            "cash": [5.0, 5.0, 5.0],
        }
    )
    parser.business_description = lambda _cik, *, as_of=None: ""
    dates = pd.date_range("2023-05-10", "2023-05-16", freq="D")
    adjusted = pd.DataFrame({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]}, index=dates)
    raw = pd.DataFrame({"AAA": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]}, index=dates)
    market = pd.Series([0.01] * (len(dates) - 1), index=dates[1:])

    snapshots = parser.parse_history(
        "AAA",
        1,
        close_wide=adjusted,
        raw_close_wide=raw,
        market_returns=market,
        as_of=pd.Timestamp("2023-07-01", tz="UTC"),
    )

    assert len(snapshots) == 1
    assert snapshots[0]["fiscal_period"] == "2023-03-31"
    assert snapshots[0]["filing_timestamp"] == pd.Timestamp("2023-05-15T16:00:00Z")
    assert engine.last_stock_dates.max() == pd.Timestamp("2023-05-14")
    assert snapshots[0]["ev"] == 235.0  # raw $24 close × 10 shares − cash


def test_sec_statement_history_joins_acceptance_times_and_quarterly_flows():
    accessions = ["a1", "a2", "a3", "a4"]
    filed_dates = ["2023-05-01", "2023-08-01", "2023-11-01", "2024-02-15"]
    accepted = [f"{day}T16:00:00Z" for day in filed_dates]
    client = _StubClient()
    client.get_json = lambda _url: {
        "filings": {
            "recent": {
                "accessionNumber": accessions,
                "acceptanceDateTime": accepted,
            }
        }
    }
    parser = USEdgarFundamentalsParser(Config.load(), client, _StubEngine())
    revenue_items = [
        {"start": "2023-01-01", "end": "2023-03-31", "filed": filed_dates[0], "form": "10-Q", "accn": accessions[0], "fy": 2023, "val": 100.0},
        {"start": "2023-01-01", "end": "2023-06-30", "filed": filed_dates[1], "form": "10-Q", "accn": accessions[1], "fy": 2023, "val": 220.0},
        {"start": "2023-04-01", "end": "2023-06-30", "filed": filed_dates[1], "form": "10-Q", "accn": accessions[1], "fy": 2023, "val": 120.0},
        {"start": "2023-01-01", "end": "2023-09-30", "filed": filed_dates[2], "form": "10-Q", "accn": accessions[2], "fy": 2023, "val": 350.0},
        {"start": "2023-07-01", "end": "2023-09-30", "filed": filed_dates[2], "form": "10-Q", "accn": accessions[2], "fy": 2023, "val": 130.0},
        {"start": "2023-01-01", "end": "2023-12-31", "filed": filed_dates[3], "form": "10-K", "accn": accessions[3], "fy": 2023, "val": 500.0},
    ]
    assets = [
        {"end": end, "filed": day, "form": form, "accn": accession, "val": value}
        for end, day, form, accession, value in zip(
            ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
            filed_dates,
            ["10-Q", "10-Q", "10-Q", "10-K"],
            accessions,
            [1000, 1100, 1200, 1300],
        )
    ]
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": revenue_items}},
                "Assets": {"units": {"USD": assets}},
            }
        }
    }

    history = parser.statement_history_frame(facts, 1)

    assert history["revenue"].tolist() == [100.0, 120.0, 130.0, 150.0]
    assert history["total_assets"].tolist() == [1000.0, 1100.0, 1200.0, 1300.0]
    assert history["filing_timestamp"].max() == pd.Timestamp("2024-02-15T16:00:00Z")




def test_reducer_rebuilds_yahoo_adjusted_prices_after_new_action(tmp_path):
    lake = Lakehouse(tmp_path / "lake")
    lake.write(
        "market_prices",
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "market": ["US", "US"],
                "price_source": ["YAHOO_FINANCE", "YAHOO_FINANCE"],
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "open": [100.0, 50.0],
                "high": [100.0, 50.0],
                "low": [100.0, 50.0],
                "close": [100.0, 50.0],
                "volume": [100.0, 200.0],
                "adj_close": [100.0, 50.0],
                "dollar_volume": [10_000.0, 10_000.0],
                "knowledge_timestamp": pd.to_datetime(
                    ["2026-01-01T23:00:00Z", "2026-01-02T23:00:00Z"], utc=True
                ),
            }
        ),
    )
    lake.write(
        "corporate_actions",
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "market": ["US"],
                "ex_date": pd.to_datetime(["2026-01-02"]),
                "action_type": ["SPLIT"],
                "split_ratio": [2.0],
                "cash_dividend": [0.0],
                "knowledge_timestamp": pd.to_datetime(["2026-01-03T00:00:00Z"], utc=True),
            }
        ),
    )
    reducer = object.__new__(V4AutonomousPortfolioReducer)
    reducer.lake = lake
    reducer.as_of = pd.Timestamp("2026-01-03")

    prices = reducer.price_snapshot("US").sort_values("date").reset_index(drop=True)

    assert prices["adj_close"].tolist() == [50.0, 50.0]
    assert prices["close"].tolist() == [100.0, 50.0]


def test_matured_feedback_uses_trading_sessions_and_only_shrinks_low_reliability_names(tmp_path):
    config = Config.load()
    config._raw["engine"]["forecast_feedback"].update(
        {
            "horizon_sessions": 5,
            "max_nonoverlap_windows": 20,
            "min_reliability_observations": 2,
            "max_reliability_discount": 0.25,
            "min_conformal_samples": 1,
        }
    )
    lake = Lakehouse(tmp_path / "lake")
    dates = pd.bdate_range("2026-01-05", periods=21)
    prices = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "market": "US",
                "price_source": "YAHOO_FINANCE",
                "date": date,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
                "adj_close": close,
                "dollar_volume": close * 1000.0,
                "knowledge_timestamp": pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=23),
            }
            for index, date in enumerate(dates)
            for ticker, close in (("AAA", 100.0 - index), ("BBB", 100.0 + index))
        ]
    )
    lake.write("market_prices", prices)
    forecasts = []
    for index in range(0, 16, 5):
        for ticker in ("AAA", "BBB"):
            forecasts.append(
                {
                    "date": dates[index],
                    "ticker": ticker,
                    "market": "US",
                    "kalman_trend": 1.0,
                    "fundamentals": 0.0,
                    "tech_evolution": 0.0,
                    "gov_policy": 0.0,
                    "geopolitics": 0.0,
                    "macro_beta": 0.0,
                    "timestral": 0.0,
                    "composite_alpha": 1.0,
                    "timestral_q05_5d_raw_pct": -1.0,
                    "timestral_q95_5d_raw_pct": 1.0,
                }
            )
    lake.write("factor_scores", pd.DataFrame(forecasts))

    reducer = object.__new__(V4AutonomousPortfolioReducer)
    reducer.config = config
    reducer.lake = lake
    reducer.as_of = pd.Timestamp(dates[-1])
    reducer.timestral_enabled = True
    reducer.feedback_cache = {}
    reducer.calibration_rows = []

    feedback = reducer._matured_forecast_feedback("US")
    adjusted, shrunk = reducer.apply_matured_stock_reliability(
        "US", pd.Series({"AAA": 1.0, "BBB": 1.0})
    )
    calibrated = reducer.calibrate_timestral_intervals(
        "US", pd.DataFrame({"timestral_q05_5d_pct": [-2.0], "timestral_q95_5d_pct": [2.0]})
    )

    assert feedback["evaluations"] == 8
    assert feedback["reliability"]["AAA"]["multiplier"] == 0.75
    assert feedback["reliability"]["BBB"]["multiplier"] == 1.0
    assert shrunk == 1
    assert adjusted["AAA"] == 0.75
    assert adjusted["BBB"] == 1.0
    assert feedback["conformal_offset_pct"] > 0.0
    assert calibrated.at[0, "timestral_q05_5d_calibrated_pct"] < -2.0
    assert calibrated.at[0, "timestral_q95_5d_calibrated_pct"] > 2.0
