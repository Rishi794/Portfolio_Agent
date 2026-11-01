from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from src.flex_model.aladdin_features import flex
from src.flex_model.runtime import FlexModelReplayRuntime


def _inputs(*, include_latest_bar: bool = False):
    dates = [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    closes = [[100.0, 50.0], [102.0, 49.0]]
    if include_latest_bar:
        dates.append(pd.Timestamp("2025-01-04"))
        closes.append([104.0, 51.0])
    rows = []
    for day, values in zip(dates, closes, strict=True):
        for ticker, close in zip(("A", "B"), values, strict=True):
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "close": close,
                    "adj_close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "knowledge_timestamp": day.tz_localize("UTC") + pd.Timedelta(hours=22),
                }
            )
    fundamentals = pd.DataFrame(
        {
            "roic_wacc_spread": [0.1, 0.2],
            "fcf_ev_yield": [0.03, 0.04],
            "piotroski_f_score": [8.0, 7.0],
        },
        index=pd.Index(["A", "B"], name="ticker"),
    )
    tech = pd.DataFrame(
        {"citation_velocity_log_ratio": [0.4, -0.2]},
        index=pd.Index(["A", "B"], name="ticker"),
    )
    prices = pd.DataFrame(rows)
    actions = pd.DataFrame(
        columns=[
            "ticker", "market", "ex_date", "action_type", "split_ratio", "cash_dividend",
            "knowledge_timestamp",
        ]
    )
    return prices, actions, fundamentals, tech


def test_replay_resumes_incrementally_and_matches_full_replay(tmp_path) -> None:
    runtime = FlexModelReplayRuntime(tmp_path / "incremental", flex)
    prices, actions, fundamentals, tech = _inputs()
    policy = pd.Series([0.5, -0.5], index=["A", "B"])
    spillover = pd.Series(dtype=float)

    first, initial_diagnostics = runtime.replay(
        market="US",
        tickers=["A", "B"],
        as_of=dt.date(2025, 1, 4),
        prices=prices,
        actions=actions,
        fundamentals=fundamentals,
        tech=tech,
        policy_scores=policy,
        spillover_scores=spillover,
    )
    assert initial_diagnostics["restored"] is False
    assert "f_fundamentals_us" in first
    assert np.any(first["f_fundamentals_us"].to_numpy() != 0.0)

    extended_prices, actions, fundamentals, tech = _inputs(include_latest_bar=True)
    incremental, diagnostics = runtime.replay(
        market="US",
        tickers=["A", "B"],
        as_of=dt.date(2025, 1, 5),
        prices=extended_prices,
        actions=actions,
        fundamentals=fundamentals,
        tech=tech,
        policy_scores=policy,
        spillover_scores=spillover,
    )
    fresh_runtime = FlexModelReplayRuntime(tmp_path / "fresh", flex)
    full, _ = fresh_runtime.replay(
        market="US",
        tickers=["A", "B"],
        as_of=dt.date(2025, 1, 5),
        prices=extended_prices,
        actions=actions,
        fundamentals=fundamentals,
        tech=tech,
        policy_scores=policy,
        spillover_scores=spillover,
    )

    assert diagnostics["restored"] is True
    assert diagnostics["replayed_bars"] == 1
    pd.testing.assert_frame_equal(incremental, full)


def test_partial_snapshot_restore_failure_rebuilds_from_clean_arena(tmp_path) -> None:
    state_dir = tmp_path / "state"
    runtime = FlexModelReplayRuntime(state_dir, flex)
    prices, actions, fundamentals, tech = _inputs()
    args = {
        "market": "US",
        "tickers": ["A", "B"],
        "as_of": dt.date(2025, 1, 4),
        "prices": prices,
        "actions": actions,
        "fundamentals": fundamentals,
        "tech": tech,
        "policy_scores": pd.Series([0.5, -0.5], index=["A", "B"]),
        "spillover_scores": pd.Series(dtype=float),
    }
    expected, _ = runtime.replay(**args)

    snapshot_path = state_dir / "flex_arena_us.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    # This fails after feature/ring restoration has started, which used to leave
    # the arena partially restored before replaying the complete history again.
    snapshot["arena"]["extra_state"]["kalman_level"] = [123.0]
    snapshot_path.write_text(json.dumps(snapshot, allow_nan=True), encoding="utf-8")

    actual, diagnostics = runtime.replay(**args)
    assert diagnostics["restored"] is False
    pd.testing.assert_frame_equal(actual, expected)
