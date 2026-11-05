"""Walk-forward backtester: leakage-free rebalancing, costs and NAV history."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.backtest.walk_forward_v4 import WalkForwardBacktesterV4


def test_walk_forward_produces_metrics_and_nav_history(e2e: dict, config, tmp_path: Path) -> None:
    state_dir = tmp_path / "bt_state"
    state_dir.mkdir()
    backtester = WalkForwardBacktesterV4(config, e2e["lake_dir"], state_dir)
    summary = backtester.run_walk_forward("US", rebalance_step_days=10, warmup=60)
    assert "error" not in summary, summary.get("error")
    assert summary["sessions_tested"] > 100
    assert summary["final_portfolio_nav"] > 0
    assert summary["final_benchmark_nav"] > 0
    for key in ("cagr_pct", "annual_vol_pct", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "total_turnover_x"):
        assert np.isfinite(summary[key]), f"{key} is not finite"
    assert summary["max_drawdown_pct"] <= 0
    assert summary["total_turnover_x"] >= 0
    assert summary["total_fees_pct_of_initial_nav"] >= 0
    assert "V5 reducer factor builder" in summary["signal_path"]
    assert "golden-copy gate unavailable" in summary["execution_model"]

    rows = [
        json.loads(line)
        for line in (state_dir / "walk_forward_nav_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == summary["sessions_tested"]
    for row in rows[:5]:
        assert set(row) >= {
            "date", "market", "strategy", "portfolio_nav", "benchmark_nav",
            "daily_return", "gross_daily_return", "transaction_cost_pct",
        }
        assert row["market"] == "US"
        assert row["strategy"] == "v5_shared_reducer_signals"
        assert row["portfolio_nav"] > 0
    # NAV path must be strictly positive throughout (a wipeout would trip the cost model, not the data)
    assert min(row["portfolio_nav"] for row in rows) > 0
    assert not (state_dir / "nav_history.jsonl").exists(), "backtest data must not pollute live ledger NAV"
    for previous_nav, current in zip([1.0, *[row["portfolio_nav"] for row in rows[:-1]]], rows):
        assert np.isfinite(current["daily_return"])
        assert np.isfinite(current["gross_daily_return"])
        assert current["transaction_cost_pct"] >= 0
        assert current["daily_return"] == pytest.approx(
            current["portfolio_nav"] / previous_nav - 1.0,
            abs=2e-7,
        )


def test_walk_forward_india_uses_the_spillover_leg(e2e: dict, config, tmp_path: Path) -> None:
    state_dir = tmp_path / "bt_state_india"
    state_dir.mkdir()
    backtester = WalkForwardBacktesterV4(config, e2e["lake_dir"], state_dir)
    summary = backtester.run_walk_forward("INDIA", rebalance_step_days=20, warmup=60)
    assert "error" not in summary, summary.get("error")
    assert summary["sessions_tested"] > 100
    assert np.isfinite(summary["sharpe_ratio"])


def test_backtest_nav_replaces_same_market_curve_without_duplicates(
    e2e: dict, config, tmp_path: Path
) -> None:
    state_dir = tmp_path / "bt_state_idempotent"
    backtester = WalkForwardBacktesterV4(config, e2e["lake_dir"], state_dir)
    first = backtester.run_walk_forward("US", rebalance_step_days=20, warmup=60)
    second = backtester.run_walk_forward("US", rebalance_step_days=20, warmup=60)
    assert first["sessions_tested"] == second["sessions_tested"]
    rows = [
        json.loads(line)
        for line in (state_dir / "walk_forward_nav_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == second["sessions_tested"]
    assert len({(row["market"], row["date"]) for row in rows}) == len(rows)


def test_daily_profile_caps_window_and_keeps_full_curve_separate(
    e2e: dict, config, tmp_path: Path
) -> None:
    state_dir = tmp_path / "bt_state_daily"
    backtester = WalkForwardBacktesterV4(config, e2e["lake_dir"], state_dir)
    full = backtester.run_walk_forward("US", rebalance_step_days=10, warmup=60)
    daily = backtester.run_walk_forward(
        "US", rebalance_step_days=5, warmup=60, test_sessions=30, profile="daily"
    )

    assert "error" not in full, full.get("error")
    assert "error" not in daily, daily.get("error")
    assert daily["profile"] == "daily"
    assert daily["test_sessions_requested"] == 30
    assert daily["sessions_tested"] == 30
    daily_rows = [
        json.loads(line)
        for line in (state_dir / "walk_forward_nav_daily.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    full_rows = [
        json.loads(line)
        for line in (state_dir / "walk_forward_nav_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(daily_rows) == 30
    assert len(full_rows) == full["sessions_tested"]
    assert daily_rows[0]["profile"] == "daily"
    assert daily_rows[0]["date"] == daily["first_test_date"]
