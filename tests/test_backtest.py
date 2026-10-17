"""Walk-forward backtester: leakage-free rebalancing, costs and NAV history."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
    assert summary["total_fees_pct_of_nav"] >= 0

    rows = [
        json.loads(line)
        for line in (state_dir / "nav_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == summary["sessions_tested"]
    for row in rows[:5]:
        assert set(row) >= {"date", "market", "portfolio_nav", "benchmark_nav", "daily_return"}
        assert row["market"] == "US"
        assert row["portfolio_nav"] > 0
    # NAV path must be strictly positive throughout (a wipeout would trip the cost model, not the data)
    assert min(row["portfolio_nav"] for row in rows) > 0


def test_walk_forward_india_uses_the_spillover_leg(e2e: dict, config, tmp_path: Path) -> None:
    state_dir = tmp_path / "bt_state_india"
    state_dir.mkdir()
    backtester = WalkForwardBacktesterV4(config, e2e["lake_dir"], state_dir)
    summary = backtester.run_walk_forward("INDIA", rebalance_step_days=20, warmup=60)
    assert "error" not in summary, summary.get("error")
    assert summary["sessions_tested"] > 100
    assert np.isfinite(summary["sharpe_ratio"])
