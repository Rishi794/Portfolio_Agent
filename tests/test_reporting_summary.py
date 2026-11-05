from __future__ import annotations

import json

from src.reporting.investment_summary_generator import InvestmentSummaryGenerator
from src.reporting.dashboard_builder import build_payload


def test_investment_summary_uses_portfolio_state_and_current_run_proposals(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "portfolio_ledger_us.json").write_text(
        json.dumps(
            {
                "market": "US",
                "cash": 50_000.0,
                "positions": {
                    "HOLD": {"shares": 10, "avg_cost": 100, "last_price": 110},
                    "REDUCE": {"shares": 5, "avg_cost": 100, "last_price": 90},
                },
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "as_of": "2026-10-30",
        "run_id": "run-123",
        "mode": "daily_eod",
        "runs": [
            {
                "market": "US",
                "tickets": [
                    {
                        "action": "BUY",
                        "ticker": "BUYME",
                        "shares": None,
                        "ref_price": 12.5,
                        "target_weight_pct": 2.0,
                    },
                    {"action": "SELL", "ticker": "REDUCE", "shares": 5, "alpha_z": -1.2},
                ],
                "quarantined_tickers": ["CHECK"],
            }
        ],
        "markets": {
            "US": {
                "forecast_feedback": {
                    "matured_evaluations": 10,
                    "nonoverlap_horizon_sessions": 5,
                    "directional_accuracy": 0.6,
                    "timestral_interval_offset_pct": 0.2,
                    "timestral_interval_samples": 10,
                    "low_reliability_names": [
                        {"ticker": "WEAK", "posterior_mean": 0.4, "alpha_multiplier": 0.8}
                    ],
                    "recent_misses": [
                        {
                            "pred_date": "2026-10-01",
                            "ticker": "MISS",
                            "pred_alpha_z": 1.0,
                            "realized_ret_pct": -2.0,
                            "failure_mode": "DIRECTIONAL_MISS",
                            "largest_signal_aligned_pillar": "kalman_trend",
                        }
                    ],
                }
            }
        },
    }

    output = tmp_path / "SUMMARY.md"
    rendered = InvestmentSummaryGenerator.render(summary, state_dir, output)

    assert rendered == output.read_text(encoding="utf-8")
    assert "BUY `BUYME`" in rendered and "0 shares" in rendered
    assert "`HOLD`" in rendered
    assert "SELL `REDUCE`" in rendered
    assert "Do not buy while data verification is unresolved" in rendered
    assert "No automatic ticker quarantine" in rendered
    assert "descriptive, not causal" in rendered
    assert "not a formal coverage guarantee" in rendered


def test_dashboard_keeps_backtest_nav_out_of_live_ledger_and_shows_separate_curve(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "nav_history.jsonl").write_text(
        '\n'.join(
            [
                json.dumps({"date": "2026-10-01", "market": "US", "equity": 101000, "cash": 20000}),
                json.dumps({"date": "2026-10-01", "market": "US", "portfolio_nav": 1.02, "benchmark_nav": 1.01}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    (state_dir / "walk_forward_nav_history.jsonl").write_text(
        json.dumps(
            {
                "date": "2026-10-01",
                "market": "US",
                "portfolio_nav": 1.02,
                "benchmark_nav": 1.01,
                "daily_return": 0.01,
            }
        ) + "\n",
        encoding="utf-8",
    )

    payload = build_payload(state_dir)
    us_book = next(book for book in payload["books"] if book["market"] == "US")
    assert len(us_book["nav"]) == 1
    assert us_book["nav"][0]["equity"] == 101000
    assert payload["walk_forward"]["US"][0]["portfolio_nav"] == 1.02
    assert payload["walk_forward"]["INDIA"] == []
