from __future__ import annotations

import json

from src.reporting.investment_summary_generator import InvestmentSummaryGenerator


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
