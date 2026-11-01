from types import SimpleNamespace

import pandas as pd
import pytest

from src.main_v4 import V4AutonomousPortfolioReducer
from src.main_v5 import format_aladdin_markdown


def test_timestral_return_reports_coverage_portfolio_contribution_and_cost_drag() -> None:
    reducer = object.__new__(V4AutonomousPortfolioReducer)
    reducer.timestral_engine = SimpleNamespace(mode="offline_fallback")
    reducer.timestral_frames = {
        "US": pd.DataFrame(
            {
                "timestral_exp_ret_5d_pct": [2.0, 4.0],
                "timestral_cvar99_pct": [3.0, 1.0],
            },
            index=["AAA", "BBB"],
        )
    }

    metrics = reducer._timestral_risk_inputs(
        "US",
        pd.Series({"AAA": 0.4, "BBB": 0.2, "NO_FORECAST": 0.2}),
        {},
        [
            {"order_value": 10_000.0, "est_cost_bps": 100.0},
            {"order_value": 5_000.0, "est_cost_bps": 100.0},
        ],
        100_000.0,
    )

    # The sleeve covers 75% of invested weight. Its 2.6667% estimate contributes
    # 1.6 percentage points to the whole account; estimated current trade costs
    # reduce that to 1.45 percentage points.
    assert metrics["timestral_forecast_coverage_pct"] == pytest.approx(75.0)
    assert metrics["timestral_forward_exp_ret_5d_covered_sleeve_pct"] == pytest.approx(2.6667)
    assert metrics["timestral_exp_ret_5d_portfolio_contribution_pct"] == pytest.approx(1.6)
    assert metrics["timestral_estimated_rebalance_cost_pct"] == pytest.approx(0.15)
    assert metrics["timestral_net_exp_ret_5d_portfolio_contribution_pct"] == pytest.approx(1.45)
    assert metrics["timestral_exp_gain_5d_portfolio_value"] == pytest.approx(1_600.0)
    assert metrics["timestral_net_exp_gain_5d_portfolio_value"] == pytest.approx(1_450.0)
    assert metrics["timestral_forward_cvar99_pct"] == pytest.approx(2.3333)


def test_timestral_return_does_not_report_zero_when_model_has_no_coverage() -> None:
    reducer = object.__new__(V4AutonomousPortfolioReducer)
    reducer.timestral_engine = SimpleNamespace(mode="offline_fallback")
    reducer.timestral_frames = {"US": pd.DataFrame(columns=["timestral_exp_ret_5d_pct"])}

    metrics = reducer._timestral_risk_inputs("US", pd.Series({"AAA": 0.5}), {}, [], 100_000.0)

    assert metrics["timestral_forecast_coverage_pct"] == 0.0
    assert metrics["timestral_exp_ret_5d_portfolio_contribution_pct"] is None


def test_issue_report_labels_forecast_estimate_and_net_account_gain() -> None:
    report = format_aladdin_markdown(
        {"as_of": "2026-11-01"},
        {
            "markets": {
                "US": {
                    "timestral_mode": "offline_fallback",
                    "timestral_forward_exp_ret_5d_pct": 2.67,
                    "timestral_exp_ret_5d_portfolio_contribution_pct": 1.6,
                    "timestral_exp_gain_5d_portfolio_value": 1600.0,
                    "timestral_net_exp_ret_5d_portfolio_contribution_pct": 1.45,
                    "timestral_net_exp_gain_5d_portfolio_value": 1450.0,
                    "timestral_forecast_coverage_pct": 75.0,
                }
            }
        },
    )

    assert "$+1600.00 gross" in report
    assert "$+1450.00 net" in report
    assert "estimates are not guaranteed gains" in report
