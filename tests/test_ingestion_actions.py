"""Corporate-action back-adjustment must survive bhavcopy int64 columns.

Regression: an int64 volume column (bhavcopy quantities arrive as whole numbers,
even-numbered prices can too) cannot store the fractional values produced by a
split factor; newer pandas raise ``LossySetitemError``/``TypeError`` instead of
silently upcasting.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.nse_bse_bhavcopy_corporate_actions import adjust_for_actions

TICKER = "RELIANCE.NS"
EX_DATE = pd.Timestamp("2026-01-05")


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [TICKER, TICKER],
            "market": ["INDIA", "INDIA"],
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "open": pd.Series([2951, 2900], dtype="int64"),
            "high": pd.Series([3001, 2950], dtype="int64"),
            "low": pd.Series([2901, 2850], dtype="int64"),
            "close": pd.Series([2941, 2905], dtype="int64"),
            "adj_close": pd.Series([2941, 2905], dtype="int64"),
            "volume": pd.Series([1001, 1500], dtype="int64"),
            "dollar_volume": pd.Series([2941 * 1001, 2905 * 1500], dtype="int64"),
        }
    )


def _action(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "ticker": TICKER,
        "market": "INDIA",
        "ex_date": EX_DATE,
        "action_type": "SPLIT",
        "split_ratio": 1.5,  # yields fractional prices *and* volume
        "cash_dividend": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_fractional_split_back_adjusts_int64_bhavcopy_columns() -> None:
    adjusted = adjust_for_actions(_prices(), _action())

    assert adjusted.loc[0, "close"] == pytest.approx(2941 / 1.5)
    assert adjusted.loc[0, "volume"] == pytest.approx(1001 * 1.5)
    # Post-ex-date rows stay exactly as traded.
    assert adjusted.loc[1, "close"] == pytest.approx(2905.0)
    assert adjusted.loc[1, "volume"] == pytest.approx(1500.0)
    # Touched columns are float64 so no later fractional assignment can fail.
    assert all(adjusted[column].dtype == "float64" for column in ("close", "adj_close", "volume"))
    # Dollar volume is rebuilt from the adjusted series.
    assert adjusted.loc[0, "dollar_volume"] == pytest.approx((2941 / 1.5) * (1001 * 1.5))


def test_dividend_only_touches_adj_close_before_ex_date() -> None:
    action = _action(action_type="DIVIDEND", split_ratio=1.0, cash_dividend=5.0)
    adjusted = adjust_for_actions(_prices(), action)

    assert adjusted.loc[0, "adj_close"] == pytest.approx(2936.0)
    assert adjusted.loc[0, "close"] == pytest.approx(2941.0)
    assert adjusted.loc[1, "adj_close"] == pytest.approx(2905.0)
    assert adjusted.loc[0, "volume"] == pytest.approx(1001.0)


def test_no_actions_still_recomputes_dollar_volume() -> None:
    adjusted = adjust_for_actions(_prices(), pd.DataFrame())

    assert adjusted.loc[0, "dollar_volume"] == pytest.approx(2941 * 1001)
    assert adjusted["volume"].dtype == "float64"
