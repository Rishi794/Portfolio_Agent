"""Stage-3 Timestral 7th pillar: covariates, quantile algebra and batching contract.

The suite never downloads the 209M checkpoint: ``timestral`` is an optional Stage-3
extra, so these tests cover (a) the deterministic offline fallback, (b) the
covariate/quantile maths and (c) the *multivariate* call contract via a stub
forecaster that records the tensors it receives.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.filtering.timestral_forecaster import (
    OUTPUT_COLUMNS,
    TIMESTRAL_QUANTILES,
    TimestralForecastEngine,
)


class _StubConfig:
    """Minimal Config stand-in: dotted ``get`` with defaults."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}

    def get(self, dotted: str, default=None):
        return self.values.get(dotted, default)


class _StubForecaster:
    """Records every forecast call; returns fixed non-crossing quantiles."""

    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def forecast(self, context, **kwargs):
        recorded = {"context": np.asarray(context)}
        for key, value in kwargs.items():
            if value is not None:
                recorded[key] = np.asarray(value)
        self.calls.append(recorded)
        _, variates, _ = np.asarray(context).shape
        horizon = int(kwargs["prediction_length"])
        base = np.linspace(-1.0, 1.0, TIMESTRAL_QUANTILES.size)
        quantiles = np.tile(base, (1, variates, horizon, 1)).astype(np.float64)
        return SimpleNamespace(quantiles=quantiles, quantile_levels=TIMESTRAL_QUANTILES)


def _panel(ticker: str, periods: int = 140, seed: int = 0, liquidity: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-03-02", periods=periods)
    returns = rng.normal(0.0004, 0.014, size=periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    volume = rng.integers(50_000, 150_000, size=periods).astype(float) * liquidity
    return pd.DataFrame(
        {
            "trade_date": dates,
            "ticker": ticker,
            "adj_close": close,
            "close": close,
            "volume": volume,
            "dollar_volume": close * volume,
        }
    )


# ------------------------------------------------------------------ covariates --
def test_known_future_calendar_flags_expiry_exdates_and_month_end() -> None:
    history = pd.bdate_range("2026-08-03", "2026-10-15")
    ex_date = dt.date(2026, 10, 19)

    cov = TimestralForecastEngine.build_known_future_calendar(
        history, prediction_length=10, market="INDIA", ex_dates_union={ex_date}
    )

    assert cov.shape == (3, len(history) + 10)
    assert set(np.unique(cov)) <= {0.0, 1.0}

    future = pd.bdate_range("2026-10-16", periods=10)  # ... 2026-10-29 (last Thursday)
    columns = {timestamp.date(): index for index, timestamp in enumerate(pd.DatetimeIndex(history).append(future))}
    assert cov[0, columns[dt.date(2026, 10, 29)]] == 1.0, "NSE expiry = last Thursday of the month"
    assert cov[0, columns[dt.date(2026, 10, 22)]] == 0.0, "non-expiry Thursday must not be flagged"
    assert cov[1, columns[ex_date]] == 1.0
    assert cov[2, columns[dt.date(2026, 10, 29)]] == 1.0, "month-end rebalancing window (Oct 29 + 3d -> Nov)"
    assert cov[2, columns[dt.date(2026, 9, 29)]] == 1.0, "month-end window applies to history rows too"
    assert cov[2, columns[dt.date(2026, 10, 20)]] == 0.0
    # History rows are only informative via ex-dates (calendar rules apply throughout).
    assert cov[0].sum() >= 1.0

    us = TimestralForecastEngine.build_known_future_calendar(
        history, prediction_length=1, market="US"
    )
    friday_index = len(history)  # 2026-10-16 is the third Friday
    assert us[0, friday_index] == 1.0


# --------------------------------------------------------------------- scoring --
def test_score_quantiles_is_the_trapezoidal_integral() -> None:
    # q(p) = p => integral over the 21 knots {0.01..0.99} is exactly
    # (0.99^2 - 0.01^2) / 2 = 0.49 (the trapezoid is exact for piecewise-linear q).
    one_step = TimestralForecastEngine.score_quantiles(
        TIMESTRAL_QUANTILES[None, :], daily_vol=0.02
    )
    assert one_step["timestral_exp_ret_5d_pct"] == pytest.approx(0.49 * 0.02 * 100.0, rel=1e-6)
    assert one_step["timestral_cvar99_pct"] == 0.0, "every quantile is positive -> no tail loss"
    assert np.isfinite(one_step["f_timestral"])

    # The 5-day expectation is the sum of the per-step quadrature.
    five_step = TimestralForecastEngine.score_quantiles(
        np.tile(TIMESTRAL_QUANTILES, (5, 1)), daily_vol=0.02
    )
    assert five_step["timestral_exp_ret_5d_pct"] == pytest.approx(5 * 0.49 * 0.02 * 100.0, rel=1e-6)

    # q(p) = 1 + p: integral = 0.98 + 0.49 = 1.47.
    shifted = TimestralForecastEngine.score_quantiles(
        (1.0 + TIMESTRAL_QUANTILES)[None, :], daily_vol=0.02
    )
    assert shifted["timestral_exp_ret_5d_pct"] == pytest.approx(1.47 * 0.02 * 100.0, rel=1e-6)


def test_score_quantiles_rewards_upside_skew_and_prices_the_tail() -> None:
    symmetric = np.tile(
        np.array([-2.33, -1.64, *np.linspace(-1.0, 1.0, 17), 1.64, 2.33])[:21], (5, 1)
    )
    # Stretch the upper half: same median, fatter right tail -> higher score.
    skewed = symmetric.copy()
    skewed[:, 11:] += 1.5
    baseline = TimestralForecastEngine.score_quantiles(symmetric, 0.02)
    upside = TimestralForecastEngine.score_quantiles(skewed, 0.02)
    assert upside["f_timestral"] > baseline["f_timestral"]

    tail = TimestralForecastEngine.score_quantiles(np.full((3, 21), -0.5), 0.02)
    assert tail["timestral_cvar99_pct"] == pytest.approx(0.5 * 0.02 * 100.0, rel=1e-6)
    assert tail["timestral_cvar99_5d_pct"] == pytest.approx(3 * 0.5 * 0.02 * 100.0, rel=1e-6)

    # Unordered / NaN input must not propagate non-finite numbers.
    dirty = np.tile(TIMESTRAL_QUANTILES, (2, 1))
    dirty[0, 0] = np.nan
    dirty[1, ::2] = dirty[1, ::-2]  # reversed rows
    metrics = TimestralForecastEngine.score_quantiles(dirty, 0.02)
    assert all(np.isfinite(value) for value in metrics.values())


# ------------------------------------------------------------------- batching --
def test_joint_multivariate_forward_pass_includes_macro_variates() -> None:
    groups = {f"T{i}.NS": _panel(f"T{i}.NS", seed=i, liquidity=1.0 + i) for i in range(6)}
    macros = pd.DataFrame(
        {"US10Y_YIELD": np.linspace(-1, 1, 140), "BRENT_OIL": np.linspace(1, -1, 140)},
        index=pd.bdate_range("2026-03-02", periods=140),
    )
    stub = _StubForecaster()
    engine = TimestralForecastEngine(
        _StubConfig({"aladdin.timestral.prediction_length": 5, "aladdin.timestral.context_length": 140}),
        forecaster=stub,
    )

    result = engine.forecast_universe_cross_section(groups, market="INDIA", macro_frame=macros)

    assert len(stub.calls) == 1, "the cross-section must be one joint (B=1, V, T) pass"
    call = stub.calls[0]
    assert call["context"].shape == (1, 6 + 2, 140)
    assert call["past_covariates"].shape == (1, 6, 140)
    assert call["known_covariates"].shape == (1, 3, 140 + 5)
    assert engine.mode == "model"
    assert sorted(result.index) == sorted(groups)
    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert np.isfinite(result["f_timestral"]).all()


def test_macro_free_call_synthesises_breadth_and_dispersion_variates() -> None:
    groups = {f"T{i}.NS": _panel(f"T{i}.NS", seed=10 + i) for i in range(4)}
    stub = _StubForecaster()
    engine = TimestralForecastEngine(_StubConfig(), forecaster=stub)

    engine.forecast_universe_cross_section(groups, market="US")

    assert stub.calls[0]["context"].shape[1] == 4 + 2, "V = stocks + breadth + dispersion"


def test_liquidity_cap_truncates_the_variate_axis() -> None:
    # Multiply the base ADV by powers of ten so the liquidity ranking cannot flip
    # on random-volume noise at the cap boundary.
    groups = {f"T{i}.NS": _panel(f"T{i}.NS", seed=20 + i, liquidity=10.0**i) for i in range(6)}
    stub = _StubForecaster()
    engine = TimestralForecastEngine(
        _StubConfig({"aladdin.timestral.max_variates": 3, "aladdin.timestral.prediction_length": 3}),
        forecaster=stub,
    )

    result = engine.forecast_universe_cross_section(groups, market="INDIA")

    assert stub.calls[0]["context"].shape[1] == 3 + 2
    assert len(result) == 3
    assert set(result.index) == {"T3.NS", "T4.NS", "T5.NS"}, "highest-dollar-volume names survive"


# ------------------------------------------------------------ offline fallback --
def test_offline_fallback_scores_every_ticker_without_the_checkpoint() -> None:
    groups = {f"T{i}.NS": _panel(f"T{i}.NS", seed=30 + i) for i in range(5)}
    engine = TimestralForecastEngine(_StubConfig({"aladdin.timestral.prediction_length": 5}))

    assert engine.mode == "offline_fallback"
    result = engine.forecast_universe_cross_section(groups, market="INDIA")

    assert engine.mode in {"offline_fallback", "model"}
    assert len(result) == len(groups)
    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert np.isfinite(result.to_numpy(dtype=float)).all()
    assert (result["timestral_cvar99_pct"] >= 0.0).all()


def test_short_history_is_dropped_and_duplicate_dates_deduplicated() -> None:
    long = _panel("LONG.NS", periods=140, seed=41)
    short = _panel("SHORT.NS", periods=10, seed=42)
    duplicated = pd.concat([long, long.tail(1)])
    engine = TimestralForecastEngine(_StubConfig(), forecaster=_StubForecaster())

    result = engine.forecast_universe_cross_section(
        {"LONG.NS": duplicated, "SHORT.NS": short}, market="US"
    )

    assert list(result.index) == ["LONG.NS"]


def test_empty_universe_returns_the_empty_contract() -> None:
    engine = TimestralForecastEngine(_StubConfig())
    result = engine.forecast_universe_cross_section({}, market="US")
    assert result.empty
    assert list(result.columns) == list(OUTPUT_COLUMNS)


def test_stalled_checkpoint_load_degrades_instead_of_hanging(monkeypatch) -> None:
    """A cold/stalled Hub download must hit the deadline, not the job timeout."""
    import time

    import src.filtering.timestral_forecaster as tf

    if not tf.TIMESTRAL_AVAILABLE:
        pytest.skip("timestral extra not installed")

    class _StalledForecaster:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # noqa: ANN002, ANN003 - signature mirror
            time.sleep(5.0)
            raise AssertionError("a stalled download must not be awaited past the deadline")

    monkeypatch.setattr(tf, "TimestralForecaster", _StalledForecaster)
    # The suite-wide conftest forces the offline path; this test exercises the
    # (normally lazy) checkpoint loader itself.
    monkeypatch.delenv("PMS_TIMESTRAL_DISABLED", raising=False)
    engine = TimestralForecastEngine(
        _StubConfig(
            {
                "aladdin.timestral.load_timeout_seconds": 0.3,
                "aladdin.timestral.prediction_length": 5,
            }
        )
    )

    started = time.monotonic()
    result = engine.forecast_universe_cross_section({"T.NS": _panel("T.NS", seed=50)}, market="INDIA")
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, "the load deadline must cut the wait short"
    assert engine.mode == "offline_fallback"
    assert "TimeoutError" in (engine.load_error or "")
    assert len(result) == 1
    assert np.isfinite(result.to_numpy(dtype=float)).all()
