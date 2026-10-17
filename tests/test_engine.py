"""Self-calibrating estimators: costs, deadbands, spillover, IC-IR and event half-life."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.filtering.self_calibrating_engine import SelfCalibratingQuantEngine


def test_statutory_cost_matches_the_configured_schedule(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    india = config.cost_profile("INDIA")
    expected_india = (
        float(india.get("stt_pct", 0.0))
        + float(india.get("stamp_duty_pct", 0.0))
        + float(india.get("exchange_txn_pct", 0.0))
        + float(india.get("sebi_turnover_pct", 0.0))
        + (float(india.get("exchange_txn_pct", 0.0)) + float(india.get("sebi_turnover_pct", 0.0)))
        * float(india.get("gst_on_fees_pct", 0.0))
    )
    assert engine.statutory_cost_fraction("INDIA") == pytest.approx(expected_india, abs=1e-12)
    assert 0.0005 < engine.statutory_cost_fraction("INDIA") < 0.003  # ~11.9 bps, statutory
    us = config.cost_profile("US")
    expected_us = (float(us.get("sec_section_31_bps", 0.0)) + float(us.get("finra_taf_bps", 0.0))) / 10_000.0
    assert engine.statutory_cost_fraction("US") == pytest.approx(expected_us, abs=1e-12)
    assert 0 < engine.statutory_cost_fraction("US") < 0.0001  # sub-1bp
    assert engine.statutory_cost_fraction("US") < engine.statutory_cost_fraction("INDIA")


def test_corwin_schultz_recovers_a_planted_spread(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    # Zero volatility: the observed high/low ARE the bid/ask quotes, so CS must
    # recover the planted relative spread almost exactly (alpha -> ln k).
    wide = 0.004
    flat = np.full(120, 100.0)
    estimate = engine.estimate_corwin_schultz_spread(
        flat * (1.0 + wide / 2.0), flat * (1.0 - wide / 2.0)
    )
    assert 0.0 < estimate < 0.05
    assert estimate == pytest.approx(wide, rel=0.05)
    tighter = engine.estimate_corwin_schultz_spread(
        flat * (1.0 + 0.0005), flat * (1.0 - 0.0005)
    )
    assert tighter == pytest.approx(0.001, rel=0.05)
    assert tighter < estimate


def test_deadband_is_the_kkt_closed_form(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    scale = float(config.e("optimizer.deadband_scale", 1.0))
    deadband = engine.estimate_deadband(0.002, 4.0, 0.02)
    expected = scale * 0.002 / (4.0 * (0.02**2) * 252.0)
    assert deadband == pytest.approx(expected, rel=1e-9)
    assert engine.estimate_deadband(0.004, 4.0, 0.02) == pytest.approx(2.0 * deadband, rel=1e-9)
    assert engine.estimate_deadband(0.002, 4.0, 0.04) < deadband


def test_risk_aversion_uses_realised_moments(config) -> None:
    rng = np.random.default_rng(6)
    engine = SelfCalibratingQuantEngine(config)
    calm = pd.Series(rng.normal(0.0006, 0.004, 400))
    gamma = engine.market_risk_aversion(calm)
    assert gamma >= float(config.e("costs.risk_aversion_floor", 0.25))
    assert gamma == pytest.approx(calm.mean() * 252.0 / (calm.var(ddof=0) * 252.0), rel=1e-6)
    assert engine.market_risk_aversion(pd.Series(dtype=float)) == float(
        config.e("costs.risk_aversion_floor", 0.25)
    )


def test_spillover_projection_orders_india_names_by_true_beta(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    rng = np.random.default_rng(7)
    rows = 90
    f1 = rng.normal(0.0, 0.01, rows)
    f2 = rng.normal(0.0, 0.01, rows)
    india = pd.DataFrame(
        {
            "IN_A": 2.0 * f1 + rng.normal(0.0, 0.0005, rows),
            "IN_B": 0.25 * f1 + rng.normal(0.0, 0.0005, rows),
            "IN_C": -1.0 * f2 + rng.normal(0.0, 0.0005, rows),
        }
    )
    us_factors = pd.DataFrame({"SOX": f1, "BRENT": f2})
    # a realistic ~2-sigma overnight shock. (A 100-sigma outlier would push the
    # Ledoit-Wolf shrinkage intensity to 1 and defensively zero every beta.)
    us_factors.iloc[-1] = [2.0 * 0.01, 1.5 * 0.01]
    projected = engine.project_us_to_india_spillover(india, us_factors)
    assert projected["IN_A"] > projected["IN_B"] > 0
    assert projected["IN_C"] < 0
    diagnostics = engine.spillover_diagnostics(india, us_factors)
    assert diagnostics["observations"] == rows
    assert diagnostics["r_squared"] > 0.5


def test_ic_ir_weights_recover_the_informative_pillar(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    rng = np.random.default_rng(9)
    pillars = ["kalman_trend", "fundamentals", "tech_evolution", "gov_policy", "geopolitics", "macro_beta"]
    factors, forwards = [], []
    for _ in range(14):
        cross = pd.DataFrame(rng.normal(size=(40, 6)), columns=pillars, index=[f"T{i}" for i in range(40)])
        forward = 0.05 * cross["kalman_trend"] + rng.normal(0.0, 0.005, 40)
        factors.append(cross)
        forwards.append(pd.Series(forward, index=cross.index))
    learned = engine.learn_rolling_ic_ir_factor_weights(factors, forwards)
    weights = np.asarray(learned["weights"])
    assert learned["mode"] == "ic_ir"
    assert learned["observations"] == 14
    assert weights.sum() == pytest.approx(1.0)
    assert int(np.argmax(weights)) == 0
    # cold start (too little history) must fall back to equal weights, not crash
    cold = engine.learn_rolling_ic_ir_factor_weights(factors[:2], forwards[:2])
    assert cold["mode"] == "equal_weight_cold_start"
    assert np.allclose(cold["weights"], np.ones(6) / 6)


def test_blend_with_prior_shrinks_until_ic_matures(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    learned = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prior = np.ones(6) / 6
    cold = engine.blend_with_prior(learned, prior, 0)
    assert np.allclose(cold, prior)
    mature = engine.blend_with_prior(learned, prior, 40)
    assert mature[0] > mature[1]
    assert mature.sum() == pytest.approx(1.0)


def test_event_half_life_is_bounded_and_data_driven(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    horizon = 40
    days = pd.bdate_range("2025-01-01", periods=140)
    tickers = ["A", "B", "C"]
    returns = pd.DataFrame(0.0, index=days, columns=tickers)
    events = []
    impact_map: dict[str, list[str]] = {}
    lam_true = np.log(2) / 12.0
    for index, start in enumerate((30, 60, 90)):
        impacted = tickers[index % len(tickers)]
        magnitude = 0.06
        for tau in range(1, horizon + 1):
            returns.loc[days[start + tau], impacted] += magnitude * np.exp(-lam_true * tau)
        event_id = f"E{index}"
        events.append(
            {
                "event_id": event_id,
                "domain": "GOV_POLICY",
                "published_timestamp": pd.Timestamp(days[start], tz="UTC"),
            }
        )
        impact_map[event_id] = [impacted]
    prices = (1.0 + returns).cumprod() * 100.0
    market = returns.mean(axis=1)
    result = engine.learn_event_half_life(
        pd.DataFrame(events),
        prices,
        market,
        domain="GOV_POLICY",
        impact_map=impact_map,
        horizon=horizon,
    )
    assert result["events"] == 3.0
    assert np.isfinite(result["half_life_days"])
    assert 3.0 <= result["half_life_days"] <= 180.0
    assert result["lambda"] > 0
    assert result["panel_rows"] >= 30


def test_event_half_life_degrades_gracefully_without_events(config) -> None:
    engine = SelfCalibratingQuantEngine(config)
    empty = pd.DataFrame(columns=["event_id", "domain", "published_timestamp"])
    result = engine.learn_event_half_life(empty, pd.DataFrame(), pd.Series(dtype=float), domain="GOV_POLICY")
    assert not np.isfinite(result["half_life_days"])
