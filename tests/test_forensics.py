"""Forensic fundamentals: Beneish, Piotroski, ROIC-WACC and the endogenous survival gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.filtering.forensics import (
    beneish_m_score,
    compute_forensic_metrics,
    estimate_survival_gate,
    forensic_risk_index,
    fundamental_quality_score,
    gate_mask,
    piotroski_f_score,
)


def _config(**forensics_overrides) -> Config:
    engine = {"forensics": {"gate_mode": "statutory", **forensics_overrides}}
    return Config({"engine": engine}, Path("."))


def test_beneish_matches_the_published_coefficients() -> None:
    # every ratio = 1 and TATA = 0 -> the 8-variable probit collapses to the intercept sum
    assert beneish_m_score(1, 1, 1, 1, 1, 1, 0, 1) == pytest.approx(-2.480, abs=1e-6)


def test_beneish_flags_earnings_manipulation() -> None:
    # aggressive revenue growth + accruals must push M above the -1.78 statutory cut
    assert beneish_m_score(2.0, 1.5, 1.4, 2.2, 1.0, 0.9, 0.15, 1.1) > -1.78
    assert beneish_m_score(0.9, 1.0, 1.0, 1.0, 1.0, 1.05, -0.02, 1.0) < -1.78


def test_piotroski_counts_all_nine_signals() -> None:
    curr = pd.Series(
        {
            "revenue": 100.0, "cogs": 50.0, "receivables": 10.0, "current_assets": 50.0,
            "total_assets": 100.0, "current_liab": 10.0, "long_term_debt": 0.0,
            "net_income": 10.0, "operating_cf": 15.0, "ppe": 30.0, "shares_out": 10.0,
            "depreciation": 2.0, "sga": 5.0,
        }
    )
    prev = pd.Series(
        {
            "revenue": 100.0, "cogs": 60.0, "receivables": 10.0, "current_assets": 40.0,
            "total_assets": 120.0, "current_liab": 20.0, "long_term_debt": 0.0,
            "net_income": 6.0, "operating_cf": 15.0, "ppe": 30.0, "shares_out": 10.0,
            "depreciation": 2.0, "sga": 5.0,
        }
    )
    assert piotroski_f_score(curr, prev) == 9


def test_compute_forensic_metrics_panel() -> None:
    rows = [
        {
            "period_end": pd.Timestamp("2024-03-31"), "filing_timestamp": pd.Timestamp("2024-05-01", tz="UTC"),
            "revenue": 100.0, "cogs": 55.0, "receivables": 12.0, "current_assets": 60.0,
            "total_assets": 200.0, "current_liab": 30.0, "long_term_debt": 20.0,
            "net_income": 12.0, "operating_cf": 18.0, "ppe": 90.0, "shares_out": 100.0,
            "depreciation": 8.0, "sga": 10.0, "ebit": 20.0, "capex": 6.0, "cash": 15.0, "inventory": 10.0,
        },
        {
            "period_end": pd.Timestamp("2025-03-31"), "filing_timestamp": pd.Timestamp("2025-05-01", tz="UTC"),
            "revenue": 130.0, "cogs": 68.0, "receivables": 15.0, "current_assets": 70.0,
            "total_assets": 220.0, "current_liab": 32.0, "long_term_debt": 22.0,
            "net_income": 16.0, "operating_cf": 24.0, "ppe": 100.0, "shares_out": 101.0,
            "depreciation": 9.0, "sga": 12.0, "ebit": 28.0, "capex": 9.0, "cash": 20.0, "inventory": 12.0,
        },
    ]
    metrics = compute_forensic_metrics(
        pd.DataFrame(rows),
        beta=1.1,
        risk_free=0.04,
        equity_risk_premium=0.045,
        effective_tax_rate=0.25,
        min_wacc=0.06,
        max_wacc=0.18,
        market_cap=1500.0,
        cash=20.0,
    )
    assert -4.84 < metrics.beneish_m_score < 2.0
    assert 0 <= metrics.piotroski_f_score <= 9
    assert metrics.wacc == pytest.approx(0.04 + 1.1 * 0.045, abs=1e-9)
    assert metrics.roic_wacc_spread == pytest.approx(metrics.roic - metrics.wacc, abs=1e-6)
    assert metrics.fcf == pytest.approx(24.0 - 9.0)
    assert metrics.ev == pytest.approx(1500.0 + 54.0 - 20.0)


def test_statutory_gate_mode_uses_published_cuts() -> None:
    frame = pd.DataFrame(
        {
            "beneish_m_score": [-2.5, -1.0, -3.0, -1.9],
            "piotroski_f_score": [7, 8, 4, 6],
            "roic_wacc_spread": [0.05, 0.02, -0.01, 0.03],
            "promoter_pledge_pct": [0.0, 0.0, 40.0, 0.0],
            "gics_sector": ["Technology"] * 4,
            "dollar_volume_20d": [1e9] * 4,
        },
        index=["A", "B", "C", "D"],
    )
    thresholds = estimate_survival_gate(frame, _config())
    mask = gate_mask(frame, thresholds, min_dollar_volume=1e7)
    assert thresholds["__GLOBAL__"].mode == "statutory"
    assert mask.tolist() == [True, False, False, True]


def test_quantile_gate_is_endogenous_per_sector() -> None:
    rng = np.random.default_rng(3)
    n = 120
    frame = pd.DataFrame(
        {
            "beneish_m_score": rng.normal(-2.0, 0.5, n),
            "piotroski_f_score": rng.integers(0, 10, n),
            "roic_wacc_spread": rng.normal(0.03, 0.06, n),
            "promoter_pledge_pct": np.abs(rng.normal(5, 8, n)),
            "gics_sector": ["Technology"] * 60 + ["Industrials"] * 60,
            "dollar_volume_20d": rng.uniform(1e7, 1e9, n),
        },
        index=[f"T{i}" for i in range(n)],
    )
    thresholds = estimate_survival_gate(frame, _config(gate_mode="quantile", survival_quantile=0.35, min_cross_section=50))
    assert thresholds["Technology"].mode == "quantile"
    assert thresholds["Technology"].sample_size == 60
    assert thresholds["Technology"].risk_cutoff is not None
    mask = gate_mask(frame, thresholds, min_dollar_volume=1e7)
    kept = mask[mask].index
    kept_tech = [t for t in kept if t in set(frame.index[:60])]
    # the estimated survival quantile of each sector survives — never the worst names
    assert 0.25 * 60 <= len(kept_tech) <= 0.45 * 60
    # compare in one common risk-index space (a per-subset index would be centred)
    risk = forensic_risk_index(frame, 0.01)
    rejected_tech = frame.index[:60].difference(kept_tech)
    assert risk.loc[kept_tech].mean() < risk.loc[rejected_tech].mean()


def test_quality_score_is_cross_sectionally_centered() -> None:
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "roic_wacc_spread": rng.normal(0.05, 0.05, 90),
            "fcf_ev_yield": rng.normal(0.04, 0.03, 90),
            "piotroski_f_score": rng.integers(0, 10, 90),
            "gics_sector": ["Technology"] * 30 + ["Financials"] * 30 + ["Energy"] * 30,
        },
        index=[f"Q{i}" for i in range(90)],
    )
    score = fundamental_quality_score(frame, (0.45, 0.35, 0.20), winsor_quantile=0.01)
    assert len(score) == 90
    assert np.isfinite(score.to_numpy()).all()
    assert abs(float(score.mean())) < 1e-9
    assert score.std() > 0.1
