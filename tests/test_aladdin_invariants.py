"""Formal Open-Aladdin mathematical invariant gate (run before the Fan-In publishes).

Five theorems are verified on every CI run:

1. **Golden-copy Byzantine quorum** — a 2-of-3 median consensus rejects a
   corrupted third vendor print; 3-way divergence quarantines; an unexplained
   8-sigma flash crash quarantines via the Hampel MAD filter.
2. **Decimal double-entry IBOR** — ``Assets == Liabilities + Equity + Income −
   Expenses`` to 8 decimals across sequential BUYs, a partial SELL and fees;
   plus the float-ledger reconciliation identity used by the V5 audit.
3. **Euler risk additivity** — ``sum(CCTR_i) == sigma_p`` and
   ``sum(PCTR_i) == 100%``.
4. **Löwdin symmetric orthogonality** — the orthogonalized factor matrix has an
   identity correlation matrix.
5. **Stable Kalman innovations + pre-trade ADV compliance** — positive
   innovation variance and an order scaled to the 5%-of-ADV participation cap.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from src.config import Config
from src.filtering.lowdin_orthogonalizer import lowdin_orthogonalize, orthogonality_error
from src.filtering.self_calibrating_engine import SelfCalibratingQuantEngine
from src.verification.aladdin_golden_copy_and_risk import (
    AladdinRiskAndScenarioEngine,
    DoubleEntryIBORLedger,
    GoldenCopyConsensusEngine,
    reconcile_double_entry_identity,
)


def test_invariant_1_golden_copy_quorum_and_mad_gate() -> None:
    engine = GoldenCopyConsensusEngine(max_source_divergence_bps=5.0, mad_sigma_threshold=8.0)
    trailing = np.linspace(2900.0, 2950.0, 25)

    # Case A: NSE + BSE agree (~1.7 bps apart); Yahoo prints a corrupted 2600.00.
    rec_valid = engine.build_golden_copy_bar(
        trade_date="2026-10-16",
        ticker="RELIANCE.NS",
        market="INDIA",
        source_prices={"NSE_UDIFF": 2950.00, "BSE_UDIFF": 2950.50, "YAHOO_FINANCE": 2600.00},
        trailing_golden_closes=trailing,
        has_corporate_action_today=False,
    )
    assert rec_valid.status == "VERIFIED_GOLDEN"
    assert rec_valid.sources_agreed == 2
    assert abs(rec_valid.golden_close - Decimal("2950.25")) < Decimal("0.0001")

    # Case B: all three vendors diverge -> divergence quarantine (takes precedence).
    rec_div = engine.build_golden_copy_bar(
        trade_date="2026-10-16",
        ticker="TCS.NS",
        market="INDIA",
        source_prices={"NSE_UDIFF": 4100.0, "BSE_UDIFF": 4150.0, "YAHOO_FINANCE": 4200.0},
        trailing_golden_closes=trailing,
    )
    assert rec_div.status == "QUARANTINED_DIVERGENCE"

    # Case C: every vendor agrees on an unexplained ~40% crash -> Hampel MAD.
    rec_mad = engine.build_golden_copy_bar(
        trade_date="2026-10-16",
        ticker="INFY.NS",
        market="INDIA",
        source_prices={"NSE_UDIFF": 1700.0, "BSE_UDIFF": 1700.0, "YAHOO_FINANCE": 1700.0},
        trailing_golden_closes=trailing,
        has_corporate_action_today=False,
    )
    assert rec_mad.status == "QUARANTINED_MAD_SPIKE"

    # Byzantine tail: with per-feed error p = 1e-3, two of three feeds failing
    # identically is ~3p^2 < 4e-6.
    assert GoldenCopyConsensusEngine.quorum_probability(1e-3) < 4e-6


def test_invariant_2_decimal_double_entry_zero_discrepancy() -> None:
    ledger = DoubleEntryIBORLedger(market="INDIA", initial_cash="1000000.00")
    ledger.post_trade("RELIANCE.NS", "BUY", shares=100, fill_price="2945.35", fee_amount="349.91")
    ledger.post_trade("TCS.NS", "BUY", shares=50, fill_price="4120.15", fee_amount="244.74")
    ledger.post_trade("RELIANCE.NS", "SELL", shares=40, fill_price="3010.80", fee_amount="143.07")

    assert ledger.verify_accounting_invariant() is True
    assert ledger.positions["RELIANCE.NS"]["shares"] == 60
    assert isinstance(ledger.accounts["ASSET:CASH"], Decimal)
    assert ledger.chain_hash != "GENESIS_ALADDIN_V5"


def test_invariant_2b_float_ledger_snapshot_reconciles() -> None:
    # start 1000 -> BUY 30 @ 10 with fee 2: cash 698, basis 300; identity holds.
    snapshot = {
        "starting_cash": 1000.0,
        "cash": 698.0,
        "realized_pnl": 0.0,
        "fees_paid": 2.0,
        "positions": {"AAA": {"shares": 30.0, "avg_cost": 10.0}},
    }
    assert reconcile_double_entry_identity(snapshot)["ok"]
    broken = {**snapshot, "cash": 700.0}
    assert not reconcile_double_entry_identity(broken)["ok"]


def test_invariant_3_euler_risk_additivity() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=252, freq="B")
    tickers = ["NVDA", "MSFT", "LLY", "RELIANCE.NS", "TCS.NS"]
    ret_df = pd.DataFrame(rng.normal(0.0005, 0.018, size=(252, 5)), index=dates, columns=tickers)
    weights = pd.Series([0.25, 0.20, 0.20, 0.20, 0.15], index=tickers)

    engine = AladdinRiskAndScenarioEngine()
    port_vol_pct, cvar_99_pct, decomp = engine.compute_euler_risk_decomposition(weights, ret_df)

    assert np.isclose(decomp["cctr_ann_pct"].sum(), port_vol_pct, atol=1e-2)
    assert np.isclose(decomp["pct_of_total_risk"].sum(), 100.0, atol=1e-2)
    assert cvar_99_pct > 0.0


def test_invariant_4_lowdin_symmetric_orthogonality() -> None:
    rng = np.random.default_rng(7)
    n_stocks = 120
    base = rng.normal(0, 1, size=(n_stocks, 1))
    raw = base + rng.normal(0, 0.4, size=(n_stocks, 6))
    columns = ["f_kalman", "f_fundamentals", "f_tech", "f_policy", "f_geopol", "f_spillover_macro"]
    raw_df = pd.DataFrame(raw, columns=columns)

    orthogonal = lowdin_orthogonalize(raw_df)
    assert list(orthogonal.columns) == columns
    assert orthogonality_error(orthogonal) < 1e-2
    identity_error = np.linalg.norm(
        orthogonal.values.T @ orthogonal.values / n_stocks - np.eye(6), ord="fro"
    )
    assert identity_error < 1e-2


def test_invariant_5_kalman_psd_and_pre_trade_adv_compliance() -> None:
    engine = SelfCalibratingQuantEngine(Config.load())
    prices = 100.0 * np.exp(np.cumsum(np.random.default_rng(11).normal(0.001, 0.015, 80)))
    denoised, _velocity, r_std = engine.innovation_adaptive_kalman(prices)
    assert r_std > 0.0
    assert len(denoised) == 80
    assert np.all(denoised > 0)

    orders = [
        {"ticker": "BAD_TICK.NS", "action": "BUY", "shares": 50, "ref_price": 100.0, "est_notional": 5000.0},
        {"ticker": "ILLIQUID.NS", "action": "BUY", "shares": 1000, "ref_price": 100.0, "est_notional": 100000.0},
    ]
    golden_map = {"BAD_TICK.NS": "QUARANTINED_DIVERGENCE", "ILLIQUID.NS": "VERIFIED_GOLDEN"}
    adv_map = {"BAD_TICK.NS": 10_000_000.0, "ILLIQUID.NS": 500_000.0}

    approved, rejected = AladdinRiskAndScenarioEngine.enforce_pre_trade_compliance_gate(
        proposed_orders=orders,
        golden_status_map=golden_map,
        adv_20d_map=adv_map,
        max_adv_participation=0.05,
    )
    assert len(rejected) == 1 and rejected[0]["ticker"] == "BAD_TICK.NS"
    assert len(approved) == 1 and approved[0]["est_notional"] <= 25000.0
    assert approved[0]["compliance_note"] == "SCALED_TO_ADV_CAP"
