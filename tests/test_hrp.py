"""True Ward-linkage HRP, alpha tilt, cap, turnover limiter and order generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.hrp_optimizer import HRPOptimizer


def _returns(n_names: int, rows: int = 400, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.01, rows)
    data = {}
    for idx in range(n_names):
        idio = rng.normal(0.0, 0.01 + 0.001 * idx, rows)
        data[f"T{idx:02d}"] = 0.8 * market + idio
    return pd.DataFrame(data)


def test_hrp_weights_respect_cap_and_cash_buffer(config) -> None:
    optimizer = HRPOptimizer(config)
    returns = _returns(30)
    alpha = pd.Series(np.linspace(1.0, -1.0, 30), index=returns.columns)
    weights = optimizer.optimize(returns, alpha, cash_buffer=0.10)
    invested = 0.90
    cap = float(config.e("optimizer.max_weight", 0.06)) * invested
    assert weights.sum() == pytest.approx(invested, abs=1e-6)  # 30 names => cap is feasible
    assert weights.max() <= cap + 1e-9
    assert (weights >= 0).all()
    assert len(weights) == 30
    # higher alpha must tilt toward positive scores on the aggregate
    assert float((weights * alpha.reindex(weights.index)).sum()) > 0


def test_infeasible_cap_leaves_the_remainder_in_cash(config) -> None:
    optimizer = HRPOptimizer(config)
    returns = _returns(8)
    alpha = pd.Series(1.0, index=returns.columns)
    weights = optimizer.optimize(returns, alpha, cash_buffer=0.10)
    cap = float(config.e("optimizer.max_weight", 0.06)) * 0.90
    assert weights.max() == pytest.approx(cap, abs=1e-9)
    assert weights.sum() == pytest.approx(8 * cap, abs=1e-9)  # hard cap => residue stays in cash
    assert weights.sum() < 0.90


def test_turnover_limiter_bounds_trading(config) -> None:
    optimizer = HRPOptimizer(config)
    returns = _returns(20)
    alpha = pd.Series(np.linspace(2.0, -2.0, 20), index=returns.columns)
    current = {ticker: (0.95 / 20 if idx % 2 == 0 else 0.0) for idx, ticker in enumerate(returns.columns)}
    limited = optimizer.optimize(returns, alpha, cash_buffer=0.05, current_weights=current)
    universe = sorted(set(limited.index) | set(current))
    current_vec = pd.Series({ticker: float(current.get(ticker, 0.0)) for ticker in universe})
    turnover = float((limited.reindex(universe).fillna(0.0) - current_vec).abs().sum())
    assert turnover <= float(config.e("optimizer.max_turnover", 0.35)) + 1e-6
    # the limiter never renormalises past the constraint: a partially-invested book
    # stays partially invested after the move
    assert 0.0 < limited.sum() <= 0.95 + 1e-6


def test_ward_seriation_is_a_permutation(config) -> None:
    optimizer = HRPOptimizer(config)
    returns = _returns(25, seed=3)
    order = optimizer.ward_seriation(returns)
    assert sorted(order) == sorted(returns.columns)
    # correlated twins must end up adjacent in the quasi-diagonal order
    returns["T00_TWIN"] = returns["T00"] * 0.999 + 0.0001
    order2 = optimizer.ward_seriation(returns)
    assert abs(order2.index("T00") - order2.index("T00_TWIN")) == 1


def test_generate_orders_respects_deadband_and_exits(config) -> None:
    optimizer = HRPOptimizer(config)
    target = pd.Series({"KEEP": 0.10, "ADD": 0.12, "DROP": 0.0})
    current = {"KEEP": 0.105, "ADD": 0.03, "DROP": 0.08}
    prices = {"KEEP": 100.0, "ADD": 50.0, "DROP": 20.0}
    deadbands = {"KEEP": 100.0, "ADD": 0.01, "DROP": 0.01}  # KEEP's band suppresses the drift
    tickets = optimizer.generate_orders(
        market="US",
        target_weights=target,
        current_weights=current,
        prices=prices,
        deadbands=deadbands,
        costs_bps={"KEEP": 4.0, "ADD": 4.0, "DROP": 4.0},
        alpha=pd.Series({"KEEP": 0.2, "ADD": 0.4, "DROP": -0.3}),
        sectors={"KEEP": "Technology", "ADD": "Energy", "DROP": "Financials"},
        equity=100_000.0,
    )
    by_ticker = {ticket.ticker: ticket for ticket in tickets}
    assert "KEEP" not in by_ticker  # inside the KKT band -> no trade
    assert by_ticker["ADD"].action == "BUY"
    assert by_ticker["ADD"].shares == pytest.approx(int((0.12 - 0.03) * 100_000 / 50.0))
    assert by_ticker["DROP"].action == "SELL"
    assert by_ticker["DROP"].reason == "full_exit"
    assert by_ticker["DROP"].shares == pytest.approx(int(0.08 * 100_000 / 20.0))


def test_generate_orders_skips_unpriced_and_micro_notional(config) -> None:
    optimizer = HRPOptimizer(config)
    tickets = optimizer.generate_orders(
        market="INDIA",
        target_weights=pd.Series({"A": 0.001, "B": 0.20}),
        current_weights={},
        prices={"A": 100.0, "B": 0.0},
        deadbands={"A": 0.0, "B": 0.0},
        costs_bps={},
        alpha=pd.Series({"A": 1.0, "B": 1.0}),
        equity=100_000.0,
    )
    assert tickets == []
