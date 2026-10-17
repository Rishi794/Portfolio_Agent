"""'/approve' execution reconciler: fills, custom fills, corporate actions, idempotency."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.execution.approve_rebalance import apply_approved_orders, parse_custom_fills
from src.state.ledger import Ledger


def _seed_us_book(state: Path, *, cash: float = 10_000.0, positions: dict | None = None) -> None:
    ledger = Ledger(state, "US", starting_cash=cash)
    snapshot = ledger.load()
    snapshot["cash"] = cash
    snapshot["positions"] = positions or {}
    ledger.save(snapshot)


def _write_pending(state: Path, orders: list[dict]) -> None:
    (state / "pending_orders_us.json").write_text(
        json.dumps({"market": "US", "orders": orders}), encoding="utf-8"
    )


def test_parse_custom_fills() -> None:
    overrides = parse_custom_fills("/approve NVDA=25@128.40 RELIANCE.NS=40@2945.50")
    assert overrides == {"NVDA": (25.0, 128.40), "RELIANCE.NS": (40.0, 2945.50)}
    assert parse_custom_fills("/approve") == {}


def test_reference_fills_book_cash_positions_and_pnl(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    _seed_us_book(
        state,
        cash=10_000.0,
        positions={"MSFT": {"ticker": "MSFT", "shares": 5.0, "avg_cost": 50.0, "sector": "Technology", "last_price": 60.0}},
    )
    _write_pending(
        state,
        [
            {"market": "US", "action": "BUY", "ticker": "AAPL", "shares": 10, "ref_price": 100.0,
             "est_cost_bps": 3.0, "sector": "Technology"},
            {"market": "US", "action": "SELL", "ticker": "MSFT", "shares": 5, "ref_price": 60.0,
             "est_cost_bps": 3.0, "sector": "Technology"},
        ],
    )
    receipt = tmp_path / "receipt.md"
    result = apply_approved_orders(state, "/approve", 42, "quant-owner", receipt)
    assert result["status"] == "booked"
    assert result["orders"] == 2

    ledger = Ledger(state, "US")
    snapshot = ledger.load()
    positions = snapshot["positions"]
    assert "MSFT" not in positions
    assert positions["AAPL"]["shares"] == pytest.approx(10.0)
    assert positions["AAPL"]["avg_cost"] == pytest.approx(100.03, abs=1e-6)  # fill + 3bps fee
    assert snapshot["cash"] == pytest.approx(9299.61, abs=0.01)
    assert snapshot["fees_paid"] == pytest.approx(0.39, abs=1e-6)
    assert snapshot["realized_pnl"] == pytest.approx(49.91, abs=0.01)

    body = receipt.read_text(encoding="utf-8")
    assert "AAPL" in body and "MSFT" in body and "42" in body

    # idempotency: the queue is cleared, a duplicate /approve books nothing
    again = apply_approved_orders(state, "/approve", 42, "quant-owner", receipt)
    assert again["orders"] == 0
    assert Ledger(state, "US").load()["cash"] == pytest.approx(9299.61, abs=0.01)


def test_custom_fills_override_shares_and_price(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    _seed_us_book(state, cash=100_000.0)
    _write_pending(
        state,
        [{"market": "US", "action": "BUY", "ticker": "AAPL", "shares": 10, "ref_price": 100.0,
          "est_cost_bps": 3.0, "sector": "Technology"}],
    )
    result = apply_approved_orders(state, "/approve AAPL=12@95.5", 7, "owner", tmp_path / "r.md")
    assert result["orders"] == 1
    position = Ledger(state, "US").load()["positions"]["AAPL"]
    assert position["shares"] == pytest.approx(12.0)
    fee = 12 * 95.5 * 3.0 / 10_000.0
    assert position["avg_cost"] == pytest.approx((12 * 95.5 + fee) / 12, abs=1e-6)
    assert position["last_price"] == pytest.approx(95.5)


def test_corporate_actions_are_reconciled_once(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    lake_dir = tmp_path / "data_lake"
    lake_dir.mkdir()
    pd.DataFrame(
        [
            {"ticker": "AAPL", "market": "US", "ex_date": pd.Timestamp("2026-01-05"), "action_type": "SPLIT",
             "split_ratio": 2.0, "cash_dividend": 0.0, "knowledge_timestamp": pd.Timestamp("2026-01-05", tz="UTC")},
            {"ticker": "AAPL", "market": "US", "ex_date": pd.Timestamp("2026-01-06"), "action_type": "DIVIDEND",
             "split_ratio": 1.0, "cash_dividend": 2.0, "knowledge_timestamp": pd.Timestamp("2026-01-06", tz="UTC")},
        ]
    ).to_parquet(lake_dir / "corporate_actions.parquet", index=False)

    _seed_us_book(
        state,
        cash=1_000.0,
        positions={"AAPL": {"ticker": "AAPL", "shares": 10.0, "avg_cost": 100.0, "sector": "Technology", "last_price": 100.0}},
    )
    _write_pending(
        state,
        [{"market": "US", "action": "BUY", "ticker": "AAPL", "shares": 5, "ref_price": 50.0,
          "est_cost_bps": 3.0, "sector": "Technology"}],
    )
    result = apply_approved_orders(
        state,
        "/approve",
        11,
        "owner",
        tmp_path / "r.md",
        lake_dir=lake_dir,
        starting_cash={"US": 1_000.0},
    )
    snapshot = Ledger(state, "US", starting_cash=1_000.0).load()
    position = snapshot["positions"]["AAPL"]
    assert position["shares"] == pytest.approx(25.0)          # 10 x 2 split + 5 bought
    assert position["avg_cost"] == pytest.approx((20 * 50.0 + 250.075) / 25.0, abs=1e-4)
    # cash: +40 dividend (2.0 x 20 post-split shares) - 250.075 purchase
    assert snapshot["cash"] == pytest.approx(789.93, abs=0.02)
    assert len(snapshot["reconciled_ex_dates"]) == 2
    assert result["orders"] == 1

    # a second approval must not re-apply the split/dividend
    _write_pending(
        state,
        [{"market": "US", "action": "BUY", "ticker": "AAPL", "shares": 1, "ref_price": 50.0,
          "est_cost_bps": 3.0, "sector": "Technology"}],
    )
    apply_approved_orders(state, "/approve", 11, "owner", tmp_path / "r2.md", lake_dir=lake_dir,
                          starting_cash={"US": 1_000.0})
    snapshot = Ledger(state, "US").load()
    assert snapshot["positions"]["AAPL"]["shares"] == pytest.approx(26.0)
    assert len(snapshot["reconciled_ex_dates"]) == 2


def test_cash_cap_never_goes_negative(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    _seed_us_book(state, cash=500.0)
    _write_pending(
        state,
        [{"market": "US", "action": "BUY", "ticker": "AAPL", "shares": 100, "ref_price": 100.0,
          "est_cost_bps": 3.0, "sector": "Technology"}],
    )
    apply_approved_orders(state, "/approve", 1, "owner", tmp_path / "r.md")
    snapshot = Ledger(state, "US").load()
    assert snapshot["cash"] >= 0.0
    assert snapshot["positions"].get("AAPL", {}).get("shares", 0) == pytest.approx(4.0)


def test_no_pending_files_is_a_noop(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    result = apply_approved_orders(state, "/approve", 3, "owner", tmp_path / "r.md")
    assert result["status"] == "no_pending_files"
    assert "No `pending_orders" in (tmp_path / "r.md").read_text(encoding="utf-8")
