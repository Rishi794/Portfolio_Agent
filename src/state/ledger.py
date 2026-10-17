"""Event-sourced portfolio ledger.

State lives in three files on the ``data-state`` branch so that every portfolio
mutation is auditable and replayable:

* ``portfolio_ledger_<market>.json``  — current snapshot (positions, cash, PnL)
* ``ledger_events.jsonl``            — append-only event log (source of truth)
* ``pending_orders_<market>.json``    — orders awaiting the ``/approve`` reconciler
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    sector: str = "UNKNOWN"
    opened_at: str = field(default_factory=_utcnow)
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        price = self.last_price or self.avg_cost
        return float(self.shares * price)


class Ledger:
    """JSON snapshot + JSONL event journal for one market."""

    def __init__(self, state_dir: str | Path, market: str, starting_cash: float = 100_000.0) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.market = market
        self.snapshot_path = self.state_dir / f"portfolio_ledger_{market.lower()}.json"
        self.events_path = self.state_dir / "ledger_events.jsonl"
        self.pending_path = self.state_dir / f"pending_orders_{market.lower()}.json"
        self.starting_cash = starting_cash

    # ------------------------------------------------------------------ state --
    def load(self) -> dict[str, Any]:
        if self.snapshot_path.exists():
            try:
                return json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return self._initial_snapshot()

    def _initial_snapshot(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "as_of": None,
            "starting_cash": self.starting_cash,
            "cash": self.starting_cash,
            "cash_buffer_target": 0.0,
            "positions": {},
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "rebalance_count": 0,
            "last_run_id": None,
        }

    def save(self, snapshot: dict[str, Any]) -> None:
        self.snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")

    # ----------------------------------------------------------------- events --
    def record_event(self, kind: str, payload: dict[str, Any], run_id: str | None = None) -> None:
        event = {
            "event_id": uuid.uuid4().hex[:12],
            "ts": _utcnow(),
            "market": self.market,
            "run_id": run_id,
            "kind": kind,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    def tail_events(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # -------------------------------------------------------------- positions --
    @staticmethod
    def positions_of(snapshot: dict[str, Any]) -> dict[str, Position]:
        return {ticker: Position(**payload) for ticker, payload in snapshot.get("positions", {}).items()}

    def mark_prices(self, snapshot: dict[str, Any], prices: dict[str, float]) -> dict[str, Any]:
        for ticker, position in snapshot.get("positions", {}).items():
            if ticker in prices and prices[ticker] == prices[ticker]:
                position["last_price"] = float(prices[ticker])
        return snapshot

    def equity(self, snapshot: dict[str, Any]) -> float:
        value = float(snapshot.get("cash", 0.0))
        for payload in snapshot.get("positions", {}).values():
            price = payload.get("last_price") or payload.get("avg_cost", 0.0)
            value += float(payload.get("shares", 0.0)) * float(price)
        return value

    def current_weights(self, snapshot: dict[str, Any]) -> dict[str, float]:
        equity = self.equity(snapshot)
        if equity <= 0:
            return {}
        out = {}
        for ticker, payload in snapshot.get("positions", {}).items():
            price = payload.get("last_price") or payload.get("avg_cost", 0.0)
            out[ticker] = float(payload.get("shares", 0.0)) * float(price) / equity
        return out

    # ------------------------------------------------------------- order queue --
    def write_pending_orders(self, orders: list[dict[str, Any]], run_id: str | None = None) -> None:
        payload = {"market": self.market, "created_at": _utcnow(), "run_id": run_id, "orders": orders}
        self.pending_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    def read_pending_orders(self) -> dict[str, Any]:
        if not self.pending_path.exists():
            return {"market": self.market, "orders": []}
        return json.loads(self.pending_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------- apply a rebalance --
    def apply_orders(
        self,
        snapshot: dict[str, Any],
        orders: list[dict[str, Any]],
        prices: dict[str, float],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply simulated/confirmed fills, updating cash, positions and realised PnL."""
        cash = float(snapshot.get("cash", self.starting_cash))
        positions = snapshot.setdefault("positions", {})
        fees = float(snapshot.get("fees_paid", 0.0))
        realized = float(snapshot.get("realized_pnl", 0.0))

        for order in orders:
            ticker = order["ticker"]
            shares = float(order.get("shares", 0.0))
            price = float(prices.get(ticker, order.get("price", 0.0)) or 0.0)
            if shares == 0.0 or price <= 0.0:
                continue
            cost_bps = float(order.get("estimated_cost_bps", 0.0))
            notional = abs(shares) * price
            fee = notional * cost_bps / 10_000.0
            fees += fee
            cash -= notional * (1.0 if shares > 0 else -1.0)
            cash -= fee

            if shares > 0:
                existing = positions.get(ticker)
                if existing:
                    total_shares = float(existing["shares"]) + shares
                    existing["avg_cost"] = (
                        float(existing["shares"]) * float(existing["avg_cost"]) + notional
                    ) / max(total_shares, 1e-9)
                    existing["shares"] = total_shares
                    existing["last_price"] = price
                else:
                    positions[ticker] = asdict(
                        Position(ticker=ticker, shares=shares, avg_cost=price, sector=order.get("sector", "UNKNOWN"), last_price=price)
                    )
            else:
                existing = positions.get(ticker)
                if existing:
                    closed = min(float(existing["shares"]), abs(shares))
                    realized += closed * (price - float(existing["avg_cost"]))
                    existing["shares"] = float(existing["shares"]) - closed
                    existing["last_price"] = price
                    if existing["shares"] <= 1e-9:
                        positions.pop(ticker, None)

        snapshot["cash"] = round(cash, 2)
        snapshot["fees_paid"] = round(fees, 2)
        snapshot["realized_pnl"] = round(realized, 2)
        snapshot["rebalance_count"] = int(snapshot.get("rebalance_count", 0)) + 1
        snapshot["last_run_id"] = run_id
        snapshot["as_of"] = dt.date.today().isoformat()
        self.save(snapshot)
        self.record_event(
            "rebalance_applied",
            {"orders": len(orders), "equity": self.equity(snapshot), "cash": snapshot["cash"]},
            run_id=run_id,
        )
        return snapshot
