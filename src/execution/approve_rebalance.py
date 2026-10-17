"""Reconcile an ``/approve`` comment into the event-sourced ledger.

Two execution modes:

1. ``/approve`` — book every pending order at its reference price plus the estimated
   Corwin-Schultz half-spread and statutory exchange fees (India: STT + stamp duty +
   exchange + SEBI + 18% GST ≈ 11.9 bps; US: SEC §31 + FINRA TAF ≈ 0.3 bps).
2. ``/approve NVDA=25@128.40 RELIANCE.NS=40@2945.50`` — book real broker fills with
   exact share counts and prices (weighted-average cost on BUY, realised PnL on SELL).

Corporate actions are reconciled first: unreconciled splits multiply the share count
and divide the cost basis; unreconciled dividends are credited to cash. Both are
recorded in ``reconciled_ex_dates`` so replaying an approval cannot double-apply them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..state.ledger import Ledger

FILL_PATTERN = re.compile(r"([A-Za-z0-9._\-]+)\s*=\s*(\d+(?:\.\d+)?)\s*@\s*(\d+(?:\.\d+)?)")


def parse_custom_fills(comment_body: str) -> dict[str, tuple[float, float]]:
    """Parse ``TICKER=shares@price`` overrides from an ``/approve`` comment."""
    overrides: dict[str, tuple[float, float]] = {}
    for match in FILL_PATTERN.finditer(comment_body or ""):
        overrides[match.group(1).upper()] = (float(match.group(2)), float(match.group(3)))
    return overrides


def _pending_files(state_dir: Path) -> list[Path]:
    return sorted(state_dir.glob("pending_orders*.json"))


def _load_orders(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, []
    if isinstance(payload, list):
        return None, payload
    return payload.get("market"), payload.get("orders", []) or []


def apply_corporate_actions(
    ledger: Ledger,
    snapshot: dict[str, Any],
    actions: pd.DataFrame,
    market: str,
    *,
    as_of: dt.date | None = None,
) -> int:
    """Apply unreconciled splits/dividends to the book; returns the number applied."""
    if actions is None or actions.empty:
        return 0
    frame = actions.copy()
    frame = frame[frame["market"].astype(str) == market]
    if frame.empty:
        return 0
    frame["ex_date"] = pd.to_datetime(frame["ex_date"]).dt.date
    today = as_of or dt.date.today()
    frame = frame[frame["ex_date"] <= today]
    reconciled = set(snapshot.get("reconciled_ex_dates", []) or [])
    positions: dict[str, dict[str, Any]] = snapshot.setdefault("positions", {})
    cash = float(snapshot.get("cash", 0.0) or 0.0)
    applied = 0

    for row in frame.sort_values("ex_date").itertuples(index=False):
        key = f"{row.ticker}:{row.ex_date}:{row.action_type}"
        if key in reconciled:
            continue
        position = positions.get(row.ticker)
        if position is None:
            reconciled.add(key)
            continue
        shares = float(position.get("shares", 0.0) or 0.0)
        ratio = float(getattr(row, "split_ratio", 1.0) or 1.0)
        dividend = float(getattr(row, "cash_dividend", 0.0) or 0.0)
        if ratio and ratio > 0 and ratio != 1.0:
            position["shares"] = shares * ratio
            avg_cost = float(position.get("avg_cost", 0.0) or 0.0)
            position["avg_cost"] = round(avg_cost / ratio, 6)
            ledger.record_event(
                "corporate_action",
                {"ticker": row.ticker, "action_type": "SPLIT", "ex_date": str(row.ex_date), "split_ratio": ratio,
                 "shares_before": shares, "shares_after": position["shares"]},
            )
            applied += 1
        if dividend > 0:
            credit = shares * dividend
            cash += credit
            ledger.record_event(
                "corporate_action",
                {"ticker": row.ticker, "action_type": "DIVIDEND", "ex_date": str(row.ex_date),
                 "cash_dividend": dividend, "credited": round(credit, 2)},
            )
            applied += 1
        reconciled.add(key)

    snapshot["cash"] = round(cash, 2)
    snapshot["reconciled_ex_dates"] = sorted(reconciled)[-500:]
    return applied


def apply_approved_orders(
    state_dir: str | Path,
    comment_body: str,
    issue_number: int,
    actor: str,
    receipt_path: str | Path,
    *,
    lake_dir: str | Path | None = None,
    starting_cash: dict[str, float] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    state = Path(state_dir)
    pending_files = _pending_files(state)
    receipt = Path(receipt_path)
    if not pending_files:
        receipt.write_text("⚠️ No `pending_orders*.json` found on the `data-state` branch.\n", encoding="utf-8")
        return {"status": "no_pending_files", "orders": 0}

    overrides = parse_custom_fills(comment_body)
    starting_cash = starting_cash or {"US": 100_000.0, "INDIA": 1_000_000.0}
    actions = _load_actions(lake_dir)

    totals = {"orders": 0, "fees": 0.0, "realized_pnl": 0.0, "markets": []}
    lines = [
        f"## ✅ Rebalance booked — issue #{issue_number} approved by @{actor}",
        "",
        f"Comment: `{(comment_body or '').strip()[:180]}`",
        "",
        f"Reference-forced fills parsed: **{len(overrides)}**"
        + (f" → {', '.join(f'`{t}`' for t in sorted(overrides))}" if overrides else " (booking at reference prices)"),
        "",
    ]

    for path in pending_files:
        market, orders = _load_orders(path)
        if not orders:
            path.write_text(json.dumps({"market": market, "orders": []}, indent=2) + "\n", encoding="utf-8")
            continue
        market = market or ("INDIA" if "india" in path.stem else "US")
        ledger = Ledger(state, market, starting_cash=starting_cash.get(market, 100_000.0))
        snapshot = ledger.load()

        applied_actions = apply_corporate_actions(ledger, snapshot, actions, market)
        if applied_actions:
            lines.append(f"- reconciled **{applied_actions}** corporate action(s) for `{market}` before fills")

        cash = float(snapshot.get("cash", 0.0) or 0.0)
        positions: dict[str, dict[str, Any]] = snapshot.setdefault("positions", {})
        currency = "₹" if market == "INDIA" else "$"
        lines.extend(
            [
                "",
                f"### {market} (`{currency}`)",
                "",
                "| Action | Ticker | Shares | Fill price | Fees | Realised PnL | Position after |",
                "| :---: | :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        realized_total = 0.0
        fee_total = 0.0
        booked = 0
        ordered = sorted(orders, key=lambda order: 0 if str(order.get("action", "")).upper() == "SELL" else 1)

        for order in ordered:
            ticker = str(order.get("ticker", "")).upper()
            action = str(order.get("action", "")).upper()
            if not ticker or action not in ("BUY", "SELL"):
                continue
            if ticker in overrides:
                shares, fill_price = overrides[ticker]
            else:
                shares = float(order.get("shares", 0.0) or 0.0)
                fill_price = float(order.get("ref_price", 0.0) or 0.0)
            if shares <= 0 or fill_price <= 0:
                continue

            cost_bps = float(order.get("est_cost_bps", 11.88 if market == "INDIA" else 3.0) or 0.0)
            position = positions.get(ticker, {"shares": 0.0, "avg_cost": 0.0, "sector": order.get("sector", "UNKNOWN")})
            held = float(position.get("shares", 0.0) or 0.0)
            avg_cost = float(position.get("avg_cost", 0.0) or 0.0)

            if action == "SELL":
                sell_shares = min(shares, held)
                if sell_shares <= 0:
                    lines.append(f"| SELL | `{ticker}` | 0 | — | — | — | skipped: no position |")
                    continue
                gross = sell_shares * fill_price
                fee = gross * cost_bps / 10_000.0
                cash += gross - fee
                realized = sell_shares * (fill_price - avg_cost) - fee
                realized_total += realized
                fee_total += fee
                new_shares = held - sell_shares
                if new_shares <= 1e-9:
                    positions.pop(ticker, None)
                else:
                    position["shares"] = new_shares
                    position["last_price"] = fill_price
                    positions[ticker] = position
                booked_shares = sell_shares
            else:
                gross = shares * fill_price
                fee = gross * cost_bps / 10_000.0
                if gross + fee > cash:  # cash cap, never allow a negative balance
                    shares = int(cash // (fill_price * (1.0 + cost_bps / 10_000.0)))
                    if shares <= 0:
                        lines.append(f"| BUY | `{ticker}` | 0 | — | — | — | skipped: insufficient cash |")
                        continue
                    gross = shares * fill_price
                    fee = gross * cost_bps / 10_000.0
                cash -= gross + fee
                fee_total += fee
                new_shares = held + shares
                position["avg_cost"] = round((held * avg_cost + gross + fee) / new_shares, 6)
                position["shares"] = new_shares
                position["last_price"] = fill_price
                position.setdefault("sector", order.get("sector", "UNKNOWN"))
                positions[ticker] = position
                realized = 0.0
                booked_shares = shares

            booked += 1
            lines.append(
                f"| {action} | `{ticker}` | {booked_shares:.0f} | {fill_price:,.2f} | {fee:,.2f} | "
                f"{realized:+,.2f} | {float(positions.get(ticker, {}).get('shares', 0.0)):.0f} |"
            )
            ledger.record_event(
                "order_fill",
                {
                    "ticker": ticker,
                    "action": action,
                    "filled_shares": booked_shares,
                    "fill_price": round(fill_price, 4),
                    "fee": round(fee, 2),
                    "realized_pnl": round(realized, 2),
                    "post_trade_shares": float(positions.get(ticker, {}).get("shares", 0.0)),
                    "post_trade_cash": round(cash, 2),
                    "issue_number": issue_number,
                    "approved_by": actor,
                    "market": market,
                },
                run_id=run_id,
            )

        snapshot["cash"] = round(cash, 2)
        snapshot["fees_paid"] = round(float(snapshot.get("fees_paid", 0.0) or 0.0) + fee_total, 2)
        snapshot["realized_pnl"] = round(float(snapshot.get("realized_pnl", 0.0) or 0.0) + realized_total, 2)
        snapshot["as_of"] = dt.date.today().isoformat()
        snapshot["last_run_id"] = run_id or snapshot.get("last_run_id")
        ledger.save(snapshot)
        lines.append("")
        lines.append(
            f"**{market} after approval:** cash `{currency}{cash:,.2f}` · equity `{currency}{ledger.equity(snapshot):,.2f}` · "
            f"positions {len(positions)} · fees booked `{currency}{fee_total:,.2f}`"
        )

        # idempotency: the queue is cleared so a duplicate '/approve' books nothing
        path.write_text(json.dumps({"market": market, "orders": [], "booked_at": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")

        totals["orders"] += booked
        totals["fees"] += fee_total
        totals["realized_pnl"] += realized_total
        totals["markets"].append(market)

    receipt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "booked", **totals}


def _load_actions(lake_dir: str | Path | None) -> pd.DataFrame:
    if not lake_dir:
        return pd.DataFrame()
    path = Path(lake_dir) / "corporate_actions.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile an /approve comment into the ledger")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--comment-body", required=True)
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--lake-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    result = apply_approved_orders(
        args.state_dir,
        args.comment_body,
        args.issue_number,
        args.actor,
        args.receipt_out,
        lake_dir=args.lake_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
