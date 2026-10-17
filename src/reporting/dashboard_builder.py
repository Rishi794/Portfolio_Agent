"""Static dual-market dashboard for GitHub Pages (self-contained HTML + Plotly CDN).

Reads the event-sourced state directory written by the reducer and the approval
reconciler: ``portfolio_ledger_<market>.json``, ``pending_orders_*.json``,
``ledger_events.jsonl`` and ``nav_history.jsonl``. No build step, no server, no cost.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>V4 Autonomous Dual-Market PMS (NSE/BSE + NYSE/NASDAQ)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#0b0f19; --panel:#111827; --border:#1f2937; --text:#f3f4f6; --muted:#9ca3af;
          --accent:#38bdf8; --green:#10b981; --red:#ef4444; --amber:#f59e0b; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--text); padding:24px; }
  header { display:flex; justify-content:space-between; align-items:flex-start; gap:16px;
           border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:24px; flex-wrap:wrap; }
  h1 { font-size:1.35rem; }
  .sub { color:var(--muted); font-size:.85rem; margin-top:6px; max-width:900px; line-height:1.45; }
  .badge { background:#1e293b; color:var(--accent); padding:6px 12px; border-radius:6px; font-size:.8rem; font-weight:600; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:20px; margin-bottom:24px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:20px; overflow-x:auto; }
  .card h2 { font-size:1.05rem; margin-bottom:14px; color:var(--accent); }
  .kpis { display:flex; gap:16px; margin-bottom:14px; flex-wrap:wrap; }
  .kpi { flex:1; min-width:130px; background:#0f172a; padding:12px 14px; border-radius:8px; border:1px solid var(--border); }
  .kpi span { display:block; font-size:.72rem; color:var(--muted); letter-spacing:.03em; text-transform:uppercase; }
  .kpi strong { font-size:1.25rem; margin-top:5px; display:block; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  th,td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--border); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; font-size:.75rem; text-transform:uppercase; letter-spacing:.03em; }
  .pos { color:var(--green); font-weight:600; }
  .neg { color:var(--red); font-weight:600; }
  .muted { color:var(--muted); }
  footer { color:var(--muted); font-size:.78rem; border-top:1px solid var(--border); padding-top:14px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>🌏 V4 Autonomous Portfolio Management System</h1>
    <div class="sub">
      Self-calibrating Kalman trend · PIT macro turbulence controller · ONNX/Student-t BH-FDR event
      projection onto 10-K business vectors · Forensic (Beneish/Piotroski) survival gate · Löwdin-pure
      orthogonal alpha · True Ward-linkage HRP · Corwin-Schultz + statutory KKT deadbands.
    </div>
  </div>
  <div class="badge" id="stamp"></div>
</header>
<div class="grid" id="books"></div>
<div class="grid">
  <div class="card">
    <h2>⏳ Pending recommendations (awaiting <code>/approve</code>)</h2>
    <table id="pending">
      <thead><tr><th>Market</th><th>Action</th><th>Ticker</th><th>Shares</th><th>Ref price</th>
      <th>Target wt</th><th>KKT band</th><th>Cost bps</th><th>α z</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="card">
    <h2>📜 Event-sourced audit ledger</h2>
    <table id="events">
      <thead><tr><th>Timestamp</th><th>Kind</th><th>Market</th><th>Ticker</th><th>Detail</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<footer id="footer"></footer>
<script>
const DATA = __PAYLOAD__;
const CURRENCY = { INDIA: "₹", US: "$" };

document.getElementById("stamp").textContent = "Updated " + DATA.generated_at;
document.getElementById("footer").textContent =
  "Run " + DATA.run_id + " · event-sourced ledger on the data-state branch · " +
  (DATA.run_mode === "synthetic_demo"
    ? "SYNTHETIC DEMO DATA (src/synthetic.py) — not a live portfolio."
    : "Live pipeline output.");

function money(symbol, value) {
  return symbol + Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function renderBook(book) {
  const symbol = CURRENCY[book.market] || "";
  const wrap = document.createElement("div");
  wrap.className = "card";
  wrap.innerHTML = `
    <h2>${book.market === "INDIA" ? "🇮🇳 India (NSE/BSE)" : "🇺🇸 US (NYSE/NASDAQ)"}</h2>
    <div class="kpis">
      <div class="kpi"><span>Total equity</span><strong>${money(symbol, book.equity)}</strong></div>
      <div class="kpi"><span>Cash reserve</span><strong>${money(symbol, book.cash)}</strong></div>
      <div class="kpi"><span>Cash %</span><strong>${book.cash_pct.toFixed(2)}%</strong></div>
      <div class="kpi"><span>Positions</span><strong>${book.holdings.length}</strong></div>
    </div>
    <div id="nav-${book.market}" style="height:250px;"></div>
    <table>
      <thead><tr><th>Ticker</th><th>Sector</th><th>Shares</th><th>Avg cost</th><th>Last</th>
      <th>Weight</th><th>P&L %</th></tr></thead>
      <tbody>${
        book.holdings.length === 0
          ? '<tr><td colspan="7" class="muted">100% cash — no equity positions currently held</td></tr>'
          : book.holdings.map(h => `<tr>
              <td><strong>${h.ticker}</strong></td>
              <td class="muted">${h.sector}</td>
              <td>${h.shares}</td>
              <td>${money(symbol, h.avg_cost)}</td>
              <td>${money(symbol, h.last_price)}</td>
              <td>${h.weight_pct.toFixed(2)}%</td>
              <td class="${h.pnl_pct >= 0 ? "pos" : "neg"}">${h.pnl_pct >= 0 ? "+" : ""}${h.pnl_pct.toFixed(2)}%</td>
            </tr>`).join("")
      }</tbody>
    </table>`;
  document.getElementById("books").appendChild(wrap);

  const nav = book.nav || [];
  Plotly.newPlot("nav-" + book.market, [
    { x: nav.map(r => r.date), y: nav.map(r => r.equity), name: "Equity", type: "scatter",
      line: { color: "#38bdf8", width: 2.5 } },
    { x: nav.map(r => r.date), y: nav.map(r => r.cash), name: "Cash reserve", type: "scatter",
      line: { color: "#f59e0b", width: 1.6, dash: "dot" } }
  ], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#9ca3af", size: 11 }, margin: { l: 50, r: 15, t: 10, b: 30 },
    legend: { orientation: "h", y: 1.15 }, xaxis: { gridcolor: "#1f2937" },
    yaxis: { gridcolor: "#1f2937", title: "Value" }
  }, { responsive: true, displayModeBar: false });
}

DATA.books.forEach(renderBook);

const pendingBody = document.querySelector("#pending tbody");
pendingBody.innerHTML = DATA.pending.length === 0
  ? '<tr><td colspan="9" class="muted">No pending orders — all targets are inside their KKT deadbands</td></tr>'
  : DATA.pending.map(o => `<tr>
      <td>${o.market}</td>
      <td class="${o.action === "BUY" ? "pos" : "neg"}">${o.action}</td>
      <td><strong>${o.ticker}</strong></td>
      <td>${o.shares}</td>
      <td>${Number(o.ref_price).toLocaleString()}</td>
      <td>${Number(o.target_weight_pct).toFixed(2)}%</td>
      <td>±${Number(o.kkt_deadband_pct).toFixed(2)}%</td>
      <td>${Number(o.est_cost_bps).toFixed(1)}</td>
      <td>${Number(o.alpha_z).toFixed(2)}</td>
    </tr>`).join("");

const eventsBody = document.querySelector("#events tbody");
eventsBody.innerHTML = DATA.events.length === 0
  ? '<tr><td colspan="5" class="muted">No ledger events recorded yet</td></tr>'
  : DATA.events.slice(0, 25).map(e => {
      const detail = e.kind === "rebalance_applied"
        ? `orders=${e.payload.orders} equity=${e.payload.equity} cash=${e.payload.cash}`
        : (e.kind === "corporate_action"
            ? `${e.payload.action_type} ${e.payload.ticker} ex=${e.payload.ex_date} ratio=${e.payload.split_ratio} div=${e.payload.cash_dividend}`
            : JSON.stringify(e.payload).slice(0, 120));
      return `<tr>
        <td>${(e.ts || "").slice(0, 16).replace("T", " ")}</td>
        <td>${e.kind}</td>
        <td>${e.market || ""}</td>
        <td><strong>${(e.payload && (e.payload.ticker || "")) || ""}</strong></td>
        <td class="muted">${detail}</td>
      </tr>`;
    }).join("");
</script>
</body>
</html>
"""


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _load_jsonl(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def build_dashboard_html(state_dir: str | Path, output_html: str | Path) -> Path:
    state = Path(state_dir)
    books: list[dict[str, Any]] = []
    for ledger_path in sorted(state.glob("portfolio_ledger_*.json")):
        ledger = _load_json(ledger_path, {})
        rows, equity, cash = _book_summary(ledger)
        nav = [
            row
            for row in _load_jsonl(state / "nav_history.jsonl", limit=4000)
            if row.get("market") == ledger.get("market")
        ][-1200:]
        books.append(
            {
                "market": ledger.get("market", ledger_path.stem.split("_")[-1].upper()),
                "equity": equity,
                "cash": cash,
                "cash_pct": (cash / equity * 100.0) if equity > 0 else 0.0,
                "holdings": rows,
                "nav": nav,
            }
        )

    pending: list[dict[str, Any]] = []
    for pending_path in sorted(state.glob("pending_orders*.json")):
        payload = _load_json(pending_path, {})
        orders = payload.get("orders", []) if isinstance(payload, dict) else payload
        for order in orders or []:
            pending.append(order)

    events = list(reversed(_load_jsonl(state / "ledger_events.jsonl", limit=200)))
    run_id = ""
    if events:
        run_id = str(events[0].get("run_id") or "")
    manifests = _load_jsonl(state / "nav_history.jsonl", limit=1)
    run_mode = "live" if books else "synthetic_demo"

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_id": run_id,
        "run_mode": run_mode,
        "books": books,
        "pending": pending,
        "events": events,
    }

    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=str))
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _book_summary(ledger: dict[str, Any]) -> tuple[list[dict[str, Any]], float, float]:
    cash = float(ledger.get("cash", ledger.get("cash_balance", 0.0)) or 0.0)
    positions: dict[str, dict[str, Any]] = ledger.get("positions", {}) or {}
    invested = 0.0
    rows: list[dict[str, Any]] = []
    for ticker, payload in positions.items():
        shares = float(payload.get("shares", 0.0) or 0.0)
        avg_cost = float(payload.get("avg_cost", 0.0) or 0.0)
        last_price = float(payload.get("last_price", avg_cost) or avg_cost)
        value = shares * last_price
        invested += value
        rows.append(
            {
                "ticker": ticker,
                "sector": payload.get("sector", "UNKNOWN"),
                "shares": round(shares, 4),
                "avg_cost": round(avg_cost, 2),
                "last_price": round(last_price, 2),
                "market_value": round(value, 2),
                "pnl_pct": ((last_price / avg_cost - 1.0) * 100.0) if avg_cost > 0 else 0.0,
            }
        )
    equity = cash + invested
    for row in rows:
        row["weight_pct"] = (row["market_value"] / equity * 100.0) if equity > 0 else 0.0
    rows.sort(key=lambda item: item["weight_pct"], reverse=True)
    return rows, round(equity, 2), round(cash, 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the static dual-market dashboard")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args(argv)
    path = build_dashboard_html(args.state_dir, args.output_html)
    print(f"dashboard written -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
