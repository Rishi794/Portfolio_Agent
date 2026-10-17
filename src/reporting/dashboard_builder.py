"""V5 Open-Aladdin dashboard: static GitHub Pages analytics portal (Plotly CDN).

Reads the event-sourced state directory written by the reducer and the approval
reconciler — ``portfolio_ledger_<market>.json``, ``pending_orders_*.json``,
``ledger_events.jsonl``, ``nav_history.jsonl`` and ``risk_attribution.json`` —
and emits one self-contained HTML file. No build step, no server, no cost.

Panels: dual-market books (equity / cash / σp / CVaR99 / Merkle badge),
walk-forward NAV, Euler risk decomposition (capital weight vs % of total risk),
market-driven scenario stress tests, pre-trade-approved pending orders and the
cryptographic ledger audit trail.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _load_jsonl(path: Path, limit: int = 400) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]


def _book_summary(
    ledger: dict[str, Any],
    market: str,
    currency: str,
    nav_rows: list[dict[str, Any]],
    risk_markets: dict[str, Any],
) -> dict[str, Any]:
    cash = float(ledger.get("cash_balance", ledger.get("cash", 0.0)) or 0.0)
    positions: list[dict[str, Any]] = []
    invested = 0.0
    for ticker, payload in (ledger.get("positions") or {}).items():
        shares = float(payload.get("shares", 0.0) or 0.0)
        avg_cost = float(payload.get("avg_cost", 0.0) or 0.0)
        last_price = float(payload.get("last_price") or avg_cost or 0.0)
        market_value = shares * last_price
        invested += market_value
        positions.append(
            {
                "ticker": ticker,
                "sector": payload.get("sector", "UNKNOWN"),
                "shares": shares,
                "avg_cost": round(avg_cost, 2),
                "last_price": round(last_price, 2),
                "market_value": round(market_value, 2),
                "pnl_pct": round((last_price / avg_cost - 1.0) * 100.0, 2) if avg_cost > 0 else 0.0,
            }
        )
    equity = cash + invested
    for position in positions:
        position["weight_pct"] = round(position["market_value"] / equity * 100.0, 2) if equity > 0 else 0.0
    positions.sort(key=lambda item: item["market_value"], reverse=True)

    nav: list[dict[str, Any]] = []
    for row in nav_rows:
        if str(row.get("market") or market) != market:
            continue
        nav.append(
            {
                "date": str(row.get("date") or row.get("as_of") or row.get("ts") or ""),
                "equity": float(row.get("equity", row.get("portfolio_nav", row.get("nav", 0.0))) or 0.0),
                "cash": float(row.get("cash", 0.0) or 0.0),
            }
        )

    return {
        "market": market,
        "currency": currency,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "invested": round(invested, 2),
        "cash_pct": round(cash / equity * 100.0, 2) if equity > 0 else 100.0,
        "positions": positions,
        "nav": nav,
        "risk": risk_markets.get(market) or {},
    }


def build_payload(state_dir: Path) -> dict[str, Any]:
    risk_doc = _load_json(state_dir / "risk_attribution.json", {})
    risk_markets = risk_doc.get("markets", {}) if isinstance(risk_doc, dict) else {}
    nav_rows = _load_jsonl(state_dir / "nav_history.jsonl", limit=1200)

    ledger_us = _load_json(state_dir / "portfolio_ledger_us.json", {"cash": 100_000.0, "positions": {}})
    ledger_in = _load_json(state_dir / "portfolio_ledger_india.json", {"cash": 1_000_000.0, "positions": {}})
    books = [
        _book_summary(ledger_us, "US", "USD", nav_rows, risk_markets),
        _book_summary(ledger_in, "INDIA", "INR", nav_rows, risk_markets),
    ]

    pending: list[dict[str, Any]] = []
    for path in sorted(state_dir.glob("pending_orders*.json")):
        payload = _load_json(path, {}) or {}
        for order in payload.get("orders", []) or []:
            pending.append({**order, "market": order.get("market") or payload.get("market") or ""})

    events = list(reversed(_load_jsonl(state_dir / "ledger_events.jsonl", limit=200)))[:60]
    has_state = any(book["equity"] > 0 for book in books) or bool(pending) or bool(events)
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run_mode": "live" if has_state else "synthetic_demo",
        "merkle_state_hash": risk_doc.get("merkle_state_hash") if isinstance(risk_doc, dict) else None,
        "books": books,
        "pending": pending,
        "events": events,
        "risk": risk_markets,
    }


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>V5 Open-Aladdin Autonomous Dual-Market PMS (NSE/BSE + NYSE/NASDAQ)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#070a12; --panel:#0f1626; --border:#1e293b; --text:#f8fafc; --muted:#94a3b8;
          --accent:#38bdf8; --gold:#f59e0b; --green:#10b981; --red:#ef4444; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--text); padding:22px; }
  header { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
           border-bottom:1px solid var(--border); padding-bottom:14px; margin-bottom:20px; }
  h1 { font-size:1.25rem; }
  .sub { color:var(--muted); font-size:.85rem; margin-top:4px; }
  .badge { background:#1e293b; color:var(--accent); padding:5px 10px; border-radius:6px;
           font-size:.78rem; font-family:ui-monospace,monospace; }
  .grid-2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(470px,1fr)); gap:18px; margin-bottom:20px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:18px; }
  .card h2 { font-size:1.02rem; margin-bottom:10px; color:var(--accent); display:flex;
             justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }
  .kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:12px; }
  .kpi { background:#090d16; padding:9px 11px; border-radius:8px; border:1px solid var(--border); }
  .kpi span { display:block; font-size:.72rem; color:var(--muted); }
  .kpi strong { font-size:1.08rem; margin-top:3px; display:block; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; margin-top:8px; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; }
  .pos { color:var(--green); font-weight:600; }
  .neg { color:var(--red); font-weight:600; }
  .muted { color:var(--muted); }
  footer { color:var(--muted); font-size:.8rem; margin-top:8px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>🛡️ V5 Open-Aladdin Autonomous Portfolio &amp; Risk Network</h1>
    <p class="sub">3-source Golden Copy quorum · Decimal double-entry IBOR · Euler risk attribution (MCTR/CCTR/PCTR) · Student-t CVaR<sub>99</sub> · MDS stress engine</p>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span class="badge" id="ts-badge">loading…</span>
    <span class="badge" id="merkle-badge"></span>
  </div>
</header>

<div class="grid-2">
  <div class="card" id="card-us">
    <h2><span>🇺🇸 US Book (NYSE / NASDAQ — USD)</span><span class="badge" id="us-merkle"></span></h2>
    <div class="kpi-row">
      <div class="kpi"><span>Total Equity</span><strong id="us-eq">—</strong></div>
      <div class="kpi"><span>Cash</span><strong id="us-cash">—</strong></div>
      <div class="kpi"><span>Ann. Vol (σp)</span><strong id="us-vol">—</strong></div>
      <div class="kpi"><span>Daily CVaR 99%</span><strong id="us-cvar" class="neg">—</strong></div>
    </div>
    <div id="nav-us" style="height:210px;"></div>
    <div id="euler-us" style="height:210px;margin-top:10px;"></div>
    <div id="stress-us" style="height:190px;margin-top:10px;"></div>
    <table><thead><tr><th>Ticker</th><th>Sector</th><th>Shares</th><th>Last Px</th><th>Weight</th><th>PnL %</th></tr></thead>
      <tbody id="pos-us"></tbody></table>
  </div>

  <div class="card" id="card-india">
    <h2><span>🇮🇳 India Book (NSE / BSE — INR)</span><span class="badge" id="india-merkle"></span></h2>
    <div class="kpi-row">
      <div class="kpi"><span>Total Equity</span><strong id="india-eq">—</strong></div>
      <div class="kpi"><span>Cash</span><strong id="india-cash">—</strong></div>
      <div class="kpi"><span>Ann. Vol (σp)</span><strong id="india-vol">—</strong></div>
      <div class="kpi"><span>Daily CVaR 99%</span><strong id="india-cvar" class="neg">—</strong></div>
    </div>
    <div id="nav-india" style="height:210px;"></div>
    <div id="euler-india" style="height:210px;margin-top:10px;"></div>
    <div id="stress-india" style="height:190px;margin-top:10px;"></div>
    <table><thead><tr><th>Ticker</th><th>Sector</th><th>Shares</th><th>Last Px</th><th>Weight</th><th>PnL %</th></tr></thead>
      <tbody id="pos-india"></tbody></table>
  </div>
</div>

<div class="grid-2">
  <div class="card">
    <h2>⏳ Pre-Trade Compliance — Approved Orders (awaiting '/approve')</h2>
    <table><thead><tr><th>Market</th><th>Action</th><th>Ticker</th><th>Shares</th><th>Ref Px</th><th>Target Wt</th><th>Deadband</th><th>α z</th><th>Note</th></tr></thead>
      <tbody id="tbl-pending"></tbody></table>
  </div>
  <div class="card">
    <h2>📜 Cryptographic IBOR Audit Trail (fills, corporate actions, compliance)</h2>
    <table><thead><tr><th>UTC</th><th>Kind</th><th>Market</th><th>Detail</th></tr></thead>
      <tbody id="tbl-events"></tbody></table>
  </div>
</div>

<footer id="footer"></footer>

<script>
const DATA = __PAYLOAD__;
const CURRENCY = {"USD": "$", "INR": "₹"};
__SCRIPT__
</script>
</body>
</html>"""


SCRIPT = """
const fmt = (curr, value) => curr + Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
const layout = (title) => ({
  title: {text: title, font: {color: '#94a3b8', size: 12}},
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {color: '#94a3b8', size: 11},
  margin: {l: 46, r: 14, t: 30, b: 30},
  xaxis: {gridcolor: '#1e293b'}, yaxis: {gridcolor: '#1e293b'},
  legend: {orientation: 'h', y: 1.18}
});

document.getElementById('ts-badge').textContent = 'Golden Copy Sync: ' + DATA.generated_at;
if (DATA.merkle_state_hash) {
  document.getElementById('merkle-badge').textContent = 'MERKLE: ' + DATA.merkle_state_hash.slice(0, 16);
}
document.getElementById('footer').textContent = DATA.run_mode === 'live'
  ? 'Live dual-market state from the data-state branch · rebuilt on every pipeline and /approve run.'
  : 'Synthetic demo (offline) — no live state found in this directory.';

function renderBook(prefix, book) {
  const curr = CURRENCY[book.currency] || '';
  const risk = book.risk || {};
  document.getElementById(prefix + '-eq').textContent = fmt(curr, book.equity);
  document.getElementById(prefix + '-cash').textContent =
    fmt(curr, book.cash) + ' (' + Number(book.cash_pct).toFixed(1) + '%)';
  document.getElementById(prefix + '-vol').textContent =
    (risk.port_ann_vol_pct !== undefined ? Number(risk.port_ann_vol_pct).toFixed(2) : '—') + '%';
  const cvar = risk.daily_fused_cvar99_pct !== undefined
    ? risk.daily_fused_cvar99_pct : risk.daily_student_t_cvar99_pct;
  document.getElementById(prefix + '-cvar').textContent =
    cvar !== undefined ? '-' + Number(cvar).toFixed(2) + '%' : '—';
  document.getElementById(prefix + '-merkle').textContent =
    risk.ledger_state_sha256 ? 'LEDGER: ' + risk.ledger_state_sha256.slice(0, 12) : 'LEDGER: genesis';

  const nav = book.nav || [];
  const navTraces = nav.length ? [
    {x: nav.map(r => r.date), y: nav.map(r => r.equity), name: 'Equity', type: 'scatter', mode: 'lines',
     line: {color: '#38bdf8', width: 2.2}},
    {x: nav.map(r => r.date), y: nav.map(r => r.cash), name: 'Cash', type: 'scatter', mode: 'lines',
     line: {color: '#64748b', width: 1.3, dash: 'dot'}}
  ] : [];
  Plotly.newPlot('nav-' + prefix, navTraces, layout('Walk-Forward Equity & Cash'),
                 {responsive: true, displayModeBar: false});

  const euler = (risk.euler_risk_table || []).slice(0, 9);
  const eulerTraces = euler.length ? [
    {x: euler.map(r => r.ticker), y: euler.map(r => r.weight_pct), name: 'Capital weight %',
     type: 'bar', marker: {color: '#38bdf8'}},
    {x: euler.map(r => r.ticker), y: euler.map(r => r.pct_of_total_risk), name: 'Euler risk share (PCTR %)',
     type: 'bar', marker: {color: '#f59e0b'}}
  ] : [];
  Plotly.newPlot('euler-' + prefix, eulerTraces,
                 Object.assign(layout('Euler Risk Decomposition (weight vs % of total risk)'), {barmode: 'group'}),
                 {responsive: true, displayModeBar: false});

  const scenarios = Object.keys(risk.stress_scenarios || {});
  const values = scenarios.map(k => risk.stress_scenarios[k]);
  Plotly.newPlot('stress-' + prefix, scenarios.length ? [{
    x: scenarios, y: values, type: 'bar',
    marker: {color: values.map(v => v >= 0 ? '#10b981' : '#ef4444')}
  }] : [], layout('Market-Driven Scenario Stress Drawdown (%)'),
                 {responsive: true, displayModeBar: false});

  const positions = book.positions || [];
  document.getElementById('pos-' + prefix).innerHTML = positions.length === 0
    ? '<tr><td colspan="6" class="muted">100% cash reserve — awaiting the next rebalance</td></tr>'
    : positions.map(p => `
      <tr>
        <td><strong>${p.ticker}</strong></td>
        <td class="muted">${p.sector || ''}</td>
        <td>${p.shares}</td>
        <td>${curr}${Number(p.last_price).toLocaleString()}</td>
        <td>${p.weight_pct}%</td>
        <td class="${p.pnl_pct >= 0 ? 'pos' : 'neg'}">${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct}%</td>
      </tr>`).join('');
}

const EMPTY_BOOK = {positions: [], nav: [], cash: 0, equity: 0, cash_pct: 100, currency: 'USD', risk: {}};
renderBook('us', DATA.books.find(b => b.market === 'US') || EMPTY_BOOK);
renderBook('india', DATA.books.find(b => b.market === 'INDIA') || EMPTY_BOOK);

const pending = DATA.pending || [];
document.getElementById('tbl-pending').innerHTML = pending.length === 0
  ? '<tr><td colspan="9" class="muted">All positions inside their KKT deadbands and compliance limits</td></tr>'
  : pending.map(o => `
    <tr>
      <td>${o.market || ''}</td>
      <td class="${o.action === 'BUY' ? 'pos' : 'neg'}">${o.action}</td>
      <td><strong>${o.ticker}</strong></td>
      <td>${o.shares}</td>
      <td>${o.ref_price}</td>
      <td>${o.target_weight_pct}%</td>
      <td>±${o.kkt_deadband_pct}%</td>
      <td>${o.alpha_z}</td>
      <td class="muted">${o.compliance_note || ''}</td>
    </tr>`).join('');

const events = DATA.events || [];
document.getElementById('tbl-events').innerHTML = events.length === 0
  ? '<tr><td colspan="4" class="muted">No ledger events recorded yet</td></tr>'
  : events.map(e => `
    <tr>
      <td>${String(e.ts || '').slice(0, 16).replace('T', ' ')}</td>
      <td>${e.kind || ''}</td>
      <td>${e.market || ''}</td>
      <td class="muted">${JSON.stringify(e.payload || {}).slice(0, 170)}</td>
    </tr>`).join('');
"""


def build_dashboard_html(state_dir: str | Path, output_html: str | Path) -> Path:
    """Render the Open-Aladdin portal; returns the written path."""
    state_path = Path(state_dir)
    output_path = Path(output_html)
    payload = build_payload(state_path)
    html_doc = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=str)).replace(
        "__SCRIPT__", SCRIPT
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V5 Open-Aladdin static dashboard builder")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args(argv)
    path = build_dashboard_html(args.state_dir, args.output_html)
    print(f"dashboard written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
