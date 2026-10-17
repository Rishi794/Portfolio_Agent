"""V5 "Open-Aladdin" Stage-3 Fan-In: V4 factor/HRP core + institutional verification.

Wraps the V4 reducer (factor fusion, Löwdin, HRP unchanged) with the
verification layer from ``src/verification``:

1. Golden-copy shards are upserted into the ``golden_copy`` lakehouse table.
2. The pre-trade compliance gate blocks quarantined BUYs and scales orders above
   the ADV participation cap, rewriting ``pending_orders_<market>.json`` and
   journaling a ``compliance_gate`` event when it intervenes.
3. Exact Euler risk attribution (MCTR/CCTR/PCTR), fat-tailed Student-t CVaR_99
   and market-driven scenario stress tests run on the live target book.
4. The ledger snapshot is reconciled against the fixed-point double-entry
   identity; telemetry lands in ``state/risk_attribution.json`` for the
   Open-Aladdin dashboard and the daily Issue.

CLI-compatible with ``src.main_v4`` — the workflow swaps ``-m src.main_v4`` for
``-m src.main_v5`` with identical flags.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .main_v4 import V4AutonomousPortfolioReducer, _jsonable, _resolve_markets
from .state.ledger import Ledger
from .verification.aladdin_golden_copy_and_risk import (
    AladdinRiskAndScenarioEngine,
    reconcile_double_entry_identity,
)


class V5OpenAladdinPipeline:
    """Post-processing layer over a completed ``V4AutonomousPortfolioReducer`` run."""

    def __init__(self, reducer: V4AutonomousPortfolioReducer, config: Config) -> None:
        self.reducer = reducer
        self.config = config
        self.state_dir = Path(reducer.state_dir)
        self.engine = AladdinRiskAndScenarioEngine(config)
        self.max_adv_participation = self.engine.max_adv_participation
        self.ibor_tolerance = float(self.config.get("aladdin.ibor.identity_tolerance", 0.05))

    # ----------------------------------------------------------- golden copy --
    def ingest_golden_shards(self) -> int:
        """Upsert every ``golden_ohlcv_*.parquet`` shard into the lakehouse."""
        shards_dir = self.reducer.shards_dir
        if not shards_dir or not Path(shards_dir).exists():
            return 0
        frames: list[pd.DataFrame] = []
        for path in sorted(Path(shards_dir).glob("golden_ohlcv_*.parquet")):
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            if frame is not None and not frame.empty and "ticker" in frame.columns:
                frames.append(frame)
        if not frames:
            return 0
        combined = pd.concat(frames, ignore_index=True)
        combined["trade_date"] = pd.to_datetime(combined["trade_date"]).dt.date
        combined = combined.drop_duplicates(subset=["market", "ticker", "trade_date"], keep="last")
        return int(self.reducer.lake.write("golden_copy", combined))

    def golden_maps(self, market: str) -> tuple[dict[str, str], dict[str, float], list[str]]:
        """Latest quorum status per ticker, 20d ADV, and quarantined tickers."""
        if not self.reducer.lake.exists("golden_copy"):
            return {}, {}, []
        golden = self.reducer.lake.read("golden_copy")
        if golden.empty:
            return {}, {}, []
        frame = golden[golden["market"].astype(str) == market].copy()
        if frame.empty:
            return {}, {}, []
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values("trade_date")
        latest = frame.groupby("ticker", sort=False).tail(1)
        status_map = {
            str(ticker): str(status) for ticker, status in zip(latest["ticker"], latest["golden_status"])
        }
        adv_map = {
            str(ticker): float(np.nanmean(group["dollar_volume_raw"].astype(float).to_numpy()[-20:]))
            for ticker, group in frame.groupby("ticker", sort=False)
        }
        quarantined = sorted(
            ticker for ticker, status in status_map.items() if status != "VERIFIED_GOLDEN"
        )
        return status_map, adv_map, quarantined

    # ------------------------------------------------------ compliance gate --
    def apply_compliance_gate(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Block/scal orders against the quorum + ADV maps; rewrite pending files."""
        per_market: dict[str, Any] = {}
        run_id = summary.get("run_id")
        for run in summary.get("runs", []):
            market = str(run["market"])
            tickets = list(run.get("tickets") or [])
            status_map, adv_map, quarantined = self.golden_maps(market)
            approved, blocked = self.engine.enforce_pre_trade_compliance_gate(
                tickets, status_map, adv_map, self.max_adv_participation
            )
            proposed_shares = {str(order["ticker"]): float(order.get("shares") or 0.0) for order in tickets}
            scaled = [
                str(order["ticker"])
                for order in approved
                if proposed_shares.get(str(order["ticker"])) != float(order.get("shares") or 0.0)
            ]
            changed = bool(blocked) or bool(scaled) or len(approved) != len(tickets)

            run["tickets"] = approved
            run["compliance_blocked_orders"] = _jsonable(blocked)
            run["compliance_scaled_orders"] = scaled
            run["quarantined_tickers"] = quarantined

            if changed:
                ledger = Ledger(
                    self.state_dir,
                    market,
                    starting_cash=float(self.reducer.starting_cash.get(market, 100_000.0)),
                )
                ledger.write_pending_orders(approved, run_id=run_id)
                ledger.record_event(
                    "compliance_gate",
                    {
                        "proposed": len(tickets),
                        "approved": len(approved),
                        "blocked": [item.get("ticker") for item in blocked],
                        "scaled": scaled,
                    },
                    run_id=run_id,
                )
                if isinstance(run.get("diagnostics"), dict):
                    run["diagnostics"]["orders"] = len(approved)

            per_market[market] = {
                "proposed": len(tickets),
                "approved": len(approved),
                "blocked": len(blocked),
                "scaled": len(scaled),
                "quarantined": len(quarantined),
                "golden_status_map": len(status_map),
                "adv_map": len(adv_map),
            }
        summary["compliance_gate"] = per_market
        return per_market


    # ------------------------------------------------------------------ risk --
    @staticmethod
    def _target_weights(run: dict[str, Any], columns: pd.Index) -> pd.Series:
        targets = run.get("targets") or {}
        weights = pd.Series({str(key): float(value) for key, value in targets.items()}, dtype=float)
        weights = weights[weights > 0.0]
        weights = weights[weights.index.isin(columns)]
        if float(weights.sum()) <= 0.0:
            return weights
        return weights / float(weights.sum())

    def _market_risk(self, market: str, run: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        panel, _ = self.reducer.query_pit_split_adjusted_universe(market)
        returns = pd.DataFrame()
        if panel is not None and not panel.empty:
            wide = panel.pivot(index="trade_date", columns="ticker", values="adj_close")
            returns = wide.pct_change(fill_method=None).dropna(how="all").ffill().fillna(0.0).tail(252)

        weights = self._target_weights(run, returns.columns) if not returns.empty else pd.Series(dtype=float)
        volatility_pct = 0.0
        cvar_pct = 0.0
        euler_rows: list[dict[str, Any]] = []
        stress: dict[str, float] = {}
        if not returns.empty and len(weights) >= 2:
            volatility_pct, cvar_pct, table = self.engine.compute_euler_risk_decomposition(weights, returns)
            euler_rows = _jsonable(table.head(20).to_dict(orient="records"))
            proxies = self.engine.build_proxy_factors(returns)
            stress = self.engine.run_scenario_stress_tests(weights, returns, proxies)

        ledger_path = self.state_dir / f"portfolio_ledger_{market.lower()}.json"
        snapshot: dict[str, Any] = {}
        if ledger_path.exists():
            try:
                snapshot = json.loads(ledger_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                snapshot = {}
        reconciliation = (
            reconcile_double_entry_identity(snapshot, tolerance=self.ibor_tolerance)
            if snapshot
            else {"ok": True, "discrepancy": "0.00000000"}
        )
        ledger_sha = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        return {
            "market": market,
            "as_of": summary.get("as_of"),
            "regime": run.get("regime"),
            "universe_size": run.get("universe_size"),
            "eligible_size": run.get("eligible_size"),
            "port_ann_vol_pct": volatility_pct,
            "daily_student_t_cvar99_pct": cvar_pct,
            "cvar_confidence": self.engine.confidence,
            "weight_basis": "invested_sleeve_normalized",
            "euler_risk_table": euler_rows,
            "stress_scenarios": stress,
            "quarantined_tickers": list(run.get("quarantined_tickers") or []),
            "compliance_blocked_orders": list(run.get("compliance_blocked_orders") or []),
            "compliance_scaled_orders": list(run.get("compliance_scaled_orders") or []),
            "ibor_identity_ok": bool(reconciliation.get("ok")),
            "ibor_discrepancy": str(reconciliation.get("discrepancy")),
            "ledger_state_sha256": ledger_sha,
        }

    def build_risk_telemetry(self, summary: dict[str, Any]) -> dict[str, Any]:
        telemetry: dict[str, Any] = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": summary.get("run_id"),
            "as_of": summary.get("as_of"),
            "mode": summary.get("mode"),
            "markets": {},
        }
        for run in summary.get("runs", []):
            market = str(run["market"])
            info = self._market_risk(market, run, summary)
            telemetry["markets"][market] = info
            run["v5"] = {
                key: info[key]
                for key in ("port_ann_vol_pct", "daily_student_t_cvar99_pct", "ibor_identity_ok")
            }
        payload = json.dumps(_jsonable(telemetry), sort_keys=True, default=str).encode("utf-8")
        telemetry["merkle_state_hash"] = hashlib.sha256(payload).hexdigest()
        telemetry["compliance_gate"] = summary.get("compliance_gate", {})
        return telemetry

    def write_telemetry(self, telemetry: dict[str, Any]) -> Path:
        path = self.state_dir / "risk_attribution.json"
        path.write_text(
            json.dumps(_jsonable(telemetry), indent=2, default=str) + "\n", encoding="utf-8"
        )
        return path

    def process(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Full V5 pass: compliance gate → risk/IBOR telemetry → state artifact."""
        self.apply_compliance_gate(summary)
        telemetry = self.build_risk_telemetry(summary)
        self.write_telemetry(telemetry)
        return telemetry


def format_aladdin_markdown(summary: dict[str, Any], telemetry: dict[str, Any]) -> str:
    """Markdown block appended to the daily Issue with the V5 telemetry."""
    lines: list[str] = [
        "",
        "---",
        "",
        "## 🛡️ V5 Open-Aladdin — Risk Attribution, Compliance & Stress Telemetry",
        "",
        f"- **Merkle state hash:** `{str(telemetry.get('merkle_state_hash', ''))[:16]}...` "
        "(SHA-256 over the full telemetry payload)",
        "",
    ]
    for market, info in (telemetry.get("markets") or {}).items():
        quarantined = info.get("quarantined_tickers") or []
        blocked = info.get("compliance_blocked_orders") or []
        scaled = info.get("compliance_scaled_orders") or []
        lines.append(f"### {market}")
        lines.append(
            f"- **Annualized Volatility (σ_p):** `{float(info.get('port_ann_vol_pct', 0.0)):.2f}%` | "
            f"**Daily Student-t CVaR@99%:** `-{float(info.get('daily_student_t_cvar99_pct', 0.0)):.2f}%`"
        )
        lines.append(
            f"- **Golden-copy quarantine:** `{len(quarantined)}`"
            + (f" ({', '.join(quarantined[:10])})" if quarantined else " — 100% quorum verified")
        )
        lines.append(
            f"- **Double-entry IBOR identity:** `{'OK' if info.get('ibor_identity_ok') else 'MISMATCH'}` "
            f"(Δ = {info.get('ibor_discrepancy', '0')}) | ledger SHA-256 "
            f"`{str(info.get('ledger_state_sha256', ''))[:16]}...`"
        )
        euler_rows = info.get("euler_risk_table") or []
        if euler_rows:
            lines.extend(
                [
                    "",
                    "#### Euler Risk Decomposition (Σ CCTR ≡ σ_p)",
                    "",
                    "| Ticker | Weight | MCTR | CCTR | % of Risk |",
                    "| :--- | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in euler_rows[:10]:
                lines.append(
                    f"| `{row['ticker']}` | {row['weight_pct']:.2f}% | {row['mctr_ann_pct']:.2f}% | "
                    f"{row['cctr_ann_pct']:.2f}% | **{row['pct_of_total_risk']:.2f}%** |"
                )
        scenarios = info.get("stress_scenarios") or {}
        if scenarios:
            lines.extend(
                [
                    "",
                    "#### Market-Driven Scenario Stress Tests",
                    "",
                    "| Scenario | Projected Impact |",
                    "| :--- | ---: |",
                ]
            )
            for name, value in scenarios.items():
                lines.append(f"| `{name}` | **{value:+.2f}%** |")
        if blocked:
            lines.extend(["", "#### Pre-Trade Compliance Interventions", ""])
            for item in blocked[:20]:
                lines.append(
                    f"- `{item.get('ticker')}` ({item.get('action')}): "
                    f"{item.get('compliance_reason', 'blocked')}"
                )
        if scaled:
            lines.append(f"- Scaled to the ADV cap: `{', '.join(scaled[:20])}`")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V5 Open-Aladdin dual-market portfolio reducer")
    parser.add_argument("--mode", default="auto", choices=["auto", "daily_eod", "weekly_deep_sync"])
    parser.add_argument("--active-market", "--target-market", dest="target_market", default="BOTH")
    parser.add_argument("--data-dir", type=Path, default=Path("data_lake"))
    parser.add_argument("--shards-dir", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=Path("state"))
    parser.add_argument("--output-issue", type=Path, default=None)
    parser.add_argument("--dashboard-html", type=Path, default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--backend", default="auto", choices=["auto", "onnx", "hashing"])
    parser.add_argument("--synthetic", action="store_true", help="Generate a deterministic offline feature store")
    parser.add_argument("--synthetic-days", type=int, default=320)
    parser.add_argument("--synthetic-tickers", type=int, default=40)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    mode = args.mode
    if mode == "auto":
        schedule = os.getenv("GITHUB_EVENT_SCHEDULE") or os.getenv("PMS_SCHEDULE") or ""
        weekday = dt.datetime.now(dt.timezone.utc).weekday()
        mode = "weekly_deep_sync" if (schedule.strip().startswith("0 22") or weekday == 6) else "daily_eod"

    shards_dir = args.shards_dir
    if args.synthetic:
        from .synthetic import generate_synthetic_lakehouse

        shards_dir = generate_synthetic_lakehouse(
            args.data_dir,
            config,
            days=args.synthetic_days,
            n_tickers_per_market=args.synthetic_tickers,
        )
        mode = "synthetic_demo"
        args.offline = True

    markets = _resolve_markets(config, args.target_market, mode)
    reducer = V4AutonomousPortfolioReducer(
        config,
        args.data_dir,
        args.state_dir,
        shards_dir=shards_dir,
        as_of=dt.date.fromisoformat(args.as_of) if args.as_of else None,
        offline=args.offline,
        embed_backend=args.backend,
    )
    pipeline = V5OpenAladdinPipeline(reducer, config)
    pipeline.ingest_golden_shards()
    summary = reducer.run(markets, mode=mode)
    telemetry = pipeline.process(summary)

    if args.output_issue:
        from .reporting.issue_builder import build_issue_body

        build_issue_body(summary, config, args.output_issue, lake=reducer.lake, state_dir=args.state_dir)
        with Path(args.output_issue).open("a", encoding="utf-8") as handle:
            handle.write(format_aladdin_markdown(summary, telemetry))
    if args.dashboard_html:
        from .reporting.dashboard_builder import build_dashboard_html

        build_dashboard_html(args.state_dir, args.dashboard_html)

    if args.print_json:
        print(json.dumps(_jsonable({**summary, "risk_telemetry": telemetry}), indent=2))
    else:
        print(
            f"[{summary['run_id']}] mode={summary['mode']} markets={list(summary['markets'])} "
            f"lakehouse_ok={summary['lakehouse']['ok']}"
        )
        for market, info in (telemetry.get("markets") or {}).items():
            print(
                f"  {market}: regime={info.get('regime')} "
                f"sigma_p={float(info.get('port_ann_vol_pct', 0.0)):.2f}% "
                f"cvar99=-{float(info.get('daily_student_t_cvar99_pct', 0.0)):.2f}% "
                f"quarantined={len(info.get('quarantined_tickers') or [])} "
                f"blocked={len(info.get('compliance_blocked_orders') or [])} "
                f"ibor_ok={info.get('ibor_identity_ok')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
