"""End-to-end synthetic pipeline: shard merge -> regime -> Löwdin -> HRP -> tickets -> reporting.

Runs entirely offline against ``src/synthetic.py`` (deterministic feature store).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting.dashboard_builder import build_dashboard_html
from src.reporting.issue_builder import build_issue_body


def test_e2e_summary_structure(e2e: dict) -> None:
    summary = e2e["summary"]
    assert summary["mode"] == "synthetic_demo"
    assert sorted(summary["markets"]) == ["INDIA", "US"]
    assert summary["lakehouse"]["ok"], summary["lakehouse"]["issues"]
    ingested = summary["ingested"]
    for table in ("market_prices", "fundamentals_pit", "tech_evolution_pit", "policy_geopol_events", "macro_external_pit"):
        assert ingested.get(table, 0) > 0, f"shard route {table} ingested nothing"
    for market in ("US", "INDIA"):
        diagnostics = summary["markets"][market]
        assert diagnostics["raw_max_abs_correlation"] > diagnostics["ortho_max_abs_correlation"] - 1e-9
        assert diagnostics["orthogonality_error"] < 1e-3
        assert diagnostics["cash_buffer"] >= 0.03
        assert 0.0 <= diagnostics["invested_weight"] <= 0.97
        assert len(diagnostics["pillar_weights"]) == len(diagnostics["pillar_names"])
        assert diagnostics["pillar_names"][:6] == [
            "kalman_trend",
            "fundamentals",
            "tech_evolution",
            "gov_policy",
            "geopolitics",
            "macro_beta",
        ]
        # Stage-3 7th pillar: Timestral is present even offline (empirical fallback),
        # and never loads the 837 MB checkpoint inside the synthetic test suite.
        assert diagnostics["pillar_names"][-1] == "timestral"
        assert diagnostics["timestral"]["mode"] in {"offline_fallback", "disabled"}
        assert abs(sum(diagnostics["pillar_weights"]) - 1.0) < 1e-6
        assert diagnostics["eligible"] > 0


def test_e2e_writes_the_lakehouse_tables(e2e: dict) -> None:
    lake = e2e["reducer"].lake
    for table in ("factor_scores", "portfolio_targets", "calibration_state", "run_manifest"):
        assert lake.exists(table), f"{table}.parquet missing"
        assert len(lake.read(table)) > 0
    targets = lake.read("portfolio_targets")
    assert set(targets["market"]) == {"US", "INDIA"}
    assert (targets["target_weight"] >= 0).all()
    assert (targets["target_weight"] <= 0.06 + 1e-9).all()
    calibration = lake.read("calibration_state")
    for scope in ("US:forensic_gate", "INDIA:forensic_gate", "US:factor_weight", "INDIA:factor_weight"):
        assert (calibration["scope"] == scope).any()
    scores = lake.read("factor_scores")
    assert "timestral" in scores.columns, "the 7th pillar must be persisted for next-run IC-IR learning"
    assert pd.to_numeric(scores["timestral"], errors="coerce").notna().any()
    assert np.isfinite(scores["composite_alpha"].to_numpy(dtype=float)).all()


def test_e2e_state_branch_artifacts(e2e: dict) -> None:
    state = e2e["state_dir"]
    for path in ("portfolio_ledger_us.json", "portfolio_ledger_india.json", "ledger_events.jsonl", "nav_history.jsonl"):
        assert (state / path).exists(), f"missing {path}"
    us = json.loads((state / "portfolio_ledger_us.json").read_text(encoding="utf-8"))
    india = json.loads((state / "portfolio_ledger_india.json").read_text(encoding="utf-8"))
    assert us["starting_cash"] == 100_000.0
    assert india["starting_cash"] == 1_000_000.0
    assert us["cash"] > 0 and india["cash"] > 0
    rows = [json.loads(line) for line in (state / "nav_history.jsonl").read_text().splitlines() if line.strip()]
    assert {row["market"] for row in rows} == {"US", "INDIA"}
    assert all({"date", "equity", "cash"} <= set(row) for row in rows)


def test_e2e_pending_orders_match_dashboard_and_approval_contract(e2e: dict) -> None:
    state = e2e["state_dir"]
    pending_files = sorted(state.glob("pending_orders*.json"))
    summary = e2e["summary"]
    total_orders = sum(len(run["tickets"]) for run in summary["runs"])
    if total_orders == 0:
        pytest.skip("synthetic run produced no trades (all targets inside their deadbands)")
    assert pending_files
    order_count = 0
    for path in pending_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["market"] in {"US", "INDIA"}
        for order in payload["orders"]:
            assert order["action"] in {"BUY", "SELL"}
            assert order["shares"] > 0
            assert order["ref_price"] > 0
            assert "est_cost_bps" in order and "sector" in order
            order_count += 1
    assert order_count == total_orders


def test_e2e_reingest_is_idempotent(e2e: dict) -> None:
    reducer = e2e["reducer"]
    before = len(reducer.lake.read("market_prices"))
    reducer.ingest_shards()  # replay the same shards (upsert semantics)
    assert len(reducer.lake.read("market_prices")) == before
    assert reducer.lake.verify()["ok"]


def test_issue_and_dashboard_render(e2e: dict, config, tmp_path: Path) -> None:
    summary = e2e["summary"]
    issue = build_issue_body(
        summary, config, tmp_path / "issue.md", lake=e2e["reducer"].lake, state_dir=e2e["state_dir"]
    )
    body = issue.read_text(encoding="utf-8")
    assert "Synthetic demo run" in body
    assert "## Approval" in body and "/approve" in body
    assert "US" in body and "INDIA" in body

    html_path = build_dashboard_html(e2e["state_dir"], tmp_path / "index.html")
    html = html_path.read_text(encoding="utf-8")
    assert "plotly" in html
    assert "Open-Aladdin" in html
    assert "__PAYLOAD__" not in html
    payload = html.split("const DATA = ", 1)[1].split(";\nconst CURRENCY", 1)[0]
    parsed = json.loads(payload)
    assert {book["market"] for book in parsed["books"]} == {"US", "INDIA"}
    for book in parsed["books"]:
        assert book["equity"] > 0
        assert book["cash"] <= book["equity"] + 1e-6


def test_v5_open_aladdin_layer_and_dashboard(e2e: dict, config, tmp_path: Path) -> None:
    """V5 layer: compliance gate → Euler/CVaR/stress telemetry → risk_attribution.json."""
    from src.main_v5 import V5OpenAladdinPipeline

    pipeline = V5OpenAladdinPipeline(e2e["reducer"], config)
    telemetry = pipeline.process(e2e["summary"])

    assert (e2e["state_dir"] / "risk_attribution.json").exists()
    assert len(telemetry["merkle_state_hash"]) == 64
    for market in ("US", "INDIA"):
        info = telemetry["markets"][market]
        assert info["port_ann_vol_pct"] >= 0.0
        assert info["daily_student_t_cvar99_pct"] >= 0.0
        # V5 forward-CVaR fusion: the reported tail is the conservative max of the
        # historical Student-t ES and the weight-weighted Timestral forward loss.
        assert info["daily_fused_cvar99_pct"] >= info["daily_student_t_cvar99_pct"] - 1e-12
        assert info["cvar_binding_source"] in {"student_t", "timestral"}
        assert isinstance(info["daily_timestral_cvar99_pct"], float)
        assert info["ibor_identity_ok"], info["ibor_discrepancy"]
        assert len(info["ledger_state_sha256"]) == 64
    # Pending-order contract survives the compliance pass (no golden shards in synthetic).
    assert telemetry["compliance_gate"]["US"]["approved"] >= 0
    for path in sorted(e2e["state_dir"].glob("pending_orders*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for order in payload["orders"]:
            assert order["action"] in {"BUY", "SELL"}
            assert float(order["shares"]) > 0.0

    html = build_dashboard_html(e2e["state_dir"], tmp_path / "v5.html").read_text(encoding="utf-8")
    assert "Open-Aladdin" in html
    assert "Euler" in html
    parsed = json.loads(html.split("const DATA = ", 1)[1].split(";\nconst CURRENCY", 1)[0])
    assert "risk" in parsed
