"""Lakehouse integrity verifier (run before publishing a GitHub Release).

Two layers of checks:

1. **Required tables** — the tables the Fan-In and the backtester actually consume
   must exist and be non-empty once the pipeline has run at least once. The
   spec-level names (``raw_ohlcv``, ``fundamentals_pit``, ``tech_evolution_pit``,
   ``policy_geopol_events``, ``corporate_actions``) are mapped onto the physical
   lakehouse files through aliases so the verifier doubles as the naming contract.
2. **Schema-driven integrity** — primary-key uniqueness, null knowledge timestamps,
   embedding widths and non-positive OHLC values (a corrupt price panel silently
   destroys every downstream estimator, so it is a hard failure).

Usage:
    uv run python -m src.storage.verify_parquet --data-dir data_lake [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .lakehouse import Lakehouse

# Spec-level table name -> physical lakehouse file stem.
REQUIRED_TABLES: dict[str, str] = {
    "raw_ohlcv": "market_prices",
    "corporate_actions": "corporate_actions",
    "fundamentals_pit": "fundamentals_pit",
    "tech_evolution_pit": "tech_evolution_pit",
    "policy_geopol_events": "policy_geopol_events",
    "macro_external_pit": "macro_external_pit",
}

_PRICE_COLUMNS = ("open", "high", "low", "close")


def verify_required_tables(lake: Lakehouse, *, require_non_empty: bool = True) -> dict[str, object]:
    """Existence / non-emptiness / OHLC sanity for the spec-mandated tables."""
    report: dict[str, object] = {}
    issues: list[str] = []
    for spec_name, physical in REQUIRED_TABLES.items():
        entry: dict[str, object] = {"physical_table": physical}
        if not lake.exists(physical):
            entry["status"] = "missing"
            issues.append(f"required table {spec_name!r} missing ({physical}.parquet)")
            report[spec_name] = entry
            continue
        frame = lake.read(physical)
        entry["rows"] = int(len(frame))
        if require_non_empty and frame.empty:
            entry["status"] = "empty"
            issues.append(f"required table {spec_name!r} is empty ({physical}.parquet)")
        else:
            entry["status"] = "ok"
        price_columns = [col for col in _PRICE_COLUMNS if col in frame.columns]
        if physical == "market_prices" and price_columns and not frame.empty:
            non_positive = {}
            for column in price_columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                bad = int((values <= 0).sum())
                if bad:
                    non_positive[column] = bad
            if non_positive:
                entry["non_positive_prices"] = non_positive
                issues.append(f"raw_ohlcv: non-positive OHLC values {non_positive}")
        report[spec_name] = entry
    return {"ok": not issues, "tables": report, "issues": issues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Parquet lakehouse integrity")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--allow-empty", action="store_true", help="Skip the non-emptiness gate (schema-only check)"
    )
    parser.add_argument(
        "--schemas-only", action="store_true", help="Skip the required-table gate (unit-test use)"
    )
    args = parser.parse_args(argv)

    lake = Lakehouse(args.data_dir)
    print(lake.describe())
    schema_report = lake.verify()

    required_report: dict[str, object] = {"ok": True, "tables": {}, "issues": []}
    if not args.schemas_only:
        required_report = verify_required_tables(lake, require_non_empty=not args.allow_empty)

    report = {
        "ok": bool(schema_report["ok"] and required_report["ok"]),
        "schema": schema_report,
        "required_tables": required_report,
    }
    print(json.dumps(report, indent=2)[:6000])
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["ok"]:
        print("FAILED: lakehouse integrity issues detected", file=sys.stderr)
        for issue in list(required_report.get("issues", [])) + list(schema_report.get("issues", [])):
            print(f"  - {issue}", file=sys.stderr)
        return 1
    print("OK: lakehouse integrity verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
