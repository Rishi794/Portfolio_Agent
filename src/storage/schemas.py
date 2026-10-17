"""Canonical lakehouse schema (DuckDB DDL) + pandas coercion rules.

Design notes
------------
* Every fact table carries an explicit *knowledge timestamp* (``vintage_timestamp``,
  ``filing_timestamp``, ``published_timestamp`` ...). Nothing is joined on economic
  period date alone: that is the leakage vector the architecture exists to close.
* ``pit_protect`` columns are the ones where a *later restatement must never
  overwrite the first public print* (Beneish's original point-in-time requirement,
  ALFRED vintage semantics). On upsert the earliest vintage wins for those rows.
* Vector columns are stored as variable-length ``FLOAT[]`` (parquet-friendly).
  ``array_dim`` enforces the embedding width at write time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: dict[str, str]  # column -> duckdb type
    primary_key: tuple[str, ...]
    knowledge_column: str | None = None
    pit_protect: tuple[str, ...] = ()
    array_dim: int | None = None


TABLE_SPECS: dict[str, TableSpec] = {
    "golden_copy": TableSpec(
        name="golden_copy",
        columns={
            "trade_date": "DATE",
            "ticker": "VARCHAR",
            "market": "VARCHAR",
            "isin": "VARCHAR",
            "open": "DOUBLE",
            "high": "DOUBLE",
            "low": "DOUBLE",
            "close": "DOUBLE",
            "volume": "DOUBLE",
            "dollar_volume_raw": "DOUBLE",
            "sources_agreed": "INTEGER",
            "max_dispersion_bps": "DOUBLE",
            "mad_z_score": "DOUBLE",
            "golden_status": "VARCHAR",
            "sha256_proof": "VARCHAR",
        },
        primary_key=("market", "ticker", "trade_date"),
    ),
    "macro_external_pit": TableSpec(
        name="macro_external_pit",
        columns={
            "series_id": "VARCHAR",
            "observation_date": "DATE",
            "vintage_timestamp": "TIMESTAMPTZ",
            "value": "DOUBLE",
            "z_score_252d": "DOUBLE",
        },
        primary_key=("series_id", "observation_date", "vintage_timestamp"),
        knowledge_column="vintage_timestamp",
    ),
    "policy_geopol_events": TableSpec(
        name="policy_geopol_events",
        columns={
            "event_id": "VARCHAR",
            "published_timestamp": "TIMESTAMPTZ",
            "domain": "VARCHAR",
            "source_agency": "VARCHAR",
            "authority_weight": "DOUBLE",
            "polarity": "DOUBLE",
            "half_life_days": "DOUBLE",
            "headline": "VARCHAR",
            "raw_text": "VARCHAR",
            "embedding": "FLOAT[]",
        },
        primary_key=("event_id",),
        knowledge_column="published_timestamp",
        array_dim=384,
    ),
    "tech_evolution_pit": TableSpec(
        name="tech_evolution_pit",
        columns={
            "ticker": "VARCHAR",
            "market": "VARCHAR",
            "knowledge_timestamp": "TIMESTAMPTZ",
            "grants_publications_180d": "INTEGER",
            "citation_velocity_log_ratio": "DOUBLE",
            "rd_to_revenue": "DOUBLE",
            "gross_margin_3y_delta": "DOUBLE",
            "capex_to_depreciation": "DOUBLE",
            "gics_sector": "VARCHAR",
            "revenue": "DOUBLE",
            "rd_expense": "DOUBLE",
            "capex": "DOUBLE",
            "depreciation": "DOUBLE",
            "gross_margin": "DOUBLE",
            "gross_margin_3y_ago": "DOUBLE",
        },
        primary_key=("ticker", "knowledge_timestamp"),
        knowledge_column="knowledge_timestamp",
        pit_protect=("citation_velocity_log_ratio",),
    ),
    "fundamentals_pit": TableSpec(
        name="fundamentals_pit",
        columns={
            "ticker": "VARCHAR",
            "market": "VARCHAR",
            "fiscal_period": "VARCHAR",
            "filing_timestamp": "TIMESTAMPTZ",
            "piotroski_f_score": "TINYINT",
            "beneish_m_score": "DOUBLE",
            "roic_wacc_spread": "DOUBLE",
            "fcf_ev_yield": "DOUBLE",
            "ev": "DOUBLE",
            "fcf": "DOUBLE",
            "roic": "DOUBLE",
            "wacc": "DOUBLE",
            "beta": "DOUBLE",
            "dollar_volume_20d": "DOUBLE",
            "gics_sector": "VARCHAR",
            "promoter_holding_pct": "DOUBLE",
            "promoter_pledge_pct": "DOUBLE",
            "institutional_flow_delta": "DOUBLE",
            "business_embedding": "FLOAT[]",
            "gate_hazard": "DOUBLE",
            "gate_pass": "BOOLEAN",
        },
        primary_key=("ticker", "filing_timestamp"),
        knowledge_column="filing_timestamp",
        pit_protect=("beneish_m_score", "piotroski_f_score"),
        array_dim=384,
    ),
    "market_prices": TableSpec(
        name="market_prices",
        columns={
            "ticker": "VARCHAR",
            "market": "VARCHAR",
            "date": "DATE",
            "open": "DOUBLE",
            "high": "DOUBLE",
            "low": "DOUBLE",
            "close": "DOUBLE",
            "volume": "DOUBLE",
            "adj_close": "DOUBLE",
            "dollar_volume": "DOUBLE",
            "knowledge_timestamp": "TIMESTAMPTZ",
        },
        primary_key=("ticker", "date"),
        knowledge_column="knowledge_timestamp",
    ),
    "corporate_actions": TableSpec(
        name="corporate_actions",
        columns={
            "ticker": "VARCHAR",
            "market": "VARCHAR",
            "ex_date": "DATE",
            "action_type": "VARCHAR",
            "split_ratio": "DOUBLE",
            "cash_dividend": "DOUBLE",
            "isin": "VARCHAR",
            "raw_narrative": "VARCHAR",
            "knowledge_timestamp": "TIMESTAMPTZ",
        },
        primary_key=("ticker", "ex_date", "action_type"),
        knowledge_column="knowledge_timestamp",
    ),
    "factor_scores": TableSpec(
        name="factor_scores",
        columns={
            "date": "DATE",
            "ticker": "VARCHAR",
            "market": "VARCHAR",
            "gics_sector": "VARCHAR",
            "kalman_trend": "DOUBLE",
            "fundamentals": "DOUBLE",
            "tech_evolution": "DOUBLE",
            "gov_policy": "DOUBLE",
            "geopolitics": "DOUBLE",
            "macro_beta": "DOUBLE",
            "timestral": "DOUBLE",
            "composite_alpha": "DOUBLE",
            "regime": "VARCHAR",
            "turbulence_d2": "DOUBLE",
            "cash_buffer": "DOUBLE",
        },
        primary_key=("date", "ticker"),
        knowledge_column="date",
    ),
    "portfolio_targets": TableSpec(
        name="portfolio_targets",
        columns={
            "date": "DATE",
            "market": "VARCHAR",
            "ticker": "VARCHAR",
            "target_weight": "DOUBLE",
            "current_weight": "DOUBLE",
            "alpha": "DOUBLE",
            "price": "DOUBLE",
            "order_shares": "DOUBLE",
            "order_value": "DOUBLE",
            "estimated_cost_bps": "DOUBLE",
            "deadband": "DOUBLE",
            "binding_constraint": "VARCHAR",
        },
        primary_key=("date", "market", "ticker"),
        knowledge_column="date",
    ),
    "spillover_us_india": TableSpec(
        name="spillover_us_india",
        columns={
            "as_of_date": "DATE",
            "ticker": "VARCHAR",
            "projected_beta_return": "DOUBLE",
            "driver_vector": "FLOAT[]",
            "r_squared": "DOUBLE",
        },
        primary_key=("as_of_date", "ticker"),
        knowledge_column="as_of_date",
    ),
    "calibration_state": TableSpec(
        name="calibration_state",
        columns={
            "as_of_date": "DATE",
            "scope": "VARCHAR",
            "parameter": "VARCHAR",
            "method": "VARCHAR",
            "value": "DOUBLE",
            "sample_size": "INTEGER",
        },
        primary_key=("as_of_date", "scope", "parameter"),
        knowledge_column="as_of_date",
    ),
    "run_manifest": TableSpec(
        name="run_manifest",
        columns={
            "run_id": "VARCHAR",
            "mode": "VARCHAR",
            "market": "VARCHAR",
            "started_at": "TIMESTAMPTZ",
            "finished_at": "TIMESTAMPTZ",
            "config_hash": "VARCHAR",
            "status": "VARCHAR",
            "details": "VARCHAR",
        },
        primary_key=("run_id",),
        knowledge_column="started_at",
    ),
}


def ddl(spec: TableSpec) -> str:
    """DuckDB ``CREATE TABLE`` statement for a spec (documentation / DQ tooling)."""
    cols = ",\n    ".join(f"{name} {dtype}" for name, dtype in spec.columns.items())
    pk = ", ".join(spec.primary_key)
    return f"CREATE TABLE IF NOT EXISTS {spec.name} (\n    {cols},\n    PRIMARY KEY ({pk})\n);"


def parquet_safe_columns(spec: TableSpec) -> list[str]:
    return list(spec.columns.keys())


__all__ = ["TABLE_SPECS", "TableSpec", "ddl", "parquet_safe_columns", "field"]
