"""Zstd Parquet lakehouse with DuckDB in-memory analytics and PIT-safe joins.

The lakehouse is a flat directory of ``<table>.parquet`` files that is shipped to
(and rotated through) GitHub Releases. Every write is an idempotent *upsert*
keyed by the table's primary key, with optional point-in-time protection: when a
restated figure would overwrite the first public print, the first print wins.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

from .schemas import TABLE_SPECS, TableSpec

_INT_TYPES = {"INTEGER", "TINYINT", "BIGINT", "SMALLINT"}
_FLOAT_TYPES = {"DOUBLE", "FLOAT", "REAL"}


class Lakehouse:
    """Directory-of-parquet store queried through DuckDB (no server, zero cost)."""

    def __init__(self, root: str | Path, tables: dict[str, TableSpec] | None = None) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tables: dict[str, TableSpec] = dict(tables or TABLE_SPECS)

    # ------------------------------------------------------------------ paths --
    def path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def spec(self, name: str) -> TableSpec:
        if name not in self.tables:
            raise KeyError(f"unknown table {name!r}")
        return self.tables[name]

    def empty(self, name: str) -> pd.DataFrame:
        spec = self.spec(name)
        return pd.DataFrame({col: pd.Series(dtype=_pandas_dtype(dtype)) for col, dtype in spec.columns.items()})

    # ------------------------------------------------------------------- read --
    def read(self, name: str) -> pd.DataFrame:
        if not self.exists(name):
            return self.empty(name)
        frame = pd.read_parquet(self.path(name))
        return self._coerce(frame, self.spec(name))

    def read_all(self) -> dict[str, pd.DataFrame]:
        return {name: self.read(name) for name in self.tables if self.exists(name)}

    # ------------------------------------------------------------------ write --
    def write(self, name: str, frame: pd.DataFrame, *, enforce_pit: bool = True) -> int:
        """Upsert ``frame`` into the table; returns the number of rows now stored."""
        spec = self.spec(name)
        incoming = self._coerce(frame, spec)
        if incoming.empty:
            if not self.exists(name):
                return 0
            return len(self.read(name))
        if self.exists(name):
            existing = self.read(name)
            combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
        else:
            combined = incoming
        combined = self._resolve(combined, spec, enforce_pit=enforce_pit)
        combined = self._coerce(combined, spec)
        tmp = self.path(name).with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(self.path(name))
        return len(combined)

    # ---------------------------------------------------------------- queries --
    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        con.execute("SET TimeZone='UTC';")
        for name in self.tables:
            if self.exists(name):
                literal = str(self.path(name)).replace("'", "''")
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{literal}');")
        return con

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        con = self.connect()
        try:
            return con.execute(sql, list(params) if params else None).df()
        finally:
            con.close()

    def snapshot_at(self, name: str, as_of: dt.date | dt.datetime | pd.Timestamp, entity: Iterable[str] | None = None) -> pd.DataFrame:
        """Latest *known* version of each entity as of ``as_of`` (no look-ahead)."""
        frame = self.read(name)
        if frame.empty:
            return frame
        spec = self.spec(name)
        knowledge = spec.knowledge_column
        if knowledge is None or knowledge not in frame.columns:
            return frame
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        known = frame[pd.to_datetime(frame[knowledge], utc=True, errors="coerce") <= cutoff]
        if known.empty:
            return frame.iloc[0:0]
        entity_cols = list(entity) if entity is not None else [
            col for col in spec.primary_key if col != knowledge
        ]
        if not entity_cols:
            return known.sort_values(knowledge).tail(1)
        known = known.sort_values(knowledge)
        return known.drop_duplicates(subset=entity_cols, keep="last").reset_index(drop=True)

    def asof_join(
        self,
        left: str | pd.DataFrame,
        right: str | pd.DataFrame,
        *,
        left_on: str,
        right_on: str,
        by: str | None = None,
        left_where: str | None = None,
        select: str = "l.*, r.*",
    ) -> pd.DataFrame:
        """Leakage-free ``ASOF JOIN`` executed inside DuckDB."""
        con = duckdb.connect(database=":memory:")
        con.execute("SET TimeZone='UTC';")
        try:
            _register(con, "l", left, self)
            _register(con, "r", right, self)
            by_clause = f"ON l.{by} = r.{by} AND l.{left_on} >= r.{right_on}" if by else f"ON l.{left_on} >= r.{right_on}"
            where_clause = f" WHERE {left_where}" if left_where else ""
            sql = f"SELECT {select} FROM l ASOF JOIN r {by_clause}{where_clause}"
            return con.execute(sql).df()
        finally:
            con.close()

    # ------------------------------------------------------------ maintenance --
    def verify(self) -> dict[str, Any]:
        """Data-quality report: PK uniqueness, null knowledge stamps, embedding width."""
        issues: list[str] = []
        tables: dict[str, Any] = {}
        for name, spec in self.tables.items():
            if not self.exists(name):
                continue
            frame = self.read(name)
            local: list[str] = []
            for col in spec.primary_key:
                if col not in frame.columns:
                    local.append(f"missing primary-key column {col}")
                    continue
                if frame[col].isna().any():
                    local.append(f"null values in primary-key column {col}")
            if len(frame):
                dupes = int(frame.duplicated(subset=list(spec.primary_key)).sum())
                if dupes:
                    local.append(f"{dupes} duplicate primary keys")
            if spec.knowledge_column and spec.knowledge_column in frame.columns:
                nulls = int(pd.to_datetime(frame[spec.knowledge_column], errors="coerce").isna().sum())
                if nulls:
                    local.append(f"{nulls} null knowledge timestamps")
            if spec.array_dim:
                for col, dtype in spec.columns.items():
                    if not dtype.endswith("[]") or col not in frame.columns:
                        continue
                    widths = {len(v) for v in frame[col].dropna() if isinstance(v, (list, np.ndarray))}
                    bad = {w for w in widths if w not in (0, spec.array_dim)}
                    if bad:
                        local.append(f"embedding column {col} has widths {sorted(bad)} != {spec.array_dim}")
            tables[name] = {"rows": int(len(frame)), "issues": local}
            issues.extend(f"{name}: {msg}" for msg in local)
        return {"ok": not issues, "tables": tables, "issues": issues}

    def describe(self) -> str:
        lines = []
        for name, spec in self.tables.items():
            rows = len(self.read(name)) if self.exists(name) else 0
            lines.append(f"{name:<24} rows={rows:<8} pk={','.join(spec.primary_key)}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- internals
    def _coerce(self, frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
        return coerce_frame(frame, spec)

    @staticmethod
    def _resolve(frame: pd.DataFrame, spec: TableSpec, *, enforce_pit: bool) -> pd.DataFrame:
        pk = [col for col in spec.primary_key if col in frame.columns]
        if not pk:
            return frame
        if enforce_pit and spec.knowledge_column and spec.knowledge_column in frame.columns:
            frame = frame.sort_values(spec.knowledge_column, kind="stable")
            return frame.drop_duplicates(subset=pk, keep="first").reset_index(drop=True)
        return frame.drop_duplicates(subset=pk, keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pandas_dtype(duckdb_type: str) -> str:
    if duckdb_type.endswith("[]"):
        return "object"
    if duckdb_type in _INT_TYPES:
        return "Int64"
    if duckdb_type in _FLOAT_TYPES:
        return "float64"
    if duckdb_type == "BOOLEAN":
        return "boolean"
    if duckdb_type == "DATE":
        return "datetime64[ns]"
    if duckdb_type == "TIMESTAMPTZ":
        return "datetime64[ns, UTC]"
    return "object"


def coerce_frame(frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Align an arbitrary DataFrame to a table spec (types, columns, ordering)."""
    out = pd.DataFrame(index=frame.index)
    for col, dtype in spec.columns.items():
        if col not in frame.columns:
            pd_dtype = _pandas_dtype(dtype)
            # pandas >= 3.0 refuses to build numeric/datetime arrays directly from
            # pd.NA scalars, so broadcast the dtype-appropriate missing sentinel.
            filler: Any = (
                np.nan
                if pd_dtype in ("float64", "boolean", "datetime64[ns]", "datetime64[ns, UTC]")
                else pd.NA
            )
            out[col] = pd.Series(filler, index=frame.index).astype(pd_dtype)
            continue
        series = frame[col]
        if dtype.endswith("[]"):
            values: list[Any] = []
            for item in series:
                if item is None or item is pd.NA or (isinstance(item, float) and np.isnan(item)):
                    values.append(None)
                    continue
                try:
                    # ravel() accepts both a plain vector and a nested singleton
                    # list (common when embeddings round-trip through JSON).
                    vector = np.asarray(item, dtype=float).ravel()
                except (TypeError, ValueError):
                    values.append(None)
                    continue
                values.append(vector.tolist())
            out[col] = pd.Series(values, index=frame.index, dtype="object")
        elif dtype in _INT_TYPES:
            out[col] = pd.to_numeric(series, errors="coerce").round().astype("Int64")
        elif dtype in _FLOAT_TYPES:
            out[col] = pd.to_numeric(series, errors="coerce").astype("float64")
        elif dtype == "BOOLEAN":
            out[col] = series.astype("boolean")
        elif dtype == "DATE":
            out[col] = pd.to_datetime(series, errors="coerce").dt.tz_localize(None).dt.normalize()
        elif dtype == "TIMESTAMPTZ":
            stamps = pd.to_datetime(series, errors="coerce", utc=True)
            out[col] = stamps
        else:
            out[col] = series.astype("object").where(series.notna(), None)
    return out.reset_index(drop=True)


def _register(con: duckdb.DuckDBPyConnection, alias: str, source: str | pd.DataFrame, lake: Lakehouse) -> None:
    if isinstance(source, pd.DataFrame):
        con.register(f"{alias}_df", source)
        con.execute(f"CREATE OR REPLACE VIEW {alias} AS SELECT * FROM {alias}_df;")
        return
    literal = str(lake.path(source)).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW {alias} AS SELECT * FROM read_parquet('{literal}');")


def manifest_row(
    run_id: str,
    mode: str,
    market: str,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    config_hash: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "mode": mode,
                "market": market,
                "started_at": started_at,
                "finished_at": finished_at,
                "config_hash": config_hash,
                "status": status,
                "details": json.dumps(details or {}, default=str)[:8000],
            }
        ]
    )
