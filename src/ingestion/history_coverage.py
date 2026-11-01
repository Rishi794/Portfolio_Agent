"""Point-in-time coverage manifest used to choose full backfill vs. daily deltas."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = 1


def build_history_coverage_manifest(prices: pd.DataFrame) -> dict[str, Any]:
    """Summarize distinct stored sessions per market/ticker for shard workers."""
    markets: dict[str, dict[str, dict[str, Any]]] = {}
    if prices is not None and not prices.empty:
        frame = prices[["market", "ticker", "date"]].copy()
        frame["market"] = frame["market"].astype(str)
        frame["ticker"] = frame["ticker"].astype(str)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame.dropna(subset=["market", "ticker", "date"])
        for (market, ticker), group in frame.groupby(["market", "ticker"], sort=True):
            dates = group["date"].drop_duplicates().sort_values()
            if dates.empty:
                continue
            markets.setdefault(str(market), {})[str(ticker)] = {
                "bar_count": int(len(dates)),
                "first_trade_date": dates.iloc[0].date().isoformat(),
                "last_trade_date": dates.iloc[-1].date().isoformat(),
            }
    return {"schema_version": SCHEMA_VERSION, "markets": markets}


def write_history_coverage_manifest(prices: pd.DataFrame, path: str | Path) -> None:
    """Atomically persist coverage; a missing/corrupt file means cold-start."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(build_history_coverage_manifest(prices), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)


def read_market_coverage(path: str | Path | None, market: str) -> dict[str, dict[str, Any]]:
    """Read one market's coverage; malformed or absent state safely cold-starts."""
    if path is None:
        return {}
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        return {}
    markets = manifest.get("markets")
    records = markets.get(market) if isinstance(markets, dict) else None
    if not isinstance(records, dict):
        return {}
    return {
        str(ticker): record
        for ticker, record in records.items()
        if isinstance(record, dict)
    }


def history_groups(
    tickers: list[str],
    coverage: Mapping[str, Mapping[str, Any]],
    *,
    minimum_bars: int,
    force_full_history: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (cold, warm); missing dates/counts are treated as cold data."""
    cold: list[str] = []
    warm: list[str] = []
    for raw_ticker in tickers:
        ticker = str(raw_ticker)
        record = coverage.get(ticker)
        try:
            count = int(record.get("bar_count", 0)) if record else 0
            last_date = dt.date.fromisoformat(str(record["last_trade_date"])) if record else None
        except (TypeError, ValueError, KeyError):
            count, last_date = 0, None
        if force_full_history or count < int(minimum_bars) or last_date is None:
            cold.append(ticker)
        else:
            warm.append(ticker)
    return cold, warm

