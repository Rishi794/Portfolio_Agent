"""Universe discovery — no hardcoded ticker lists anywhere in the codebase.

US   : SEC ``company_tickers.json`` (official, free, no key) filtered to the
       exchanges configured in ``config/universe.yaml``.
India: NSE index constituent CSV (Nifty 500) served from ``nsearchives``.

A ``static_csv`` loader is available for custom universes (``--universe-csv``).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from .base import HttpClient, SourceUnavailable


class UniverseLoader:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client

    # -------------------------------------------------------------------- US ----
    def load_us(self, limit: int | None = None) -> pd.DataFrame:
        spec = self.config.market("US")["universe_source"]
        payload = self.client.get_json(spec["url"])
        rows = []
        priority = [str(x).lower() for x in spec.get("priority_exchanges", [])]
        for entry in (payload.values() if isinstance(payload, dict) else payload):
            ticker = str(entry.get("ticker", "")).upper()
            if not ticker or any(ch in ticker for ch in "^$."):
                continue
            rows.append({"ticker": ticker, "cik": int(entry["cik_str"]), "name": entry.get("title", "")})
        frame = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
        frame = frame.sort_values("ticker").reset_index(drop=True)
        frame["market"] = "US"
        frame["exchange_priority"] = frame["ticker"].map(lambda _: len(priority))
        cap = limit if limit is not None else int(self.config.market("US").get("max_symbols", 500))
        return frame.head(cap)

    # ----------------------------------------------------------------- INDIA ----
    def load_india(self, limit: int | None = None) -> pd.DataFrame:
        spec = self.config.market("INDIA")["universe_source"]
        text = self.client.get_text(spec["url"])
        from io import StringIO

        frame = pd.read_csv(StringIO(text))
        symbol_col = spec.get("symbol_column", "Symbol")
        name_col = spec.get("name_column", "Company Name")
        sector_col = spec.get("sector_column")
        suffix = str(self.config.market("INDIA").get("symbol_suffix", ".NS"))
        out = pd.DataFrame(
            {
                "ticker": frame[symbol_col].astype(str).str.strip() + suffix,
                "nse_symbol": frame[symbol_col].astype(str).str.strip(),
                "name": frame[name_col].astype(str).str.strip() if name_col in frame.columns else "",
                "gics_sector": frame[sector_col].astype(str).str.strip() if sector_col and sector_col in frame.columns else "UNKNOWN",
            }
        )
        out["market"] = "INDIA"
        cap = limit if limit is not None else int(self.config.market("INDIA").get("max_symbols", 500))
        return out.drop_duplicates(subset=["ticker"]).head(cap).reset_index(drop=True)

    # ------------------------------------------------------------------ misc ----
    @staticmethod
    def load_static_csv(path: str | Path, market: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "ticker" not in frame.columns:
            raise ValueError("static universe CSV requires a 'ticker' column")
        frame["market"] = market
        return frame

    def load(self, market: str, limit: int | None = None, static_csv: str | Path | None = None) -> pd.DataFrame:
        if static_csv:
            return self.load_static_csv(static_csv, market)
        if market == "US":
            return self.load_us(limit=limit)
        if market == "INDIA":
            return self.load_india(limit=limit)
        raise KeyError(f"no loader for market {market!r}")

    def load_or_fallback(self, market: str, lake, limit: int | None = None) -> pd.DataFrame:
        """Universe from the network, else from the last persisted universe table."""
        try:
            return self.load(market, limit=limit)
        except (SourceUnavailable, Exception):
            cached = lake.read("fundamentals_pit")
            if not cached.empty:
                subset = cached[cached["market"] == market]
                if not subset.empty:
                    return (
                        subset[["ticker", "gics_sector"]]
                        .drop_duplicates(subset=["ticker"])
                        .assign(market=market)
                        .reset_index(drop=True)
                    )
            raise SourceUnavailable(f"universe for {market} unavailable and no cached universe exists")


def shard_of(items: list[Any], shard_id: int, total_shards: int) -> list[Any]:
    """Deterministic modulo sharding used by the parallel GitHub Actions matrix."""
    return [item for idx, item in enumerate(items) if idx % max(total_shards, 1) == shard_id]


def knowledge_timestamp_for_close(day: dt.date, market: str, config: Config) -> dt.datetime:
    """Conservative knowledge timestamp: end-of-day close publication, in UTC."""
    hour_utc = 22 if market == "US" else 12
    return dt.datetime.combine(day, dt.time(hour_utc, 0), tzinfo=dt.timezone.utc)
