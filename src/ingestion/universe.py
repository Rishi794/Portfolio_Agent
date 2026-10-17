"""Universe discovery — no hardcoded ticker lists anywhere in the codebase.

US   : SEC ``company_tickers.json`` (official, free, no key). SEC blocks generic
       User-Agents (403) and requires a declared contact, sent via
       ``SEC_USER_AGENT``; datacenter IPs are also sometimes blocked outright, so
       the official NASDAQ Trader symbol directory is the configured fallback.
India: NSE index constituent CSV (Nifty 500) served from ``nsearchives``, with
       the legacy ``archives`` host as fallback.

Every source is declared in ``config/universe.yaml`` as a primary spec plus a
``fallback`` chain, so a blocked endpoint degrades to the next one instead of
failing the run. A ``static_csv`` loader serves custom universes
(``--universe-csv``).
"""

from __future__ import annotations

import datetime as dt
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from .base import HttpClient, SourceUnavailable

# CSV / pipe-delimited endpoints (NASDAQ symbol directory, NSE archives) answer
# 406 Not Acceptable to a JSON-only ``Accept`` header, so text fetches override it.
TEXT_ACCEPT = {"Accept": "text/csv, text/plain, */*"}


class UniverseLoader:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client

    # -------------------------------------------------------------------- US ----
    def load_us(self, limit: int | None = None) -> pd.DataFrame:
        spec = self.config.market("US")["universe_source"]
        frame = self._load_with_fallback("US", spec)
        priority = [str(x).lower() for x in spec.get("priority_exchanges", [])]
        frame = frame.sort_values("ticker").reset_index(drop=True)
        frame["market"] = "US"
        frame["exchange_priority"] = frame["ticker"].map(lambda _: len(priority))
        cap = limit if limit is not None else int(self.config.market("US").get("max_symbols", 500))
        return frame.head(cap)

    def _load_us_sec_company_tickers(self, spec: dict[str, Any]) -> pd.DataFrame:
        payload = self.client.get_json(spec["url"])
        rows = []
        for entry in (payload.values() if isinstance(payload, dict) else payload):
            ticker = str(entry.get("ticker", "")).upper()
            if not ticker or any(ch in ticker for ch in "^$."):
                continue
            rows.append({"ticker": ticker, "cik": int(entry["cik_str"]), "name": entry.get("title", "")})
        return pd.DataFrame(rows).drop_duplicates(subset=["ticker"])

    def _load_us_nasdaq_symdir(self, spec: dict[str, Any]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for url in spec["urls"]:
            frame = pd.read_csv(
                StringIO(self.client.get_text(url, extra_headers=TEXT_ACCEPT)), sep="|", dtype=str
            )
            symbol_col = "Symbol" if "Symbol" in frame.columns else "ACT Symbol"
            if symbol_col not in frame.columns or "Security Name" not in frame.columns:
                raise SourceUnavailable(f"unrecognized NASDAQ symbol directory payload at {url}")
            frame = frame.rename(columns={symbol_col: "ticker", "Security Name": "name"})
            frame = frame[frame["ticker"].notna()].copy()
            frame["ticker"] = frame["ticker"].str.strip().str.upper()
            frame = frame[frame["ticker"].str.fullmatch(r"[A-Z][A-Z0-9.\-]*", na=False)]
            if "Test Issue" in frame.columns:
                frame = frame[frame["Test Issue"].fillna("").str.strip().str.upper() == "N"]
            frames.append(frame[["ticker", "name"]])
        if not frames:
            raise SourceUnavailable("NASDAQ symbol directory returned no parsable frames")
        frame = pd.concat(frames, ignore_index=True)
        # Warrants / units / preferreds / rights / notes are not ordinary equities.
        junk = r"\b(?:Preferred|Warrants?|Rights?|Units?|Depositary|Notes?)\b"
        frame = frame[~frame["name"].fillna("").str.contains(junk, case=False, regex=True)]
        # NASDAQ uses dotted class shares (BRK.A); SEC / Yahoo use dashes (BRK-A).
        frame["ticker"] = frame["ticker"].str.replace(".", "-", regex=False)
        frame["cik"] = 0
        return frame.drop_duplicates(subset=["ticker"])

    # ----------------------------------------------------------------- INDIA ----
    def load_india(self, limit: int | None = None) -> pd.DataFrame:
        spec = self.config.market("INDIA")["universe_source"]
        frame = self._load_with_fallback("INDIA", spec)
        cap = limit if limit is not None else int(self.config.market("INDIA").get("max_symbols", 500))
        return frame.drop_duplicates(subset=["ticker"]).head(cap).reset_index(drop=True)

    def _load_india_nse_index_csv(self, spec: dict[str, Any]) -> pd.DataFrame:
        frame = pd.read_csv(StringIO(self.client.get_text(spec["url"], extra_headers=TEXT_ACCEPT)))
        symbol_col = spec.get("symbol_column", "Symbol")
        if symbol_col not in frame.columns:
            raise SourceUnavailable(
                f"unexpected NSE payload at {spec['url']} (columns: {[str(c) for c in list(frame.columns)[:6]]})"
            )
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
        return out

    # -------------------------------------------------------------- fallbacks ---
    def _load_with_fallback(self, market: str, spec: dict[str, Any]) -> pd.DataFrame:
        """Walk the configured source chain; the first non-empty frame wins."""
        errors: list[str] = []
        for candidate in self._source_chain(spec):
            source_type = str(candidate.get("type", ""))
            try:
                frame = self._load_single(market, source_type, candidate)
            except (SourceUnavailable, KeyError, ValueError, pd.errors.ParserError) as exc:
                errors.append(f"{source_type or '?'}: {exc}")
                continue
            if frame is not None and not frame.empty:
                return frame
            errors.append(f"{source_type or '?'}: empty frame")
        raise SourceUnavailable(f"{market} universe unavailable from every configured source ({'; '.join(errors)})")

    @staticmethod
    def _source_chain(spec: dict[str, Any]) -> list[dict[str, Any]]:
        chain = [spec]
        fallback = spec.get("fallback") or []
        chain.extend([fallback] if isinstance(fallback, dict) else list(fallback))
        return chain

    def _load_single(self, market: str, source_type: str, spec: dict[str, Any]) -> pd.DataFrame:
        if market == "US" and source_type == "sec_company_tickers":
            return self._load_us_sec_company_tickers(spec)
        if market == "US" and source_type == "nasdaq_symdir":
            return self._load_us_nasdaq_symdir(spec)
        if market == "INDIA" and source_type == "nse_index_csv":
            return self._load_india_nse_index_csv(spec)
        raise ValueError(f"unknown universe source type {source_type!r} for market {market}")

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
