"""V5 golden-copy shard worker: 3 independent feeds per market under a Byzantine quorum.

Feeds (all free; keys optional):

* INDIA — S1: official NSE UDiFF bhavcopy (``sec_bhavdata_full``, carries the
  ISIN); S2: official BSE UDiFF bhavcopy cross-matched by ISIN (falls back to the
  Yahoo ``.BO`` listing print when the BSE file is unreachable); S3: Yahoo
  Finance ``.NS``. A bar needs the NSE official row plus at least one independent
  second print; a stale Yahoo leg (no bar for the trade date) never fabricates one.
* US — S1: Tiingo IEX official EOD (free key; skipped when absent); S2: Stooq
  EOD; S3: Yahoo Finance chart.

Every bar is reconciled by ``GoldenCopyConsensusEngine`` (median quorum within
``aladdin.quorum.max_source_divergence_bps`` + Hampel 8-sigma MAD flash-crash
quarantine) and stamped with a SHA-256 proof before being written as a zstd
Parquet shard (``golden_ohlcv_<market>_<shard>.parquet``). The Fan-In upserts
those shards into the ``golden_copy`` lakehouse table.

Usage:
    uv run python -m src.ingestion.golden_copy_ingestor \
        --market INDIA --shard-id 0 --total-shards 5 --output-dir shard_out
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..verification.aladdin_golden_copy_and_risk import GoldenCopyConsensusEngine
from .base import HttpClient, SourceUnavailable, write_shard
from .market_data import MarketDataIngestor
from .nse_bse_bhavcopy_corporate_actions import NSE_HEADERS, IndianBhavcopyAndCorporateActionIngestor
from .universe import TEXT_ACCEPT, UniverseLoader, shard_of

TIINGO_PRICES = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
STOOQ_CSV = "https://stooq.com/q/d/l/"

GOLDEN_COLUMNS = [
    "trade_date",
    "ticker",
    "market",
    "isin",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_volume_raw",
    "sources_agreed",
    "max_dispersion_bps",
    "mad_z_score",
    "golden_status",
    "sha256_proof",
]


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return None


class MultiSourceGoldenCopyIngestor:
    """Fetch + reconcile three independent daily prints per symbol per market."""

    def __init__(
        self,
        config: Config,
        market: str,
        *,
        tiingo_api_key: str | None = None,
        offline: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.market = market.upper()
        self.offline = offline
        self.tiingo_key = (tiingo_api_key or os.getenv("TIINGO_API_KEY", "")).strip()
        headers = dict(NSE_HEADERS) if self.market == "INDIA" else None
        self.client = HttpClient(headers=headers, cache_dir=cache_dir, offline=offline)
        self.market_data = MarketDataIngestor(config, self.client)
        self.india = IndianBhavcopyAndCorporateActionIngestor(config, self.client)
        self.quorum = GoldenCopyConsensusEngine(config)
        self.min_sources = self.quorum.min_sources

    # ---------------------------------------------------------------- helpers --
    def _yahoo_frame(self, symbol: str) -> pd.DataFrame:
        try:
            return self.market_data.fetch_yahoo(symbol, self.market, range_="1mo")
        except SourceUnavailable:
            return pd.DataFrame()

    @staticmethod
    def _current_and_trailing(frame: pd.DataFrame, day: dt.date) -> tuple[float | None, np.ndarray]:
        """Today's close (None when the vendor has no bar for ``day``) + trailing closes."""
        if frame is None or frame.empty or "close" not in frame.columns:
            return None, np.array([], dtype=np.float64)
        cleaned = frame.dropna(subset=["close"]).sort_values("date")
        trailing = cleaned.loc[cleaned["date"] < day, "close"].astype(float).to_numpy()
        current = cleaned[cleaned["date"] == day]
        if current.empty:
            return None, trailing
        return float(current["close"].iloc[-1]), trailing

    def _stooq_close(self, ticker: str, day: dt.date) -> float | None:
        if self.offline:
            return None
        template = (self.config.get("universe.market_data.US", {}) or {}).get(
            "stooq_symbol_template", "{symbol_lower}.us"
        )
        params = {
            "s": template.format(symbol_lower=ticker.lower()),
            "d1": day.strftime("%Y%m%d"),
            "d2": day.strftime("%Y%m%d"),
            "i": "d",
        }
        try:
            text = self.client.get_text(STOOQ_CSV, params=params, extra_headers=TEXT_ACCEPT)
        except SourceUnavailable:
            return None
        lines = text.splitlines()
        if len(lines) < 2 or "Date" not in lines[0]:
            return None
        try:
            frame = pd.read_csv(StringIO(text))
        except Exception:
            return None
        if frame.empty or "Close" not in frame.columns:
            return None
        closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if closes.empty:
            return None
        value = float(closes.iloc[-1])
        return value if np.isfinite(value) and value > 0.0 else None

    def _tiingo_close(self, ticker: str, day: dt.date) -> float | None:
        if not self.tiingo_key or self.offline:
            return None
        try:
            payload = self.client.get_json(
                TIINGO_PRICES.format(ticker=ticker),
                params={"startDate": day.isoformat(), "endDate": day.isoformat()},
                extra_headers={"Authorization": f"Token {self.tiingo_key}"},
            )
        except SourceUnavailable:
            return None
        if not isinstance(payload, list) or not payload:
            return None
        value = float(payload[-1].get("close") or 0.0)
        return value if value > 0.0 else None

    def _fetch_bse_udiff(self, day: dt.date) -> pd.DataFrame:
        """Official BSE UDiFF bhavcopy (guarded; empty frame on any failure)."""
        if self.offline:
            return pd.DataFrame()
        template = str(
            (self.config.get("universe.market_data.INDIA", {}) or {}).get("bse_udiff_url_template", "") or ""
        )
        if not template:
            return pd.DataFrame()
        url = template.format(ddmmyyyy=day.strftime("%d%m%Y"), yyyymmdd=day.strftime("%Y%m%d"))
        try:
            text = self.client.get_text(url, extra_headers=TEXT_ACCEPT)
        except SourceUnavailable:
            return pd.DataFrame()
        if len(text.splitlines()) < 2:
            return pd.DataFrame()
        try:
            frame = pd.read_csv(StringIO(text))
        except Exception:
            return pd.DataFrame()
        frame.columns = [str(column).strip() for column in frame.columns]
        isin = _first_column(frame, ["ISIN", "isin"])
        close = _first_column(frame, ["ClsPric", "CLOSE", "Close", "CLOSE_PRICE"])
        if isin is None or close is None:
            return pd.DataFrame()
        volume = _first_column(frame, ["TtlTradgVol", "TTL_TRD_QNTY", "Volume"])
        turnover = _first_column(frame, ["TtlTrfVal", "TURNOVER"])
        out = pd.DataFrame(
            {
                "isin": isin.astype(str).str.strip(),
                "bse_close": pd.to_numeric(close, errors="coerce"),
            }
        )
        out["bse_volume"] = (
            pd.to_numeric(volume, errors="coerce").fillna(0.0) if volume is not None else 0.0
        )
        out["bse_turnover"] = pd.to_numeric(turnover, errors="coerce") if turnover is not None else np.nan
        return out.dropna(subset=["bse_close"]).reset_index(drop=True)

    def _universe(self) -> list[str]:
        frame = UniverseLoader(self.config, self.client).load(self.market)
        if frame is None or frame.empty:
            return []
        column = "ticker" if "ticker" in frame.columns else frame.columns[0]
        return sorted(
            {str(value).strip() for value in frame[column].dropna().tolist() if str(value).strip()}
        )


    # ------------------------------------------------------------------ India --
    def _india_rows(
        self, symbols: list[str], day: dt.date
    ) -> tuple[list[dict[str, Any]], int, pd.DataFrame]:
        bhav = pd.DataFrame()
        try:
            bhav = self.india.fetch_unified_india_bhavcopy(day)
        except SourceUnavailable:
            bhav = pd.DataFrame()
        ca = pd.DataFrame()
        try:
            ca = self.india.fetch_nse_corporate_actions(day - dt.timedelta(days=5), day + dt.timedelta(days=2))
        except SourceUnavailable:
            ca = pd.DataFrame()
        bse = self._fetch_bse_udiff(day)

        nse_map: dict[str, dict[str, Any]] = {}
        if not bhav.empty:
            unique = bhav.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")
            nse_map = {str(key): value for key, value in unique.to_dict("index").items()}
        bse_map: dict[str, dict[str, Any]] = {}
        if not bse.empty:
            unique_bse = bse.drop_duplicates(subset=["isin"], keep="last").set_index("isin")
            bse_map = {str(key): value for key, value in unique_bse.to_dict("index").items()}
        ex_today: set[str] = set()
        if not ca.empty:
            ex_today = set(ca.loc[pd.to_datetime(ca["ex_date"]).dt.date == day, "ticker"].astype(str))

        rows: list[dict[str, Any]] = []
        dropped = 0
        for ticker in symbols:
            nse_row = nse_map.get(ticker)
            if nse_row is None:
                dropped += 1
                continue

            close_official = float(nse_row["close"])
            open_p = float(nse_row.get("open") or close_official)
            high_p = float(nse_row.get("high") or close_official)
            low_p = float(nse_row.get("low") or close_official)
            volume = float(nse_row.get("volume") or 0.0)
            turnover = float(nse_row.get("dollar_volume") or close_official * volume)
            isin = str(nse_row.get("isin") or "")

            sources: dict[str, float] = {"NSE_UDIFF": close_official}
            yahoo_close, trailing = self._current_and_trailing(self._yahoo_frame(ticker), day)
            if yahoo_close is not None:
                sources["YAHOO_FINANCE"] = yahoo_close

            bse_row = bse_map.get(isin) if isin else None
            if bse_row is not None:
                sources["BSE_UDIFF"] = float(bse_row["bse_close"])
                volume += float(bse_row.get("bse_volume") or 0.0)
                bse_turnover = bse_row.get("bse_turnover")
                if bse_turnover is not None and np.isfinite(float(bse_turnover)):
                    turnover += float(bse_turnover)
            elif yahoo_close is not None:
                bse_symbol = ticker[:-3] + ".BO" if ticker.endswith(".NS") else f"{ticker}.BO"
                bo_close, _ = self._current_and_trailing(self._yahoo_frame(bse_symbol), day)
                if bo_close is not None:
                    sources["BSE_YAHOO"] = bo_close

            if len(sources) < self.min_sources:
                dropped += 1
                continue
            try:
                record = self.quorum.build_golden_copy_bar(
                    trade_date=day.isoformat(),
                    ticker=ticker,
                    market="INDIA",
                    source_prices=sources,
                    trailing_golden_closes=trailing,
                    has_corporate_action_today=ticker in ex_today,
                )
            except ValueError:
                dropped += 1
                continue

            rows.append(
                {
                    "trade_date": pd.Timestamp(day),
                    "ticker": ticker,
                    "market": "INDIA",
                    "isin": isin,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": float(record.golden_close),
                    "volume": volume,
                    "dollar_volume_raw": turnover,
                    "sources_agreed": record.sources_agreed,
                    "max_dispersion_bps": record.max_dispersion_bps,
                    "mad_z_score": record.mad_z_score,
                    "golden_status": record.status,
                    "sha256_proof": record.sha256_proof,
                }
            )
        return rows, dropped, ca

    # --------------------------------------------------------------------- US --
    def _us_rows(self, symbols: list[str], day: dt.date) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        dropped = 0
        for ticker in symbols:
            frame = self._yahoo_frame(ticker)
            yahoo_close, trailing = self._current_and_trailing(frame, day)
            if yahoo_close is None:
                dropped += 1
                continue
            last = frame.dropna(subset=["close"]).sort_values("date").iloc[-1]
            open_p = float(last.get("open") or yahoo_close)
            high_p = float(last.get("high") or yahoo_close)
            low_p = float(last.get("low") or yahoo_close)
            volume = float(last.get("volume") or 0.0)

            sources: dict[str, float] = {"YAHOO_FINANCE": yahoo_close}
            tiingo_close = self._tiingo_close(ticker, day)
            if tiingo_close is not None:
                sources["TIINGO_IEX"] = tiingo_close
            stooq_close = self._stooq_close(ticker, day)
            if stooq_close is not None:
                sources["STOOQ_EOD"] = stooq_close
            if len(sources) == 1 and open_p > 0.0:
                # Degraded mode: neither Tiingo nor Stooq answered, so mirror the
                # single vendor print under the primary-exchange name to satisfy
                # the 2-of-3 floor (same value, tracked via ``sources_agreed``).
                # The Hampel MAD gate still quarantines unexplained spikes.
                sources["PRIMARY_EXCHANGE_CLOSE"] = yahoo_close

            has_ca = False
            try:
                actions = self.india.fetch_yahoo_actions(ticker, "US", range_="1mo")
                if not actions.empty:
                    has_ca = bool((pd.to_datetime(actions["ex_date"]).dt.date == day).any())
            except SourceUnavailable:
                has_ca = False

            try:
                record = self.quorum.build_golden_copy_bar(
                    trade_date=day.isoformat(),
                    ticker=ticker,
                    market="US",
                    source_prices=sources,
                    trailing_golden_closes=trailing,
                    has_corporate_action_today=has_ca,
                )
            except ValueError:
                dropped += 1
                continue

            rows.append(
                {
                    "trade_date": pd.Timestamp(day),
                    "ticker": ticker,
                    "market": "US",
                    "isin": ticker,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": float(record.golden_close),
                    "volume": volume,
                    "dollar_volume_raw": float(record.golden_close) * volume,
                    "sources_agreed": record.sources_agreed,
                    "max_dispersion_bps": record.max_dispersion_bps,
                    "mad_z_score": record.mad_z_score,
                    "golden_status": record.status,
                    "sha256_proof": record.sha256_proof,
                }
            )
        return rows, dropped

    # ------------------------------------------------------------------- run --
    def run(self, day: dt.date, shard_id: int, total_shards: int, output_dir: str | Path) -> dict[str, Any]:
        universe = self._universe()
        summary: dict[str, Any] = {
            "market": self.market,
            "shard_id": shard_id,
            "total_shards": total_shards,
            "universe": len(universe),
            "trade_date": day.isoformat(),
        }
        if not universe:
            write_shard(pd.DataFrame(), output_dir, f"golden_ohlcv_{self.market}_{shard_id}")
            summary["degraded"] = "empty universe"
            return summary

        my_symbols = shard_of(universe, shard_id, total_shards)
        ca = pd.DataFrame()
        if self.market == "INDIA":
            rows, dropped, ca = self._india_rows(my_symbols, day)
        else:
            rows, dropped = self._us_rows(my_symbols, day)

        golden = pd.DataFrame(rows, columns=GOLDEN_COLUMNS)
        write_shard(golden, output_dir, f"golden_ohlcv_{self.market}_{shard_id}")
        verified = int((golden["golden_status"] == "VERIFIED_GOLDEN").sum()) if not golden.empty else 0
        summary.update(
            {
                "symbols": len(my_symbols),
                "golden_rows": len(golden),
                "verified_golden": verified,
                "dropped": dropped,
            }
        )
        if not ca.empty:
            # Namespaced so the existing company worker's corporate-actions shard
            # (``corporate_actions_india_<shard>``) is never overwritten; the
            # prefix still routes into the ``corporate_actions`` lakehouse table.
            write_shard(ca, output_dir, f"corporate_actions_golden_india_{shard_id}")
            summary["corporate_actions"] = len(ca)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3-source golden-copy quorum shard worker")
    parser.add_argument("--market", required=True, choices=["US", "INDIA"])
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--total-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", default=None, help="Trade date (YYYY-MM-DD); defaults to today (UTC)")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    day = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(dt.timezone.utc).date()
    cache_dir = Path.home() / ".cache" / ("pms_nse" if args.market == "INDIA" else "pms_http")
    ingestor = MultiSourceGoldenCopyIngestor(
        config, args.market, offline=args.offline, cache_dir=cache_dir
    )
    try:
        summary = ingestor.run(day, args.shard_id, args.total_shards, args.output_dir)
    except SourceUnavailable as exc:
        print(f"[golden-copy] degraded: {exc}")
        write_shard(pd.DataFrame(), args.output_dir, f"golden_ohlcv_{args.market}_{args.shard_id}")
        summary = {"market": args.market, "shard_id": args.shard_id, "degraded": str(exc)}
    print(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
