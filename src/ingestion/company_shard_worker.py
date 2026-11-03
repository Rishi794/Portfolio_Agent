"""Unified company-shard worker (Stage 2A of the parallel DAG).

Each of the 10 matrix jobs ([US-0..4, INDIA-0..4]) writes three Zstd Parquet shards:

    ohlcv_<market>_<shard>.parquet             -> market_prices
    corporate_actions_<market>_<shard>.parquet -> corporate_actions
    fundamentals_<market>_<shard>.parquet      -> fundamentals_pit (+ business_embedding)

Raw exchange OHLCV and vendor-adjusted Yahoo series are kept distinct. Corporate
actions are applied later by the reducer using only actions known by its as-of time,
so retrospective corrections do not contaminate point-in-time backtests.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..filtering.embedding_projector import EventProjector, get_backend
from ..filtering.self_calibrating_engine import SelfCalibratingQuantEngine
from ..storage.price_integrity import clean_ohlc_rows
from .base import HttpClient, write_shard
from .fundamentals_edgar_indian_xbrl import (
    NSE_HEADERS,
    SEC_HEADERS,
    IndiaFundamentalsParser,
    USEdgarFundamentalsParser,
    load_price_panel,
)
from .history_coverage import history_groups, read_market_coverage
from .market_data import MarketDataIngestor, trailing_trading_days
from .nse_bse_bhavcopy_corporate_actions import IndianBhavcopyAndCorporateActionIngestor
from .universe import UniverseLoader, shard_of


class UnifiedCompanyShardWorker:
    def __init__(
        self,
        config: Config,
        market: str,
        shard_id: int,
        total_shards: int,
        output_dir: Path,
        *,
        lake_dir: Path | None = None,
        offline: bool = False,
        lookback_days: int = 5,
        coverage_manifest: Path | None = None,
        force_full_history: bool = False,
        embed_backend: str = "auto",
        limit: int | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        self.config = config
        self.market = market
        self.shard_id = int(shard_id)
        self.total_shards = int(total_shards)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lake_dir = Path(lake_dir) if lake_dir else None
        self.offline = offline
        self.lookback_days = int(lookback_days)
        self.minimum_bars = int(config.e("historical_ingestion.min_bars_for_incremental", 512))
        self.cold_start_days = int(config.e("historical_ingestion.cold_start_calendar_days", 1825))
        self.delta_sessions = int(config.e("historical_ingestion.warm_delta_sessions", 5))
        self.minimum_warm_days = max(self.lookback_days, self.delta_sessions * 2 + 1)
        self.max_official_sessions = int(
            config.e("historical_ingestion.max_india_official_refresh_sessions", 30)
        )
        self.force_full_history = bool(force_full_history or self.lookback_days > 800)
        self.coverage_manifest = Path(coverage_manifest) if coverage_manifest else None
        self.history_coverage = read_market_coverage(self.coverage_manifest, market)
        self.cold_start_tickers: set[str] = set()
        self.limit = limit
        self.symbols = symbols

        headers = SEC_HEADERS if market == "US" else NSE_HEADERS
        self.client = HttpClient(headers=headers, cache_dir=Path.home() / ".cache" / "pms_http", offline=offline)
        self.client.headers.update(headers)
        self.sec_client = HttpClient(headers=SEC_HEADERS, cache_dir=Path.home() / ".cache" / "pms_sec", offline=offline)
        self.sec_client.headers.update(SEC_HEADERS)
        self.nse_client = HttpClient(headers=NSE_HEADERS, cache_dir=Path.home() / ".cache" / "pms_nse", offline=offline)
        self.nse_client.headers.update(NSE_HEADERS)

        self.market_data = MarketDataIngestor(config, self.client)
        self.india_actions = IndianBhavcopyAndCorporateActionIngestor(config, self.nse_client)
        self.engine = SelfCalibratingQuantEngine(config)
        self.projector = EventProjector(config, backend=get_backend(embed_backend))

    # ------------------------------------------------------------------ setup ---
    def universe(self) -> pd.DataFrame:
        if self.symbols:
            return pd.DataFrame({"ticker": self.symbols, "market": self.market, "name": self.symbols})
        loader = UniverseLoader(self.config, self.client)
        frame = loader.load(self.market, limit=self.limit)
        return frame

    # --------------------------------------------------------------- ingestion ---
    def ingest_prices(self, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        cold, warm = history_groups(
            symbols,
            self.history_coverage,
            minimum_bars=self.minimum_bars,
            force_full_history=self.force_full_history,
        )
        self.cold_start_tickers = set(cold)
        warm_calendar_days = self._warm_calendar_days(warm)

        if self.market == "US":
            price_frames = []
            if cold:
                frame = self.market_data.ingest_us(
                    cold, range_=_range_from_days(self.cold_start_days)
                )
                if not frame.empty:
                    price_frames.append(frame)
            if warm:
                frame = self.market_data.ingest_us(
                    warm, range_=_range_from_days(warm_calendar_days)
                )
                if not frame.empty:
                    price_frames.append(frame)

            action_frames = []
            # Price deltas are small, but action history has no reliable "no events"
            # coverage marker. Re-read 5y for every symbol so missing historical
            # actions (including prior partial runs) are repaired idempotently.
            for symbol in symbols:
                frame = self.india_actions.fetch_yahoo_actions(symbol, "US", range_="5y")
                if not frame.empty:
                    action_frames.append(frame)
            prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
            actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
            return self._dedupe_prices(prices), self._dedupe_actions(actions)

        official_sessions = self.max_official_sessions if cold else min(
            self.max_official_sessions, max(self.delta_sessions, warm_calendar_days)
        )
        days = trailing_trading_days(dt.date.today(), official_sessions)
        bhavcopy_frames = []
        for day in days:
            frame = self.india_actions.fetch_unified_india_bhavcopy(day)
            if not frame.empty:
                bhavcopy_frames.append(frame[frame["ticker"].isin(symbols)])
        official_prices = pd.concat(bhavcopy_frames, ignore_index=True) if bhavcopy_frames else pd.DataFrame()
        yahoo_history_frames = []
        if cold:
            frame = self.market_data.ingest_yahoo_history(
                cold, "INDIA", range_=_range_from_days(self.cold_start_days)
            )
            if not frame.empty:
                yahoo_history_frames.append(frame)
        if warm:
            frame = self.market_data.ingest_yahoo_history(
                warm, "INDIA", range_=_range_from_days(warm_calendar_days)
            )
            if not frame.empty:
                yahoo_history_frames.append(frame)
        price_frames = [*yahoo_history_frames]
        if not official_prices.empty:
            # Official exchange rows are appended last and win same-session ties.
            price_frames.append(official_prices)
        prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()

        nse_actions = self.india_actions.fetch_nse_corporate_actions(
            dt.date.today() - dt.timedelta(days=self.cold_start_days), dt.date.today() + dt.timedelta(days=2)
        )
        yahoo_actions_frames = []
        for symbol in symbols:
            frame = self.india_actions.fetch_yahoo_actions(symbol, "INDIA", range_="5y")
            if not frame.empty:
                yahoo_actions_frames.append(frame)
        yahoo_actions = (
            pd.concat(yahoo_actions_frames, ignore_index=True)
            if yahoo_actions_frames
            else pd.DataFrame()
        )
        if not nse_actions.empty and not prices.empty:
            nse_actions = nse_actions[nse_actions["ticker"].isin(prices["ticker"].unique())]
        actions = pd.concat(
            [frame for frame in (nse_actions, yahoo_actions) if not frame.empty],
            ignore_index=True,
        ) if not nse_actions.empty or not yahoo_actions.empty else pd.DataFrame()
        return self._dedupe_prices(prices), self._dedupe_actions(actions)

    def _warm_calendar_days(self, symbols: list[str]) -> int:
        """Cover missed sessions plus an overlap, bounded by the cold-start range."""
        if not symbols:
            return max(self.minimum_warm_days, 15)
        today = dt.date.today()
        gaps = []
        for symbol in symbols:
            record = self.history_coverage.get(symbol, {})
            try:
                last_date = dt.date.fromisoformat(str(record["last_trade_date"]))
            except (KeyError, TypeError, ValueError):
                continue
            gaps.append(max((today - last_date).days, 0))
        days = max([self.minimum_warm_days, *(gap + 3 for gap in gaps)])
        return min(days, self.cold_start_days)

    @staticmethod
    def _dedupe_prices(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame() if frame is None else frame
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out, _removed = clean_ohlc_rows(out)
        return (
            out.sort_values(["ticker", "date"])
            .drop_duplicates(["ticker", "date"], keep="last")
            .reset_index(drop=True)
        )

    @staticmethod
    def _dedupe_actions(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame() if frame is None else frame
        out = frame.copy()
        out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.normalize()
        # Keep the first row so an official NSE record wins over the Yahoo copy.
        return out.drop_duplicates(
            subset=["ticker", "ex_date", "action_type"], keep="first"
        ).reset_index(drop=True)

    def ingest_fundamentals(
        self,
        universe: pd.DataFrame,
        *,
        history_symbols: set[str] | None = None,
    ) -> pd.DataFrame:
        close_wide = pd.DataFrame()
        raw_close_wide = pd.DataFrame()
        volume_wide = pd.DataFrame()
        market_returns = pd.Series(dtype=float)
        if self.lake_dir:
            if history_symbols:
                close_wide, volume_wide, market_returns, raw_close_wide = load_price_panel(
                    self.lake_dir, self.market, include_raw_close=True
                )
            else:
                close_wide, volume_wide, market_returns = load_price_panel(self.lake_dir, self.market)

        rows: list[dict[str, Any]] = []
        if self.market == "US":
            parser = USEdgarFundamentalsParser(self.config, self.sec_client, self.engine)
            risk_free = float(self.config.e("optimizer.wacc.min_wacc", 0.06))
            for item in universe.to_dict("records"):
                cik = item.get("cik")
                if cik is None or (isinstance(cik, float) and np.isnan(cik)):
                    continue
                try:
                    ticker = str(item["ticker"])
                    sector = str(item.get("gics_sector", "UNKNOWN") or "UNKNOWN")
                    if history_symbols and ticker in history_symbols:
                        history = parser.parse_history(
                            ticker,
                            int(cik),
                            sector=sector,
                            close_wide=close_wide if not close_wide.empty else None,
                            raw_close_wide=raw_close_wide if not raw_close_wide.empty else None,
                            volume_wide=volume_wide if not volume_wide.empty else None,
                            market_returns=market_returns if not market_returns.empty else None,
                            risk_free=risk_free,
                        )
                        if history:
                            rows.extend(history)
                            continue
                    row = parser.parse(
                        ticker,
                        int(cik),
                        sector=sector,
                        close_wide=close_wide if not close_wide.empty else None,
                        market_returns=market_returns if not market_returns.empty else None,
                        risk_free=risk_free,
                    )
                except Exception:
                    continue
                if row:
                    rows.append(row)
        else:
            parser = IndiaFundamentalsParser(self.config, self.nse_client)
            for item in universe.to_dict("records"):
                symbol = str(item.get("nse_symbol") or item["ticker"]).replace(".NS", "")
                try:
                    row = parser.parse(symbol, name=str(item.get("name", "")), sector=str(item.get("gics_sector", "UNKNOWN") or "UNKNOWN"))
                except Exception:
                    continue
                if row:
                    rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        texts = [
            str(row.get("business_description") or f"{row.get('ticker', '')} {row.get('gics_sector', '')}").strip()
            for row in frame.to_dict("records")
        ]
        unique_texts = list(dict.fromkeys(texts))
        vectors = self.projector.embed(unique_texts)
        vector_by_text = {text: vector.tolist() for text, vector in zip(unique_texts, vectors)}
        frame["business_embedding"] = [vector_by_text[text] for text in texts]
        return frame

    # -------------------------------------------------------------------- run ---
    def run(self) -> dict[str, int]:
        universe = self.universe()
        sharded = universe.iloc[shard_of(list(range(len(universe))), self.shard_id, self.total_shards)] if not universe.empty else universe
        symbols = [str(symbol) for symbol in sharded["ticker"].tolist()]

        prices, actions = self.ingest_prices(symbols)
        if not prices.empty:
            if "dollar_volume" not in prices.columns:
                prices["dollar_volume"] = pd.to_numeric(prices["close"], errors="coerce") * pd.to_numeric(
                    prices["volume"], errors="coerce"
                ).fillna(0.0)
            write_shard(prices, self.output_dir, f"ohlcv_{self.market.lower()}_{self.shard_id}")
        if not actions.empty:
            write_shard(actions, self.output_dir, f"corporate_actions_{self.market.lower()}_{self.shard_id}")

        fundamentals = self.ingest_fundamentals(
            sharded, history_symbols=self.cold_start_tickers
        )
        if not fundamentals.empty:
            write_shard(fundamentals, self.output_dir, f"fundamentals_{self.market.lower()}_{self.shard_id}")

        return {
            "symbols": len(symbols),
            "price_rows": int(len(prices)),
            "action_rows": int(len(actions)),
            "fundamental_rows": int(len(fundamentals)),
            "cold_start_symbols": len(self.cold_start_tickers),
            "warm_delta_symbols": max(len(symbols) - len(self.cold_start_tickers), 0),
        }


def _range_from_days(days: int) -> str:
    if days <= 30:
        return "1mo"
    if days <= 95:
        return "3mo"
    if days <= 190:
        return "6mo"
    if days <= 400:
        return "1y"
    if days <= 800:
        return "2y"
    return "5y"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified company shard worker (OHLCV + actions + fundamentals)")
    parser.add_argument("--market", required=True, choices=["US", "INDIA"])
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--total-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lake-dir", type=Path, default=None)
    parser.add_argument("--lookback-days", type=int, default=5,
                        help="Warm-symbol overlap in calendar days; cold symbols still backfill 5y")
    parser.add_argument("--coverage-manifest", type=Path, default=None)
    parser.add_argument("--force-full-history", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--backend", default="auto", choices=["auto", "onnx", "hashing"])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    worker = UnifiedCompanyShardWorker(
        config,
        args.market,
        args.shard_id,
        args.total_shards,
        args.output_dir,
        lake_dir=args.lake_dir,
        offline=args.offline,
        lookback_days=args.lookback_days,
        coverage_manifest=args.coverage_manifest,
        force_full_history=args.force_full_history,
        embed_backend=args.backend,
        limit=args.limit,
        symbols=args.symbols,
    )
    summary = worker.run()
    print(f"[{args.market} shard {args.shard_id}/{args.total_shards}] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
