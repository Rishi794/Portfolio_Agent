"""Unified company-shard worker (Stage 2A of the parallel DAG).

Each of the 10 matrix jobs ([US-0..4, INDIA-0..4]) writes three Zstd Parquet shards:

    ohlcv_<market>_<shard>.parquet             -> market_prices
    corporate_actions_<market>_<shard>.parquet -> corporate_actions
    fundamentals_<market>_<shard>.parquet      -> fundamentals_pit (+ business_embedding)

Splits/bonuses/dividends are back-adjusted *from the corporate-action table* before
the shard is written, so no artificial price jump ever reaches the Kalman filter, the
HRP covariance or the Corwin-Schultz spread estimator.
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
from .base import HttpClient, write_shard
from .fundamentals_edgar_indian_xbrl import (
    NSE_HEADERS,
    SEC_HEADERS,
    IndiaFundamentalsParser,
    USEdgarFundamentalsParser,
    load_price_panel,
)
from .market_data import MarketDataIngestor, trailing_trading_days
from .nse_bse_bhavcopy_corporate_actions import IndianBhavcopyAndCorporateActionIngestor, adjust_for_actions
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
        lookback_days: int = 420,
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
        if self.market == "US":
            prices = self.market_data.ingest_us(symbols, range_=_range_from_days(self.lookback_days))
            actions_frames = []
            for symbol in symbols:
                frame = self.india_actions.fetch_yahoo_actions(symbol, "US")
                if not frame.empty:
                    actions_frames.append(frame)
            actions = pd.concat(actions_frames, ignore_index=True) if actions_frames else pd.DataFrame()
            return prices, actions

        days = trailing_trading_days(dt.date.today(), min(self.lookback_days, 30))
        bhavcopy_frames = []
        for day in days:
            frame = self.india_actions.fetch_unified_india_bhavcopy(day)
            if not frame.empty:
                bhavcopy_frames.append(frame[frame["ticker"].isin(symbols)])
        prices = pd.concat(bhavcopy_frames, ignore_index=True) if bhavcopy_frames else pd.DataFrame()

        if prices.empty:  # Yahoo fallback for India
            prices = self.market_data.ingest_india(symbols, days)

        nse_actions = self.india_actions.fetch_nse_corporate_actions(
            dt.date.today() - dt.timedelta(days=self.lookback_days), dt.date.today() + dt.timedelta(days=5)
        )
        yahoo_actions_frames = [self.india_actions.fetch_yahoo_actions(symbol, "INDIA") for symbol in symbols[:60]]
        yahoo_actions = (
            pd.concat([frame for frame in yahoo_actions_frames if not frame.empty], ignore_index=True)
            if any(not frame.empty for frame in yahoo_actions_frames)
            else pd.DataFrame()
        )
        if not nse_actions.empty and not prices.empty:
            nse_actions = nse_actions[nse_actions["ticker"].isin(prices["ticker"].unique())]
        actions = pd.concat([nse_actions, yahoo_actions], ignore_index=True) if not nse_actions.empty else yahoo_actions
        return prices, actions

    def ingest_fundamentals(self, universe: pd.DataFrame) -> pd.DataFrame:
        close_wide = pd.DataFrame()
        market_returns = pd.Series(dtype=float)
        if self.lake_dir:
            close_wide, _, market_returns = load_price_panel(self.lake_dir, self.market)

        rows: list[dict[str, Any]] = []
        if self.market == "US":
            parser = USEdgarFundamentalsParser(self.config, self.sec_client, self.engine)
            risk_free = float(self.config.e("optimizer.wacc.min_wacc", 0.06))
            for item in universe.to_dict("records"):
                cik = item.get("cik")
                if cik is None or (isinstance(cik, float) and np.isnan(cik)):
                    continue
                try:
                    row = parser.parse(
                        str(item["ticker"]),
                        int(cik),
                        sector=str(item.get("gics_sector", "UNKNOWN") or "UNKNOWN"),
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
        vectors = self.projector.embed(texts)
        frame["business_embedding"] = [vector.tolist() for vector in vectors]
        return frame

    # -------------------------------------------------------------------- run ---
    def run(self) -> dict[str, int]:
        universe = self.universe()
        sharded = universe.iloc[shard_of(list(range(len(universe))), self.shard_id, self.total_shards)] if not universe.empty else universe
        symbols = [str(symbol) for symbol in sharded["ticker"].tolist()]

        prices, actions = self.ingest_prices(symbols)
        if not prices.empty:
            prices = adjust_for_actions(prices, actions)
            if "dollar_volume" not in prices.columns:
                prices["dollar_volume"] = pd.to_numeric(prices["close"], errors="coerce") * pd.to_numeric(
                    prices["volume"], errors="coerce"
                ).fillna(0.0)
            write_shard(prices, self.output_dir, f"ohlcv_{self.market.lower()}_{self.shard_id}")
        if not actions.empty:
            write_shard(actions, self.output_dir, f"corporate_actions_{self.market.lower()}_{self.shard_id}")

        fundamentals = self.ingest_fundamentals(sharded)
        if not fundamentals.empty:
            write_shard(fundamentals, self.output_dir, f"fundamentals_{self.market.lower()}_{self.shard_id}")

        return {
            "symbols": len(symbols),
            "price_rows": int(len(prices)),
            "action_rows": int(len(actions)),
            "fundamental_rows": int(len(fundamentals)),
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
    parser.add_argument("--lookback-days", type=int, default=420)
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
        embed_backend=args.backend,
        limit=args.limit,
        symbols=args.symbols,
    )
    summary = worker.run()
    print(f"[{args.market} shard {args.shard_id}/{args.total_shards}] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
