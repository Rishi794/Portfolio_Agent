"""Market data: OHLCV ingestion for US (NYSE/NASDAQ) and India (NSE/BSE).

Sources (all free, no paid subscription):
* Yahoo Finance chart API  — ``query1.finance.yahoo.com/v8/finance/chart``
* Stooq daily CSV          — fallback for US symbols
* NSE ``sec_bhavdata_full`` — full-day security-wise bhavcopy for India

Splits/dividends: the Yahoo payload's ``adjclose`` is stored in ``adj_close``; the
lakehouse keeps *raw* OHLCV and the engine consumes ``adj_close`` for returns so
that historical returns are already split/dividend adjusted (no hardcoded factor).
"""

from __future__ import annotations

import argparse
import datetime as dt
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .base import HttpClient, SourceUnavailable, write_shard
from .universe import knowledge_timestamp_for_close

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"


class MarketDataIngestor:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client

    # ------------------------------------------------------------------ Yahoo ---
    def fetch_yahoo(self, symbol: str, market: str, range_: str = "2y", interval: str = "1d") -> pd.DataFrame:
        payload = self.client.get_json(
            YAHOO_CHART.format(symbol=symbol), params={"range": range_, "interval": interval}
        )
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame()
        node = result[0]
        stamps = node.get("timestamp") or []
        quote = ((node.get("indicators") or {}).get("quote") or [{}])[0]
        adj = ((node.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
        rows = []
        for idx, ts in enumerate(stamps):
            day = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
            close = _pick(quote.get("close"), idx)
            if close is None or not np.isfinite(close):
                continue
            rows.append(
                {
                    "ticker": symbol,
                    "market": market,
                    "date": day,
                    "open": _pick(quote.get("open"), idx),
                    "high": _pick(quote.get("high"), idx),
                    "low": _pick(quote.get("low"), idx),
                    "close": close,
                    "volume": _pick(quote.get("volume"), idx),
                    "adj_close": _pick(adj, idx) if adj else close,
                    "knowledge_timestamp": knowledge_timestamp_for_close(day, market, self.config),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ Stooq ---
    def fetch_stooq(self, symbol: str, market: str) -> pd.DataFrame:
        template = (self.config.get("universe.market_data.US", {}) or {}).get(
            "stooq_symbol_template", "{symbol_lower}.us"
        )
        text = self.client.get_text(STOOQ_CSV.format(symbol=template.format(symbol_lower=symbol.lower())))
        if "Date" not in text.splitlines()[0]:
            return pd.DataFrame()
        frame = pd.read_csv(StringIO(text))
        frame = frame.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["ticker"] = symbol
        frame["market"] = market
        frame["adj_close"] = frame["close"]
        frame["knowledge_timestamp"] = [knowledge_timestamp_for_close(d, market, self.config) for d in frame["date"]]
        return frame[["ticker", "market", "date", "open", "high", "low", "close", "volume", "adj_close", "knowledge_timestamp"]]

    # ------------------------------------------------------------------ India ---
    def fetch_nse_bhavcopy(self, day: dt.date, market: str = "INDIA") -> pd.DataFrame:
        template = (self.config.get("universe.market_data.INDIA", {}) or {}).get("bhavcopy_url_template", "")
        url = template.format(ddmmyyyy=day.strftime("%d%m%Y"))
        text = self.client.get_text(url)
        lines = text.splitlines()
        if len(lines) < 2:
            return pd.DataFrame()
        frame = pd.read_csv(StringIO(text))
        frame.columns = [col.strip() for col in frame.columns]
        series_col = "SERIES" if "SERIES" in frame.columns else None
        if series_col:
            frame = frame[frame[series_col].astype(str).str.strip().isin(["EQ", "BE", "BZ"])]
        symbol_col = "SYMBOL" if "SYMBOL" in frame.columns else "Symbol"
        suffix = str(self.config.market("INDIA").get("symbol_suffix", ".NS"))
        out = pd.DataFrame(
            {
                "ticker": frame[symbol_col].astype(str).str.strip() + suffix,
                "market": market,
                "date": day,
                "open": _num(frame, ["OPEN_PRICE", "Open"]),
                "high": _num(frame, ["HIGH_PRICE", "High"]),
                "low": _num(frame, ["LOW_PRICE", "Low"]),
                "close": _num(frame, ["CLOSE_PRICE", "Close"]),
                "volume": _num(frame, ["TTL_TRD_QNTY", "TotalTradedQuantity", "Volume"]),
            }
        )
        out["adj_close"] = out["close"]
        out["knowledge_timestamp"] = knowledge_timestamp_for_close(day, market, self.config)
        return out.dropna(subset=["close"]).reset_index(drop=True)

    # ------------------------------------------------------------------- main ---
    def ingest_us(self, symbols: list[str], range_: str = "2y") -> pd.DataFrame:
        """Yahoo with dynamic trailing split adjustment, falling back to Stooq per symbol."""
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            try:
                frame = self.fetch_yahoo(symbol, "US", range_=range_)
            except SourceUnavailable:
                try:
                    frame = self.fetch_stooq(symbol, "US")
                except SourceUnavailable:
                    continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def ingest_india(self, symbols: list[str], days: list[dt.date]) -> pd.DataFrame:
        frames = []
        for day in days:
            try:
                frame = self.fetch_nse_bhavcopy(day)
            except SourceUnavailable:
                continue
            if not frame.empty:
                frames.append(frame[frame["ticker"].isin(symbols)])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])


def _pick(values: list[Any] | None, idx: int) -> float | None:
    if not values or idx >= len(values):
        return None
    value = values[idx]
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _num(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name].astype(str).str.replace(",", ""), errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def trailing_trading_days(as_of: dt.date, n: int) -> list[dt.date]:
    """Simple backward calendar walk; the exchange-calendar gate trims non-sessions."""
    days = []
    day = as_of
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day -= dt.timedelta(days=1)
    return sorted(days)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OHLCV ingestion (US + India)")
    parser.add_argument("--market", required=True, choices=["US", "INDIA"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--universe-csv", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    client = HttpClient(cache_dir=Path.home() / ".cache" / "pms_http", offline=args.offline)
    ingestor = MarketDataIngestor(config, client)

    if args.symbols:
        symbols = list(args.symbols)
    else:
        from .universe import UniverseLoader

        loader = UniverseLoader(config, client)
        universe = loader.load(args.market, static_csv=args.universe_csv)
        symbols = universe["ticker"].tolist()

    if args.market == "US":
        frame = ingestor.ingest_us(symbols)
    else:
        frame = ingestor.ingest_india(symbols, trailing_trading_days(dt.date.today(), args.lookback_days))
    path = write_shard(frame, args.output_dir, f"market_prices_{args.market.lower()}")
    print(f"wrote {len(frame)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
