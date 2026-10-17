"""Point-in-Time macro (FRED/ALFRED vintages) + cross-asset external factors.

Leakage control
---------------
FRED's ``realtime_start``/``realtime_end`` parameters expose the *vintage* of every
observation, i.e. what a market participant could have known on a given date. We
store ``(series_id, observation_date, vintage_timestamp)`` so the engine can always
reconstruct the information set as of any past date — this is what makes the
backtest and the live regime detector use the same code path.

Cross-asset prices (VIX, DXY, Brent, Copper, Gold, SOX, NDX, US10Y real yield) come
from the free Yahoo chart API with a Stooq CSV fallback; knowledge timestamps are
pinned to the close of the trading day, never to the *observation* date alone.
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

FRED_OBSERVATIONS = "https://api.stlouisfed.org/fred/series/observations"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"


class MacroExternalIngestor:
    def __init__(self, config: Config, client: HttpClient, fred_api_key: str | None = None) -> None:
        self.config = config
        self.client = client
        self.fred_api_key = fred_api_key or ""
        self.vintage_hour_utc = int(config.e("macro.vintage_hour_utc", 23))

    # ------------------------------------------------------------- FRED / ALFRED
    def fetch_fred_pit(
        self,
        as_of_date: dt.date | str,
        series_ids: dict[str, str] | None = None,
        limit: int = 300,
    ) -> pd.DataFrame:
        if not self.fred_api_key:
            return pd.DataFrame(columns=["series_id", "observation_date", "vintage_timestamp", "value"])
        mapping = series_ids or dict(self.config.get("universe.fred_series", {}) or {})
        as_of = as_of_date.isoformat() if isinstance(as_of_date, dt.date) else str(as_of_date)
        rows: list[dict[str, Any]] = []
        for series_id, label in mapping.items():
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "realtime_start": as_of,
                "realtime_end": as_of,
                "limit": limit,
                "sort_order": "desc",
            }
            try:
                payload = self.client.get_json(FRED_OBSERVATIONS, params=params)
            except SourceUnavailable:
                continue
            vintage_date = payload.get("realtime_start") or as_of
            for obs in payload.get("observations", []):
                raw = obs.get("value")
                if raw in (None, "."):
                    continue
                # Conservatively stamp the vintage at the end of the release day (UTC).
                vintage = dt.datetime.fromisoformat(vintage_date).replace(
                    hour=self.vintage_hour_utc, minute=0, second=0, tzinfo=dt.timezone.utc
                )
                rows.append(
                    {
                        "series_id": label,
                        "observation_date": obs["date"],
                        "vintage_timestamp": vintage,
                        "value": float(raw),
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- cross-assets
    def fetch_yahoo_series(self, symbol: str, series_id: str, range_: str = "2y") -> pd.DataFrame:
        payload = self.client.get_json(
            YAHOO_CHART.format(symbol=symbol), params={"range": range_, "interval": "1d"}
        )
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame()
        node = result[0]
        stamps = node.get("timestamp") or []
        quote = ((node.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        rows = []
        for idx, ts in enumerate(stamps):
            if idx >= len(closes) or closes[idx] is None:
                continue
            stamp = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
            day = stamp.date()
            rows.append(
                {
                    "series_id": series_id,
                    "observation_date": day,
                    "vintage_timestamp": dt.datetime.combine(
                        day, dt.time(hour=self.vintage_hour_utc, minute=0), tzinfo=dt.timezone.utc
                    ),
                    "value": float(closes[idx]),
                }
            )
        return pd.DataFrame(rows)

    def fetch_stooq_series(self, symbol: str, series_id: str) -> pd.DataFrame:
        text = self.client.get_text(STOOQ_CSV.format(symbol=symbol.lower()))
        if not text or "Date" not in text.splitlines()[0]:
            return pd.DataFrame()
        frame = pd.read_csv(StringIO(text))
        rows = []
        for _, row in frame.iterrows():
            try:
                day = pd.to_datetime(row["Date"]).date()
                value = float(row["Close"])
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "series_id": series_id,
                    "observation_date": day,
                    "vintage_timestamp": dt.datetime.combine(
                        day, dt.time(hour=self.vintage_hour_utc), tzinfo=dt.timezone.utc
                    ),
                    "value": value,
                }
            )
        return pd.DataFrame(rows)

    def fetch_cross_assets(self, symbols: dict[str, str] | None = None) -> pd.DataFrame:
        mapping = dict(symbols or self.config.get("universe.cross_assets.symbols", {}) or {})
        frames: list[pd.DataFrame] = []
        for series_id, symbol in mapping.items():
            try:
                frame = self.fetch_yahoo_series(symbol, series_id)
            except SourceUnavailable:
                frame = pd.DataFrame()
            if frame.empty:
                fallback = (self.config.get("universe.cross_assets.sources", {}) or {}).get("fallback")
                if fallback == "stooq_csv":
                    try:
                        frame = self.fetch_stooq_series(symbol.lstrip("^"), series_id)
                    except SourceUnavailable:
                        frame = pd.DataFrame()
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ build ---
    def build(self, as_of_date: dt.date | str | None = None) -> pd.DataFrame:
        as_of = as_of_date or dt.date.today().isoformat()
        macro = self.fetch_fred_pit(as_of)
        external = self.fetch_cross_assets()
        frames = [frame for frame in (macro, external) if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=["series_id", "observation_date", "vintage_timestamp", "value", "z_score_252d"])
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["series_id", "observation_date", "vintage_timestamp"]
        )
        window = int(self.config.e("macro.zscore_lookback_days", 252))
        winsor = float(self.config.e("macro.winsor_quantile", 0.01))
        combined["z_score_252d"] = _rolling_z(combined, window=window, winsor=winsor)
        return combined.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _rolling_z(frame: pd.DataFrame, *, window: int, winsor: float) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for series_id, group in frame.groupby("series_id", sort=False):
        series = group.sort_values("observation_date")["value"].astype(float)
        if winsor > 0:
            lower, upper = series.quantile(winsor), series.quantile(1 - winsor)
            series = series.clip(lower, upper)
        mean = series.rolling(window, min_periods=max(10, window // 5)).mean()
        std = series.rolling(window, min_periods=max(10, window // 5)).std()
        z = (series - mean) / std.replace(0.0, np.nan)
        out.loc[group.sort_values("observation_date").index] = z.to_numpy()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Macro (FRED ALFRED PIT) + cross-asset ingestion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", default=None, help="Vintage date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    import os

    config = Config.load()
    client = HttpClient(cache_dir=Path.home() / ".cache" / "pms_http", offline=args.offline)
    ingestor = MacroExternalIngestor(config, client, fred_api_key=os.getenv("FRED_API_KEY"))
    frame = ingestor.build(args.as_of)
    path = write_shard(frame, args.output_dir, "macro_external_pit")
    print(f"wrote {len(frame)} macro/external rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
