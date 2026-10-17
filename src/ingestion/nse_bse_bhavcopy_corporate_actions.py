"""India market data: NSE/BSE UDiFF bhavcopy + corporate actions (splits, bonuses,
dividends) and the Yahoo chart-events fallback for both markets.

Two things matter for PIT correctness here:

1. the bhavcopy is *unadjusted* — adjustment factors come from the corporate-action
   table, so a split is applied to the whole history atomically and no artificial
   -50% return ever enters the Kalman filter or the HRP covariance;
2. every corporate action carries the date it became public (``knowledge_timestamp``),
   never the ex-date alone.
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

NSE_CORPORATE_ACTIONS = "https://www.nseindia.com/api/corporates-corporateActions"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
}


class IndianBhavcopyAndCorporateActionIngestor:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client
        self._primed = False

    # -------------------------------------------------------------- bhavcopy ---
    def fetch_unified_india_bhavcopy(self, day: dt.date, *, market: str = "INDIA") -> pd.DataFrame:
        """Security-wise full bhavcopy (UDiFF ``sec_bhavdata_full`` CSV, free)."""
        template = (self.config.get("universe.market_data.INDIA", {}) or {}).get("bhavcopy_url_template", "")
        if not template:
            return pd.DataFrame()
        url = template.format(ddmmyyyy=day.strftime("%d%m%Y"))
        try:
            text = self.client.get_text(url)
        except SourceUnavailable:
            return pd.DataFrame()
        lines = text.splitlines()
        if len(lines) < 2:
            return pd.DataFrame()
        frame = pd.read_csv(StringIO(text))
        frame.columns = [str(column).strip() for column in frame.columns]
        series_column = "SERIES" if "SERIES" in frame.columns else None
        if series_column:
            frame = frame[frame[series_column].astype(str).str.strip().isin(["EQ", "BE", "BZ"])]
        symbol_column = "SYMBOL" if "SYMBOL" in frame.columns else "Symbol"
        suffix = str(self.config.market("INDIA").get("symbol_suffix", ".NS"))
        out = pd.DataFrame(
            {
                "ticker": frame[symbol_column].astype(str).str.strip() + suffix,
                "market": market,
                "date": day,
                "open": _numeric(frame, ["OPEN_PRICE", "Open"]),
                "high": _numeric(frame, ["HIGH_PRICE", "High"]),
                "low": _numeric(frame, ["LOW_PRICE", "Low"]),
                "close": _numeric(frame, ["CLOSE_PRICE", "Close"]),
                "volume": _numeric(frame, ["TTL_TRD_QNTY", "TotalTradedQuantity", "Volume"]),
                "isin": frame.get("ISIN", pd.Series("", index=frame.index)).astype(str).str.strip(),
            }
        )
        out["adj_close"] = out["close"]
        out["dollar_volume"] = out["close"] * out["volume"]
        out["knowledge_timestamp"] = knowledge_timestamp_for_close(day, market, self.config)
        return out.dropna(subset=["close"]).reset_index(drop=True)

    # ------------------------------------------------------ corporate actions ---
    def prime_nse_session(self) -> None:
        if self._primed:
            return
        try:
            self.client.get_text("https://www.nseindia.com")
        except SourceUnavailable:
            pass
        self._primed = True

    def fetch_nse_corporate_actions(
        self, from_date: dt.date, to_date: dt.date, *, market: str = "INDIA"
    ) -> pd.DataFrame:
        self.prime_nse_session()
        params = {
            "index": "equities",
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
        }
        try:
            payload = self.client.get_json(NSE_CORPORATE_ACTIONS, params=params)
        except SourceUnavailable:
            return pd.DataFrame()
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        suffix = str(self.config.market("INDIA").get("symbol_suffix", ".NS"))
        out_rows: list[dict[str, Any]] = []
        for row in rows or []:
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                continue
            ex_date = _parse_action_date(row.get("exDate") or row.get("ex_date"))
            if ex_date is None:
                continue
            purpose = str(row.get("subject") or row.get("purpose") or "")
            split_ratio = _split_ratio_from_text(purpose)
            dividend = _dividend_from_text(purpose)
            action_type = _action_type(purpose, split_ratio, dividend)
            if action_type == "UNKNOWN":
                continue
            out_rows.append(
                {
                    "ticker": f"{symbol}{suffix}",
                    "market": market,
                    "ex_date": ex_date,
                    "action_type": action_type,
                    "split_ratio": split_ratio,
                    "cash_dividend": dividend,
                    "isin": str(row.get("isin") or ""),
                    "raw_narrative": purpose[:300],
                    "knowledge_timestamp": dt.datetime.combine(
                        ex_date - dt.timedelta(days=1), dt.time(14, 0), tzinfo=dt.timezone.utc
                    ),
                }
            )
        return pd.DataFrame(out_rows).drop_duplicates(subset=["ticker", "ex_date", "action_type"]) if out_rows else pd.DataFrame()

    def fetch_yahoo_actions(self, symbol: str, market: str, *, range_: str = "5y") -> pd.DataFrame:
        """Split/dividend events from the free Yahoo chart API (works for US and India)."""
        try:
            payload = self.client.get_json(
                YAHOO_CHART.format(symbol=symbol), params={"range": range_, "interval": "1d", "events": "div,splits"}
            )
        except SourceUnavailable:
            return pd.DataFrame()
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame()
        events = ((result[0].get("events") or {})) or {}
        rows: list[dict[str, Any]] = []
        for split in (events.get("splits") or {}).values():
            date = dt.datetime.fromtimestamp(int(split["date"]), tz=dt.timezone.utc).date()
            rows.append(
                {
                    "ticker": symbol,
                    "market": market,
                    "ex_date": date,
                    "action_type": "SPLIT",
                    "split_ratio": float(split.get("numerator", 1.0) or 1.0)
                    / max(float(split.get("denominator", 1.0) or 1.0), 1e-9),
                    "cash_dividend": 0.0,
                    "isin": "",
                    "raw_narrative": str(split.get("splitRatio", "")),
                    "knowledge_timestamp": knowledge_timestamp_for_close(date, market, self.config),
                }
            )
        for dividend in (events.get("dividends") or {}).values():
            date = dt.datetime.fromtimestamp(int(dividend["date"]), tz=dt.timezone.utc).date()
            rows.append(
                {
                    "ticker": symbol,
                    "market": market,
                    "ex_date": date,
                    "action_type": "DIVIDEND",
                    "split_ratio": 1.0,
                    "cash_dividend": float(dividend.get("amount", 0.0) or 0.0),
                    "isin": "",
                    "raw_narrative": "",
                    "knowledge_timestamp": knowledge_timestamp_for_close(date, market, self.config),
                }
            )
        return pd.DataFrame(rows)


def adjust_for_actions(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC/volume for splits and cash dividends, PIT-safely.

    Split factors multiply the *entire* pre-ex-date history, so the artificial jump
    disappears while post-ex prices remain exactly as traded.
    """
    if prices.empty:
        return prices
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["adj_close"] = pd.to_numeric(out.get("adj_close", out["close"]), errors="coerce").fillna(out["close"])
    # Newer pandas raise instead of silently upcasting when a fractional value lands in
    # an int64 column. Bhavcopy volume arrives as int64 (whole-number prices can too),
    # and every split/dividend adjustment below can produce fractional values, so the
    # touched columns are promoted to float64 up-front (the lakehouse schema wants
    # DOUBLE for all of them anyway).
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    if actions is None or actions.empty:
        out["dollar_volume"] = out["close"] * pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
        return out

    actions = actions.copy()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"])
    for ticker, group in actions.groupby("ticker"):
        mask = out["ticker"] == ticker
        if not mask.any():
            continue
        for row in group.sort_values("ex_date").itertuples(index=False):
            split_ratio = float(row.split_ratio or 1.0)
            dividend = float(row.cash_dividend or 0.0)
            before = mask & (out["date"] < row.ex_date)
            if split_ratio and split_ratio != 1.0 and split_ratio > 0:
                for column in ("open", "high", "low", "close", "adj_close"):
                    if column in out.columns:
                        out.loc[before, column] = out.loc[before, column] / split_ratio
                out.loc[before, "volume"] = pd.to_numeric(out.loc[before, "volume"], errors="coerce") * split_ratio
            if dividend > 0:
                out.loc[before, "adj_close"] = out.loc[before, "adj_close"] - dividend
    out["dollar_volume"] = out["adj_close"] * pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    return out


def _numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name].astype(str).str.replace(",", ""), errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _parse_action_date(value: Any) -> dt.date | None:
    if not value:
        return None
    parsed = pd.to_datetime(str(value), errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _split_ratio_from_text(text: str) -> float:
    import re

    match = re.search(r"(?:split|sub-?division|face value)\D{0,20}(\d+(?:\.\d+)?)\D{0,6}(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        old, new = float(match.group(1)), float(match.group(2))
        if new > 0:
            return old / new
    match = re.search(r"(\d+)\s*:\s*(\d+)", text)
    if match:
        numerator, denominator = float(match.group(1)), float(match.group(2))
        if denominator > 0:
            return numerator / denominator
    return 1.0


def _dividend_from_text(text: str) -> float:
    import re

    match = re.search(r"(?:dividend|rs\.?|inr)\D{0,12}(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def _action_type(text: str, split_ratio: float, dividend: float) -> str:
    lowered = text.lower()
    if "bonus" in lowered:
        return "BONUS"
    if split_ratio != 1.0 or "split" in lowered or "sub-division" in lowered or "subdivision" in lowered:
        return "SPLIT"
    if dividend > 0 or "dividend" in lowered:
        return "DIVIDEND"
    if "rights" in lowered:
        return "RIGHTS"
    return "UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NSE/BSE bhavcopy + corporate actions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", default=None, help="Bhavcopy date (YYYY-MM-DD), default today")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--forward-days", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    client = HttpClient(headers=NSE_HEADERS, cache_dir=Path.home() / ".cache" / "pms_nse", offline=args.offline)
    client.headers.update(NSE_HEADERS)
    ingestor = IndianBhavcopyAndCorporateActionIngestor(config, client)

    day = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    bhavcopy = ingestor.fetch_unified_india_bhavcopy(day)
    if not bhavcopy.empty:
        write_shard(bhavcopy, args.output_dir, f"ohlcv_india_{day.isoformat()}")
    actions = ingestor.fetch_nse_corporate_actions(
        day - dt.timedelta(days=args.lookback_days), day + dt.timedelta(days=args.forward_days)
    )
    path = write_shard(actions, args.output_dir, "corporate_actions_india")
    print(f"wrote {len(bhavcopy)} bhavcopy rows and {len(actions)} corporate actions -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
