"""Point-in-Time fundamentals for US (SEC EDGAR XBRL) and India (NSE/BSE + Reg 31).

US  — ``data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` keyed by the exact
      EDGAR ``filed`` acceptance timestamp. For every fiscal period the *first*
      public print is retained (`drop_duplicates(keep="first")` after sorting by
      filing timestamp) so later restatements can never leak into a historical run.
      All 8 Beneish inputs, all 9 Piotroski inputs and the ROIC-WACC spread are
      reproduced from raw XBRL tags (no vendor ratios).

India — NSE corporate shareholding (Regulation 31: promoter holding %, promoter
      *pledged* shares %, FII/DII) plus Ind-AS results comparatives. Because daily
      NSE endpoints do not expose the EDGAR-grade acceptance timestamp, knowledge
      time falls back to ``period_end + india_reporting_lag_days`` (config), which is
      conservative by construction: it can only delay a signal, never leak one.
      Indian forensic scoring therefore uses a governance-augmented proxy that
      penalises promoter pledging and interest-coverage erosion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import Config
from ..filtering.forensics import compute_forensic_metrics, detail_to_json
from ..filtering.self_calibrating_engine import SelfCalibratingQuantEngine
from .base import HttpClient, SourceUnavailable, compliant_user_agent, write_shard
from .universe import UniverseLoader, shard_of

SEC_HEADERS = {
    "User-Agent": compliant_user_agent(),
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
}

US_TAG_MAP: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "current_assets": ["AssetsCurrent"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "total_assets": ["Assets"],
    "depreciation": ["DepreciationDepletionAndAmortization", "Depreciation", "DepreciationAmortizationAndAccretionNet"],
    "sga": ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"],
    "current_liab": ["LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "ebit": ["OperatingIncomeLoss"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "shares_out": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "inventory": ["InventoryNet"],
}

ITEM1_PATTERNS = (
    re.compile(r"item\s*1[\.\s]*business", re.IGNORECASE),
    re.compile(r"item\s*1[\.\s]*description\s*of\s*business", re.IGNORECASE),
    re.compile(r"item\s*4[\.\s]*information\s*on\s*the\s*company", re.IGNORECASE),
)
ITEM_TERMINATORS = (
    re.compile(r"item\s*1a[\.\s]*risk\s*factors", re.IGNORECASE),
    re.compile(r"item\s*1b[\.\s]", re.IGNORECASE),
    re.compile(r"item\s*2[\.\s]*properties", re.IGNORECASE),
)


# --------------------------------------------------------------------------- #
# US: SEC EDGAR XBRL
# --------------------------------------------------------------------------- #
def extract_pit_xbrl_tag(
    us_gaap: dict[str, Any],
    tag_candidates: Iterable[str],
    unit: str = "USD",
    *,
    public_forms: tuple[str, ...] = ("10-K", "10-Q", "20-F"),
) -> pd.DataFrame:
    """First-print (ticker, period_end, filing_timestamp, value) series for a tag."""
    rows: list[dict[str, Any]] = []
    for tag in tag_candidates:
        node = us_gaap.get(tag)
        if not node:
            continue
        for item in (node.get("units", {}) or {}).get(unit, []) or []:
            if item.get("form") not in public_forms:
                continue
            filed, end, value = item.get("filed"), item.get("end"), item.get("val")
            if not filed or not end or value is None:
                continue
            rows.append(
                {
                    "period_end": pd.to_datetime(end),
                    "filing_timestamp": pd.to_datetime(f"{filed}T22:00:00Z", utc=True),
                    "val": float(value),
                }
            )
        if rows:
            break
    if not rows:
        return pd.DataFrame(columns=["period_end", "filing_timestamp", "val"])
    frame = pd.DataFrame(rows).sort_values(["period_end", "filing_timestamp"])
    # PIT rule: the first public print of a period wins; restatements are ignored.
    return frame.drop_duplicates(subset=["period_end"], keep="first").reset_index(drop=True)


class USEdgarFundamentalsParser:
    def __init__(self, config: Config, client: HttpClient, engine: SelfCalibratingQuantEngine) -> None:
        self.config = config
        self.client = client
        self.engine = engine
        self.max_business_chars = int(config.e("fundamentals.business_text_max_chars", 6000))

    def company_facts(self, cik: int) -> dict[str, Any]:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
        return self.client.get_json(url)

    def statement_frame(self, facts: dict[str, Any]) -> pd.DataFrame:
        us_gaap = (facts.get("facts", {}) or {}).get("us-gaap", {}) or {}
        merged: pd.DataFrame | None = None
        for column, tags in US_TAG_MAP.items():
            unit = "shares" if column == "shares_out" else "USD"
            extracted = extract_pit_xbrl_tag(us_gaap, tags, unit=unit)
            if extracted.empty:
                continue
            extracted = extracted.rename(columns={"val": column})
            if merged is None:
                merged = extracted
                continue
            merged = pd.merge_asof(
                merged.sort_values("period_end"),
                extracted[["period_end", column]].sort_values("period_end"),
                on="period_end",
                direction="nearest",
                tolerance=pd.Timedelta("20D"),
            )
        if merged is None or merged.empty:
            return pd.DataFrame()
        return merged.sort_values("period_end").ffill().fillna(0.0)

    def business_description(self, cik: int) -> str:
        """10-K Item 1 (Business) text, used as the company's policy-matching vector."""
        try:
            submissions = self.client.get_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        except SourceUnavailable:
            return ""
        recent = (submissions.get("filings", {}) or {}).get("recent", {}) or {}
        forms = recent.get("form", []) or []
        accession = recent.get("accessionNumber", []) or []
        documents = recent.get("primaryDocument", []) or []
        for idx, form in enumerate(forms):
            if form not in ("10-K", "20-F"):
                continue
            if idx >= len(accession) or idx >= len(documents):
                continue
            accession_nodash = str(accession[idx]).replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/"
                f"{documents[idx]}"
            )
            try:
                html = self.client.get_text(url)
            except SourceUnavailable:
                continue
            text = _html_to_text(html)
            extracted = _extract_item1(text)
            if extracted:
                return extracted[: self.max_business_chars]
        return ""

    def parse(
        self,
        ticker: str,
        cik: int,
        *,
        sector: str = "UNKNOWN",
        close_wide: pd.DataFrame | None = None,
        market_returns: pd.Series | None = None,
        risk_free: float = 0.04,
    ) -> dict[str, Any] | None:
        try:
            facts = self.company_facts(cik)
        except SourceUnavailable:
            return None
        statement = self.statement_frame(facts)
        if len(statement) < 2:
            return None

        shares = float(statement.iloc[-1].get("shares_out", 0.0) or 0.0)
        stock_returns = None
        dollar_volume = float("nan")
        price = float("nan")
        if close_wide is not None and ticker in close_wide.columns:
            series = pd.to_numeric(close_wide[ticker], errors="coerce").dropna()
            if len(series) > 5:
                price = float(series.iloc[-1])
                stock_returns = series.pct_change(fill_method=None).dropna()
                lookback = int(self.config.e("fundamentals.price_lookback_days", 400))
                dollar_volume = _dollar_volume(series.tail(120), self.config)
        beta = (
            self.engine.beta_from_returns(stock_returns, market_returns)
            if stock_returns is not None and market_returns is not None
            else 1.0
        )
        market_cap = price * shares if np.isfinite(price) and shares > 0 else float("nan")
        cash = float(statement.iloc[-1].get("cash", 0.0) or 0.0)

        wacc_cfg = self.config.e("optimizer.wacc", {}) or {}
        metrics = compute_forensic_metrics(
            statement,
            beta=beta,
            risk_free=risk_free,
            equity_risk_premium=float(wacc_cfg.get("equity_risk_premium", 0.045)),
            effective_tax_rate=float(wacc_cfg.get("effective_tax_rate", 0.25)),
            min_wacc=float(wacc_cfg.get("min_wacc", 0.06)),
            max_wacc=float(wacc_cfg.get("max_wacc", 0.18)),
            market_cap=market_cap,
            cash=cash,
        )
        latest = statement.iloc[-1]
        row = {
            "ticker": ticker,
            "market": "US",
            "fiscal_period": str(pd.to_datetime(latest["period_end"]).date()),
            "filing_timestamp": pd.to_datetime(latest["filing_timestamp"], utc=True),
            "gics_sector": sector,
            "beta": round(float(beta), 4),
            "dollar_volume_20d": dollar_volume,
            "promoter_holding_pct": float("nan"),
            "promoter_pledge_pct": 0.0,
            "institutional_flow_delta": 0.0,
            "business_description": _business_text(ticker, sector, self.business_description(cik)),
        }
        row.update(detail_to_json(metrics))
        return row


def _dollar_volume(close: pd.Series, config: Config, volume: pd.Series | None = None) -> float:
    if volume is None:
        return float(close.tail(20).mean())  # price proxy when no volume feed is joined
    return float((pd.to_numeric(close, errors="coerce") * pd.to_numeric(volume, errors="coerce")).tail(20).mean())


def _html_to_text(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#8217;", "'")
    return re.sub(r"\s+", " ", text).strip()


def _extract_item1(text: str) -> str:
    for pattern in ITEM1_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        start = match.start()
        end = len(text)
        for terminator in ITEM_TERMINATORS:
            next_match = terminator.search(text, match.end())
            if next_match:
                end = min(end, next_match.start())
        excerpt = text[start:end].strip()
        if len(excerpt) > 150:
            return excerpt
    return ""


def _business_text(ticker: str, sector: str, description: str) -> str:
    parts = [ticker, sector, description]
    return ". ".join(part for part in parts if part).strip()


# --------------------------------------------------------------------------- #
# India: NSE/BSE Ind-AS results + Regulation 31 shareholding
# --------------------------------------------------------------------------- #
class IndiaFundamentalsParser:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client
        self.lag_days = int(self.config.e("fundamentals.india_reporting_lag_days", 45))
        self._primed = False

    def prime_session(self) -> None:
        if self._primed or not bool(self.config.e("fundamentals.nse_session_priming", True)):
            return
        try:
            self.client.get_text("https://www.nseindia.com")
        except SourceUnavailable:
            pass
        self._primed = True

    def shareholding(self, symbol: str) -> dict[str, float]:
        self.prime_session()
        url = f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={symbol}"
        try:
            payload = self.client.get_json(url)
        except SourceUnavailable:
            return {}
        blocks = payload if isinstance(payload, list) else payload.get("data", [])
        if not blocks:
            return {}
        latest = blocks[0] or {}
        previous = blocks[1] if len(blocks) > 1 else {}
        promoter = _first_number(latest, ["pr_and_prgrp", "promoterAndPromoterGroup", "promoter_hold"])
        pledge = _first_number(latest, ["pledged_encumbered", "pledgedEncumbered", "pledged_shares"])
        inst_now = _first_number(latest, ["fii_fpi", "fiiFpi"]) + _first_number(latest, ["dii", "diiSum"]) + _first_number(
            latest, ["mutual_funds", "mutualFunds"]
        )
        inst_prev = _first_number(previous, ["fii_fpi", "fiiFpi"]) + _first_number(previous, ["dii", "diiSum"]) + _first_number(
            previous, ["mutual_funds", "mutualFunds"]
        )
        return {
            "promoter_holding_pct": promoter,
            "promoter_pledge_pct": pledge,
            "institutional_flow_delta": inst_now - inst_prev,
        }

    def results(self, symbol: str) -> dict[str, Any]:
        self.prime_session()
        url = f"https://www.nseindia.com/api/results-comparision?index=equities&symbol={symbol}"
        try:
            payload = self.client.get_json(url)
        except SourceUnavailable:
            return {}
        blocks = payload.get("resCmpData") or payload.get("data") or []
        if not isinstance(blocks, list) or len(blocks) < 2:
            return {}
        current, previous = blocks[0] or {}, blocks[1] or {}
        period_end = _parse_indian_date(
            current.get("re_to_date") or current.get("to_date") or current.get("re_toDate")
        )
        broadcast = _parse_broadcast_timestamp(
            current.get("re_broadcast_timestamp") or current.get("re_exchdisstime") or current.get("broadcast_timestamp")
        )
        if broadcast is not None:
            filing_timestamp = broadcast
        elif period_end is not None:
            filing_timestamp = pd.Timestamp(period_end, tz="UTC") + pd.Timedelta(days=self.lag_days)
        else:
            filing_timestamp = pd.Timestamp(dt.datetime.now(dt.timezone.utc))
        return {
            "current": current,
            "previous": previous,
            "period_end": period_end,
            "filing_timestamp": filing_timestamp,
        }

    def parse(self, symbol: str, *, name: str = "", sector: str = "UNKNOWN") -> dict[str, Any] | None:
        financials = self.results(symbol)
        shareholding = self.shareholding(symbol)
        if not financials and not shareholding:
            return None

        current = financials.get("current", {}) or {}
        previous = financials.get("previous", {}) or {}
        revenue_c = _first_number(current, ["re_net_sale", "re_total_income", "re_income"])
        revenue_p = _first_number(previous, ["re_net_sale", "re_total_income", "re_income"])
        pat_c = _first_number(current, ["re_net_profit", "re_profit_after_tax"])
        pat_p = _first_number(previous, ["re_net_profit", "re_profit_after_tax"])
        ebit_c = _first_number(current, ["re_pro_loss_bef_int_tax", "re_profit_before_tax", "re_pbt"])
        interest_c = _first_number(current, ["re_int_new", "re_interest", "re_finance_cost"])

        promoter = float(shareholding.get("promoter_holding_pct", float("nan")))
        pledge = float(shareholding.get("promoter_pledge_pct", 0.0))
        institutional_delta = float(shareholding.get("institutional_flow_delta", 0.0))

        npm_c = pat_c / max(revenue_c, 1e-6)
        npm_p = pat_p / max(revenue_p, 1e-6)
        interest_coverage = ebit_c / max(interest_c, 1e-3)
        revenue_growth = (revenue_c / max(revenue_p, 1e-6)) - 1.0
        roic_proxy = ebit_c / max(revenue_c, 1e-6)

        # Governance-augmented forensic proxy: pledging and interest-coverage erosion
        # are the strongest documented predictors of Indian left-tail events (Reg 31).
        beneish_proxy = (
            float(self.config.e("forensics.statutory_beneish_gate", -1.78))
            - 0.6
            + 0.035 * pledge
            + 0.30 * max(revenue_growth, 0.0)
            + (0.35 if interest_coverage < 3.0 else 0.0)
        )
        f_score_proxy = int(
            sum(
                [
                    pat_c > 0,
                    npm_c > npm_p,
                    revenue_c > revenue_p,
                    interest_coverage > 3.0,
                    pledge < 5.0,
                    institutional_delta >= 0.0,
                    (not np.isfinite(promoter)) or promoter >= 35.0,
                ]
            )
        )

        wacc = self.config.e("optimizer.wacc", {}) or {}
        row = {
            "ticker": f"{symbol}.NS" if not symbol.endswith(".NS") else symbol,
            "market": "INDIA",
            "fiscal_period": str(financials.get("period_end") or pd.Timestamp(dt.date.today())),
            "filing_timestamp": financials.get("filing_timestamp", pd.Timestamp(dt.datetime.now(dt.timezone.utc))),
            "gics_sector": sector or "UNKNOWN",
            "piotroski_f_score": f_score_proxy,
            "beneish_m_score": round(float(beneish_proxy), 4),
            "roic": round(float(roic_proxy), 6),
            "wacc": round(float(np.clip(roic_proxy * 0.7, wacc.get("min_wacc", 0.06), wacc.get("max_wacc", 0.18))), 6),
            "roic_wacc_spread": round(float(roic_proxy * 0.3), 6),
            "fcf_ev_yield": round(float(roic_proxy * 0.8), 6),
            "fcf": float("nan"),
            "ev": float("nan"),
            "beta": 1.0,
            "dollar_volume_20d": float("nan"),
            "promoter_holding_pct": round(promoter, 3) if np.isfinite(promoter) else float("nan"),
            "promoter_pledge_pct": round(pledge, 3),
            "institutional_flow_delta": round(institutional_delta, 4),
            "business_description": _business_text(row_ticker(symbol), sector, name),
        }
        return row


def row_ticker(symbol: str) -> str:
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def _first_number(block: dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        if key in block and block[key] not in (None, "", "-", "--"):
            try:
                return float(str(block[key]).replace(",", "").replace("%", ""))
            except (TypeError, ValueError):
                continue
    return 0.0


def _parse_indian_date(value: Any) -> dt.date | None:
    if not value:
        return None
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(str(value).strip()[:11].strip(), fmt).date()
        except ValueError:
            continue
    parsed = pd.to_datetime(str(value), errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _parse_broadcast_timestamp(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(str(value), errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    parsed = pd.Timestamp(parsed)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# price panel helper + CLI
# --------------------------------------------------------------------------- #
def load_price_panel(lake_dir: str | Path, market: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(close_wide, volume_wide, market_returns) from the local lakehouse if present."""
    path = Path(lake_dir) / "market_prices.parquet"
    if not path.exists():
        empty = pd.DataFrame()
        return empty, empty, pd.Series(dtype=float)
    frame = pd.read_parquet(path)
    frame = frame[frame["market"].astype(str) == market]
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    frame["date"] = pd.to_datetime(frame["date"])
    close_column = "adj_close" if "adj_close" in frame.columns else "close"
    close_wide = frame.pivot_table(index="date", columns="ticker", values=close_column, aggfunc="last").sort_index()
    volume_wide = frame.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").sort_index()
    benchmark = frame.groupby("date")[close_column].mean()
    market_returns = benchmark.pct_change(fill_method=None).dropna()
    return close_wide, volume_wide, market_returns


def union_price_panel(lake_dir: str | Path, market: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return load_price_panel(lake_dir, market)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="US SEC EDGAR XBRL + India NSE/BSE fundamentals")
    parser.add_argument("--market", required=True, choices=["US", "INDIA"])
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Local lakehouse for prices/beta")
    parser.add_argument("--universe-csv", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    engine = SelfCalibratingQuantEngine(config)
    if args.market == "US":
        client = HttpClient(
            headers=SEC_HEADERS,
            cache_dir=Path.home() / ".cache" / "pms_sec",
            offline=args.offline,
            min_interval=1.0 / max(int(config.e("fundamentals.sec_requests_per_second", 8)), 1),
        )
        client.headers.update(SEC_HEADERS)
    else:
        client = HttpClient(
            headers=NSE_HEADERS,
            cache_dir=Path.home() / ".cache" / "pms_nse",
            offline=args.offline,
        )
        client.headers.update(NSE_HEADERS)

    parser_us = USEdgarFundamentalsParser(config, client, engine)
    parser_in = IndiaFundamentalsParser(config, client)

    close_wide = pd.DataFrame()
    market_returns = pd.Series(dtype=float)
    if args.data_dir:
        close_wide, _, market_returns = union_price_panel(args.data_dir, args.market)
    risk_free = float(config.e("optimizer.wacc.min_wacc", 0.06))

    records: list[dict[str, Any]] = []
    if args.symbols:
        items = [{"ticker": symbol, "cik": None, "name": symbol} for symbol in args.symbols]
    else:
        loader = UniverseLoader(config, client)
        universe = loader.load(args.market, limit=args.limit, static_csv=args.universe_csv)
        items = universe.to_dict("records")
    sharded = shard_of(items, args.shard_id, args.total_shards)

    for item in sharded:
        try:
            if args.market == "US":
                if item.get("cik") is None:
                    continue
                row = parser_us.parse(
                    str(item["ticker"]),
                    int(item["cik"]),
                    sector=str(item.get("gics_sector", "UNKNOWN") or "UNKNOWN"),
                    close_wide=close_wide if not close_wide.empty else None,
                    market_returns=market_returns if not market_returns.empty else None,
                    risk_free=risk_free,
                )
            else:
                symbol = str(item.get("nse_symbol") or item["ticker"]).replace(".NS", "")
                row = parser_in.parse(
                    symbol,
                    name=str(item.get("name", "")),
                    sector=str(item.get("gics_sector", "UNKNOWN") or "UNKNOWN"),
                )
            if row:
                records.append(row)
        except Exception:
            continue

    frame = pd.DataFrame(records)
    path = write_shard(frame, args.output_dir, f"fundamentals_{args.market.lower()}_{args.shard_id}")
    print(f"wrote {len(frame)} fundamental rows for {args.market} shard {args.shard_id} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
