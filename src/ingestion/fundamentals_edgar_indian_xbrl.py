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
US_FLOW_COLUMNS = frozenset(
    {"revenue", "cogs", "depreciation", "sga", "net_income", "operating_cf", "ebit", "capex"}
)

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
    """First-print (period_end, filed_at, value) series for a tag.

    CompanyFacts publishes a filing *date*, not an acceptance time. Until the
    submission's exact ``acceptanceDateTime`` is joined by accession number, the
    conservative fallback is midnight UTC on the following day, which cannot leak
    an after-close filing into the filing date's decision.
    """
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
            fallback_file_time = pd.Timestamp(filed, tz="UTC") + pd.Timedelta(days=1)
            rows.append(
                {
                    "period_end": pd.to_datetime(end),
                    "filing_timestamp": fallback_file_time,
                    "accession_number": str(item.get("accn") or "").replace("-", ""),
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


def extract_pit_xbrl_quarterly_flow(
    us_gaap: dict[str, Any],
    tag_candidates: Iterable[str],
    acceptance_by_accession: dict[str, pd.Timestamp],
    unit: str = "USD",
    *,
    public_forms: tuple[str, ...] = ("10-K", "10-Q", "20-F"),
) -> pd.DataFrame:
    """Extract discrete quarterly flow values from SEC duration facts.

    Prefer facts whose duration is a quarter. Where only a cumulative YTD or
    annual fact exists, subtract the prior cumulative fact for the same fiscal
    year and start date. If that subtraction cannot be supported, retain the
    period with a missing value instead of treating YTD as a standalone quarter.
    """
    rows: list[dict[str, Any]] = []
    for tag in tag_candidates:
        node = us_gaap.get(tag)
        if not node:
            continue
        for item in (node.get("units", {}) or {}).get(unit, []) or []:
            filed, start, end, value = (
                item.get("filed"), item.get("start"), item.get("end"), item.get("val")
            )
            if item.get("form") not in public_forms or not filed or not start or not end or value is None:
                continue
            try:
                start_date = pd.Timestamp(start).normalize()
                end_date = pd.Timestamp(end).normalize()
                fallback = pd.Timestamp(filed, tz="UTC") + pd.Timedelta(days=1)
                accession = str(item.get("accn") or "").replace("-", "")
                rows.append(
                    {
                        "period_end": end_date,
                        "start": start_date,
                        "duration_days": int((end_date - start_date).days + 1),
                        "filing_timestamp": acceptance_by_accession.get(accession, fallback),
                        "accession_number": accession,
                        "fy": item.get("fy"),
                        "val": float(value),
                    }
                )
            except (TypeError, ValueError, OverflowError):
                continue
        if rows:
            break
    if not rows:
        return pd.DataFrame(
            columns=["period_end", "filing_timestamp", "accession_number", "val"]
        )

    facts = pd.DataFrame(rows).sort_values(["period_end", "filing_timestamp"])
    quarterly: dict[pd.Timestamp, dict[str, Any]] = {}
    cumulative: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for period_end, group in facts.groupby("period_end", sort=True):
        direct = group[group["duration_days"].between(70, 110)].copy()
        if not direct.empty:
            direct["duration_distance"] = (direct["duration_days"] - 91).abs()
            selected = direct.sort_values(
                ["filing_timestamp", "duration_distance"], kind="stable"
            ).iloc[0]
            quarterly[pd.Timestamp(period_end)] = {
                "period_end": pd.Timestamp(period_end),
                "filing_timestamp": selected["filing_timestamp"],
                "accession_number": selected["accession_number"],
                "val": float(selected["val"]),
            }

        # Keep the longest same-period duration as the YTD anchor (Q1's 90-day
        # fact is also its YTD anchor; later quarters can subtract it if a
        # standalone quarter fact is absent).
        accumulated = group[group["duration_days"].between(70, 400)]
        if accumulated.empty:
            continue
        selected = accumulated.sort_values(
            ["duration_days", "filing_timestamp"], ascending=[False, True], kind="stable"
        ).iloc[0]
        fiscal_year_value = selected["fy"]
        fiscal_year = (
            str(int(fiscal_year_value))
            if pd.notna(fiscal_year_value)
            else str(pd.Timestamp(selected["start"]).year)
        )
        cumulative[(fiscal_year, pd.Timestamp(period_end))] = selected.to_dict()

    for (fiscal_year, period_end), selected in cumulative.items():
        prior_rows = [
            (prior_end, row)
            for (prior_fy, prior_end), row in cumulative.items()
            if prior_fy == fiscal_year
            and prior_end < period_end
            and pd.Timestamp(row["start"]) == pd.Timestamp(selected["start"])
        ]
        prior = max(prior_rows, key=lambda pair: pair[0])[1] if prior_rows else None
        value = float("nan")
        knowledge_time = selected["filing_timestamp"]
        if prior is not None:
            value = float(selected["val"]) - float(prior["val"])
            knowledge_time = max(knowledge_time, prior["filing_timestamp"])
        if period_end in quarterly:
            continue
        quarterly[period_end] = {
            "period_end": period_end,
            "filing_timestamp": knowledge_time,
            "accession_number": selected["accession_number"],
            "val": value,
        }

    if not quarterly:
        return pd.DataFrame(
            columns=["period_end", "filing_timestamp", "accession_number", "val"]
        )
    return pd.DataFrame(quarterly.values()).sort_values("period_end").reset_index(drop=True)


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
            if column in US_FLOW_COLUMNS:
                extracted = extract_pit_xbrl_quarterly_flow(us_gaap, tags, {}, unit=unit)
            else:
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

    def statement_history_frame(self, facts: dict[str, Any], cik: int) -> pd.DataFrame:
        """Join first-print facts while retaining the latest public timestamp per row."""
        us_gaap = (facts.get("facts", {}) or {}).get("us-gaap", {}) or {}
        try:
            submissions = self.client.get_json(
                f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
            )
        except SourceUnavailable:
            submissions = {}
        recent = ((submissions.get("filings", {}) or {}).get("recent", {}) or {})
        accessions = recent.get("accessionNumber", []) or []
        acceptance_times = recent.get("acceptanceDateTime", []) or []
        acceptance_by_accession: dict[str, pd.Timestamp] = {}
        for accession, accepted in zip(accessions, acceptance_times):
            if not accession or not accepted:
                continue
            stamp = pd.to_datetime(accepted, utc=True, errors="coerce")
            if pd.notna(stamp):
                acceptance_by_accession[str(accession).replace("-", "")] = stamp

        merged: pd.DataFrame | None = None
        filed_columns: list[str] = []
        for column, tags in US_TAG_MAP.items():
            unit = "shares" if column == "shares_out" else "USD"
            if column in US_FLOW_COLUMNS:
                extracted = extract_pit_xbrl_quarterly_flow(
                    us_gaap, tags, acceptance_by_accession, unit=unit
                )
            else:
                extracted = extract_pit_xbrl_tag(us_gaap, tags, unit=unit)
            if extracted.empty:
                continue
            filed_column = f"{column}__filed_at"
            filed_columns.append(filed_column)
            extracted["filing_timestamp"] = [
                acceptance_by_accession.get(accession, fallback)
                for accession, fallback in zip(
                    extracted["accession_number"], extracted["filing_timestamp"]
                )
            ]
            extracted = extracted.rename(
                columns={"val": column, "filing_timestamp": filed_column}
            )
            if merged is None:
                merged = extracted[["period_end", column, filed_column]]
                continue
            right = extracted[["period_end", column, filed_column]].sort_values("period_end")
            merged = pd.merge_asof(
                merged.sort_values("period_end"),
                right,
                on="period_end",
                direction="nearest",
                tolerance=pd.Timedelta("20D"),
            )
        if merged is None or merged.empty:
            return pd.DataFrame()

        # A row only becomes usable once all values it uses, including carried
        # values from older periods, were public. Carry the per-column knowledge
        # timestamps alongside the values before taking the row-level maximum.
        merged = merged.sort_values("period_end").reset_index(drop=True)
        value_columns = [column for column in US_TAG_MAP if column in merged.columns]
        for column in value_columns:
            filed_column = f"{column}__filed_at"
            merged[[column, filed_column]] = merged[[column, filed_column]].ffill()
        merged["filing_timestamp"] = merged[filed_columns].max(axis=1)
        return merged.dropna(subset=["filing_timestamp"]).drop_duplicates(
            subset=["period_end"], keep="last"
        ).reset_index(drop=True)

    def business_description(
        self, cik: int, *, as_of: pd.Timestamp | None = None
    ) -> str:
        """10-K Item 1 (Business) text, used as the company's policy-matching vector."""
        try:
            submissions = self.client.get_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        except SourceUnavailable:
            return ""
        recent = (submissions.get("filings", {}) or {}).get("recent", {}) or {}
        forms = recent.get("form", []) or []
        accession = recent.get("accessionNumber", []) or []
        documents = recent.get("primaryDocument", []) or []
        filing_dates = recent.get("filingDate", []) or []
        acceptance_times = recent.get("acceptanceDateTime", []) or []
        cutoff = None
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        for idx, form in enumerate(forms):
            if form not in ("10-K", "20-F"):
                continue
            if idx >= len(accession) or idx >= len(documents):
                continue
            if cutoff is not None:
                if idx >= len(filing_dates):
                    continue
                accepted = acceptance_times[idx] if idx < len(acceptance_times) else None
                knowledge_time = pd.to_datetime(accepted, errors="coerce", utc=True)
                if pd.isna(knowledge_time):
                    filing_day = pd.to_datetime(filing_dates[idx], errors="coerce", utc=True)
                    if pd.isna(filing_day):
                        continue
                    # Company submissions do not always include acceptance times.
                    # The next-day fallback is intentionally conservative.
                    knowledge_time = filing_day.normalize() + pd.Timedelta(days=1)
                if knowledge_time > cutoff:
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

    def parse_history(
        self,
        ticker: str,
        cik: int,
        *,
        sector: str = "UNKNOWN",
        close_wide: pd.DataFrame | None = None,
        raw_close_wide: pd.DataFrame | None = None,
        volume_wide: pd.DataFrame | None = None,
        market_returns: pd.Series | None = None,
        risk_free: float = 0.04,
        as_of: pd.Timestamp | None = None,
        history_years: int = 5,
    ) -> list[dict[str, Any]]:
        """Emit distinct SEC-known statement snapshots for leakage-free replay."""
        try:
            facts = self.company_facts(cik)
        except SourceUnavailable:
            return []
        statements = self.statement_history_frame(facts, cik)
        if len(statements) < 2:
            return []

        cutoff = pd.Timestamp(as_of or dt.datetime.now(dt.timezone.utc))
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        earliest_period = cutoff.tz_localize(None) - pd.DateOffset(years=max(int(history_years), 1))
        statements = statements.copy()
        statements["filing_timestamp"] = pd.to_datetime(
            statements["filing_timestamp"], utc=True, errors="coerce"
        )
        statements["period_end"] = pd.to_datetime(
            statements["period_end"], utc=True, errors="coerce"
        ).dt.tz_convert(None).dt.normalize()
        statements = statements[
            (statements["period_end"] >= earliest_period)
            & (statements["filing_timestamp"] <= cutoff)
        ]
        if len(statements) < 2:
            return []

        snapshots: list[dict[str, Any]] = []
        stamps = sorted(statements["filing_timestamp"].dropna().unique())
        wacc_cfg = self.config.e("optimizer.wacc", {}) or {}
        history_close = _date_indexed_panel(close_wide)
        raw_history_close = _date_indexed_panel(raw_close_wide)
        history_volume = _date_indexed_panel(volume_wide)
        market_history = _date_indexed_series(market_returns)
        for raw_stamp in stamps:
            stamp = pd.Timestamp(raw_stamp).tz_convert("UTC")
            known = statements[
                statements["filing_timestamp"] <= stamp
            ].sort_values("period_end")
            if len(known) < 2:
                continue
            current = known.iloc[-1]
            period_end = pd.Timestamp(current["period_end"])
            # Price-based quantities are marked using the latest fully completed
            # session before disclosure. Excluding the disclosure-date bar avoids
            # assuming a decision could use a close that was not yet available.
            price_cutoff = stamp.tz_localize(None).normalize()
            series = pd.Series(dtype=float)
            market_series = pd.Series(dtype=float)
            if not history_close.empty and ticker in history_close.columns:
                series = pd.to_numeric(
                    history_close.loc[history_close.index < price_cutoff, ticker], errors="coerce"
                ).dropna()
            if not market_history.empty:
                market_series = market_history[market_history.index < price_cutoff].dropna()
            stock_returns = series.pct_change(fill_method=None).dropna()
            beta = (
                self.engine.beta_from_returns(stock_returns, market_series)
                if not stock_returns.empty and not market_series.empty
                else 1.0
            )
            raw_series = pd.Series(dtype=float)
            if not raw_history_close.empty and ticker in raw_history_close.columns:
                raw_series = pd.to_numeric(
                    raw_history_close.loc[raw_history_close.index < price_cutoff, ticker],
                    errors="coerce",
                ).dropna()
            price = float(raw_series.iloc[-1]) if not raw_series.empty else float("nan")
            dollar_volume = float("nan")
            if (
                not raw_series.empty
                and not history_volume.empty
                and ticker in history_volume.columns
            ):
                volume_series = pd.to_numeric(
                    history_volume.loc[history_volume.index < price_cutoff, ticker],
                    errors="coerce",
                )
                aligned_volume = volume_series.reindex(raw_series.index).dropna()
                aligned_price = raw_series.reindex(aligned_volume.index)
                if not aligned_volume.empty:
                    dollar_volume = float((aligned_price * aligned_volume).tail(20).mean())
            shares = float(current.get("shares_out", 0.0) or 0.0)
            market_cap = price * shares if np.isfinite(price) and shares > 0 else float("nan")
            cash = float(current.get("cash", 0.0) or 0.0)
            statement_values = known.drop(
                columns=[
                    column
                    for column in known.columns
                    if column == "filing_timestamp" or column.endswith("__filed_at")
                ],
                errors="ignore",
            )
            metrics = compute_forensic_metrics(
                statement_values,
                beta=beta,
                risk_free=risk_free,
                equity_risk_premium=float(wacc_cfg.get("equity_risk_premium", 0.045)),
                effective_tax_rate=float(wacc_cfg.get("effective_tax_rate", 0.25)),
                min_wacc=float(wacc_cfg.get("min_wacc", 0.06)),
                max_wacc=float(wacc_cfg.get("max_wacc", 0.18)),
                market_cap=market_cap,
                cash=cash,
            )
            description = self.business_description(cik, as_of=stamp)
            row = {
                "ticker": ticker,
                "market": "US",
                "fiscal_period": str(period_end.date()),
                "filing_timestamp": stamp,
                "gics_sector": sector,
                "beta": round(float(beta), 4),
                "dollar_volume_20d": dollar_volume,
                "promoter_holding_pct": float("nan"),
                "promoter_pledge_pct": 0.0,
                "institutional_flow_delta": 0.0,
                "business_description": _business_text(ticker, sector, description),
            }
            row.update(detail_to_json(metrics))
            snapshots.append(row)
        return snapshots

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
def load_price_panel(
    lake_dir: str | Path,
    market: str,
    *,
    include_raw_close: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series] | tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame
]:
    """(adjusted close, volume, market returns), optionally with raw closes."""
    path = Path(lake_dir) / "market_prices.parquet"
    if not path.exists():
        empty = pd.DataFrame()
        result = (empty, empty, pd.Series(dtype=float))
        return (*result, empty) if include_raw_close else result
    frame = pd.read_parquet(path)
    frame = frame[frame["market"].astype(str) == market]
    if frame.empty:
        empty = pd.DataFrame()
        result = (empty, empty, pd.Series(dtype=float))
        return (*result, empty) if include_raw_close else result
    frame["date"] = pd.to_datetime(frame["date"])
    close_column = "adj_close" if "adj_close" in frame.columns else "close"
    close_wide = frame.pivot_table(index="date", columns="ticker", values=close_column, aggfunc="last").sort_index()
    volume_wide = frame.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").sort_index()
    benchmark = frame.groupby("date")[close_column].mean()
    market_returns = benchmark.pct_change(fill_method=None).dropna()
    result = (close_wide, volume_wide, market_returns)
    if not include_raw_close:
        return result
    raw_close_wide = frame.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    ).sort_index()
    return (*result, raw_close_wide)


def union_price_panel(lake_dir: str | Path, market: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return load_price_panel(lake_dir, market)


def _date_indexed_panel(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce", utc=True).tz_convert(None))
    out.index = index.normalize()
    return out.sort_index()


def _date_indexed_series(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    out = pd.to_numeric(series, errors="coerce").copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce", utc=True).tz_convert(None))
    out.index = index.normalize()
    return out.sort_index()


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
