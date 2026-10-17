"""Technology-evolution ingestion: OpenAlex (free, no key) + USPTO PatentsView
(free key optional) + R&D intensity pulled from the PIT fundamentals store.

Measured quantities (all PIT):
* ``grants_publications_180d``          — citation-weighted output in the trailing 180 days;
* ``citation_velocity_log_ratio``       — log(180d citation-weighted output / 3-year semi-annual baseline);
* ``rd_to_revenue``, ``gross_margin_3y_delta``, ``capex_to_depreciation`` — the
  "productive R&D" and "CapEx modernisation" legs of the Tech Evolution score.

The point of mixing patents with R&D productivity is to filter press-release hype:
a company claiming AI leadership without patents, without R&D intensity, and without
gross-margin expansion earns no technology alpha.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .base import HttpClient, SourceUnavailable, write_shard
from .universe import UniverseLoader, shard_of

OPENALEX_HEADERS = {
    "User-Agent": "AutonomousPMS/4.0 (mailto:quant-research@autonomous-pms.org)",
    "Accept": "application/json",
}


class PatentAndTechEvolutionIngestor:
    def __init__(self, config: Config, client: HttpClient, patentsview_api_key: str = "") -> None:
        self.config = config
        self.client = client
        self.pv_key = patentsview_api_key
        sources = config.get("universe.event_sources.technology", {}) or {}
        self.openalex_institutions = sources.get("openalex_institutions", "https://api.openalex.org/institutions")
        self.openalex_works = sources.get("openalex_works", "https://api.openalex.org/works")
        self.patentsview_search = sources.get("patentsview_search", "https://search.patentsview.org/api/v1/patent/")

    # --------------------------------------------------------------- OpenAlex ---
    def fetch_openalex(self, company_name: str, *, recent_days: int = 180, baseline_days: int = 1080) -> dict[str, float]:
        today = dt.date.today()
        recent_cutoff = (today - dt.timedelta(days=recent_days)).isoformat()
        baseline_cutoff = (today - dt.timedelta(days=baseline_days)).isoformat()
        try:
            institutions = self.client.get_json(
                self.openalex_institutions, params={"search": company_name, "per-page": 1}
            )
        except SourceUnavailable:
            return {}
        results = (institutions or {}).get("results") or []
        if not results:
            return {}
        institution_id = str(results[0].get("id", "")).rsplit("/", 1)[-1]
        if not institution_id:
            return {}
        try:
            works = self.client.get_json(
                self.openalex_works,
                params={
                    "filter": f"institutions.id:{institution_id},from_publication_date:{baseline_cutoff}",
                    "select": "publication_date,cited_by_count",
                    "per-page": 200,
                },
            )
        except SourceUnavailable:
            return {}
        recent_weighted = 0.0
        baseline_weighted = 0.0
        recent_count = 0
        for work in (works or {}).get("results", []) or []:
            published = str(work.get("publication_date") or "")
            cites = float(work.get("cited_by_count", 0) or 0)
            weight = 1.0 + float(np.log1p(cites))
            baseline_weighted += weight
            if published and published >= recent_cutoff:
                recent_weighted += weight
                recent_count += 1
        semi_annual_baseline = baseline_weighted * (recent_days / max(baseline_days, 1))
        ratio = float(np.log((1.0 + recent_weighted) / (1.0 + semi_annual_baseline)))
        return {
            "openalex_recent_weighted": recent_weighted,
            "openalex_baseline_weighted": semi_annual_baseline,
            "grants_publications_180d": float(recent_count),
            "openalex_log_ratio": ratio,
        }

    # ------------------------------------------------------------ PatentsView ---
    def fetch_patentsview(self, company_name: str, *, baseline_days: int = 1080, recent_days: int = 180) -> dict[str, float]:
        if not self.pv_key:
            return {}
        today = dt.date.today()
        baseline_cutoff = (today - dt.timedelta(days=baseline_days)).isoformat()
        recent_cutoff = (today - dt.timedelta(days=recent_days)).isoformat()
        payload = {
            "q": {
                "_and": [
                    {"_text_phrase": {"assignees.assignee_organization": company_name}},
                    {"_gte": {"patent_date": baseline_cutoff}},
                ]
            },
            "f": ["patent_id", "patent_date", "patent_num_times_cited_by_us_patents"],
            "o": {"per_page": 500},
        }
        try:
            response = self.client.post_json(self.patentsview_search, payload, extra_headers={"X-Api-Key": self.pv_key})
        except SourceUnavailable:
            return {}
        recent_weighted = 0.0
        baseline_weighted = 0.0
        recent_count = 0
        for patent in (response or {}).get("patents", []) or []:
            date = str(patent.get("patent_date") or "")
            cites = float(patent.get("patent_num_times_cited_by_us_patents", 0) or 0)
            weight = 1.5 * (1.0 + float(np.log1p(cites)))
            baseline_weighted += weight
            if date and date >= recent_cutoff:
                recent_weighted += weight
                recent_count += 1
        semi_annual_baseline = baseline_weighted * (recent_days / max(baseline_days, 1))
        ratio = float(np.log((1.0 + recent_weighted) / (1.0 + semi_annual_baseline)))
        return {
            "patents_recent_weighted": recent_weighted,
            "patents_baseline_weighted": semi_annual_baseline,
            "patents_180d": float(recent_count),
            "patentsview_log_ratio": ratio,
        }

    # ------------------------------------------------------------------ merge ---
    def fetch_company_tech_velocity(self, ticker: str, company_name: str, market: str) -> dict[str, Any]:
        openalex = self.fetch_openalex(company_name) if company_name else {}
        patents = self.fetch_patentsview(company_name) if company_name else {}
        weights: list[tuple[float, float]] = []
        if "openalex_log_ratio" in openalex:
            weights.append((openalex["openalex_log_ratio"], max(openalex.get("openalex_baseline_weighted", 1.0), 1.0)))
        if "patentsview_log_ratio" in patents:
            weights.append((patents["patentsview_log_ratio"], max(patents.get("patents_baseline_weighted", 1.0), 1.0)))
        if weights:
            total_weight = sum(weight for _, weight in weights)
            blended = sum(ratio * weight for ratio, weight in weights) / max(total_weight, 1e-9)
        else:
            blended = 0.0
        return {
            "market": market,
            "ticker": ticker,
            "knowledge_timestamp": dt.datetime.now(dt.timezone.utc),
            "grants_publications_180d": int(
                openalex.get("grants_publications_180d", 0) + patents.get("patents_180d", 0)
            ),
            "citation_velocity_log_ratio": round(float(blended), 6),
            "rd_to_revenue": float("nan"),
            "gross_margin_3y_delta": float("nan"),
            "capex_to_depreciation": float("nan"),
        }

    # --------------------------------------------------------------- universe ---
    def fetch_universe(self, *, limit: int | None = None, symbols: list[str] | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        loader = UniverseLoader(self.config, self.client)
        for market in self.config.markets():
            if symbols:
                items = [{"ticker": symbol, "name": symbol} for symbol in symbols]
            else:
                try:
                    universe = loader.load(market, limit=limit)
                except Exception:
                    continue
                items = universe.to_dict("records")
            for item in items:
                ticker = str(item.get("ticker"))
                name = str(item.get("name") or ticker.replace(".NS", ""))
                try:
                    rows.append(self.fetch_company_tech_velocity(ticker, name, market))
                except Exception:
                    continue
        return pd.DataFrame(rows)

    def enrich_with_fundamentals(self, tech: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Attach productive-R&D and CapEx-modernisation legs from the PIT fundamental store."""
        if tech.empty or fundamentals is None or fundamentals.empty:
            return tech
        latest = (
            fundamentals.sort_values("filing_timestamp")
            .drop_duplicates(subset=["ticker"], keep="last")
            .set_index("ticker")
        )
        out = tech.copy()
        for column in ("rd_to_revenue", "gross_margin_3y_delta", "capex_to_depreciation", "revenue", "rd_expense", "capex", "depreciation", "gross_margin"):
            if column in latest.columns:
                out[column] = out["ticker"].map(latest[column])
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAlex + PatentsView technology velocity")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    import os

    config = Config.load()
    client = HttpClient(
        headers=OPENALEX_HEADERS,
        cache_dir=Path.home() / ".cache" / "pms_openalex",
        offline=args.offline,
    )
    client.headers.update(OPENALEX_HEADERS)
    ingestor = PatentAndTechEvolutionIngestor(
        config, client, patentsview_api_key=os.getenv("PATENTSVIEW_API_KEY", "")
    )
    frame = ingestor.fetch_universe(limit=args.limit)
    if not frame.empty:
        frame = frame.iloc[shard_of(list(range(len(frame))), args.shard_id, args.total_shards)]
    if args.data_dir and not frame.empty:
        fundamentals_path = Path(args.data_dir) / "fundamentals_pit.parquet"
        if fundamentals_path.exists():
            frame = ingestor.enrich_with_fundamentals(frame, pd.read_parquet(fundamentals_path))
    path = write_shard(frame, args.output_dir, "tech_patents_velocity")
    print(f"wrote {len(frame)} technology rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
