"""Geopolitical shock ingestion — GDELT 2.0 Doc API (free, no key) + sanctions lists.

GDELT's Doc API returns syndicated headlines; the raw feed is therefore dominated
by duplicate cables. Denoising happens in two stages:

1. ingestion emits one row per article with an explicit publication timestamp and
   the base authority weight from ``config/engine.yaml``;
2. :mod:`src.filtering.embedding_projector` performs MinHash-style deduplication
   (near-duplicate headline clustering), turns cluster size into the *mentions*
   term of the Goldstein severity weight, and fills the semantic polarity.

The Goldstein magnitude itself is taken from the optional OFAC/EU sanctions feed
when present, otherwise the semantic severity proxy from the projector is used.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from .base import HttpClient, SourceUnavailable, write_shard


class GeopoliticsIngestor:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client
        self.spec = dict(config.get("universe.event_sources.geopolitics", {}) or {})

    def fetch_query(self, name: str, query: str, max_records: int = 40, timespan: str = "24h") -> pd.DataFrame:
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(max_records),
            "format": "json",
            "timespan": timespan,
            "sort": "hybridrel",
        }
        try:
            payload = self.client.get_json(self.spec["endpoint"], params=params)
        except SourceUnavailable:
            return pd.DataFrame()
        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        domain = name if name.startswith("GEOPOL") else f"GEOPOL_{name}"
        base_authority = float((self.config.e("events.authority_weights", {}) or {}).get("gdelt", 0.75))
        rows: list[dict[str, Any]] = []
        for idx, article in enumerate(articles):
            headline = (article.get("title") or "").strip()
            if not headline:
                continue
            published = _parse_seendate(article.get("seendate")) or dt.datetime.now(dt.timezone.utc)
            rows.append(
                {
                    "event_id": f"GDELT_{name}_{published:%Y%m%d%H%M%S}_{idx:03d}",
                    "published_timestamp": published,
                    "domain": domain,
                    "source_agency": article.get("domain", "gdelt"),
                    "authority_weight": base_authority,
                    "authority_class": "gdelt_article",
                    "polarity": None,
                    "half_life_days": self.config.half_life_default("GEOPOLITICAL"),
                    "headline": headline[:500],
                    "raw_text": f"{headline}. {article.get('language', '')}",
                    "embedding": None,
                }
            )
        return pd.DataFrame(rows)

    def fetch_sanctions(self) -> pd.DataFrame:
        """OFAC SDN list as high-authority, unanimously negative supply-chain events."""
        url = (self.config.get("universe.event_sources.sanctions", {}) or {}).get("ofac_sdn")
        if not url:
            return pd.DataFrame()
        try:
            text = self.client.get_text(url)
        except SourceUnavailable:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        now = dt.datetime.now(dt.timezone.utc)
        for line in text.splitlines()[:2000]:
            parts = [part.strip().strip('"') for part in line.split(",")]
            if len(parts) < 2:
                continue
            name = parts[1]
            if not name or name.lower() in {"name", "sdn name"}:
                continue
            rows.append(
                {
                    "event_id": f"OFAC_{abs(hash(name)) % 10**12:012d}",
                    "published_timestamp": now,
                    "domain": "GEOPOLITICAL",
                    "source_agency": "US_OFAC_SDN",
                    "authority_weight": 1.0,
                    "authority_class": "sanction",
                    "polarity": -1.0,
                    "half_life_days": self.config.half_life_default("GOV_POLICY"),
                    "headline": f"OFAC SDN designation: {name}",
                    "raw_text": f"OFAC specially designated national sanction designation for {name}.",
                    "embedding": None,
                }
            )
        return pd.DataFrame(rows).drop_duplicates(subset=["event_id"]) if rows else pd.DataFrame()

    def fetch(self, max_records: int = 40, timespan: str = "24h", include_sanctions: bool = False) -> pd.DataFrame:
        frames = []
        for name, query in (self.spec.get("queries", {}) or {}).items():
            frame = self.fetch_query(name, str(query), max_records=max_records, timespan=timespan)
            if not frame.empty:
                frames.append(frame)
        if include_sanctions:
            sanctions = self.fetch_sanctions()
            if not sanctions.empty:
                frames.append(sanctions)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _parse_seendate(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return dt.datetime.strptime(value, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def make_client(
    config: Config,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> HttpClient:
    """Pacing-aware client for the GDELT Doc API (one call per documented interval).

    GDELT answers bursts with an HTTP 429 whose body is a *plain-text* notice;
    unless every caller of this endpoint shares the same pacing + retry floor the
    whole geopolitics pillar degrades on the first burst. Used by this module's CLI
    and by ``src.filtering.embedding_projector --ingest-domain geopol_gdelt``.
    """
    geopolitics = dict(config.get("universe.event_sources.geopolitics", {}) or {})
    interval = float(geopolitics.get("min_interval_seconds", 5.0) or 0.0)
    return HttpClient(
        cache_dir=cache_dir,
        offline=offline,
        min_interval=interval,
        retry_after_floor=interval,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GDELT geopolitical ingestion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=40)
    parser.add_argument("--timespan", default="24h")
    parser.add_argument("--include-sanctions", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    client = make_client(config, cache_dir=Path.home() / ".cache" / "pms_http", offline=args.offline)
    frame = GeopoliticsIngestor(config, client).fetch(
        max_records=args.max_records, timespan=args.timespan, include_sanctions=args.include_sanctions
    )
    path = write_shard(frame, args.output_dir, "geopol_gdelt")
    print(f"wrote {len(frame)} geopolitical events -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
