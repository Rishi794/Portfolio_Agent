"""India government / regulator policy ingestion: PIB (Cabinet & ministries),
RBI press releases and SEBI circulars (all free RSS/XML, no key).

Captured instruments include PLI scheme disbursements, Defence Acquisition Council
approvals, Union Budget capex, import/export duty changes, GST Council realignments
(PIB/Cabinet); MPC repo/SDF/MSF decisions, CRR changes, LAF liquidity and NBFC risk
weight circulars (RBI); F&O margin rules and promoter-disclosure enforcement (SEBI).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from .base import HttpClient, SourceUnavailable, write_shard

_WHITESPACE = re.compile(r"\s+")
_TAGS = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", _TAGS.sub(" ", text or "")).strip()


class IndiaPolicyIngestor:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client
        self.spec = dict(config.get("universe.event_sources.india_policy", {}) or {})

    def fetch(self, limit_per_feed: int = 40) -> pd.DataFrame:
        domain = str(self.spec.get("domain", "GOV_POLICY_INDIA"))
        rows: list[dict[str, Any]] = []
        for source_tag, url in self.spec.get("feeds", []):
            try:
                text = self.client.get_text(url)
            except SourceUnavailable:
                continue
            try:
                root = ET.fromstring(text)
            except ET.ParseError:
                continue
            for idx, item in enumerate(root.findall(".//item")[:limit_per_feed]):
                title = _clean(item.findtext("title") or "")
                description = _clean(item.findtext("description") or "")
                body = f"{title}. {description}".strip()
                if len(body) < 20:
                    continue
                published = _parse_pubdate(item.findtext("pubDate")) or dt.datetime.now(dt.timezone.utc)
                rows.append(
                    {
                        "event_id": f"{source_tag}_{published:%Y%m%d%H%M}_{abs(hash(body)) % 10**8:08d}",
                        "published_timestamp": published,
                        "domain": domain,
                        "source_agency": source_tag,
                        "authority_weight": _authority_for(source_tag, self.config),
                        "authority_class": _authority_class(source_tag),
                        "polarity": None,
                        "half_life_days": self.config.half_life_default(domain),
                        "headline": title[:500],
                        "raw_text": body[:4000],
                        "embedding": None,
                    }
                )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).drop_duplicates(subset=["event_id"])
        return frame.reset_index(drop=True)


def _authority_for(source_tag: str, config: Config) -> float:
    weights = config.e("events.authority_weights", {}) or {}
    if "RBI" in source_tag:
        return float(weights.get("final_rule", 1.0))          # MPC decisions are enacted immediately
    if "SEBI" in source_tag:
        return float(weights.get("final_rule", 1.0))          # circulars are binding once issued
    if "CABINET" in source_tag:
        return float(weights.get("final_rule", 1.0))
    return float(weights.get("proposed_rule", 0.40))


def _authority_class(source_tag: str) -> str:
    if any(key in source_tag for key in ("RBI", "SEBI", "CABINET")):
        return "enacted"
    return "announcement"


def _parse_pubdate(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = dt.datetime.strptime(value, fmt)
            return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="India policy ingestion (PIB / RBI / SEBI)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    client = HttpClient(
        cache_dir=Path.home() / ".cache" / "pms_http",
        offline=args.offline,
        headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"},
    )
    frame = IndiaPolicyIngestor(config, client).fetch()
    path = write_shard(frame, args.output_dir, "policy_in_pib_rbi_sebi")
    print(f"wrote {len(frame)} India policy events -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
