"""US government policy ingestion — Federal Register API (free, no key required).

Only *actionable* instruments are collected; political rhetoric never enters the
event store. Authority weights follow the documented ladder:

    Final Rule / Enacted Statute = 1.00
    Executive Order              = 0.85
    Proposed Rule / Notice       = 0.40
    Press rhetoric               = 0.05  (excluded here by construction)

Weights live in ``config/engine.yaml: events.authority_weights``.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from .base import HttpClient, SourceUnavailable, write_shard

PUBLICATION_HOUR_UTC = 14  # FR documents are public 08:45 ET; 14:00 UTC is a safe gate


def _authority_weight(doc_type: str, config: Config) -> tuple[float, str]:
    weights = config.e("events.authority_weights", {}) or {}
    normalized = (doc_type or "").strip().lower()
    if "final" in normalized or normalized in {"rule"}:
        return float(weights.get("final_rule", 1.0)), "final_rule"
    if "presidential" in normalized or "executive" in normalized:
        return float(weights.get("executive_order", 0.85)), "executive_order"
    if "proposed" in normalized or "prorule" in normalized or "notice" in normalized:
        return float(weights.get("proposed_rule", 0.40)), "proposed_rule"
    return float(weights.get("press_rhetoric", 0.05)), "press_rhetoric"


class FederalRegisterIngestor:
    def __init__(self, config: Config, client: HttpClient) -> None:
        self.config = config
        self.client = client
        self.spec = dict(config.get("universe.event_sources.us_policy", {}) or {})

    def fetch(self, since_date: dt.date | str, until_date: dt.date | str | None = None, per_page: int = 50) -> pd.DataFrame:
        since = since_date.isoformat() if isinstance(since_date, dt.date) else str(since_date)
        params: list[tuple[str, str]] = [
            ("conditions[publication_date][gte]", since),
            ("per_page", str(per_page)),
            ("order", "newest"),
            ("fields[]", "document_number"),
            ("fields[]", "title"),
            ("fields[]", "abstract"),
            ("fields[]", "type"),
            ("fields[]", "publication_date"),
            ("fields[]", "agencies"),
            ("fields[]", "html_url"),
            ("fields[]", "effective_on"),
        ]
        if until_date:
            params.append(
                ("conditions[publication_date][lte]", until_date.isoformat() if isinstance(until_date, dt.date) else str(until_date))
            )
        for doc_type in self.spec.get("document_types", ["RULE", "PRESDOCU"]):
            params.append(("conditions[type][]", str(doc_type)))
        for agency in self.spec.get("agencies", []):
            params.append(("conditions[agencies][]", str(agency)))

        try:
            payload = self.client.get_json(self.spec["endpoint"], params=_pairs_to_dict(params))
        except SourceUnavailable:
            return pd.DataFrame()

        domain = str(self.spec.get("domain", "GOV_POLICY"))
        rows: list[dict[str, Any]] = []
        for doc in payload.get("results", []):
            abstract = (doc.get("abstract") or "").strip()
            title = (doc.get("title") or "").strip()
            text = f"{title}. {abstract}".strip()
            if len(text) < 25:
                continue
            authority, label = _authority_weight(doc.get("type", ""), self.config)
            published = dt.datetime.combine(
                dt.date.fromisoformat(doc["publication_date"]),
                dt.time(hour=PUBLICATION_HOUR_UTC),
                tzinfo=dt.timezone.utc,
            )
            rows.append(
                {
                    "event_id": f"FEDREG_{doc['document_number']}",
                    "published_timestamp": published,
                    "domain": domain,
                    "source_agency": ",".join(
                        str(agency.get("slug", "")) for agency in (doc.get("agencies") or [])[:2]
                    ),
                    "authority_weight": authority,
                    "authority_class": label,
                    "polarity": None,
                    "half_life_days": self.config.half_life_default(domain),
                    "headline": title[:500],
                    "raw_text": text[:4000],
                    "embedding": None,
                }
            )
        return pd.DataFrame(rows)


def _pairs_to_dict(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Collapse repeated keys into the list form httpx sends as repeated params."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            existing = out[key]
            out[key] = existing + [value] if isinstance(existing, list) else [existing, value]
        else:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="US Federal Register policy ingestion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--since", default=None, help="YYYY-MM-DD (default: 7 days back)")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    client = HttpClient(cache_dir=Path.home() / ".cache" / "pms_http", offline=args.offline)
    since = args.since or (dt.date.today() - dt.timedelta(days=7)).isoformat()
    frame = FederalRegisterIngestor(config, client).fetch(since)
    path = write_shard(frame, args.output_dir, "policy_us_fedreg")
    print(f"wrote {len(frame)} policy events -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
