"""Deterministic offline feature store.

``--synthetic`` builds a miniature but structurally faithful lakehouse so the whole
pipeline (shard routing, PIT views, regime detection, Löwdin fusion, HRP, tickets,
reporting) can be exercised with **zero network calls** — this is what CI smoke tests
and `scripts/run_local_demo.sh` use, and it lets a reviewer see real output in ~5 s.

It is *synthetic*: numbers are generated, not market data. It never claims to be a
backtest and is always labelled as such in the run manifest and trade ticket.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .config import Config
from .filtering.embedding_projector import EventProjector, HashingBackend
from .ingestion.base import write_shard

SECTORS = ["Technology", "Industrials", "Financials", "Health Care", "Energy", "Consumer Staples"]

POLICY_HEADLINES: dict[str, tuple[str, str]] = {
    "Technology": (
        "Final rule: export control licence granted for advanced semiconductor tooling and AI compute",
        "Notice of proposed rulemaking: antitrust probe opened into semiconductor supply chain practices",
    ),
    "Industrials": (
        "Approved: infrastructure capex expansion package and defence order award for domestic manufacturers",
        "Tariff hike and safeguard duty imposed on imported heavy engineering components",
    ),
    "Financials": (
        "Rate cut approved; risk weight easing announced for bank lending to productive sectors",
        "Regulatory penalty and penalty order issued over disclosure violations",
    ),
    "Health Care": (
        "Regulatory clearance granted for new drug manufacturing capacity and incentive disbursement",
        "Price control order and windfall tax recommended for pharmaceutical imports",
    ),
    "Energy": (
        "Approved: production linked incentive for solar and grid modernisation capex",
        "Export ban and windfall tax proposed on refined fuel exports amid supply chain blockade",
    ),
    "Consumer Staples": (
        "GST rate reduction approved; subsidy for food processing capex announced",
        "Recall order and penalty issued over misstatement in packaged goods labelling",
    ),
}

GEOPOL_HEADLINES = [
    "Strait blockade escalations threaten crude and freight supply chain",
    "Sanctions package and export controls extended to critical minerals",
    "Punitive tariff hike on machinery imports announced by trading partner",
    "Energy supply disruption risk rises after pipeline halt",
]


def generate_synthetic_lakehouse(
    root: str | dt.date,
    config: Config,
    *,
    days: int = 320,
    n_tickers_per_market: int = 40,
    seed: int = 20261017,
) -> "dt.Path":  # type: ignore[valid-type]
    """Write a complete synthetic feature store and return the shards directory."""
    from pathlib import Path

    root_path = Path(root)
    shards_dir = root_path.parent / "synthetic_shards" if root_path.name == "data_lake" else root_path / "synthetic_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(dt.date.today()).normalize()
    trading_days = pd.bdate_range(end=as_of, periods=days)

    projector = EventProjector(config, backend=HashingBackend(config.e("events.embedding_dim", 384)))

    for market in ("US", "INDIA"):
        tickers = [
            f"US{i:02d}" if market == "US" else f"IN{i:02d}.NS" for i in range(n_tickers_per_market)
        ]
        sectors = {ticker: SECTORS[i % len(SECTORS)] for i, ticker in enumerate(tickers)}
        market_shock = rng.normal(0.0004, 0.010, len(trading_days))
        sector_shock = {sector: rng.normal(0.0002, 0.006, len(trading_days)) for sector in SECTORS}

        price_frames = []
        action_frames = []
        for ticker in tickers:
            drift = rng.normal(0.0003, 0.0004)
            idio = rng.normal(0.0, 0.012, len(trading_days))
            returns = drift + 0.9 * market_shock + 0.7 * sector_shock[sectors[ticker]] + idio
            close = 100.0 * np.exp(np.cumsum(returns))
            high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, len(trading_days))))
            low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, len(trading_days))))
            open_ = low + (high - low) * rng.uniform(0.1, 0.9, len(trading_days))
            volume = rng.integers(50_000, 5_000_000, len(trading_days)).astype(float)
            price_frames.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "market": market,
                        "date": trading_days,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "adj_close": close,
                        "dollar_volume": close * volume,
                        "knowledge_timestamp": [
                            pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=22 if market == "US" else 12)
                            for day in trading_days
                        ],
                    }
                )
            )
            if rng.random() < 0.25:
                split_ratio = float(rng.choice([2.0, 5.0, 10.0]))
                ex_index = int(rng.integers(days // 3, days - 5))
                action_frames.append(
                    pd.DataFrame(
                        [
                            {
                                "ticker": ticker,
                                "market": market,
                                "ex_date": trading_days[ex_index],
                                "action_type": "SPLIT",
                                "split_ratio": split_ratio,
                                "cash_dividend": 0.0,
                                "isin": f"SYN{market[:2]}{abs(hash(ticker)) % 10**8:08d}",
                                "raw_narrative": f"synthetic 1:{split_ratio:.0f} split",
                                "knowledge_timestamp": pd.Timestamp(trading_days[ex_index], tz="UTC"),
                            }
                        ]
                    )
                )

        prices = pd.concat(price_frames, ignore_index=True)
        actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
        write_shard(prices, shards_dir, f"ohlcv_{market.lower()}_0")
        if not actions.empty:
            write_shard(actions, shards_dir, f"corporate_actions_{market.lower()}_0")

        # ---- fundamentals (with business embeddings for the event projector) ----
        rows = []
        filing_ts = pd.Timestamp(trading_days[-5], tz="UTC") + pd.Timedelta(hours=13)
        for ticker in tickers:
            sector = sectors[ticker]
            description = (
                f"{ticker} operates in {sector.lower()}, manufacturing equipment, supplying government "
                f"and industrial customers, with capital expenditure on production capacity, exports and "
                f"a supply chain spanning domestic and imported components."
            )
            rows.append(
                {
                    "ticker": ticker,
                    "market": market,
                    "fiscal_period": str((filing_ts - pd.Timedelta(days=90)).date()),
                    "filing_timestamp": filing_ts,
                    "gics_sector": sector,
                    "beneish_m_score": float(rng.normal(-2.05, 0.45)),
                    "piotroski_f_score": int(np.clip(rng.normal(5.6, 2.0), 0, 9)),
                    "roic_wacc_spread": float(rng.normal(0.03, 0.07)),
                    "fcf_ev_yield": float(np.clip(rng.normal(0.035, 0.03), -0.05, 0.15)),
                    "roic": float(rng.normal(0.11, 0.06)),
                    "wacc": float(np.clip(rng.normal(0.085, 0.015), 0.05, 0.2)),
                    "fcf": float(rng.normal(2e9, 5e8)),
                    "ev": float(rng.normal(4e10, 5e9)),
                    "beta": float(np.clip(rng.normal(1.0, 0.25), 0.3, 2.2)),
                    "dollar_volume_20d": float(rng.uniform(2e7, 8e8)),
                    "promoter_holding_pct": float(np.clip(rng.normal(52, 12), 0, 90)) if market == "INDIA" else float("nan"),
                    "promoter_pledge_pct": float(np.clip(rng.normal(3, 6), 0, 45)) if market == "INDIA" else 0.0,
                    "institutional_flow_delta": float(rng.normal(0.1, 0.7)),
                    "business_description": description,
                }
            )
        fundamentals = pd.DataFrame(rows)
        vectors = projector.embed(fundamentals["business_description"].tolist())
        fundamentals["business_embedding"] = [vector.tolist() for vector in vectors]
        write_shard(fundamentals, shards_dir, f"fundamentals_{market.lower()}_0")

        # ---- technology evolution -------------------------------------------------
        tech = pd.DataFrame(
            {
                "ticker": tickers,
                "market": market,
                "knowledge_timestamp": filing_ts,
                "grants_publications_180d": rng.integers(0, 40, len(tickers)),
                "citation_velocity_log_ratio": rng.normal(0.0, 0.45, len(tickers)),
                "rd_to_revenue": np.clip(rng.normal(0.045, 0.03, len(tickers)), 0.0, 0.25),
                "gross_margin_3y_delta": rng.normal(0.01, 0.04, len(tickers)),
                "capex_to_depreciation": np.clip(rng.normal(1.4, 0.6, len(tickers)), 0.2, 4.0),
                "gics_sector": [sectors[ticker] for ticker in tickers],
            }
        )
        write_shard(tech, shards_dir, f"tech_evolution_{market.lower()}_0")

    # ---- events ------------------------------------------------------------------
    event_rows = []
    for idx, (sector, (tailwind, headwind)) in enumerate(POLICY_HEADLINES.items()):
        for offset, text in enumerate((tailwind, headwind)):
            published = pd.Timestamp(as_of, tz="UTC") - pd.Timedelta(days=int(rng.integers(1, 45)))
            event_rows.append(
                {
                    "event_id": f"SYN_FEDREG_{sector.upper()}_{offset}",
                    "published_timestamp": published,
                    "domain": "GOV_POLICY",
                    "source_agency": "synthetic-register",
                    "authority_weight": 1.0 if offset == 0 else 0.4,
                    "authority_class": "final_rule" if offset == 0 else "proposed_rule",
                    "polarity": None,
                    "half_life_days": 60.0,
                    "headline": text[:200],
                    "raw_text": f"{text}. The rule concerns the {sector.lower()} sector supply chain and capacity.",
                    "embedding": None,
                }
            )
            india_published = published - pd.Timedelta(days=2)
            event_rows.append(
                {
                    "event_id": f"SYN_PIB_{sector.upper()}_{offset}",
                    "published_timestamp": india_published,
                    "domain": "GOV_POLICY_INDIA",
                    "source_agency": "INDIA_PIB_CABINET",
                    "authority_weight": 1.0 if offset == 0 else 0.4,
                    "authority_class": "enacted" if offset == 0 else "announcement",
                    "polarity": None,
                    "half_life_days": 60.0,
                    "headline": text[:200],
                    "raw_text": f"{text}. Cabinet note for the {sector.lower()} sector and domestic capacity.",
                    "embedding": None,
                }
            )
    for idx, headline in enumerate(GEOPOL_HEADLINES):
        published = pd.Timestamp(as_of, tz="UTC") - pd.Timedelta(days=int(rng.integers(1, 30)))
        event_rows.append(
            {
                "event_id": f"SYN_GDELT_{idx}",
                "published_timestamp": published,
                "domain": "GEOPOLITICAL",
                "source_agency": "synthetic-gdelt",
                "authority_weight": 0.75,
                "authority_class": "gdelt_article",
                "polarity": None,
                "half_life_days": 14.0,
                "headline": headline,
                "raw_text": f"{headline}. Crude, freight and semiconductor supply chain implications for exporters and importers.",
                "embedding": None,
            }
        )
        event_rows.append(
            {
                "event_id": f"SYN_GDELT_INDIA_{idx}",
                "published_timestamp": published,
                "domain": "POLITICAL_INDIA",
                "source_agency": "synthetic-gdelt",
                "authority_weight": 0.75,
                "authority_class": "gdelt_article",
                "polarity": None,
                "half_life_days": 21.0,
                "headline": f"India policy continuity: {headline}",
                "raw_text": f"India parliament coalition budget debate. {headline}. Policy continuity and capex implications.",
                "embedding": None,
            }
        )
    events = projector.enrich(pd.DataFrame(event_rows), as_of=pd.Timestamp(as_of, tz="UTC").to_pydatetime())
    write_shard(events, shards_dir, "events_policy_geopol")

    # ---- macro & external --------------------------------------------------------
    macro_rows = []
    for series_id, level, vol, freq, periods in (
        ("VIX", 17.0, 0.12, "D", days),
        ("DXY", 103.0, 0.004, "D", days),
        ("BRENT", 82.0, 0.018, "D", days),
        ("YIELD_CURVE_10Y2Y", 0.35, 0.09, "D", days),
        ("HY_CREDIT_SPREAD", 3.4, 0.06, "D", days),
        ("CPI_INDEX", 320.0, 0.003, "ME", max(days // 21, 60)),
    ):
        index = pd.date_range(end=as_of, periods=int(periods), freq=freq)
        values = level * np.exp(np.cumsum(rng.normal(0.0, vol * 0.1, len(index))))
        for day, value in zip(index, values):
            macro_rows.append(
                {
                    "series_id": series_id,
                    "observation_date": day,
                    "vintage_timestamp": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23),
                    "value": float(value),
                    "z_score_252d": float("nan"),
                }
            )
    write_shard(pd.DataFrame(macro_rows), shards_dir, "macro_external_pit")

    return shards_dir
