"""Embedding pipeline: hashing backend, MinHash dedup, polarity, BH-FDR-gated projection."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.filtering.embedding_projector import (
    EventProjector,
    HashingBackend,
    PolarityScorer,
    minhash_clusters,
)

DIM = 384


def test_hashing_backend_is_deterministic_and_normalized() -> None:
    backend = HashingBackend(DIM)
    texts = ["semiconductor export licence granted for advanced tooling", "consumer staples food processing"]
    first = backend.encode(texts)
    second = backend.encode(texts)
    assert first.shape == (2, DIM)
    assert np.allclose(first, second)  # reproducible across processes (sha1, not hash())
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    same = backend.encode([texts[0], texts[0]])
    assert float(same[0] @ same[1]) == pytest.approx(1.0, abs=1e-6)
    disjoint = float(first[0] @ first[1])
    assert disjoint < 0.35


def test_minhash_clusters_group_syndicated_texts() -> None:
    texts = [
        "Government approves production linked incentive for solar manufacturing",
        "Government approves production linked incentive for solar manufacturing",
        "Strait blockade escalations threaten crude and freight supply chain",
    ]
    representatives, sizes = minhash_clusters(texts, threshold=0.7)
    assert representatives == [0, 2]
    assert sizes == [2, 1]


def test_polarity_lexicon_fallback_signs_headlines(config) -> None:
    scorer = PolarityScorer(HashingBackend(DIM), config)
    assert scorer.mode == "lexicon_fallback"
    scores = scorer.score(
        [
            "Cabinet approval granted subsidy for capex expansion and PLI incentive",
            "Export ban and penalty probe opened over sanctions and tariff hike",
        ],
        np.zeros((2, DIM)),
    )
    assert scores[0] > 0
    assert scores[1] < 0
    neutral = scorer.score(["Quarterly filing submitted to the exchange"], np.zeros((1, DIM)))
    assert neutral[0] == 0.0


def test_projector_gates_tailwind_event_onto_matching_company(config) -> None:
    projector = EventProjector(config, backend=HashingBackend(DIM))
    company_texts = {
        "SEMI": "Advanced semiconductor tooling manufacturer supplying AI compute fabs with export licence capabilities",
        "FOOD": "Consumer staples food processing company producing packaged goods and beverages",
    }
    company_vectors = projector.embed_company_texts(company_texts)
    published = pd.Timestamp(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2))
    events = pd.DataFrame(
        [
            {
                "event_id": "E_TAIL",
                "published_timestamp": published,
                "domain": "GOV_POLICY",
                "source_agency": "synthetic-register",
                "authority_weight": 1.0,
                "polarity": 1.0,
                "half_life_days": 60.0,
                "headline": "Final rule: export control licence granted for advanced semiconductor tooling and AI compute",
                "raw_text": "Export licence granted for advanced semiconductor tooling and AI compute fabs.",
                "embedding": projector.embed(
                    [
                        "Final rule: export control licence granted for advanced semiconductor tooling and AI compute. "
                        "Export licence granted for advanced semiconductor tooling and AI compute fabs."
                    ]
                )[0].tolist(),
            },
            {
                "event_id": "E_HEAD",
                "published_timestamp": published,
                "domain": "GOV_POLICY",
                "source_agency": "synthetic-register",
                "authority_weight": 0.4,
                "polarity": -1.0,
                "half_life_days": 60.0,
                "headline": "Notice: export ban and penalty order for consumer goods imports",
                "raw_text": "Export ban and penalty order for consumer goods imports.",
                "embedding": projector.embed(
                    ["Notice: export ban and penalty order for consumer goods imports. Export ban and penalty order."]
                )[0].tolist(),
            },
        ]
    )
    result = projector.project_to_tickers(company_vectors, events, tau_override=0.1)
    assert result.diagnostics["tau_hat"] == 0.1
    scores = result.scores
    assert set(scores.index) == {"SEMI", "FOOD"}
    assert scores["SEMI"] > 0
    assert scores["SEMI"] > scores["FOOD"]
    assert np.isfinite(scores.to_numpy()).all()


def test_projector_prior_gate_when_panel_is_small(config) -> None:
    projector = EventProjector(config, backend=HashingBackend(DIM))
    company_vectors = projector.embed_company_texts({"A": "industrial machinery manufacturer"})
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "published_timestamp": pd.Timestamp(dt.datetime.now(dt.timezone.utc)),
                "domain": "GEOPOLITICAL",
                "authority_weight": 0.75,
                "polarity": -0.5,
                "half_life_days": 14.0,
                "embedding": projector.embed(["shipping blockade disrupts industrial machinery imports"])[0].tolist(),
            }
        ]
    )
    result = projector.project_to_tickers(company_vectors, events)
    assert result.diagnostics["source"] == 2.0  # prior tau (panel below fdr_min_pairs)
    assert np.isfinite(result.scores["A"])


def test_enrich_deduplicates_and_assigns_defaults(config) -> None:
    projector = EventProjector(config, backend=HashingBackend(DIM))
    now = pd.Timestamp(dt.datetime.now(dt.timezone.utc))
    rows = []
    headlines = [
        ("POL1", "GOV_POLICY", "Government approves subsidy for grid capex"),
        ("POL1_dup", "GOV_POLICY", "Government approves subsidy for grid capex"),
        ("GEO1", "GEOPOLITICAL", "Strait blockade escalates and threatens crude freight supply chain"),
    ]
    for event_id, domain, headline in headlines:
        rows.append(
            {
                "event_id": event_id,
                "published_timestamp": now - pd.Timedelta(days=3),
                "domain": domain,
                "source_agency": "synthetic",
                "authority_weight": 0.75,
                "polarity": None,
                "half_life_days": None,
                "headline": headline,
                "raw_text": headline + ".",
                "embedding": None,
            }
        )
    enriched = projector.enrich(pd.DataFrame(rows))
    assert len(enriched) == 2  # syndicated duplicate collapsed
    assert enriched["embedding"].apply(len).eq(DIM).all()
    assert enriched["mention_cluster_size"].max() == 2.0
    assert enriched["duplicates_removed"].max() == 1.0
    assert enriched["half_life_days"].between(3.0, 180.0).all()
    assert enriched["authority_weight"].between(0.0, 1.0).all()
    # the surviving policy event must carry the lexicon polarity sign
    policy_row = enriched[enriched["domain"] == "GOV_POLICY"].iloc[0]
    assert policy_row["polarity"] > 0
