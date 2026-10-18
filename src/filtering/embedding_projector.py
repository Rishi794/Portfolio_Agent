"""Event embedding + bipartite event-to-ticker projection.

Pipeline
--------
1. ingest policy / geopolitical events (Federal Register, PIB/RBI/SEBI, GDELT);
2. embed each event into R^384 (quantized ``BAAI/bge-small-en-v1.5`` on CPU via
   ONNX Runtime; a dependency-free hashing backend is used when the ONNX stack is
   unavailable, e.g. offline CI);
3. MinHash-style near-duplicate clustering removes syndicated cables and converts
   cluster size into the *mentions* term of the authority weight;
4. polarity is assigned on a semantic axis (or the documented lexicon fallback);
5. the event is projected onto each company's 10-K business-description embedding
   with the analytically derived Student-t / Benjamini-Hochberg similarity gate.

Everything after step 2 is pure numpy: the gate threshold tau_hat is *derived* from
the exact null distribution of cosine similarity on S^{383}, never hardcoded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion.base import HttpClient, SourceUnavailable, write_shard
from .statistics import adaptive_cosine_gate, exponential_decay

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']+")
DEFAULT_EMBED_DIM = 384


# --------------------------------------------------------------------------- #
# semantic backends
# --------------------------------------------------------------------------- #
class HashingBackend:
    """Deterministic signed hashing-trick bag-of-ngrams embedder.

    Dependency-free and reproducible (uses sha1, not salted ``hash()``). It is a
    *fallback*: it preserves lexical overlap, not deep semantics. Runs offline.
    """

    name = "hashing-fallback"
    is_semantic = False

    def __init__(self, dim: int = DEFAULT_EMBED_DIM) -> None:
        self.dim = int(dim)

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(str(text).lower())
            if not tokens:
                continue
            for token in tokens:
                index, sign = self._bucket(token)
                out[row, index] += sign
            for first, second in zip(tokens, tokens[1:]):
                index, sign = self._bucket(f"{first}_{second}")
                out[row, index] += 0.5 * sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-12)).astype(np.float32)


class OnnxBgeBackend:
    """Quantized ``Xenova/bge-small-en-v1.5`` (ONNX Runtime, CPU, ~33 MB).

    Downloads once into the GitHub Actions cache (``~/.cache/onnx_models``) and then
    encodes hundreds of documents per second on a free 2-core runner.
    """

    name = "onnx-bge-small-en-v1.5"
    is_semantic = True
    dim = DEFAULT_EMBED_DIM

    def __init__(self, cache_dir: str | Path | None = None, repo_id: str = "Xenova/bge-small-en-v1.5") -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        cache = Path(cache_dir or (Path.home() / ".cache" / "onnx_models")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(
            repo_id=repo_id, filename="onnx/model_quantized.onnx", cache_dir=str(cache)
        )
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir=str(cache))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._input_names = {meta.name for meta in self.session.get_inputs()}

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encodings = self.tokenizer.encode_batch(batch)
            input_ids = np.array([item.ids for item in encodings], dtype=np.int64)
            attention = np.array([item.attention_mask for item in encodings], dtype=np.int64)
            feed: dict[str, np.ndarray] = {"input_ids": input_ids, "attention_mask": attention}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
            outputs = self.session.run(None, feed)
            cls = outputs[0][:, 0, :]
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            vectors.append((cls / np.maximum(norms, 1e-12)).astype(np.float32))
        return np.vstack(vectors)


def get_backend(prefer: str = "auto", cache_dir: str | Path | None = None) -> Any:
    """Return the best available embedding backend ('auto' → ONNX, else hashing)."""
    prefer = (prefer or "auto").lower()
    if prefer in ("auto", "onnx"):
        try:
            return OnnxBgeBackend(cache_dir=cache_dir)
        except Exception:
            if prefer == "onnx":
                raise
    return HashingBackend()


# --------------------------------------------------------------------------- #
# polarity
# --------------------------------------------------------------------------- #
class PolarityScorer:
    def __init__(self, backend: Any, config: Config) -> None:
        self.backend = backend
        self.config = config
        taxonomy = config.section("polarity")
        self.axis_scale = float(taxonomy.get("axis_projection_scale", 4.0))
        self.lexicon_scale = float(taxonomy.get("lexicon_scale", 1.5))
        anchors = taxonomy.get("axis_anchors", {}) or {}
        self.tailwind_anchor = str(anchors.get("tailwind", ""))
        self.headwind_anchor = str(anchors.get("headwind", ""))
        lexicon = taxonomy.get("fallback_lexicon", {}) or {}
        self.tailwind_terms = {str(term).lower() for term in lexicon.get("tailwind_terms", [])}
        self.headwind_terms = {str(term).lower() for term in lexicon.get("headwind_terms", [])}
        self.mode = "embedding_axis" if getattr(backend, "is_semantic", False) else "lexicon_fallback"

    def score(self, texts: Sequence[str], embeddings: np.ndarray) -> np.ndarray:
        if self.mode == "embedding_axis" and embeddings.size:
            anchors = self.backend.encode([self.tailwind_anchor, self.headwind_anchor])
            axis = anchors[0] - anchors[1]
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                raw = (embeddings @ (axis / norm)) * self.axis_scale
                return np.tanh(raw)
        return self._lexicon(texts)

    def _lexicon(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=float)
        tailwind_hits: list[int] = []
        for row, text in enumerate(texts):
            tokens = set(_TOKEN.findall(str(text).lower()))
            tailwind = len(tokens & self.tailwind_terms)
            headwind = len(tokens & self.headwind_terms)
            tailwind_hits.append(tailwind)
            total = tailwind + headwind
            if total == 0:
                out[row] = 0.0
            else:
                out[row] = np.tanh(self.lexicon_scale * (tailwind - headwind) / total)
        return out


# --------------------------------------------------------------------------- #
# near-duplicate clustering (MinHash-style, no external dependency)
# --------------------------------------------------------------------------- #
def _shingles(text: str, k: int = 4) -> set[int]:
    tokens = _TOKEN.findall(str(text).lower())
    if len(tokens) < k:
        return {hash(tuple(tokens))} if tokens else set()
    return {hash(tuple(tokens[i : i + k])) for i in range(len(tokens) - k + 1)}


def _minhash_signature(shingles: set[int], num_perm: int) -> np.ndarray:
    if not shingles:
        return np.full(num_perm, np.iinfo(np.int64).max, dtype=np.int64)
    signature = np.full(num_perm, np.iinfo(np.int64).max, dtype=np.int64)
    for seed in range(num_perm):
        best = np.iinfo(np.int64).max
        for shingle in shingles:
            hashed = (shingle * 2654435761 + seed * 40503) & ((1 << 63) - 1)
            if hashed < best:
                best = hashed
        signature[seed] = best
    return signature


def minhash_clusters(texts: Sequence[str], *, num_perm: int = 32, threshold: float = 0.7) -> tuple[list[int], list[int]]:
    """Greedy LSH clustering; returns (representative indices, cluster sizes)."""
    signatures = [_minhash_signature(_shingles(text), num_perm) for text in texts]
    representatives: list[int] = []
    sizes: list[int] = []
    for idx, signature in enumerate(signatures):
        assigned = False
        for pos, rep in enumerate(representatives):
            similarity = float(np.mean(signatures[rep] == signature))
            if similarity >= threshold:
                sizes[pos] += 1
                assigned = True
                break
        if not assigned:
            representatives.append(idx)
            sizes.append(1)
    order = np.argsort(representatives)
    return [representatives[i] for i in order], [sizes[i] for i in order]


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
@dataclass
class ProjectionResult:
    scores: pd.Series
    diagnostics: dict[str, Any]


class EventProjector:
    def __init__(self, config: Config, backend: Any | None = None) -> None:
        self.config = config
        self.spec_backend = str(config.e("events.backend", "auto"))
        self.backend = backend or get_backend(self.spec_backend)
        self.dim = int(config.e("events.embedding_dim", DEFAULT_EMBED_DIM))
        self.polarity = PolarityScorer(self.backend, config)
        self.prior_tau = float(config.e("events.gate_tau_prior", 0.62))
        self.prior_eta = float(config.e("events.prior_eta", 1.0))
        self.min_pairs = int(config.e("events.fdr_min_pairs", 100))
        self.alpha_scale = float(config.e("events.fdr_alpha_scale", 1.0))
        self.fallback_p = float(config.e("events.fdr_fallback_p", 0.05))
        self.mention_saturation = float(
            (config.e("events.authority_weights", {}) or {}).get("gdelt_mention_saturation", 100.0)
        )
        self.max_half_life = float(config.e("events.half_life_max_days", 180.0))
        self.min_half_life = float(config.e("events.half_life_min_days", 3.0))

    # ---------------------------------------------------------------- encode ---
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.asarray(self.backend.encode(list(texts)), dtype=np.float32)
        if vectors.shape[1] != self.dim:
            vectors = _resize(vectors, self.dim)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def embed_company_texts(self, mapping: dict[str, str]) -> pd.DataFrame:
        tickers = list(mapping)
        vectors = self.embed([mapping[ticker] for ticker in tickers])
        return pd.DataFrame(vectors, index=tickers)

    # -------------------------------------------------------------- enrich -----
    def enrich(self, events: pd.DataFrame, *, as_of: dt.datetime | None = None) -> pd.DataFrame:
        """Deduplicate, embed, score polarity and set authority/half-life on events."""
        if events is None or events.empty:
            return pd.DataFrame(columns=_EVENT_COLUMNS)
        frame = events.copy()
        for column in ("embedding", "polarity", "authority_weight", "half_life_days", "headline", "raw_text", "domain"):
            if column not in frame.columns:
                frame[column] = None
        frame["published_timestamp"] = pd.to_datetime(frame["published_timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["published_timestamp"]).sort_values("published_timestamp")
        frame = frame.drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)

        texts = [
            f"{row.headline or ''}. {row.raw_text or ''}".strip()
            for row in frame.itertuples(index=False)
        ]
        vectors = self.embed(texts)
        polarities = self.polarity.score(texts, vectors)

        representatives, cluster_sizes = minhash_clusters(texts, threshold=0.7)
        keep_mask = np.zeros(len(frame), dtype=bool)
        keep_mask[representatives] = True
        mentions = np.zeros(len(frame), dtype=float)
        mentions[representatives] = np.asarray(cluster_sizes, dtype=float)

        deduped = frame.loc[keep_mask].reset_index(drop=True)
        deduped_vectors = vectors[keep_mask]
        deduped_polarities = polarities[keep_mask]
        deduped_mentions = mentions[keep_mask]

        base = pd.to_numeric(deduped["authority_weight"], errors="coerce").fillna(0.75)
        is_media = deduped.get("authority_class", pd.Series("", index=deduped.index)).astype(str).str.startswith("gdelt")
        mention_factor = np.minimum(1.0, deduped_mentions / max(self.mention_saturation, 1.0))
        severity = np.abs(deduped_polarities) if self.backend.is_semantic else np.ones(len(deduped))
        authority = base.to_numpy(dtype=float) * mention_factor
        authority = np.where(is_media.to_numpy(), authority * severity, base.to_numpy(dtype=float))
        authority = np.clip(authority, 0.0, 1.0)

        half_lives = pd.to_numeric(deduped["half_life_days"], errors="coerce").to_numpy(dtype=float)
        defaults = np.array([self.config.half_life_default(str(domain)) for domain in deduped["domain"]])
        half_lives = np.where(np.isfinite(half_lives) & (half_lives > 0), half_lives, defaults)
        half_lives = np.clip(half_lives, self.min_half_life, self.max_half_life)

        deduped["embedding"] = [vector.tolist() for vector in deduped_vectors]
        deduped["polarity"] = deduped_polarities
        deduped["authority_weight"] = authority
        deduped["half_life_days"] = half_lives
        deduped["mention_cluster_size"] = deduped_mentions
        deduped["duplicates_removed"] = np.where(mention_factor > 0, deduped_mentions - 1.0, 0.0)
        return deduped

    # ------------------------------------------------------------ projection ---
    def project_to_tickers(
        self,
        company_embeddings: pd.DataFrame,
        events: pd.DataFrame,
        *,
        as_of: dt.datetime | None = None,
        tau_override: float | None = None,
    ) -> ProjectionResult:
        index = company_embeddings.index if company_embeddings is not None else pd.Index([], name="ticker")
        if company_embeddings is None or company_embeddings.empty or events is None or events.empty:
            return ProjectionResult(pd.Series(0.0, index=index, dtype=float), {"tau_hat": float("nan"), "events": 0.0})

        now = as_of or dt.datetime.now(dt.timezone.utc)
        if isinstance(now, pd.Timestamp):
            now = now.to_pydatetime()
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt.timezone.utc)

        events = events.copy()
        events = events[events["embedding"].notna()]
        published = pd.to_datetime(events["published_timestamp"], utc=True, errors="coerce")
        valid = published.notna()
        events = events[valid]
        published = published[valid]
        if events.empty:
            return ProjectionResult(pd.Series(0.0, index=index, dtype=float), {"tau_hat": float("nan"), "events": 0.0})

        event_vectors = np.vstack([np.asarray(vec, dtype=np.float32) for vec in events["embedding"]])
        age_days = np.array(
            [max((now - stamp.to_pydatetime()).total_seconds() / 86_400.0, 0.0) for stamp in published],
            dtype=float,
        )

        half_lives = pd.to_numeric(events["half_life_days"], errors="coerce").fillna(
            self.config.half_life_default("GEOPOLITICAL")
        ).to_numpy(dtype=float)
        authority = pd.to_numeric(events["authority_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        polarity = pd.to_numeric(events["polarity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        decay = exponential_decay(age_days, half_lives)

        company_vectors = np.asarray(company_embeddings.to_numpy(dtype=np.float32), dtype=np.float32)
        cosine = np.clip(company_vectors @ event_vectors.T, -0.9999, 0.9999)

        if tau_override is not None:
            tau_hat = float(tau_override)
            gate_stats = {"tau_hat": tau_hat, "source": 1.0}
        elif cosine.size >= self.min_pairs:
            gate = adaptive_cosine_gate(
                cosine,
                self.dim,
                alpha_scale=self.alpha_scale,
                min_pairs=self.min_pairs,
                fallback_p=self.fallback_p,
            )
            tau_hat = float(gate["tau_hat"])
            gate_stats = dict(gate)
            gate_stats["source"] = 0.0
        else:
            tau_hat = self.prior_tau
            gate_stats = {"tau_hat": tau_hat, "pairs": float(cosine.size), "source": 2.0}

        gated = np.maximum(0.0, (cosine - tau_hat) / max(1.0 - tau_hat, 1e-6))
        weighted = authority * polarity * decay
        numerator = gated @ weighted
        denominator = float(np.sum(authority * decay)) + self.prior_eta
        scores = pd.Series(numerator / denominator, index=index, dtype=float)

        diagnostics = {
            **gate_stats,
            "events": float(len(events)),
            "mean_abs_polarity": float(np.mean(np.abs(polarity))) if len(polarity) else 0.0,
            "max_gated_similarity": float(gated.max()) if gated.size else 0.0,
            "backend": self.backend.name,
            "polarity_mode": self.polarity.mode,
            "prior_eta": self.prior_eta,
        }
        return ProjectionResult(scores, diagnostics)


# --------------------------------------------------------------------------- #
# helpers + CLI
# --------------------------------------------------------------------------- #
_EVENT_COLUMNS = [
    "event_id",
    "published_timestamp",
    "domain",
    "source_agency",
    "authority_weight",
    "authority_class",
    "polarity",
    "half_life_days",
    "headline",
    "raw_text",
    "embedding",
]


def _resize(vectors: np.ndarray, dim: int) -> np.ndarray:
    if vectors.shape[1] > dim:
        return vectors[:, :dim]
    pad = np.zeros((vectors.shape[0], dim - vectors.shape[1]), dtype=np.float32)
    return np.hstack([vectors, pad])


def company_text_from_row(row: pd.Series) -> str:
    parts = [
        str(row.get("name", "") or ""),
        str(row.get("gics_sector", "") or ""),
        str(row.get("business_description", "") or ""),
    ]
    return ". ".join(part for part in parts if part).strip()


def ingest_domain(domain: str, config: Config, client: HttpClient, *, file: Path | None = None) -> pd.DataFrame:
    """Dispatch to the ingestion module for a domain shard (or read a cached shard)."""
    if file is not None and Path(file).exists():
        return pd.read_parquet(file)
    if domain == "policy_us_fedreg":
        from ..ingestion.policy_federal_register import FederalRegisterIngestor

        since = (dt.date.today() - dt.timedelta(days=7)).isoformat()
        return FederalRegisterIngestor(config, client).fetch(since)
    if domain == "policy_in_pib_rbi_sebi":
        from ..ingestion.policy_india import IndiaPolicyIngestor

        return IndiaPolicyIngestor(config, client).fetch()
    if domain == "geopol_gdelt":
        from ..ingestion.geopolitics_gdelt import GeopoliticsIngestor

        return GeopoliticsIngestor(config, client).fetch()
    if domain == "macro_us_in":
        from ..ingestion.macro_fred_pit import MacroExternalIngestor

        import os

        return MacroExternalIngestor(config, client, fred_api_key=os.getenv("FRED_API_KEY")).build()
    if domain.startswith("tech_"):
        from ..ingestion.tech_patents_openalex import PatentAndTechEvolutionIngestor

        return PatentAndTechEvolutionIngestor(config, client).fetch_universe()
    raise KeyError(f"unknown domain {domain!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy / geopolitical ONNX embedding projector")
    parser.add_argument("--ingest-domain", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-shard", type=Path, default=None, help="Source parquet instead of network")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--backend", default=None, choices=[None, "auto", "onnx", "hashing"])
    args = parser.parse_args(argv)

    config = Config.load()
    # GDELT requires its own pacing; every other domain keeps the shared defaults.
    if args.ingest_domain == "geopol_gdelt":
        from ..ingestion.geopolitics_gdelt import make_client

        client = make_client(config, cache_dir=Path.home() / ".cache" / "pms_http", offline=args.offline)
    else:
        client = HttpClient(
            cache_dir=Path.home() / ".cache" / "pms_http",
            offline=args.offline,
            headers={"Accept": "application/json, application/rss+xml, text/xml, */*"},
        )
    projector = EventProjector(config) if not args.backend else EventProjector(
        config, backend=get_backend(args.backend)
    )
    events = ingest_domain(args.ingest_domain, config, client, file=args.input_shard)
    enriched = projector.enrich(events)
    if not enriched.empty:
        enriched["knowledge_timestamp"] = args.as_of or dt.datetime.now(dt.timezone.utc).isoformat()
    path = write_shard(enriched, args.output_dir, f"events_{args.ingest_domain}")
    print(
        f"[{projector.backend.name} / {projector.polarity.mode}] wrote {len(enriched)} events -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
