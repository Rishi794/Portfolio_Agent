# Autonomous PMS V4 — Dual-Market Quantitative Portfolio System

**US (NYSE/NASDAQ) + India (NSE/BSE), 6-domain multi-factor pipeline, Point-in-Time
lakehouse, self-calibrating estimators, $0 runtime on GitHub Actions.**

No paid data, no servers, no cron VPS. The whole system runs on free GitHub-hosted
runners, stores its feature store as Parquet on a rolling GitHub Release, keeps its
event-sourced portfolio ledger on an orphan `data-state` branch, and publishes a
Plotly dashboard to GitHub Pages.

> ⚠️ **This is quantitative research infrastructure, not investment advice.** Trades
> only ever happen when a human collaborator comments `/approve` on the generated
> GitHub Issue.

---

## 1. What it does

| Domain | Signal | Source (all free) |
| :-- | :-- | :-- |
| **① External market factors** | Kalman trend on log-price (Myers–Tapley adaptive Q/R), cross-asset complex (VIX, DXY, Brent, SOX, yields) | Yahoo chart API, Stooq fallback |
| **② Macroeconomic conditions** | 6-dim PIT macro vector → Mahalanobis turbulence `d²` → regime + cash buffer | FRED (`FRED_API_KEY` free key; degrades gracefully without it) |
| **③ Geopolitical** | GDELT tension events embedded and projected onto company vectors | GDELT 2.0 DOC API |
| **④ Government policies** | Federal Register (US), PIB/RBI/SEBI (India) – rule stage, authority weight | Federal Register API, PIB/RBI/SEBI |
| **⑤ Technology evolution** | Citation/publication velocity, productive R&D, capex-to-depreciation modernisation | OpenAlex (keyless), optional PatentsView |
| **⑥ Company fundamentals** | Beneish M-Score, Piotroski F-Score, ROIC–WACC spread, FCF/EV, 10-K business-text embeddings | SEC EDGAR XBRL, NSE/BSE Ind-AS results & Reg-31 pledges |

The three hard problems the architecture solves explicitly:

1. **Mixed-frequency / lag mismatch** — every fact row carries a *knowledge
   timestamp* (`vintage_timestamp`, `filing_timestamp`, `published_timestamp`,
   `knowledge_timestamp`). Reads are always `ASOF`-gated: macro vintages use
   first-print-wins (ALFRED semantics), SEC facts use the first XBRL print, India
   filings use the broadcast timestamp or a conservative statutory reporting lag.
   Nothing is ever joined on the fiscal period alone.
2. **Macro/policy → ticker projection** — events and macro states live in different
   spaces from tickers. Both policy text and 10-K business descriptions are embedded
   into the same 384-dim space and linked by a bipartite cosine similarity whose
   acceptance threshold `τ̂` is derived from the **exact** spherical-null distribution
   of cosine similarity (`t = z·√((D−1)/(1−z²)) ~ Student-t(D−1)`) plus a
   Benjamini–Hochberg FDR step-up at `α = 1/√M`. No hardcoded 0.62.
3. **Multicollinearity** — the six pillars are correlated by construction. Before the
   optimizer they are fused with **symmetric (Löwdin) orthogonalization**
   `F⊥ = F·C_F^(−1/2)`, which decorrelates while minimizing `‖F⊥ − F‖_F` (keeps the
   economic interpretation of each pillar, unlike PCA rotation).

---

## 2. Zero-cost runtime topology

```
schedule/dispatch
   │
   ├── prepare            (mode = daily_eod | weekly_deep_sync, target market)
   │
   ├── company-shards     matrix: [US-0..4, INDIA-0..4]   ──┐   OHLCV + corporate actions
   │                                                       │   + PIT fundamentals + embeddings
   ├── domain-shards      macro PIT · FedReg · PIB/RBI/SEBI│
   │                      · GDELT · OpenAlex/PatentsView  ─┤
   │                                                       │
   └── fan-in  ◀──────────────────── artifacts ────────────┘
          DuckDB merge → regime cash buffer → Löwdin fusion
          → IC-IR pillar weights → True-Ward HRP → tickets
          → Issue (human approval surface) → ledger commit
          → data_lake → rolling release → Pages deploy
```

* **Concurrency**: a single global lock (`pms-v4-global-state-lock`) serializes every
  workflow — the DAG, the weekly deep sync and the `/approve` reconciler never race
  on the ledger.
* **State**: `data-state` orphan branch holds `portfolio_ledger_<market>.json`,
  `ledger_events.jsonl`, `pending_orders_<market>.json`, `nav_history.jsonl`.
* **Lakehouse**: flat Zstd Parquet (`data_lake/<table>.parquet`) shipped to the
  rolling `v-data-latest` release; GitHub's cache carries it between runs.

---

## 3. Self-calibrating parameters (no magic numbers)

Every threshold is estimated, and the estimator is recorded in the `calibration_state`
table and the trade ticket:

| Quantity | Estimator |
| :-- | :-- |
| Kalman process/observation noise `Q̂, R̂` | innovation covariance matching (Mehra 1970; Myers & Tapley 1976), PSD-projected |
| Event similarity gate `τ̂` | exact Student-t spherical null + BH-FDR at `α=1/√M` |
| Event half-life `λ̂` | impulse-response MLE per domain, clipped to config bounds |
| Round-trip cost `ĉ` | Corwin–Schultz (2012) high-low spread + statutory schedule + √-law impact |
| No-trade band `δ̂` | KKT: `δ = ĉ / (γ̂ σ̂²)` with realised risk aversion `γ̂ = μ_mkt/σ²_mkt` |
| Pillar weights `ω_t` | rolling IC-IR (Grinold–Kahn) shrunk toward the regime prior |
| Forensic gate | per-sector survival quantiles of a PIT risk index (Beneish/Piotroski/pledge), statutory fallback |
| Regime cash buffer | `cash_min + (cash_max−cash_min)·σ(γ_m(d² − χ²₆,₉₀))` |
| Covariance | Ledoit–Wolf shrinkage (sklearn) |
| Pairwise redundancy | MinHash-style shingle clustering with configured Jaccard cut |

**Two categories of numbers legitimately live in `config/`** and are labelled there:

* **Statutory / exogenous constants** — Indian STT, stamp duty, exchange txn + SEBI
  turnover fees and 18% GST on fees; US SEC §31 and FINRA TAF; Beneish's published
  −1.78 cut and Piotroski's 6 (Beneish 1999, Piotroski 2000); χ²₆ 90th percentile
  10.64. These are law or literature, not tuning knobs.
* **Publication timing gates** — when a datum becomes *knowable* (US EOD 22:00 UTC,
  India bhavcopy 12:00 UTC, FRED 23:00 UTC vintage hour, India reporting lag default
  45 days). These are timing constants that must be conservative, not statistical
  parameters; each is commented as such.

Everything else — weights, gates, bands, windows, quantiles — is estimated at run
time and stored in `calibration_state` with its method and sample size.

---

## 4. Quickstart (turnkey)

```bash
# 0. one-time bootstrap of the portfolio state branch (India ₹1,000,000 · US $100,000)
gh auth status                       # needs a token with repo + workflow scope
bash scripts/bootstrap_state_branch.sh

# 1. kick the first full DAG run
gh workflow run parallel_dag_pipeline.yml -f target_market=BOTH
```

**One-time GitHub settings** (Settings → …):

* **Pages → Build and deployment → Source: GitHub Actions** — the fan-in stage
  publishes the dashboard artifact as the Pages site.
* **Actions → General → Workflow permissions: Read and write** — the fan-in stage
  commits the ledger snapshot back to the `data-state` branch and refreshes the
  rolling `v-data-latest` release.
* Optional: add repository secrets `FRED_API_KEY` and `PATENTSVIEW_API_KEY`
  (Actions → Secrets and variables). Both feeds degrade gracefully without them.

The scheduled crons only arm **after this repository has a run of activity on the
default branch**; the `cron_heartbeat` workflow keeps them alive month after month,
and every schedule tick is gated by the exchange calendar (holidays exit in Stage 1
without starting any worker).

Then:

* the pipeline opens **“📈 V4 Rebalance Recommendation — YYYY-MM-DD”** with the full
  ticket, calibration table and order list;
* comment `/approve` to book every pending order at reference price + estimated
  spread/statutory fees, or `/approve NVDA=25@128.40 RELIANCE.NS=40@2945.50` to book
  real broker fills;
* splits and dividends are reconciled automatically before fills and recorded in
  `reconciled_ex_dates` (replaying an approval is idempotent);
* the dashboard redeploys to GitHub Pages on every run.

### Local demo (no network, no keys)

```bash
uv sync --all-extras
bash scripts/run_local_demo.sh          # synthetic feature store → ticket + dashboard
# or manually:
uv run python -m src.main_v4 --synthetic \
  --data-dir data_lake --state-dir state \
  --output-issue issue.md --dashboard-html pages/index.html --backend hashing
```

The synthetic run is clearly labelled in the manifest, the Issue and the dashboard —
it exercises every production code path with generated numbers.

### Walk-forward validation

```bash
uv run python -m src.backtest.walk_forward_v4 \
  --lake-dir data_lake --state-dir state --market BOTH
```

PIT-sliced rebalancing, Corwin–Schultz + statutory costs, KKT deadbands, and
CAGR/vol/Sharpe/Sortino/max-DD/turnover metrics written to `nav_history.jsonl`.

---

## 5. Repository layout

```
config/            engine.yaml (all priors/statutory constants), universe.yaml,
                   regime_weights.yaml, polarity.yaml
src/config.py      dotted-path config + fingerprint
src/storage/       schemas.py (table specs), lakehouse.py (PIT upsert + DuckDB),
                   verify_parquet.py (required tables + integrity, exit 1 on failure)
src/state/         ledger.py (event-sourced snapshots + pending queues)
src/ingestion/     base (retry/cache/offline), universe, market_data, macro_fred_pit,
                   policy_federal_register, policy_india, geopolitics_gdelt,
                   fundamentals_edgar_indian_xbrl, tech_patents_openalex,
                   nse_bse_bhavcopy_corporate_actions, company_shard_worker
src/filtering/     statistics (exact null + BH-FDR + shrinkage), lowdin_orthogonalizer,
                   kalman_smoother, forensics, regime_turbulence,
                   embedding_projector (ONNX/hashing, MinHash, polarity),
                   self_calibrating_engine
src/portfolio/     hrp_optimizer (True Ward HRP + α tilt + cap/turnover + tickets)
src/backtest/      walk_forward_v4
src/reporting/     issue_builder (trade ticket), dashboard_builder (Plotly Pages)
src/execution/     approve_rebalance (/approve reconciler, custom fills, CA)
src/main_v4.py     Stage-3 Fan-In reducer + CLI
src/synthetic.py   deterministic offline feature store (CI/demo)
tests/             offline unit + integration suite (no network)
scripts/           bootstrap_state_branch.sh, run_local_demo.sh
.github/workflows/ parallel_dag_pipeline.yml, weekly_deep_ingestion.yml,
                   approve_rebalance.yml
```

---

## 6. Data-source notes and honest deviations

* **Embedding backend** — primary is a quantized `Xenova/bge-small-en-v1.5` ONNX model
  (CPU, ~33 MB, cached on the runner). `onnxruntime`, `tokenizers` and
  `huggingface-hub` are an **optional extra** (`uv sync --extra onnx`) rather than core
  dependencies. When they are absent (offline CI, minimal installs) the projector
  automatically falls back to a dependency-free signed **hashing embedding**; the
  active backend and polarity mode are printed in the ticket and stored on every event,
  so the fallback can never be mistaken for the semantic model.
* **Prices** — fetched through an in-repo `httpx` client for the Yahoo chart API
  (with Stooq fallback for US, NSE bhavcopy for India). `adj_close` is the vendor's
  split/dividend-adjusted series; a re-fetch refreshes it (market data is a derived
  feed, not a public print), while *fundamental and macro* rows keep strict
  first-print-wins semantics. Price PIT discipline is enforced at read time
  (`knowledge_timestamp ≤ as_of`). Because returns are invariant to the adjustment
  scale, backtest metrics are unaffected; the live ledger additionally reconciles
  every split/dividend against `corporate_actions` at approval time.
* **India knowledge time** — NSE results carry a broadcast timestamp when available,
  otherwise `period_end + fundamentals.india_reporting_lag_days` (45). The fallback is
  conservative by construction (it can only make the system later, never earlier).
  India forensic gates therefore lean on governance proxies (Reg-31 promoter pledge,
  interest coverage) in addition to Beneish/Piotroski.
* **FRED** — with no `FRED_API_KEY` the macro pillar degrades to a neutral state
  rather than crashing; set the secret to enable the full vintage history.
* **Timing vs statistics** — see §3: publication hours and reporting lags are gates
  about *knowability*, not tuned statistics, and are documented as such in
  `config/engine.yaml`.
* **Hashing fallback is lexical, not semantic** — in fallback mode, event ↔ company
  similarity reflects token overlap. The FDR machinery still operates exactly; the
  only degradation is embedding quality.

---

## 7. Tests

```bash
uv run pytest            # 60+ offline tests, no network, no keys
```

Covers: exact spherical null + BH-FDR, Löwdin orthogonality/scale invariance,
adaptive Kalman noise learning, Beneish/Piotroski/gates, regime turbulence + crisis
classification, Ward HRP/caps/turnover/deadband tickets, Corwin–Schultz recovery,
spillover projection, IC-IR recovery, half-life MLE, MinHash dedup + polarity +
gated projection, PIT first-print-wins/snapshots/embedding-width validation,
`/approve` fills + custom fills + corporate actions + idempotency, a full synthetic
end-to-end pipeline run, and the walk-forward backtester.

---

## 8. License

MIT. Data sources remain subject to their own terms; keep request rates polite
(the HTTP client throttles and caches by default).
