"""Stage-3 Fan-In reducer: DuckDB merge, self-calibrating factor fusion, HRP, tickets.

Run modes
---------
``daily_eod``        Mon-Fri fast path: prices, macro, events, regime, HRP, tickets.
``weekly_deep_sync`` Sunday: SEC/NSE XBRL fundamentals, patents, 10-K embeddings.
``auto``             derived from the workflow schedule / CLI flags.

The reducer is deliberately deterministic given ``(config, lakehouse, as_of)``: every
number it emits is either read from the lakehouse or estimated from it, and all
estimator settings come from ``config/``. ``--synthetic`` bypasses the network
entirely (used by the offline CI smoke test and for local demos).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .filtering.embedding_projector import EventProjector, get_backend
from .filtering.forensics import (
    estimate_survival_gate,
    fundamental_quality_score,
    gate_mask,
)
from .filtering.lowdin_orthogonalizer import (
    condition_number,
    lowdin_orthogonalize,
    max_abs_correlation,
    orthogonality_error,
)
from .filtering.regime_turbulence import MacroRegimeDetector, RegimeAllocation
from .filtering.self_calibrating_engine import SelfCalibratingQuantEngine
from .filtering.statistics import sector_zscore, winsorize, zscore
from .portfolio.hrp_optimizer import HRPOptimizer
from .state.ledger import Ledger
from .storage.lakehouse import Lakehouse, manifest_row

PILLARS = ("kalman_trend", "fundamentals", "tech_evolution", "gov_policy", "geopolitics", "macro_beta")

SHARD_ROUTES: dict[str, str] = {
    "ohlcv": "market_prices",
    "market_prices": "market_prices",
    "corporate_actions": "corporate_actions",
    "fundamentals": "fundamentals_pit",
    "events": "policy_geopol_events",
    "macro_external_pit": "macro_external_pit",
    "tech_patents_velocity": "tech_evolution_pit",
    "tech_evolution": "tech_evolution_pit",
    "spillover": "spillover_us_india",
}

# Market data is a *vendor-derived* feed, not a public print: re-ingesting the same
# (ticker, date) must refresh ``adj_close`` (Yahoo re-adjusts for later splits).
# PIT discipline for prices is enforced at *read* time instead: ``price_snapshot``
# only exposes rows whose ``knowledge_timestamp`` <= as_of. Fundamental/macro prints
# keep first-print-wins semantics.
LATEST_WINS_TABLES = {"market_prices", "corporate_actions", "policy_geopol_events", "spillover_us_india"}


@dataclass
class MarketRun:
    market: str
    regime: RegimeAllocation
    universe_size: int
    eligible_size: int
    target_weights: pd.Series
    alpha: pd.Series
    tickets: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class V4AutonomousPortfolioReducer:
    def __init__(
        self,
        config: Config,
        lake_dir: str | Path,
        state_dir: str | Path,
        *,
        shards_dir: str | Path | None = None,
        as_of: dt.date | None = None,
        offline: bool = False,
        embed_backend: str = "auto",
        starting_cash: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.lake = Lakehouse(lake_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir = Path(shards_dir) if shards_dir else None
        self.as_of = pd.Timestamp(as_of or dt.date.today()).normalize()
        self.offline = offline
        self.embed_backend = embed_backend
        self.starting_cash = starting_cash or {"US": 100_000.0, "INDIA": 1_000_000.0}
        self.engine = SelfCalibratingQuantEngine(config)
        self.regime_detector = MacroRegimeDetector(config)
        self.optimizer = HRPOptimizer(config)
        self.run_id = f"{uuid.uuid4().hex[:8]}-{self.as_of:%Y%m%d}"
        self.calibration_rows: list[dict[str, Any]] = []

    # ===================================================================== ingest
    def ingest_shards(self) -> dict[str, int]:
        written: dict[str, int] = {}
        if not self.shards_dir or not self.shards_dir.exists():
            return written
        for path in sorted(self.shards_dir.glob("*.parquet")):
            stem = path.stem
            route = next((table for prefix, table in SHARD_ROUTES.items() if stem.startswith(prefix)), None)
            if route is None:
                continue
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            if "market" not in frame.columns and route in {"market_prices", "fundamentals_pit", "corporate_actions"}:
                market = "INDIA" if "india" in stem else "US"
                frame = frame.assign(market=market)
            count = self.lake.write(route, frame, enforce_pit=route not in LATEST_WINS_TABLES)
            written[route] = written.get(route, 0) + count
        return written

    # ====================================================================== view
    def price_snapshot(self, market: str) -> pd.DataFrame:
        prices = self.lake.read("market_prices")
        if prices.empty:
            return prices
        frame = prices[prices["market"].astype(str) == market].copy()
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[pd.to_datetime(frame["knowledge_timestamp"], utc=True) <= pd.Timestamp(self.as_of, tz="UTC")]
        frame = frame[frame["date"] <= self.as_of]
        frame["adj_close"] = pd.to_numeric(frame.get("adj_close"), errors="coerce").fillna(
            pd.to_numeric(frame["close"], errors="coerce")
        )
        frame = frame.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
        return frame

    def close_wide(self, market: str) -> pd.DataFrame:
        frame = self.price_snapshot(market)
        if frame.empty:
            return pd.DataFrame()
        return frame.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()

    def query_pit_split_adjusted_universe(self, market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Long-format PIT price panel + fundamentals snapshot (used by the backtester)."""
        frame = self.price_snapshot(market)
        if frame.empty:
            return pd.DataFrame(), self.fundamentals_snapshot(market)
        long = frame.rename(columns={"date": "trade_date"})[
            ["trade_date", "ticker", "market", "open", "high", "low", "close", "adj_close", "volume", "dollar_volume"]
        ].copy()
        long["adj_high"] = long["high"]
        long["adj_low"] = long["low"]
        if "dollar_volume" not in long.columns or long["dollar_volume"].isna().all():
            long["dollar_volume"] = long["adj_close"] * pd.to_numeric(long["volume"], errors="coerce").fillna(0.0)
        return long, self.fundamentals_snapshot(market)

    def macro_pit(self) -> pd.DataFrame:
        macro = self.lake.read("macro_external_pit")
        if macro.empty:
            return macro
        macro = macro[pd.to_datetime(macro["vintage_timestamp"], utc=True) <= pd.Timestamp(self.as_of, tz="UTC")]
        return macro.sort_values("observation_date")

    def events_snapshot(self) -> pd.DataFrame:
        events = self.lake.read("policy_geopol_events")
        if events.empty:
            return events
        events = events[pd.to_datetime(events["published_timestamp"], utc=True) <= pd.Timestamp(self.as_of, tz="UTC")]
        return events.reset_index(drop=True)

    def fundamentals_snapshot(self, market: str) -> pd.DataFrame:
        frame = self.lake.snapshot_at("fundamentals_pit", self.as_of, entity=["ticker"])
        if frame.empty:
            return frame
        frame = frame[frame["market"].astype(str) == market].copy()
        if frame.empty:
            return frame
        frame = frame.set_index("ticker")
        for column in ("piotroski_f_score", "beneish_m_score", "roic_wacc_spread", "fcf_ev_yield"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def tech_snapshot(self, market: str) -> pd.DataFrame:
        frame = self.lake.snapshot_at("tech_evolution_pit", self.as_of, entity=["ticker"])
        if frame.empty:
            return frame
        frame = frame[frame["market"].astype(str) == market]
        if frame.empty:
            return frame
        return frame.set_index("ticker")

    def company_embeddings(self, market: str) -> pd.DataFrame:
        frame = self.fundamentals_snapshot(market)
        if frame.empty or "business_embedding" not in frame.columns:
            return pd.DataFrame()
        rows = {}
        for ticker, embedding in frame["business_embedding"].items():
            if embedding is None:
                continue
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.size == 0:
                continue
            rows[ticker] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).T

    # ===================================================================== factors
    def build_returns(self, market: str, lookback_days: int | None = None) -> pd.DataFrame:
        wide = self.close_wide(market)
        if wide.empty:
            return wide
        lookback = int(lookback_days or self.config.e("optimizer.cov_lookback_days", 160))
        returns = wide.tail(lookback).pct_change(fill_method=None)
        return returns.dropna(how="all")

    def factor_macro_beta(self, market: str, returns: pd.DataFrame, macro_matrix: pd.DataFrame) -> pd.Series:
        if returns.empty or macro_matrix.empty:
            return pd.Series(0.0, index=returns.columns, dtype=float)
        shocks = macro_matrix.reindex(returns.index).ffill().dropna()
        if len(shocks) < int(self.config.e("macro_beta.min_observations", 60)):
            return pd.Series(0.0, index=returns.columns, dtype=float)
        aligned = returns.reindex(shocks.index).dropna(how="all")
        x = shocks.to_numpy(dtype=float)
        lam = float(self.config.e("macro_beta.ridge_lambda", 1e-3))
        gram = x.T @ x + lam * np.eye(x.shape[1])
        expected: dict[str, float] = {}
        latest = x[-1]
        for ticker in aligned.columns:
            y = aligned[ticker]
            mask = y.notna().to_numpy()
            if mask.sum() < int(self.config.e("macro_beta.min_observations", 60)):
                continue
            beta = np.linalg.solve(gram, x[mask].T @ y.to_numpy(dtype=float)[mask])
            expected[ticker] = float(beta @ latest)
        if not expected:
            return pd.Series(0.0, index=returns.columns, dtype=float)
        scores = pd.Series(expected, dtype=float)

        blend = float(self.config.e("spillover.blend_weight", 0.5))
        spin = self.spillover_scores(market, returns)
        if not spin.empty:
            common = scores.index.intersection(spin.index)
            scores.loc[common] = (1.0 - blend) * zscore(scores.loc[common], 0.01) + blend * zscore(spin.loc[common], 0.01)
        return zscore(winsorize(scores, 0.01), 0.0)

    def spillover_scores(self, market: str, returns: pd.DataFrame) -> pd.Series:
        provider = str(self.config.e("spillover.provider_market", "US"))
        if market == provider:
            return pd.Series(dtype=float)
        us_wide = self.close_wide(provider)
        if us_wide.empty or returns.empty:
            return pd.Series(dtype=float)

        spec = self.config.get("universe.indices_for_spillover", {}) or {}
        drivers = [str(t) for t in (spec.get("provider") or [])]
        adr = [str(t) for t in (spec.get("adr_proxies") or [])]
        us_returns = us_wide.pct_change(fill_method=None)
        identified = [ticker for ticker in drivers + adr if ticker in us_returns.columns]

        series: dict[str, pd.Series] = {ticker: us_returns[ticker] for ticker in identified}

        if len(identified) < 2:
            # Cross-asset ETF/index tickers are not in the US price universe yet: fall back to
            # the external macro complex (all PIT) so the spillover leg still has real drivers.
            external = self.macro_pit()
            if external is not None and not external.empty:
                cross = external[external["series_id"].isin(["VIX", "DXY", "BRENT", "YIELD_CURVE_10Y2Y", "HY_CREDIT_SPREAD"])]
                if not cross.empty:
                    wide = cross.pivot_table(index="observation_date", columns="series_id", values="value", aggfunc="last")
                    wide.index = pd.to_datetime(wide.index)
                    wide = wide.ffill()
                    for column in wide.columns:
                        value = pd.to_numeric(wide[column], errors="coerce")
                        series[f"EXT_{column}"] = value.diff() if column in ("VIX", "YIELD_CURVE_10Y2Y", "HY_CREDIT_SPREAD") else value.pct_change()
        if not series:
            return pd.Series(dtype=float)

        factors = pd.DataFrame(series).replace([np.inf, -np.inf], np.nan)
        # Overnight discipline: the US close (23:15 UTC) precedes the *next* Indian session,
        # so US factors are lagged one session before they are projected onto India returns.
        factors = factors.shift(1).dropna(how="all")
        if factors.empty:
            return pd.Series(dtype=float)

        aligned_index = factors.index.intersection(returns.index)
        if len(aligned_index) < int(self.config.e("spillover.min_observations", 30)):
            return pd.Series(dtype=float)
        india_returns = returns.reindex(aligned_index)
        us_factors = factors.reindex(aligned_index).fillna(0.0)
        projected = self.engine.project_us_to_india_spillover(india_returns, us_factors)
        diagnostics = self.engine.spillover_diagnostics(india_returns, us_factors)
        self.calibration_rows.append(
            {
                "as_of_date": self.as_of,
                "scope": f"{market}:spillover",
                "parameter": "r_squared",
                "method": "ledoit_wolf_overnight_lead_lag",
                "value": float(diagnostics.get("r_squared", np.nan)),
                "sample_size": int(diagnostics.get("observations", 0)),
            }
        )
        if not projected.empty:
            stored = projected.rename("projected_beta_return").reset_index()
            stored.columns = ["ticker", "projected_beta_return"]
            stored["as_of_date"] = self.as_of
            stored["driver_vector"] = [factors.iloc[-1].fillna(0.0).to_numpy(dtype=float).tolist()] * len(stored)
            stored["r_squared"] = float(diagnostics.get("r_squared", np.nan))
            self.lake.write("spillover_us_india", stored, enforce_pit=False)
        return projected

    def factor_events(
        self,
        market: str,
        company_vectors: pd.DataFrame,
        events: pd.DataFrame,
        *,
        domain_key: str,
    ) -> tuple[pd.Series, dict[str, Any]]:
        if company_vectors.empty or events.empty:
            return pd.Series(0.0, index=company_vectors.index, dtype=float), {"events": 0.0}
        domains = (self.config.e("events.market_domains", {}) or {}).get(market, {}).get(domain_key, [])
        if not domains:
            return pd.Series(0.0, index=company_vectors.index, dtype=float), {"events": 0.0}
        subset = events[events["domain"].astype(str).isin([str(domain) for domain in domains])]
        if subset.empty:
            return pd.Series(0.0, index=company_vectors.index, dtype=float), {"events": 0.0}
        projector = EventProjector(self.config, backend=get_backend(self.embed_backend))
        result = projector.project_to_tickers(company_vectors, subset, as_of=pd.Timestamp(self.as_of, tz="UTC").to_pydatetime())
        diagnostics = dict(result.diagnostics)
        diagnostics["domain_key"] = domain_key
        diagnostics["domains"] = ",".join(map(str, domains))
        return result.scores, diagnostics

    def factor_tech_evolution(self, market: str, tech: pd.DataFrame, universe_index: pd.Index) -> pd.Series:
        if tech.empty:
            return pd.Series(0.0, index=universe_index, dtype=float)
        frame = tech.reindex(universe_index).copy()
        weights = self.config.e("tech_evolution.weights", [0.45, 0.35, 0.20]) or [0.45, 0.35, 0.20]
        pa = pd.to_numeric(frame.get("citation_velocity_log_ratio"), errors="coerce").fillna(0.0)
        rd_ratio = pd.to_numeric(frame.get("rd_to_revenue"), errors="coerce").fillna(0.0)
        gm_delta = pd.to_numeric(frame.get("gross_margin_3y_delta"), errors="coerce").fillna(0.0)
        cmr = pd.to_numeric(frame.get("capex_to_depreciation"), errors="coerce").fillna(1.0)
        work = pd.DataFrame(
            {
                "patent_acceleration": pa,
                "productive_rd": rd_ratio * (1.0 + gm_delta),
                "capex_modernisation": cmr,
                "gics_sector": frame.get("gics_sector", pd.Series("UNKNOWN", index=frame.index)),
            }
        )
        score = (
            float(weights[0]) * sector_zscore(work, "patent_acceleration", winsor_quantile=0.01)
            + float(weights[1]) * sector_zscore(work, "productive_rd", winsor_quantile=0.01)
            + float(weights[2]) * sector_zscore(work, "capex_modernisation", winsor_quantile=0.01)
        )
        return zscore(score, 0.0)

    def factor_fundamentals(self, market: str, fundamentals: pd.DataFrame, universe_index: pd.Index) -> tuple[pd.Series, pd.Series]:
        if fundamentals.empty:
            return (
                pd.Series(0.0, index=universe_index, dtype=float),
                pd.Series(False, index=universe_index, dtype=bool),
            )
        frame = fundamentals.reindex(universe_index)
        frame = frame.assign(
            gics_sector=frame.get("gics_sector", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN")
        )
        frame["dollar_volume_20d"] = pd.to_numeric(frame.get("dollar_volume_20d"), errors="coerce")
        thresholds = estimate_survival_gate(frame, self.config)
        mask = gate_mask(
            frame,
            thresholds,
            min_dollar_volume=float(self.config.e("forensics.min_dollar_volume", 15_000_000.0)),
        )
        quality_weights = tuple(self.config.e("forensics.quality_weights", [0.45, 0.35, 0.20]))
        quality = fundamental_quality_score(
            frame,
            quality_weights,  # type: ignore[arg-type]
            winsor_quantile=float(self.config.e("forensics.winsor_quantile", 0.01)),
        )
        score = zscore(quality, 0.0).where(pd.Series(True, index=frame.index), 0.0)
        gate_rows = {
            sector: {
                "beneish_max": threshold.beneish_max,
                "fscore_min": threshold.fscore_min,
                "mode": threshold.mode,
                "sample": threshold.sample_size,
            }
            for sector, threshold in thresholds.items()
        }
        self.calibration_rows.append(
            {
                "as_of_date": self.as_of,
                "scope": f"{market}:forensic_gate",
                "parameter": "gate_mode",
                "method": str(self.config.e("forensics.gate_mode", "quantile")),
                "value": float(len(gate_rows)),
                "sample_size": int(len(frame)),
            }
        )
        for pillar_sector, payload in gate_rows.items():
            self.calibration_rows.append(
                {
                    "as_of_date": self.as_of,
                    "scope": f"{market}:gate:{pillar_sector}",
                    "parameter": "beneish_max",
                    "method": payload["mode"],
                    "value": float(payload["beneish_max"]),
                    "sample_size": int(payload["sample"]),
                }
            )
        return score, mask

    def build_factor_matrix(
        self,
        market: str,
        regime: RegimeAllocation,
        macro_matrix: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        returns = self.build_returns(market)
        diagnostics: dict[str, Any] = {"markets": market}
        if returns.empty:
            return pd.DataFrame(), diagnostics
        close_wide = self.close_wide(market)
        universe_index = returns.columns

        trend = self.engine.kalman_trend_scores(close_wide, min_observations=int(self.config.e("macro_beta.min_observations", 60) // 2))
        trend = zscore(trend.reindex(universe_index).fillna(0.0), 0.0)

        fundamentals = self.fundamentals_snapshot(market)
        quality, gate = self.factor_fundamentals(market, fundamentals, universe_index)

        tech = self.tech_snapshot(market)
        tech_scores = self.factor_tech_evolution(market, tech, universe_index)

        company_vectors = self.company_embeddings(market)
        events = self.events_snapshot()
        policy_scores, policy_diag = self.factor_events(market, company_vectors, events, domain_key="gov_policy")
        geopol_scores, geopol_diag = self.factor_events(market, company_vectors, events, domain_key="geopolitics")
        policy_scores = zscore(policy_scores.reindex(universe_index).fillna(0.0), 0.0)
        geopol_scores = zscore(geopol_scores.reindex(universe_index).fillna(0.0), 0.0)

        macro_beta = self.factor_macro_beta(market, returns, macro_matrix)
        macro_beta = zscore(macro_beta.reindex(universe_index).fillna(0.0), 0.0)

        factors = pd.DataFrame(
            {
                "kalman_trend": trend,
                "fundamentals": quality,
                "tech_evolution": tech_scores,
                "gov_policy": policy_scores,
                "geopolitics": geopol_scores,
                "macro_beta": macro_beta,
            }
        )
        factors = factors[list(PILLARS)]
        metadata = pd.DataFrame(
            {
                "eligible": gate.reindex(universe_index).fillna(False),
                "price": close_wide.iloc[-1].reindex(universe_index) if not close_wide.empty else np.nan,
                "gics_sector": pd.Series("UNKNOWN", index=universe_index, dtype="object"),
            },
            index=universe_index,
        )
        if not fundamentals.empty and "gics_sector" in fundamentals.columns:
            metadata["gics_sector"] = fundamentals["gics_sector"].reindex(universe_index).fillna("UNKNOWN")
        factors = factors.join(metadata)
        diagnostics.update(
            {
                "policy": policy_diag,
                "geopolitics": geopol_diag,
                "universe": len(universe_index),
                "eligible": int(metadata["eligible"].sum()),
                "raw_max_abs_correlation": max_abs_correlation(factors[list(PILLARS)]),
                "raw_condition_number": condition_number(factors[list(PILLARS)]),
            }
        )
        return factors, diagnostics

    # ======================================================================= fuse
    def pillar_weights(self, market: str, regime: RegimeAllocation, factors: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        mode = str(self.config.e("factor_learning.mode", "ic_ir"))
        prior = regime.domain_weights
        if mode != "ic_ir":
            return prior, {"mode": "regime_prior", "observations": 0}
        history = self._historical_ic_panels(market)
        if not history["factors"]:
            return prior, {"mode": "regime_prior_cold_start", "observations": 0}
        learned = self.engine.learn_rolling_ic_ir_factor_weights(history["factors"], history["forward_returns"])
        observations = int(learned.get("observations", 0))
        weights = self.engine.blend_with_prior(
            np.asarray(learned["weights"], dtype=float), prior, observations
        )
        return weights, {
            "mode": learned.get("mode", mode),
            "observations": observations,
            "ic_mean": np.round(np.asarray(learned.get("ic_mean", np.zeros(len(PILLARS))), dtype=float), 4).tolist(),
            "learned_weights": np.round(np.asarray(learned["weights"], dtype=float), 4).tolist(),
        }

    def _historical_ic_panels(self, market: str, *, max_dates: int = 120) -> dict[str, list[Any]]:
        scores = self.lake.read("factor_scores")
        out: dict[str, list[Any]] = {"factors": [], "forward_returns": []}
        if scores.empty:
            return out
        frame = scores[scores["market"].astype(str) == market].copy()
        if frame.empty:
            return out
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame["date"] < self.as_of]
        horizon = int(self.config.e("factor_learning.horizon_days", 5))
        close_wide = self.close_wide(market)
        if close_wide.empty:
            return out
        forward = close_wide.shift(-horizon) / close_wide - 1.0
        dates = sorted(frame["date"].unique())[-max_dates:]
        for date in dates:
            cross = frame[frame["date"] == date].set_index("ticker")
            if len(cross) < 10:
                continue
            raw = cross[list(PILLARS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            ortho = lowdin_orthogonalize(
                raw,
                winsor_quantile=float(self.config.e("lowdin.winsor_quantile", 0.02)),
                eigenvalue_floor=float(self.config.e("lowdin.eigenvalue_floor", 1e-4)),
                min_observations=int(self.config.e("lowdin.min_observations", 8)),
            )
            realized = forward.asof(date).reindex(ortho.index)
            if realized.isna().all():
                continue
            out["factors"].append(ortho)
            out["forward_returns"].append(realized.fillna(0.0))
        return out

    def fuse(
        self,
        market: str,
        factors: pd.DataFrame,
        regime: RegimeAllocation,
        *,
        season_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
        eligible = factors["eligible"].astype(bool) if "eligible" in factors.columns else pd.Series(True, index=factors.index)
        raw = factors[list(PILLARS)].loc[eligible]
        if raw.empty:
            return pd.Series(dtype=float), pd.DataFrame(), {"reason": "no_eligible_names"}
        ortho = lowdin_orthogonalize(
            raw,
            winsor_quantile=float(self.config.e("lowdin.winsor_quantile", 0.02)),
            eigenvalue_floor=float(self.config.e("lowdin.eigenvalue_floor", 1e-4)),
            min_observations=int(self.config.e("lowdin.min_observations", 8)),
        )
        weights, weight_diagnostics = self.pillar_weights(market, regime, factors)
        alpha = pd.Series(ortho.to_numpy(dtype=float) @ np.asarray(weights, dtype=float), index=ortho.index, dtype=float)
        alpha = zscore(alpha, 0.0)
        diagnostics = {
            "pillar_weights": np.round(np.asarray(weights, dtype=float), 4).tolist(),
            "orthogonality_error": orthogonality_error(ortho),
            "ortho_max_abs_correlation": max_abs_correlation(ortho),
            **weight_diagnostics,
            **(season_diagnostics or {}),
        }
        for pillar, weight in zip(PILLARS, diagnostics["pillar_weights"]):
            self.calibration_rows.append(
                {
                    "as_of_date": self.as_of,
                    "scope": f"{market}:factor_weight",
                    "parameter": pillar,
                    "method": str(weight_diagnostics.get("mode", "ic_ir")),
                    "value": float(weight),
                    "sample_size": int(weight_diagnostics.get("observations", 0) or 0),
                }
            )
        return alpha, ortho, diagnostics

    # ==================================================================== allocate
    def allocate(
        self,
        market: str,
        alpha: pd.Series,
        returns: pd.DataFrame,
        prices: pd.Series,
        cash_buffer: float,
        current_weights: dict[str, float],
        sectors: pd.Series,
        equity: float,
    ) -> tuple[pd.Series, list[dict[str, Any]], dict[str, Any]]:
        if alpha.empty:
            return pd.Series(dtype=float), [], {"reason": "no_alpha"}
        subset = [ticker for ticker in alpha.index if ticker in returns.columns]
        if len(subset) < 3:
            return pd.Series(dtype=float), [], {"reason": "insufficient_eligible_prices"}
        returns_panel = returns[subset].dropna(how="all").fillna(0.0).tail(160)
        target = self.optimizer.optimize(
            returns_panel,
            alpha.loc[subset].reindex(returns_panel.columns),
            cash_buffer=cash_buffer,
            current_weights=current_weights,
        )

        market_returns = returns_panel.mean(axis=1)
        risk_aversion = self.engine.market_risk_aversion(market_returns)
        price_frame = self.price_snapshot(market)
        deadbands: dict[str, float] = {}
        costs_bps: dict[str, float] = {}
        for ticker in target.index:
            history = price_frame[price_frame["ticker"] == ticker].tail(60)
            estimate = self.engine.estimate_transaction_cost(
                market=market,
                high=history["high"].to_numpy(dtype=float) if not history.empty else None,
                low=history["low"].to_numpy(dtype=float) if not history.empty else None,
                close=history["adj_close"].to_numpy(dtype=float) if not history.empty else None,
                order_value=float(target.get(ticker, 0.0)) * equity,
                adv=float(history["dollar_volume"].tail(20).mean()) if not history.empty and "dollar_volume" in history else float("nan"),
                market_returns=market_returns,
            )
            deadbands[ticker] = estimate.deadband
            costs_bps[ticker] = estimate.total * 10_000.0

        tickets = self.optimizer.generate_orders(
            market=market,
            target_weights=target,
            current_weights=current_weights,
            prices=prices,
            deadbands=deadbands,
            costs_bps=costs_bps,
            alpha=alpha,
            sectors=sectors.reindex(target.index).fillna("UNKNOWN").to_dict(),
            equity=equity,
        )
        diagnostics = {
            "target_names": int((target > 0).sum()),
            "invested_weight": float(target.sum()),
            "cash_buffer": float(cash_buffer),
            "market_risk_aversion": float(risk_aversion),
            "median_deadband_pct": float(np.median(list(deadbands.values())) * 100.0) if deadbands else 0.0,
            "median_cost_bps": float(np.median(list(costs_bps.values()))) if costs_bps else 0.0,
            "orders": len(tickets),
        }
        return target, [ticket.to_dict() for ticket in tickets], diagnostics

    # ========================================================================= run
    def run_market(self, market: str, macro_matrix: pd.DataFrame) -> MarketRun:
        regime = self.regime_detector.detect(macro_matrix)
        factors, diagnostics = self.build_factor_matrix(market, regime, macro_matrix)
        if factors.empty:
            return MarketRun(market, regime, 0, 0, pd.Series(dtype=float), pd.Series(dtype=float), [], diagnostics)

        alpha, ortho, fusion_diagnostics = self.fuse(market, factors, regime)
        diagnostics.update(fusion_diagnostics)
        diagnostics["regime"] = regime.regime_name
        diagnostics["turbulence_d2"] = float(regime.turbulence_d2)
        diagnostics["cash_buffer"] = float(regime.cash_buffer)

        ledger = Ledger(self.state_dir, market, starting_cash=self.starting_cash.get(market, 100_000.0))
        snapshot = ledger.load()
        prices = factors["price"] if "price" in factors.columns else pd.Series(dtype=float)
        snapshot = ledger.mark_prices(snapshot, prices.to_dict())
        snapshot["as_of"] = self.as_of.date().isoformat()
        ledger.save(snapshot)
        equity = ledger.equity(snapshot)
        current_weights = ledger.current_weights(snapshot)
        returns = self.build_returns(market)

        target, tickets, allocation_diagnostics = self.allocate(
            market,
            alpha,
            returns,
            prices,
            regime.cash_buffer,
            current_weights,
            factors["gics_sector"] if "gics_sector" in factors.columns else pd.Series("UNKNOWN", index=factors.index),
            equity,
        )
        diagnostics.update(allocation_diagnostics)

        scores_table = self.factor_score_table(market, ortho, alpha, regime)
        if not scores_table.empty:
            self.lake.write("factor_scores", scores_table)

        if target is not None and not target.empty:
            targets_table = pd.DataFrame(
                {
                    "date": self.as_of,
                    "market": market,
                    "ticker": target.index,
                    "target_weight": target.to_numpy(dtype=float),
                    "current_weight": [float(current_weights.get(ticker, 0.0)) for ticker in target.index],
                    "alpha": [float(alpha.get(ticker, 0.0)) for ticker in target.index],
                    "price": [float(prices.get(ticker, np.nan)) for ticker in target.index],
                    "order_shares": [
                        float(next((ticket["shares"] for ticket in tickets if ticket["ticker"] == ticker), 0.0))
                        for ticker in target.index
                    ],
                    "order_value": [
                        float(next((ticket["order_value"] for ticket in tickets if ticket["ticker"] == ticker), 0.0))
                        for ticker in target.index
                    ],
                    "estimated_cost_bps": [
                        float(next((ticket["est_cost_bps"] for ticket in tickets if ticket["ticker"] == ticker), 0.0))
                        for ticker in target.index
                    ],
                    "deadband": [
                        float(next((ticket["kkt_deadband_pct"] for ticket in tickets if ticket["ticker"] == ticker), 0.0)) / 100.0
                        for ticker in target.index
                    ],
                    "binding_constraint": ["deadband" if not any(t["ticker"] == ticker for t in tickets) else "trade" for ticker in target.index],
                }
            )
            self.lake.write("portfolio_targets", targets_table)

        if tickets:
            ledger.write_pending_orders(tickets, run_id=self.run_id)
            ledger.record_event("recommendation", {"orders": len(tickets), "regime": regime.regime_name}, run_id=self.run_id)
        self._append_nav(market, ledger, equity)

        eligible = int(factors["eligible"].sum()) if "eligible" in factors.columns else 0
        return MarketRun(market, regime, int(len(factors)), eligible, target, alpha, tickets, diagnostics)

    def factor_score_table(self, market: str, ortho: pd.DataFrame, alpha: pd.Series, regime: RegimeAllocation) -> pd.DataFrame:
        if ortho.empty:
            return pd.DataFrame()
        table = ortho.copy()
        table["market"] = market
        table["composite_alpha"] = alpha.reindex(table.index)
        table["regime"] = regime.regime_name
        table["turbulence_d2"] = float(regime.turbulence_d2)
        table["cash_buffer"] = float(regime.cash_buffer)
        table = table.reset_index().rename(columns={"index": "ticker"})
        if "ticker" not in table.columns:
            table = table.rename(columns={table.columns[0]: "ticker"})
        table["date"] = self.as_of
        return table

    def _append_nav(self, market: str, ledger: Ledger, equity: float) -> None:
        snapshot = ledger.load()
        nav_path = self.state_dir / "nav_history.jsonl"
        row = {
            "date": self.as_of.date().isoformat(),
            "market": market,
            "equity": round(float(equity), 2),
            "cash": round(float(snapshot.get("cash", 0.0)), 2),
            "positions": len(snapshot.get("positions", {})),
            "run_id": self.run_id,
        }
        with nav_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    # =================================================================== pipeline
    def lowdin_orthogonalize_matrix(self, factor_frame: pd.DataFrame) -> pd.DataFrame:
        """Thin wrapper used by the walk-forward backtester (raw pillar matrix -> Löwdin)."""
        return lowdin_orthogonalize(
            factor_frame,
            winsor_quantile=float(self.config.e("lowdin.winsor_quantile", 0.02)),
            eigenvalue_floor=float(self.config.e("lowdin.eigenvalue_floor", 1e-4)),
            min_observations=int(self.config.e("lowdin.min_observations", 8)),
        )

    def allocate_true_ward_hrp(self, returns: pd.DataFrame, alpha: pd.Series, cash_buffer: float) -> pd.Series:
        """True Ward-linkage HRP allocation (used by the backtester and the live path)."""
        return self.optimizer.optimize(returns, alpha, cash_buffer=cash_buffer)

    def run(self, markets: Sequence[str], mode: str = "daily_eod") -> dict[str, Any]:
        started = dt.datetime.now(dt.timezone.utc)
        ingested = self.ingest_shards()
        macro = self.macro_pit()
        macro_matrix = self.regime_detector.build_macro_matrix(macro)
        results: dict[str, Any] = {}
        runs: list[dict[str, Any]] = []
        for market in markets:
            run = self.run_market(market, macro_matrix)
            results[market] = run.diagnostics
            # JSON-serializable run records: the CLI prints this summary directly
            # and the issue builder / tests consume ``run["tickets"]``.
            runs.append(
                {
                    "market": run.market,
                    "regime": run.regime.regime_name,
                    "universe_size": run.universe_size,
                    "eligible_size": run.eligible_size,
                    "targets": _jsonable(run.target_weights.round(6).to_dict()),
                    "alpha": _jsonable(run.alpha.round(6).to_dict()),
                    "tickets": _jsonable(run.tickets),
                    "diagnostics": _jsonable(run.diagnostics),
                }
            )

        if self.calibration_rows:
            self.lake.write("calibration_state", pd.DataFrame(self.calibration_rows))

        finished = dt.datetime.now(dt.timezone.utc)
        manifest = manifest_row(
            self.run_id,
            mode,
            ",".join(markets),
            started,
            finished,
            self.config.fingerprint(),
            "ok",
            {"ingested": ingested, "diagnostics": {market: _jsonable(diag) for market, diag in results.items()}},
        )
        self.lake.write("run_manifest", manifest)

        return {
            "run_id": self.run_id,
            "as_of": self.as_of.date().isoformat(),
            "mode": mode,
            "ingested": ingested,
            "markets": results,
            "runs": runs,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "lakehouse": self.lake.verify(),
        }


def _jsonable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(item) for item in payload]
    if isinstance(payload, (np.floating, np.integer)):
        return float(payload)
    if isinstance(payload, (pd.Timestamp, dt.datetime, dt.date)):
        return str(payload)
    return payload


# ============================================================================ CLI
def _resolve_markets(config: Config, target: str, mode: str) -> list[str]:
    """Accept ``BOTH``/``ALL``, a single market, or the expanded ``US,INDIA`` form
    emitted by the Stage-1 exchange-calendar gate."""
    normalized = target.upper().strip()
    if normalized in ("BOTH", "ALL", ""):
        return config.markets()
    if "," in normalized:
        return [part.strip() for part in normalized.split(",") if part.strip()]
    return [normalized]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V4 autonomous dual-market portfolio reducer")
    parser.add_argument("--mode", default="auto", choices=["auto", "daily_eod", "weekly_deep_sync"])
    parser.add_argument("--active-market", "--target-market", dest="target_market", default="BOTH")
    parser.add_argument("--data-dir", type=Path, default=Path("data_lake"))
    parser.add_argument("--shards-dir", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=Path("state"))
    parser.add_argument("--output-issue", type=Path, default=None)
    parser.add_argument("--dashboard-html", type=Path, default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--backend", default="auto", choices=["auto", "onnx", "hashing"])
    parser.add_argument("--synthetic", action="store_true", help="Generate a deterministic offline feature store")
    parser.add_argument("--synthetic-days", type=int, default=320)
    parser.add_argument("--synthetic-tickers", type=int, default=40)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    config = Config.load()
    mode = args.mode
    if mode == "auto":
        schedule = os.getenv("GITHUB_EVENT_SCHEDULE") or os.getenv("PMS_SCHEDULE") or ""
        weekday = dt.datetime.now(dt.timezone.utc).weekday()
        mode = "weekly_deep_sync" if (schedule.strip().startswith("0 22") or weekday == 6) else "daily_eod"

    shards_dir = args.shards_dir
    if args.synthetic:
        from .synthetic import generate_synthetic_lakehouse

        shards_dir = generate_synthetic_lakehouse(
            args.data_dir,
            config,
            days=args.synthetic_days,
            n_tickers_per_market=args.synthetic_tickers,
        )
        mode = "synthetic_demo"
        args.offline = True

    markets = _resolve_markets(config, args.target_market, mode)
    reducer = V4AutonomousPortfolioReducer(
        config,
        args.data_dir,
        args.state_dir,
        shards_dir=shards_dir,
        as_of=dt.date.fromisoformat(args.as_of) if args.as_of else None,
        offline=args.offline,
        embed_backend=args.backend,
    )
    summary = reducer.run(markets, mode=mode)

    if args.output_issue:
        from .reporting.issue_builder import build_issue_body

        build_issue_body(summary, config, args.output_issue, lake=reducer.lake, state_dir=args.state_dir)
    if args.dashboard_html:
        from .reporting.dashboard_builder import build_dashboard_html

        build_dashboard_html(args.state_dir, args.dashboard_html)

    if args.print_json:
        print(json.dumps(_jsonable(summary), indent=2))
    else:
        print(
            f"[{summary['run_id']}] mode={summary['mode']} markets={list(summary['markets'])} "
            f"lakehouse_ok={summary['lakehouse']['ok']}"
        )
        for market, diagnostics in summary["markets"].items():
            print(
                f"  {market}: regime={diagnostics.get('regime')} "
                f"d2={float(diagnostics.get('turbulence_d2', float('nan')) or float('nan')):.2f} "
                f"cash={float(diagnostics.get('cash_buffer', 0.0) or 0.0):.1%} "
                f"eligible={diagnostics.get('eligible')} orders={diagnostics.get('orders')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
