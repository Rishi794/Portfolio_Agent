"""Point-in-Time walk-forward backtester for US (NYSE/NASDAQ) and India (NSE/BSE).

Leakage discipline
------------------
On every rebalancing session ``t`` the engine slices **only** data stamped at or
before ``t``: prices/corporate actions up to ``t``, SEC/NSE filings with
``filing_timestamp <= t``, macro vintages published by ``t``, and news/patents
published by ``t``. The signal is then realized out-of-sample from ``t`` to
``t + Δ`` — features never see their own outcome.

Costs
-----
Corwin-Schultz high-low effective spread + statutory friction (Indian
STT/stamp-duty/exchange/SEBI + GST; US SEC §31 + FINRA TAF) + the endogenous KKT
no-trade band. A trade only happens if the drift exceeds the band.

Outputs ``state/nav_history.jsonl`` rows (``portfolio_nav``, ``benchmark_nav``,
``daily_return``) that the GitHub Pages dashboard plots directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..filtering.forensics import estimate_survival_gate, fundamental_quality_score, gate_mask
from ..filtering.lowdin_orthogonalizer import lowdin_orthogonalize
from ..filtering.self_calibrating_engine import SelfCalibratingQuantEngine
from ..main_v4 import PILLARS, V4AutonomousPortfolioReducer


class WalkForwardBacktesterV4:
    def __init__(self, config: Config, lake_dir: str | Path, state_dir: str | Path) -> None:
        self.config = config
        self.lake_dir = Path(lake_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.quant = SelfCalibratingQuantEngine(config)
        self.reducer = V4AutonomousPortfolioReducer(config, self.lake_dir, self.state_dir)

    # ------------------------------------------------------------------ helpers --
    def _fundamentals_history(self, market: str) -> pd.DataFrame:
        frame = self.reducer.lake.read("fundamentals_pit")
        if frame.empty:
            return frame
        frame = frame[frame["market"].astype(str) == market].copy()
        frame["filing_timestamp"] = pd.to_datetime(frame["filing_timestamp"], utc=True, errors="coerce")
        return frame.sort_values("filing_timestamp")

    def _tech_history(self, market: str) -> pd.DataFrame:
        frame = self.reducer.lake.read("tech_evolution_pit")
        if frame.empty:
            return frame
        frame = frame[frame["market"].astype(str) == market].copy()
        frame["knowledge_timestamp"] = pd.to_datetime(frame["knowledge_timestamp"], utc=True, errors="coerce")
        return frame.sort_values("knowledge_timestamp")

    def _events(self) -> pd.DataFrame:
        events = self.reducer.lake.read("policy_geopol_events")
        if events.empty:
            return events
        events = events.copy()
        events["published_timestamp"] = pd.to_datetime(events["published_timestamp"], utc=True, errors="coerce")
        return events.dropna(subset=["published_timestamp"]).sort_values("published_timestamp")

    def _snapshot_before(self, frame: pd.DataFrame, column: str, stamp: pd.Timestamp) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame[frame[column] <= stamp]

    # ------------------------------------------------------------------- signals --
    def _event_pillar(
        self,
        market: str,
        stamp: pd.Timestamp,
        events: pd.DataFrame,
        fundamentals_hist: pd.DataFrame,
        universe: list[str],
        domain_key: str,
    ) -> pd.Series:
        if events.empty or "embedding" not in events.columns:
            return pd.Series(0.0, index=universe, dtype=float)
        domains = (self.config.e("events.market_domains", {}) or {}).get(market, {}).get(domain_key, [])
        subset = events[
            events["domain"].astype(str).isin([str(domain) for domain in domains])
            & (events["published_timestamp"] <= stamp)
        ]
        if subset.empty:
            return pd.Series(0.0, index=universe, dtype=float)
        # company vectors known at `stamp` (first 10-K available at that time)
        known = self._snapshot_before(fundamentals_hist, "filing_timestamp", stamp)
        if known.empty:
            return pd.Series(0.0, index=universe, dtype=float)
        known = known.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")
        vectors: dict[str, np.ndarray] = {}
        for ticker in universe:
            if ticker not in known.index:
                continue
            embedding = known.loc[ticker, "business_embedding"] if "business_embedding" in known.columns else None
            if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
                continue
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.size == 0:
                continue
            vectors[ticker] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if not vectors:
            return pd.Series(0.0, index=universe, dtype=float)
        from ..filtering.embedding_projector import EventProjector, get_backend

        projector = EventProjector(self.config, backend=get_backend("hashing"))
        result = projector.project_to_tickers(
            pd.DataFrame(vectors).T,
            subset,
            as_of=stamp.to_pydatetime(),
        )
        return result.scores.reindex(universe).fillna(0.0)

    # --------------------------------------------------------------------- engine --
    def run_walk_forward(self, market: str, rebalance_step_days: int = 5, warmup: int = 60) -> dict[str, Any]:
        ohlcv_df, _ = self.reducer.query_pit_split_adjusted_universe(market)
        if ohlcv_df.empty:
            return {"market": market, "error": "No historical OHLCV data in the lakehouse"}

        lookback_days = int(self.config.e("optimizer.cov_lookback_days", 160))
        prices = ohlcv_df.copy()
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        all_dates = sorted(prices["trade_date"].unique())
        if len(all_dates) < warmup + 20:
            return {"market": market, "error": f"Need >= {warmup + 20} trading days; found {len(all_dates)}"}

        close = prices.pivot_table(index="trade_date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
        high = prices.pivot_table(index="trade_date", columns="ticker", values="adj_high", aggfunc="last").sort_index()
        low = prices.pivot_table(index="trade_date", columns="ticker", values="adj_low", aggfunc="last").sort_index()
        volume = prices.pivot_table(index="trade_date", columns="ticker", values="volume", aggfunc="last").sort_index()
        returns = close.pct_change(fill_method=None)

        macro_matrix = self.reducer.regime_detector.build_macro_matrix(self.reducer.macro_pit())
        fundamentals_hist = self._fundamentals_history(market)
        tech_hist = self._tech_history(market)
        events = self._events()

        current_weights = pd.Series(0.0, index=close.columns, dtype=float)
        portfolio_nav = 1.0
        benchmark_nav = 1.0
        nav_records: list[dict[str, Any]] = []
        ortho_snapshots: list[pd.DataFrame] = []
        forward_snapshots: list[pd.Series] = []
        total_turnover = 0.0
        total_fees = 0.0

        for idx in range(warmup, len(all_dates) - 1):
            stamp = pd.Timestamp(all_dates[idx], tz="UTC")
            next_date = all_dates[idx + 1]

            if (idx - warmup) % rebalance_step_days == 0:
                history = close.iloc[: idx + 1]
                hist_returns = returns.iloc[: idx + 1]
                universe = list(close.columns)

                # 1. Kalman trend (Myers-Tapley, learned Q/R)
                trend = self.quant.kalman_trend_scores(history, min_observations=20)

                # 2. PIT fundamentals: forensic gate + quality score
                known = self._snapshot_before(fundamentals_hist, "filing_timestamp", stamp)
                if not known.empty:
                    known = known.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")
                    known["dollar_volume_20d"] = pd.to_numeric(known.get("dollar_volume_20d"), errors="coerce")
                    known["gics_sector"] = known.get("gics_sector", pd.Series("UNKNOWN", index=known.index)).fillna("UNKNOWN")
                    thresholds = estimate_survival_gate(known, self.config)
                    mask = gate_mask(
                        known,
                        thresholds,
                        min_dollar_volume=float(self.config.e("forensics.min_dollar_volume", 15_000_000.0)),
                    )
                    quality = fundamental_quality_score(
                        known,
                        tuple(self.config.e("forensics.quality_weights", [0.45, 0.35, 0.20])),
                        winsor_quantile=float(self.config.e("forensics.winsor_quantile", 0.01)),
                    )
                else:
                    mask = pd.Series(True, index=universe)
                    quality = pd.Series(0.0, index=universe)

                # 3. Technology evolution (patents / productive R&D)
                tech_known = self._snapshot_before(tech_hist, "knowledge_timestamp", stamp)
                if not tech_known.empty:
                    tech_known = tech_known.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")
                    tech_scores = self.reducer.factor_tech_evolution(market, tech_known, pd.Index(universe))
                else:
                    tech_scores = pd.Series(0.0, index=universe, dtype=float)

                # 4./5. Policy & geopolitics via the PIT event projector
                policy = self._event_pillar(market, stamp, events, fundamentals_hist, universe, "gov_policy")
                geopol = self._event_pillar(market, stamp, events, fundamentals_hist, universe, "geopolitics")

                # 6. Macro / external beta (+ US->India spillover inside the reducer)
                # macro_matrix carries a tz-naive index (observation dates); `stamp`
                # is tz-aware UTC, so compare like with like.
                macro_asof = stamp.tz_localize(None) if macro_matrix.index.tz is None else stamp
                macro_shocks = macro_matrix[macro_matrix.index <= macro_asof]
                macro_beta = self.reducer.factor_macro_beta(
                    market, hist_returns.tail(lookback_days), macro_shocks
                )

                def _z(series: pd.Series) -> pd.Series:
                    from ..filtering.statistics import zscore as _zscore

                    return _zscore(series.reindex(universe).fillna(0.0), 0.0)

                raw_factors = pd.DataFrame(
                    {
                        "kalman_trend": _z(trend) if not trend.empty else 0.0,
                        "fundamentals": _z(quality),
                        "tech_evolution": _z(tech_scores),
                        "gov_policy": _z(policy),
                        "geopolitics": _z(geopol),
                        "macro_beta": _z(macro_beta),
                    }
                )
                eligible = mask.reindex(universe).fillna(True).astype(bool)
                raw_factors = raw_factors[list(PILLARS)]

                if raw_factors.sum().abs().sum() <= 0.0 or int(eligible.sum()) < 5:
                    # No signal at all: stay in cash for this leg rather than pretend.
                    ortho = pd.DataFrame()
                    alpha = pd.Series(dtype=float)
                    weights = np.zeros(len(PILLARS))
                else:
                    ortho = lowdin_orthogonalize(
                        raw_factors.loc[eligible],
                        winsor_quantile=float(self.config.e("lowdin.winsor_quantile", 0.02)),
                        eigenvalue_floor=float(self.config.e("lowdin.eigenvalue_floor", 1e-4)),
                        min_observations=int(self.config.e("lowdin.min_observations", 8)),
                    )
                    if len(ortho_snapshots) >= int(self.config.e("factor_learning.min_ic_observations", 5)):
                        learned = self.quant.learn_rolling_ic_ir_factor_weights(
                            ortho_snapshots[-40:], forward_snapshots[-40:]
                        )
                        weights = self.quant.blend_with_prior(
                            np.asarray(learned["weights"], dtype=float),
                            np.ones(len(PILLARS)) / len(PILLARS),
                            int(learned.get("observations", 0)),
                        )
                    else:
                        weights = np.ones(len(PILLARS)) / len(PILLARS)
                    alpha = pd.Series(ortho.to_numpy(dtype=float) @ weights, index=ortho.index, dtype=float)
                    alpha = _z(alpha)

                # store snapshot for the IC learning (forward return realized below)
                fwd_end = min(idx + rebalance_step_days, len(all_dates) - 1)
                realized_forward = (close.iloc[fwd_end] / close.iloc[idx] - 1.0).reindex(ortho.index if not ortho.empty else [])
                if not ortho.empty:
                    ortho_snapshots.append(ortho)
                    forward_snapshots.append(realized_forward.fillna(0.0))

                if alpha.empty:
                    target = pd.Series(dtype=float)
                else:
                    top = alpha[alpha > 0].sort_values(ascending=False).head(30).index.tolist()
                    market_returns_hist = hist_returns.mean(axis=1)
                    market_vol = float(market_returns_hist.tail(60).std() * np.sqrt(252))
                    market_sharpe = (
                        float(market_returns_hist.tail(60).mean() * 252 / max(market_vol, 1e-9))
                        if market_vol > 0
                        else 0.0
                    )
                    cash_buffer = float(
                        np.clip(
                            market_vol / 0.45,
                            float(self.config.e("turbulence.cash_min", 0.03)),
                            float(self.config.e("turbulence.cash_max", 0.45)),
                        )
                    )
                    if top:
                        target = self.reducer.allocate_true_ward_hrp(
                            hist_returns[top].tail(lookback_days).dropna(how="all").fillna(0.0),
                            alpha.loc[top],
                            cash_buffer,
                        )
                    else:
                        target = pd.Series(dtype=float)

                # cost-aware rebalancing: only trade outside the stock-specific deadband
                rebalanced = current_weights.copy()
                step_cost = 0.0
                for ticker in rebalanced.index:
                    current_w = float(current_weights.get(ticker, 0.0))
                    target_w = float(target.get(ticker, 0.0)) if not target.empty else 0.0
                    drift = target_w - current_w
                    if abs(drift) < 1e-6:
                        continue
                    window_slice = slice(max(0, idx - 60), idx + 1)
                    cost_fraction, deadband = self.quant.estimate_corwin_schultz_spread_and_deadband(
                        high=high[ticker].iloc[window_slice].to_numpy(dtype=float),
                        low=low[ticker].iloc[window_slice].to_numpy(dtype=float),
                        close=close[ticker].iloc[window_slice].to_numpy(dtype=float),
                        volume=volume[ticker].iloc[window_slice].to_numpy(dtype=float),
                        market=market,
                        market_sharpe=market_sharpe if "market_sharpe" in locals() else 0.0,
                        market_vol=market_vol if "market_vol" in locals() else 0.2,
                    )
                    if abs(drift) >= deadband or (target_w <= 1e-9 and current_w > 0):
                        rebalanced[ticker] = target_w
                        step_cost += abs(drift) * cost_fraction
                        total_turnover += abs(drift)
                step_fee = step_cost
                total_fees += step_fee
                current_weights = rebalanced
                portfolio_nav *= 1.0 - step_fee

            day_return = returns.iloc[idx + 1]
            port_day_return = float((current_weights * day_return.reindex(current_weights.index).fillna(0.0)).sum())
            bench_day_return = float(day_return.mean()) if day_return.notna().any() else 0.0
            portfolio_nav *= 1.0 + port_day_return
            benchmark_nav *= 1.0 + bench_day_return
            nav_records.append(
                {
                    "date": pd.Timestamp(next_date).strftime("%Y-%m-%d"),
                    "market": market,
                    "portfolio_nav": round(portfolio_nav, 6),
                    "benchmark_nav": round(benchmark_nav, 6),
                    "daily_return": round(port_day_return, 6),
                    "invested_weight": round(float(current_weights.sum()), 4),
                }
            )

        if not nav_records:
            return {"market": market, "error": "no sessions simulated"}

        frame = pd.DataFrame(nav_records)
        rets = frame["daily_return"].to_numpy(dtype=float)
        ann_return = float(np.mean(rets) * 252)
        ann_vol = float(np.std(rets) * np.sqrt(252)) + 1e-12
        downside = float(np.std(np.minimum(0.0, rets)) * np.sqrt(252)) + 1e-12
        nav = frame["portfolio_nav"].to_numpy(dtype=float)
        running_max = np.maximum.accumulate(nav)
        drawdowns = (nav - running_max) / np.maximum(running_max, 1e-12)

        with (self.state_dir / "nav_history.jsonl").open("a", encoding="utf-8") as handle:
            for record in nav_records:
                handle.write(json.dumps(record) + "\n")

        return {
            "market": market,
            "sessions_tested": int(len(frame)),
            "final_portfolio_nav": round(portfolio_nav, 4),
            "final_benchmark_nav": round(benchmark_nav, 4),
            "cagr_pct": round(ann_return * 100, 2),
            "annual_vol_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(ann_return / ann_vol, 3),
            "sortino_ratio": round(ann_return / downside, 3),
            "max_drawdown_pct": round(float(np.min(drawdowns)) * 100, 2),
            "total_turnover_x": round(total_turnover, 3),
            "total_fees_pct_of_nav": round(total_fees * 100, 4),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V4 PIT walk-forward backtester")
    parser.add_argument("--lake-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--market", default="BOTH", choices=["INDIA", "US", "BOTH"])
    parser.add_argument("--rebalance-step-days", type=int, default=5)
    args = parser.parse_args(argv)

    config = Config.load()
    backtester = WalkForwardBacktesterV4(config, args.lake_dir, args.state_dir)
    markets = ["INDIA", "US"] if args.market == "BOTH" else [args.market]
    for market in markets:
        summary = backtester.run_walk_forward(market, rebalance_step_days=args.rebalance_step_days)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
