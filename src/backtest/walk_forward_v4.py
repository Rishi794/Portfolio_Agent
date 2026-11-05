"""Point-in-time walk-forward evaluation using the shared V5 feature and allocation path.

The historical module name is retained for existing commands. The backtest now calls
the same reducer feature builder, Flex replay, Timestral forecaster, HRP allocator, and
transaction-cost estimator as the live pipeline. Historical golden-copy quorum evidence
is not available for the full OHLCV history, so the report states that limitation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..filtering.statistics import zscore
from ..flex_model.aladdin_features import flex as aladdin_flex
from ..flex_model.runtime import FlexModelReplayRuntime
from ..main_v4 import V4AutonomousPortfolioReducer


class WalkForwardBacktesterV4:
    """Walk-forward strategy replay over the V5 reducer's point-in-time features."""

    def __init__(self, config: Config, lake_dir: str | Path, state_dir: str | Path) -> None:
        self.config = config
        self.lake_dir = Path(lake_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.reducer = V4AutonomousPortfolioReducer(config, self.lake_dir, self.state_dir)
        # Feature generation may estimate US→India spillover, but a backtest must not
        # mutate the production lakehouse while replaying historical dates.
        self.reducer.persist_derived_features = False
        self._flex_temp = tempfile.TemporaryDirectory(prefix="pms-walk-forward-flex-")
        if bool(config.get("aladdin.flex_model.enabled", True)):
            self.reducer.flex_runtime = FlexModelReplayRuntime(self._flex_temp.name, aladdin_flex)

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        rendered = json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _profile_suffix(profile: str) -> str:
        return "" if profile == "full" else f"_{profile}"

    @classmethod
    def _nav_filename(cls, profile: str) -> str:
        if profile == "full":
            return "walk_forward_nav_history.jsonl"
        return f"walk_forward_nav{cls._profile_suffix(profile)}.jsonl"

    def _replace_market_nav(
        self,
        market: str,
        records: list[dict[str, Any]],
        *,
        profile: str,
    ) -> None:
        """Replace this profile/market curve without touching live or other NAV."""
        path = self.state_dir / self._nav_filename(profile)
        existing: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and str(row.get("market")) != market:
                    existing.append(row)
        combined = existing + records
        combined.sort(key=lambda row: (str(row.get("market", "")), str(row.get("date", ""))))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in combined),
            encoding="utf-8",
        )
        temporary.replace(path)

    def run_walk_forward(
        self,
        market: str,
        rebalance_step_days: int = 5,
        warmup: int = 60,
        test_sessions: int | None = None,
        profile: str = "full",
    ) -> dict[str, Any]:
        if rebalance_step_days < 1:
            return {"market": market, "error": "rebalance_step_days must be positive"}
        if warmup < 1:
            return {"market": market, "error": "warmup must be positive"}
        if profile not in {"full", "daily"}:
            return {"market": market, "error": "profile must be 'full' or 'daily'"}
        if test_sessions is not None and test_sessions < 1:
            return {"market": market, "error": "test_sessions must be positive"}

        # Start each market from the same full available history, even when this
        # object already replayed another market and left its reducer as-of in the past.
        self.reducer.as_of = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        prices, _ = self.reducer.query_pit_split_adjusted_universe(market)
        if prices.empty:
            return {"market": market, "error": "No historical OHLCV data in the lakehouse"}

        prices = prices.copy()
        prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="coerce")
        if isinstance(prices["trade_date"].dtype, pd.DatetimeTZDtype):
            prices["trade_date"] = prices["trade_date"].dt.tz_localize(None)
        prices = prices.dropna(subset=["trade_date"])
        all_dates = sorted(pd.Timestamp(value).normalize() for value in prices["trade_date"].unique())
        if len(all_dates) < warmup + 20:
            return {
                "market": market,
                "error": f"Need at least {warmup + 20} trading sessions; found {len(all_dates)}",
            }
        test_start_idx = max(warmup, len(all_dates) - test_sessions - 1) if test_sessions else warmup

        close = prices.pivot_table(
            index="trade_date", columns="ticker", values="adj_close", aggfunc="last"
        ).sort_index()
        returns = close.pct_change(fill_method=None)
        lookback_days = int(self.config.e("optimizer.cov_lookback_days", 160))
        canonical_pillars = list(self.reducer.pillar_names())
        current_weights = pd.Series(0.0, index=close.columns, dtype=float)
        starting_cash = float(self.reducer.starting_cash.get(market, 100_000.0))
        if not np.isfinite(starting_cash) or starting_cash <= 0.0:
            return {"market": market, "error": "starting cash must be positive and finite"}

        portfolio_nav = 1.0
        benchmark_nav = 1.0
        nav_records: list[dict[str, Any]] = []
        ortho_snapshots: list[pd.DataFrame] = []
        forward_snapshots: list[pd.Series] = []
        total_turnover = 0.0
        total_fee_value = 0.0
        feature_modes: dict[str, set[str]] = {"timestral": set(), "flex": set()}
        min_ic_observations = int(self.config.e("factor_learning.min_ic_observations", 5))

        for idx in range(test_start_idx, len(all_dates) - 1):
            stamp = all_dates[idx]
            next_date = all_dates[idx + 1]
            step_fee = 0.0

            # Rebalance on the first test session, then maintain the requested
            # cadence. A rolling daily window can start on any calendar index.
            if (idx - test_start_idx) % rebalance_step_days == 0:
                self.reducer.as_of = stamp
                # Feedback is date-dependent. Clear the reducer cache so every
                # simulated rebalance sees only labels that have matured by this date.
                self.reducer.feedback_cache.pop(market, None)
                macro_matrix = self.reducer.regime_detector.build_macro_matrix(self.reducer.macro_pit())
                regime = self.reducer.regime_detector.detect(macro_matrix)
                factors, feature_diagnostics = self.reducer.build_factor_matrix(
                    market, regime, macro_matrix
                )
                timestral = (feature_diagnostics.get("timestral") or {}).get("mode")
                if timestral:
                    feature_modes["timestral"].add(str(timestral))
                flex = feature_diagnostics.get("flex_model") or {}
                if flex:
                    feature_modes["flex"].add("enabled")
                else:
                    feature_modes["flex"].add("disabled")

                eligible = (
                    factors["eligible"].reindex(close.columns).fillna(False).astype(bool)
                    if not factors.empty and "eligible" in factors.columns
                    else pd.Series(False, index=close.columns)
                )
                raw_factors = (
                    factors.reindex(close.columns)
                    .reindex(columns=canonical_pillars, fill_value=0.0)
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                ortho = pd.DataFrame()
                alpha = pd.Series(dtype=float)
                if int(eligible.sum()) >= 5 and float(raw_factors.loc[eligible].abs().to_numpy().sum()) > 0.0:
                    ortho = self.reducer.lowdin_orthogonalize_matrix(raw_factors.loc[eligible])
                    prior = self.reducer._pillar_prior(regime, canonical_pillars)
                    if len(ortho_snapshots) >= min_ic_observations:
                        learned = self.reducer.engine.learn_rolling_ic_ir_factor_weights(
                            ortho_snapshots[-40:], forward_snapshots[-40:]
                        )
                        weights = self.reducer.engine.blend_with_prior(
                            np.asarray(learned["weights"], dtype=float),
                            prior,
                            int(learned.get("observations", 0)),
                        )
                    else:
                        weights = prior
                    alpha = pd.Series(
                        ortho.to_numpy(dtype=float) @ weights,
                        index=ortho.index,
                        dtype=float,
                    )
                    alpha = zscore(alpha, 0.0)

                # The label for this snapshot ends no later than the next scheduled
                # rebalance, so it cannot affect the signal that generated the label.
                fwd_end = min(idx + rebalance_step_days, len(all_dates) - 1)
                if not ortho.empty:
                    realized = close.iloc[fwd_end].div(close.iloc[idx]).sub(1.0).reindex(ortho.index)
                    ortho_snapshots.append(ortho)
                    forward_snapshots.append(realized.replace([np.inf, -np.inf], np.nan).fillna(0.0))

                hist_returns = returns.iloc[: idx + 1].tail(lookback_days)
                prices_asof = (
                    factors["price"].reindex(close.columns)
                    if not factors.empty and "price" in factors.columns
                    else pd.Series(dtype=float)
                )
                sectors = (
                    factors["gics_sector"].reindex(close.columns).fillna("UNKNOWN")
                    if not factors.empty and "gics_sector" in factors.columns
                    else pd.Series("UNKNOWN", index=close.columns, dtype="object")
                )
                if not alpha.empty:
                    alpha, _shrunk = self.reducer.apply_matured_stock_reliability(market, alpha)
                    equity = portfolio_nav * starting_cash
                    target, tickets, _allocation_diagnostics = self.reducer.allocate(
                        market,
                        alpha,
                        hist_returns,
                        prices_asof,
                        regime.cash_buffer,
                        current_weights.to_dict(),
                        sectors,
                        equity,
                    )
                else:
                    target = pd.Series(dtype=float)
                    tickets = []

                # Execute only orders the shared live allocator would emit (deadband,
                # minimum notional, statutory friction, half spread, impact, turnover).
                # Historical golden-copy quorum data does not cover the full OHLCV
                # history, so the V5 pre-trade verification gate is reported separately.
                rebalanced = current_weights.copy()
                equity = max(portfolio_nav * starting_cash, 1e-12)
                step_turnover = 0.0
                fee_value = 0.0
                for order in tickets:
                    ticker = str(order.get("ticker", ""))
                    side = 1.0 if str(order.get("action", "")).upper() == "BUY" else -1.0
                    notional = float(order.get("order_value") or 0.0)
                    cost_bps = max(float(order.get("est_cost_bps") or 0.0), 0.0)
                    if not ticker or not np.isfinite(notional) or notional <= 0.0:
                        continue
                    executed_weight = notional / equity
                    rebalanced.loc[ticker] = max(
                        float(rebalanced.get(ticker, 0.0)) + side * executed_weight,
                        0.0,
                    )
                    step_turnover += executed_weight
                    fee_value += notional * cost_bps / 10_000.0
                step_fee = min(max(fee_value / equity, 0.0), 0.99)
                total_turnover += step_turnover
                total_fee_value += fee_value
                if step_fee > 0.0:
                    rebalanced = rebalanced / (1.0 - step_fee)
                current_weights = rebalanced
                portfolio_nav *= 1.0 - step_fee

            day_return = returns.iloc[idx + 1].reindex(current_weights.index).fillna(0.0)
            gross_port_return = float((current_weights * day_return).sum())
            gross_port_return = max(gross_port_return, -0.999999)
            benchmark_return = (
                float(returns.iloc[idx + 1].mean())
                if returns.iloc[idx + 1].notna().any()
                else 0.0
            )
            nav_before_day = portfolio_nav
            portfolio_nav *= 1.0 + gross_port_return
            benchmark_nav *= 1.0 + benchmark_return
            net_port_return = portfolio_nav / max(nav_before_day / max(1.0 - step_fee, 1e-12), 1e-12) - 1.0
            # Drift holdings by realized returns; this is necessary for the next
            # rebalance's turnover and fees to reflect the actual post-return book.
            if 1.0 + gross_port_return > 1e-12:
                current_weights = current_weights.mul(1.0 + day_return).div(1.0 + gross_port_return)
            nav_records.append(
                {
                    "date": pd.Timestamp(next_date).strftime("%Y-%m-%d"),
                    "market": market,
                    "profile": profile,
                    "strategy": "v5_shared_reducer_signals",
                    "portfolio_nav": round(portfolio_nav, 8),
                    "benchmark_nav": round(benchmark_nav, 8),
                    "daily_return": round(net_port_return, 8),
                    "gross_daily_return": round(gross_port_return, 8),
                    "transaction_cost_pct": round(step_fee * 100.0, 8),
                    "invested_weight": round(float(current_weights.sum()), 6),
                }
            )

        if not nav_records:
            return {"market": market, "error": "No sessions simulated"}

        self._replace_market_nav(market, nav_records, profile=profile)
        frame = pd.DataFrame(nav_records)
        net_returns = frame["daily_return"].to_numpy(dtype=float)
        annual_vol = float(np.std(net_returns, ddof=0) * np.sqrt(252.0))
        downside_deviation = float(np.sqrt(np.mean(np.minimum(net_returns, 0.0) ** 2) * 252.0))
        annual_arithmetic_return = float(np.mean(net_returns) * 252.0)
        first_test_date = pd.Timestamp(nav_records[0]["date"])
        last_test_date = pd.Timestamp(nav_records[-1]["date"])
        elapsed_days = max((last_test_date - first_test_date).days, 1)
        cagr = float(portfolio_nav ** (365.25 / elapsed_days) - 1.0)
        benchmark_cagr = float(benchmark_nav ** (365.25 / elapsed_days) - 1.0)
        nav = np.concatenate(([1.0], frame["portfolio_nav"].to_numpy(dtype=float)))
        drawdowns = (nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)
        summary = {
            "market": market,
            "profile": profile,
            "strategy": "v5_shared_reducer_signals",
            "signal_path": "V5 reducer factor builder (Flex replay + Timestral when enabled)",
            "execution_model": (
                "shared live HRP allocator and estimated transaction costs; "
                "historical golden-copy gate unavailable"
            ),
            "sessions_tested": int(len(frame)),
            "test_sessions_requested": test_sessions,
            "first_test_date": first_test_date.strftime("%Y-%m-%d"),
            "last_test_date": last_test_date.strftime("%Y-%m-%d"),
            "final_portfolio_nav": round(portfolio_nav, 6),
            "final_benchmark_nav": round(benchmark_nav, 6),
            "total_return_pct": round((portfolio_nav - 1.0) * 100.0, 2),
            "benchmark_total_return_pct": round((benchmark_nav - 1.0) * 100.0, 2),
            "cagr_pct": round(cagr * 100.0, 2),
            "benchmark_cagr_pct": round(benchmark_cagr * 100.0, 2),
            "annual_vol_pct": round(annual_vol * 100.0, 2),
            "sharpe_ratio": round(annual_arithmetic_return / max(annual_vol, 1e-12), 3),
            "sortino_ratio": round(annual_arithmetic_return / max(downside_deviation, 1e-12), 3),
            "max_drawdown_pct": round(float(np.min(drawdowns)) * 100.0, 2),
            "total_turnover_x": round(total_turnover, 3),
            "total_fees_pct_of_initial_nav": round(total_fee_value / starting_cash * 100.0, 4),
            "feature_modes": {key: sorted(value) for key, value in feature_modes.items()},
        }
        suffix = self._profile_suffix(profile)
        self._write_json_atomic(
            self.state_dir / f"walk_forward_summary{suffix}_{market.lower()}.json",
            summary,
        )
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V5 point-in-time walk-forward backtester")
    parser.add_argument("--lake-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--market", default="BOTH", choices=["INDIA", "US", "BOTH"])
    parser.add_argument("--rebalance-step-days", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--test-sessions", type=int, default=None)
    parser.add_argument("--profile", choices=["daily", "full"], default="full")
    args = parser.parse_args(argv)

    config = Config.load()
    backtester = WalkForwardBacktesterV4(config, args.lake_dir, args.state_dir)
    markets = ["INDIA", "US"] if args.market == "BOTH" else [args.market]
    summaries = {
        market: backtester.run_walk_forward(
            market,
            rebalance_step_days=args.rebalance_step_days,
            warmup=args.warmup,
            test_sessions=args.test_sessions,
            profile=args.profile,
        )
        for market in markets
    }
    report = {
        "strategy": "v5_shared_reducer_signals",
        "markets": summaries,
    }
    suffix = backtester._profile_suffix(args.profile)
    backtester._write_json_atomic(
        backtester.state_dir / f"walk_forward_summary{suffix}.json",
        report,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if any("error" in value for value in summaries.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
