"""Self-calibrating quant engine: every hyperparameter is estimated from data.

Implements, in closed form:

1. ``innovation_adaptive_kalman``      — Myers-Tapley / Mehra Q_t, R_t learning.
2. ``estimate_corwin_schultz_spread``  — effective bid-ask spread from daily H/L.
3. ``statutory_cost_fraction``         — exchange/regulator frictions (IN STT+GST, US SEC/TAF).
4. ``estimate_deadband``               — KKT no-trade band delta = c / (gamma * sigma^2).
5. ``project_us_to_india_spillover``   — US→India overnight lead-lag beta matrix.
6. ``learn_event_half_life``           — impulse-response MLE of decay lambda_d.
7. ``learn_rolling_ic_ir_factor_weights`` — Grinold-Kahn IC-IR pillar weights.
8. ``market_risk_aversion``            — realised gamma_hat = mu_mkt / sigma_mkt^2.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .kalman_smoother import AdaptiveKalmanFilter
from .statistics import shrunk_covariance

EPS = 1e-12


@dataclass
class TransactionCostEstimate:
    statutory: float
    half_spread: float
    impact: float
    total: float
    deadband: float
    stock_vol_daily: float
    market_risk_aversion: float


class SelfCalibratingQuantEngine:
    def __init__(self, config) -> None:
        self.config = config
        self.kalman = AdaptiveKalmanFilter.from_config(config)

    # ------------------------------------------------------------------ 1. Q/R --
    def innovation_adaptive_kalman(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        result = self.kalman.filter(prices)
        return (
            np.asarray(result["level"], dtype=float),
            np.asarray(result["velocity"], dtype=float),
            float(result["r_std"]),
        )

    def kalman_trend_scores(self, close_wide: pd.DataFrame, *, min_observations: int = 30) -> pd.Series:
        scores: dict[str, float] = {}
        for ticker in close_wide.columns:
            series = pd.to_numeric(close_wide[ticker], errors="coerce").dropna()
            if len(series) < min_observations:
                continue
            scores[ticker] = self.kalman.trend_score(series.to_numpy(dtype=float))
        return pd.Series(scores, dtype=float)

    # ------------------------------------------------------- 2. Corwin-Schultz --
    def estimate_corwin_schultz_spread(
        self,
        high: np.ndarray,
        low: np.ndarray,
        *,
        window: int | None = None,
        denom_floor: float | None = None,
    ) -> float:
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        mask = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0)
        h, l = h[mask], l[mask]
        if h.size < 5:
            return 0.0
        window = int(window or self.config.e("costs.corwin_schultz.window", 20))
        denom_floor = float(denom_floor or self.config.e("costs.corwin_schultz.denom_floor", 1e-8))

        h_t, l_t = h[1:], l[1:]
        h_p, l_p = h[:-1], l[:-1]
        beta = np.square(np.log(np.maximum(h_t / np.maximum(l_t, denom_floor), 1.0))) + np.square(
            np.log(np.maximum(h_p / np.maximum(l_p, denom_floor), 1.0))
        )
        gamma = np.square(
            np.log(np.maximum(np.maximum(h_t, h_p) / np.maximum(np.minimum(l_t, l_p), denom_floor), 1.0))
        )
        root2 = np.sqrt(2.0)
        denom = 3.0 - 2.0 * root2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(np.maximum(gamma / denom, 0.0))
        spread = np.maximum(0.0, 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha)))
        tail = spread[-window:]
        return float(np.mean(tail)) if tail.size else 0.0

    # --------------------------------------------------------- 3. statutory ---
    def statutory_cost_fraction(self, market: str) -> float:
        profile = self.config.cost_profile(market)
        if not profile:
            return 0.0
        if "stt_pct" in profile:  # India delivery: STT + stamp + txn + SEBI + GST on fees
            fees = float(profile.get("exchange_txn_pct", 0.0)) + float(profile.get("sebi_turnover_pct", 0.0))
            gst = fees * float(profile.get("gst_on_fees_pct", 0.0))
            return (
                float(profile.get("stt_pct", 0.0))
                + float(profile.get("stamp_duty_pct", 0.0))
                + fees
                + gst
            )
        total_bps = float(profile.get("sec_section_31_bps", 0.0)) + float(profile.get("finra_taf_bps", 0.0))
        return total_bps / 10_000.0

    # ------------------------------------------------------------ 4. deadband --
    @staticmethod
    def market_risk_aversion(market_returns: pd.Series, *, floor: float = 0.25) -> float:
        returns = pd.to_numeric(market_returns, errors="coerce").dropna()
        if len(returns) < 20:
            return float(floor)
        mu = float(returns.mean() * 252.0)
        var = float(returns.var(ddof=0) * 252.0)
        if var <= EPS:
            return float(floor)
        return float(max(abs(mu) / var, floor))

    def estimate_transaction_cost(
        self,
        *,
        market: str,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        close: np.ndarray | None = None,
        order_value: float = 0.0,
        adv: float = float("nan"),
        market_returns: pd.Series | None = None,
    ) -> TransactionCostEstimate:
        statutory = self.statutory_cost_fraction(market)
        spread = (
            self.estimate_corwin_schultz_spread(high, low)
            if high is not None and low is not None
            else 0.0
        )
        returns = np.diff(np.log(np.maximum(np.asarray(close, dtype=float), 1e-8))) if close is not None else np.array([])
        returns = returns[np.isfinite(returns)]
        stock_vol = float(np.std(returns[-60:])) if returns.size else 0.02

        coefficient = float(self.config.e("costs.impact.sqrt_law_coefficient", 1.0))
        participation = (order_value / adv) if (adv and np.isfinite(adv) and adv > 0) else 0.0
        impact = coefficient * stock_vol * np.sqrt(max(participation, 0.0))

        gamma = self.market_risk_aversion(
            market_returns if market_returns is not None else pd.Series(dtype=float),
            floor=float(self.config.e("costs.risk_aversion_floor", 0.25)),
        )
        total = statutory + 0.5 * spread + impact
        deadband = self.estimate_deadband(total, gamma, stock_vol)
        return TransactionCostEstimate(
            statutory=statutory,
            half_spread=0.5 * spread,
            impact=impact,
            total=total,
            deadband=deadband,
            stock_vol_daily=stock_vol,
            market_risk_aversion=gamma,
        )

    def estimate_deadband(self, cost_fraction: float, risk_aversion: float, daily_vol: float) -> float:
        """KKT no-trade band: delta = c / (gamma * sigma^2), annualised variance."""
        variance = max(daily_vol**2 * 252.0, 1e-6)
        scale = float(self.config.e("optimizer.deadband_scale", 1.0))
        return float(scale * cost_fraction / max(risk_aversion * variance, 1e-6))

    def estimate_corwin_schultz_spread_and_deadband(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        market: str,
        market_sharpe: float,
        market_vol: float,
    ) -> tuple[float, float]:
        """(total one-way cost fraction, KKT deadband) — convenience wrapper.

        ``market_sharpe`` / ``market_vol`` are accepted for API compatibility with the
        walk-forward backtester; the realised risk-aversion floor from config is used
        when the passed implied gamma is not finite.
        """
        del volume  # participation is supplied through ``order_value``/``adv`` in the live path
        implied_gamma = abs(float(market_sharpe)) / max(float(market_vol), 1e-6) if np.isfinite(market_sharpe) else float("nan")
        statutory = self.statutory_cost_fraction(market)
        spread = self.estimate_corwin_schultz_spread(high, low)
        returns = np.diff(np.log(np.maximum(np.asarray(close, dtype=float), 1e-8)))
        returns = returns[np.isfinite(returns)]
        stock_vol = float(np.std(returns[-60:])) if returns.size else 0.02
        total = statutory + 0.5 * spread
        gamma = implied_gamma if np.isfinite(implied_gamma) and implied_gamma > 0 else float(
            self.config.e("costs.risk_aversion_floor", 0.25)
        )
        return float(total), self.estimate_deadband(total, gamma, stock_vol)

    # ----------------------------------------------------------- 5. spillover --
    def project_us_to_india_spillover(
        self,
        india_returns: pd.DataFrame,
        us_factors: pd.DataFrame,
    ) -> pd.Series:
        aligned = pd.concat([india_returns, us_factors], axis=1, join="inner").dropna()
        n_india = india_returns.shape[1]
        if len(aligned) < int(self.config.e("spillover.min_observations", 30)) or n_india == 0:
            return pd.Series(0.0, index=india_returns.columns, dtype=float)
        joint = shrunk_covariance(aligned.to_numpy(dtype=float))
        cov_in_us = joint[:n_india, n_india:]
        cov_us_us = joint[n_india:, n_india:]
        beta = cov_in_us @ np.linalg.pinv(cov_us_us)
        shock = np.nan_to_num(us_factors.iloc[-1].to_numpy(dtype=float), nan=0.0)
        projected = beta @ shock
        return pd.Series(projected, index=india_returns.columns, dtype=float)

    def spillover_diagnostics(self, india_returns: pd.DataFrame, us_factors: pd.DataFrame) -> dict[str, float]:
        aligned = pd.concat([india_returns, us_factors], axis=1, join="inner").dropna()
        n_india = india_returns.shape[1]
        if len(aligned) < 30 or n_india == 0 or us_factors.shape[1] == 0:
            return {"observations": float(len(aligned)), "r_squared": float("nan")}
        joint = shrunk_covariance(aligned.to_numpy(dtype=float))
        cov_in_us = joint[:n_india, n_india:]
        cov_us_us = joint[n_india:, n_india:]
        explained = 0.0
        total = 0.0
        for i in range(n_india):
            beta = cov_in_us[i] @ np.linalg.pinv(cov_us_us)
            explained += float(beta @ cov_us_us @ beta)
            total += float(joint[i, i])
        return {
            "observations": float(len(aligned)),
            "r_squared": float(explained / total) if total > EPS else float("nan"),
        }

    # -------------------------------------------------------- 6. event decay ---
    def learn_event_half_life(
        self,
        events: pd.DataFrame,
        price_panel: pd.DataFrame,
        market_returns: pd.Series,
        *,
        domain: str,
        impact_map: dict[str, list[str]] | None = None,
        horizon: int | None = None,
    ) -> dict[str, float]:
        """Impulse-response MLE: y_{e,tau} = ln A - lambda * tau + eta => half-life."""
        horizon = int(horizon or self.config.e("events.impulse_horizon_days", 40))
        subset = events[events["domain"].astype(str).str.startswith(domain, na=False)]
        if subset.empty or price_panel.empty or len(price_panel) < horizon + 5:
            return {"half_life_days": float("nan"), "lambda": float("nan"), "events": 0.0}

        returns = price_panel.pct_change(fill_method=None)
        market = pd.to_numeric(market_returns, errors="coerce").reindex(returns.index).fillna(0.0)
        index = list(returns.index)

        tau_values: list[float] = []
        magnitude_values: list[float] = []
        baseline_sigma = float(returns.stack().std()) if returns.size else 0.02

        for row in subset.itertuples(index=False):
            event_id = getattr(row, "event_id")
            published = pd.Timestamp(getattr(row, "published_timestamp"))
            if published.tzinfo is None:
                published = published.tz_localize("UTC")
            future = [day for day in index if pd.Timestamp(day).tz_localize("UTC") > published]
            if len(future) < horizon + 2:
                continue
            start = index.index(future[0])
            tickers = (impact_map or {}).get(event_id) or []
            if not tickers:
                continue
            window_index = index[start : start + horizon]
            for ticker in tickers:
                if ticker not in returns.columns:
                    continue
                abnormal = (returns[ticker] - market).reindex(window_index).fillna(0.0)
                for tau, magnitude in enumerate(np.abs(abnormal.to_numpy(dtype=float)), start=0):
                    excess = magnitude - baseline_sigma
                    if excess <= 0:
                        continue
                    tau_values.append(float(tau))
                    magnitude_values.append(float(np.log(max(excess, 1e-6))))

        if len(tau_values) < 30:
            return {"half_life_days": float("nan"), "lambda": float("nan"), "events": float(len(subset))}

        tau = np.asarray(tau_values)
        y = np.asarray(magnitude_values)
        tau_centered = tau - tau.mean()
        slope = float(np.sum(tau_centered * (y - y.mean())) / max(np.sum(tau_centered**2), EPS))
        lam = float(max(-slope, 1.0 / max(horizon, 1)))
        half_life = float(np.log(2.0) / max(lam, EPS))
        half_life = float(
            np.clip(
                half_life,
                float(self.config.e("events.half_life_min_days", 3.0)),
                float(self.config.e("events.half_life_max_days", 180.0)),
            )
        )
        return {"half_life_days": half_life, "lambda": lam, "events": float(len(subset)), "panel_rows": float(len(tau))}

    # -------------------------------------------------------------- 7. IC-IR ---
    def learn_rolling_ic_ir_factor_weights(
        self,
        historical_ortho_factors: Sequence[pd.DataFrame],
        historical_forward_returns: Sequence[pd.Series],
    ) -> dict[str, Any]:
        """Grinold-Kahn IC-IR weights with Bayesian shrinkage toward equal weight."""
        if not historical_ortho_factors:
            return {"weights": np.array([]), "ic_mean": np.array([]), "ic_var": np.array([]), "observations": 0}
        n_pillars = historical_ortho_factors[0].shape[1]
        records: list[list[float]] = []
        for factors, forward in zip(historical_ortho_factors, historical_forward_returns):
            common = factors.index.intersection(forward.index)
            if len(common) < 10:
                continue
            forward_ranks = pd.to_numeric(forward.loc[common], errors="coerce").rank(pct=True)
            row = []
            for column in factors.columns:
                factor_ranks = pd.to_numeric(factors.loc[common, column], errors="coerce").rank(pct=True)
                corr = factor_ranks.corr(forward_ranks)
                row.append(0.0 if corr is None or not np.isfinite(corr) else float(corr))
            records.append(row)

        min_obs = int(self.config.e("factor_learning.min_ic_observations", 5))
        if len(records) < min_obs:
            return {
                "weights": np.ones(n_pillars) / n_pillars,
                "ic_mean": np.zeros(n_pillars),
                "ic_var": np.zeros(n_pillars),
                "observations": len(records),
                "mode": "equal_weight_cold_start",
            }

        ic = np.asarray(records, dtype=float)
        mean_ic = ic.mean(axis=0)
        var_ic = ic.var(axis=0)
        prior = float(self.config.e("factor_learning.prior_precision_scale", 1.0)) / np.sqrt(len(records))
        raw = np.maximum(0.0, mean_ic / (var_ic + prior))
        if raw.sum() <= 1e-8:
            weights = np.ones(n_pillars) / n_pillars
            mode = "equal_weight_no_ic"
        else:
            weights = raw / raw.sum()
            mode = "ic_ir"
        return {
            "weights": weights,
            "ic_mean": mean_ic,
            "ic_var": var_ic,
            "observations": len(records),
            "mode": mode,
        }

    # ------------------------------------------------------ 8. regime blending --
    def blend_with_prior(self, learned: np.ndarray, prior: np.ndarray, observations: int, *, scale: float = 1.0) -> np.ndarray:
        """Shrink IC-IR weights toward the regime prior until the IC sample matures."""
        if learned.size != prior.size or learned.size == 0:
            return prior
        min_obs = int(self.config.e("factor_learning.min_ic_observations", 5))
        reliability = float(np.clip(observations / max(min_obs * 4.0, 1.0), 0.0, 1.0)) * float(scale)
        blended = reliability * learned + (1.0 - reliability) * prior
        total = blended.sum()
        return blended / total if total > 0 else prior

    # --------------------------------------------------------- 9. fundamentals --
    def beta_from_returns(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            return 1.0
        cov = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
        var = float(np.var(aligned.iloc[:, 1], ddof=0))
        if var <= EPS:
            return 1.0
        return float(np.clip(cov / var, 0.0, 3.0))

    def risk_free_from_macro(self, macro_pit: pd.DataFrame, *, default: float = 0.04) -> float:
        if macro_pit is None or macro_pit.empty:
            return float(self.config.e("optimizer.wacc.min_wacc", 0.06))
        candidates = ["US10Y_REAL_YIELD", "US10Y_NOMINAL"]
        frame = macro_pit[macro_pit["series_id"].isin(candidates)]
        if frame.empty:
            return default
        latest = (
            frame.sort_values("observation_date")
            .groupby("series_id")
            .tail(1)
            .set_index("series_id")["value"]
        )
        for series in candidates:
            if series in latest.index:
                return float(np.clip(float(latest[series]) / 100.0, 0.0, 0.15))
        return default
