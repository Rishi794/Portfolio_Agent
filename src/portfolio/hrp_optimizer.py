"""Hierarchical Risk Parity (López de Prado 2016) with an alpha tilt and a
cost-aware rebalancer.

Steps
-----
1. correlation -> distance ``d = sqrt(0.5 (1 - rho))`` -> **Ward** linkage;
2. optimal leaf ordering (quasi-diagonalization) to keep similar assets adjacent;
3. recursive bisection allocating inverse-variance weights inside each cluster;
4. alpha tilt: ``w ~ w_HRP * exp(tilt * clip(z_alpha))`` (a Bayesian tilt away from
   pure risk parity toward expected return, strength from config);
5. weight cap + cash buffer scaling, then a KKT deadband filter that suppresses
   any trade whose expected alpha gain cannot pay the stock-specific friction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from ..filtering.statistics import shrunk_covariance, winsorize, zscore

EPS = 1e-12


@dataclass
class OrderTicket:
    market: str
    action: str
    ticker: str
    shares: float
    ref_price: float
    target_weight_pct: float
    current_weight_pct: float
    kkt_deadband_pct: float
    est_cost_bps: float
    alpha_z: float
    sector: str = "UNKNOWN"
    order_value: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "action": self.action,
            "ticker": self.ticker,
            "shares": self.shares,
            "ref_price": round(self.ref_price, 4),
            "target_weight_pct": round(self.target_weight_pct, 4),
            "current_weight_pct": round(self.current_weight_pct, 4),
            "kkt_deadband_pct": round(self.kkt_deadband_pct, 4),
            "est_cost_bps": round(self.est_cost_bps, 4),
            "alpha_z": round(self.alpha_z, 4),
            "sector": self.sector,
            "order_value": round(self.order_value, 2),
            "reason": self.reason,
        }


class HRPOptimizer:
    def __init__(self, config) -> None:
        self.config = config
        self.linkage_method = str(config.e("optimizer.hrp_linkage", "ward"))
        self.alpha_tilt = float(config.e("optimizer.alpha_tilt_strength", 0.35))
        self.max_weight = float(config.e("optimizer.max_weight", 0.06))
        self.max_turnover = float(config.e("optimizer.max_turnover", 0.35))
        self.min_position_value = float(config.e("optimizer.min_position_value", 1000.0))
        self.cov_shrinkage = str(config.e("optimizer.cov_shrinkage", "ledoit_wolf"))

    # ------------------------------------------------------------ clustering --
    def ward_seriation(self, returns: pd.DataFrame) -> list[str]:
        columns = list(returns.columns)
        if len(columns) < 3:
            return columns
        data = returns.to_numpy(dtype=float)
        correlation = np.corrcoef(data, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0)
        np.fill_diagonal(correlation, 1.0)
        distance = np.sqrt(np.maximum(0.5 * (1.0 - correlation), 0.0))
        condensed = _condensed(distance)
        if condensed.size == 0 or np.allclose(condensed, 0.0):
            return columns
        link = linkage(condensed, method=self.linkage_method)
        leaves = dendrogram(link, no_plot=True)["leaves"]
        return [columns[i] for i in leaves]

    # ------------------------------------------------------- recursive tree --
    def hrp_weights(self, returns: pd.DataFrame, order: Sequence[str] | None = None) -> pd.Series:
        columns = list(order) if order is not None else self.ward_seriation(returns)
        if not columns:
            return pd.Series(dtype=float)
        if len(columns) == 1:
            return pd.Series([1.0], index=columns, dtype=float)
        cov = shrunk_covariance(returns[columns].to_numpy(dtype=float), method=self.cov_shrinkage)
        weights = pd.Series(1.0, index=columns, dtype=float)
        clusters: list[list[str]] = [list(columns)]
        while clusters:
            split: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                mid = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]
                var_left = self._cluster_variance(cov, columns, left)
                var_right = self._cluster_variance(cov, columns, right)
                alpha = 1.0 - var_left / max(var_left + var_right, EPS)
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha
                split.extend([left, right])
            clusters = split
        total = float(weights.sum())
        return weights / total if total > 0 else pd.Series(1.0 / len(columns), index=columns)

    @staticmethod
    def _cluster_variance(cov: np.ndarray, columns: Sequence[str], items: Sequence[str]) -> float:
        positions = [columns.index(item) for item in items]
        sub = cov[np.ix_(positions, positions)]
        diagonal = np.diag(sub)
        inverted = 1.0 / np.maximum(diagonal, EPS)
        parity = inverted / inverted.sum()
        return float(parity @ sub @ parity)

    # ---------------------------------------------------------------- optimize --
    def optimize(
        self,
        returns: pd.DataFrame,
        alpha: pd.Series,
        *,
        cash_buffer: float,
        current_weights: Mapping[str, float] | None = None,
    ) -> pd.Series:
        if returns.empty:
            return pd.Series(dtype=float)
        columns = [column for column in returns.columns if column in alpha.index]
        if not columns:
            return pd.Series(dtype=float)
        order = self.ward_seriation(returns[columns])
        base = self.hrp_weights(returns[columns], order)
        curated = [column for column in order if column in base.index]
        base = base.reindex(curated).fillna(0.0)

        clipped_alpha = zscore(winsorize(alpha.reindex(curated), 0.02), 0.0).clip(-3.0, 3.0)
        if float(np.abs(clipped_alpha).sum()) > 0.0:
            tilted = base * np.exp(self.alpha_tilt * clipped_alpha)
        else:
            tilted = base.copy()
        if float(tilted.sum()) <= EPS:
            tilted = base.copy()
        tilted = tilted / max(float(tilted.sum()), EPS)

        invested = float(np.clip(1.0 - cash_buffer, 0.0, 1.0))
        cap = self.max_weight * invested
        # The concentration cap is a hard risk constraint: if fewer than 1/cap names are
        # eligible the book is deliberately left *under-invested* (the remainder stays in
        # the cash reserve) instead of silently breaching the cap.
        weights = self._enforce_cap(tilted * invested, cap)

        if current_weights:
            weights = self._limit_turnover(weights, current_weights)
            weights = self._enforce_cap(weights, cap)
        return weights.sort_values(ascending=False)

    def _enforce_cap(self, weights: pd.Series, cap: float) -> pd.Series:
        """Cap each weight at ``cap`` and redistribute the deficit to uncapped names.

        The total is never renormalised: when the cap is infeasible the residue is left
        unallocated (held as cash) rather than violating the constraint.
        """
        if cap <= 0 or weights.empty:
            return weights
        capped = weights.clip(upper=cap)
        for _ in range(50):
            deficit = float(weights.sum() - capped.sum())
            if deficit <= 1e-9:
                break
            room = (cap - capped).clip(lower=0.0)
            if float(room.sum()) <= 1e-12:
                break
            capped = capped + room * (deficit / float(room.sum()))
            capped = capped.clip(upper=cap)
        return capped

    def _limit_turnover(self, target: pd.Series, current: Mapping[str, float]) -> pd.Series:
        """Move only a fraction of the way from the current book to the target.

        The blended vector is deliberately NOT renormalised back to the invested
        budget: doing so would trade more than ``max_turnover`` whenever the current
        book is under-invested, silently breaching the constraint.
        """
        universe = sorted(set(target.index) | set(current))
        target_vec = target.reindex(universe).fillna(0.0)
        current_vec = pd.Series({ticker: float(current.get(ticker, 0.0)) for ticker in universe})
        turnover = float((target_vec - current_vec).abs().sum())
        if turnover <= self.max_turnover or turnover <= EPS:
            return target.sort_values(ascending=False)
        scale = self.max_turnover / turnover
        blended = current_vec + (target_vec - current_vec) * scale
        return blended.sort_values(ascending=False)

    # ------------------------------------------------------------------ orders --
    def generate_orders(
        self,
        *,
        market: str,
        target_weights: pd.Series,
        current_weights: Mapping[str, float],
        prices: Mapping[str, float],
        deadbands: Mapping[str, float],
        costs_bps: Mapping[str, float],
        alpha: pd.Series,
        sectors: Mapping[str, str] | None = None,
        equity: float = 100_000.0,
    ) -> list[OrderTicket]:
        sectors = sectors or {}
        universe = sorted(set(target_weights.index) | set(current_weights))
        tickets: list[OrderTicket] = []
        for ticker in universe:
            price = float(prices.get(ticker, 0.0) or 0.0)
            target = float(target_weights.get(ticker, 0.0))
            current = float(current_weights.get(ticker, 0.0))
            drift = target - current
            deadband = float(deadbands.get(ticker, 0.0))
            is_exit = target <= 1e-9 and current > 0.0
            if abs(drift) <= 1e-6:
                continue
            if not is_exit and abs(drift) < deadband:
                continue
            notional = abs(drift) * equity
            if not is_exit and notional < self.min_position_value:
                continue
            if price <= 0:
                continue
            shares = float(int(notional // price))
            if shares <= 0 and not is_exit:
                continue
            action = "BUY" if drift > 0 else "SELL"
            tickets.append(
                OrderTicket(
                    market=market,
                    action=action,
                    ticker=ticker,
                    shares=shares,
                    ref_price=price,
                    target_weight_pct=target * 100.0,  # placeholder, converted to invested-% by caller if desired
                    current_weight_pct=current * 100.0,
                    kkt_deadband_pct=deadband * 100.0,
                    est_cost_bps=float(costs_bps.get(ticker, 0.0)),
                    alpha_z=float(alpha.get(ticker, 0.0)),
                    sector=str(sectors.get(ticker, "UNKNOWN")),
                    order_value=shares * price,
                    reason="alpha_below_deadband" if not is_exit else "full_exit",
                )
            )
        tickets.sort(key=lambda ticket: abs(ticket.order_value), reverse=True)
        return tickets


def _condensed(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0]
    if n < 2:
        return np.array([])
    rows, cols = np.triu_indices(n, k=1)
    return distance[rows, cols]


def weights_from_scores(scores: pd.Series, *, top_n: int | None = None) -> pd.Series:
    """Fallback equal-weight allocation over positive-alpha names."""
    positive = scores[scores > 0].sort_values(ascending=False)
    if top_n:
        positive = positive.head(top_n)
    if positive.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(positive), index=positive.index)
