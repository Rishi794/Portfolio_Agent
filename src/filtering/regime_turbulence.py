"""Tier-1 macro regime controller.

Builds the Point-in-Time 6-dimensional macro/external vector

    m_t = [ d20 YC slope, d20 HY OAS, VIX percentile(252d),
            d20 log DXY, d20 log Brent, CPI surprise ]

and computes the squared Mahalanobis turbulence index d_t^2 with a Ledoit-Wolf
shrunk covariance. Under multivariate normality d_t^2 ~ chi2(6); the 90th percentile
(10.64) anchors the sigmoid that maps turbulence to the cash reserve buffer, while
growth/inflation signs pick one of four structural regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .statistics import mahalanobis_d2, shrunk_covariance, winsorize, zscore

MACRO_VECTOR_COLUMNS = (
    "yield_curve_20d_change",
    "hy_spread_20d_change",
    "vix_percentile_252d",
    "dxy_20d_log_return",
    "brent_20d_log_return",
    "cpi_surprise",
)


@dataclass
class RegimeAllocation:
    regime_name: str
    turbulence_d2: float
    cash_buffer: float
    domain_weights: np.ndarray
    growth_score: float = 0.0
    inflation_score: float = 0.0
    hy_spread_z: float = 0.0
    cpi_surprise_z: float = 0.0
    observations: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "regime": self.regime_name,
            "turbulence_d2": round(self.turbulence_d2, 4),
            "cash_buffer": round(self.cash_buffer, 4),
            "growth_score": round(self.growth_score, 4),
            "inflation_score": round(self.inflation_score, 4),
            "hy_spread_z": round(self.hy_spread_z, 4),
            "cpi_surprise_z": round(self.cpi_surprise_z, 4),
            "observations": self.observations,
        }


class MacroRegimeDetector:
    def __init__(self, config) -> None:
        self.config = config
        self.delta_lookback = int(config.e("macro.delta_lookback_days", 20))
        self.z_window = int(config.e("macro.zscore_lookback_days", 252))
        self.min_observations = int(config.e("macro.min_observations", 60))
        self.vix_window = int(config.e("macro.vix_percentile_window", 252))
        self.cpi_ma_months = int(config.e("macro.cpi_ma_months", 12))
        self.winsor = float(config.e("macro.winsor_quantile", 0.01))
        self.regime_weights = config.regime_weights()
        self.fallback_regime = config.fallback_regime()

    # ------------------------------------------------------------------ matrix --
    def build_macro_matrix(self, macro_pit: pd.DataFrame) -> pd.DataFrame:
        """Pivot the PIT macro table into a daily 6-column regime vector."""
        if macro_pit is None or macro_pit.empty:
            return pd.DataFrame(columns=list(MACRO_VECTOR_COLUMNS))

        frame = macro_pit.copy()
        frame["observation_date"] = pd.to_datetime(frame["observation_date"])
        # Restatements: keep the FIRST *vintage* per (series, period) -> no revision
        # look-ahead. The sort must include the vintage timestamp and be stable, or
        # same-period rows could be reordered by an unstable sort and the revision win.
        sort_columns = ["series_id", "observation_date"]
        if "vintage_timestamp" in frame.columns:
            frame["vintage_timestamp"] = pd.to_datetime(
                frame["vintage_timestamp"], utc=True, errors="coerce"
            )
            sort_columns.append("vintage_timestamp")
        frame = frame.sort_values(sort_columns, kind="stable")
        frame = frame.drop_duplicates(subset=["series_id", "observation_date"], keep="first")
        panel = frame.pivot_table(index="observation_date", columns="series_id", values="value", aggfunc="last")
        panel = panel.sort_index()

        daily_index = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
        panel = panel.reindex(daily_index).ffill(limit=int(self.config.e("macro.ffill_limit_days", 40)))

        columns: dict[str, pd.Series] = {}
        if "YIELD_CURVE_10Y2Y" in panel:
            curve = pd.to_numeric(panel["YIELD_CURVE_10Y2Y"], errors="coerce")
            columns["yield_curve_20d_change"] = curve.diff(self.delta_lookback)
        if "HY_CREDIT_SPREAD" in panel:
            spread = pd.to_numeric(panel["HY_CREDIT_SPREAD"], errors="coerce")
            columns["hy_spread_20d_change"] = spread.diff(self.delta_lookback)
        if "VIX" in panel:
            vix = pd.to_numeric(panel["VIX"], errors="coerce")
            window = max(self.vix_window // 4, 20)
            columns["vix_percentile_252d"] = vix.rolling(window, min_periods=window // 4).rank(pct=True)
        if "DXY" in panel:
            dxy = pd.to_numeric(panel["DXY"], errors="coerce")
            columns["dxy_20d_log_return"] = np.log(dxy / dxy.shift(self.delta_lookback))
        if "BRENT" in panel:
            brent = pd.to_numeric(panel["BRENT"], errors="coerce")
            columns["brent_20d_log_return"] = np.log(brent / brent.shift(self.delta_lookback))
        if "CPI_INDEX" in panel:
            cpi = pd.to_numeric(panel["CPI_INDEX"], errors="coerce")
            columns["cpi_surprise"] = _cpi_surprise(cpi, self.cpi_ma_months)

        matrix = pd.DataFrame(columns)
        for column in MACRO_VECTOR_COLUMNS:
            if column not in matrix.columns:
                matrix[column] = np.nan
        matrix = matrix[list(MACRO_VECTOR_COLUMNS)]
        standardized = pd.DataFrame(index=matrix.index)
        for column in MACRO_VECTOR_COLUMNS:
            standardized[column] = zscore(winsorize(matrix[column], self.winsor).ffill(), 0.0)
        return standardized.dropna(how="all")

    # ----------------------------------------------------------------- detect ---
    def detect(self, macro_matrix: pd.DataFrame) -> RegimeAllocation:
        clean = macro_matrix.replace([np.inf, -np.inf], np.nan)
        clean = clean.dropna()
        prior_weights = self.regime_weights.get(self.fallback_regime, np.ones(6) / 6.0)
        if len(clean) < self.min_observations:
            cash_prior = float(self.config.e("turbulence.cash_min", 0.03))
            return RegimeAllocation(
                regime_name=self.fallback_regime,
                turbulence_d2=0.0,
                cash_buffer=cash_prior,
                domain_weights=prior_weights,
                observations=int(len(clean)),
                diagnostics={"reason": "insufficient_macro_history"},
            )

        window = min(len(clean), self.z_window)
        sample = clean.tail(window).to_numpy(dtype=float)
        cov = shrunk_covariance(sample, method=str(self.config.e("turbulence.shrinkage", "ledoit_wolf")))
        mean_vec = sample.mean(axis=0)
        latest = sample[-1]
        d2 = mahalanobis_d2(latest, mean_vec, cov)

        cash_min = float(self.config.e("turbulence.cash_min", 0.03))
        cash_max = float(self.config.e("turbulence.cash_max", 0.45))
        gamma_m = float(self.config.e("turbulence.gamma_m", 0.5))
        chi2_q90 = float(self.config.e("turbulence.chi2_q_90", 10.64))
        cash_buffer = cash_min + (cash_max - cash_min) / (1.0 + np.exp(-gamma_m * (d2 - chi2_q90)))

        hy_z = float(latest[list(MACRO_VECTOR_COLUMNS).index("hy_spread_20d_change")])
        cpi_z = float(latest[list(MACRO_VECTOR_COLUMNS).index("cpi_surprise")])
        curve_z = float(latest[list(MACRO_VECTOR_COLUMNS).index("yield_curve_20d_change")])
        growth_score = float(np.tanh(0.5 * (curve_z - hy_z)))

        cls = self.config.e("turbulence.classification", {}) or {}
        crisis_d2 = float(cls.get("crisis_d2", 12.5))
        crisis_hy = float(cls.get("crisis_hy_z", 1.5))
        stag_cpi = float(cls.get("stagflation_cpi_z", 0.75))
        stag_hy = float(cls.get("stagflation_hy_z", 0.5))
        reflation_cpi = float(cls.get("reflation_cpi_z", 0.5))

        if d2 >= crisis_d2 or hy_z >= crisis_hy:
            regime = "CRISIS_TURBULENCE"
        elif cpi_z >= stag_cpi and hy_z >= stag_hy:
            regime = "STAGFLATION_STRESS"
        elif cpi_z >= reflation_cpi and hy_z < stag_hy:
            regime = "INFLATIONARY_BOOM"
        else:
            regime = "LOW_VOL_EXPANSION"

        cash_floor_band, cash_ceiling_band = self.config.regime_cash_range(regime)
        cash_buffer = float(np.clip(cash_buffer, cash_floor_band, cash_ceiling_band))
        weights = self.regime_weights.get(regime, prior_weights)

        return RegimeAllocation(
            regime_name=regime,
            turbulence_d2=d2,
            cash_buffer=cash_buffer,
            domain_weights=weights,
            growth_score=growth_score,
            inflation_score=float(np.tanh(cpi_z)),
            hy_spread_z=hy_z,
            cpi_surprise_z=cpi_z,
            observations=int(len(clean)),
            diagnostics={
                "cov_window": int(window),
                "shrunk_cov_trace": float(np.trace(cov)),
                "chi2_q90": chi2_q90,
                "vintage_safe": True,
            },
        )

    def run(self, macro_pit: pd.DataFrame) -> tuple[RegimeAllocation, pd.DataFrame]:
        matrix = self.build_macro_matrix(macro_pit)
        return self.detect(matrix), matrix


def _cpi_surprise(cpi_index: pd.Series, ma_months: int) -> pd.Series:
    """YoY inflation minus its own trailing moving average (a PIT inflation surprise)."""
    if cpi_index.dropna().empty:
        return pd.Series(np.nan, index=cpi_index.index)
    monthly = cpi_index.resample("ME").last().dropna()
    if monthly.empty:
        return pd.Series(np.nan, index=cpi_index.index)
    yoy = monthly.pct_change(12) * 100.0
    surprise = yoy - yoy.rolling(max(ma_months, 3), min_periods=3).mean()
    return surprise.reindex(cpi_index.index, method="ffill")
