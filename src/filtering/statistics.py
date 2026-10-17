"""Cross-sectional statistics: winsorization, shrinkage covariance, exact
spherical-null hypothesis testing and Benjamini-Hochberg FDR control.

Key result used throughout the event pipeline
---------------------------------------------
Let u, v ~ Uniform(S^{D-1}) and independent (the null: an event is unrelated to a
company). By rotational invariance fix u = e_1; then z = u'v is the first
coordinate of a random unit vector, hence

    z^2 ~ Beta(1/2, (D-1)/2)
    t   = z * sqrt((D-1) / (1 - z^2)) ~ Student-t(D-1)          (exact)

so cosine similarity admits *exact* p-values without simulation. Those p-values
feed a Benjamini-Hochberg step-up procedure whose critical value defines the data
adaptive similarity gate tau_hat — no hardcoded 0.62.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import t as student_t
from sklearn.covariance import LedoitWolf


# --------------------------------------------------------------------------- #
# cross-sectional transforms
# --------------------------------------------------------------------------- #
def winsorize(series: pd.Series, quantile: float) -> pd.Series:
    # Force float: clipping integer (nullable Int64) columns with float bounds
    # raises on pandas >= 3.0, and every downstream use is continuous anyway.
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    if quantile <= 0 or values.dropna().empty:
        return values
    lower, upper = values.quantile(quantile), values.quantile(1.0 - quantile)
    return values.clip(lower, upper)


def zscore(series: pd.Series, winsor_quantile: float = 0.0, *, ddof: int = 0) -> pd.Series:
    values = winsorize(series, winsor_quantile)
    std = float(values.std(ddof=ddof))
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=values.index, dtype="float64")
    return (values - float(values.mean())) / std


def sector_zscore(frame: pd.DataFrame, column: str, sector_column: str = "gics_sector", winsor_quantile: float = 0.0) -> pd.Series:
    """Standardize a column within GICS/NIC sector, then re-standardize globally."""
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    if sector_column not in frame.columns:
        return zscore(frame[column], winsor_quantile)
    grouped = frame.groupby(sector_column, dropna=False)[column]
    out = pd.Series(0.0, index=frame.index, dtype="float64")
    for _, idx in grouped.groups.items():
        out.loc[idx] = zscore(frame.loc[idx, column], winsor_quantile)
    return zscore(out, 0.0)


def shrunk_covariance(matrix: np.ndarray, *, method: str = "ledoit_wolf") -> np.ndarray:
    """PSD-projected covariance: Ledoit-Wolf shrinkage, or diagonal fallback."""
    data = np.asarray(matrix, dtype=float)
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 3:
        diag = np.diag(np.maximum(np.var(data, axis=0) if len(data) else np.ones(data.shape[1]), 1e-12))
        return diag
    if method == "ledoit_wolf":
        try:
            return LedoitWolf().fit(data).covariance_
        except Exception:
            pass
    return np.diag(np.maximum(np.var(data, axis=0), 1e-12))


def psd_project(matrix: np.ndarray, floor: float) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    return eigvecs @ np.diag(np.maximum(eigvals, floor)) @ eigvecs.T


def mahalanobis_d2(vector: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    diff = np.asarray(vector, dtype=float) - np.asarray(mean, dtype=float)
    inv = np.linalg.pinv(cov)
    return float(diff @ inv @ diff)


# --------------------------------------------------------------------------- #
# spherical null hypothesis testing
# --------------------------------------------------------------------------- #
def cosine_pvalues(cosine: np.ndarray, dim: int) -> np.ndarray:
    """Exact two-sided p-values for dot products of unit vectors on S^{D-1}."""
    clipped = np.clip(np.asarray(cosine, dtype=float), -1.0 + 1e-12, 1.0 - 1e-12)
    dof = max(dim - 1, 1)
    t_stats = clipped * np.sqrt(dof / np.maximum(1.0 - clipped**2, 1e-18))
    return student_t.sf(np.abs(t_stats), df=dof) * 2.0


def cosine_statistics(cosine: np.ndarray, dim: int) -> np.ndarray:
    clipped = np.clip(np.asarray(cosine, dtype=float), -1.0 + 1e-12, 1.0 - 1e-12)
    dof = max(dim - 1, 1)
    return clipped * np.sqrt(dof / np.maximum(1.0 - clipped**2, 1e-18))


def t_to_cosine(t_stat: float, dim: int) -> float:
    dof = max(dim - 1, 1)
    return float(t_stat / np.sqrt(dof + t_stat**2))


def beta_null_pdf(squared_cosine: np.ndarray, dim: int) -> np.ndarray:
    """Analytic null density of z^2 ~ Beta(1/2, (D-1)/2) (diagnostics/tests)."""
    return beta_dist.pdf(np.clip(squared_cosine, 0.0, 1.0), 0.5, max((dim - 1) / 2.0, 0.5))


def bh_stepup(p_values: np.ndarray, alpha: float) -> tuple[float | None, int]:
    """Benjamini-Hochberg step-up. Returns (critical p-value, k*) or (None, 0)."""
    flat = np.sort(np.asarray(p_values, dtype=float).ravel())
    flat = flat[np.isfinite(flat)]
    m = len(flat)
    if m == 0:
        return None, 0
    thresholds = (np.arange(1, m + 1) / m) * alpha
    significant = flat <= thresholds
    if not significant.any():
        return None, 0
    k_star = int(np.where(significant)[0][-1])
    return float(flat[k_star]), k_star + 1


def adaptive_cosine_gate(
    cosine: np.ndarray,
    dim: int,
    *,
    alpha_scale: float = 1.0,
    min_pairs: int = 100,
    fallback_p: float = 0.05,
) -> dict[str, float]:
    """Data-adaptive tau_hat from the exact spherical null + BH-FDR control.

    ``alpha_FDR`` self-normalizes as ``alpha_scale / sqrt(M)`` so that growing the
    event x company panel does not inflate false discoveries.
    """
    pairs = np.asarray(cosine, dtype=float).ravel()
    pairs = pairs[np.isfinite(pairs)]
    m = pairs.shape[0]
    if m == 0:
        return {"tau_hat": 1.0, "p_crit": float("nan"), "k_star": 0.0, "alpha_fdr": float("nan"), "pairs": 0.0}
    alpha = float(alpha_scale) / np.sqrt(max(m, min_pairs))
    p_values = cosine_pvalues(pairs, dim)
    p_crit, k_star = bh_stepup(p_values, alpha)
    if p_crit is None:
        p_crit = float(np.quantile(p_values, min(max(fallback_p, 1e-6), 0.5)))
        k_star = 0
    t_crit = float(student_t.isf(p_crit / 2.0, df=max(dim - 1, 1)))
    return {
        "tau_hat": t_to_cosine(t_crit, dim),
        "p_crit": float(p_crit),
        "k_star": float(k_star),
        "alpha_fdr": alpha,
        "pairs": float(m),
    }


def exponential_decay(age_days: np.ndarray, half_life_days: np.ndarray) -> np.ndarray:
    ages = np.maximum(np.asarray(age_days, dtype=float), 0.0)
    half_lives = np.maximum(np.asarray(half_life_days, dtype=float), 1e-6)
    return np.exp(-np.log(2.0) * ages / half_lives)


def robust_z(values: pd.Series, window: int | None = None) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if window:
        mean = values.rolling(window, min_periods=max(5, window // 5)).mean()
        std = values.rolling(window, min_periods=max(5, window // 5)).std()
    else:
        mean, std = values.mean(), values.std()
    return (values - mean) / std.replace(0.0, np.nan)
