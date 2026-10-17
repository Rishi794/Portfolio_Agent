"""Symmetric (Löwdin) orthogonalization of the cross-sectional factor matrix.

Given the standardized factor matrix F (N x K) with correlation matrix
C_F = F'F / N, the symmetric orthogonalization

    F_perp = F @ C_F^{-1/2} = F @ V Lambda^{-1/2} V'

minimizes ||F_perp - F||_F^2 over all orthogonal matrices with unit-variance
columns, i.e. it removes factor double-counting while staying as close as possible
to the economic interpretation of the raw factors (Löwdin 1950; symmetric
orthogonalization in signal processing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lowdin_orthogonalize(
    raw: pd.DataFrame,
    *,
    winsor_quantile: float = 0.02,
    eigenvalue_floor: float = 1e-4,
    min_observations: int = 8,
) -> pd.DataFrame:
    frame = raw.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    standardized = frame.copy()
    for column in standardized.columns:
        series = standardized[column]
        if winsor_quantile > 0 and len(series) > 1:
            lower, upper = series.quantile(winsor_quantile), series.quantile(1.0 - winsor_quantile)
            series = series.clip(lower, upper)
        std = float(series.std(ddof=0))
        standardized[column] = (series - float(series.mean())) / (std if std > 1e-12 else 1.0)

    n_rows, n_cols = standardized.shape
    if n_rows <= n_cols or n_rows < min_observations:
        return standardized

    matrix = standardized.to_numpy(dtype=float)
    # Constant columns have zero variance and cannot be whitened; exclude them
    # from the transform (they stay zero) so the informative subspace is
    # orthogonalized exactly instead of flooding the clipping floor.
    standard_deviations = standardized.std(axis=0, ddof=0).to_numpy(dtype=float)
    informative = np.flatnonzero(standard_deviations > 1e-12)
    if informative.size == 0 or n_rows <= informative.size:
        return standardized

    block = matrix[:, informative]
    corr = block.T @ block / n_rows
    eigvals, eigvecs = np.linalg.eigh((corr + corr.T) / 2.0)
    eigvals = np.maximum(eigvals, eigenvalue_floor)
    inverse_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    orthogonal = np.zeros_like(matrix)
    orthogonal[:, informative] = block @ inverse_sqrt
    return pd.DataFrame(orthogonal, index=raw.index, columns=raw.columns)


def correlation_matrix(factors: pd.DataFrame) -> pd.DataFrame:
    frame = factors.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if len(frame) < 2:
        return pd.DataFrame(np.eye(frame.shape[1]), index=frame.columns, columns=frame.columns)
    corr = np.corrcoef(frame.to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=frame.columns, columns=frame.columns)


def condition_number(factors: pd.DataFrame) -> float:
    corr = correlation_matrix(factors).to_numpy(dtype=float)
    eigvals = np.linalg.eigvalsh((corr + corr.T) / 2.0)
    eigvals = np.maximum(eigvals, 1e-12)
    return float(eigvals.max() / eigvals.min())


def orthogonality_error(orthogonal: pd.DataFrame, n_rows: int | None = None) -> float:
    """Frobenius distance of (1/N) F_perp' F_perp from the identity.

    Constant (zero-variance) columns cannot be whitened and are excluded from the
    measurement, matching ``lowdin_orthogonalize``'s informative-subspace transform.
    """
    matrix = orthogonal.to_numpy(dtype=float)
    n = n_rows or matrix.shape[0]
    if n == 0 or matrix.size == 0:
        return 0.0
    gram = matrix.T @ matrix / n
    diagonal = np.diag(gram)
    keep = diagonal > 1e-12
    if not keep.any():
        return 0.0
    block = gram[np.ix_(keep, keep)]
    return float(np.linalg.norm(block - np.eye(int(keep.sum())), ord="fro"))


def max_abs_correlation(factors: pd.DataFrame) -> float:
    # np.array(...) forces a writable copy: pandas >= 3.0 may hand back a read-only
    # view from to_numpy(), and np.fill_diagonal mutates in place.
    corr = np.array(np.abs(correlation_matrix(factors)), dtype=float, copy=True)
    np.fill_diagonal(corr, 0.0)
    return float(corr.max()) if corr.size else 0.0
