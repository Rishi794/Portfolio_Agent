"""Symmetric (Löwdin) orthogonalization: multicollinearity removal without rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.filtering.lowdin_orthogonalizer import (
    condition_number,
    correlation_matrix,
    lowdin_orthogonalize,
    max_abs_correlation,
    orthogonality_error,
)


def _collinear_panel(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(size=n)
    return pd.DataFrame(
        {
            "kalman_trend": common + 0.1 * rng.normal(size=n),
            "fundamentals": 0.95 * common + 0.15 * rng.normal(size=n),   # ~0.99 correlated
            "tech_evolution": 0.6 * common + 0.6 * rng.normal(size=n),
            "gov_policy": rng.normal(size=n),
            "geopolitics": rng.normal(size=n),
            "macro_beta": 0.3 * common + 0.9 * rng.normal(size=n),
        }
    )


def test_lowdin_removes_collinearity() -> None:
    raw = _collinear_panel()
    ortho = lowdin_orthogonalize(raw, winsor_quantile=0.02)
    assert max_abs_correlation(raw) > 0.9
    assert max_abs_correlation(ortho) < 0.02
    assert orthogonality_error(ortho) < 1e-3
    assert condition_number(ortho) == pytest.approx(1.0, abs=0.02)
    # dimension and column identity are preserved
    assert list(ortho.columns) == list(raw.columns)
    assert ortho.shape == raw.shape
    assert np.isfinite(ortho.to_numpy()).all()


def test_lowdin_is_scale_invariant_and_keeps_rank() -> None:
    raw = _collinear_panel(n=200, seed=9)
    scaled = raw.copy()
    scaled["fundamentals"] *= 1e6
    a = lowdin_orthogonalize(raw)
    b = lowdin_orthogonalize(scaled)
    # the orthogonalized coordinates are invariant to a re-scaling of the input
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-9)
    ranks = [
        np.linalg.matrix_rank(raw.to_numpy()),
        np.linalg.matrix_rank(lowdin_orthogonalize(raw).to_numpy()),
    ]
    assert ranks[0] == ranks[1] == raw.shape[1]


def test_lowdin_falls_back_when_wide() -> None:
    """N <= K: nothing to orthogonalize against, so it must return standardized factors."""
    raw = _collinear_panel(n=4, seed=2)
    out = lowdin_orthogonalize(raw, min_observations=8)
    assert out.shape == raw.shape
    assert np.isfinite(out.to_numpy()).all()
    assert np.allclose(out.mean(axis=0).to_numpy(), 0.0, atol=1e-9)


def test_correlation_matrix_diagonal_is_one() -> None:
    corr = correlation_matrix(_collinear_panel(n=60))
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
