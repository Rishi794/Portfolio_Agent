"""Macro turbulence controller: Mahalanobis cash buffer and regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.filtering.regime_turbulence import MACRO_VECTOR_COLUMNS, MacroRegimeDetector


def _matrix(rng: np.random.Generator, rows: int) -> pd.DataFrame:
    return pd.DataFrame(rng.normal(0.0, 1.0, size=(rows, 6)), columns=list(MACRO_VECTOR_COLUMNS))


def test_detect_is_deterministic_and_controls_cash(config) -> None:
    rng = np.random.default_rng(0)
    detector = MacroRegimeDetector(config)
    normal = detector.detect(_matrix(rng, 240))
    assert normal.regime_name in set(config.regime_weights())
    assert normal.observations == 240
    assert np.isfinite(normal.turbulence_d2)
    low, high = config.regime_cash_range(normal.regime_name)
    assert low <= normal.cash_buffer <= high
    assert normal.domain_weights.sum() == pytest.approx(1.0)
    assert len(normal.domain_weights) == len(MACRO_VECTOR_COLUMNS)


def test_crisis_shock_raises_turbulence_and_cash(config) -> None:
    rng = np.random.default_rng(1)
    detector = MacroRegimeDetector(config)
    base = _matrix(rng, 240)
    # hy_spread_20d_change = +5z, cpi_surprise = +4z on the last three rows
    shock_rows = pd.DataFrame([[0.0, 5.0, 3.0, 0.0, 0.0, 4.0]] * 3, columns=list(MACRO_VECTOR_COLUMNS))
    shocked = pd.concat([base, shock_rows], ignore_index=True)
    normal = detector.detect(base)
    stressed = detector.detect(shocked)
    assert stressed.regime_name == "CRISIS_TURBULENCE"
    assert stressed.turbulence_d2 > normal.turbulence_d2
    assert stressed.cash_buffer > normal.cash_buffer
    assert config.regime_cash_range("CRISIS_TURBULENCE")[0] <= stressed.cash_buffer


def test_stagflation_branch(config) -> None:
    rng = np.random.default_rng(2)
    detector = MacroRegimeDetector(config)
    base = _matrix(rng, 240)
    # mild but persistent inflation + credit stress, turbulence below the crisis cut
    rows = pd.DataFrame([[0.0, 1.0, 0.5, 0.0, 0.0, 1.2]] * 5, columns=list(MACRO_VECTOR_COLUMNS))
    allocation = detector.detect(pd.concat([base, rows], ignore_index=True))
    assert allocation.regime_name in {"STAGFLATION_STRESS", "CRISIS_TURBULENCE"}
    assert allocation.inflation_score > 0


def test_insufficient_history_returns_fallback(config) -> None:
    detector = MacroRegimeDetector(config)
    allocation = detector.detect(_matrix(np.random.default_rng(3), 12))
    assert allocation.regime_name == config.fallback_regime()
    assert allocation.cash_buffer == pytest.approx(config.e("turbulence.cash_min", 0.03), abs=1e-9)
    assert allocation.diagnostics["reason"] == "insufficient_macro_history"


def test_build_macro_matrix_has_no_revision_lookahead(config) -> None:
    """A revised vintage with a *later* vintage timestamp must not change the matrix."""
    rng = np.random.default_rng(4)
    days = pd.date_range("2025-01-01", periods=120, freq="D")
    rows = []
    for day in days:
        rows.append(
            {"series_id": "VIX", "observation_date": day, "vintage_timestamp": pd.Timestamp(day, tz="UTC"),
             "value": float(18.0 + rng.normal(0, 1))}
        )
        rows.append(
            {"series_id": "HY_CREDIT_SPREAD", "observation_date": day, "vintage_timestamp": pd.Timestamp(day, tz="UTC"),
             "value": float(3.5 + rng.normal(0, 0.1))}
        )
        rows.append(
            {"series_id": "YIELD_CURVE_10Y2Y", "observation_date": day, "vintage_timestamp": pd.Timestamp(day, tz="UTC"),
             "value": float(0.3 + rng.normal(0, 0.05))}
        )
    revised = list(rows)
    for row in rows[:60]:  # every early point gets a wild revised print one month later
        revised.append({**row, "vintage_timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "value": row["value"] * 10.0})
    frame = pd.DataFrame(rows)
    detector = MacroRegimeDetector(config)
    clean = detector.build_macro_matrix(frame)
    dirty = detector.build_macro_matrix(pd.DataFrame(revised))
    pd.testing.assert_frame_equal(clean, dirty)
    assert set(clean.columns) == set(MACRO_VECTOR_COLUMNS)
