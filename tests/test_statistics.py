"""Exact spherical-null statistics and Benjamini-Hochberg FDR control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from src.filtering.statistics import (
    adaptive_cosine_gate,
    beta_null_pdf,
    bh_stepup,
    cosine_pvalues,
    exponential_decay,
    shrunk_covariance,
    t_to_cosine,
    zscore,
)


def test_bh_stepup_known_example() -> None:
    p_values = np.array([0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    critical, k_star = bh_stepup(p_values, alpha=0.05)
    # thresholds are i/m * alpha = 0.005, 0.010, 0.015 ... -> only the first two pass
    assert k_star == 2
    assert critical == pytest.approx(0.002)


def test_bh_stepup_none_significant() -> None:
    critical, k_star = bh_stepup(np.array([0.4, 0.5, 0.9]), alpha=0.05)
    assert critical is None and k_star == 0


def test_cosine_pvalues_are_calibrated_under_the_spherical_null() -> None:
    rng = np.random.default_rng(7)
    dim = 384
    u = rng.normal(size=(2000, dim))
    v = rng.normal(size=(2000, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p_values = cosine_pvalues(np.einsum("ij,ij->i", u, v), dim)
    # exact transform => p-values uniform on (0, 1)
    assert kstest(p_values, "uniform").pvalue > 0.01


def test_beta_null_density_integrates_to_one() -> None:
    # z^2 ~ Beta(1/2, (D-1)/2). The density has an integrable x^{-1/2} singularity
    # at 0, so integrate in z = sqrt(x): integral = \int f(z^2) 2z dz, which is finite.
    z_grid = np.linspace(1e-9, 1.0, 40001)
    density = beta_null_pdf(z_grid**2, 384) * 2.0 * z_grid
    integral = np.trapezoid(density, z_grid)
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_t_to_cosine_round_trip() -> None:
    for cosine in (0.05, 0.2, 0.5):
        t_stat = cosine * np.sqrt(383.0 / (1.0 - cosine**2))
        assert t_to_cosine(t_stat, 384) == pytest.approx(cosine, abs=1e-9)


def test_adaptive_gate_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(11)
    dim = 384
    noise = rng.normal(size=(400, dim))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    anchor = rng.normal(size=dim)
    anchor /= np.linalg.norm(anchor)
    # planted matches: the anchor plus a small orthogonal perturbation -> cos ~ 0.89
    signal = anchor + 0.5 * noise[:40]
    signal /= np.linalg.norm(signal, axis=1, keepdims=True)
    noise_cosines = noise @ anchor
    signal_cosines = signal @ anchor
    assert signal_cosines.mean() > 0.8  # sanity: the planted matches really are similar
    gate = adaptive_cosine_gate(np.concatenate([noise_cosines, signal_cosines]), dim)
    tau = gate["tau_hat"]
    assert 0.0 < tau < 1.0
    assert gate["k_star"] >= 30  # the planted matches are recovered
    # the gate keeps the planted similarities and rejects the pure-noise ones
    assert (signal_cosines > tau).mean() >= 0.5
    assert (noise_cosines > tau).mean() <= 0.02


def test_exponential_decay_half_life() -> None:
    decays = exponential_decay(np.array([0.0, 10.0, 20.0]), np.array([10.0, 10.0, 10.0]))
    assert decays[0] == pytest.approx(1.0)
    assert decays[1] == pytest.approx(0.5)
    assert decays[2] == pytest.approx(0.25)


def test_zscore_and_shrunk_covariance_are_psd() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 100.0])
    assert zscore(series).std(ddof=0) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(40, 5)) @ rng.normal(size=(5, 9))
    cov = shrunk_covariance(matrix)
    assert cov.shape == (9, 9)
    assert np.linalg.eigvalsh(cov).min() > 0
