"""Adaptive Kalman: Q and R must be learned, not configured."""

from __future__ import annotations

import numpy as np

from src.filtering.kalman_smoother import AdaptiveKalmanFilter
from src.filtering.self_calibrating_engine import SelfCalibratingQuantEngine


def test_velocity_tracks_a_known_slope() -> None:
    prices = 100.0 * np.exp(np.cumsum(np.full(120, 0.01)))
    result = AdaptiveKalmanFilter().filter(prices)
    velocity = np.asarray(result["velocity"])
    assert velocity[-1] > 0.005
    assert abs(velocity[-1] - 0.01) < 0.005
    assert 0.0 < float(result["r_std"]) < 0.05
    assert np.all(np.asarray(result["q_diag"]) > 0)


def test_noise_estimates_are_data_driven() -> None:
    rng = np.random.default_rng(4)
    steps = 200
    # Smooth drift + i.i.d. observation noise: the model-consistent setup in which
    # R is identifiable (a pure random walk has no observation noise to recover).
    trend = np.cumsum(np.full(steps, 0.002))
    calm = 100.0 * np.exp(trend + rng.normal(0.0, 0.002, steps))
    wild = 100.0 * np.exp(trend + rng.normal(0.0, 0.05, steps))
    calm_r = float(AdaptiveKalmanFilter().filter(calm)["r_std"])
    wild_r = float(AdaptiveKalmanFilter().filter(wild)["r_std"])
    # observation noise is *estimated*: the noisy series must yield a larger R
    assert wild_r > 3.0 * calm_r
    assert wild_r > 0.01
    assert calm_r < 0.01


def test_trend_score_signs() -> None:
    engine = SelfCalibratingQuantEngine(_stub_config())
    up = 50.0 * np.exp(np.cumsum(np.full(90, 0.004)))
    down = 50.0 * np.exp(np.cumsum(np.full(90, -0.004)))
    assert engine.kalman.trend_score(up) > 0
    assert engine.kalman.trend_score(down) < 0


def test_short_series_is_safe() -> None:
    result = AdaptiveKalmanFilter().filter(np.array([10.0, 11.0]))
    assert len(np.asarray(result["velocity"])) == 2
    assert np.isfinite(np.asarray(result["level"])).all()


def _stub_config():
    from src.config import Config

    return Config.load()
