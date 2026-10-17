"""Adaptive Kalman trend filter with *learned* noise covariances.

State-space model per stock (2-state local linear trend on log price):

    s_t = F s_{t-1} + w_t,   w ~ N(0, Q)        F = [[1, 1], [0, 1]]
    y_t = H s_t + v_t,       v ~ N(0, R)        H = [1, 0],  y = log price

Neither Q nor R is hardcoded. R is estimated in closed form from the data: for a
local linear trend observed with i.i.d. noise, the lag-1 autocovariance of the
first differences of the log price equals -R exactly (level increments cancel at
lag 1 for a locally constant slope), so

    R_hat = max( -gamma_1(delta log p) , eps )

Q is estimated online by correction-covariance matching (Myers & Tapley 1976):

    Q_hat_t = max( (1/W) sum_{k<W} (K nu)(K nu)' + P_{t|t} - F P_{t-1|t-1} F' , eps )

with the process-noise estimate projected onto the PSD cone by eigenvalue clipping
so finite-sample noise cannot destroy the filter. Estimating R in closed form
(rather than from innovations) is what makes the pair identifiable: innovation
covariance matching alone admits any split of the total innovation variance
between Q and R and would drive R to the floor.
"""

from __future__ import annotations

import numpy as np

F_MATRIX = np.array([[1.0, 1.0], [0.0, 1.0]])
H_MATRIX = np.array([[1.0, 0.0]])


def observation_noise_from_autocovariance(diffs: np.ndarray) -> float:
    """Closed-form observation-noise estimate from first differences.

    For a local linear trend observed with i.i.d. measurement noise, the lag-1
    autocovariance of the first differences equals ``-R`` exactly (the level
    increments cancel at lag one when the slope is locally constant), so
    ``R_hat = max(-gamma_1, 0)`` -- fully data-driven, never configured.
    """
    values = np.asarray(diffs, dtype=float)
    if values.size < 3:
        return 0.0
    centered = values - float(np.mean(values))
    gamma_1 = float(np.dot(centered[1:], centered[:-1])) / float(centered.size - 1)
    return max(-gamma_1, 0.0)


class AdaptiveKalmanFilter:
    """Innovation-covariance-matching Kalman filter (no fixed Q/R)."""

    def __init__(
        self,
        *,
        min_observations: int = 10,
        window_sqrt_scale: float = 1.0,
        min_window: int = 10,
        init_process_frac: tuple[float, float] = (0.10, 0.01),
        innovation_floor: float = 1e-10,
        eigenvalue_floor: float = 1e-10,
    ) -> None:
        self.min_observations = int(min_observations)
        self.window_sqrt_scale = float(window_sqrt_scale)
        self.min_window = int(min_window)
        self.init_process_frac = init_process_frac
        self.innovation_floor = float(innovation_floor)
        self.eigenvalue_floor = float(eigenvalue_floor)

    @classmethod
    def from_config(cls, config) -> "AdaptiveKalmanFilter":
        spec = config.e("kalman", {}) or {}
        frac = tuple(spec.get("init_process_frac", [0.10, 0.01]))
        return cls(
            min_observations=int(spec.get("min_observations", 10)),
            window_sqrt_scale=float(spec.get("window_sqrt_scale", 1.0)),
            min_window=int(spec.get("min_window", 10)),
            init_process_frac=(float(frac[0]), float(frac[1])),
            innovation_floor=float(spec.get("innovation_floor", 1e-10)),
            eigenvalue_floor=float(spec.get("eigenvalue_floor", 1e-10)),
        )

    def filter(self, prices: np.ndarray) -> dict[str, np.ndarray | float]:
        values = np.asarray(prices, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        n = values.size
        if n < 3:
            zeros = np.zeros(max(n, 0))
            return {"level": values, "velocity": zeros, "r_std": 0.0, "q_diag": zeros, "n_obs": n}

        log_price = np.log(values)
        diffs = np.diff(log_price)
        init_var = float(np.var(diffs[: min(20, diffs.size)])) + 1e-12

        state = np.array([log_price[0], float(np.mean(diffs[: min(5, diffs.size)]))])
        covariance = np.eye(2) * init_var
        Q = np.diag([init_var * self.init_process_frac[0], init_var * self.init_process_frac[1]])
        # Closed-form, model-consistent observation-noise estimate.
        R = max(observation_noise_from_autocovariance(diffs), self.innovation_floor)

        window = max(self.min_window, int(np.sqrt(n) * self.window_sqrt_scale))
        eye = np.eye(2)
        level = np.zeros(n)
        velocity = np.zeros(n)
        # Ring buffers for the Myers-Tapley process-noise estimate.
        corrections: list[np.ndarray] = []
        posterior_cov: list[np.ndarray] = []
        previous_cov: list[np.ndarray] = []
        q_diag_last = np.diag(Q).copy()
        r_last = float(R)

        for t in range(n):
            state_pred = F_MATRIX @ state
            cov_pred = F_MATRIX @ covariance @ F_MATRIX.T + Q
            pred_s = float((H_MATRIX @ cov_pred @ H_MATRIX.T)[0, 0])

            innovation = float(log_price[t] - (H_MATRIX @ state_pred)[0])

            if len(corrections) >= window:
                # Myers-Tapley process noise (exact in expectation):
                #   Q = E[(K nu)(K nu)'] + E[P^+] - F E[P^+_{t-1}] F'
                recent_corr = np.asarray(corrections[-window:], dtype=float)
                mean_post = np.mean(np.asarray(posterior_cov[-window:]), axis=0)
                mean_prev = np.mean(np.asarray(previous_cov[-window:]), axis=0)
                raw_q = (recent_corr.T @ recent_corr) / float(window)
                raw_q = raw_q + mean_post - F_MATRIX @ mean_prev @ F_MATRIX.T
                eigvals, eigvecs = np.linalg.eigh((raw_q + raw_q.T) / 2.0)
                Q = eigvecs @ np.diag(np.maximum(eigvals, self.eigenvalue_floor)) @ eigvecs.T

            s_t = pred_s + R
            gain = (cov_pred @ H_MATRIX.T) / max(s_t, self.innovation_floor)

            correction = gain.flatten() * innovation
            state = state_pred + correction
            posterior = (eye - gain @ H_MATRIX) @ cov_pred

            corrections.append(correction.copy())
            posterior_cov.append(posterior.copy())
            previous_cov.append(covariance.copy())
            covariance = posterior
            q_diag_last = np.diag(Q).copy()
            r_last = float(R)

            level[t] = state[0]
            velocity[t] = state[1]

        return {
            "level": np.exp(level),
            "velocity": velocity,
            "r_std": float(np.sqrt(max(r_last, 0.0))),
            "q_diag": q_diag_last,
            "n_obs": n,
        }

    def trend_score(self, prices: np.ndarray) -> float:
        """Risk-scaled trend: Kalman velocity divided by observation noise scale."""
        result = self.filter(prices)
        velocity = np.asarray(result["velocity"], dtype=float)
        if velocity.size < 2:
            return 0.0
        scale = max(float(result["r_std"]), 1e-8)
        recent = velocity[-1]
        return float(np.sign(recent) * min(abs(recent) / scale, 25.0))
