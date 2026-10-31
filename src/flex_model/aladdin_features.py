"""Aladdin feature registry built from event-local vector kernels.

V5 replays PIT lakehouse bars and fact snapshots through these handlers, persists
their arena state, and blends their outputs into the production factor matrix.
"""

from __future__ import annotations

import numpy as np

from .engine import EventKind, FlexModelRegistry, ZeroAllocArena


flex = FlexModelRegistry()


@flex.handler(
    on=EventKind.CORP_ACTION_EX_DATE,
    markets={"INDIA", "US"},
    initial_value=1.0,
)
def cumulative_price_adjustment_factor(arena: ZeroAllocArena) -> np.ndarray:
    """Apply the vendor-normalized price adjustment multiplier for each ticker."""
    previous = arena.read("cumulative_price_adjustment_factor")
    action = arena.event_payload.get("price_adjustment_factor")
    if action is None:
        raise ValueError("CORP_ACTION_EX_DATE requires price_adjustment_factor")
    return previous * action


@flex.eval(
    reads=["cumulative_price_adjustment_factor"],
    markets={"INDIA", "US"},
    output=False,
)
def split_adjusted_close(arena: ZeroAllocArena) -> np.ndarray:
    """Apply the cumulative ex-date factor to raw close values supplied by feed."""
    raw_close = arena.event_payload.get("close")
    if raw_close is None:
        raise ValueError("EOD_BAR requires close")
    factor = arena.read("cumulative_price_adjustment_factor")
    out = np.empty_like(raw_close, dtype=np.float64)
    np.divide(raw_close, factor, out=out, where=factor != 0.0)
    out[factor == 0.0] = np.nan
    return out


@flex.handler(
    on=EventKind.EOD_BAR,
    reads=["split_adjusted_close"],
    markets={"INDIA", "US"},
    ring_window=20,
    output=True,
)
def f_kalman_velocity(arena: ZeroAllocArena) -> np.ndarray:
    """Streaming local-linear trend velocity with adaptive innovation scale."""
    price = np.maximum(arena.read("split_adjusted_close"), 1e-12)
    log_price = np.log(price)
    initialized = arena.ensure_state("kalman_initialized", dtype=np.bool_)
    level = arena.ensure_state("kalman_level")
    velocity = arena.ensure_state("kalman_velocity")
    variance = arena.ensure_state("kalman_variance", fill=1e-4)
    valid = np.isfinite(log_price)
    innovation = np.where(valid, log_price - (level + velocity), np.nan)
    ring = arena.push_ring("f_kalman_velocity", innovation)
    squared = np.square(ring)
    ring_count = np.sum(np.isfinite(squared), axis=1)
    innovation_var = np.divide(
        np.nansum(squared, axis=1), ring_count,
        out=np.full(len(arena.tickers), 1e-8), where=ring_count > 0,
    )
    scale2 = np.maximum(innovation_var, 1e-8)
    gain = np.clip(variance / (variance + scale2), 0.01, 0.99)
    next_level = level + velocity + gain * innovation
    next_velocity = velocity + 0.1 * gain * innovation
    new_names = valid & ~initialized
    continuing = valid & initialized
    level[new_names] = log_price[new_names]
    velocity[new_names] = 0.0
    level[continuing] = next_level[continuing]
    velocity[continuing] = next_velocity[continuing]
    variance[valid] = np.maximum((1.0 - gain[valid]) * variance[valid] + 1e-8, 1e-10)
    initialized[valid] = True
    output = arena.read("f_kalman_velocity").copy()
    output[valid] = velocity[valid] / np.sqrt(scale2[valid])
    return output


@flex.handler(
    on=EventKind.EOD_BAR,
    markets={"INDIA", "US"},
    ring_window=20,
    output=True,
)
def corwin_schultz_effective_spread(arena: ZeroAllocArena) -> np.ndarray:
    """Rolling high/low range proxy; calibrated CS estimation stays in V5 batch path."""
    high = arena.event_payload.get("high")
    low = arena.event_payload.get("low")
    if high is None or low is None:
        raise ValueError("EOD_BAR spread feature requires high and low")
    valid = np.isfinite(high) & np.isfinite(low) & (high > 0.0) & (low > 0.0)
    safe_high = np.maximum(np.nan_to_num(high, nan=1.0), 1e-12)
    safe_low = np.maximum(np.nan_to_num(low, nan=1.0), 1e-12)
    range_sq = np.where(valid, np.square(np.log(np.maximum(safe_high / safe_low, 1.0))), np.nan)
    rolling = arena.push_ring("corwin_schultz_effective_spread", range_sq)
    with np.errstate(invalid="ignore"):
        total = np.nansum(rolling, axis=1)
        count = np.sum(np.isfinite(rolling), axis=1)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    output = arena.read("corwin_schultz_effective_spread").copy()
    output[valid] = 0.25 * np.sqrt(np.maximum(mean[valid], 0.0))
    return output


@flex.handler(
    on=EventKind.XBRL_FUNDAMENTAL_FILING,
    markets={"INDIA", "US"},
)
def core_fundamental_state(arena: ZeroAllocArena) -> np.ndarray:
    """Updated only on a filing event; feature inputs are normalized upstream."""
    roic = arena.event_payload.get("roic_spread")
    fcf = arena.event_payload.get("fcf_yield")
    score = arena.event_payload.get("piotroski_f_score")
    if roic is None or fcf is None or score is None:
        raise ValueError("XBRL filing requires roic_spread, fcf_yield, and piotroski_f_score")
    candidate = 0.45 * roic + 0.35 * fcf + 0.20 * (score / 9.0)
    mask = np.asarray(
        arena.event_payload.get("core_event_mask", arena.event_payload.get("event_mask", np.ones(len(arena.tickers)))),
        dtype=bool,
    )
    output = arena.read("core_fundamental_state").copy()
    output[mask] = candidate[mask]
    return output


@flex.handler(
    on=EventKind.XBRL_FUNDAMENTAL_FILING,
    markets={"INDIA"},
)
def india_promoter_pledge_penalty(arena: ZeroAllocArena) -> np.ndarray:
    """India-only pledge and institutional-flow adjustment."""
    pledge = arena.event_payload.get("promoter_pledge_pct")
    flow = arena.event_payload.get("institutional_flow_delta")
    if pledge is None or flow is None:
        raise ValueError("India filing requires promoter_pledge_pct and institutional_flow_delta")
    candidate = 0.10 * flow - 0.005 * pledge
    mask = np.asarray(
        arena.event_payload.get("pledge_event_mask", arena.event_payload.get("event_mask", np.ones(len(arena.tickers)))),
        dtype=bool,
    )
    output = arena.read("india_promoter_pledge_penalty").copy()
    output[mask] = candidate[mask]
    return output


@flex.eval(
    reads=["core_fundamental_state", "india_promoter_pledge_penalty"],
    markets={"INDIA"},
)
def f_fundamentals_india(arena: ZeroAllocArena) -> np.ndarray:
    return arena.read("core_fundamental_state") + arena.read("india_promoter_pledge_penalty")


@flex.eval(reads=["core_fundamental_state"], markets={"US"})
def f_fundamentals_us(arena: ZeroAllocArena) -> np.ndarray:
    return arena.read("core_fundamental_state")


@flex.handler(
    on=EventKind.POLICY_OR_GEOPOL_SHOCK,
    markets={"INDIA", "US"},
    output=True,
)
def f_policy_geopol_accumulator(arena: ZeroAllocArena) -> np.ndarray:
    previous = arena.read("f_policy_geopol_accumulator")
    decay = arena.event_payload.get("decay_factor")
    shock = arena.event_payload.get("bge_projected_shock")
    if decay is None or shock is None:
        raise ValueError("policy event requires decay_factor and bge_projected_shock")
    return previous * decay + shock


@flex.handler(
    on=EventKind.PATENT_RND_UPDATE,
    markets={"INDIA", "US"},
    output=True,
)
def f_tech_patent_velocity(arena: ZeroAllocArena) -> np.ndarray:
    velocity = arena.event_payload.get("citation_velocity_log_ratio")
    if velocity is None:
        raise ValueError("patent event requires citation_velocity_log_ratio")
    mask = np.asarray(arena.event_payload.get("event_mask", np.ones(len(arena.tickers))), dtype=bool)
    output = arena.read("f_tech_patent_velocity").copy()
    output[mask] = velocity[mask]
    return output


@flex.handler(
    on=EventKind.US_OVERNIGHT_SPILLOVER,
    markets={"INDIA"},
    output=True,
)
def f_us_to_india_spillover(arena: ZeroAllocArena) -> np.ndarray:
    projected = arena.event_payload.get("projected_us_spillover")
    if projected is None:
        raise ValueError("US overnight spillover event requires projected_us_spillover")
    return projected
