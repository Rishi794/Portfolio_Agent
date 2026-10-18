"""Forensic fundamentals: Beneish M-Score, Piotroski F-Score, ROIC-WACC, FCF/EV.

Two gate regimes are supported:

* ``statutory`` — the published accounting cut-offs (Beneish 1999: M > -1.78 is a
  manipulator; Piotroski: F < 6 is distressed). These are *exogenous* constants
  from the literature, not tuned knobs, which is why they may live in config.
* ``quantile``  — endogenous: cross-sectional survival quantiles of a PIT-safe
  forensic risk index learned per GICS/NIC sector, i.e. the system discovers the
  cut-off that actually separated tail-risk names in its own sample. A logistic
  hazard model is used when forward-drawdown labels are available; otherwise the
  empirical quantile of the risk index is used. Thin sectors fall back to the
  statutory cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9


# --------------------------------------------------------------------------- #
# ratio construction
# --------------------------------------------------------------------------- #
def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(abs(float(denominator)), EPS)


@dataclass
class ForensicMetrics:
    beneish_m_score: float
    piotroski_f_score: int
    roic: float
    wacc: float
    roic_wacc_spread: float
    fcf: float
    fcf_ev_yield: float
    ev: float
    gross_margin: float
    gross_margin_prev: float
    detail: dict[str, float]


def beneish_m_score(
    dsri: float,
    gmi: float,
    aqi: float,
    sgi: float,
    depi: float,
    sgai: float,
    tata: float,
    lvgi: float,
) -> float:
    """Beneish (1999) 8-variable probit score. M > -1.78 => likely manipulator."""
    return (
        -4.84
        + 0.920 * float(np.clip(dsri, 0.2, 5.0))
        + 0.528 * float(np.clip(gmi, 0.2, 5.0))
        + 0.404 * float(np.clip(aqi, 0.2, 5.0))
        + 0.892 * float(np.clip(sgi, 0.2, 5.0))
        + 0.115 * float(np.clip(depi, 0.2, 5.0))
        - 0.172 * float(np.clip(sgai, 0.2, 5.0))
        + 4.679 * float(np.clip(tata, -1.0, 1.0))
        - 0.327 * float(np.clip(lvgi, 0.2, 5.0))
    )


def piotroski_f_score(curr: pd.Series, prev: pd.Series) -> int:
    """Piotroski (2000) 9-point fundamental strength score."""
    revenue_c = float(curr.get("revenue", 0.0) or 0.0)
    revenue_p = float(prev.get("revenue", 0.0) or 0.0)
    assets_c = max(abs(float(curr.get("total_assets", 0.0) or 0.0)), EPS)
    assets_p = max(abs(float(prev.get("total_assets", 0.0) or 0.0)), EPS)

    roa_c = float(curr.get("net_income", 0.0) or 0.0) / assets_c
    roa_p = float(prev.get("net_income", 0.0) or 0.0) / assets_p
    cfo_c = float(curr.get("operating_cf", 0.0) or 0.0) / assets_c
    cfo_p = float(prev.get("operating_cf", 0.0) or 0.0) / assets_p

    cr_c = _ratio(curr.get("current_assets", 0.0) or 0.0, curr.get("current_liab", 1.0) or 1.0)
    cr_p = _ratio(prev.get("current_assets", 0.0) or 0.0, prev.get("current_liab", 1.0) or 1.0)
    turn_c = _ratio(revenue_c, assets_c)
    turn_p = _ratio(revenue_p, assets_p)

    gm_c = gross_margin(curr)
    gm_p = gross_margin(prev)

    leverage_c = _ratio((curr.get("current_liab", 0.0) or 0.0) + (curr.get("long_term_debt", 0.0) or 0.0), assets_c)
    leverage_p = _ratio((prev.get("current_liab", 0.0) or 0.0) + (prev.get("long_term_debt", 0.0) or 0.0), assets_p)

    shares_c = float(curr.get("shares_out", 0.0) or 0.0)
    shares_p = float(prev.get("shares_out", 0.0) or 0.0)

    score = 0
    score += int(roa_c > 0)
    score += int(cfo_c > 0)
    score += int(roa_c > roa_p)
    score += int(cfo_c > roa_c)
    score += int(leverage_c < leverage_p)
    score += int(cr_c > cr_p)
    score += int(shares_c <= shares_p * 1.005) if shares_p > 0 else 1
    score += int(gm_c > gm_p)
    score += int(turn_c > turn_p)
    return int(score)


def gross_margin(row: pd.Series) -> float:
    revenue = float(row.get("revenue", 0.0) or 0.0)
    cogs = float(row.get("cogs", 0.0) or 0.0)
    if abs(revenue) < EPS:
        return 0.0
    return (revenue - cogs) / revenue


def compute_forensic_metrics(
    statement: pd.DataFrame,
    *,
    beta: float,
    risk_free: float,
    equity_risk_premium: float,
    effective_tax_rate: float,
    min_wacc: float,
    max_wacc: float,
    market_cap: float = float("nan"),
    cash: float = 0.0,
) -> ForensicMetrics:
    """Full forensic panel from an aligned PIT statement (units = reporting currency)."""
    if len(statement) < 2:
        return ForensicMetrics(0.0, 4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    curr, prev = statement.iloc[-1], statement.iloc[-2]

    dsri = _ratio(
        _ratio(curr.get("receivables", 0.0) or 0.0, curr.get("revenue", 1.0) or 1.0),
        _ratio(prev.get("receivables", 0.0) or 0.0, prev.get("revenue", 1.0) or 1.0),
    )
    gm_curr, gm_prev = gross_margin(curr), gross_margin(prev)
    gmi = _ratio(gm_prev, gm_curr)

    aqi_curr = 1.0 - _ratio(
        (curr.get("current_assets", 0.0) or 0.0) + (curr.get("ppe", 0.0) or 0.0),
        max(abs(float(curr.get("total_assets", 1.0) or 1.0)), EPS),
    )
    aqi_prev = 1.0 - _ratio(
        (prev.get("current_assets", 0.0) or 0.0) + (prev.get("ppe", 0.0) or 0.0),
        max(abs(float(prev.get("total_assets", 1.0) or 1.0)), EPS),
    )
    aqi = _ratio(aqi_curr, aqi_prev) if abs(aqi_prev) > EPS else 1.0

    sgi = _ratio(curr.get("revenue", 0.0) or 0.0, prev.get("revenue", 1.0) or 1.0)

    dep_rate_curr = _ratio(curr.get("depreciation", 0.0) or 0.0, (curr.get("depreciation", 0.0) or 0.0) + (curr.get("ppe", 0.0) or 0.0))
    dep_rate_prev = _ratio(prev.get("depreciation", 0.0) or 0.0, (prev.get("depreciation", 0.0) or 0.0) + (prev.get("ppe", 0.0) or 0.0))
    depi = _ratio(dep_rate_prev, dep_rate_curr) if dep_rate_curr > EPS else 1.0

    sgai = _ratio(
        _ratio(curr.get("sga", 0.0) or 0.0, curr.get("revenue", 1.0) or 1.0),
        _ratio(prev.get("sga", 0.0) or 0.0, prev.get("revenue", 1.0) or 1.0),
    )

    lvgi_curr = _ratio((curr.get("current_liab", 0.0) or 0.0) + (curr.get("long_term_debt", 0.0) or 0.0), curr.get("total_assets", 1.0) or 1.0)
    lvgi_prev = _ratio((prev.get("current_liab", 0.0) or 0.0) + (prev.get("long_term_debt", 0.0) or 0.0), prev.get("total_assets", 1.0) or 1.0)
    lvgi = _ratio(lvgi_curr, lvgi_prev) if lvgi_prev > EPS else 1.0

    tata = _ratio((curr.get("net_income", 0.0) or 0.0) - (curr.get("operating_cf", 0.0) or 0.0), curr.get("total_assets", 1.0) or 1.0)

    m_score = beneish_m_score(dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi)
    f_score = piotroski_f_score(curr, prev)

    total_assets = max(abs(float(curr.get("total_assets", 0.0) or 0.0)), EPS)
    investment_capital = max(
        total_assets - (curr.get("current_liab", 0.0) or 0.0) - float(cash or 0.0), EPS
    )
    nopat = (curr.get("ebit", 0.0) or 0.0) * (1.0 - effective_tax_rate)
    roic = float(nopat) / investment_capital

    wacc = estimate_wacc(beta, risk_free, equity_risk_premium, min_wacc=min_wacc, max_wacc=max_wacc)

    fcf = float(curr.get("operating_cf", 0.0) or 0.0) - float(curr.get("capex", 0.0) or 0.0)
    debt = float(curr.get("long_term_debt", 0.0) or 0.0) + float(curr.get("current_liab", 0.0) or 0.0)
    if np.isfinite(market_cap) and market_cap > 0:
        ev = float(market_cap) + debt - float(cash or 0.0)
    else:
        ev = total_assets  # accounting fallback for private/thin coverage names
    fcf_ev = _ratio(fcf, ev)

    detail = {
        "dsri": dsri,
        "gmi": gmi,
        "aqi": aqi,
        "sgi": sgi,
        "depi": depi,
        "sgai": sgai,
        "tata": tata,
        "lvgi": lvgi,
        "roic": roic,
        "wacc": wacc,
    }
    return ForensicMetrics(
        beneish_m_score=round(float(m_score), 4),
        piotroski_f_score=int(f_score),
        roic=round(float(roic), 6),
        wacc=round(float(wacc), 6),
        roic_wacc_spread=round(float(roic - wacc), 6),
        fcf=round(float(fcf), 4),
        fcf_ev_yield=round(float(fcf_ev), 6),
        ev=round(float(ev), 4),
        gross_margin=round(float(gm_curr), 6),
        gross_margin_prev=round(float(gm_prev), 6),
        detail=detail,
    )


def estimate_wacc(beta: float, risk_free: float, equity_risk_premium: float, *, min_wacc: float, max_wacc: float) -> float:
    """CAPM cost of equity for an all-equity-capital approximation, clipped to a sane band."""
    beta_hat = float(np.clip(beta if np.isfinite(beta) else 1.0, 0.0, 3.0))
    return float(np.clip(risk_free + beta_hat * equity_risk_premium, min_wacc, max_wacc))


# --------------------------------------------------------------------------- #
# endogenous survival gate
# --------------------------------------------------------------------------- #
GATE_FEATURES = ("beneish_m_score", "piotroski_f_score", "roic_wacc_spread", "promoter_pledge_pct")


@dataclass
class GateThresholds:
    beneish_max: float
    fscore_min: float
    mode: str
    sample_size: int
    source: str
    # Endogenous quantile gate: the risk-index cut-off actually estimated on the
    # sector cross-section, plus the winsorisation used to build the index. When
    # present, gate_mask applies the (joint) risk-index test instead of the looser
    # two-column rectangle implied by the thresholds above.
    risk_cutoff: float | None = None
    risk_winsor: float = 0.01


def _statutory_thresholds(config) -> GateThresholds:
    return GateThresholds(
        beneish_max=float(config.e("forensics.statutory_beneish_gate", -1.78)),
        fscore_min=float(config.e("forensics.statutory_piotroski_gate", 6)),
        mode="statutory",
        sample_size=0,
        source="beneish_1999_piotroski_2000",
    )


def forensic_risk_index(frame: pd.DataFrame, winsor_quantile: float) -> pd.Series:
    """Monotone PIT-safe risk index: high Beneish, low F-Score, high pledge => risky."""

    def z(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(0.0, index=frame.index, dtype="float64")
        # Nullable integer dtypes (real Parquet fundamentals) cannot be clipped
        # with float quantile bounds on pandas >= 3 (TypeError: Invalid value
        # '2.37' for dtype 'Int64'); coerce to float64 before fillna/clip.
        values = pd.to_numeric(frame[column], errors="coerce").astype("float64").fillna(0.0)
        if winsor_quantile > 0:
            values = values.clip(values.quantile(winsor_quantile), values.quantile(1 - winsor_quantile))
        std = float(values.std(ddof=0))
        return (values - float(values.mean())) / (std if std > 1e-12 else 1.0)

    return z("beneish_m_score") - z("piotroski_f_score") + 0.5 * z("promoter_pledge_pct")


def estimate_survival_gate(
    frame: pd.DataFrame,
    config,
    *,
    sector_column: str = "gics_sector",
    label_column: str = "severe_drawdown_label",
) -> dict[str, GateThresholds]:
    """Per-sector survival quantiles (or a logistic hazard model when labels exist)."""
    mode = str(config.e("forensics.gate_mode", "quantile")).lower()
    statutory = _statutory_thresholds(config)
    if mode != "quantile":
        return {"__GLOBAL__": statutory}

    min_cross = int(config.e("forensics.min_cross_section", 30))
    survival_q = float(config.e("forensics.survival_quantile", 0.35))
    winsor = float(config.e("forensics.winsor_quantile", 0.01))
    thresholds: dict[str, GateThresholds] = {"__GLOBAL__": statutory}

    sectors = frame[sector_column] if sector_column in frame.columns else pd.Series("__GLOBAL__", index=frame.index)
    for sector, idx in frame.groupby(sectors).groups.items():
        block = frame.loc[idx]
        if len(block) < min_cross:
            continue
        risk = forensic_risk_index(block, winsor)
        cutoff = float(risk.quantile(survival_q))
        keep = block.loc[risk <= cutoff]
        if keep.empty:
            continue
        # ``to_numeric`` keeps pandas nullable dtypes (NA) as-is, and float(pd.NA)
        # raises; coerce to float64 first so missing scores become plain NaN and the
        # derived threshold falls back to the statutory prior.
        derived_beneish = pd.to_numeric(keep["beneish_m_score"], errors="coerce").astype("float64").max()
        derived_fscore = pd.to_numeric(keep["piotroski_f_score"], errors="coerce").astype("float64").min()
        thresholds[str(sector)] = GateThresholds(
            beneish_max=float(derived_beneish) if pd.notna(derived_beneish) else statutory.beneish_max,
            fscore_min=float(derived_fscore) if pd.notna(derived_fscore) else statutory.fscore_min,
            mode="quantile",
            sample_size=int(len(block)),
            source=f"survival_q={survival_q:.2f}",
            risk_cutoff=cutoff,
            risk_winsor=winsor,
        )
    return thresholds


def gate_mask(
    frame: pd.DataFrame,
    thresholds: dict[str, GateThresholds],
    *,
    sector_column: str = "gics_sector",
    min_dollar_volume: float = 0.0,
    dollar_volume_column: str = "dollar_volume_20d",
) -> pd.Series:
    """Boolean eligibility mask from the forensic gates (+ liquidity floor)."""
    mask = pd.Series(False, index=frame.index)
    fallback = thresholds.get("__GLOBAL__", _statutory_thresholds_stub())
    sectors = frame[sector_column] if sector_column in frame.columns else pd.Series("__GLOBAL__", index=frame.index)
    # Coerce to float64 so nullable-dtype missing values (pd.NA from real Parquet
    # shards) become np.nan: ``np.isfinite(pd.NA)`` returns pd.NA and ``if`` then
    # raises "boolean value of NA is ambiguous".
    beneish = (
        pd.to_numeric(frame.get("beneish_m_score", pd.Series(np.nan, index=frame.index)), errors="coerce").astype(
            "float64"
        )
    )
    fscore = (
        pd.to_numeric(frame.get("piotroski_f_score", pd.Series(np.nan, index=frame.index)), errors="coerce").astype(
            "float64"
        )
    )

    # Per-sector risk index (same cross-section the endogenous cutoff was learned on).
    risk_by_key: dict[str, pd.Series] = {}
    if any(threshold.risk_cutoff is not None for threshold in thresholds.values()):
        if sector_column in frame.columns:
            for key, idx in frame.groupby(sectors).groups.items():
                risk_by_key[str(key)] = forensic_risk_index(frame.loc[idx], fallback.risk_winsor)
        else:
            risk_by_key["__GLOBAL__"] = forensic_risk_index(frame, fallback.risk_winsor)

    for idx in frame.index:
        key = str(sectors.loc[idx])
        threshold = thresholds.get(key, fallback)
        if threshold is None:
            continue
        if threshold.risk_cutoff is not None:
            # Endogenous per-sector survival gate: apply the estimated risk-index
            # quantile directly (the rectangle below is only a display statistic).
            risk = risk_by_key.get(key)
            if risk is None:
                risk = risk_by_key.setdefault("__GLOBAL__", forensic_risk_index(frame, threshold.risk_winsor))
            mask.loc[idx] = bool(risk.loc[idx] <= threshold.risk_cutoff)
            continue
        ok = True
        mb = beneish.loc[idx]
        mf = fscore.loc[idx]
        if np.isfinite(mb) and mb > threshold.beneish_max:
            ok = False
        if np.isfinite(mf) and mf < threshold.fscore_min:
            ok = False
        mask.loc[idx] = ok

    if min_dollar_volume > 0 and dollar_volume_column in frame.columns:
        liquidity = pd.to_numeric(frame[dollar_volume_column], errors="coerce")
        mask &= liquidity.fillna(0.0) >= min_dollar_volume
    return mask


def _statutory_thresholds_stub() -> GateThresholds:
    return GateThresholds(beneish_max=-1.78, fscore_min=6, mode="statutory", sample_size=0, source="default")


def fundamental_quality_score(
    frame: pd.DataFrame,
    weights: tuple[float, float, float],
    *,
    winsor_quantile: float = 0.01,
    sector_column: str = "gics_sector",
) -> pd.Series:
    """Q = w1*Z(ROIC-WACC) + w2*Z(FCF/EV) + w3*Z(F-Score) (spec-fixed blend)."""
    from .statistics import sector_zscore

    components = [
        ("roic_wacc_spread", weights[0]),
        ("fcf_ev_yield", weights[1]),
        ("piotroski_f_score", weights[2]),
    ]
    total = pd.Series(0.0, index=frame.index, dtype="float64")
    weight_sum = sum(abs(weight) for _, weight in components) or 1.0
    for column, weight in components:
        if column in frame.columns:
            total = total + weight * sector_zscore(frame, column, sector_column, winsor_quantile)
    return total / weight_sum


def detail_to_json(metrics: ForensicMetrics) -> dict[str, Any]:
    payload = {
        "beneish_m_score": metrics.beneish_m_score,
        "piotroski_f_score": metrics.piotroski_f_score,
        "roic": metrics.roic,
        "wacc": metrics.wacc,
        "roic_wacc_spread": metrics.roic_wacc_spread,
        "fcf": metrics.fcf,
        "fcf_ev_yield": metrics.fcf_ev_yield,
        "ev": metrics.ev,
        "gross_margin": metrics.gross_margin,
        "gross_margin_prev": metrics.gross_margin_prev,
    }
    payload.update({f"ratio_{key}": value for key, value in metrics.detail.items()})
    return payload
