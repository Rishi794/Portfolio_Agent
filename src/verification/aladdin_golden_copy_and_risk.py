"""Open-Aladdin verification core: golden-copy quorum + Decimal IBOR + Euler risk.

Three institutional verification layers, all deterministic and offline-testable:

1. ``GoldenCopyConsensusEngine`` — 2-of-N median quorum (within
   ``aladdin.quorum.max_source_divergence_bps``) over independent price feeds plus
   a Hampel/MAD flash-crash quarantine, stamped with a SHA-256 proof per bar.
   With per-feed error ``p``, two independent feeds sharing the *same* bad print
   has probability ~``3p^2`` (binomial tail); the divergence gate is evaluated
   first and takes precedence over the MAD flag (a vendor disagreement is
   stronger evidence than an unexplained move that multiple vendors printed).
2. ``DoubleEntryIBORLedger`` — 28-digit fixed-point ``Decimal`` double-entry
   bookkeeping with the invariant ``Assets == Liabilities + Equity + Income -
   Expenses`` and a SHA-256 hash chain over account states. Also provides the
   reconciliation helper for the float ledger snapshots written by the V4 layer.
3. ``AladdinRiskAndScenarioEngine`` — exact Euler volatility attribution
   (``MCTR_i = (Sigma w)_i / sigma_p``, ``CCTR_i = w_i * MCTR_i``,
   ``sum(CCTR) == sigma_p``), fat-tailed Student-t ``CVaR_99``, market-driven
   scenario stress propagation (``B = Sigma_RF Sigma_F^{-1}``) and the pre-trade
   compliance gate (quarantine blocks + ADV participation cap).

Every policy/scenario constant is read from ``config/aladdin.yaml``; every
statistical quantity (covariance, tail index, MAD scale) is estimated at runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, getcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.covariance import LedoitWolf

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

getcontext().prec = 28  # institutional fixed-point precision

ZERO_DEC = Decimal("0.00000000")
QUANT_8 = Decimal("0.00000001")
QUANT_2 = Decimal("0.01")

# Closed-form normal-consistency factor for the median absolute deviation.
MAD_CONSISTENCY = 1.4826


def to_dec(val: float | int | str | Decimal) -> Decimal:
    """Exact decimal from the value's shortest repr, quantized to 8dp (banker's)."""
    if isinstance(val, Decimal):
        return val.quantize(QUANT_8, rounding=ROUND_HALF_EVEN)
    return Decimal(str(val)).quantize(QUANT_8, rounding=ROUND_HALF_EVEN)


def _config_default(config: "Config | None", path: str, default: Any) -> Any:
    if config is None:
        return default
    value = config.get(path, default)
    return default if value is None else value


# =============================================================================
# 1. Multi-source golden-copy consensus (Byzantine 2-of-N quorum + Hampel MAD)
# =============================================================================
@dataclass(frozen=True)
class GoldenCopyRecord:
    trade_date: str
    ticker: str
    market: str
    golden_close: Decimal
    sources_agreed: int
    max_dispersion_bps: float
    mad_z_score: float
    status: str  # VERIFIED_GOLDEN | QUARANTINED_DIVERGENCE | QUARANTINED_MAD_SPIKE
    sha256_proof: str


class GoldenCopyConsensusEngine:
    """2-of-N median quorum over independent price feeds.

    With per-feed error probability ``p`` the probability that two independent
    feeds share the *same* bad print is ~``3p^2`` (binomial tail, N=3): that is
    the Byzantine robustness argument for never trusting a single vendor. The
    divergence gate runs first; the MAD flash-crash flag only downgrades bars
    that were otherwise golden (a vendor disagreement is stronger evidence).
    """

    def __init__(
        self,
        config: "Config | None" = None,
        *,
        max_source_divergence_bps: float | None = None,
        mad_sigma_threshold: float | None = None,
        min_sources: int | None = None,
    ) -> None:
        self.max_div_bps = float(
            max_source_divergence_bps
            if max_source_divergence_bps is not None
            else _config_default(config, "aladdin.quorum.max_source_divergence_bps", 5.0)
        )
        self.mad_threshold = float(
            mad_sigma_threshold
            if mad_sigma_threshold is not None
            else _config_default(config, "aladdin.quorum.mad_sigma_threshold", 8.0)
        )
        self.min_sources = int(
            min_sources
            if min_sources is not None
            else _config_default(config, "aladdin.quorum.min_sources", 2)
        )

    def build_golden_copy_bar(
        self,
        trade_date: str,
        ticker: str,
        market: str,
        source_prices: dict[str, float],
        trailing_golden_closes: np.ndarray | list[float],
        has_corporate_action_today: bool = False,
    ) -> GoldenCopyRecord:
        valid = {
            src: float(price)
            for src, price in source_prices.items()
            if price is not None and np.isfinite(float(price)) and float(price) > 0.0
        }
        if len(valid) < self.min_sources:
            raise ValueError(
                f"[GOLDEN_COPY_HALT] {ticker}: {len(valid)} usable sources < min_sources={self.min_sources}"
            )

        values = np.array(list(valid.values()), dtype=np.float64)
        median_price = float(np.median(values))
        deviations_bps = np.abs(values - median_price) / median_price * 10_000.0
        consensus_mask = deviations_bps <= self.max_div_bps
        agreed_count = int(np.sum(consensus_mask))

        if agreed_count >= 2:
            golden_value = float(np.mean(values[consensus_mask]))
            max_dispersion = float(np.max(deviations_bps[consensus_mask]))
            status = "VERIFIED_GOLDEN"
        else:
            golden_value = median_price
            max_dispersion = float(np.max(deviations_bps))
            status = "QUARANTINED_DIVERGENCE"

        mad_z = 0.0
        trailing = np.asarray(trailing_golden_closes, dtype=np.float64)
        if status == "VERIFIED_GOLDEN" and len(trailing) >= 20 and not has_corporate_action_today:
            history = np.append(trailing[-21:], golden_value)
            log_returns = np.diff(np.log(np.maximum(history, 1e-8)))
            median_return = float(np.median(log_returns[:-1]))
            mad = float(np.median(np.abs(log_returns[:-1] - median_return)))
            robust_sigma = max(MAD_CONSISTENCY * mad, 1e-4)
            mad_z = float(abs(log_returns[-1] - median_return) / robust_sigma)
            if mad_z > self.mad_threshold:
                status = "QUARANTINED_MAD_SPIKE"

        golden_decimal = to_dec(round(golden_value, 4))
        payload = f"{trade_date}|{ticker}|{market}|{golden_decimal}|{agreed_count}|{status}"
        proof = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        return GoldenCopyRecord(
            trade_date=str(trade_date),
            ticker=ticker,
            market=market,
            golden_close=golden_decimal,
            sources_agreed=agreed_count,
            max_dispersion_bps=round(max_dispersion, 3),
            mad_z_score=round(mad_z, 3),
            status=status,
            sha256_proof=proof,
        )

    @staticmethod
    def quorum_probability(p: float, n: int = 3) -> float:
        """P(>= 2 of ``n`` feeds share the same bad print) — the Byzantine tail."""
        from math import comb

        return float(sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(2, n + 1)))


# =============================================================================
# 2. Fixed-point Decimal double-entry IBOR (zero rounding drift)
# =============================================================================
class DoubleEntryIBORLedger:
    """Institutional Investment Book of Record with balanced journal legs.

    Accounts (all ``Decimal``, 28-digit precision)::

        ASSET:CASH | ASSET:EQUITY_COST_BASIS | EQUITY:INITIAL_CAPITAL
        EQUITY:REALIZED_PNL | INCOME:DIVIDENDS | EXPENSE:TAXES_AND_FEES

    Invariant: ``ASSET:CASH + ASSET:EQUITY_COST_BASIS == EQUITY:INITIAL_CAPITAL
    + EQUITY:REALIZED_PNL + INCOME:DIVIDENDS - EXPENSE:TAXES_AND_FEES`` to 8dp.
    """

    def __init__(self, market: str, initial_cash: str = "1000000.00") -> None:
        self.market = market
        self.accounts: dict[str, Decimal] = {
            "ASSET:CASH": to_dec(initial_cash),
            "ASSET:EQUITY_COST_BASIS": ZERO_DEC,
            "EQUITY:INITIAL_CAPITAL": to_dec(initial_cash),
            "EQUITY:REALIZED_PNL": ZERO_DEC,
            "INCOME:DIVIDENDS": ZERO_DEC,
            "EXPENSE:TAXES_AND_FEES": ZERO_DEC,
        }
        self.positions: dict[str, dict[str, Decimal | int]] = {}
        self.chain_hash: str = hashlib.sha256(b"GENESIS_ALADDIN_V5").hexdigest()

    # ------------------------------------------------------------- invariant --
    def verify_accounting_invariant(self) -> bool:
        """Raise unless the double-entry invariant holds to 8 decimals."""
        total_assets = self.accounts["ASSET:CASH"] + self.accounts["ASSET:EQUITY_COST_BASIS"]
        total_equity = (
            self.accounts["EQUITY:INITIAL_CAPITAL"]
            + self.accounts["EQUITY:REALIZED_PNL"]
            + self.accounts["INCOME:DIVIDENDS"]
            - self.accounts["EXPENSE:TAXES_AND_FEES"]
        )
        discrepancy = abs(total_assets - total_equity)
        if discrepancy > ZERO_DEC:
            raise RuntimeError(f"[IBOR_INVARIANT_VIOLATION] imbalance {discrepancy}")

        position_basis = sum(
            (Decimal(str(payload["shares"])) * Decimal(str(payload["avg_cost"]))).quantize(
                QUANT_8, rounding=ROUND_HALF_EVEN
            )
            for payload in self.positions.values()
        )
        if abs(position_basis - self.accounts["ASSET:EQUITY_COST_BASIS"]) > Decimal("0.0001"):
            raise RuntimeError("[IBOR_INVARIANT_VIOLATION] position cost-basis sum mismatch")
        return True

    # --------------------------------------------------------------- trades --
    def post_trade(
        self, ticker: str, action: str, shares: int, fill_price: str | float, fee_amount: str | float
    ) -> dict[str, str]:
        share_dec = Decimal(str(shares))
        price_dec = to_dec(fill_price)
        fee_dec = to_dec(fee_amount)
        gross_dec = (share_dec * price_dec).quantize(QUANT_8, rounding=ROUND_HALF_EVEN)

        position = self.positions.get(ticker, {"shares": 0, "avg_cost": ZERO_DEC})
        old_shares = int(position["shares"])
        old_avg = Decimal(str(position["avg_cost"]))
        normalized = action.upper()

        if normalized == "BUY":
            outflow = gross_dec + fee_dec
            if outflow > self.accounts["ASSET:CASH"]:
                raise ValueError(f"[PRE_TRADE_REJECT] insufficient cash for {ticker} BUY")
            self.accounts["ASSET:CASH"] -= outflow
            self.accounts["ASSET:EQUITY_COST_BASIS"] += gross_dec
            self.accounts["EXPENSE:TAXES_AND_FEES"] += fee_dec
            new_shares = old_shares + shares
            new_avg = ((Decimal(old_shares) * old_avg) + gross_dec) / Decimal(new_shares)
            self.positions[ticker] = {
                "shares": new_shares,
                "avg_cost": new_avg.quantize(QUANT_8, rounding=ROUND_HALF_EVEN),
            }
        elif normalized == "SELL":
            if shares > old_shares:
                raise ValueError(f"[PRE_TRADE_REJECT] cannot short-sell {ticker} in a delivery book")
            cost_basis_removed = (share_dec * old_avg).quantize(QUANT_8, rounding=ROUND_HALF_EVEN)
            self.accounts["ASSET:CASH"] += gross_dec - fee_dec
            self.accounts["ASSET:EQUITY_COST_BASIS"] -= cost_basis_removed
            self.accounts["EQUITY:REALIZED_PNL"] += gross_dec - cost_basis_removed
            self.accounts["EXPENSE:TAXES_AND_FEES"] += fee_dec
            remaining = old_shares - shares
            if remaining == 0:
                self.positions.pop(ticker, None)
            else:
                self.positions[ticker]["shares"] = remaining
        else:
            raise ValueError(f"unsupported action {action!r}")

        self.verify_accounting_invariant()
        state_blob = json.dumps({key: str(value) for key, value in self.accounts.items()}, sort_keys=True)
        self.chain_hash = hashlib.sha256(f"{self.chain_hash}|{state_blob}".encode("utf-8")).hexdigest()
        return {"ticker": ticker, "action": normalized, "merkle_chain_hash": self.chain_hash}


def reconcile_double_entry_identity(snapshot: dict[str, Any], tolerance: float = 0.05) -> dict[str, Any]:
    """Reconcile a V4 float ledger snapshot against the double-entry identity.

    The V4 ``Ledger.apply_orders`` arithmetic satisfies, exactly:

        cash + sum(shares * avg_cost) == starting_cash + realized_pnl - fees_paid

    (BUY legs move cost basis and cash by the same notional; SELL legs move
    realized PnL and cash by the same gross amount; fees always reduce cash).
    Rounded 2dp writes introduce sub-cent drift, hence ``tolerance``.
    """
    cash = to_dec(snapshot.get("cash", 0.0))
    starting = to_dec(snapshot.get("starting_cash", 0.0))
    realized = to_dec(snapshot.get("realized_pnl", 0.0))
    fees = to_dec(snapshot.get("fees_paid", 0.0))
    basis = ZERO_DEC
    for payload in (snapshot.get("positions") or {}).values():
        basis += (to_dec(payload.get("shares", 0.0)) * to_dec(payload.get("avg_cost", 0.0))).quantize(
            QUANT_8, rounding=ROUND_HALF_EVEN
        )
    discrepancy = (cash + basis - (starting + realized - fees)).quantize(QUANT_8, rounding=ROUND_HALF_EVEN)
    return {
        "ok": abs(discrepancy) <= to_dec(tolerance),
        "discrepancy": str(discrepancy),
        "assets": str(cash + basis),
        "equity_plus_income_minus_expenses": str(starting + realized - fees),
    }


# =============================================================================
# 3. Aladdin Euler risk attribution, Student-t CVaR and scenario stress engine
# =============================================================================
class AladdinRiskAndScenarioEngine:
    """Institution-grade risk analytics over the live portfolio.

    * **Euler decomposition** — exact additivity ``sum(CCTR_i) == sigma_p`` by the
      homogeneous-function theorem: ``MCTR = Sigma w / sigma_p``.
    * **Student-t ``CVaR_99``** — MLE-fitted tail index on the realized daily
      series (fat tails, not naive Gaussian VaR).
    * **MDS stress** — ``B = Sigma_RF @ pinv(Sigma_F)``, shock ``w' B delta_F``.
    * **Pre-trade compliance** — golden-copy quarantine blocks + ADV cap.
    """

    # Fallback definitions; ``config/aladdin.yaml`` overrides both blocks.
    DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
        "2008_LEHMAN_CREDIT_FREEZE": {"HY_CREDIT_SPREAD": 4.5, "VIX": 3.8, "SOX_TECH": -0.28, "BRENT_OIL": -0.35},
        "2022_STAGFLATION_RATE_SHOCK": {"US10Y_YIELD": 2.5, "BRENT_OIL": 0.40, "DXY_DOLLAR": 0.12, "SOX_TECH": -0.22},
        "STRAIT_ENERGY_AND_CHIP_BLOCKADE": {"BRENT_OIL": 0.45, "SOX_TECH": -0.30, "USD_INR": 0.06, "VIX": 2.8},
        "INDIA_FISCAL_CAPEX_SURGE": {"USD_INR": -0.03, "BRENT_OIL": -0.15, "HY_CREDIT_SPREAD": -0.8, "SOX_TECH": 0.10},
    }
    DEFAULT_PROXY_FACTORS: dict[str, dict[str, float]] = {
        "US10Y_YIELD": {"market_return": -0.5},
        "BRENT_OIL": {"market_return": 0.3},
        "HY_CREDIT_SPREAD": {"market_return": -1.2, "market_volatility": 1.0},
        "VIX": {"market_volatility": 2.0},
        "SOX_TECH": {"market_return": 1.3},
        "DXY_DOLLAR": {"market_return": -0.35},
        "USD_INR": {"market_return": -0.4},
    }

    def __init__(self, config: "Config | None" = None, *, max_adv_participation: float | None = None) -> None:
        self.confidence = float(_config_default(config, "aladdin.risk.cvar_confidence", 0.99))
        bounds = _config_default(config, "aladdin.risk.cvar_nu_bounds", [3.1, 30.0]) or [3.1, 30.0]
        self.nu_min, self.nu_max = float(bounds[0]), float(bounds[1])
        self.additivity_tolerance = float(
            _config_default(config, "aladdin.risk.euler_additivity_tolerance", 1.0e-10)
        )
        scenarios = _config_default(config, "aladdin.stress_scenarios", None) or self.DEFAULT_SCENARIOS
        self.stress_scenarios = {
            str(name): {str(factor): float(value) for factor, value in (shocks or {}).items()}
            for name, shocks in scenarios.items()
        }
        proxies = _config_default(config, "aladdin.proxy_factors", None) or self.DEFAULT_PROXY_FACTORS
        self.proxy_factors = {
            str(factor): {str(term): float(coeff) for term, coeff in (terms or {}).items()}
            for factor, terms in proxies.items()
        }
        self.max_adv_participation = float(
            max_adv_participation
            if max_adv_participation is not None
            else _config_default(config, "aladdin.compliance.max_adv_participation", 0.05)
        )

    # -------------------------------------------------------- Euler + CVaR --
    def compute_euler_risk_decomposition(
        self, weights: pd.Series, returns_df: pd.DataFrame
    ) -> tuple[float, float, pd.DataFrame]:
        """Exact Euler decomposition + Student-t CVaR; returns (vol%, cvar%, table)."""
        tickers = list(weights.index)
        w = np.asarray(weights.values, dtype=np.float64)
        clean = (
            returns_df[tickers]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(how="all")
            .ffill()
            .fillna(0.0)
        )

        lw = LedoitWolf().fit(clean.values)
        cov_annual = lw.covariance_ * 252.0

        port_var = float(w.T @ cov_annual @ w)
        port_vol = float(np.sqrt(max(port_var, 1e-16)))

        mctr = (cov_annual @ w) / port_vol
        cctr = w * mctr
        if not np.isclose(float(np.sum(cctr)), port_vol, atol=self.additivity_tolerance, rtol=1e-8):
            raise ArithmeticError("Euler additivity violated: sum(CCTR) != sigma_p")
        pctr = (cctr / port_vol) * 100.0

        daily_port_returns = clean.values @ w
        alpha = self.confidence
        if float(np.nanstd(daily_port_returns)) > 0.0:
            nu, loc, scale = student_t.fit(daily_port_returns)
            nu = float(np.clip(nu, self.nu_min, self.nu_max))
            quantile = float(student_t.ppf(1.0 - alpha, df=nu))
            density = float(student_t.pdf(quantile, df=nu))
            daily_cvar = -float(loc) + float(scale) * (density / (1.0 - alpha)) * (
                (nu + quantile**2) / (nu - 1.0)
            )
        else:
            daily_cvar = 0.0

        table = pd.DataFrame(
            {
                "ticker": tickers,
                "weight_pct": np.round(w * 100.0, 4),
                "mctr_ann_pct": np.round(mctr * 100.0, 4),
                "cctr_ann_pct": np.round(cctr * 100.0, 4),
                "pct_of_total_risk": np.round(pctr, 4),
            }
        ).sort_values("pct_of_total_risk", ascending=False)

        return round(port_vol * 100.0, 4), round(daily_cvar * 100.0, 4), table

    # ----------------------------------------------------- scenario stresses --
    def build_proxy_factors(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Declared linear proxies of the scenario factors from the equity panel."""
        base = {
            "market_return": returns_df.mean(axis=1),
            "market_volatility": returns_df.std(axis=1),
        }
        columns: dict[str, pd.Series] = {}
        for factor, terms in self.proxy_factors.items():
            series: pd.Series | None = None
            for term, coeff in terms.items():
                component = base.get(term)
                if component is None:
                    continue
                series = component * coeff if series is None else series + component * coeff
            if series is not None:
                columns[factor] = series
        return pd.DataFrame(columns, index=returns_df.index)

    def run_scenario_stress_tests(
        self,
        weights: pd.Series,
        stock_returns: pd.DataFrame,
        macro_factor_changes: pd.DataFrame,
    ) -> dict[str, float]:
        """Project exogenous shock vectors through ``B = Sigma_RF Sigma_F^{-1}``."""
        tickers = list(weights.index)
        scenario_names = list(self.stress_scenarios)
        aligned = pd.concat([stock_returns[tickers], macro_factor_changes], axis=1).dropna()
        if len(aligned) < 30 or macro_factor_changes.shape[1] == 0:
            return {name: 0.0 for name in scenario_names}

        n_stocks = len(tickers)
        joint_cov = LedoitWolf().fit(aligned.values).covariance_
        cov_rf = joint_cov[:n_stocks, n_stocks:]
        cov_ff = joint_cov[n_stocks:, n_stocks:]
        beta_stress = cov_rf @ np.linalg.pinv(cov_ff)

        factor_names = list(macro_factor_changes.columns)
        w = np.asarray(weights.values, dtype=np.float64)
        outcomes: dict[str, float] = {}
        for name, shock_map in self.stress_scenarios.items():
            delta = np.array([shock_map.get(factor, 0.0) for factor in factor_names], dtype=np.float64)
            outcomes[name] = round(float(np.dot(w, beta_stress @ delta) * 100.0), 2)
        return outcomes

    # ------------------------------------------------- pre-trade compliance --
    @staticmethod
    def enforce_pre_trade_compliance_gate(
        proposed_orders: list[dict[str, Any]],
        golden_status_map: dict[str, str],
        adv_20d_map: dict[str, float],
        max_adv_participation: float = 0.05,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Block quarantined BUYs; scale any order above the ADV participation cap.

        Returns ``(approved, rejected_or_scaled)``. Orders whose notional is below
        the cap pass through untouched apart from an ``est_notional`` stamp.
        """
        approved: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for order in proposed_orders:
            ticker = order["ticker"]
            action = str(order.get("action", "")).upper()
            status = golden_status_map.get(ticker, "VERIFIED_GOLDEN")
            if status != "VERIFIED_GOLDEN" and action == "BUY":
                rejected.append({**order, "compliance_reason": f"BLOCKED: {status}"})
                continue

            ref_price = float(order.get("ref_price") or 0.0)
            shares = float(order.get("shares") or 0.0)
            notional = float(
                order.get("est_notional")
                or order.get("order_value")
                or ref_price * shares
            )
            adv_value = float(adv_20d_map.get(ticker, float("inf")))
            max_allowed = adv_value * float(max_adv_participation)

            if max_allowed > 0.0 and notional > max_allowed:
                scaled_shares = int(max_allowed // ref_price) if ref_price > 0.0 else 0
                if scaled_shares <= 0:
                    rejected.append({**order, "compliance_reason": "BLOCKED: exceeds ADV liquidity cap"})
                    continue
                scaled_notional = round(scaled_shares * ref_price, 2)
                approved.append(
                    {
                        **order,
                        "shares": scaled_shares,
                        "order_value": scaled_notional,
                        "est_notional": scaled_notional,
                        "compliance_note": "SCALED_TO_ADV_CAP",
                    }
                )
                continue

            approved.append({**order, "est_notional": notional})

        return approved, rejected
