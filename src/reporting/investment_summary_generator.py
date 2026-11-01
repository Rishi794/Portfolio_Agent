"""Compact daily invest/hold/avoid summary based on the actual V5 run output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..state.ledger import Ledger


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and abs(number) != float("inf") else default


class InvestmentSummaryGenerator:
    """Render actionable recommendations and learned forecast diagnostics."""

    @classmethod
    def render(
        cls,
        summary: dict[str, Any],
        state_dir: str | Path,
        output_path: str | Path,
        *,
        starting_cash: dict[str, float] | None = None,
    ) -> str:
        starting_cash = starting_cash or {"US": 100_000.0, "INDIA": 1_000_000.0}
        runs = {
            str(run.get("market")): run
            for run in summary.get("runs", [])
            if isinstance(run, dict) and run.get("market")
        }
        lines = [
            f"# Portfolio Summary — {summary.get('as_of', 'unknown date')}",
            "",
            f"Run `{summary.get('run_id', 'n/a')}` · mode `{summary.get('mode', 'n/a')}`",
            "",
            "> Orders are proposals awaiting approval. Forecasts and calibration statistics do not guarantee returns.",
            "",
        ]

        for market, run in runs.items():
            lines.extend([f"## {market}", ""])
            ledger = Ledger(state_dir, market, starting_cash=float(starting_cash.get(market, 100_000.0)))
            snapshot = ledger.load()
            current = ledger.current_weights(snapshot)
            orders = list(run.get("tickets") or [])
            buys = [order for order in orders if str(order.get("action", "")).upper() == "BUY"]
            sells = [order for order in orders if str(order.get("action", "")).upper() == "SELL"]
            sell_tickers = {str(order.get("ticker")) for order in sells}
            held = sorted(ticker for ticker, weight in current.items() if weight > 0.0 and ticker not in sell_tickers)

            lines.append("### What to invest in")
            if buys:
                for order in buys:
                    lines.append(
                        f"- **BUY `{order.get('ticker')}`**: {_number(order.get('shares')):.0f} shares at "
                        f"reference {_number(order.get('ref_price')):,.2f}; target "
                        f"{_number(order.get('target_weight_pct')):.2f}% of equity."
                    )
            else:
                lines.append("- No BUY proposal cleared the optimizer and transaction-cost deadband.")
            lines.append("")

            lines.append("### What to hold")
            if held:
                lines.append("- " + ", ".join(f"`{ticker}`" for ticker in held))
            else:
                lines.append("- No existing positions were identified for a hold recommendation.")
            lines.append("")

            lines.append("### What not to invest in / proposed reductions")
            if sells:
                for order in sells:
                    lines.append(
                        f"- **SELL `{order.get('ticker')}`**: {_number(order.get('shares')):.0f} shares; "
                        f"alpha Z {_number(order.get('alpha_z')):+.2f}."
                    )
            quarantined = sorted(set(map(str, run.get("quarantined_tickers", []) or [])))
            if quarantined:
                lines.append(
                    "- **Do not buy while data verification is unresolved:** "
                    + ", ".join(f"`{ticker}`" for ticker in quarantined)
                    + "."
                )
            if not sells and not quarantined:
                lines.append("- No sell proposal or data-quality quarantine was reported.")
            lines.append("")

            diagnostics = (summary.get("markets", {}).get(market, {}) or {})
            feedback = diagnostics.get("forecast_feedback", {}) or {}
            lines.append("### Self-reflection from matured forecasts")
            accuracy = feedback.get("directional_accuracy")
            accuracy_text = "n/a" if accuracy is None else f"{float(accuracy) * 100.0:.1f}%"
            evaluations = int(feedback.get("matured_evaluations", 0) or 0)
            if evaluations:
                lines.append(
                    f"- Evaluated {evaluations} non-overlapping "
                    f"{feedback.get('nonoverlap_horizon_sessions', 5)}-session forecasts; directional accuracy "
                    f"{accuracy_text}. Existing IC-IR learning continues to update factor weights from matured returns."
                )
            else:
                lines.append("- No matured forecast labels were available for feedback in this run.")
            interval_offset = _number(feedback.get("timestral_interval_offset_pct"))
            interval_samples = int(feedback.get("timestral_interval_samples", 0) or 0)
            lines.append(
                f"- Empirical additive widening for the Timestral 5th/95th percentile band: "
                f"{interval_offset:.3f} percentage points ({interval_samples} matured intervals). "
                "The summed daily quantiles are an interval proxy; this adjustment is not a formal coverage "
                "guarantee or a 99% CVaR calibration."
            )
            weak = feedback.get("low_reliability_names", []) or []
            if weak:
                lines.append(
                    "- Bounded alpha shrink for low-reliability names: "
                    + ", ".join(
                        f"`{item['ticker']}` ({item['posterior_mean']:.0%}, "
                        f"×{item['alpha_multiplier']:.2f})"
                        for item in weak
                    )
                    + ". No automatic ticker quarantine is applied from hit rate alone."
                )
            misses = feedback.get("recent_misses", []) or []
            if misses:
                lines.append("- Recent high-conviction misses (largest aligned factor is descriptive, not causal):")
                for miss in misses[:5]:
                    lines.append(
                        f"  - `{miss.get('pred_date')}` `{miss.get('ticker')}`: alpha "
                        f"{_number(miss.get('pred_alpha_z')):+.2f}, realized "
                        f"{_number(miss.get('realized_ret_pct')):+.2f}%, "
                        f"{miss.get('failure_mode')}; largest aligned factor "
                        f"`{miss.get('largest_signal_aligned_pillar') or 'n/a'}`."
                    )
            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
        return markdown
