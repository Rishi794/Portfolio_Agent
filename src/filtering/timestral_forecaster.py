"""Mistral **Timestral** zero-shot probabilistic forecaster — the Stage-3 7th pillar.

Wraps ``mistralai/Timestral-209M`` (209M params, 24 layers, d_model 768, 32-step
patches; 837 MB fp32) through the official ``timestral`` inference package
(``pip install git+https://github.com/mistralai/timestral.git``, Apache-2.0 code /
Modified-MIT weights).

Why the *multivariate* call and not a per-stock loop
----------------------------------------------------
Timestral has two attention stacks: **time attention** and **variate attention**.
Forecasting one stock at a time (``V = 1``) disables the variate stack entirely;
packing the surviving cross-section **plus macro/spillover drivers** into a single
``(B, V, T)`` context lets the 24 variate-attention layers price cross-stock,
sector and macro-to-stock lead-lag structure in one forward pass. The same trick
collapses N sequential 140 ms passes into one batched pass.

Point-in-time discipline
------------------------
The engine never reads the lakehouse itself. Callers hand it *already PIT-filtered*
OHLCV (knowledge timestamp <= as_of, trade_date <= as_of). The only forward-looking
inputs are:

* **known-future calendar covariates** — exchange rules (derivative expiry,
  month-end rebalancing) and *announced* corporate-action ex-dates; and
* the model's own forecast.

Artificially lagging the calendar is not required (expiry schedules are public
ahead of time); silently using an ex-date whose announcement post-dates ``as_of``
is the leak to avoid, which is why the caller must filter ``corporate_actions`` on
``knowledge_timestamp`` before building ``ex_dates_by_ticker``.

Quantile algebra
----------------
Timestral emits 21 non-crossing quantiles ``q(p)`` for ``p in {0.01 .. 0.99}``.
Because :math:`E[R] = \\int_0^1 F^{-1}(p)\\,dp`, the distributional expected
return is the trapezoidal integral over the 21 levels; upside/downside semi-
dispersions come from ``q(0.90) - q(0.50)`` and ``q(0.50) - q(0.05)``, giving the
tail-penalised score

``f_timestral = (mu + 0.5 * (upside - downside)) / max(downside, eps)``.

All quantities are produced in *standardised* units and rescaled by the stock's
own trailing daily log-return volatility, so names trading at ₹50 and ₹5,000 are
scored on the same footing.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - the Stage-3 extra is only installed in the fan-in job
    import torch
    from timestral import TimestralForecaster

    TIMESTRAL_AVAILABLE = True
except Exception:  # pragma: no cover - ImportError or an unusable torch build
    torch = None  # type: ignore[assignment]
    TIMESTRAL_AVAILABLE = False

#: The 21 non-crossing quantile levels emitted by every Timestral checkpoint.
TIMESTRAL_QUANTILES = np.array(
    [
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
    ],
    dtype=np.float64,
)

_LEVEL = {level: index for index, level in enumerate(TIMESTRAL_QUANTILES.tolist())}
_Q01, _Q05, _Q50, _Q90 = _LEVEL[0.01], _LEVEL[0.05], _LEVEL[0.50], _LEVEL[0.90]

#: Must match ``PILLARS`` in ``src.main_v4`` — the pillar name registered in the lakehouse.
TIMESTRAL_PILLAR = "timestral"

OUTPUT_COLUMNS = (
    "f_timestral",
    "timestral_exp_ret_5d_pct",
    "timestral_cvar99_pct",
    "timestral_cvar99_5d_pct",
)


def _cfg(config: Any, path: str, default: Any) -> Any:
    """Read ``aladdin.<path>`` from a :class:`src.config.Config`, tolerating ``None``."""
    if config is None:
        return default
    return config.get(f"aladdin.{path}", default)


class TimestralForecastEngine:
    """Multivariate, covariate-conditioned Timestral forecaster.

    The model is loaded **lazily** on the first forecast call: constructing the
    engine is free, so unit tests and the parallel Stage-2A shards never touch the
    Hub. If the ``timestral`` package is absent, or the checkpoint cannot be
    loaded, the engine degrades to a deterministic empirical-quantile fallback and
    reports ``mode == "offline_fallback"`` — the pipeline never hard-fails on a
    missing 837 MB artifact.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        model_id: str | None = None,
        device: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        threads: int | None = None,
        local_files_only: bool | None = None,
        forecaster: Any = None,
    ) -> None:
        self.config = config
        self.model_id = str(model_id or _cfg(config, "timestral.model_id", "mistralai/Timestral-209M"))
        self.device = device or str(_cfg(config, "timestral.device", "auto"))
        self.cache_dir = str(
            Path(str(cache_dir or _cfg(config, "timestral.cache_dir", "~/.cache/huggingface"))).expanduser()
        )
        self.threads = threads if threads is not None else _cfg(config, "timestral.cpu_threads", None)
        self.local_files_only = bool(
            local_files_only if local_files_only is not None else _cfg(config, "timestral.local_files_only", False)
        )
        self.require_model = bool(_cfg(config, "timestral.require_model", False))

        #: The loaded upstream forecaster (``TimestralForecaster``) or the injected stub.
        self._forecaster: Any = forecaster
        self._load_error: str | None = None
        #: What the *last* forecast actually used: a stub counts as the model path.
        self._last_mode = "model" if forecaster is not None else "offline_fallback"

    # ------------------------------------------------------------------ model --
    @property
    def available(self) -> bool:
        """Whether the real checkpoint can be used (the package imports)."""
        return TIMESTRAL_AVAILABLE or self._forecaster is not None

    @property
    def mode(self) -> str:
        """``"model"`` when the last forecast used the checkpoint, else ``"offline_fallback"``."""
        return self._last_mode

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_forecaster(self) -> None:
        if self._forecaster is not None or self._load_error is not None:
            return
        # Explicit kill-switch so hermetic test suites / deterministic runs can force
        # the empirical fallback even where the Stage-3 extra is installed.
        if os.getenv("PMS_TIMESTRAL_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
            self._load_error = "disabled by PMS_TIMESTRAL_DISABLED"
            if self.require_model:
                raise RuntimeError(self._load_error)
            return
        if not TIMESTRAL_AVAILABLE:
            self._load_error = "timestral package not installed (pip install git+https://github.com/mistralai/timestral.git)"
            if self.require_model:
                raise RuntimeError(self._load_error)
            return
        try:
            if str(self.device) == "cpu" or str(_cfg(self.config, "timestral.device", "auto")) == "cpu":
                torch.set_num_threads(int(self.threads or os.cpu_count() or 4))
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:  # pragma: no cover - older torch builds
                    pass
            kwargs: dict[str, Any] = {"cache_dir": self.cache_dir or None}
            if str(self.device) not in {"auto", ""}:
                kwargs["device"] = self.device
            if self.local_files_only:
                kwargs["local_files_only"] = True
            self._forecaster = TimestralForecaster.from_pretrained(self.model_id, **kwargs)
        except Exception as error:  # noqa: BLE001 - degrade, never crash the fan-in
            self._load_error = f"{type(error).__name__}: {error}"
            if self.require_model:
                raise
            self._forecaster = None

    # ------------------------------------------------------------- covariates --
    @staticmethod
    def build_known_future_calendar(
        history_dates: pd.DatetimeIndex,
        prediction_length: int,
        market: str,
        ex_dates_union: set[dt.date] | None = None,
    ) -> np.ndarray:
        """Binary known-future rows over ``T_context + H``: expiry / ex-date / month-end.

        All-0-1 rows are interpreted by Timestral as *known-future indicators*, not
        as numeric covariates, so they may legitimately cover the horizon.
        """
        ex_set = ex_dates_union or set()
        last = pd.Timestamp(history_dates[-1])
        future = pd.bdate_range(start=last + pd.Timedelta(days=1), periods=int(prediction_length))
        full = pd.DatetimeIndex(history_dates).append(future)

        covariates = np.zeros((3, len(full)), dtype=np.float32)
        for index, timestamp in enumerate(full):
            day = timestamp.date()
            if market == "INDIA":
                # NSE F&O: last Thursday of the month (no expiry in a next-month week).
                is_expiry = day.weekday() == 3 and (day + dt.timedelta(days=7)).month != day.month
            else:
                # US: third Friday (monthly options expiry).
                is_expiry = day.weekday() == 4 and 15 <= day.day <= 21
            covariates[0, index] = 1.0 if is_expiry else 0.0
            covariates[1, index] = 1.0 if day in ex_set else 0.0
            covariates[2, index] = 1.0 if (day + dt.timedelta(days=3)).month != day.month else 0.0
        return covariates

    # ---------------------------------------------------------------- scoring --
    @staticmethod
    def score_quantiles(quantile_horizon: np.ndarray, daily_vol: float) -> dict[str, float]:
        """Trapezoidal quadrature over the 21 levels -> alpha + tail metrics.

        ``quantile_horizon`` is ``(H, 21)`` in standardised log-return units;
        ``daily_vol`` rescales back to the stock's own return scale.
        """
        matrix = np.asarray(quantile_horizon, dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if matrix.ndim != 2 or matrix.shape[1] != TIMESTRAL_QUANTILES.size or matrix.size == 0:
            return {name: 0.0 for name in OUTPUT_COLUMNS}
        matrix = np.sort(matrix, axis=1)  # non-crossing guard (model guarantees it)

        level_widths = np.diff(TIMESTRAL_QUANTILES)  # (20,)
        step_expectation = np.sum(
            0.5 * (matrix[:, 1:] + matrix[:, :-1]) * level_widths[None, :], axis=1
        )
        expected_standardised = float(np.sum(step_expectation))
        expected_return = expected_standardised * daily_vol

        median = matrix[:, _Q50]
        upside = float(np.mean(matrix[:, _Q90] - median)) * daily_vol
        downside = max(float(np.mean(median - matrix[:, _Q05])) * daily_vol, 1e-12)

        score = (expected_return + 0.5 * (upside - downside)) / downside

        # q(0.01) is the model's 1%-quantile loss (a forward VaR-type number); the
        # V5 fusion takes max(historical Student-t ES99, this), which is conservative
        # because a quantile is always weaker than the expected shortfall beyond it.
        cvar_1d_pct = abs(min(0.0, float(matrix[0, _Q01]) * daily_vol)) * 100.0
        cvar_horizon_pct = float(np.sum(np.maximum(0.0, -matrix[:, _Q01])) * daily_vol) * 100.0

        return {
            "f_timestral": round(float(score), 4),
            "timestral_exp_ret_5d_pct": round(expected_return * 100.0, 3),
            "timestral_cvar99_pct": round(cvar_1d_pct, 3),
            "timestral_cvar99_5d_pct": round(cvar_horizon_pct, 3),
        }

    # ------------------------------------------------------------- inference ---
    def _extract_quantiles(self, forecast: Any, n_variates: int, horizon: int) -> np.ndarray:
        """Normalise the upstream ``Forecast`` into a ``(V, H, 21)`` float64 array."""
        quantiles = getattr(forecast, "quantiles", None)
        if quantiles is not None:
            array = np.asarray(quantiles.detach() if hasattr(quantiles, "detach") else quantiles, dtype=np.float64)
            if array.ndim == 4:  # (B, V, H, Q)
                return array[0, :, :horizon, :]
            if array.ndim == 3:  # (V, H, Q) already unbatched
                return array[:, :horizon, :]
        if hasattr(forecast, "quantile"):
            slices = []
            for level in TIMESTRAL_QUANTILES:
                values = forecast.quantile(float(level))
                values = np.asarray(values.detach() if hasattr(values, "detach") else values, dtype=np.float64)
                if values.ndim == 3:
                    values = values[0]
                elif values.ndim == 1:
                    values = values[None, :]
                slices.append(values[:, :horizon])
            return np.stack(slices, axis=-1)
        raw = np.asarray(forecast, dtype=np.float64)
        if raw.ndim == 4:
            return raw[0, :, :horizon, :]
        return raw.reshape(n_variates, horizon, TIMESTRAL_QUANTILES.size)

    # ----------------------------------------------------------- orchestration --
    def forecast_universe_cross_section(
        self,
        ohlcv_groups: Mapping[str, pd.DataFrame],
        market: str,
        *,
        sectors: pd.Series | None = None,
        ex_dates_by_ticker: Mapping[str, set[dt.date]] | None = None,
        macro_frame: pd.DataFrame | None = None,
        prediction_length: int | None = None,
    ) -> pd.DataFrame:
        """Joint ``(B=1, V, T)`` forecast for the whole candidate cross-section.

        Returns a frame indexed by ticker with :data:`OUTPUT_COLUMNS`. Empty when no
        ticker has enough history. Macro columns (already standardised by the regime
        detector) are appended as extra variates, or market-breadth/dispersion
        variates are synthesised when no macro matrix is available.
        """
        horizon = int(prediction_length or _cfg(self.config, "timestral.prediction_length", 5))
        context_length = int(_cfg(self.config, "timestral.context_length", 256))
        min_context = int(_cfg(self.config, "timestral.min_context", 64))
        max_variates = int(_cfg(self.config, "timestral.max_variates", 128))
        vol_window = int(_cfg(self.config, "timestral.vol_window", 60))
        adv_window = int(_cfg(self.config, "timestral.adv_window", 20))
        vol_floor = float(_cfg(self.config, "timestral.vol_floor", 1e-6))
        use_sectors = bool(_cfg(self.config, "timestral.use_sector_covariate", True))
        ex_map = ex_dates_by_ticker or {}

        prepared: list[dict[str, Any]] = []
        for ticker, group in ohlcv_groups.items():
            if group is None or group.empty or "adj_close" not in group.columns:
                continue
            frame = group.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            if len(frame) < min_context:
                continue
            frame = frame.tail(context_length)
            dates = pd.DatetimeIndex(pd.to_datetime(frame["trade_date"]))
            closes = pd.to_numeric(frame["adj_close"], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(closes).any():
                continue
            previous = np.concatenate([[closes[0]], closes[:-1]])
            with np.errstate(divide="ignore", invalid="ignore"):
                log_returns = np.log(np.maximum(closes, 1e-8) / np.maximum(previous, 1e-8))
            log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
            daily_vol = float(np.std(log_returns[-vol_window:]))
            if not np.isfinite(daily_vol) or daily_vol <= vol_floor:
                continue

            volumes = pd.to_numeric(frame.get("volume", pd.Series(np.nan, index=frame.index)), errors="coerce")
            average_volume = volumes.rolling(adv_window, min_periods=1).mean().to_numpy(dtype=np.float64)
            abnormal = np.log(
                np.maximum(volumes.to_numpy(dtype=np.float64), 1.0) / np.maximum(average_volume, 1.0)
            )

            liquidity = float(
                np.nanmean(
                    pd.to_numeric(frame.get("dollar_volume", pd.Series(np.nan, index=frame.index)), errors="coerce")
                    .tail(adv_window)
                    .to_numpy(dtype=np.float64)
                )
            )
            prepared.append(
                {
                    "ticker": str(ticker),
                    "dates": dates,
                    "target": (log_returns / daily_vol).astype(np.float32),
                    "past": np.nan_to_num(abnormal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                    "vol": daily_vol,
                    "liquidity": liquidity if np.isfinite(liquidity) else 0.0,
                }
            )

        if not prepared:
            return pd.DataFrame(columns=list(OUTPUT_COLUMNS), index=pd.Index([], name="ticker"))

        # Cap the variate count by liquidity: 24-layer variate attention over 128
        # names is the CPU sweet spot; the tail of the cross-section adds little
        # and costs quadratically.
        prepared.sort(key=lambda item: item["liquidity"], reverse=True)
        prepared = prepared[:max_variates]

        index = pd.DatetimeIndex(sorted(set().union(*(set(item["dates"]) for item in prepared))))
        index = index[-context_length:]
        target_panel = pd.DataFrame(
            {item["ticker"]: pd.Series(item["target"], index=item["dates"]) for item in prepared}
        ).reindex(index)
        past_panel = pd.DataFrame(
            {item["ticker"]: pd.Series(item["past"], index=item["dates"]) for item in prepared}
        ).reindex(index)

        variate_blocks = [target_panel]
        if macro_frame is not None and not macro_frame.empty:
            macro = macro_frame.reindex(index).ffill()
            macro = macro.loc[:, macro.notna().any(axis=0)]
            if not macro.empty:
                variate_blocks.append(macro.astype(np.float32))
        if len(variate_blocks) == 1:
            variate_blocks.append(target_panel.mean(axis=1).rename("__MKT_BREADTH__").to_frame().astype(np.float32))
            variate_blocks.append(
                target_panel.std(axis=1).fillna(0.0).rename("__MKT_DISPERSION__").to_frame().astype(np.float32)
            )
        joint = pd.concat(variate_blocks, axis=1)
        # ``ascontiguousarray`` hands upstream a writable copy: a bare ``.T`` view is
        # flagged as read-only by the tensor bridge, and the model may write in place.
        context = np.ascontiguousarray(joint.to_numpy(dtype=np.float32).T[None, ...])  # (B=1, V, T)

        past_covariates = np.ascontiguousarray(past_panel.to_numpy(dtype=np.float32).T[None, ...])  # (B=1, C_past, T)

        history = pd.DatetimeIndex(target_panel.index)
        ex_dates_union: set[dt.date] = set()
        for ticker_dates in ex_map.values():
            ex_dates_union.update(ticker_dates)
        known = np.ascontiguousarray(
            self.build_known_future_calendar(history, horizon, market, ex_dates_union)[None, ...]
        )

        categorical: np.ndarray | None = None
        if use_sectors and sectors is not None and not sectors.empty:
            codes, _ = pd.factorize(sectors.reindex(target_panel.columns).astype("object"), use_na_sentinel=True)
            row = np.where(codes < 0, np.nan, codes).astype(np.float32)[:, None]
            categorical = np.ascontiguousarray(
                np.repeat(row, context.shape[-1], axis=1).T[None, ...]
            )  # (B=1, C_cat=1, T)

        self._ensure_forecaster()
        quantiles: np.ndarray | None = None
        if self._forecaster is not None:
            try:
                quantiles = self._call_model(context, past_covariates, known, categorical, horizon)
            except Exception as error:  # noqa: BLE001 - degrade for this run, keep the pipeline alive
                self._load_error = f"forecast failed: {type(error).__name__}: {error}"
                if self.require_model:
                    raise
                quantiles = None
        if quantiles is None:
            quantiles = self._fallback_quantiles(context, horizon)
            self._last_mode = "offline_fallback"
        else:
            self._last_mode = "model"

        rows: list[dict[str, Any]] = []
        for position, item in enumerate(prepared):
            metrics = self.score_quantiles(quantiles[position], item["vol"])
            rows.append({"ticker": item["ticker"], **metrics})
        return pd.DataFrame(rows).set_index("ticker")

    def _call_model(
        self,
        context: np.ndarray,
        past_covariates: np.ndarray,
        known: np.ndarray,
        categorical: np.ndarray | None,
        horizon: int,
    ) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "prediction_length": int(horizon),
            "past_covariates": past_covariates,
            "known_covariates": known,
        }
        if categorical is not None:
            kwargs["categorical_past_covariates"] = categorical
        if torch is not None and hasattr(torch, "inference_mode"):  # pragma: no cover - real torch path
            with torch.inference_mode():
                forecast = self._forecaster.forecast(context, **kwargs)
        else:  # stub forecaster in tests, or a torch build without inference_mode
            forecast = self._forecaster.forecast(context, **kwargs)
        return self._extract_quantiles(forecast, context.shape[1], horizon)

    @staticmethod
    def _fallback_quantiles(context: np.ndarray, horizon: int) -> np.ndarray:
        """Deterministic empirical-quantile fallback over the standardised context.

        Used when the Stage-3 extra is absent (unit tests, shards, cold cache):
        quantiles of the trailing window plus the recent drift, tiled over the
        horizon. Slower-moving than the transformer, but honest about its mode.
        """
        values = np.nan_to_num(np.asarray(context, dtype=np.float64)[0], nan=0.0)
        n_variates = values.shape[0]
        window = values[:, -min(values.shape[1], 60) :]
        levels = TIMESTRAL_QUANTILES[None, :]
        empirical = np.quantile(window, TIMESTRAL_QUANTILES, axis=1).T  # (V, 21)
        drift = np.mean(values[:, -min(values.shape[1], 10) :], axis=1, keepdims=True)
        per_step = empirical + drift
        return np.repeat(per_step[:, None, :], int(horizon), axis=1)
