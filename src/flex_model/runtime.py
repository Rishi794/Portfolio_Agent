"""Point-in-time event replay adapter for the V5 batch reducer.

The reducer owns lakehouse reads and portfolio construction; this adapter turns
those already PIT-filtered rows into Flex events, persists the arena snapshot in
the state directory, and returns ticker-aligned feature outputs for factor fusion.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .engine import EventKind, FlexModelRegistry, ZeroAllocArena


class FlexModelReplayRuntime:
    """Replay OHLCV and current PIT snapshots through compiled Flex event plans."""

    _SCHEMA_VERSION = 1
    _PRICE_DIGEST_COLUMNS = (
        "ticker", "date", "close", "adj_close", "high", "low", "knowledge_timestamp"
    )
    _ACTION_DIGEST_COLUMNS = (
        "ticker", "ex_date", "action_type", "split_ratio", "cash_dividend", "knowledge_timestamp"
    )

    def __init__(self, state_dir: str | Path, registry: FlexModelRegistry) -> None:
        self.state_dir = Path(state_dir)
        self.registry = registry
        self._plans = {
            (market, event): registry.compile_plan(market, event)
            for market in ("US", "INDIA")
            for event in EventKind
        }

    def replay(
        self,
        *,
        market: str,
        tickers: pd.Index | list[str] | tuple[str, ...],
        as_of: dt.date | pd.Timestamp,
        prices: pd.DataFrame,
        actions: pd.DataFrame,
        fundamentals: pd.DataFrame,
        tech: pd.DataFrame,
        policy_scores: pd.Series,
        spillover_scores: pd.Series,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        names = tuple(sorted({str(ticker) for ticker in tickers}))
        if not names:
            return pd.DataFrame(index=pd.Index([], name="ticker")), {"events": 0, "restored": False}

        arena = self.registry.allocate_arena(names, market)
        price_frame = self._prepare_prices(prices, names, as_of)
        action_frame = self._prepare_actions(actions, market, as_of)
        state_path = self.state_dir / f"flex_arena_{market.lower()}.json"
        saved = self._read_snapshot(state_path)
        checkpoint = self._compatible_checkpoint(saved, arena, price_frame, action_frame, as_of)
        restored = checkpoint is not None
        if restored:
            try:
                arena.restore_state(saved["arena"])
            except (KeyError, TypeError, ValueError):
                restored = False
                # Restore validates several arrays in sequence and may have copied
                # some of them before discovering a later mismatch. A fresh arena is
                # required before replaying from the beginning, or history is applied
                # on top of partially restored state.
                arena = self.registry.allocate_arena(names, market)
        if restored:
            replay_prices = price_frame[pd.to_datetime(price_frame["date"]) > checkpoint]
            replay_actions = action_frame[pd.to_datetime(action_frame["ex_date"]) > checkpoint]
        else:
            replay_prices = price_frame
            replay_actions = action_frame

        event_count = self._replay_market_events(arena, market, names, replay_prices, replay_actions)
        self._clear_snapshot_features(arena)
        event_count += self._apply_fundamental_snapshot(arena, market, names, fundamentals)
        event_count += self._apply_policy_snapshot(arena, names, policy_scores)
        event_count += self._apply_tech_snapshot(arena, names, tech)
        event_count += self._apply_spillover_snapshot(arena, market, names, spillover_scores)

        output_names = [
            node.name
            for node in self.registry._market_nodes(market).values()
            if node.is_model_output
        ]
        features = arena.export_feature_matrix(output_names)
        last_bar = pd.to_datetime(price_frame["date"]).max() if not price_frame.empty else checkpoint
        if pd.isna(last_bar):
            last_bar = checkpoint
        price_digest = self._frame_digest(price_frame, self._PRICE_DIGEST_COLUMNS)
        if last_bar is not None and not pd.isna(last_bar):
            actions_at_checkpoint = action_frame[pd.to_datetime(action_frame["ex_date"]) <= last_bar]
        else:
            actions_at_checkpoint = action_frame.iloc[0:0]
        action_digest = self._frame_digest(actions_at_checkpoint, self._ACTION_DIGEST_COLUMNS)
        self._write_snapshot(
            state_path,
            {
                "schema_version": self._SCHEMA_VERSION,
                "market": market,
                "as_of": str(pd.Timestamp(as_of).date()),
                "tickers": list(names),
                "last_bar_date": str(pd.Timestamp(last_bar).date()) if last_bar is not None else None,
                "price_digest": price_digest,
                "action_digest": action_digest,
                "arena": arena.export_state(),
            },
        )
        return features, {
            "events": int(event_count),
            "replayed_bars": int(replay_prices["date"].nunique()) if not replay_prices.empty else 0,
            "restored": restored,
            "last_bar_date": str(pd.Timestamp(last_bar).date()) if last_bar is not None else None,
            "state_file": state_path.name,
        }

    def _replay_market_events(
        self,
        arena: ZeroAllocArena,
        market: str,
        names: tuple[str, ...],
        prices: pd.DataFrame,
        actions: pd.DataFrame,
    ) -> int:
        price_dates = set(pd.to_datetime(prices["date"]).dt.normalize()) if not prices.empty else set()
        action_dates = set(pd.to_datetime(actions["ex_date"]).dt.normalize()) if not actions.empty else set()
        event_count = 0
        for day in sorted(price_dates | action_dates):
            day_actions = actions[pd.to_datetime(actions["ex_date"]).dt.normalize() == day] if not actions.empty else actions
            if not day_actions.empty:
                # The ingestion path has already materialized announced split/dividend
                # adjustments into its OHLC history. Send neutral factors here so the
                # state node observes the event without applying the adjustment twice.
                self._plans[(market, EventKind.CORP_ACTION_EX_DATE)].execute(
                    arena, {"price_adjustment_factor": np.ones(len(names), dtype=np.float64)}
                )
                event_count += len(day_actions)

            day_prices = prices[pd.to_datetime(prices["date"]).dt.normalize() == day]
            if day_prices.empty:
                continue
            grouped = day_prices.drop_duplicates("ticker", keep="last").set_index("ticker")
            payload = {
                "close": self._aligned(grouped, names, "adj_close", fallback="close"),
                "high": self._aligned(grouped, names, "high"),
                "low": self._aligned(grouped, names, "low"),
            }
            self._plans[(market, EventKind.EOD_BAR)].execute(arena, payload)
            event_count += 1
        return event_count

    def _apply_fundamental_snapshot(
        self, arena: ZeroAllocArena, market: str, names: tuple[str, ...], frame: pd.DataFrame
    ) -> int:
        if frame is None or frame.empty:
            return 0
        fields = {
            "roic_spread": "roic_wacc_spread",
            "fcf_yield": "fcf_ev_yield",
            "piotroski_f_score": "piotroski_f_score",
        }
        aligned = {key: self._aligned(frame, names, column) for key, column in fields.items()}
        core_mask = np.logical_and.reduce([np.isfinite(aligned[key]) for key in fields])
        payload: dict[str, np.ndarray] = {**aligned, "core_event_mask": core_mask}
        if market == "INDIA":
            pledge = self._aligned(frame, names, "promoter_pledge_pct")
            flow = self._aligned(frame, names, "institutional_flow_delta")
            payload.update(
                promoter_pledge_pct=pledge,
                institutional_flow_delta=flow,
                pledge_event_mask=np.isfinite(pledge) & np.isfinite(flow),
            )
        if not core_mask.any() and not np.asarray(payload.get("pledge_event_mask", []), dtype=bool).any():
            return 0
        self._plans[(market, EventKind.XBRL_FUNDAMENTAL_FILING)].execute(arena, payload)
        return 1

    def _apply_policy_snapshot(self, arena: ZeroAllocArena, names: tuple[str, ...], scores: pd.Series) -> int:
        if scores is None or scores.empty:
            return 0
        shock = self._aligned(scores.to_frame("score"), names, "score")
        self._plans[(arena.market, EventKind.POLICY_OR_GEOPOL_SHOCK)].execute(
            arena,
            {"decay_factor": np.zeros(len(names)), "bge_projected_shock": np.nan_to_num(shock)},
        )
        return 1

    def _apply_tech_snapshot(
        self, arena: ZeroAllocArena, names: tuple[str, ...], frame: pd.DataFrame
    ) -> int:
        if frame is None or frame.empty or "citation_velocity_log_ratio" not in frame.columns:
            return 0
        velocity = self._aligned(frame, names, "citation_velocity_log_ratio")
        mask = np.isfinite(velocity)
        if not mask.any():
            return 0
        self._plans[(arena.market, EventKind.PATENT_RND_UPDATE)].execute(
            arena,
            {"citation_velocity_log_ratio": np.nan_to_num(velocity), "event_mask": mask},
        )
        return 1

    def _apply_spillover_snapshot(
        self,
        arena: ZeroAllocArena,
        market: str,
        names: tuple[str, ...],
        scores: pd.Series,
    ) -> int:
        if market != "INDIA" or scores is None or scores.empty:
            return 0
        projected = self._aligned(scores.to_frame("score"), names, "score")
        self._plans[(market, EventKind.US_OVERNIGHT_SPILLOVER)].execute(
            arena, {"projected_us_spillover": np.nan_to_num(projected)}
        )
        return 1

    @staticmethod
    def _clear_snapshot_features(arena: ZeroAllocArena) -> None:
        """Clear latest-snapshot nodes before applying the current PIT snapshots.

        OHLCV state is incremental. Filing, event and technology nodes represent the
        current PIT snapshot, so stale values from a previous snapshot must not survive
        when a row is withdrawn or the run is moved back to an earlier as-of date.
        """
        for feature in (
            "core_fundamental_state",
            "india_promoter_pledge_penalty",
            "f_fundamentals_india",
            "f_fundamentals_us",
            "f_tech_patent_velocity",
            "f_policy_geopol_accumulator",
            "f_us_to_india_spillover",
        ):
            if feature in arena.node_to_slot:
                arena.read(feature)[:] = 0.0

    @staticmethod
    def _prepare_prices(prices: pd.DataFrame, names: tuple[str, ...], as_of: dt.date | pd.Timestamp) -> pd.DataFrame:
        if prices is None or prices.empty:
            return pd.DataFrame(columns=["ticker", "date", "close", "adj_close", "high", "low"])
        frame = prices.copy()
        frame["ticker"] = frame["ticker"].astype(str)
        frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
        frame = frame[frame["ticker"].isin(names) & frame["date"].notna()]
        frame = frame[frame["date"] <= pd.Timestamp(as_of).normalize()]
        if "knowledge_timestamp" in frame.columns:
            known_at = pd.to_datetime(frame["knowledge_timestamp"], utc=True, errors="coerce")
            cutoff = pd.Timestamp(as_of)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            frame = frame[known_at.notna() & (known_at <= cutoff)]
        columns = [
            column
            for column in FlexModelReplayRuntime._PRICE_DIGEST_COLUMNS
            if column in frame
        ]
        return frame[columns].sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")

    @staticmethod
    def _prepare_actions(actions: pd.DataFrame, market: str, as_of: dt.date | pd.Timestamp) -> pd.DataFrame:
        if actions is None or actions.empty:
            return pd.DataFrame(columns=["ticker", "market", "ex_date", "action_type", "split_ratio", "cash_dividend", "knowledge_timestamp"])
        frame = actions.copy()
        frame["market"] = frame["market"].astype(str)
        frame["ticker"] = frame["ticker"].astype(str)
        frame["ex_date"] = pd.to_datetime(frame["ex_date"], errors="coerce")
        frame["knowledge_timestamp"] = pd.to_datetime(frame["knowledge_timestamp"], utc=True, errors="coerce")
        cutoff = pd.Timestamp(as_of)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return frame[
            (frame["market"] == market)
            & frame["ticker"].notna()
            & frame["ex_date"].notna()
            & (frame["ex_date"] <= pd.Timestamp(as_of).normalize())
            & (frame["knowledge_timestamp"] <= cutoff)
        ].sort_values(["ex_date", "ticker", "action_type"])

    def _compatible_checkpoint(
        self,
        saved: Mapping[str, Any] | None,
        arena: ZeroAllocArena,
        prices: pd.DataFrame,
        actions: pd.DataFrame,
        as_of: dt.date | pd.Timestamp,
    ) -> pd.Timestamp | None:
        if not saved or saved.get("schema_version") != self._SCHEMA_VERSION:
            return None
        if saved.get("market") != arena.market or tuple(saved.get("tickers") or ()) != arena.tickers:
            return None
        saved_as_of = saved.get("as_of")
        if saved_as_of and pd.Timestamp(saved_as_of).date() > pd.Timestamp(as_of).date():
            return None
        raw_date = saved.get("last_bar_date")
        if not raw_date:
            return None
        checkpoint = pd.Timestamp(raw_date).normalize()
        old_prices = prices[pd.to_datetime(prices["date"]) <= checkpoint]
        old_actions = actions[pd.to_datetime(actions["ex_date"]) <= checkpoint]
        if self._frame_digest(old_prices, self._PRICE_DIGEST_COLUMNS) != saved.get("price_digest"):
            return None
        if self._frame_digest(old_actions, self._ACTION_DIGEST_COLUMNS) != saved.get("action_digest"):
            return None
        return checkpoint

    @staticmethod
    def _aligned(frame: pd.DataFrame, names: tuple[str, ...], column: str, *, fallback: str | None = None) -> np.ndarray:
        source = column if column in frame.columns else fallback
        if source is None or source not in frame.columns:
            return np.full(len(names), np.nan, dtype=np.float64)
        values = pd.to_numeric(frame[source], errors="coerce").reindex(names)
        return values.to_numpy(dtype=np.float64)

    @staticmethod
    def _frame_digest(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
        available = [column for column in columns if column in frame.columns]
        canonical = frame[available].copy()
        if not canonical.empty:
            sort_columns = [column for column in ("date", "ex_date", "ticker", "action_type") if column in canonical]
            canonical = canonical.sort_values(sort_columns).reset_index(drop=True)
        digest = hashlib.sha256()
        digest.update("\0".join(available).encode("utf-8"))
        if not canonical.empty:
            digest.update(pd.util.hash_pandas_object(canonical, index=False).values.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _read_snapshot(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = FlexModelReplayRuntime._json_arrays(snapshot)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(serializable, separators=(",", ":"), allow_nan=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    @staticmethod
    def _json_arrays(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): FlexModelReplayRuntime._json_arrays(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FlexModelReplayRuntime._json_arrays(item) for item in value]
        return value
