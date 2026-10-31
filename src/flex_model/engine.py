"""Static event-plan compiler and preallocated NumPy feature/state arena.

The arena avoids per-node DataFrame construction and preallocates feature vectors
and ring buffers. Python/NumPy kernels can still allocate temporary arrays; this
is allocation-conscious infrastructure, not a hard real-time zero-allocation
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Mapping

import numpy as np


class EventKind(Enum):
    CORP_ACTION_EX_DATE = auto()
    EOD_BAR = auto()
    XBRL_FUNDAMENTAL_FILING = auto()
    POLICY_OR_GEOPOL_SHOCK = auto()
    PATENT_RND_UPDATE = auto()
    US_OVERNIGHT_SPILLOVER = auto()


MARKETS = frozenset({"INDIA", "US"})
VectorKernel = Callable[["ZeroAllocArena"], np.ndarray]


@dataclass
class ZeroAllocArena:
    """Market feature slots and fixed-size per-ticker rolling buffers."""

    tickers: tuple[str, ...]
    market: str
    node_to_slot: dict[str, int]
    values: np.ndarray
    ring_buffers: dict[str, np.ndarray] = field(default_factory=dict)
    ring_cursors: dict[str, int] = field(default_factory=dict)
    extra_state: dict[str, np.ndarray] = field(default_factory=dict)
    event_payload: Mapping[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def allocate(
        cls,
        tickers: list[str] | tuple[str, ...],
        market: str,
        node_names: list[str] | tuple[str, ...],
        buffer_specs: Mapping[str, int],
        initial_values: Mapping[str, float] | None = None,
    ) -> "ZeroAllocArena":
        _validate_market(market)
        names = tuple(str(ticker) for ticker in tickers)
        if len(names) != len(set(names)):
            raise ValueError("tickers must be unique")
        nodes = tuple(node_names)
        if len(nodes) != len(set(nodes)):
            raise ValueError("node names must be unique")
        slots = {name: idx for idx, name in enumerate(nodes)}
        values = np.zeros((len(names), len(nodes)), dtype=np.float64, order="C")
        for feature, initial in (initial_values or {}).items():
            if feature not in slots:
                raise ValueError(f"initial value provided for unknown node {feature!r}")
            values[:, slots[feature]] = float(initial)
        rings: dict[str, np.ndarray] = {}
        for name, window in buffer_specs.items():
            if int(window) < 1:
                raise ValueError(f"ring buffer {name!r} must have a positive window")
            rings[name] = np.full((len(names), int(window)), np.nan, dtype=np.float64, order="C")
        return cls(
            tickers=names,
            market=market,
            node_to_slot=slots,
            values=values,
            ring_buffers=rings,
            ring_cursors={name: 0 for name in rings},
        )

    def read(self, node_name: str) -> np.ndarray:
        """Return a mutable, zero-copy vector view for a declared feature slot."""
        try:
            return self.values[:, self.node_to_slot[node_name]]
        except KeyError as error:
            raise KeyError(f"feature {node_name!r} is not allocated for {self.market}") from error

    def write(self, node_name: str, vec: np.ndarray) -> None:
        target = self.read(node_name)
        values = np.asarray(vec)
        if values.shape != target.shape:
            raise ValueError(
                f"feature {node_name!r} has shape {values.shape}; expected {target.shape}"
            )
        np.copyto(target, values, casting="same_kind")

    def push_ring(self, buf_name: str, vec: np.ndarray) -> np.ndarray:
        try:
            buf = self.ring_buffers[buf_name]
        except KeyError as error:
            raise KeyError(f"ring buffer {buf_name!r} was not allocated") from error
        values = np.asarray(vec)
        if values.shape != (len(self.tickers),):
            raise ValueError(f"ring values must have shape ({len(self.tickers)},)")
        cursor = self.ring_cursors[buf_name]
        np.copyto(buf[:, cursor], values, casting="same_kind")
        self.ring_cursors[buf_name] = (cursor + 1) % buf.shape[1]
        return buf

    def ensure_state(self, name: str, *, fill: float = 0.0, dtype=np.float64) -> np.ndarray:
        """Return an allocated per-ticker state vector, allocating only on first use."""
        if name not in self.extra_state:
            self.extra_state[name] = np.full(len(self.tickers), fill, dtype=dtype)
        state = self.extra_state[name]
        if state.shape != (len(self.tickers),):
            raise ValueError(f"state {name!r} has invalid shape {state.shape}")
        return state

    def set_payload(self, payload: Mapping[str, np.ndarray]) -> None:
        """Bind event vectors after validating their ticker alignment."""
        for key, value in payload.items():
            array = np.asarray(value)
            if array.shape != (len(self.tickers),):
                raise ValueError(
                    f"event payload {key!r} has shape {array.shape}; "
                    f"expected ({len(self.tickers)},) aligned to arena tickers"
                )
        self.event_payload = payload

    def export_feature_matrix(self, feature_names: list[str] | tuple[str, ...]):
        """Copy selected features into a pandas frame at the batch/model boundary."""
        import pandas as pd

        unknown = [name for name in feature_names if name not in self.node_to_slot]
        if unknown:
            raise KeyError(f"features are not allocated for {self.market}: {unknown}")
        data = {name: self.read(name).copy() for name in feature_names}
        return pd.DataFrame(data, index=self.tickers)

    def export_state(self) -> dict[str, np.ndarray | dict[str, object]]:
        """Return an owned snapshot suitable for persistence by the caller."""
        return {
            "values": self.values.copy(),
            "ring_buffers": {name: value.copy() for name, value in self.ring_buffers.items()},
            "ring_cursors": dict(self.ring_cursors),
            "extra_state": {name: value.copy() for name, value in self.extra_state.items()},
        }

    def restore_state(self, snapshot: Mapping[str, object]) -> None:
        """Restore an arena snapshot after strict shape and key validation."""
        values = np.asarray(snapshot.get("values"))
        if values.shape != self.values.shape:
            raise ValueError("snapshot feature matrix shape does not match arena")
        np.copyto(self.values, values, casting="same_kind")
        for name, buffer in self.ring_buffers.items():
            saved = np.asarray((snapshot.get("ring_buffers") or {}).get(name))
            if saved.shape != buffer.shape:
                raise ValueError(f"snapshot ring buffer {name!r} shape does not match arena")
            np.copyto(buffer, saved, casting="same_kind")
        cursors = snapshot.get("ring_cursors") or {}
        for name, cursor in self.ring_cursors.items():
            saved_cursor = int(cursors.get(name, -1))
            if not 0 <= saved_cursor < self.ring_buffers[name].shape[1]:
                raise ValueError(f"snapshot ring cursor {name!r} is invalid")
            self.ring_cursors[name] = saved_cursor
        saved_states = snapshot.get("extra_state") or {}
        restored: dict[str, np.ndarray] = {}
        for name, raw in saved_states.items():
            saved = np.asarray(raw)
            if saved.shape != (len(self.tickers),):
                raise ValueError(f"snapshot state {name!r} does not match arena ticker shape")
            if name in self.extra_state:
                state = self.extra_state[name]
                if saved.dtype != state.dtype:
                    raise ValueError(f"snapshot state {name!r} dtype does not match arena")
                np.copyto(state, saved)
                restored[name] = state
            else:
                restored[name] = saved.copy()
        self.extra_state = restored


@dataclass(frozen=True)
class FlexNode:
    name: str
    kind: str
    fn: VectorKernel
    reads: tuple[str, ...]
    markets: frozenset[str]
    event_triggers: frozenset[EventKind] = frozenset()
    ring_buffer_window: int = 0
    is_model_output: bool = False
    initial_value: float = 0.0


@dataclass(frozen=True)
class CompiledExecutionPlan:
    market: str
    event_kind: EventKind
    ordered_steps: tuple[FlexNode, ...]

    def execute(self, arena: ZeroAllocArena, payload: Mapping[str, np.ndarray]) -> None:
        if arena.market != self.market:
            raise ValueError(f"plan is for {self.market}, arena is for {arena.market}")
        arena.set_payload(payload)
        for step in self.ordered_steps:
            result = step.fn(arena)
            arena.write(step.name, result)

    def describe_plan(self) -> list[str]:
        return [
            f"{idx:02d}. [{node.kind}] {node.name} (reads={list(node.reads)})"
            for idx, node in enumerate(self.ordered_steps, 1)
        ]


class FlexModelRegistry:
    """Declarative registry with market/event filtering and deterministic plans."""

    def __init__(self) -> None:
        self.nodes: dict[str, FlexNode] = {}

    @staticmethod
    def _markets(markets: set[str] | frozenset[str] | None) -> frozenset[str]:
        result = frozenset(markets or MARKETS)
        invalid = result - MARKETS
        if invalid or not result:
            raise ValueError(f"markets must be a non-empty subset of {sorted(MARKETS)}; got {sorted(result)}")
        return result

    def _register(self, node: FlexNode) -> None:
        if not node.name.isidentifier():
            raise ValueError(f"invalid feature name {node.name!r}")
        if node.name in self.nodes:
            raise ValueError(f"feature {node.name!r} is already registered")
        if node.ring_buffer_window < 0:
            raise ValueError("ring buffer window cannot be negative")
        if len(node.reads) != len(set(node.reads)):
            raise ValueError(f"feature {node.name!r} declares duplicate reads")
        self.nodes[node.name] = node

    def handler(
        self,
        *,
        on: EventKind | set[EventKind] | frozenset[EventKind],
        reads: list[str] | tuple[str, ...] | None = None,
        markets: set[str] | frozenset[str] | None = None,
        ring_window: int = 0,
        output: bool = False,
        initial_value: float = 0.0,
    ):
        triggers = frozenset({on} if isinstance(on, EventKind) else on)
        if not triggers or any(not isinstance(kind, EventKind) for kind in triggers):
            raise ValueError("handlers require one or more EventKind triggers")

        def decorator(fn: VectorKernel) -> VectorKernel:
            self._register(
                FlexNode(
                    fn.__name__, "HANDLER", fn, tuple(reads or ()), self._markets(markets),
                    triggers, int(ring_window), bool(output), float(initial_value)
                )
            )
            return fn

        return decorator

    def eval(
        self,
        *,
        reads: list[str] | tuple[str, ...] | None = None,
        markets: set[str] | frozenset[str] | None = None,
        output: bool = True,
        initial_value: float = 0.0,
    ):
        def decorator(fn: VectorKernel) -> VectorKernel:
            self._register(
                FlexNode(
                    fn.__name__, "EVAL", fn, tuple(reads or ()), self._markets(markets),
                    is_model_output=bool(output), initial_value=float(initial_value)
                )
            )
            return fn

        return decorator

    def _market_nodes(self, market: str) -> dict[str, FlexNode]:
        _validate_market(market)
        nodes = {name: node for name, node in self.nodes.items() if market in node.markets}
        for node in nodes.values():
            for dependency in node.reads:
                if dependency not in self.nodes:
                    raise ValueError(f"{node.name!r} reads unregistered feature {dependency!r}")
                if market not in self.nodes[dependency].markets:
                    raise ValueError(
                        f"{node.name!r} ({market}) reads {dependency!r}, which is not registered for {market}"
                    )
        indegree = {name: 0 for name in nodes}
        consumers: dict[str, list[str]] = {name: [] for name in nodes}
        for name, node in nodes.items():
            for dependency in node.reads:
                indegree[name] += 1
                consumers[dependency].append(name)
        ready = [name for name, degree in indegree.items() if degree == 0]
        seen = 0
        while ready:
            name = ready.pop()
            seen += 1
            for consumer in consumers[name]:
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    ready.append(consumer)
        if seen != len(nodes):
            cycle = sorted(name for name, degree in indegree.items() if degree)
            raise ValueError(f"cycle detected in Flex graph for {market}: {cycle}")
        return nodes

    def allocate_arena(self, tickers: list[str] | tuple[str, ...], market: str) -> ZeroAllocArena:
        nodes = self._market_nodes(market)
        return ZeroAllocArena.allocate(
            tickers,
            market,
            list(nodes),
            {name: node.ring_buffer_window for name, node in nodes.items() if node.ring_buffer_window},
            {name: node.initial_value for name, node in nodes.items()},
        )

    def compile_plan(self, market: str, event_kind: EventKind) -> CompiledExecutionPlan:
        """Compile only triggered handlers and evals needed for reachable outputs."""
        if not isinstance(event_kind, EventKind):
            raise TypeError("event_kind must be an EventKind")
        nodes = self._market_nodes(market)
        triggered = {
            name for name, node in nodes.items()
            if node.kind == "HANDLER" and event_kind in node.event_triggers
        }
        active = set(triggered)

        # Find downstream evals affected by this event. Only model outputs and the
        # intermediate evals required to compute them are then retained.
        affected = set(triggered)
        changed = True
        while changed:
            changed = False
            for name, node in nodes.items():
                if node.kind == "EVAL" and name not in affected and any(dep in affected for dep in node.reads):
                    affected.add(name)
                    changed = True
        needed = {name for name in affected if nodes[name].is_model_output}
        active.update(needed)
        # Triggered handlers run even when no model output consumes them, so their
        # declared Eval inputs must also be prepared before their handler executes.
        stack = list(active)
        while stack:
            name = stack.pop()
            node = nodes[name]
            for dependency in node.reads:
                dep_node = nodes[dependency]
                if dep_node.kind == "EVAL" and dependency not in active:
                    active.add(dependency)
                    stack.append(dependency)
                elif dep_node.kind == "HANDLER" and dependency in triggered:
                    active.add(dependency)
                    if dependency not in stack:
                        stack.append(dependency)
        # Triggered state handlers must run even when they are not model outputs.
        active.update(triggered)

        # Every active node's dependencies that are also active must be scheduled first.
        degree = {name: 0 for name in active}
        downstream: dict[str, list[str]] = {name: [] for name in active}
        for name in active:
            for dependency in nodes[name].reads:
                if dependency in active:
                    degree[name] += 1
                    downstream[dependency].append(name)
        ready = sorted(name for name, count in degree.items() if count == 0)
        ordered: list[FlexNode] = []
        while ready:
            name = ready.pop(0)
            ordered.append(nodes[name])
            for consumer in sorted(downstream[name]):
                degree[consumer] -= 1
                if degree[consumer] == 0:
                    ready.append(consumer)
                    ready.sort()
        if len(ordered) != len(active):
            cycle = sorted(name for name, count in degree.items() if count)
            raise ValueError(f"cycle detected in Flex graph for {market}/{event_kind.name}: {cycle}")
        return CompiledExecutionPlan(market, event_kind, tuple(ordered))


def _validate_market(market: str) -> None:
    if market not in MARKETS:
        raise ValueError(f"unsupported market {market!r}; expected one of {sorted(MARKETS)}")
