"""Event-driven feature registry and static execution-plan compiler."""

from .engine import (
    CompiledExecutionPlan,
    EventKind,
    FlexModelRegistry,
    FlexNode,
    ZeroAllocArena,
)

__all__ = [
    "CompiledExecutionPlan",
    "EventKind",
    "FlexModelRegistry",
    "FlexNode",
    "ZeroAllocArena",
]
