from __future__ import annotations

import numpy as np
import pytest

from src.flex_model.aladdin_features import flex
from src.flex_model.engine import EventKind, FlexModelRegistry


def test_market_and_event_plans_prune_unrelated_nodes() -> None:
    india = flex.compile_plan("INDIA", EventKind.XBRL_FUNDAMENTAL_FILING)
    us = flex.compile_plan("US", EventKind.XBRL_FUNDAMENTAL_FILING)
    assert [node.name for node in india.ordered_steps] == [
        "core_fundamental_state",
        "india_promoter_pledge_penalty",
        "f_fundamentals_india",
    ]
    assert [node.name for node in us.ordered_steps] == [
        "core_fundamental_state",
        "f_fundamentals_us",
    ]

    us_arena = flex.allocate_arena(["A", "B"], "US")
    assert "india_promoter_pledge_penalty" not in us_arena.node_to_slot


def test_eod_plan_reads_state_updated_by_prior_corporate_action() -> None:
    arena = flex.allocate_arena(["A", "B"], "INDIA")
    action_plan = flex.compile_plan("INDIA", EventKind.CORP_ACTION_EX_DATE)
    action_plan.execute(
        arena,
        {"price_adjustment_factor": np.array([0.5, 1.0])},
    )
    eod_plan = flex.compile_plan("INDIA", EventKind.EOD_BAR)
    eod_plan.execute(
        arena,
        {
            "close": np.array([50.0, 100.0]),
            "high": np.array([51.0, 101.0]),
            "low": np.array([49.0, 99.0]),
        },
    )
    assert arena.read("split_adjusted_close") == pytest.approx([100.0, 100.0])
    assert np.isfinite(arena.read("f_kalman_velocity")).all()
    assert arena.ring_cursors["corwin_schultz_effective_spread"] == 1


def test_event_payload_and_feature_shapes_are_checked() -> None:
    arena = flex.allocate_arena(["A", "B"], "US")
    plan = flex.compile_plan("US", EventKind.POLICY_OR_GEOPOL_SHOCK)
    with pytest.raises(ValueError, match="shape"):
        plan.execute(arena, {"decay_factor": np.ones(1), "bge_projected_shock": np.ones(2)})
    with pytest.raises(ValueError, match="requires"):
        plan.execute(arena, {"decay_factor": np.ones(2)})


def test_arena_snapshot_restores_values_ring_buffers_and_state() -> None:
    arena = flex.allocate_arena(["A", "B"], "INDIA")
    plan = flex.compile_plan("INDIA", EventKind.EOD_BAR)
    plan.execute(
        arena,
        {"close": np.array([10.0, 20.0]), "high": np.array([11.0, 21.0]), "low": np.array([9.0, 19.0])},
    )
    snapshot = arena.export_state()
    restored = flex.allocate_arena(["A", "B"], "INDIA")
    restored.restore_state(snapshot)
    assert np.array_equal(restored.values, arena.values)
    assert np.array_equal(restored.ring_buffers["f_kalman_velocity"], arena.ring_buffers["f_kalman_velocity"], equal_nan=True)
    assert restored.ring_cursors == arena.ring_cursors
    assert np.array_equal(restored.extra_state["kalman_level"], arena.extra_state["kalman_level"])


def test_compiler_rejects_cycles_and_missing_market_dependencies() -> None:
    registry = FlexModelRegistry()

    @registry.eval(reads=["second"])
    def first(arena):
        return arena.read("second")

    @registry.eval(reads=["first"])
    def second(arena):
        return arena.read("first")

    with pytest.raises(ValueError, match="cycle detected"):
        registry.compile_plan("US", EventKind.EOD_BAR)

    broken = FlexModelRegistry()

    @broken.handler(on=EventKind.EOD_BAR, markets={"US"})
    def us_only(arena):
        return np.zeros(len(arena.tickers))

    @broken.eval(reads=["us_only"], markets={"INDIA"})
    def india_output(arena):
        return arena.read("us_only")

    with pytest.raises(ValueError, match="not registered for INDIA"):
        broken.compile_plan("INDIA", EventKind.EOD_BAR)


def test_triggered_handler_eval_inputs_are_scheduled_even_without_model_output() -> None:
    registry = FlexModelRegistry()

    @registry.eval(output=False)
    def current_mid(arena):
        return arena.event_payload["mid"]

    @registry.handler(on=EventKind.EOD_BAR, reads=["current_mid"])
    def update_buffer(arena):
        return arena.read("update_buffer") + arena.read("current_mid")

    plan = registry.compile_plan("US", EventKind.EOD_BAR)
    assert [node.name for node in plan.ordered_steps] == ["current_mid", "update_buffer"]


def test_export_feature_matrix_is_a_copy_at_the_boundary() -> None:
    arena = flex.allocate_arena(["A"], "US")
    arena.write("f_policy_geopol_accumulator", np.array([0.7]))
    frame = arena.export_feature_matrix(["f_policy_geopol_accumulator"])
    frame.iloc[0, 0] = -1.0
    assert arena.read("f_policy_geopol_accumulator")[0] == pytest.approx(0.7)
