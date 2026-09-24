# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the out-of-band tool-call provenance context channel."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from chrys.foundation.tool_call_context import (
    MAX_CONTEXT_SEGMENT_KEYS,
    MAX_CONTEXT_VALUE_CHARS,
    get_tool_context,
    resolve_tool_call_context,
    sanitize_context_segment,
    set_tool_context,
    set_tool_context_builder,
)


def _tool(name: str = "probe") -> SimpleNamespace:
    return SimpleNamespace(name=name)


# --------------------------------------------------------------------------- #
# Static segment: set/get
# --------------------------------------------------------------------------- #


def test_set_get_round_trip() -> None:
    tool = _tool()
    set_tool_context(tool, {"server_name": "linear", "remote_name": "create_issue"})
    assert get_tool_context(tool) == {"server_name": "linear", "remote_name": "create_issue"}


def test_get_returns_none_when_unset_or_empty() -> None:
    assert get_tool_context(_tool()) is None
    tool = _tool()
    set_tool_context(tool, {})
    assert get_tool_context(tool) is None


def test_set_sanitizes_static_segment_at_registration() -> None:
    tool = _tool()
    set_tool_context(tool, {"ok": "value", "bad": ["not", "scalar"], "": "empty-key"})
    assert get_tool_context(tool) == {"ok": "value"}


def test_set_with_oversized_segment_stores_empty_and_does_not_raise() -> None:
    """A provenance cap must never prevent a tool from loading (exact-or-absent)."""
    tool = _tool()
    oversized = {f"key_{i}": "v" * 100 for i in range(12)}
    set_tool_context(tool, oversized)
    assert get_tool_context(tool) is None


# --------------------------------------------------------------------------- #
# Segment sanitizer (single authority for both write points)
# --------------------------------------------------------------------------- #


def test_sanitize_keeps_scalars_and_drops_non_scalars() -> None:
    segment = {"s": "text", "i": 3, "f": 1.5, "b": True, "d": {"nested": 1}, "l": [1], "n": None}
    assert sanitize_context_segment(segment) == {"s": "text", "i": 3, "f": 1.5, "b": True}


def test_sanitize_drops_over_cap_value_with_its_key_never_truncates() -> None:
    exact = "x" * MAX_CONTEXT_VALUE_CHARS
    over = "x" * (MAX_CONTEXT_VALUE_CHARS + 1)
    result = sanitize_context_segment({"exact": exact, "over": over})
    assert result == {"exact": exact}, "over-cap identifiers must vanish, not be mangled"


def test_sanitize_drops_non_finite_floats_and_output_is_strict_json_safe() -> None:
    """NaN/Infinity would reach session.json as non-standard JSON — exact-or-absent applies."""
    segment = {
        "nan": float("nan"),
        "inf": float("inf"),
        "ninf": float("-inf"),
        "ok_float": 1.5,
        "ok_int": 3,
    }
    result = sanitize_context_segment(segment)
    assert result == {"ok_float": 1.5, "ok_int": 3}
    json.dumps(result, allow_nan=False)  # must not raise: strict-JSON invariant


def test_resolve_drops_non_finite_builder_values() -> None:
    tool = _tool()
    set_tool_context_builder(tool, lambda args: {"score": float("inf"), "skill_name": "pdf"})
    assert resolve_tool_call_context(tool, {}) == {"skill_name": "pdf"}


def test_sanitize_round_trips_identifiers_byte_exact() -> None:
    name = "scripts/Rendér-PDF_v2.py"
    assert sanitize_context_segment({"script_name": name})["script_name"] == name


def test_sanitize_drops_whole_segment_over_key_bound() -> None:
    segment = {f"k{i}": "v" for i in range(MAX_CONTEXT_SEGMENT_KEYS + 1)}
    assert sanitize_context_segment(segment) == {}


def test_sanitize_drops_whole_segment_over_serialized_budget() -> None:
    # Each value passes the per-value cap, but the segment total blows the budget.
    segment = {f"key_{i}": "v" * 200 for i in range(8)}
    assert sanitize_context_segment(segment) == {}


# --------------------------------------------------------------------------- #
# Resolve: static + builder merge
# --------------------------------------------------------------------------- #


def test_resolve_without_builder_returns_static_copy() -> None:
    tool = _tool()
    set_tool_context(tool, {"server_name": "linear"})
    resolved = resolve_tool_call_context(tool, {})
    assert resolved == {"server_name": "linear"}
    resolved["mutated"] = True
    assert get_tool_context(tool) == {"server_name": "linear"}, "resolve must not expose the stored segment"


def test_resolve_merges_builder_output_from_args() -> None:
    tool = _tool()

    def builder(args: Any) -> dict[str, Any]:
        return {"skill_name": str(args.get("skill_name", "")).lower()}

    set_tool_context_builder(tool, builder)
    assert resolve_tool_call_context(tool, {"skill_name": "PDF"}) == {"skill_name": "pdf"}


def test_resolve_static_wins_on_collision_and_keeps_disjoint_builder_keys() -> None:
    tool = _tool()
    set_tool_context(tool, {"server_name": "static-truth"})
    set_tool_context_builder(tool, lambda args: {"server_name": "builder-lie", "extra": "kept"})
    assert resolve_tool_call_context(tool, {}) == {"server_name": "static-truth", "extra": "kept"}


def test_resolve_builder_exception_keeps_static_only() -> None:
    tool = _tool()
    set_tool_context(tool, {"server_name": "linear"})

    def builder(args: Any) -> dict[str, Any]:
        raise RuntimeError("provenance must never break tool execution")

    set_tool_context_builder(tool, builder)
    assert resolve_tool_call_context(tool, {}) == {"server_name": "linear"}


def test_resolve_builder_non_mapping_output_keeps_static_only() -> None:
    tool = _tool()
    set_tool_context(tool, {"server_name": "linear"})
    set_tool_context_builder(tool, lambda args: "not a mapping")
    assert resolve_tool_call_context(tool, {}) == {"server_name": "linear"}


def test_resolve_over_budget_builder_segment_drops_whole_while_static_stands() -> None:
    """Per-segment atomicity: a bad builder segment cannot take the static segment down."""
    tool = _tool()
    set_tool_context(tool, {"server_name": "linear"})
    set_tool_context_builder(tool, lambda args: {f"key_{i}": "v" * 200 for i in range(8)})
    assert resolve_tool_call_context(tool, {}) == {"server_name": "linear"}


def test_resolve_over_budget_static_segment_still_resolves_builder() -> None:
    tool = _tool()
    set_tool_context(tool, {f"key_{i}": "v" * 200 for i in range(8)})
    set_tool_context_builder(tool, lambda args: {"skill_name": "pdf"})
    assert resolve_tool_call_context(tool, {}) == {"skill_name": "pdf"}


def test_resolve_with_nothing_registered_returns_empty() -> None:
    assert resolve_tool_call_context(_tool(), {"any": "args"}) == {}
