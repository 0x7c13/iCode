# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Value envelope helpers: canonical JSON, strict JSON checks, wire shapes, default combine."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.service.workflows.sdk import SourceValue, WorkflowValue
from chrys.service.workflows.values import (
    ValueShapeError,
    canonical_json,
    check_json_value,
    default_combine,
    source_to_wire,
    value_from_wire,
    value_to_wire,
)


def test_canonical_json_is_sorted_compact_and_keeps_unicode() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "é", None, True]}) == '{"a":[1.5,"é",null,true],"b":1}'


def test_canonical_json_rejects_non_finite_floats() -> None:
    with pytest.raises(ValueError, match="Out of range"):
        canonical_json({"x": float("nan")})


def test_check_json_value_accepts_nested_json() -> None:
    check_json_value({"a": [1, 2.0, True, None, {"b": "c", "d": []}]})


@pytest.mark.parametrize(
    "payload",
    [(1,), {1: "a"}, {"a": {"b": object()}}, float("inf"), b"x", {"a": [float("nan")]}],
)
def test_check_json_value_rejects_non_json(payload: Any) -> None:
    with pytest.raises(ValueShapeError):
        check_json_value(payload)


def test_value_round_trips_through_the_wire() -> None:
    value = WorkflowValue(text="t", data={"k": [1, "x"]})
    assert value_to_wire(value) == {"text": "t", "data": {"k": [1, "x"]}}
    assert value_from_wire(value_to_wire(value)) == value
    assert value_from_wire({"text": "only"}) == WorkflowValue(text="only")


@pytest.mark.parametrize("payload", ["x", {"text": 1}, {"text": "a", "data": {1: 2}}, {"data": None}])
def test_value_from_wire_rejects_bad_shapes(payload: Any) -> None:
    with pytest.raises(ValueShapeError):
        value_from_wire(payload)


def test_source_serializes_its_identity_and_value() -> None:
    source = SourceValue("b", "opaque", WorkflowValue("v", {"n": 1}))
    assert source_to_wire(source) == {
        "node_id": "b",
        "activation_id": "opaque",
        "value": {"text": "v", "data": {"n": 1}},
    }


def test_default_combine_titles_each_source_in_declared_order() -> None:
    sources = (
        SourceValue("b", "b@iter#1", WorkflowValue("one", {"dropped": True})),
        SourceValue("c", "c@iter#1", WorkflowValue("two")),
    )
    assert default_combine(sources) == WorkflowValue(text="## b\none\n\n## c\ntwo")
