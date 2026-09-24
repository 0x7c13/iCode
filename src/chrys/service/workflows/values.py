# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Main-process helpers for the ``WorkflowValue`` envelope: canonical JSON, wire shapes, default combine."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from chrys.service.workflows.sdk import JsonValue, SourceValue, WorkflowValue


class ValueShapeError(ValueError):
    """A value crossing the process boundary is not the envelope the contract defines."""


def canonical_json(payload: Any) -> str:
    """Strict, deterministic JSON: sorted keys, no NaN/Infinity, UTF-8 text kept as-is."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def check_json_value(payload: Any, *, path: str = "data") -> None:
    """Reject anything outside strict recursive JSON (bool/int/float/str/None/list/dict)."""
    if payload is None or isinstance(payload, bool | str):
        return
    if isinstance(payload, int):
        return
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueShapeError(f"{path}: non-finite float is not JSON.")
        return
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            check_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(payload, dict):
        for key, item in payload.items():
            if not isinstance(key, str):
                raise ValueShapeError(f"{path}: object keys must be str, got {type(key).__name__}.")
            check_json_value(item, path=f"{path}.{key}")
        return
    raise ValueShapeError(f"{path}: {type(payload).__name__} is not a JSON value.")


def value_to_wire(value: WorkflowValue) -> dict[str, Any]:
    return {"text": value.text, "data": value.data}


def value_from_wire(payload: Any, *, where: str = "value") -> WorkflowValue:
    if not isinstance(payload, dict):
        raise ValueShapeError(f"{where}: expected an object, got {type(payload).__name__}.")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueShapeError(f"{where}.text: expected str.")
    data: JsonValue = payload.get("data")
    check_json_value(data, path=f"{where}.data")
    return WorkflowValue(text=text, data=data)


def source_to_wire(source: SourceValue) -> dict[str, Any]:
    return {
        "node_id": source.node_id,
        "activation_id": source.activation_id,
        "value": value_to_wire(source.value),
    }


def default_combine(sources: Sequence[SourceValue]) -> WorkflowValue:
    """Fan-in without a user combine: titled concatenation of every present source's text."""
    parts = [f"## {source.node_id}\n{source.value.text}" for source in sources]
    return WorkflowValue(text="\n\n".join(parts))
