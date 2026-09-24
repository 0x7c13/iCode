# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Compact, persistent metadata for one completed tool invocation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

EXECUTION_STAMP_KEY = "_exec_stamp"
"""Per-result ``additional_properties`` key for effective invocation metadata."""

EFFECTIVE_ARGS_MAX_CHARS = 2_048
"""Maximum stored canonical argument characters; longer values are middle-truncated."""

EXECUTION_STAMP_DIGEST_HEX_CHARS = 16
"""Number of leading SHA-256 hex characters stored for the full argument payload."""

ExecutionOutcome = Literal["ok", "error"]


def write_execution_stamp(
    metadata: MutableMapping[str, Any],
    arguments: Any,
    *,
    outcome: ExecutionOutcome,
    error_kind: str | None = None,
) -> None:
    """Write the canonical execution stamp for one completed invocation."""
    try:
        stamp = build_execution_stamp(
            arguments,
            outcome=outcome,
            error_kind=error_kind,
        )
    except Exception:
        # Execution stamps are observability/durability metadata. A custom
        # typed argument that cannot be represented must never turn an
        # otherwise successful tool invocation into a tool failure.
        logger.debug("Failed to serialize tool execution stamp", exc_info=True)
        return
    metadata[EXECUTION_STAMP_KEY] = stamp


def build_execution_stamp(
    arguments: Any,
    *,
    outcome: ExecutionOutcome,
    error_kind: str | None = None,
) -> dict[str, str]:
    """Return a compact stamp whose digest covers the uncapped canonical arguments."""
    serialized = canonicalize_effective_arguments(arguments)
    stamp = {
        "effective_args_digest": hashlib.sha256(serialized.encode("utf-8", errors="surrogatepass")).hexdigest()[
            :EXECUTION_STAMP_DIGEST_HEX_CHARS
        ],
        "effective_args": _middle_truncate(serialized, EFFECTIVE_ARGS_MAX_CHARS),
        "outcome": outcome,
    }
    if error_kind:
        stamp["error_kind"] = error_kind
    return stamp


def canonicalize_effective_arguments(arguments: Any) -> str:
    """Serialize JSON-like effective arguments deterministically and compactly."""
    payload = arguments.model_dump(mode="json") if isinstance(arguments, BaseModel) else arguments
    return json.dumps(
        _normalize_json_containers(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _normalize_json_containers(value: Any) -> Any:
    """Normalize typed container shapes before deterministic JSON encoding.

    Pydantic validation can turn JSON object keys into typed Python values
    such as :class:`date`. ``json.dumps(default=...)`` never applies its
    default handler to mapping keys, so normalize those keys explicitly.
    If normalization would collapse distinct keys, preserve the mapping as
    sorted key/value pairs instead of dropping data.
    """
    if isinstance(value, BaseModel):
        return _normalize_json_containers(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_json_containers(asdict(value))
    if isinstance(value, Mapping):
        normalized_items = [
            (_canonical_mapping_key(key), _normalize_json_containers(nested)) for key, nested in value.items()
        ]
        keys = [key for key, _nested in normalized_items]
        if len(keys) == len(set(keys)):
            return dict(normalized_items)
        items = [[_normalize_json_containers(key), _normalize_json_containers(nested)] for key, nested in value.items()]
        items.sort(key=_canonical_sort_key)
        return {"__mapping_items__": items}
    if isinstance(value, list | tuple):
        return [_normalize_json_containers(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [_normalize_json_containers(item) for item in value]
        items.sort(key=_canonical_sort_key)
        return items
    return value


def _canonical_mapping_key(value: Any) -> str:
    """Render one Python mapping key as its stable JSON-object key."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return json.dumps(value, ensure_ascii=False)
    converted = _json_default(value)
    if isinstance(converted, str):
        return converted
    return _canonical_sort_key(_normalize_json_containers(converted))


def _canonical_sort_key(value: Any) -> str:
    """Return a deterministic ordering key for normalized container values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def execution_stamp_from_metadata(metadata: Mapping[str, Any]) -> dict[str, str] | None:
    """Return a validated shallow copy of a stored execution stamp."""
    value = metadata.get(EXECUTION_STAMP_KEY)
    if not isinstance(value, Mapping):
        return None
    digest = value.get("effective_args_digest")
    arguments = value.get("effective_args")
    outcome = value.get("outcome")
    if not isinstance(digest, str) or not isinstance(arguments, str) or outcome not in {"ok", "error"}:
        return None
    stamp = {
        "effective_args_digest": digest,
        "effective_args": arguments,
        "outcome": outcome,
    }
    error_kind = value.get("error_kind")
    if isinstance(error_kind, str) and error_kind:
        stamp["error_kind"] = error_kind
    return stamp


def _middle_truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "…"
    available = max_chars - len(marker)
    left = (available + 1) // 2
    right = available - left
    return f"{value[:left]}{marker}{value[len(value) - right :]}"


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return str(value)
