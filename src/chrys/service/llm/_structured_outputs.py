# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Helpers shared by the Chat Completions and Responses wire clients for OpenAI Structured Outputs.

Includes strict-schema compatibility, schema materialization, fine-tuned-model
constraint rejection, and response-format name sanitisation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from openai.lib._pydantic import _ensure_strict_json_schema

logger = logging.getLogger(__name__)

# Structured-output schema names have a stricter, ASCII-only rule than the
# Chat Completions author-name field: [A-Za-z0-9_-], max 64 chars.
_INVALID_RESPONSE_FORMAT_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_response_format_name(name: object) -> str:
    """Sanitize a response-format schema name; fall back to ``"response"``."""
    sanitized = _INVALID_RESPONSE_FORMAT_NAME_CHARS_RE.sub("", str(name)) if name is not None else ""
    return sanitized[:64] or "response"


# Composition and dependency keywords Structured Outputs rejects under strict
# mode. anyOf is the supported one; single-entry allOf is inlined away by the
# strictifier, so any allOf that remains is the unsupported multi-entry form.
# The dependent* keywords need flagging by name: their values hold property
# names or arbitrary subschemas, which the object-closure leg cannot detect.
_STRICT_UNSUPPORTED_COMPOSITION_KEYWORDS = frozenset(
    {"allOf", "oneOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas"}
)


def _strict_schema_branch_incompatible(schema: Any) -> bool:
    """Return whether a strictified tree still contains strict-incompatible branches.

    The SDK strictifier only traverses keywords it supports (``properties``,
    ``items``, ``anyOf``, single-entry ``allOf``, ``$defs``), so objects
    inside anything else stay open — and the unsupported composition
    keywords are rejected by strict mode even with closed members. Flag all
    four shapes: a remaining unsupported composition keyword, an
    object-typed node not explicitly closed with ``additionalProperties:
    false`` (the strictifier closes every object it actually visits), an
    object-typed node whose ``required`` does not list exactly its
    ``properties`` (strict mode demands every field be required, and the
    strictifier stamps that only on objects it visits — an untraversed
    object can arrive pre-closed but under-required), and any other
    explicitly non-false ``additionalProperties``. The walk is
    deliberately context-blind — a literal property *named* like a keyword
    can false-positive — because the only cost is falling back to
    non-strict.
    """
    if isinstance(schema, Mapping):
        schema_type = schema.get("type")
        is_object_typed = schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type)
        if is_object_typed:
            if schema.get("additionalProperties") is not False:
                return True
            properties = schema.get("properties")
            if isinstance(properties, Mapping):
                required = schema.get("required")
                # Entries must be vetted before the set comparison: set()
                # raises on unhashable entries, and set equality alone would
                # let duplicate names satisfy the completeness check.
                if (
                    not isinstance(required, list)
                    or len(required) != len(properties)
                    or not all(isinstance(entry, str) for entry in required)
                    or set(required) != set(properties)
                ):
                    return True
        return any(
            key in _STRICT_UNSUPPORTED_COMPOSITION_KEYWORDS
            or (key == "additionalProperties" and value is not False)
            or _strict_schema_branch_incompatible(value)
            for key, value in schema.items()
        )
    if isinstance(schema, list):
        return any(_strict_schema_branch_incompatible(item) for item in schema)
    return False


def _strict_mode_incompatible(schema: Mapping[str, Any]) -> bool:
    """Return whether a strictified tree still violates the strict-mode contract.

    The SDK strictifier transforms but does not certify: Structured Outputs
    additionally requires the root to be an object schema without a
    root-level ``anyOf``, and non-object, type-less, and root-``anyOf``
    inputs all pass through it unchanged. Any violation falls back to
    non-strict rather than 400 on the wire.
    """
    if schema.get("type") != "object" or "anyOf" in schema:
        return True
    return _strict_schema_branch_incompatible(schema)


def _materialize_json_structure(value: Any) -> Any:
    """Recursively materialize mappings and sequences into plain containers.

    The schema copier where ``copy.deepcopy`` would fail: read-only views
    such as ``MappingProxyType`` are not deep-copyable (deepcopy falls back
    to pickling them, which raises), and callers hand schemas nested inside
    frozen structures. Mappings become dicts, lists and tuples become lists;
    scalars are shared — JSON-shaped schema leaves are immutable, and the
    in-place strictifier only ever mutates containers.
    """
    if isinstance(value, Mapping):
        return {key: _materialize_json_structure(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_materialize_json_structure(item) for item in value]
    return value


_FINE_TUNED_MODEL_PREFIX = "ft:"

# Constraint keywords Structured Outputs accepts on base models but rejects
# on fine-tuned ones: string bounds/pattern/format, number bounds, array
# bounds, and patternProperties.
_FINE_TUNE_UNSUPPORTED_CONSTRAINT_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "multipleOf",
        "patternProperties",
        "minItems",
        "maxItems",
    }
)


def _contains_fine_tune_unsupported_keyword(schema: Any) -> bool:
    if isinstance(schema, Mapping):
        return any(
            key in _FINE_TUNE_UNSUPPORTED_CONSTRAINT_KEYWORDS or _contains_fine_tune_unsupported_keyword(value)
            for key, value in schema.items()
        )
    if isinstance(schema, list):
        return any(_contains_fine_tune_unsupported_keyword(item) for item in schema)
    return False


def _fine_tuned_model_rejects_strict_schema(model: Any, schema: Any) -> bool:
    """Return whether strict mode must be withheld for a fine-tuned model.

    Fine-tuned (``ft:``-prefixed) models reject constraint keywords that
    base models accept under strict mode, so a schema carrying any of them
    goes to the wire non-strict for those models only. The walk is
    context-blind like the branch walk: a property literally named after a
    constraint keyword false-positives, and the only cost is the safe
    non-strict fallback.
    """
    if not (isinstance(model, str) and model.startswith(_FINE_TUNED_MODEL_PREFIX)):
        return False
    return _contains_fine_tune_unsupported_keyword(schema)


def _strictify_response_schema(schema: dict[str, Any], *, model: str | None = None) -> tuple[dict[str, Any], bool]:
    """Return a strict-compatible copy or the original schema with strict mode disabled."""
    try:
        # Strict mode rejects schemas missing ``required`` or nested
        # ``additionalProperties: false``; the SDK's recursive
        # strictifier fills both exactly the way it does for model classes.
        working = _materialize_json_structure(schema)
        working = _ensure_strict_json_schema(working, path=(), root=working)
    except Exception:
        # Shapes the strictifier rejects go to the wire non-strict,
        # exactly as the caller wrote them: the API enforces schema
        # completeness only when strict is set.
        logger.debug("response_format schema not strictifiable; sending non-strict", exc_info=True)
    else:
        # The strictifier transforms without certifying: a non-object
        # or ``anyOf`` root and an explicitly non-false
        # ``additionalProperties`` all survive it, and strict mode
        # rejects each — such schemas also go to the wire non-strict.
        if _strict_mode_incompatible(working):
            logger.debug("response_format schema incompatible with strict mode; sending non-strict")
        elif _fine_tuned_model_rejects_strict_schema(model, working):
            logger.debug("response_format constraint keywords unsupported for fine-tuned models; sending non-strict")
        else:
            return working, True
    return schema, False
