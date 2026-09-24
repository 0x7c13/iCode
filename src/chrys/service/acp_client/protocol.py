# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACP protocol validation and retained payload caps."""

from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any, Literal

from chrys.foundation.text.images import MAX_IMAGE_BYTES
from chrys.foundation.util.unicode_scalars import find_unpaired_surrogate

_MAX_PAYLOAD_BYTES = 1024 * 1024
_MAX_PAYLOAD_DEPTH = 32
_MAX_COLLECTION_ITEMS = 4_096
_MAX_STRING_CHARS = 256 * 1024
_MAX_BASE64_IMAGE_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
_MAX_REQUEST_ID_CHARS = 4_096
_MAX_METHOD_CHARS = 1_024

_PERMISSION_METHOD = "session/request_permission"
_ASK_USER_METHOD = "_chrys/request_input"


def validate_json_scalar_tree(value: Any) -> None:
    """Validate every JSON key, value, and sequence element recursively."""
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite.")
        return
    if type(value) is str:
        _validate_surrogates(value)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings.")
            _validate_surrogates(key)
            validate_json_scalar_tree(item)
        return
    if type(value) in {list, tuple}:
        for item in value:
            validate_json_scalar_tree(item)
        return
    raise ValueError(f"Unsupported JSON value type: {type(value).__name__}")


def _validate_surrogates(value: str) -> None:
    if find_unpaired_surrogate(value) >= 0:
        raise ValueError("JSON strings cannot contain unpaired surrogates.")


def encode_protocol_json(value: Any) -> bytes:
    """Encode a protocol-bound value after the shared scalar validation pass."""
    validate_json_scalar_tree(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def parse_protocol_json(raw: bytes) -> Any:
    """Parse one JSON value while rejecting JavaScript non-finite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON numeric constant: {value}")

    parsed = json.loads(raw, parse_constant=reject_constant)
    validate_json_scalar_tree(parsed)
    return parsed


_ENVELOPE_MEMBERS = frozenset({"jsonrpc", "id", "method", "params", "result", "error"})


def validate_json_rpc_envelope(message: Any) -> Literal["request", "notification", "response"]:
    """Validate the strict JSON-RPC envelope accepted by the SDK-facing reader."""
    if type(message) is not dict:
        raise ValueError("JSON-RPC frame root must be an object.")
    # The per-kind caps run on params/result only, and the SDK re-parses the
    # complete frame; an unknown top-level member would ride around both.
    if not _ENVELOPE_MEMBERS.issuperset(message):
        raise ValueError("JSON-RPC frame contains unknown envelope members.")
    if message.get("jsonrpc") != "2.0" or type(message.get("jsonrpc")) is not str:
        raise ValueError("JSON-RPC frame must declare version 2.0.")

    has_id = "id" in message
    has_method = "method" in message
    has_result = "result" in message
    has_error = "error" in message

    if has_id and type(message["id"]) not in {str, int}:
        raise ValueError("JSON-RPC ids must be exact strings or integers.")
    # Ids and methods are retained (inbound map keys, outgoing records) and
    # echoed into response frames, so they carry their own §7.4-style caps.
    if has_id and type(message["id"]) is str and len(message["id"]) > _MAX_REQUEST_ID_CHARS:
        raise ValueError("JSON-RPC id exceeds the retained-field cap.")
    if has_id and type(message["id"]) is int and not -(1 << 63) <= message["id"] < 1 << 63:
        raise ValueError("JSON-RPC id exceeds the interoperable integer range.")
    if has_method and type(message["method"]) is not str:
        raise ValueError("JSON-RPC methods must be strings.")
    if has_method and len(message["method"]) > _MAX_METHOD_CHARS:
        raise ValueError("JSON-RPC method exceeds the retained-field cap.")
    if "params" in message:
        if not has_method:
            raise ValueError("JSON-RPC responses cannot contain params.")
        if type(message["params"]) is not dict:
            raise ValueError("JSON-RPC params must be an object.")

    if has_method:
        if has_result or has_error:
            raise ValueError("JSON-RPC requests cannot contain result or error.")
        return "request" if has_id else "notification"

    if not has_id or has_result == has_error:
        raise ValueError("JSON-RPC responses require an id and exactly one of result or error.")
    if has_error:
        error = message["error"]
        if type(error) is not dict:
            raise ValueError("JSON-RPC error must be an object.")
        if type(error.get("code")) is not int or type(error.get("message")) is not str:
            raise ValueError("JSON-RPC error requires an exact integer code and string message.")
    return "response"


def _measure_payload(value: Any, *, depth: int = 0) -> tuple[int, int]:
    if depth > _MAX_PAYLOAD_DEPTH:
        raise ValueError("ACP payload nesting is too deep.")
    if type(value) is str:
        if len(value) > _MAX_STRING_CHARS:
            raise ValueError("ACP payload string is too large.")
        return len(value.encode("utf-8", errors="surrogatepass")), 1
    if type(value) is dict:
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("ACP payload object has too many fields.")
        total = 0
        items = 1
        for key, item in value.items():
            if len(key) > _MAX_STRING_CHARS:
                raise ValueError("ACP payload string is too large.")
            total += len(key.encode("utf-8", errors="surrogatepass"))
            child_bytes, child_items = _measure_payload(item, depth=depth + 1)
            total += child_bytes
            items += child_items
        return total, items
    if type(value) in {list, tuple}:
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("ACP payload sequence has too many items.")
        total = 0
        items = 1
        for item in value:
            child_bytes, child_items = _measure_payload(item, depth=depth + 1)
            total += child_bytes
            items += child_items
        return total, items
    return 16, 1


def _validate_payload_caps(value: Any) -> None:
    _total_bytes, total_items = _measure_payload(value)
    # Per-collection checks alone admit 4096-wide children at every level;
    # the aggregate bound is what actually caps the retained object count.
    if total_items > _MAX_COLLECTION_ITEMS:
        raise ValueError("ACP payload has too many items in total.")
    if len(encode_protocol_json(value)) > _MAX_PAYLOAD_BYTES:
        raise ValueError("ACP retained payload exceeds the byte limit.")


def _validate_update_image_data(data: str) -> None:
    """Validate one image payload before exempting its encoding overhead."""
    if not data:
        raise ValueError("ACP image payload is empty.")
    if len(data) > _MAX_BASE64_IMAGE_CHARS:
        raise ValueError("ACP image payload exceeds the supported size limit.")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ACP image payload is not valid base64.") from exc
    if not decoded:
        raise ValueError("ACP image payload is empty.")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("ACP image payload exceeds the supported size limit.")


def _payload_without_tool_image_data(value: Any) -> Any:
    """Replace validated tool-image encodings for retained-cap accounting."""
    if type(value) is not dict:
        return value
    update = value.get("update")
    if type(update) is not dict or update.get("sessionUpdate") not in {"tool_call", "tool_call_update"}:
        return value
    content = update.get("content")
    if type(content) is not list:
        return value

    bounded_content: list[Any] | None = None
    for index, item in enumerate(content):
        if type(item) is not dict or item.get("type") != "content":
            continue
        block = item.get("content")
        if type(block) is not dict or block.get("type") != "image" or type(block.get("mimeType")) is not str:
            continue
        data = block.get("data")
        if type(data) is not str:
            continue
        _validate_update_image_data(data)
        if bounded_content is None:
            bounded_content = list(content)
        bounded_content[index] = {**item, "content": {**block, "data": ""}}

    if bounded_content is None:
        return value
    return {**value, "update": {**update, "content": bounded_content}}


def _validate_update_payload_caps(value: Any) -> None:
    """Apply retained caps while excluding validated tool-image encoding overhead."""
    _validate_payload_caps(_payload_without_tool_image_data(value))
