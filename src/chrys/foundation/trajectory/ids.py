# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Identifier minting and format classification for trajectory events.

IDs are classified **by origin, not by suffix**:

* Chrys-generated analytics IDs (``event_id``, ``operation_id``, ``turn_id``,
  ``analytics_item_id``, ...) are 32 lowercase hex characters. Any ID field
  not registered below defaults to this class.
* Existing Chrys IDs keep their native shape: ``session_id`` is a canonical
  UUID string (pre-refactor sessions: 12 hex), ``invocation_id`` /
  ``sub_agent_log_id`` are 12 lowercase hex characters, profile IDs are
  bounded opaque strings compared by fingerprint, and a compressed block's
  ``compressed_context_id`` keeps the ``ctx_`` + 8 hex shape the session
  already persists and shows (so pre-existing blocks stay referenceable).
* External opaque IDs (provider ``response_id``, ``child_session_id``, ...)
  and user-configured identifiers such as ``hook_id`` are bounded printable
  strings and never validated as hex. A spilled tool result's ``artifact_id``
  is the artifact's file *name* — bounded, printable and separator-free, so a
  path can never be recorded in its place.

Reference fields inherit the format of the ID they point to, so validation
runs uniformly over every ``*_id`` key without per-field allow-lists.
"""

from __future__ import annotations

import re
import secrets
from enum import Enum

ANALYTICS_ID_HEX_LENGTH = 32
INVOCATION_ID_HEX_LENGTH = 12
OPAQUE_ID_MAX_LENGTH = 512

_ANALYTICS_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_INVOCATION_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_COMPRESSED_CONTEXT_ID_RE = re.compile(r"^ctx_[0-9a-f]{8}$")
ARTIFACT_ID_MAX_LENGTH = 128
_ARTIFACT_ID_SEPARATORS = frozenset({"/", "\\", "\x00"})


class IdClass(Enum):
    """Format class of an identifier field."""

    ANALYTICS = "analytics"
    SESSION = "session"
    INVOCATION = "invocation"
    COMPRESSED_CONTEXT = "compressed_context"
    ARTIFACT = "artifact"
    PROFILE = "profile"
    OPAQUE = "opaque"


_OPAQUE_FIELDS = frozenset(
    {
        "response_id",
        "child_session_id",
        "hook_id",
        "provider_request_id",
        "transport_exchange_id",
    }
)
_SESSION_FIELDS = frozenset({"session_id", "origin_session_id"})
_INVOCATION_FIELDS = frozenset({"invocation_id", "sub_agent_log_id"})
_COMPRESSED_CONTEXT_FIELDS = frozenset({"compressed_context_id"})
_ARTIFACT_FIELDS = frozenset({"artifact_id"})
_PROFILE_FIELDS = frozenset({"agent_profile_id", "model_profile_id"})


def new_analytics_id() -> str:
    """Mint a fresh 32-hex analytics identifier."""
    return secrets.token_hex(ANALYTICS_ID_HEX_LENGTH // 2)


def classify_id_field(field_name: str) -> IdClass:
    """Return the format class for an ID-bearing field name.

    Registered exact names win; everything else that looks like an ID
    (``*_id``) is an analytics ID by default, which is exactly the
    "any new ID falls into the analytics class unless registered" rule.
    """
    if field_name in _OPAQUE_FIELDS:
        return IdClass.OPAQUE
    if field_name in _SESSION_FIELDS:
        return IdClass.SESSION
    if field_name in _INVOCATION_FIELDS:
        return IdClass.INVOCATION
    if field_name in _COMPRESSED_CONTEXT_FIELDS:
        return IdClass.COMPRESSED_CONTEXT
    if field_name in _ARTIFACT_FIELDS:
        return IdClass.ARTIFACT
    if field_name in _PROFILE_FIELDS:
        return IdClass.PROFILE
    return IdClass.ANALYTICS


def is_id_field(field_name: str) -> bool:
    """Return whether a payload/envelope key carries an identifier."""
    return field_name == "item_id" or field_name.endswith("_id")


def is_valid_analytics_id(value: object) -> bool:
    """Return whether *value* is a 32-hex analytics identifier."""
    return isinstance(value, str) and _ANALYTICS_ID_RE.fullmatch(value) is not None


def is_valid_session_id(value: object) -> bool:
    """Return whether *value* is a canonical UUID string (or a legacy 12-hex id)."""
    if not isinstance(value, str):
        return False
    return _SESSION_ID_RE.fullmatch(value) is not None or _INVOCATION_ID_RE.fullmatch(value) is not None


def is_valid_invocation_id(value: object) -> bool:
    """Return whether *value* is a 12-hex sub-agent invocation identifier."""
    return isinstance(value, str) and _INVOCATION_ID_RE.fullmatch(value) is not None


def is_valid_compressed_context_id(value: object) -> bool:
    """Return whether *value* is a session compressed-block identifier (``ctx_`` + 8 hex)."""
    return isinstance(value, str) and _COMPRESSED_CONTEXT_ID_RE.fullmatch(value) is not None


def is_valid_artifact_id(value: object) -> bool:
    """Return whether *value* is a bare artifact file name (never a path)."""
    if not isinstance(value, str) or not value or len(value) > ARTIFACT_ID_MAX_LENGTH:
        return False
    if value in {".", ".."} or any(ch in _ARTIFACT_ID_SEPARATORS for ch in value):
        return False
    return all(ch.isprintable() for ch in value)


def is_valid_opaque_id(value: object) -> bool:
    """Return whether *value* is a bounded printable opaque identifier."""
    if not isinstance(value, str) or not value or len(value) > OPAQUE_ID_MAX_LENGTH:
        return False
    return all(ch.isprintable() for ch in value)


def is_valid_id(field_name: str, value: object) -> bool:
    """Validate *value* against the format class implied by *field_name*."""
    id_class = classify_id_field(field_name)
    if id_class is IdClass.ANALYTICS:
        return is_valid_analytics_id(value)
    if id_class is IdClass.SESSION:
        return is_valid_session_id(value)
    if id_class is IdClass.INVOCATION:
        return is_valid_invocation_id(value)
    if id_class is IdClass.COMPRESSED_CONTEXT:
        return is_valid_compressed_context_id(value)
    if id_class is IdClass.ARTIFACT:
        return is_valid_artifact_id(value)
    return is_valid_opaque_id(value)
