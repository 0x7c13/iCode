# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Canonical Chrys tool-kind vocabulary and out-of-band kind channel."""

from __future__ import annotations

import logging
from typing import Any, Final

logger = logging.getLogger(__name__)

KIND_SHELL: Final[str] = "shell"
KIND_FILESYSTEM_READ: Final[str] = "filesystem.read"
KIND_FILESYSTEM_WRITE: Final[str] = "filesystem.write"
KIND_SEARCH: Final[str] = "search"
KIND_WEB_SEARCH: Final[str] = "web_search"
KIND_WEB_FETCH: Final[str] = "web_fetch"
KIND_ASK_USER: Final[str] = "ask_user"
KIND_SLEEP: Final[str] = "sleep"
KIND_SUB_AGENT: Final[str] = "sub_agent"
KIND_MCP: Final[str] = "mcp"
KIND_DOC_CONVERTER: Final[str] = "doc_converter"
KIND_SKILL: Final[str] = "skill"
KIND_CONTEXT: Final[str] = "context"
KIND_TODO: Final[str] = "todo"

# Canonical set of Chrys tool kinds - the form shared by YAML, hook matchers,
# approval-override keys and runtime events.
TOOL_KINDS: Final[frozenset[str]] = frozenset(
    {
        KIND_SHELL,
        KIND_FILESYSTEM_READ,
        KIND_FILESYSTEM_WRITE,
        KIND_SEARCH,
        KIND_WEB_SEARCH,
        KIND_WEB_FETCH,
        KIND_ASK_USER,
        KIND_SLEEP,
        KIND_SUB_AGENT,
        KIND_MCP,
        KIND_DOC_CONVERTER,
        KIND_SKILL,
        KIND_CONTEXT,
        KIND_TODO,
    }
)

TOOL_CALL_KIND_METADATA_KEY: Final[str] = "_chrys_tool_kind"
"""Structured tool kind persisted on assistant ``function_call`` contents.

Lives at the foundation layer so the kernel tool loop can stamp it on
calls that never reach the service-layer middleware (pre-pipeline argument
validation failures); ``chrys.service.session.message_metadata`` re-exports
it for existing importers.
"""

_KIND_ATTR: Final[str] = "chrys_kind"

# Pre-P1 runtime kinds were namespaced (``chrys.shell``); the deleted
# YAML<->runtime conversion layer also accepted that form directly in config.
_LEGACY_KIND_PREFIX: Final[str] = "chrys."


def strip_legacy_kind_prefix(value: str, *, allow_tool_suffix: bool = False, source: str = "") -> str:
    """Strip the legacy ``chrys.`` prefix from a config-supplied kind, warning once per call.

    Bounded compat for pre-P1 YAML: only values that were *meaningful* under
    the old prefixed scheme are rewritten:

    - ``chrys.<known kind>`` (``chrys.shell`` -> ``shell``);
    - with ``allow_tool_suffix=True``, ``chrys.<known kind>.<tool>`` too
      (approval-override keys; ``chrys.filesystem.write.write_file`` ->
      ``filesystem.write.write_file``).

    Anything else (bare values, ``chrys.``-prefixed strings that never matched
    a kind - possibly literal tool names) passes through unchanged.
    """
    if not value.startswith(_LEGACY_KIND_PREFIX):
        return value
    bare = value[len(_LEGACY_KIND_PREFIX) :]
    recognized = bare in TOOL_KINDS or (
        allow_tool_suffix and any(bare.startswith(f"{kind}.") and len(bare) > len(kind) + 1 for kind in TOOL_KINDS)
    )
    if not recognized:
        return value
    logger.warning(
        "Legacy tool-kind value %r in %s - the 'chrys.' runtime prefix was removed; use %r.",
        value,
        source or "config",
        bare,
    )
    return bare


def set_tool_kind(tool: Any, kind: str) -> None:
    """Classify *tool* with a Chrys kind, out of band."""
    setattr(tool, _KIND_ATTR, kind)


def get_tool_kind(tool: Any) -> str | None:
    """Return the Chrys kind of *tool*, or ``None`` when unclassified."""
    return getattr(tool, _KIND_ATTR, None)
