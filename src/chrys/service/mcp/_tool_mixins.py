# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared MCP transport tool mixins."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol


class _NoPrePagePingMixin:
    """Disable the owned engine's pre-pagination ``send_ping`` health check.

    The owned ``MCPTool._ensure_connected`` sends ``ping`` before every
    ``list_tools`` / ``list_prompts`` page and treats *any* failure (including
    ``method_not_found`` from servers that don't implement the optional ``ping``
    utility) as a dead session — triggering ``connect(reset=True)``, which
    closes the transport (DELETE on Streamable HTTP) and re-handshakes (POST
    initialize + background GET). With a non-implementing server this recurses
    into a tight POST → GET → DELETE loop until ``request_timeout`` fires.

    The check is also redundant: ``_ensure_connected`` only runs immediately
    after ``session.initialize()`` succeeds, when the session is freshly
    handshaken. Real transport loss surfaces from the actual ``list_*`` call as
    ``ClosedResourceError`` and propagates up to ``__aenter__`` cleanly.
    Runtime tool calls (``call_tool`` / ``get_prompt``) have their own
    reconnect-on-``ClosedResourceError`` paths that don't go through
    ``_ensure_connected``.
    """

    async def _ensure_connected(self) -> None:
        return


def _content_has_meaningful_payload(content: list[Any]) -> bool:
    """Return True if ``content`` carries any non-empty/non-text payload.

    Used to decide whether to fall back to ``CallToolResult.structuredContent``.
    A list that is empty, or that contains only whitespace-only ``TextContent``
    items, is considered to have no meaningful payload — those are the cases
    where a server is relying on the structured field to deliver its result.
    Any non-text item (image, audio, resource, link, etc.) counts as
    meaningful regardless of whether ``structuredContent`` is also set, since
    ``MCPTool._parse_tool_result_from_mcp`` already renders those usefully.
    """
    from mcp import types

    for item in content:
        if isinstance(item, types.TextContent):
            if (item.text or "").strip():
                return True
        else:
            return True
    return False


if TYPE_CHECKING:

    class _StructuredContentFallbackBase(Protocol):
        def _parse_tool_result_from_mcp(self, mcp_type: Any) -> Any: ...

else:
    _StructuredContentFallbackBase = object


class _StructuredContentFallbackMixin(_StructuredContentFallbackBase):
    """Surface ``CallToolResult.structuredContent`` when ``content`` is empty.

    Per the MCP spec, a server may return its payload in
    ``structuredContent`` (a JSON object) and leave ``content`` empty or
    populated with only an empty text fallback for clients that don't yet
    consume the structured field. The owned ``MCPTool._parse_tool_result_from_mcp``
    walks ``content`` exclusively, so it would return an empty result to the
    model without this fallback.

    This mixin overrides the parser to detect the "no meaningful content"
    case and return a single ``Content.from_text`` carrying a JSON dump
    of ``structuredContent`` instead.  When ``content`` already has a
    meaningful payload, the owned MCPTool parser is used unchanged — a
    server that emits both a populated text fallback *and*
    ``structuredContent`` keeps its existing rendering (the structured
    field is a duplicate in that case).
    """

    def _parse_tool_result_from_mcp(self, mcp_type: Any) -> Any:
        from chrys.kernel import Content

        structured = getattr(mcp_type, "structuredContent", None)
        content = list(getattr(mcp_type, "content", None) or [])
        if structured is not None and not _content_has_meaningful_payload(content):
            try:
                payload = json.dumps(structured, default=str, ensure_ascii=False)
            except TypeError, ValueError:
                payload = str(structured)
            return [Content.from_text(payload)]
        return super()._parse_tool_result_from_mcp(mcp_type)  # type: ignore[misc]
