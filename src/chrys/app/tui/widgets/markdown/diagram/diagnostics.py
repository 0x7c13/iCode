# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Localized display adapters for locale-neutral diagram diagnostics."""

from __future__ import annotations

from collections.abc import Callable

from chrys.foundation.i18n import MessageDef, MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

from .model import Diagnostic, DiagnosticCode, DiagnosticSeverity

_UNABLE_TO_RENDER = msg(
    "tui.diagram.diagnostic.unable_to_render",
    fallback="Mermaid diagram could not be rendered",
)
_ERROR_AT_LINE = msg(
    "tui.diagram.diagnostic.error_at_line",
    fallback="Line {line}: {detail}",
)
_WARNING_AT_LINE = msg(
    "tui.diagram.diagnostic.warning_at_line",
    fallback="Warning on line {line}: {detail}",
)
_INVALID_SYNTAX = msg(
    "tui.diagram.diagnostic.invalid_syntax",
    fallback="Unsupported or malformed Mermaid syntax",
)
_LIMIT_EXCEEDED = msg(
    "tui.diagram.diagnostic.limit_exceeded",
    fallback="The diagram exceeds renderer limits",
)
_CONFLICTING_DECLARATION = msg(
    "tui.diagram.diagnostic.conflicting_declaration",
    fallback="A node was declared more than once; the latest label is used",
)
_EMPTY_SOURCE = msg(
    "tui.diagram.diagnostic.empty_source",
    fallback="The Mermaid diagram is empty",
)
_UNSUPPORTED_TYPE = msg(
    "tui.diagram.diagnostic.unsupported_type",
    fallback="Expected a supported Mermaid diagram type",
)
_NO_NODES = msg(
    "tui.diagram.diagnostic.no_nodes",
    fallback="The diagram contains no nodes",
)
_MORE_DIAGNOSTICS = msg(
    "tui.diagram.diagnostic.more",
    fallback="… and {count} more diagnostic",
    plural_fallback="… and {count} more diagnostics",
)

_LIMIT_CODES = {
    DiagnosticCode.NODE_LIMIT,
    DiagnosticCode.EDGE_LIMIT,
    DiagnosticCode.SOURCE_LIMIT,
    DiagnosticCode.CANVAS_LIMIT,
}


def _detail_definition(code: DiagnosticCode) -> MessageDef:
    if code in _LIMIT_CODES:
        return _LIMIT_EXCEEDED
    if code is DiagnosticCode.NODE_REDECLARED:
        return _CONFLICTING_DECLARATION
    if code is DiagnosticCode.EMPTY_SOURCE:
        return _EMPTY_SOURCE
    if code is DiagnosticCode.UNSUPPORTED_DIAGRAM_TYPE:
        return _UNSUPPORTED_TYPE
    if code is DiagnosticCode.NO_NODES:
        return _NO_NODES
    return _INVALID_SYNTAX


def _render(reference: MessageRef, renderer: Callable[[MessageRef], str] | None) -> str:
    return format_message(reference) if renderer is None else renderer(reference)


def render_diagnostic_heading(renderer: Callable[[MessageRef], str] | None = None) -> str:
    """Render the diagnostic canvas heading for the active locale."""
    return _render(_UNABLE_TO_RENDER.bind(), renderer)


def render_diagnostic(diagnostic: Diagnostic, renderer: Callable[[MessageRef], str] | None = None) -> str:
    """Render one locale-neutral diagnostic for the active locale."""
    detail = _render(_detail_definition(diagnostic.code).bind(), renderer)
    if diagnostic.line <= 0:
        return detail
    definition = _WARNING_AT_LINE if diagnostic.severity is DiagnosticSeverity.WARNING else _ERROR_AT_LINE
    return _render(definition.bind(line=diagnostic.line, detail=detail), renderer)


def render_more_diagnostics(count: int, renderer: Callable[[MessageRef], str] | None = None) -> str:
    """Render a bounded-canvas overflow line for hidden diagnostics."""
    return _render(_MORE_DIAGNOSTICS.bind(count=count), renderer)


__all__ = ["render_diagnostic", "render_diagnostic_heading", "render_more_diagnostics"]
