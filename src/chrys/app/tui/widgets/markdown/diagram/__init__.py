# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline Mermaid-subset compiler for terminal Markdown rendering."""

from __future__ import annotations

from collections.abc import Callable

from chrys.foundation.i18n import MessageRef

from .layout import compile_ir
from .model import CompiledDiagram, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagramKind
from .parser import parse_mermaid

__all__ = [
    "CompiledDiagram",
    "Diagnostic",
    "DiagnosticCode",
    "DiagnosticSeverity",
    "DiagramKind",
    "compile_mermaid",
]


def compile_mermaid(
    source: str,
    *,
    render_message: Callable[[MessageRef], str] | None = None,
) -> CompiledDiagram:
    """Compile bounded Mermaid source into immutable plain terminal rows.

    Malformed, unsupported, and oversized user input produces a diagnostic
    canvas. Expected input failures do not raise exceptions.
    """
    return compile_ir(source, parse_mermaid(source), render_message=render_message)
