# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline terminal mathematics; failures always retain the original source."""

from __future__ import annotations

from dataclasses import dataclass

from .layout import inline, layout
from .parser import MathError, parse_math


@dataclass(frozen=True, slots=True)
class CompiledMath:
    """One source's reusable display and linear forms, or a source-only failure."""

    source: str
    rows: tuple[str, ...] = ()
    width: int = 0
    linear: str = ""


def compile_math(source: str) -> CompiledMath:
    """Compile a bounded TeX subset without algebraic rewriting or execution."""
    try:
        node = parse_math(source)
        result = layout(node)
        linear = inline(node)
        if not linear.strip() or not result.width:
            return CompiledMath(source)
    except MathError:
        return CompiledMath(source)
    return CompiledMath(source, result.rows, result.width, linear)


__all__ = ["CompiledMath", "compile_math"]
