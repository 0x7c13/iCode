# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Messages emitted by terminal diagram presentation widgets."""

from __future__ import annotations

from textual.message import Message

from .model import CompiledDiagram


class DiagramOpenRequested(Message):
    """Request presentation of a compiled diagram in an expanded viewer."""

    def __init__(self, diagram: CompiledDiagram) -> None:
        super().__init__()
        self.diagram = diagram
        self.source = diagram.source
