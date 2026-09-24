# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow SDK: the eight public names a workflow file imports from ``chrys.workflows``.

This package is the single source of the SDK. It is pure standard library,
imports nothing from the rest of chrys, and stays on Python 3.9 syntax and
runtime APIs, because the worker host injects a copy of it into whatever
interpreter runs the user's workflow file (``chrys/workflows/`` inside the
injected artifact). Keep it that way: the py39 contract harness compiles and
imports it under a real 3.9 interpreter.
"""

from ._builder import (
    BuilderScope,
    EdgeDefinition,
    LoopDefinition,
    NodeDefinition,
    NodeHandle,
    Retry,
    Workflow,
    WorkflowBuilder,
    WorkflowDefinition,
    WorkflowValidationError,
)
from ._values import JsonValue, NodeContext, SourceValue, WorkflowValue

__all__ = [
    "BuilderScope",
    "NodeContext",
    "NodeHandle",
    "Retry",
    "SourceValue",
    "Workflow",
    "WorkflowBuilder",
    "WorkflowValue",
]

# Names below are part of the SDK module but not of the authoring API: the
# worker host and the main-process manifest reader use them.
_INTERNAL = (
    EdgeDefinition,
    JsonValue,
    LoopDefinition,
    NodeDefinition,
    WorkflowDefinition,
    WorkflowValidationError,
)
