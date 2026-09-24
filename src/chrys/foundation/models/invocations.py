# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable invocation identities, independent of execution and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class InvocationOrigin:
    """A logical caller operation, including its explicitly bound ancestry."""

    kind: Literal["turn", "sub_agent", "workflow_node"]
    session_id: str
    invocation_id: str
    parent: InvocationOrigin | None

    def __post_init__(self) -> None:
        if self.kind not in ("turn", "sub_agent", "workflow_node") or not self.invocation_id:
            raise ValueError("A live invocation requires a kind and invocation identity")

    @property
    def root(self) -> InvocationOrigin:
        """The outermost bound ancestor: the chat turn or workflow node an operation runs under, or itself."""
        origin = self
        while origin.parent is not None:
            origin = origin.parent
        return origin


@dataclass(frozen=True, slots=True)
class ParentToolOccurrence:
    """Both parent tool identifiers, scoped to the parent invocation."""

    parent_invocation_id: str
    parent_event_call_id: str
    parent_provider_call_id: str


@dataclass(frozen=True, slots=True)
class PassHandle:
    """An immutable cancellation target; never a provider call identifier."""

    invocation_id: str
    pass_id: str
