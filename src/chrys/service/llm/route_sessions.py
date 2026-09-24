# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Stable LLM route-session id derivation for auxiliary request streams."""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

if TYPE_CHECKING:
    from chrys.service.profiles.models.schema import ModelProfile

# Chrys-owned UUIDv5 namespace for deriving opaque LLM routing session ids.
# Changing this changes every derived auxiliary/sub-agent route session id.
_LLM_ROUTE_SESSION_NAMESPACE = UUID("6ddd36f6-03d9-4e49-8558-fb0456857f55")
_SEPARATOR = "\x1f"

llm_route_session_id: ContextVar[str] = ContextVar("llm_route_session_id", default="")
llm_parent_session_id: ContextVar[str] = ContextVar("llm_parent_session_id", default="")


def derive_llm_route_session_id(
    parent_session_id: str | None,
    *,
    route_kind: str,
    route_parts: Sequence[str] = (),
    model_profile: ModelProfile,
) -> str:
    """Return a stable child route-session id for a distinct LLM request stream."""
    if not parent_session_id:
        return uuid4().hex

    from chrys.service.llm.clients import effective_model_base_url

    seed = _SEPARATOR.join(
        (
            parent_session_id,
            route_kind,
            *route_parts,
            model_profile.id,
            model_profile.provider,
            model_profile.api_style,
            model_profile.model_id,
            effective_model_base_url(model_profile),
        )
    )
    return uuid5(_LLM_ROUTE_SESSION_NAMESPACE, seed).hex
