# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable service-tier specifications and outcomes for external ACP agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from acp.schema import (
    AgentCapabilities,
    AuthMethodAgent,
    EnvVarAuthMethod,
    Implementation,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionModelState,
    SessionModeState,
    TerminalAuthMethod,
)

from chrys import __version__


@dataclass(frozen=True, slots=True, kw_only=True)
class AcpAgentSpec:
    """Fully resolved launch and session specification for one ACP attempt."""

    command: str
    cwd: str
    stderr_log_path: Path
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    additional_directories: tuple[str, ...] = ()
    session_mode: str = ""
    model_id: str = ""
    config_options: Mapping[str, str | bool] = field(default_factory=dict)
    best_effort_options: bool = False
    handshake_timeout_seconds: float = 30.0
    idle_timeout_seconds: float = 600.0
    client_info: Implementation = field(
        default_factory=lambda: Implementation(name="chrys", version=__version__),
    )
    depth: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PermissionDecision:
    """A callback decision for an agent-originated permission request."""

    action: Literal["allow", "deny", "cancelled"]
    option_id: str | None = None

    @classmethod
    def allow(cls, option_id: str) -> PermissionDecision:
        return cls(action="allow", option_id=option_id)

    @classmethod
    def deny(cls, option_id: str | None = None) -> PermissionDecision:
        return cls(action="deny", option_id=option_id)

    @classmethod
    def cancelled(cls) -> PermissionDecision:
        return cls(action="cancelled")


@dataclass(frozen=True, slots=True, kw_only=True)
class AcpPromptUsage:
    """Strictly validated per-attempt ACP token usage."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None
    thought_tokens: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AcpPromptOutcome:
    """Result of one prompt request after all earlier updates have drained."""

    stop_reason: Literal["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"]
    usage: AcpPromptUsage | None
    user_message_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AcpHandshakeInfo:
    """Agent and session metadata collected by ``connect`` and ``open_session``."""

    session_id: str
    agent_info: Implementation | None
    auth_methods: tuple[EnvVarAuthMethod | TerminalAuthMethod | AuthMethodAgent, ...]
    capabilities: AgentCapabilities
    modes: SessionModeState | None
    models: SessionModelState | None
    config_options: tuple[SessionConfigOptionSelect | SessionConfigOptionBoolean, ...]
