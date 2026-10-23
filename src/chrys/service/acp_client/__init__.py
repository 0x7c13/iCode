# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""External ACP agent client core."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .errors import (
    AcpAuthRequiredError,
    AcpClientError,
    AcpConfigError,
    AcpConnectError,
    AcpIdleTimeoutError,
    AcpOperation,
    AcpRefusalError,
    AcpSpawnError,
    AcpTransportError,
    classify_acp_error,
    classify_protocol_frame_error,
    classify_spawn_error,
)
from .protocol import (
    encode_protocol_json,
    parse_protocol_json,
    validate_json_rpc_envelope,
    validate_json_scalar_tree,
)
from .spec import (
    AcpAgentSpec,
    AcpHandshakeInfo,
    AcpPromptOutcome,
    AcpPromptUsage,
    PermissionDecision,
)

if TYPE_CHECKING:
    from .client import (
        AcpAgentClient,
        AcpClientCallbacks,
        AcpClientWaitController,
        AcpUpdateSink,
    )

# The client loads on first use; specs and errors above stay light.
_EXPORTS = {
    "AcpAgentClient": "client",
    "AcpClientCallbacks": "client",
    "AcpClientWaitController": "client",
    "AcpUpdateSink": "client",
}

__all__ = [
    "AcpAgentClient",
    "AcpAgentSpec",
    "AcpAuthRequiredError",
    "AcpClientCallbacks",
    "AcpClientError",
    "AcpClientWaitController",
    "AcpConfigError",
    "AcpConnectError",
    "AcpHandshakeInfo",
    "AcpIdleTimeoutError",
    "AcpOperation",
    "AcpPromptOutcome",
    "AcpPromptUsage",
    "AcpRefusalError",
    "AcpSpawnError",
    "AcpTransportError",
    "AcpUpdateSink",
    "PermissionDecision",
    "classify_acp_error",
    "classify_protocol_frame_error",
    "classify_spawn_error",
    "encode_protocol_json",
    "parse_protocol_json",
    "validate_json_rpc_envelope",
    "validate_json_scalar_tree",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the ACP client."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the default module attributes plus facade exports."""
    return sorted({*globals(), *__all__})
