# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""External ACP agent client core."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import (
        AcpAgentClient,
        AcpClientCallbacks,
        AcpClientWaitController,
        AcpUpdateSink,
    )


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

_EXPORTS = {
    "AcpAgentClient": "chrys.service.acp_client.client",
    "AcpClientCallbacks": "chrys.service.acp_client.client",
    "AcpClientWaitController": "chrys.service.acp_client.client",
    "AcpUpdateSink": "chrys.service.acp_client.client",
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
    """Resolve public feature exports only when requested."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = vars(import_module(module_name))[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include public exports without importing their implementations."""
    return sorted({*globals(), *__all__})
