# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""External ACP agent client core."""

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
