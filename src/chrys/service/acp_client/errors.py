# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Total, phase-aware ACP client error classification."""

from __future__ import annotations

import asyncio
import errno
from enum import StrEnum
from typing import Any

from acp.exceptions import RequestError

from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.platform.process import MissingWorkingDirectoryError

_DETERMINISTIC_REQUEST_CODES = frozenset({-32700, -32600, -32601, -32602, -32002})
_AUTH_REQUIRED_CODE = -32000
_INTERNAL_ERROR_CODE = -32603
# ETXTBSY is deliberately absent: it clears when the concurrent writer of the
# executable finishes, so the same launch can succeed on retry.
_DETERMINISTIC_ERRNOS = frozenset(
    {
        errno.E2BIG,
        errno.EACCES,
        errno.EINVAL,
        errno.EISDIR,
        errno.ELOOP,
        errno.ENAMETOOLONG,
        errno.ENOENT,
        errno.ENOEXEC,
        errno.ENOTDIR,
        errno.EPERM,
    }
)
_DETERMINISTIC_WINERRORS = frozenset({2, 3, 5, 10, 87, 123, 193, 203, 206, 216, 267, 740})


class AcpOperation(StrEnum):
    """Protocol operation being classified."""

    SPAWN = "spawn"
    INITIALIZE = "initialize"
    NEW_SESSION = "session/new"
    SET_MODE = "session/set_mode"
    SET_MODEL = "session/set_model"
    SET_CONFIG_OPTION = "session/set_config_option"
    PROMPT = "session/prompt"
    CLOSE_SESSION = "session/close"


class AcpClientError(RuntimeError):
    """Base for sanitized client failures safe to expose to model-visible code."""

    def __init__(self, detail: str, *, cause: BaseException | None = None) -> None:
        super().__init__(_sanitize_detail(detail))
        self.detail = _sanitize_detail(detail)
        self.cause = cause


class AcpConfigError(AcpClientError):
    """Deterministic configuration or protocol incompatibility."""


class AcpSpawnError(AcpClientError):
    """Deterministic process launch failure."""


class AcpConnectError(AcpClientError):
    """Transient failure in the provably stateless connect window."""


class AcpTransportError(AcpClientError):
    """Ambiguous post-stateful transport or protocol failure."""

    def __init__(
        self,
        detail: str,
        *,
        cause: BaseException | None = None,
        usage: Any | None = None,
    ) -> None:
        super().__init__(detail, cause=cause)
        self.usage = usage


class AcpIdleTimeoutError(AcpTransportError):
    """Prompt inactivity timeout while no human decision is outstanding."""


class AcpAuthRequiredError(AcpClientError):
    """The remote requires an authentication method Chrys cannot perform."""

    def __init__(
        self,
        detail: str = "The ACP agent requires authentication.",
        *,
        method_names: tuple[str, ...] = (),
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(detail, cause=cause)
        self.method_names = method_names


class AcpRefusalError(AcpClientError):
    """Terminal remote refusal, distinct from a transport pause."""


def _sanitize_detail(detail: str) -> str:
    line = str(detail).replace("\r", " ").replace("\n", " ").strip()
    if not line:
        return "ACP operation failed."
    return line[:500]


def _request_auth_methods(error: RequestError) -> tuple[str, ...]:
    data = error.data
    if not isinstance(data, dict):
        return ()
    raw_methods = data.get("authMethods", data.get("auth_methods", data.get("methods", ())))
    if not isinstance(raw_methods, list | tuple):
        return ()
    names: list[str] = []
    for item in raw_methods:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = item.get("name", item.get("id"))
            if not isinstance(value, str):
                continue
        else:
            continue
        sanitized = value.replace("\r", " ").replace("\n", " ").strip()[:100]
        if sanitized:
            names.append(sanitized)
    return tuple(names)


def is_remote_error(error: AcpClientError) -> bool:
    """Return whether the remote agent itself answered with the JSON-RPC error behind *error*."""
    return isinstance(error.cause, RequestError)


def is_deterministic_request_error(error: RequestError) -> bool:
    """Return whether retrying the same request cannot change its rejection."""
    return type(error.code) is int and error.code in _DETERMINISTIC_REQUEST_CODES


def is_best_effort_option_rejection(error: RequestError) -> bool:
    """Return whether a configured option may be skipped under best-effort policy."""
    return type(error.code) is int and error.code in {-32601, -32602}


def classify_spawn_error(error: BaseException) -> AcpSpawnError | AcpConnectError:
    """Classify every subprocess-boundary failure as deterministic or transient."""
    if isinstance(error, TypeError | ValueError):
        return AcpSpawnError("The ACP process launch configuration is invalid.", cause=error)
    if isinstance(error, MissingWorkingDirectoryError):
        return AcpSpawnError(
            f"The ACP agent's working directory no longer exists: {surrogate_safe_text(error.path)}", cause=error
        )
    if isinstance(error, OSError):
        winerror = getattr(error, "winerror", None)
        if error.errno in _DETERMINISTIC_ERRNOS or winerror in _DETERMINISTIC_WINERRORS:
            return AcpSpawnError("The ACP process could not be launched with this configuration.", cause=error)
        return AcpConnectError("The ACP process could not be started due to a transient system error.", cause=error)
    return AcpConnectError("The ACP process could not be started.", cause=error)


def classify_protocol_frame_error(
    error: BaseException,
    *,
    stateful: bool,
    complete: bool,
) -> AcpConfigError | AcpConnectError | AcpTransportError:
    """Classify a syntactically invalid complete line versus EOF-truncated frame."""
    if stateful:
        return AcpTransportError("The ACP agent sent an invalid protocol frame.", cause=error)
    if complete:
        return AcpConfigError("The ACP executable did not produce valid ACP JSON-RPC.", cause=error)
    return AcpConnectError("The ACP process ended while writing a protocol frame.", cause=error)


def classify_acp_error(
    error: BaseException,
    *,
    operation: AcpOperation,
    stateful: bool,
) -> AcpClientError:
    """Map every known SDK/validation/transport failure to one settled policy class."""
    if isinstance(error, AcpClientError):
        return error
    if operation is AcpOperation.SPAWN:
        return classify_spawn_error(error)
    if isinstance(error, RequestError):
        if type(error.code) is int and error.code == _AUTH_REQUIRED_CODE:
            return AcpAuthRequiredError(method_names=_request_auth_methods(error), cause=error)
        if is_deterministic_request_error(error):
            return AcpConfigError(f"The ACP agent rejected {operation.value}.", cause=error)
        if type(error.code) is int and error.code == _INTERNAL_ERROR_CODE:
            if stateful:
                return AcpTransportError(f"The ACP agent failed during {operation.value}.", cause=error)
            return AcpConnectError(f"The ACP agent failed during {operation.value}.", cause=error)
        if stateful:
            return AcpTransportError(f"The ACP agent returned an unknown error during {operation.value}.", cause=error)
        return AcpConnectError(f"The ACP agent returned an unknown error during {operation.value}.", cause=error)
    if _is_validation_error(error):
        if operation is AcpOperation.INITIALIZE:
            return AcpConfigError("The ACP agent returned an incompatible initialize response.", cause=error)
        if stateful:
            return AcpTransportError(f"The ACP agent returned a malformed {operation.value} response.", cause=error)
        return AcpConnectError(f"The ACP agent returned a malformed {operation.value} response.", cause=error)
    if isinstance(error, TimeoutError | ConnectionError | EOFError | OSError | asyncio.CancelledError):
        if stateful:
            return AcpTransportError(f"The ACP transport failed during {operation.value}.", cause=error)
        return AcpConnectError(f"The ACP transport failed during {operation.value}.", cause=error)
    if stateful:
        return AcpTransportError(f"The ACP client failed during {operation.value}.", cause=error)
    return AcpConnectError(f"The ACP client failed during {operation.value}.", cause=error)


def _is_validation_error(error: BaseException) -> bool:
    error_type = type(error)
    return error_type.__name__ == "ValidationError" and error_type.__module__.startswith("pydantic")
