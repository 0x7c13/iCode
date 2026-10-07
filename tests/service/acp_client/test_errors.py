# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exhaustive ACP error classification tests."""

from __future__ import annotations

import errno

import pytest
from acp.exceptions import RequestError
from acp.schema import InitializeResponse, PromptResponse

from chrys.foundation.platform.process import MissingWorkingDirectoryError
from chrys.service.acp_client import (
    AcpAuthRequiredError,
    AcpConfigError,
    AcpConnectError,
    AcpOperation,
    AcpSpawnError,
    AcpTransportError,
    classify_acp_error,
    classify_protocol_frame_error,
    classify_spawn_error,
)

_ALL_PROTOCOL_OPERATIONS = (
    AcpOperation.INITIALIZE,
    AcpOperation.NEW_SESSION,
    AcpOperation.SET_MODE,
    AcpOperation.SET_MODEL,
    AcpOperation.SET_CONFIG_OPTION,
    AcpOperation.PROMPT,
    AcpOperation.CLOSE_SESSION,
)


@pytest.mark.parametrize(
    "error",
    [
        RequestError.parse_error(),
        RequestError.invalid_request(),
        RequestError.method_not_found("method"),
        RequestError.invalid_params(),
        RequestError.resource_not_found("resource"),
    ],
)
@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("operation", _ALL_PROTOCOL_OPERATIONS)
def test_deterministic_request_codes_are_always_config_errors(
    error: RequestError,
    stateful: bool,
    operation: AcpOperation,
) -> None:
    classified = classify_acp_error(error, operation=operation, stateful=stateful)

    assert isinstance(classified, AcpConfigError)


def test_auth_required_is_total_and_carries_methods() -> None:
    error = RequestError.auth_required({"authMethods": [{"name": "Login"}, "Environment"]})

    classified = classify_acp_error(error, operation=AcpOperation.NEW_SESSION, stateful=True)

    assert isinstance(classified, AcpAuthRequiredError)
    assert classified.method_names == ("Login", "Environment")


@pytest.mark.parametrize("code", [-32603, -31999, 42, True, 1.5, "vendor"])
def test_transient_or_unknown_codes_split_at_stateful_boundary(code: object) -> None:
    error = RequestError(code, "remote error")

    before = classify_acp_error(error, operation=AcpOperation.INITIALIZE, stateful=False)
    after = classify_acp_error(error, operation=AcpOperation.NEW_SESSION, stateful=True)

    assert isinstance(before, AcpConnectError)
    assert isinstance(after, AcpTransportError)


def test_malformed_initialize_is_config_but_malformed_prompt_is_transport() -> None:
    with pytest.raises(Exception) as initialize_error:
        InitializeResponse.model_validate({"protocolVersion": "bad"})
    with pytest.raises(Exception) as prompt_error:
        PromptResponse.model_validate({"stopReason": "unknown"})

    assert isinstance(
        classify_acp_error(
            initialize_error.value,
            operation=AcpOperation.INITIALIZE,
            stateful=False,
        ),
        AcpConfigError,
    )
    assert isinstance(
        classify_acp_error(
            prompt_error.value,
            operation=AcpOperation.PROMPT,
            stateful=True,
        ),
        AcpTransportError,
    )


@pytest.mark.parametrize(
    "operation",
    [
        AcpOperation.NEW_SESSION,
        AcpOperation.SET_MODE,
        AcpOperation.SET_MODEL,
        AcpOperation.SET_CONFIG_OPTION,
        AcpOperation.PROMPT,
        AcpOperation.CLOSE_SESSION,
    ],
)
def test_malformed_responses_split_at_the_stateful_boundary_for_every_operation(
    operation: AcpOperation,
) -> None:
    with pytest.raises(Exception) as validation_error:
        PromptResponse.model_validate({"stopReason": "unknown"})

    before = classify_acp_error(validation_error.value, operation=operation, stateful=False)
    after = classify_acp_error(validation_error.value, operation=operation, stateful=True)

    assert isinstance(before, AcpConnectError)
    assert isinstance(after, AcpTransportError)


@pytest.mark.parametrize(
    "error_number",
    [
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
    ],
)
def test_spawn_deterministic_errno_table(error_number: int) -> None:
    assert isinstance(classify_spawn_error(OSError(error_number, "launch failed")), AcpSpawnError)


@pytest.mark.skipif(not hasattr(errno, "ETXTBSY"), reason="ETXTBSY is POSIX-only")
def test_spawn_etxtbsy_is_transient_because_the_writer_can_finish() -> None:
    assert isinstance(classify_spawn_error(OSError(errno.ETXTBSY, "text file busy")), AcpConnectError)


@pytest.mark.parametrize("winerror", [2, 3, 5, 10, 87, 123, 193, 203, 206, 216, 267, 740])
def test_spawn_deterministic_winerror_table(winerror: int) -> None:
    error = OSError("launch failed")
    error.winerror = winerror

    assert isinstance(classify_spawn_error(error), AcpSpawnError)


def test_spawn_unknown_oserror_is_retryable_connect_failure() -> None:
    assert isinstance(classify_spawn_error(OSError(errno.EAGAIN, "busy")), AcpConnectError)


@pytest.mark.parametrize("error", [TypeError("bad env"), ValueError("embedded null")])
def test_spawn_python_boundary_validation_errors_are_deterministic(error: BaseException) -> None:
    assert isinstance(classify_spawn_error(error), AcpSpawnError)


def test_spawn_into_a_deleted_working_directory_names_the_directory() -> None:
    error = MissingWorkingDirectoryError("/work/gone-\udcff")

    classified = classify_spawn_error(error)

    assert isinstance(classified, AcpSpawnError)
    assert classified.detail == r"The ACP agent's working directory no longer exists: /work/gone-\udcff"
    assert classified.cause is error


def test_complete_invalid_and_truncated_frames_split_before_stateful_phase() -> None:
    assert isinstance(
        classify_protocol_frame_error(ValueError("bad"), stateful=False, complete=True),
        AcpConfigError,
    )
    assert isinstance(
        classify_protocol_frame_error(EOFError(), stateful=False, complete=False),
        AcpConnectError,
    )
    assert isinstance(
        classify_protocol_frame_error(ValueError("bad"), stateful=True, complete=True),
        AcpTransportError,
    )
    assert isinstance(
        classify_protocol_frame_error(EOFError(), stateful=True, complete=False),
        AcpTransportError,
    )


@pytest.mark.parametrize(
    "operation",
    [
        AcpOperation.INITIALIZE,
        AcpOperation.NEW_SESSION,
        AcpOperation.SET_MODE,
        AcpOperation.SET_MODEL,
        AcpOperation.SET_CONFIG_OPTION,
        AcpOperation.PROMPT,
        AcpOperation.CLOSE_SESSION,
    ],
)
def test_unknown_exception_classification_is_total_for_every_operation(operation: AcpOperation) -> None:
    error = RuntimeError("unexpected")

    before = classify_acp_error(error, operation=operation, stateful=False)
    after = classify_acp_error(error, operation=operation, stateful=True)

    assert isinstance(before, AcpConnectError)
    assert isinstance(after, AcpTransportError)
