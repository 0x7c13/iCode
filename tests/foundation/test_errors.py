# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for :mod:`chrys.foundation.errors`."""

from __future__ import annotations

import asyncio
import errno
import ssl
import sys
from collections.abc import Callable
from typing import Any

import pytest

from chrys.foundation.errors import ErrorKind, classify_error, clean_error_message
from chrys.foundation.errors._walk import iter_explicit_graph
from tests.support.network_faults import (
    INJECTED_V4,
    INJECTED_V6,
    NetworkFaults,
    SimulatedWindowsError,
    os_error,
    refused,
    socket_error_text,
    win_proactor_reset,
)
from tests.support.provider_errors import API_HOST, openai_network


def test_single_member_exception_groups_expose_the_leaf() -> None:
    error = ExceptionGroup("task group", [ExceptionGroup("nested", [ValueError("invalid UTF-8")])])
    error.__suppress_context__ = True
    assert clean_error_message(error) == "invalid UTF-8"
    assert clean_error_message(_make_chained(error)) == "invalid UTF-8"


def test_multiple_member_exception_group_is_not_arbitrarily_reduced() -> None:
    error = ExceptionGroup("two failures", [ValueError("first"), ValueError("second")])
    assert clean_error_message(_make_chained(error)) == str(error)


@pytest.mark.parametrize("message", ["MCP initialize failed: invalid UTF-8", "Background export stopped: disk full"])
@pytest.mark.parametrize("cancellation_detail", ["", "Cancelled via cancel scope"])
def test_cancelled_inner_keeps_diagnostic_wrapper(message: str, cancellation_detail: str) -> None:
    error = RuntimeError(message)
    error.__cause__ = asyncio.CancelledError(cancellation_detail)
    assert clean_error_message(error) == str(error)


def test_bare_nested_cancellation_keeps_the_generic_diagnostic() -> None:
    inner = asyncio.CancelledError()
    outer = asyncio.CancelledError()
    outer.__cause__ = inner
    assert clean_error_message(outer) == clean_error_message(inner)
    assert outer.__cause__ is inner


def _make_status_error(
    message: str,
    *,
    body: object,
    status_code: int = 400,
    response_text: str | None = None,
) -> Exception:
    cls = type("BadRequestError", (Exception,), {})
    exc = cls(message)
    exc.status_code = status_code  # type: ignore[attr-defined]
    exc.body = body  # type: ignore[attr-defined]
    if response_text is not None:
        exc.response = type("Response", (), {"text": response_text})()  # type: ignore[attr-defined]
    return exc


def _make_chained(cause: Exception, wrapper_msg: str = "service failed") -> Exception:
    try:
        raise RuntimeError(wrapper_msg) from cause
    except RuntimeError as exc:
        return exc


def _make_named_chained(cls_name: str, message: str, cause: Exception) -> Exception:
    cls = type(cls_name, (Exception,), {})
    try:
        raise cls(message) from cause
    except Exception as exc:
        return exc


@pytest.mark.parametrize(
    ("cls_name", "expected"),
    [
        ("ConnectError", "Connection failed (ConnectError)"),
        ("ConnectionResetError", "Connection reset (ConnectionResetError)"),
        ("ReadError", "Read failed (ReadError)"),
        ("ReadTimeout", "Read timed out (ReadTimeout)"),
        ("RemoteProtocolError", "Remote protocol error (RemoteProtocolError)"),
        ("LocalProtocolError", "Local protocol error (LocalProtocolError)"),
        ("ProxyError", "Proxy error (ProxyError)"),
    ],
)
def test_clean_error_message_falls_back_for_empty_http_transport_errors(cls_name: str, expected: str) -> None:
    cls = type(cls_name, (Exception,), {})
    exc = cls(TimeoutError())
    assert str(exc) == ""
    assert clean_error_message(exc) == expected


def test_clean_error_message_uses_empty_cause_type_over_wrapper_noise() -> None:
    class ReadTimeout(Exception):
        pass

    try:
        raise RuntimeError("Chat Completions request failed") from ReadTimeout(TimeoutError())
    except RuntimeError as exc:
        assert clean_error_message(exc) == "Read timed out (ReadTimeout)"


def test_clean_error_message_uses_empty_context_type_over_wrapper_noise() -> None:
    class RemoteProtocolError(Exception):
        pass

    try:
        raise RemoteProtocolError(TimeoutError())
    except RemoteProtocolError:
        try:
            raise RuntimeError("Chat Completions request failed")
        except RuntimeError as exc:
            assert clean_error_message(exc) == "Remote protocol error (RemoteProtocolError)"


def test_clean_error_message_exposes_tls_cause_hidden_by_sdk_connection_error() -> None:
    tls_error = ssl.SSLCertVerificationError(
        1,
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1016)",
    )
    transport_error = _make_named_chained("ConnectError", str(tls_error), tls_error)
    sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)
    wrapped = _make_chained(sdk_error, "Chat Completions request failed: Connection error.")

    assert clean_error_message(wrapped) == (
        "Connection error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "self-signed certificate (_ssl.c:1016)"
    )


def test_clean_error_message_preserves_generic_sdk_connection_error_without_cause() -> None:
    sdk_error = type("APIConnectionError", (Exception,), {})("Connection error.")

    assert clean_error_message(sdk_error) == "Connection error."


def test_clean_error_message_exposes_invalid_url_hidden_by_sdk_connection_error() -> None:
    url_error = type("InvalidURL", (Exception,), {})("Invalid port: 'not-a-port'")
    sdk_error = _make_named_chained("APIConnectionError", "Connection error.", url_error)
    wrapped = _make_chained(sdk_error, "Chat Completions request failed: Connection error.")

    assert clean_error_message(wrapped) == "Connection error: Invalid port: 'not-a-port'"


_REFUSED_LEAF = socket_error_text(errno.ECONNREFUSED)


def _refuse_v4(faults: NetworkFaults) -> None:
    faults.resolve_to(API_HOST, INJECTED_V4)
    faults.refuse(INJECTED_V4)


async def test_clean_error_message_shows_the_socket_error_below_all_connection_attempts_failed() -> None:
    exc = await openai_network(_refuse_v4)

    assert clean_error_message(exc) == f"All connection attempts failed: {_REFUSED_LEAF}"
    assert clean_error_message(_make_chained(exc, "Chat Completions request failed: Connection error.")) == (
        f"Connection error: {_REFUSED_LEAF}"
    )


async def test_the_socket_error_never_names_the_peer_address() -> None:
    # asyncio's message names the address it dialed; this text reaches the
    # model through sub-agent results, MCP startup errors and ACP.
    exc = await openai_network(_refuse_v4)
    leaves = [node for node in iter_explicit_graph(exc) if isinstance(node, ConnectionRefusedError)]

    assert [INJECTED_V4 in str(leaf) for leaf in leaves] == [True]
    assert INJECTED_V4 not in clean_error_message(exc)
    assert INJECTED_V4 not in clean_error_message(_make_chained(exc, "Connection error."))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (os_error(errno.ENETUNREACH), refused),
        (refused, os_error(errno.ENETUNREACH)),
    ],
    ids=["refused-last", "refused-first"],
)
async def test_a_dual_stack_group_shows_the_attempt_that_named_its_kind(
    first: Callable[[tuple[Any, ...]], OSError], second: Callable[[tuple[Any, ...]], OSError]
) -> None:
    def arrange(faults: NetworkFaults) -> None:
        faults.resolve_to(API_HOST, INJECTED_V6, INJECTED_V4)
        faults.fail_connect(INJECTED_V6, first)
        faults.fail_connect(INJECTED_V4, second)

    exc = await openai_network(arrange)

    # The refused attempt got furthest and names the group, whichever ran last.
    assert classify_error(exc).kind is ErrorKind.CONNECTION_REFUSED
    assert clean_error_message(exc) == f"All connection attempts failed: {_REFUSED_LEAF}"


@pytest.mark.parametrize("absent_first", [True, False], ids=["absent-first", "absent-last"])
async def test_an_address_family_the_machine_lacks_abstains(absent_first: bool) -> None:
    absent, no_route = os_error(errno.EAFNOSUPPORT), os_error(errno.ENETUNREACH)

    def arrange(faults: NetworkFaults) -> None:
        faults.resolve_to(API_HOST, INJECTED_V6, INJECTED_V4)
        faults.fail_connect(INJECTED_V6, absent if absent_first else no_route)
        faults.fail_connect(INJECTED_V4, no_route if absent_first else absent)

    result = classify_error(await openai_network(arrange))

    # The attempt that never left the machine outranks nothing, so the one
    # that had no route names the group and still proves the first hop failed.
    assert (result.kind, result.failed_at_first_hop) == (ErrorKind.NO_ROUTE, True)


async def test_an_unrecognized_attempt_that_names_the_group_shows_its_socket_error() -> None:
    def arrange(faults: NetworkFaults) -> None:
        faults.resolve_to(API_HOST, INJECTED_V6, INJECTED_V4)
        faults.fail_connect(INJECTED_V6, os_error(errno.EACCES))
        faults.fail_connect(INJECTED_V4, os_error(errno.ENETUNREACH))

    exc = await openai_network(arrange)

    assert classify_error(exc).kind is ErrorKind.CONNECTION_FAILED
    assert clean_error_message(exc) == f"All connection attempts failed: {socket_error_text(errno.EACCES)}"


def test_a_file_error_keeps_its_file_name() -> None:
    connect_error = type("ConnectError", (Exception,), {})("All connection attempts failed")
    connect_error.__cause__ = FileNotFoundError(errno.ENOENT, "Connect call failed", "/run/gateway.sock")

    assert clean_error_message(_make_chained(connect_error)) == (
        f"All connection attempts failed: {socket_error_text(errno.ENOENT)}: '/run/gateway.sock'"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Winsock codes need the Windows system's words")
def test_a_winsock_code_without_a_winerror_keeps_the_system_words() -> None:
    # A selector loop raises OSError(err, "Connect call failed ...") with no ``winerror``.
    connect_error = type("ConnectError", (Exception,), {})("All connection attempts failed")
    connect_error.__cause__ = OSError(10061, "Connect call failed ('10.1.2.3', 443)")

    detail = clean_error_message(_make_chained(connect_error)).removeprefix("All connection attempts failed: ")

    assert detail.startswith("[Errno 10061] ")
    assert detail.removeprefix("[Errno 10061] ").strip()
    assert "Unknown error" not in detail
    assert "10.1.2.3" not in detail


def test_a_windows_socket_error_keeps_the_system_words() -> None:
    leaf = SimulatedWindowsError(errno.EINVAL, "The remote computer refused the network connection", 1225)
    connect_error = type("ConnectError", (Exception,), {})("All connection attempts failed")
    connect_error.__cause__ = leaf

    assert clean_error_message(_make_chained(connect_error)) == (
        "All connection attempts failed: [WinError 1225] The remote computer refused the network connection"
    )


_NETNAME_DELETED = 64
_NETNAME_DELETED_WORDS = "The specified network name is no longer available."


async def test_a_proactor_reset_without_its_winerror_keeps_the_system_words() -> None:
    def arrange(faults: NetworkFaults) -> None:
        faults.resolve_to(API_HOST, INJECTED_V4)
        faults.fail_connect(INJECTED_V4, win_proactor_reset(_NETNAME_DELETED, _NETNAME_DELETED_WORDS))

    exc = await openai_network(arrange)

    assert classify_error(exc).kind is ErrorKind.CONNECTION_LOST
    # The stand-in EINVAL names nothing: never "[Errno 22] Invalid argument".
    assert clean_error_message(exc) == f"All connection attempts failed: {_NETNAME_DELETED_WORDS}"
    assert clean_error_message(_make_chained(exc, "Chat Completions request failed: Connection error.")) == (
        f"Connection error: {_NETNAME_DELETED_WORDS}"
    )


def test_clean_error_message_ignores_stale_context_below_transport_error() -> None:
    connect_error_cls = type("ConnectError", (Exception,), {})
    stale_transport_error: Exception | None = None
    try:
        raise ValueError("unrelated prior failure")
    except ValueError:
        try:
            raise connect_error_cls("Connection refused")
        except Exception as transport_error:
            stale_transport_error = transport_error

    assert stale_transport_error is not None
    assert isinstance(stale_transport_error.__context__, ValueError)
    sdk_error = _make_named_chained("APIConnectionError", "Connection error.", stale_transport_error)
    wrapped = _make_chained(sdk_error, "Chat Completions request failed: Connection error.")

    assert clean_error_message(wrapped) == "Connection error: Connection refused"


def test_clean_error_message_respects_suppressed_context() -> None:
    try:
        raise TimeoutError
    except TimeoutError:
        try:
            raise RuntimeError("not retryable") from None
        except RuntimeError as exc:
            assert exc.__suppress_context__ is True
            assert clean_error_message(exc) == "not retryable"


@pytest.mark.parametrize(
    ("body", "expected_detail"),
    [
        ({"message": "prompt is too long"}, "prompt is too long"),
        ({"error": {"message": "invalid tool schema"}}, "invalid tool schema"),
        ({"detail": [{"msg": "model does not support tools"}]}, "model does not support tools"),
    ],
)
def test_clean_error_message_includes_provider_status_body(body: Any, expected_detail: str) -> None:
    exc = _make_status_error("Error code: 400", body=body)

    assert clean_error_message(exc) == f"Error code: 400 - {expected_detail}"


def test_clean_error_message_uses_provider_body_from_wrapped_status_error() -> None:
    cause = _make_status_error("Error code: 400", body={"message": "chat template rejected tool call"})
    wrapped = _make_chained(cause, "Chat Completions request failed: Error code: 400")

    assert clean_error_message(wrapped) == "Error code: 400 - chat template rejected tool call"


def test_clean_error_message_drops_the_class_prefix_older_wrappers_wrote() -> None:
    legacy = RuntimeError(
        "<class 'chrys.service.llm.chat_completions.client.ChatCompletionsClient'> "
        "service failed to complete the prompt: Connection error."
    )

    assert clean_error_message(legacy) == "service failed to complete the prompt: Connection error."


def test_clean_error_message_uses_response_text_when_status_body_is_empty() -> None:
    exc = _make_status_error("Error code: 400", body=None, response_text="plain gateway rejection")

    assert clean_error_message(exc) == "Error code: 400 - plain gateway rejection"


def test_clean_error_message_preserves_status_code_when_provider_body_is_empty() -> None:
    exc = _make_status_error("Error code: 400", body="")

    assert clean_error_message(exc) == "Error code: 400"


def test_clean_error_message_handles_circular_provider_body() -> None:
    body: dict[str, object] = {}
    body["error"] = body
    exc = _make_status_error("Error code: 400", body=body)

    assert clean_error_message(exc) == "Error code: 400 - {'error': {...}}"


def test_clean_error_message_keeps_cause_text_over_outer_status_metadata() -> None:
    wrapped = _make_chained(ValueError("specific provider failure"), "Error code: 400")
    wrapped.status_code = 400  # type: ignore[attr-defined]
    wrapped.body = {"message": "less specific wrapper body"}  # type: ignore[attr-defined]

    assert clean_error_message(wrapped) == "specific provider failure"
