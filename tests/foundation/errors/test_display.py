# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What each classified error says to the user: the message, the proxy wording, and when the offline hint is asked for."""

from __future__ import annotations

import logging
import socket
import sys
from collections.abc import Callable

import httpx
import pytest

from chrys.foundation.errors import (
    ROUTE_EXTENSION_KEY,
    ErrorKind,
    Origin,
    ProviderResponseError,
    RouteFacts,
    classify_error,
)
from chrys.foundation.errors import display as display_module
from chrys.foundation.errors.classify import ErrorClassification
from chrys.foundation.errors.display import DISPLAY_WITH_HINT, ErrorDescription, describe_error, display_fields
from chrys.foundation.errors.kinds import NETWORK_KINDS
from chrys.foundation.errors.network import codes_for
from chrys.foundation.i18n import Localizer, MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.kernel.exceptions import ChatClientContentFilterException
from tests.support.provider_errors import API_HOST, api_request, raised_from

_TARGET = Origin("https", API_HOST, 443)
_PROXY = Origin("http", "127.0.0.1", 7897)
_DIRECT = RouteFacts(_TARGET, None, first_hop_reached=False)
_PROXIED = RouteFacts(_TARGET, _PROXY, first_hop_reached=False)

# Every kind's direct wording; None falls back to the raw text.
_DIRECT_KEYS: dict[ErrorKind, str | None] = {
    ErrorKind.NO_ROUTE: "error.kind.no_route",
    ErrorKind.HOST_UNREACHABLE: "error.kind.host_unreachable",
    ErrorKind.DNS_FAILED: "error.kind.dns_failed",
    ErrorKind.CONNECTION_REFUSED: "error.kind.connection_refused",
    ErrorKind.CONNECT_TIMEOUT: "error.kind.connect_timeout",
    ErrorKind.CONNECTION_FAILED: "error.kind.connection_failed",
    ErrorKind.CONNECTION_LOST: "error.kind.connection_lost",
    ErrorKind.READ_TIMEOUT: "error.kind.read_timeout",
    ErrorKind.WRITE_TIMEOUT: "error.kind.write_timeout",
    # A proxy verdict needs a proxy to name; a direct route has none.
    ErrorKind.PROXY_REJECTED: "error.kind.network_generic",
    ErrorKind.PROXY_AUTH_FAILED: "error.kind.network_generic",
    ErrorKind.TLS_FAILED: "error.kind.tls_failed",
    ErrorKind.INVALID_ENDPOINT: "error.kind.invalid_endpoint",
    ErrorKind.RATE_LIMITED: "error.kind.rate_limited",
    ErrorKind.QUOTA_EXHAUSTED: "error.kind.quota_exhausted",
    ErrorKind.OVERLOADED: "error.kind.overloaded",
    ErrorKind.SERVER_ERROR: "error.kind.server_error",
    ErrorKind.CONTEXT_OVERFLOW: "error.kind.context_overflow",
    ErrorKind.PAYLOAD_TOO_LARGE: "error.kind.payload_too_large",
    ErrorKind.AUTH_FAILED: "error.kind.auth_failed",
    ErrorKind.REQUEST_REJECTED: None,
    ErrorKind.CONTENT_FILTERED: "error.kind.content_filtered",
    ErrorKind.STREAM_TRUNCATED: "error.kind.stream_truncated",
    ErrorKind.STREAM_STALLED: None,
    ErrorKind.INVALID_RESPONSE: None,
    ErrorKind.UNKNOWN: None,
}
_HOSTLESS = {
    "error.kind.network_generic",
    "error.kind.invalid_endpoint",
    "error.kind.rate_limited",
    "error.kind.quota_exhausted",
    "error.kind.overloaded",
    "error.kind.server_error",
    "error.kind.context_overflow",
    "error.kind.payload_too_large",
    "error.kind.auth_failed",
    "error.kind.content_filtered",
    "error.kind.stream_truncated",
}
# Sorted: parametrize ids must match on every xdist worker.
_NETWORK_KINDS = sorted(NETWORK_KINDS)


class _Probe:
    def __init__(self, answer: bool | None) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self) -> bool | None:
        self.calls += 1
        return self.answer


def _classified(
    monkeypatch: pytest.MonkeyPatch,
    kind: ErrorKind,
    route: RouteFacts | None,
    *,
    failed_at_first_hop: bool = False,
) -> None:
    """Pin the classifier's verdict: these tests are about what a verdict says, not how it is reached."""
    verdict = ErrorClassification(
        kind=kind,
        retryable=True,
        route=route,
        failed_at_first_hop=failed_at_first_hop,
        from_model_service=route is not None,
    )
    monkeypatch.setattr(display_module, "classify_error", lambda _exc: verdict)


def _describe(probe: Callable[[], bool | None] | None = None) -> ErrorDescription | None:
    return describe_error(RuntimeError("stand-in"), route_probe=probe or _Probe(True))


def _key(ref: MessageRef | None) -> str | None:
    return None if ref is None else ref.definition.key


def _args(ref: MessageRef) -> dict[str, object]:
    return dict(ref.args)


def test_the_table_covers_every_kind() -> None:
    assert set(_DIRECT_KEYS) == set(ErrorKind)


@pytest.mark.parametrize("kind", list(ErrorKind), ids=str)
def test_each_kind_direct(monkeypatch: pytest.MonkeyPatch, kind: ErrorKind) -> None:
    _classified(monkeypatch, kind, _DIRECT)

    description = _describe()

    expected = _DIRECT_KEYS[kind]
    if expected is None:
        assert description is None
        return
    assert description is not None
    assert (description.kind, _key(description.message), description.hint) == (kind, expected, None)
    assert _args(description.message) == ({} if expected in _HOSTLESS else {"host": API_HOST})


@pytest.mark.parametrize("kind", _NETWORK_KINDS, ids=str)
def test_a_network_failure_without_a_route_keeps_the_raw_text(monkeypatch: pytest.MonkeyPatch, kind: ErrorKind) -> None:
    # Only the LLM client stamps a route: without one, nothing says this was the model service.
    _classified(monkeypatch, kind, None)
    probe = _Probe(False)

    assert (_describe(probe), probe.calls) == (None, 0)


def test_a_routeless_transport_failure_from_another_client_keeps_the_raw_text() -> None:
    exc = raised_from(RuntimeError("MCP connect failed"), httpx.ConnectError("[Errno 61] Connection refused"))
    assert classify_error(exc).kind is ErrorKind.CONNECTION_FAILED
    probe = _Probe(False)

    assert (describe_error(exc, route_probe=probe), probe.calls) == (None, 0)


def _status_error(status: int, *, routed: bool) -> BaseException:
    request = api_request()
    if routed:
        request.extensions = {**request.extensions, ROUTE_EXTENSION_KEY: _DIRECT}
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


@pytest.mark.parametrize(("status", "kind"), [(401, ErrorKind.AUTH_FAILED), (429, ErrorKind.RATE_LIMITED)])
def test_another_clients_http_status_never_blames_the_model_service(status: int, kind: ErrorKind) -> None:
    # An MCP server's 401 must not send the user to the model profile's API key.
    other = raised_from(RuntimeError("MCP server 'docs' failed to connect"), _status_error(status, routed=False))
    model = raised_from(RuntimeError("Model request failed"), _status_error(status, routed=True))

    assert (classify_error(other).kind, classify_error(model).kind) == (kind, kind)
    assert describe_error(other) is None
    description = describe_error(model)
    assert description is not None
    assert description.kind is kind


def test_model_response_types_speak_for_the_model_service_without_a_route() -> None:
    # Raised by Chrys's model clients for a response, with no request attached.
    filtered = ChatClientContentFilterException("blocked")
    truncated = ProviderResponseError("stream_truncated", "the stream ended early", retryable=True)

    for exc, key in ((filtered, "error.kind.content_filtered"), (truncated, "error.kind.stream_truncated")):
        description = describe_error(raised_from(RuntimeError("run failed"), exc))
        assert description is not None
        assert _key(description.message) == key


@pytest.mark.parametrize(
    ("target", "label"),
    [
        (Origin("https", "api.example.test", 443), "api.example.test"),
        (Origin("http", "api.example.test", 80), "api.example.test"),
        (Origin("http", "127.0.0.1", 8080), "127.0.0.1:8080"),
        (Origin("https", "api.example.test", 80), "api.example.test:80"),
        (Origin("https", "2001:db8::5", 443), "[2001:db8::5]"),
        (Origin("http", "::1", 11434), "[::1]:11434"),
    ],
)
def test_the_host_names_its_port_only_when_not_the_default(
    monkeypatch: pytest.MonkeyPatch, target: Origin, label: str
) -> None:
    _classified(monkeypatch, ErrorKind.CONNECTION_REFUSED, RouteFacts(target, None, first_hop_reached=False))

    description = _describe()

    assert description is not None
    assert _args(description.message) == {"host": label}


@pytest.mark.parametrize(
    ("kind", "first_hop", "key"),
    [
        (ErrorKind.CONNECTION_REFUSED, True, "error.kind.proxy_unreachable"),
        (ErrorKind.DNS_FAILED, True, "error.kind.proxy_unreachable"),
        (ErrorKind.HOST_UNREACHABLE, True, "error.kind.proxy_unreachable"),
        (ErrorKind.NO_ROUTE, True, "error.kind.proxy_unreachable"),
        (ErrorKind.PROXY_REJECTED, False, "error.kind.proxy_rejected"),
        (ErrorKind.PROXY_AUTH_FAILED, False, "error.kind.proxy_auth_failed"),
        (ErrorKind.CONNECT_TIMEOUT, False, "error.kind.via_proxy_failed"),
        (ErrorKind.READ_TIMEOUT, False, "error.kind.via_proxy_failed"),
        (ErrorKind.WRITE_TIMEOUT, False, "error.kind.via_proxy_failed"),
        (ErrorKind.CONNECTION_FAILED, False, "error.kind.via_proxy_failed"),
        (ErrorKind.CONNECTION_LOST, False, "error.kind.via_proxy_failed"),
        # An untrusted certificate through a proxy is most often the proxy's own.
        (ErrorKind.TLS_FAILED, False, "error.kind.tls_failed"),
        # The base URL itself is wrong, whichever hop the request took.
        (ErrorKind.INVALID_ENDPOINT, True, "error.kind.invalid_endpoint"),
    ],
)
def test_proxied_wording_follows_first_hop_evidence(
    monkeypatch: pytest.MonkeyPatch, kind: ErrorKind, first_hop: bool, key: str
) -> None:
    _classified(monkeypatch, kind, _PROXIED, failed_at_first_hop=first_hop)
    probe = _Probe(False)

    description = _describe(probe)

    assert description is not None
    assert (_key(description.message), description.hint) == (key, None)
    # A proxied request says nothing about this machine's own route.
    assert probe.calls == 0
    expected_args = {
        "error.kind.proxy_unreachable": {"proxy": "127.0.0.1:7897"},
        "error.kind.proxy_auth_failed": {"proxy": "127.0.0.1:7897"},
        "error.kind.tls_failed": {"host": API_HOST},
        "error.kind.invalid_endpoint": {},
    }.get(key, {"host": API_HOST, "proxy": "127.0.0.1:7897"})
    assert _args(description.message) == expected_args


@pytest.mark.parametrize("kind", [ErrorKind.DNS_FAILED, ErrorKind.CONNECT_TIMEOUT, ErrorKind.CONNECTION_FAILED])
@pytest.mark.parametrize(("answer", "hinted"), [(False, True), (True, False), (None, False)])
def test_the_offline_hint_follows_the_probe(
    monkeypatch: pytest.MonkeyPatch, kind: ErrorKind, answer: bool | None, hinted: bool
) -> None:
    _classified(monkeypatch, kind, _DIRECT)
    probe = _Probe(answer)

    description = _describe(probe)

    assert description is not None
    assert probe.calls == 1
    assert _key(description.message) == _DIRECT_KEYS[kind]
    assert _key(description.hint) == ("error.hint.maybe_offline" if hinted else None)


@pytest.mark.parametrize(
    "kind",
    [
        kind
        for kind in _NETWORK_KINDS
        if kind not in {ErrorKind.DNS_FAILED, ErrorKind.CONNECT_TIMEOUT, ErrorKind.CONNECTION_FAILED}
    ],
    ids=str,
)
def test_kinds_that_name_their_cause_never_probe(monkeypatch: pytest.MonkeyPatch, kind: ErrorKind) -> None:
    _classified(monkeypatch, kind, _DIRECT)
    probe = _Probe(False)

    description = _describe(probe)

    assert description is not None
    assert (probe.calls, description.hint) == (0, None)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "192.168.1.20", "10.1.2.3", "fd00::7", "nas.local"])
def test_a_local_or_private_target_never_probes(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    _classified(monkeypatch, ErrorKind.CONNECT_TIMEOUT, RouteFacts(Origin("http", host, 8000), None, False))
    probe = _Probe(False)

    description = _describe(probe)

    assert description is not None
    assert (probe.calls, description.hint) == (0, None)


def test_a_real_routed_chain_names_the_host_that_failed_to_resolve() -> None:
    codes = codes_for(sys.platform)
    if codes is None:
        pytest.skip("no code table for this OS")
    request = api_request()
    request.extensions = {**request.extensions, ROUTE_EXTENSION_KEY: _DIRECT}
    leaf = socket.gaierror(codes.eai_again, "temporary failure in name resolution")
    exc = raised_from(RuntimeError("Connection error."), raised_from(httpx.ConnectError("x", request=request), leaf))
    probe = _Probe(None)

    description = describe_error(exc, route_probe=probe)

    assert description is not None
    assert (_key(description.message), _args(description.message)) == ("error.kind.dns_failed", {"host": API_HOST})
    assert probe.calls == 1


@pytest.mark.parametrize("absent_first", [True, False], ids=["absent-first", "absent-last"])
def test_a_missing_address_family_never_hides_an_unreachable_proxy(absent_first: bool) -> None:
    codes = codes_for(sys.platform)
    if codes is None:
        pytest.skip("no code table for this OS")
    request = api_request()
    request.extensions = {**request.extensions, ROUTE_EXTENSION_KEY: _PROXIED}
    absent = OSError(codes.eafnosupport, "Address family not supported by protocol")
    no_route = OSError(codes.enetunreach, "Connect call failed ('127.0.0.1', 7897)")
    attempts = ExceptionGroup("attempts", [absent, no_route] if absent_first else [no_route, absent])
    connect = raised_from(httpx.ConnectError("x", request=request), raised_from(OSError("All attempts"), attempts))

    description = describe_error(raised_from(RuntimeError("Connection error."), connect), route_probe=_Probe(None))

    assert description is not None
    assert (_key(description.message), _args(description.message)) == (
        "error.kind.proxy_unreachable",
        {"proxy": "127.0.0.1:7897"},
    )


def test_display_fields_returns_message_and_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _classified(monkeypatch, ErrorKind.DNS_FAILED, _DIRECT)
    description = _describe(_Probe(False))
    assert description is not None
    monkeypatch.setattr(
        display_module,
        "describe_error",
        lambda _exc, *, retry_notice=False, max_context_tokens=None: description,
    )

    message, hint = display_fields(RuntimeError("stand-in"))

    assert (_key(message), _key(hint)) == ("error.kind.dns_failed", "error.hint.maybe_offline")


@pytest.mark.parametrize(("retry_notice", "key"), [(True, "retry.stream_stalled"), (False, None)])
def test_only_a_retry_notice_names_a_stalled_stream(
    monkeypatch: pytest.MonkeyPatch, retry_notice: bool, key: str | None
) -> None:
    # A paused sub-agent's card already labels a stall by its pause reason.
    _classified(monkeypatch, ErrorKind.STREAM_STALLED, None)

    message, hint = display_fields(RuntimeError("stand-in"), retry_notice=retry_notice)

    assert (_key(message), hint) == (key, None)


@pytest.mark.parametrize(
    ("retry_notice", "key"), [(True, "retry.context_overflow"), (False, "error.kind.context_overflow")]
)
def test_a_retry_notice_for_a_context_overflow_announces_the_resend_after_compacting(
    monkeypatch: pytest.MonkeyPatch, retry_notice: bool, key: str
) -> None:
    _classified(monkeypatch, ErrorKind.CONTEXT_OVERFLOW, _DIRECT)

    message, hint = display_fields(RuntimeError("stand-in"), retry_notice=retry_notice)

    assert (_key(message), hint) == (key, None)


_NAMES_131072 = "400: This model's maximum context length is 131072 tokens."
_MISMATCH = "error.kind.context_overflow_config_mismatch"


@pytest.mark.parametrize(
    ("text", "configured", "key", "args"),
    [
        (
            _NAMES_131072,
            200_000,
            _MISMATCH,
            {"configured_max_context_tokens": 200_000, "server_max_context_tokens": 131_072},
        ),
        (_NAMES_131072, 131_072, "error.kind.context_overflow", {}),
        (_NAMES_131072, None, "error.kind.context_overflow", {}),
        ("400: prompt is too long", 200_000, "error.kind.context_overflow", {}),
    ],
    ids=["server_limit_below_profile", "server_limit_equal", "window_unknown", "no_limit_named"],
)
def test_an_overflow_says_which_window_to_set_when_the_server_names_a_smaller_one(
    monkeypatch: pytest.MonkeyPatch, text: str, configured: int | None, key: str, args: dict[str, object]
) -> None:
    _classified(monkeypatch, ErrorKind.CONTEXT_OVERFLOW, _DIRECT)

    message, hint = display_fields(RuntimeError(text), max_context_tokens=configured)

    assert message is not None
    assert (_key(message), _args(message), hint) == (key, args, None)


def test_display_fields_is_empty_for_raw_text_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    _classified(monkeypatch, ErrorKind.UNKNOWN, _DIRECT)

    assert display_fields(RuntimeError("stand-in")) == (None, None)


def test_display_fields_swallows_its_own_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(_exc: BaseException) -> ErrorClassification:
        raise ValueError("classifier bug")

    monkeypatch.setattr(display_module, "classify_error", broken)

    with caplog.at_level(logging.DEBUG, logger=display_module.__name__):
        assert display_fields(RuntimeError("stand-in")) == (None, None)

    [record] = [record for record in caplog.records if record.name == display_module.__name__]
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)


def test_the_hint_joins_as_each_locale_joins() -> None:
    zh = Localizer("zh-Hans")

    assert format_message(DISPLAY_WITH_HINT.bind(message="A.", hint="B.")) == "A. B."
    assert zh.render(DISPLAY_WITH_HINT.bind(message="甲。", hint="乙。")) == "甲。乙。"
