# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The explicit exception graph: which edges it follows, in what order, and where it stops."""

from __future__ import annotations

from typing import NoReturn

import httpx

from chrys.foundation.errors._walk import iter_explicit_graph, iter_full_chain, request_of
from tests.support.provider_errors import raised_from, raised_while_handling


class _Holder(Exception):
    """An exception keeping another in ``args``, as httpcore and ChatClientException do."""


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_cause_subtree_is_walked_before_exceptions_in_args() -> None:
    deep = ValueError("deep")
    cause = raised_from(KeyError("cause"), deep)
    held = OSError("held")
    top = raised_from(_Holder("top", held), cause)

    assert list(iter_explicit_graph(top)) == [top, cause, deep, held]


def test_group_members_follow_the_cause() -> None:
    cause = ValueError("cause")
    first, second = OSError("first"), OSError("second")
    group = raised_from(ExceptionGroup("attempts", [first, second]), cause)

    assert list(iter_explicit_graph(group)) == [group, cause, first, second]


def test_each_exception_is_yielded_once_even_in_a_cycle() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__cause__ = b
    b.__cause__ = a
    shared = OSError("shared")
    holder = raised_from(_Holder("holder", shared), shared)

    assert list(iter_explicit_graph(a)) == [a, b]
    assert list(iter_explicit_graph(holder)) == [holder, shared]


def test_implicit_context_is_never_followed() -> None:
    stale = _StatusError("stale 413", 413)

    def raise_fresh() -> NoReturn:
        raise httpx.ConnectError("All connection attempts failed")

    fresh = raised_while_handling(stale, raise_fresh)

    assert stale in list(iter_full_chain(fresh))
    assert list(iter_explicit_graph(fresh)) == [fresh]


def test_walk_stops_below_a_node_whose_request_got_a_response() -> None:
    parse_failure = ValueError("bad json")
    status = raised_from(_StatusError("Error code: 500", 500), parse_failure)
    top = raised_from(RuntimeError("wrapper"), status)

    assert list(iter_explicit_graph(top)) == [top, status]


def test_request_of_reads_only_an_attached_request() -> None:
    request = httpx.Request("GET", "https://api.example.test/v1/models")

    assert request_of(httpx.ConnectError("failed", request=request)) is request
    assert request_of(httpx.ConnectError("failed")) is None
    assert request_of(ValueError("no request")) is None
