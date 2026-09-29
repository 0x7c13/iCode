# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exception-graph walks shared by the error classifier and formatter.

Two walks with different trust:

* :func:`iter_full_chain` follows ``__cause__`` and unsuppressed
  ``__context__``.  Implicit context can be stale — an exception that was
  being handled when a fresh request failed — so only the legacy retry layers
  and the owner-terminal veto read it.
* :func:`iter_explicit_graph` follows only edges a raiser set on purpose:
  ``__cause__``, exceptions held in ``args`` (httpcore keeps its root cause in
  ``args[0]`` after ``raise ... from None``; ``ChatClientException`` keeps its
  inner exception in ``args[1]``) and exception-group members.  It never
  follows ``__context__`` and does not descend below a node whose request got
  an HTTP response: anything deeper belongs to response processing, not to
  this request's failure.  Every classification rule other than the legacy
  layers reads this walk.
"""

from __future__ import annotations

from collections.abc import Iterator


def iter_full_chain(exc: BaseException, *, unwrap_single_groups: bool = False) -> Iterator[BaseException]:
    """Yield an exception and its explicit/implicit cause chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        yield cur
        seen.add(id(cur))
        if unwrap_single_groups and isinstance(cur, BaseExceptionGroup) and len(cur.exceptions) == 1:
            cur = cur.exceptions[0]
        else:
            cur = (
                cur.__cause__ if cur.__cause__ is not None else (None if cur.__suppress_context__ else cur.__context__)
            )


def has_response_status(exc: BaseException) -> bool:
    """Return whether *exc* proves that its request received an HTTP response."""
    if getattr(exc, "status_code", None) is not None:
        return True
    response = getattr(exc, "response", None)
    return response is not None and getattr(response, "status_code", None) is not None


def explicit_children(exc: BaseException) -> list[BaseException]:
    """Return *exc*'s explicit children: cause, exceptions in ``args``, group members."""
    children: list[BaseException] = []
    if exc.__cause__ is not None:
        children.append(exc.__cause__)
    children.extend(arg for arg in exc.args if isinstance(arg, BaseException))
    if isinstance(exc, BaseExceptionGroup):
        children.extend(exc.exceptions)
    return children


def iter_explicit_graph(exc: BaseException) -> Iterator[BaseException]:
    """Yield *exc* and its explicit causes depth-first, pre-order.

    Children are visited cause first, then exceptions in ``args``, then group
    members.  Each exception is yielded once, so cycles terminate.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        if has_response_status(node):
            continue
        stack.extend(reversed(explicit_children(node)))


def request_of(exc: BaseException) -> object | None:
    """Return the HTTP request an SDK or httpx exception carries, if any."""
    try:
        # httpx raises RuntimeError from ``.request`` when none was attached.
        return getattr(exc, "request", None)
    except RuntimeError:
        return None
