# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Route facts: where a request was going, and through which first hop.

The LLM HTTP client stamps a :class:`RouteFacts` snapshot on every request it
sends (``request.extensions[ROUTE_EXTENSION_KEY]``), redirects and SDK retries
included.  The classifier reads it back from the failed request carried on the
exception chain, so a verdict about a failure never depends on process-global
state read at classification time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from chrys.foundation.net.url import hostname

from ._walk import request_of

ROUTE_EXTENSION_KEY = "chrys.route"

_DEFAULT_PORTS = {"http": 80, "https": 443, "socks5": 1080, "socks5h": 1080}


@dataclass(frozen=True, slots=True)
class Origin:
    """A scheme/host/port identity; the host is lowercase, IPv6 without brackets."""

    scheme: str
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class RouteFacts:
    """One request's route, as its client resolved it when the request left."""

    target: Origin
    # The proxy this client used for ``target``, credentials stripped.
    proxy: Origin | None
    # Whether this process had EVER received a response through the same
    # first hop before this request was sent — never proof it is reachable now.
    first_hop_reached: bool

    @property
    def first_hop(self) -> Origin:
        return self.proxy or self.target


def origin_of(url: object) -> Origin | None:
    """Return the origin of *url* (a string or an ``httpx.URL``), or None when it has none."""
    try:
        parts = urlsplit(str(url))
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname
    if not scheme or not host:
        return None
    port = port if port is not None else _DEFAULT_PORTS.get(scheme)
    if port is None:
        return None
    with contextlib.suppress(ValueError):
        host = hostname(host)
    return Origin(scheme, host, port)


def route_of(explicit: Iterable[BaseException]) -> RouteFacts | None:
    """Return the route snapshot of the deepest explicit node whose request carries one.

    After a redirect the inner transport error holds the redirected request,
    so the deepest snapshot names the target that actually failed.
    """
    found: RouteFacts | None = None
    for node in explicit:
        request = request_of(node)
        extensions = getattr(request, "extensions", None) if request is not None else None
        facts = extensions.get(ROUTE_EXTENSION_KEY) if isinstance(extensions, Mapping) else None
        if isinstance(facts, RouteFacts):
            found = facts
    return found
