# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Stamp route facts on every LLM request, and remember which first hops ever answered.

The request hook runs before httpx sends each request — SDK retries and
redirects included — and stamps a :class:`~chrys.foundation.errors.RouteFacts`
snapshot into its extensions.  Every ``httpx.Request`` copies the extensions it
is built with, so a redirect starts from a copy holding the previous hop's
snapshot, which the hook replaces.  The response hook records the first hop as
reached: any HTTP response through it proves the process could reach it at
least once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from chrys.foundation.errors import ROUTE_EXTENSION_KEY, Origin, RouteFacts, origin_of

if TYPE_CHECKING:
    import httpx

    from chrys.service.llm.proxy_route import ProxyRouter

# First hops (a proxy, or the target when direct) that returned at least one
# HTTP response in this process. Never cleared at runtime: it is evidence that
# a name once resolved, not proof it resolves now.
_REACHED: set[Origin] = set()


def record_reached(first_hop: Origin) -> None:
    _REACHED.add(first_hop)


def has_reached(first_hop: Origin) -> bool:
    return first_hop in _REACHED


def reset_reached_first_hops() -> None:
    """Forget every reached first hop; tests only."""
    _REACHED.clear()


def build_route_hooks(router: ProxyRouter) -> dict[str, list[Callable[..., Any]]]:
    """Return httpx event hooks that stamp and record *router*'s routes; list them first."""

    async def stamp(request: httpx.Request) -> None:
        target = origin_of(request.url)
        if target is None:
            request.extensions.pop(ROUTE_EXTENSION_KEY, None)
            return
        proxy = router.proxy_for(request.url)
        request.extensions[ROUTE_EXTENSION_KEY] = RouteFacts(
            target=target, proxy=proxy, first_hop_reached=has_reached(proxy or target)
        )

    async def record(response: httpx.Response) -> None:
        facts = response.request.extensions.get(ROUTE_EXTENSION_KEY)
        if isinstance(facts, RouteFacts):
            record_reached(facts.first_hop)

    return {"request": [stamp], "response": [record]}
