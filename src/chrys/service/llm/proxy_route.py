# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which proxy an LLM HTTP client sends a URL through.

Both SDK default clients route by the same table httpx builds: the
environment's proxy map (``HTTP(S)_PROXY``, ``ALL_PROXY``, ``NO_PROXY``) with
the client's explicit mounts laid over it, sorted most specific first, first
match wins.  :class:`ProxyRouter` rebuilds that table once, when the client is
built, so a request hook can name its first hop without reaching into the
client's transports.  ``tests/service/llm/test_proxy_route_contract.py`` pins
it to ``httpx.AsyncClient._transport_for_url`` for both SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chrys.foundation.errors import Origin, origin_of
from chrys.foundation.util.httpx_helpers import BYPASS_PROXY_MOUNTS

if TYPE_CHECKING:
    import httpx
    from httpx._utils import URLPattern


@dataclass(frozen=True, slots=True)
class ProxyRouter:
    """A client's proxy table: URL patterns, most specific first, to a proxy origin or direct."""

    routes: tuple[tuple[URLPattern, Origin | None], ...]

    @classmethod
    def from_client_config(cls, *, bypass_proxy: bool) -> ProxyRouter:
        """Build the table the client built at the same moment from the same environment.

        Call it next to the client's construction: httpx and the SDKs read the
        proxy environment once, when the client is built.
        """
        from httpx._utils import URLPattern, get_environment_proxies

        table: dict[URLPattern, Origin | None] = {
            URLPattern(key): None if url is None else origin_of(url) for key, url in get_environment_proxies().items()
        }
        if bypass_proxy:
            table.update({URLPattern(key): None for key in BYPASS_PROXY_MOUNTS})
        return cls(tuple(sorted(table.items())))

    def proxy_for(self, url: httpx.URL) -> Origin | None:
        """Return the proxy *url* goes through (credentials never kept), or None when it goes direct."""
        for pattern, proxy in self.routes:
            if pattern.matches(url):
                return proxy
        return None
