# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The endpoint that issued a reasoning payload, so the payload replays only there.

Anthropic thinking signatures and redacted thinking, Responses encrypted
reasoning and Chat Completions ``reasoning_details`` are state only the
endpoint that issued them can read back. A client stamps such reasoning with
its protocol and the origin (scheme, host and port) of its base URL, and
replays it only to a client of the same protocol and origin; the model does
not count. Such reasoning without a stamp (received before stamps existed, or
from a base URL with no origin) replays as it did before; a stamp on it that
does not read as this version's never replays.

Anthropic thinking text goes with its signature. Other plaintext reasoning is
not gated: Chat Completions plaintext fields are never stamped, and a
message's stamp gates only its ``reasoning_details``; a plaintext Responses
dialect replays reasoning text whatever its stamp.

An origin is a coarse identity: tenants behind one origin on different paths,
another key or organization on the same URL, and a router that switches the
service behind it all look alike.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from chrys.foundation.errors.route import origin_of

REASONING_ORIGIN_KEY = "_chrys_reasoning_origin"
"""Content ``additional_properties`` key of the stamp; Chat Completions also sets it on the message."""

_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReasoningOrigin:
    """The endpoint one client sends to."""

    protocol: str
    origin: str

    @classmethod
    def of(cls, protocol: str, base_url: object) -> ReasoningOrigin | None:
        """The endpoint of a *protocol* client at *base_url*; None when the URL has no origin."""
        found = origin_of(base_url)
        if found is None:
            return None
        host = f"[{found.host}]" if ":" in found.host else found.host
        return cls(protocol, f"{found.scheme}://{host}:{found.port}")

    def stamp(self, properties: MutableMapping[str, Any]) -> None:
        """Record this endpoint as the issuer of the reasoning *properties* belong to."""
        properties[REASONING_ORIGIN_KEY] = self.stamp_value()

    def stamp_value(self) -> dict[str, Any]:
        """The stamp this endpoint writes, a fresh dict each time."""
        return {"v": _VERSION, "protocol": self.protocol, "origin": self.origin}


def replays_to(properties: Mapping[str, Any], endpoint: ReasoningOrigin | None) -> bool:
    """Whether reasoning with *properties* may be sent to *endpoint* (None: an unknown one)."""
    if REASONING_ORIGIN_KEY not in properties:
        return True
    return endpoint is not None and properties[REASONING_ORIGIN_KEY] == endpoint.stamp_value()
