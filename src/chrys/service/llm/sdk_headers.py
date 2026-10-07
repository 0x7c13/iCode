# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Static request headers for the provider SDK client subclasses.

It imports no provider SDK, so each SDK's subclass module loads only its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:

    class _HeadersBase(Protocol):
        _custom_headers: Mapping[str, str]

        @property
        def auth_headers(self) -> dict[str, str]: ...

        @property
        def default_headers(self) -> dict[str, Any]: ...

else:
    _HeadersBase = object


class CaseFoldedDefaultHeaders(_HeadersBase):
    """Send each static header name once, whatever case a profile spells it in.

    The pinned SDKs merge their own headers, Chrys's and the profile's as
    dictionaries, case-sensitively, so a profile's ``user-agent`` or
    ``openai-organization`` would go out beside, or be dropped for, the
    ``User-Agent`` or ``OpenAI-Organization`` the SDK or Chrys sets. Names
    are folded under their first spelling, or the auth header's, and the
    client's custom headers (Chrys's and the profile's, the SDK's
    ``_custom_headers``) are laid over last, as they are when spelled alike.
    """

    @property
    def default_headers(self) -> dict[str, Any]:
        spellings = {name.casefold(): name for name in self.auth_headers}
        folded: dict[str, Any] = {}
        for headers in (super().default_headers, self._custom_headers):
            for name, value in headers.items():
                folded[spellings.setdefault(name.casefold(), name)] = value
        return folded
