# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Charset validation for values that travel in HTTP headers.

httpx encodes header names and values as ASCII, so one stray non-ASCII
or control character in an API key, model id, or custom header aborts
the request locally — an opaque ``UnicodeEncodeError`` raised before
anything reaches the provider.  The validators here reject exactly that
class of value and deliberately nothing more: no provider key prefixes,
no length rules, no model-id grammar.  Custom gateways and
OpenAI-compatible servers keep working, and a genuinely wrong-but-
encodable credential still fails server-side with a clear 401.

Values are constrained to printable ASCII (U+0020 through U+007E) with
no leading or trailing space: h11 rejects field values with outer
whitespace (``LocalProtocolError``), so an edge space fails locally just
like a non-ASCII character — internal spaces remain allowed.  HTAB is
technically legal in RFC 9110 field values but never needed by API
credentials or model ids; rejecting all control characters also keeps
CR/LF header smuggling impossible.

Secret-bearing fields (API keys, header values) are never echoed back:
their messages report only the 1-based character position.
"""

from __future__ import annotations

import string

# RFC 9110 ``token`` — the grammar for HTTP field (header) names.
_HEADER_NAME_SPECIALS = "!#$%&'*+-.^_`|~"
_HEADER_NAME_CHARS = frozenset(_HEADER_NAME_SPECIALS + string.ascii_letters + string.digits)


def find_non_printable_ascii(value: str) -> int | None:
    """Return the index of the first character outside printable ASCII, else ``None``."""
    for index, char in enumerate(value):
        if not "\x20" <= char <= "\x7e":
            return index
    return None


def _describe_char(char: str) -> str:
    code = ord(char)
    if code < 0x20 or code == 0x7F:
        return f"a control character (U+{code:04X})"
    return f"unsupported character {char!r} (U+{code:04X})"


def _describe_outer_space(value: str) -> str | None:
    """Describe a leading/trailing space, else ``None``.

    Only plain spaces need this check: every other whitespace character
    is a control character and already fails the printable-ASCII scan.
    """
    if value.startswith(" "):
        return "starts with a space"
    if value.endswith(" "):
        return "ends with a space"
    return None


def api_key_charset_error(value: str) -> str | None:
    """Return an error message when *value* cannot be sent as an HTTP credential.

    API keys ride in ``Authorization`` / ``x-api-key`` headers, so the
    whole key must be printable ASCII.  The key is a secret: the message
    reports the character position, never the content.
    """
    index = find_non_printable_ascii(value)
    if index is not None:
        return (
            f"API key contains a non-ASCII or control character at position {index + 1}; "
            "only printable ASCII is supported."
        )
    outer_space = _describe_outer_space(value)
    if outer_space is not None:
        return f"API key {outer_space}; HTTP header values cannot carry leading or trailing whitespace."
    return None


def model_id_charset_error(value: str) -> str | None:
    """Return an error message when *value* cannot ride in an HTTP header.

    Model ids travel in the ``Chrys-Model-Id`` header as well as the
    request body, so they carry the same printable-ASCII constraint as
    headers.  Not a secret — the offending character is echoed so the
    fix is obvious.
    """
    if not value:
        return "Model ID is empty."
    index = find_non_printable_ascii(value)
    if index is not None:
        return (
            f"Model ID contains {_describe_char(value[index])} at position {index + 1}; "
            "only printable ASCII is supported."
        )
    outer_space = _describe_outer_space(value)
    if outer_space is not None:
        return f"Model ID {outer_space}; HTTP header values cannot carry leading or trailing whitespace."
    return None


def header_name_charset_error(name: str) -> str | None:
    """Return an error message when *name* is not a valid HTTP header name.

    Header names must match the RFC 9110 ``token`` grammar; a name is
    not a secret, so both the name and the offending character are
    echoed.
    """
    if not name:
        return "Header name is empty."
    for index, char in enumerate(name):
        if char not in _HEADER_NAME_CHARS:
            return (
                f"Header name {name!r} contains {_describe_char(char)} at position {index + 1}; "
                f"header names may only use letters, digits, and {_HEADER_NAME_SPECIALS} characters."
            )
    return None


def header_value_charset_error(name: str, value: str) -> str | None:
    """Return an error message when a header *value* cannot be encoded on the wire.

    Header values routinely carry credentials, so like API keys the
    message reports the position only, never the content.  *name* is
    echoed to identify the offending header.
    """
    index = find_non_printable_ascii(value)
    if index is not None:
        return (
            f"Header {name!r} value contains a non-ASCII or control character at position {index + 1}; "
            "only printable ASCII is supported."
        )
    outer_space = _describe_outer_space(value)
    if outer_space is not None:
        return f"Header {name!r} value {outer_space}; HTTP header values cannot carry leading or trailing whitespace."
    return None
