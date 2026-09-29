# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Legacy retryability rules, kept verbatim behind :func:`classify_error`.

The three transient layers (type name, status code, phrase) read the full
chain, implicit ``__context__`` included; new rules read only explicit
evidence and live in the classifier.  The deterministic-connection veto moved
to :mod:`.network`, which reads only explicit evidence.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._text import _clean_exception_text
from ._walk import iter_full_chain
from .network import RETRYABLE_PHRASES, is_deterministic_connection_error

# HTTP status codes that are transient and worth retrying after SDK
# retries are exhausted.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})

# SDK exception class names (checked by name to avoid hard imports).
RETRYABLE_TYPE_NAMES = frozenset(
    {
        # OpenAI SDK
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        # Bare ``openai.APIError`` is only raised in ``openai/_streaming.py``
        # when the upstream emits an in-band SSE ``error`` event mid-stream.
        # ``type(exc).__name__`` returns the most-derived class, so this name
        # match catches only the base class — subclasses (APIStatusError,
        # BadRequestError, etc.) keep their existing classification.
        "APIError",
        # httpx / httpcore transport failures.  These often carry no
        # message when wrapping a bare OS/timeout exception on Windows
        # (e.g. ``ReadTimeout(TimeoutError())``), so string fallback alone
        # cannot classify them.
        "CloseError",
        "ConnectError",
        "ConnectTimeout",
        "DecodingError",
        "NetworkError",
        "PoolTimeout",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "WriteError",
        "WriteTimeout",
        # Anthropic SDK
        "OverloadedError",
        "ServiceUnavailableError",
    }
)


_OWNER_TERMINAL_TYPE_NAMES = frozenset({"LastWordsGenerationError", "TerminalResponseValidationError"})


def is_owner_terminal(chain: Iterable[BaseException]) -> bool:
    """Return whether an owning component already concluded this failure.

    These failures have already exhausted or terminally concluded their owning
    component's policy, so an outer retry would replay a larger unit of work
    without making the failure recoverable.  Reads the full chain.
    """
    return any(type(exc).__name__ in _OWNER_TERMINAL_TYPE_NAMES for exc in chain)


def is_transient(chain: Iterable[BaseException]) -> bool:
    """Apply the three transient layers to a full chain.

    1. Exception type — Python builtins and SDK types (via class name)
    2. HTTP status code — ``status_code`` attribute on SDK exceptions
    3. String matching — fallback for transport-level errors
    """
    chain = tuple(chain)
    for exc in chain:
        # Python built-in transient exceptions
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        # SDK exception types (by name to avoid hard imports)
        if type(exc).__name__ in RETRYABLE_TYPE_NAMES:
            return True
        # HTTP status code (works for both OpenAI and Anthropic SDKs)
        status = getattr(exc, "status_code", None)
        if status is not None and status in RETRYABLE_STATUS_CODES:
            return True
    # Fallback: string matching for transport-level errors
    for exc in chain:
        error_msg = _clean_exception_text(exc).lower()
        if any(phrase in error_msg for phrase in RETRYABLE_PHRASES):
            return True
    return False


def is_retryable(e: BaseException) -> bool:
    """The legacy retry decision: owner veto, deterministic veto, transient layers.

    Deterministic non-retryable root causes take precedence over the
    three-layer transient-error strategy.  The classifier decides both vetoes
    itself and calls :func:`is_transient`; the corpus pins this whole rule as
    the reference for every case the classifier left unchanged.
    """
    chain = tuple(iter_full_chain(e))
    if is_owner_terminal(chain):
        return False
    if is_deterministic_connection_error(e):
        # SDKs wrap deterministic transport, protocol, DNS, and TLS failures
        # in retryable APIConnectionError/ConnectError types. The root cause
        # must win: repeating the same request cannot fix it.
        return False
    return is_transient(chain)
