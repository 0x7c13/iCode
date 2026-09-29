# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The classifier against the corpus, and what stale implicit context may never do."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import NoReturn

import pytest

from chrys.foundation.errors import ErrorKind, _legacy, classify_error, is_retryable
from chrys.foundation.errors._walk import iter_full_chain
from tests.foundation.errors._corpus import CORPUS, FRESH_FAILURES, STALE_PROVIDER_ERRORS, Case
from tests.support.provider_errors import raised_while_handling


@pytest.mark.parametrize("case", CORPUS, ids=lambda case: case.id)
async def test_corpus_classifies_as_recorded(case: Case) -> None:
    exc = await case.build()

    result = classify_error(exc)

    assert (result.kind, result.retryable) == (case.kind, case.retryable)
    assert is_retryable(exc) is case.retryable


@pytest.mark.parametrize("case", [case for case in CORPUS if not case.changed], ids=lambda case: case.id)
async def test_unchanged_cases_keep_the_legacy_retry_decision(case: Case) -> None:
    exc = await case.build()

    assert classify_error(exc).retryable is _legacy.is_retryable(exc)


@pytest.mark.parametrize("case", CORPUS, ids=lambda case: case.id)
async def test_the_classifier_decides_each_legacy_veto_once(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    exc = await case.build()
    owner_checks: list[object] = []
    real_owner_terminal = _legacy.is_owner_terminal

    def count_owner_terminal(chain: Iterable[BaseException]) -> bool:
        owner_checks.append(chain)
        return real_owner_terminal(chain)

    def refuse(_exc: BaseException) -> NoReturn:
        raise AssertionError("the classifier already judged the connection failure")

    monkeypatch.setattr(_legacy, "is_owner_terminal", count_owner_terminal)
    monkeypatch.setattr(_legacy, "is_deterministic_connection_error", refuse)

    assert classify_error(exc).retryable is case.retryable
    assert len(owner_checks) == 1


# The kind each fresh failure classifies as on its own.
_FRESH_KINDS = {
    "httpx-connect-error": ErrorKind.CONNECTION_FAILED,
    "api-connection-error": ErrorKind.CONNECTION_FAILED,
    "httpx-read-timeout": ErrorKind.READ_TIMEOUT,
    "remote-protocol-error": ErrorKind.CONNECTION_LOST,
}


@pytest.mark.parametrize("fresh", sorted(FRESH_FAILURES))
@pytest.mark.parametrize("stale", sorted(STALE_PROVIDER_ERRORS))
async def test_stale_context_never_steers_kind_or_retry(stale: str, fresh: str) -> None:
    build_stale: Callable[[], Awaitable[BaseException]] = STALE_PROVIDER_ERRORS[stale]
    raise_fresh: Callable[[], NoReturn] = FRESH_FAILURES[fresh]
    stale_error = await build_stale()
    stale_kind = classify_error(stale_error).kind

    exc = raised_while_handling(stale_error, raise_fresh)

    # The stale error really is on the full chain, so the legacy layers can see it.
    assert stale_error in list(iter_full_chain(exc))
    result = classify_error(exc)
    assert result.retryable is True
    assert _legacy.is_retryable(exc) is True
    assert result.kind != stale_kind
    assert result.kind is _FRESH_KINDS[fresh]
