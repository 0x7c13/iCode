# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Drive real httpcore timeout scopes from network events, not scheduler speed."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import anyio
import httpcore._backends.anyio as httpcore_anyio
import pytest


class HttpcoreDeadlines:
    """Keep transport deadlines parked until the scripted operation stalls.

    Only httpcore's AnyIO backend gets a shadow module. Socket operations,
    TLS handshakes, cancellation and exception mapping still run normally.
    Callers bound the whole request with a separate test deadline.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pending: dict[anyio.CancelScope, float | None] = {}
        shadow = ModuleType("anyio")
        shadow.__dict__.update(vars(anyio), fail_after=self.fail_after)
        monkeypatch.setattr(httpcore_anyio, "anyio", shadow)

    @contextmanager
    def fail_after(self, delay: float | None, shield: bool = False) -> Iterator[anyio.CancelScope]:
        with anyio.fail_after(None, shield=shield) as scope:
            self.pending[scope] = delay
            try:
                yield scope
            finally:
                del self.pending[scope]

    def expire(self, expected_delay: float) -> None:
        """Expire the single in-flight network operation after its peer observes it."""
        assert len(self.pending) == 1, (
            f"Expected one pending httpcore timeout scope for delay {expected_delay!r}; pending={self.pending!r}"
        )
        scope, delay = next(iter(self.pending.items()))
        assert delay == expected_delay, (
            f"Expected httpcore timeout delay {expected_delay!r}, got {delay!r}; pending={self.pending!r}"
        )
        scope.deadline = float("-inf")


@contextmanager
def httpcore_deadlines() -> Iterator[HttpcoreDeadlines]:
    """Own the scoped backend shadow and report leaks without masking body exceptions."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        deadlines = HttpcoreDeadlines(monkeypatch)
        try:
            yield deadlines
        except BaseException as exc:
            if deadlines.pending:
                exc.add_note(f"Pending httpcore timeout scopes at exit: {deadlines.pending!r}")
            raise
        else:
            assert not deadlines.pending, f"Pending httpcore timeout scopes at exit: {deadlines.pending!r}"
