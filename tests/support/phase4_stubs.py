# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared Phase 4 test stubs: fake ``LastWordsGenerator`` + reminder middleware.

The real classes reach into the LLM client and Chrys middleware machinery;
tests that exercise compaction / Phase 4 wiring need lightweight stand-ins
that just record what they were asked to do.
"""

from __future__ import annotations

from chrys.service.agent_middleware.system_reminder import (
    DropRoundBreakerState,
    ManifestEntry,
    Phase4RetrySnapshot,
)


class StubLastWordsGenerator:
    """Test stub for ``LastWordsGenerator`` — records calls, returns fixed text."""

    def __init__(self, text: str = "[STUB LAST_WORDS progress note]") -> None:
        self.text = text
        self.calls: list[dict] = []
        self.breaker_trips: list[str] = []
        self.committed_publishes = 0

    async def publish_breaker_trip(self, failure_reason: str) -> None:
        self.breaker_trips.append(failure_reason)

    async def publish_committed(self) -> None:
        self.committed_publishes += 1

    async def generate(
        self,
        scoped_groups: list,
        previous_last_words: str | None,
        *,
        degraded_opener: bool,
        has_continuation_nudges: bool,
        completer: object | None = None,
        tokenizer: object | None = None,
        system_overhead_tokens: int = 0,
        tool_definition_tokens: int = 0,
        request_overhead_tokens: int = 0,
        calibration_ratio: float = 1.0,
        spend_side_call_tokens: object | None = None,
    ) -> str:
        self.calls.append(
            {
                "scoped_groups": list(scoped_groups),
                "previous_last_words": previous_last_words,
                "has_continuation_nudges": has_continuation_nudges,
                "degraded_opener": degraded_opener,
                "completer": completer,
                "tokenizer": tokenizer,
                "system_overhead_tokens": system_overhead_tokens,
                "tool_definition_tokens": tool_definition_tokens,
                "request_overhead_tokens": request_overhead_tokens,
                "calibration_ratio": calibration_ratio,
                "spend_side_call_tokens": spend_side_call_tokens,
            }
        )
        return self.text


class StubReminderMiddleware:
    """Test stub for ``SystemReminderMiddleware.set/get_last_words``."""

    def __init__(self) -> None:
        self._last_words: str | None = None
        self._last_words_todo: str | None = None
        self._manifest: list[dict] = []
        self._breaker = DropRoundBreakerState()
        self._context_pressure_notified = False
        self.refresh_calls: list[list] = []

    def get_last_words(self) -> str | None:
        return self._last_words

    def set_last_words(self, text: str | None) -> None:
        self._last_words = text or None

    def append_manifest(self, entries: list[ManifestEntry]) -> None:
        self._manifest.extend(entry.to_state() for entry in entries)

    def mark_manifest_records_unavailable(self, relative_paths: frozenset[str] | set[str]) -> None:
        for entry in self._manifest:
            if entry.get("relative_path") in relative_paths:
                entry["available"] = False

    def get_last_words_manifest(self) -> list[dict]:
        return list(self._manifest)

    def snapshot_phase4_retry_state(self) -> Phase4RetrySnapshot:
        """Mirror the real middleware's attempt-local content snapshot."""
        manifest = tuple(
            entry.to_state() for value in self._manifest if (entry := ManifestEntry.from_state(value)) is not None
        )
        return Phase4RetrySnapshot(
            last_words=self._last_words,
            last_words_todo=self._last_words_todo,
            last_words_manifest=manifest,
        )

    def restore_phase4_retry_state(self, snapshot: Phase4RetrySnapshot) -> None:
        """Restore note content while deliberately retaining breaker state."""
        self._last_words = snapshot.last_words
        self._last_words_todo = snapshot.last_words_todo
        self._manifest = [
            entry.to_state()
            for value in snapshot.last_words_manifest
            if (entry := ManifestEntry.from_state(value)) is not None
        ]

    def get_drop_round_breaker(self) -> DropRoundBreakerState:
        return self._breaker

    def set_drop_round_breaker(self, breaker: DropRoundBreakerState) -> None:
        self._breaker = breaker

    def get_last_words_breaker_state(self) -> dict | None:
        return self._breaker.to_state() if self._breaker != DropRoundBreakerState() else None

    def claim_context_pressure_notification(self) -> bool:
        if self._context_pressure_notified:
            return False
        self._context_pressure_notified = True
        return True

    def release_context_pressure_notification(self) -> None:
        self._context_pressure_notified = False

    def refresh_last_words_reminder(self, messages: list) -> int | None:
        """Record the refresh request; never rewrites the stub's message list."""
        self.refresh_calls.append(messages)
        return None

    def render_last_words_reminder_text(self) -> str | None:
        """Mirror the real renderer's contract: wrapped text when state exists, else None."""
        from chrys.service.agent_middleware.system_reminder import _wrap

        if not self._last_words and not self._manifest:
            return None
        lines = [f"[LAST_WORDS] {self._last_words or ''}".rstrip()]
        lines.extend(f"manifest: {entry.get('relative_path', '')}" for entry in self._manifest)
        return _wrap("\n".join(lines))
