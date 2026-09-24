# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual ``events.Callback`` dispatch fast path."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import textual.message_pump as message_pump_mod
from textual import events, on
from textual._context import active_app, prevent_message_types_stack
from textual.message import Message
from textual.message_pump import MessagePump

from chrys.foundation.patches.textual_callback_dispatch import apply_runtime_patch


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[Any, ...]] = []

    def warning(self, *args: Any) -> None:
        self.warnings.append(args)


class _FakeApp:
    def __init__(self) -> None:
        self._closing = False
        self._is_devtools_connected = False
        self._logger = _FakeLogger()
        self._screen = object()
        self.features: set[str] = set()

    @property
    def screen(self) -> object:
        return self._screen


@pytest.fixture(autouse=True)
def _reset_runtime_patch() -> Iterator[_FakeApp]:
    original_dispatch = getattr(MessagePump._dispatch_message, "_original", MessagePump._dispatch_message)
    MessagePump._dispatch_message = original_dispatch
    app = _FakeApp()
    app_token = active_app.set(app)
    prevent_token = prevent_message_types_stack.set([set()])
    try:
        yield app
    finally:
        active_app.reset(app_token)
        prevent_message_types_stack.reset(prevent_token)
        MessagePump._dispatch_message = original_dispatch


async def test_callback_fast_path_invokes_callback_and_flushes_next_callbacks() -> None:
    apply_runtime_patch()
    pump = MessagePump()
    seen: list[str] = []

    def callback() -> None:
        seen.append("callback")
        pump.call_next(lambda: seen.append("next"))

    await pump._dispatch_message(events.Callback(callback))

    assert seen == ["callback", "next"]


async def test_callback_fast_path_preserves_prevent_context_and_message_hook() -> None:
    apply_runtime_patch()
    pump = MessagePump()
    seen: list[bool] = []
    hook_seen: list[Message] = []

    class Blocked(Message):
        pass

    def callback() -> None:
        seen.append(Blocked in pump._get_prevented_messages())

    message = events.Callback(callback)
    message._prevent.add(Blocked)
    hook_token = message_pump_mod.message_hook_context_var.set(hook_seen.append)
    try:
        await pump._dispatch_message(message)
    finally:
        message_pump_mod.message_hook_context_var.reset(hook_token)

    assert seen == [True]
    assert pump._get_prevented_messages() == set()
    assert hook_seen == [message]


async def test_callback_fast_path_honors_prevent_default_from_message_hook() -> None:
    apply_runtime_patch()
    pump = MessagePump()
    seen: list[str] = []
    pump.call_next(lambda: seen.append("next"))

    message = events.Callback(lambda: seen.append("callback"))

    def message_hook(hooked: Message) -> None:
        hooked.prevent_default()

    hook_token = message_pump_mod.message_hook_context_var.set(message_hook)
    try:
        await pump._dispatch_message(message)
    finally:
        message_pump_mod.message_hook_context_var.reset(hook_token)

    assert seen == ["next"]


async def test_callback_fast_path_uses_textual_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_runtime_patch()
    pump = MessagePump()
    seen: list[object] = []

    async def fake_invoke(callback: Any, *params: object) -> object:
        seen.append(("invoke", params))
        return callback()

    monkeypatch.setattr(message_pump_mod, "invoke", fake_invoke)

    await pump._dispatch_message(events.Callback(lambda: seen.append("callback")))

    assert seen == [("invoke", ()), "callback"]


async def test_callback_fast_path_skips_callback_when_app_is_closing(_reset_runtime_patch: _FakeApp) -> None:
    apply_runtime_patch()
    _reset_runtime_patch._closing = True
    pump = MessagePump()
    seen: list[str] = []
    pump._next_callbacks.append(events.Callback(lambda: seen.append("next")))

    await pump._dispatch_message(events.Callback(lambda: seen.append("callback")))

    assert seen == ["next"]


async def test_custom_callback_handler_uses_original_dispatch_path() -> None:
    apply_runtime_patch()
    seen: list[str] = []

    class CustomPump(MessagePump):
        async def _on_callback(self, event: events.Callback) -> None:
            seen.append("custom")
            event.prevent_default()

    pump = CustomPump()

    await pump._dispatch_message(events.Callback(lambda: seen.append("callback")))

    assert seen == ["custom"]


async def test_decorated_callback_handler_uses_original_dispatch_path() -> None:
    apply_runtime_patch()
    seen: list[str] = []

    class DecoratedPump(MessagePump):
        @on(events.Callback)
        async def handle_callback(self, event: events.Callback) -> None:
            seen.append("decorated")
            event.prevent_default()

    pump = DecoratedPump()

    await pump._dispatch_message(events.Callback(lambda: seen.append("callback")))

    assert seen == ["decorated"]


async def test_custom_on_event_uses_original_dispatch_path() -> None:
    apply_runtime_patch()
    seen: list[str] = []

    class CustomEventPump(MessagePump):
        async def on_event(self, event: events.Event) -> None:
            seen.append("event")
            await super().on_event(event)

    pump = CustomEventPump()

    await pump._dispatch_message(events.Callback(lambda: seen.append("callback")))

    assert seen == ["event", "callback"]


def test_callback_fast_path_runtime_patch_is_idempotent() -> None:
    apply_runtime_patch()
    first = MessagePump._dispatch_message
    apply_runtime_patch()
    second = MessagePump._dispatch_message

    assert first is second
