# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: fast-path default Textual ``events.Callback`` dispatch.

Textual schedules many internal callbacks while scrolling.  The default
``events.Callback`` handler only guards app shutdown / missing-screen cases and
then invokes the stored callable, but Textual still routes it through the full
event dispatch pipeline: handler-name lookup, MRO dispatch lookup, event
logging, and generic ``on_event`` indirection.

This runtime patch specializes that one internal event when the receiving pump
uses Textual's default event and callback handlers. Pumps with custom
``on_event``, ``_on_callback``, or ``on_callback`` handlers stay on the
original dispatch path.
"""

from __future__ import annotations

from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_callback_dispatch_fast_path"


def apply_runtime_patch() -> None:
    """Install a default ``events.Callback`` fast path on ``MessagePump``."""
    try:
        import textual.message_pump as message_pump_mod
        from textual import events
        from textual.message import Message
        from textual.message_pump import MessagePump
    except ImportError:
        return

    existing = MessagePump._dispatch_message
    if getattr(existing, _RUNTIME_PATCH_MARKER, False):
        return

    original_dispatch_message = existing
    default_callback_handler_cache: dict[type, bool] = {}
    callback_message_mro = tuple(t for t in events.Callback.__mro__ if issubclass(t, Message))

    def uses_default_callback_handler(pump: Any) -> bool:
        """Return true when ``pump`` would use Textual's default callback path."""
        pump_cls = type(pump)
        cached = default_callback_handler_cache.get(pump_cls)
        if cached is not None:
            return cached

        for cls in pump_cls.__mro__:
            method = cls.__dict__.get("on_event")
            if method is not None:
                if cls is not MessagePump and method is not MessagePump.on_event:
                    default_callback_handler_cache[pump_cls] = False
                    return False
                break

        for cls in pump_cls.__mro__:
            decorated_handlers = cls.__dict__.get("_decorated_handlers")
            if decorated_handlers:
                for message_cls in callback_message_mro:
                    if decorated_handlers.get(message_cls):
                        default_callback_handler_cache[pump_cls] = False
                        return False

            method = cls.__dict__.get("_on_callback") or cls.__dict__.get("on_callback")
            if method is None:
                continue
            cached = cls is MessagePump and method is MessagePump.on_callback
            default_callback_handler_cache[pump_cls] = cached
            return cached

        default_callback_handler_cache[pump_cls] = False
        return False

    async def _dispatch_message(self: Any, message: Any) -> None:
        _rich_traceback_guard = True
        if not isinstance(message, events.Callback) or not uses_default_callback_handler(self):
            await original_dispatch_message(self, message)
            return

        if message.no_dispatch:
            return

        try:
            message_hook = message_pump_mod.message_hook_context_var.get()
        except LookupError:
            pass
        else:
            message_hook(message)

        with self.prevent(*message._prevent):
            app = self.app
            if not message._no_default_action and not app._closing:
                try:
                    _ = app.screen
                except Exception:
                    self.log.warning(f"Not invoking timer callback {message.callback!r} because there is no screen.")
                else:
                    await message_pump_mod.invoke(message.callback)
            if self._next_callbacks:
                await self._flush_next_callbacks()

    setattr(_dispatch_message, _RUNTIME_PATCH_MARKER, True)
    _dispatch_message._original = original_dispatch_message
    _dispatch_message._default_callback_handler_cache = default_callback_handler_cache
    _dispatch_message._uses_default_callback_handler = uses_default_callback_handler
    MessagePump._dispatch_message = _dispatch_message
