# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: cache ``MessagePump._get_dispatch_methods`` MRO walks.

Problem
-------
Textual's ``MessagePump._get_dispatch_methods`` walks ``self.__class__.__mro__``
on every dispatched message, performs ``cls.__dict__.get`` lookups for both
``_decorated_handlers`` and convention-named handlers, and rebuilds a
``methods_dispatched`` set each call.  During a chat-panel scroll on a widget
tree with many subscribers, ~190 ``events.Callback`` messages are dispatched
per frame, and ``_get_dispatch_methods`` accounts for ~2-3 ms/frame of pure
re-walking of static class data.

Solution
--------
Per ``(self_class, handler_name, message_class)`` triple, pre-compute the
ordered sequence of ``(cls, unbound_method)`` groups that the original walk
would yield, plus a flag noting whether any decorated handler with selectors
is involved.

On dispatch:

* If the plan has no selector handlers, take a fast path that iterates the
  cached groups and yields ``(cls, bound_method)`` pairs directly — no MRO
  walk, no ``__dict__.get`` lookups, no dedup-set rebuild.  ``_no_default_action``
  is still honored at class boundaries to match original semantics.
* If the plan has any selector handlers (decorated with CSS-style selectors),
  fall through to the original implementation.  Selector matching depends on
  runtime widget tree state and isn't safely cacheable.

The plan dict is keyed by ``(class, str, class)`` — class objects are immutable
as far as ``_decorated_handlers`` is concerned (populated at class definition
by the ``_MessagePumpMeta`` metaclass) and live for the process lifetime,
so a plain dict is a safe long-lived cache.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_dispatch_cache_patched"


@dataclass(frozen=True, slots=True)
class _DispatchPlan:
    """Pre-computed dispatch plan for one ``(self_cls, handler_name, message_cls)`` triple.

    ``steps`` is the ordered list of MRO classes that contribute handlers,
    each paired with its tuple of unbound methods to yield in order.  Empty
    classes are omitted so the fast path doesn't pay for them.

    ``has_selectors`` is True iff any decorated handler in the plan declared
    a non-empty selectors dict.  Selector matching depends on the live widget
    tree, so when this flag is set the patched ``_get_dispatch_methods`` falls
    through to the original implementation instead of using the plan.
    """

    steps: tuple[tuple[type, tuple[Callable[..., Any], ...]], ...]
    has_selectors: bool


def _build_dispatch_plan(self_cls: type, method_name: str, message_cls: type) -> _DispatchPlan:
    """Replicate the static portion of the original MRO walk and capture the result.

    Mirrors ``MessagePump._get_dispatch_methods`` from Textual 8.2.7 — see
    ``message_pump.py`` for the reference behavior.
    """
    from textual.message import Message

    message_mro = [t for t in message_cls.__mro__ if issubclass(t, Message)]

    methods_dispatched: set[Callable[..., Any]] = set()
    has_selectors = False
    steps: list[tuple[type, tuple[Callable[..., Any], ...]]] = []

    for cls in self_cls.__mro__:
        step_methods: list[Callable[..., Any]] = []

        decorated_handlers = cls.__dict__.get("_decorated_handlers")
        if decorated_handlers:
            for msg_class in message_mro:
                for method, selectors in decorated_handlers.get(msg_class, []):
                    if method in methods_dispatched:
                        continue
                    if selectors:
                        # Selector handlers depend on live ``message._sender`` /
                        # widget tree state and can't be safely pre-resolved.
                        has_selectors = True
                    else:
                        step_methods.append(method)
                        methods_dispatched.add(method)

        method = cls.__dict__.get(f"_{method_name}") or cls.__dict__.get(method_name)
        if method is not None and not getattr(method, "_textual_on", None):
            step_methods.append(method)

        if step_methods:
            steps.append((cls, tuple(step_methods)))

    return _DispatchPlan(steps=tuple(steps), has_selectors=has_selectors)


def apply_runtime_patch() -> None:
    """Install the cached ``_get_dispatch_methods`` on ``MessagePump`` in this process."""
    try:
        from textual.message_pump import MessagePump
    except ImportError:
        return

    existing = MessagePump._get_dispatch_methods
    if getattr(existing, _RUNTIME_PATCH_MARKER, False):
        return

    original_get_dispatch_methods = existing
    plan_cache: dict[tuple[type, str, type], _DispatchPlan] = {}

    def _get_dispatch_methods(self: Any, method_name: str, message: Any) -> Any:
        self_cls = type(self)
        message_cls = type(message)
        key = (self_cls, method_name, message_cls)

        plan = plan_cache.get(key)
        if plan is None:
            plan = _build_dispatch_plan(self_cls, method_name, message_cls)
            plan_cache[key] = plan

        if plan.has_selectors:
            yield from original_get_dispatch_methods(self, method_name, message)
            return

        for step_cls, unbound_methods in plan.steps:
            if message._no_default_action:
                return
            for unbound in unbound_methods:
                yield step_cls, unbound.__get__(self, step_cls)

    setattr(_get_dispatch_methods, _RUNTIME_PATCH_MARKER, True)
    # Expose the cache and builder for tests / introspection without leaking
    # them into module-level state.  These attributes are part of the patch's
    # observable surface — tests can clear ``_plan_cache`` between runs to
    # exercise both cold and warm paths.
    _get_dispatch_methods._plan_cache = plan_cache
    _get_dispatch_methods._build_plan = _build_dispatch_plan
    _get_dispatch_methods._original = original_get_dispatch_methods
    MessagePump._get_dispatch_methods = _get_dispatch_methods
