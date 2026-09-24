# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the cached ``MessagePump._get_dispatch_methods`` patch.

These tests pin behavioral equivalence with the original Textual
implementation while exercising the static plan cache.  The most important
regression test is ``test_patched_output_matches_original_for_*`` — if any
edge case diverges from the upstream walk, those tests fail.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.message import Message
from textual.message_pump import MessagePump
from textual.widgets import Static

from chrys.foundation.patches import textual_dispatch_cache
from chrys.foundation.patches.textual_dispatch_cache import (
    _build_dispatch_plan,
    apply_runtime_patch,
)


@pytest.fixture(autouse=True)
def _patch_and_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Exercise a fresh installation and restore the incoming dispatcher afterward."""
    incoming = MessagePump._get_dispatch_methods
    original = getattr(incoming, "_original", incoming)
    # Share the test's patch stack so any later overrides unwind first.
    monkeypatch.setattr(MessagePump, "_get_dispatch_methods", original)
    apply_runtime_patch()
    installed = MessagePump._get_dispatch_methods
    try:
        yield
    finally:
        installed._plan_cache.clear()  # type: ignore[attr-defined]


def _original_dispatch(instance: MessagePump, handler_name: str, message: Message) -> list:
    """Call the saved pre-patch ``_get_dispatch_methods`` directly."""
    original = MessagePump._get_dispatch_methods._original  # type: ignore[attr-defined]
    return list(original(instance, handler_name, message))


# ---------------------------------------------------------------------------
# Test message classes.  Auto-derived ``handler_name`` follows ``on_<snake>``.
# ---------------------------------------------------------------------------


class _Ping(Message):
    """Convention handler target: ``on_ping``."""


class _ControlMsg(Message):
    """Selector-target message: exposes ``control`` so ``@on(_, "#x")`` works."""

    @property
    def control(self):  # type: ignore[override]
        return self._sender


# ---------------------------------------------------------------------------
# Convention-handler MRO order.
# ---------------------------------------------------------------------------


class _Base(MessagePump):
    def _on_ping(self, message: _Ping) -> None: ...  # pragma: no cover - signature only


class _Sub(_Base):
    def _on_ping(self, message: _Ping) -> None: ...  # pragma: no cover - signature only


def test_convention_handlers_yield_in_mro_order() -> None:
    instance = _Sub()
    yields = list(instance._get_dispatch_methods("on_ping", _Ping()))
    classes = [cls for cls, _method in yields]
    assert classes == [_Sub, _Base]
    # Each yielded method is bound to ``instance`` and resolves through
    # the cls that defined it (not via inherited lookup).
    for cls, method in yields:
        assert method.__self__ is instance
        assert method.__func__ is cls.__dict__["_on_ping"]


def test_underscored_handler_takes_precedence_over_public() -> None:
    class _PreferredUnderscore(MessagePump):
        def _on_ping(self, message: _Ping) -> None: ...

        def on_ping(self, message: _Ping) -> None: ...

    instance = _PreferredUnderscore()
    yields = list(instance._get_dispatch_methods("on_ping", _Ping()))
    assert len(yields) == 1, yields
    _cls, method = yields[0]
    assert method.__func__ is _PreferredUnderscore.__dict__["_on_ping"]


# ---------------------------------------------------------------------------
# Decorated handlers, no selectors.
# ---------------------------------------------------------------------------


class _DecoratedNoSelector(MessagePump):
    @on(_Ping)
    def handle_ping(self, message: _Ping) -> None: ...

    # Convention handler also defined to verify both fire and order.
    def _on_ping(self, message: _Ping) -> None: ...


def test_decorated_handler_without_selectors_yields() -> None:
    instance = _DecoratedNoSelector()
    yields = list(instance._get_dispatch_methods("on_ping", _Ping()))
    funcs = [m.__func__ for _cls, m in yields]
    # Decorated handler fires first (decorated handlers come before convention),
    # then the convention handler.
    assert _DecoratedNoSelector.__dict__["handle_ping"] in funcs
    assert _DecoratedNoSelector.__dict__["_on_ping"] in funcs
    assert funcs.index(_DecoratedNoSelector.__dict__["handle_ping"]) < funcs.index(
        _DecoratedNoSelector.__dict__["_on_ping"]
    )


# ---------------------------------------------------------------------------
# Decorated handlers WITH selectors → plan flags ``has_selectors`` and
# patched ``_get_dispatch_methods`` delegates to the original.
# ---------------------------------------------------------------------------


class _DecoratedWithSelector(MessagePump):
    @on(_ControlMsg, "#some-id")
    def handle_selector(self, message: _ControlMsg) -> None: ...

    def _on_control_msg(self, message: _ControlMsg) -> None: ...


@pytest.mark.parametrize("sender_id", ["some-id", "other-id", None])
def test_selector_handler_marks_plan_and_delegates_to_original(sender_id: str | None) -> None:
    instance = _DecoratedWithSelector()
    plan = _build_dispatch_plan(_DecoratedWithSelector, "on_control_msg", _ControlMsg)
    assert plan.has_selectors is True

    # Patched output must equal the original's output exactly.
    msg = _ControlMsg()
    if sender_id is not None:
        msg.set_sender(Static(id=sender_id))
    patched = list(instance._get_dispatch_methods("on_control_msg", msg))
    original = _original_dispatch(instance, "on_control_msg", msg)
    assert patched == original
    functions = [method.__func__ for _cls, method in patched]
    assert (_DecoratedWithSelector.handle_selector in functions) is (sender_id == "some-id")


async def test_selector_matches_live_widget_state_with_a_warm_plan() -> None:
    calls: list[str] = []

    class ControlMsg(_ControlMsg):
        pass

    class _SelectorApp(App[None]):
        def compose(self) -> ComposeResult:
            yield Static(id="sender")

        @on(ControlMsg, ".selected")
        def handle_selected(self, _message: ControlMsg) -> None:
            calls.append("selected")

        def on_control_msg(self, _message: ControlMsg) -> None:
            calls.append("default")

    app = _SelectorApp()
    async with app.run_test():
        sender = app.query_one("#sender", Static)
        # The same sender and dispatch key alternate between nonmatching and
        # matching states. Only the handler plan, not the selector result, is static.
        for selected in (False, True, False):
            sender.set_class(selected, "selected")
            calls.clear()
            await app._on_message(ControlMsg().set_sender(sender))
            assert calls == (["selected", "default"] if selected else ["default"])


# ---------------------------------------------------------------------------
# ``_no_default_action`` breaks between MRO class boundaries — not within.
# ---------------------------------------------------------------------------


class _ParentBreak(MessagePump):
    def _on_ping(self, message: _Ping) -> None: ...


class _ChildBreak(_ParentBreak):
    def _on_ping(self, message: _Ping) -> None: ...


def test_no_default_action_breaks_between_class_boundaries() -> None:
    instance = _ChildBreak()
    msg = _Ping()
    gen = instance._get_dispatch_methods("on_ping", msg)

    first_cls, _first_method = next(gen)
    assert first_cls is _ChildBreak

    msg._no_default_action = True
    # The next class (_ParentBreak) must be skipped.
    assert list(gen) == []


# ---------------------------------------------------------------------------
# Dedup: a decorated handler reachable via both decorator and convention naming
# is yielded once.  Textual filters it out of the convention path via
# ``_textual_on``.
# ---------------------------------------------------------------------------


class _DecoratedAndNamed(MessagePump):
    @on(_Ping)
    def _on_ping(self, message: _Ping) -> None: ...


def test_decorated_method_with_convention_name_yields_once() -> None:
    instance = _DecoratedAndNamed()
    yields = list(instance._get_dispatch_methods("on_ping", _Ping()))
    assert len(yields) == 1, yields
    _cls, method = yields[0]
    assert method.__func__ is _DecoratedAndNamed.__dict__["_on_ping"]


# ---------------------------------------------------------------------------
# ``on_idle`` and other non-message handler names still work.
# ---------------------------------------------------------------------------


class _IdleHandler(MessagePump):
    def _on_idle(self, message: Message) -> None: ...


def test_on_idle_handler_name_works() -> None:
    instance = _IdleHandler()
    # ``on_idle`` is called explicitly with an ``events.Idle`` instance, but
    # the dispatch walk is name-driven, so any Message subclass exercises
    # the same code path.
    from textual import events

    yields = list(instance._get_dispatch_methods("on_idle", events.Idle()))
    assert len(yields) == 1
    _cls, method = yields[0]
    assert method.__func__ is _IdleHandler.__dict__["_on_idle"]


# ---------------------------------------------------------------------------
# Cache hit returns the same yield sequence as cold path.
# ---------------------------------------------------------------------------


def test_cache_hit_yields_match_cache_miss() -> None:
    instance = _Sub()

    with patch.object(
        textual_dispatch_cache, "_build_dispatch_plan", autospec=True, side_effect=_build_dispatch_plan
    ) as build:
        cold = list(instance._get_dispatch_methods("on_ping", _Ping()))
        warm = list(instance._get_dispatch_methods("on_ping", _Ping()))
    build.assert_called_once_with(_Sub, "on_ping", _Ping)

    cold_keys = [(cls, m.__func__) for cls, m in cold]
    warm_keys = [(cls, m.__func__) for cls, m in warm]
    assert cold_keys == warm_keys


def test_cache_survives_across_instances() -> None:
    a = _Sub()
    b = _Sub()

    with patch.object(
        textual_dispatch_cache, "_build_dispatch_plan", autospec=True, side_effect=_build_dispatch_plan
    ) as build:
        list(a._get_dispatch_methods("on_ping", _Ping()))

        cache = MessagePump._get_dispatch_methods._plan_cache  # type: ignore[attr-defined]
        assert (_Sub, "on_ping", _Ping) in cache

        # Second instance should reuse the same plan and yield correctly.
        yields = list(b._get_dispatch_methods("on_ping", _Ping()))
    build.assert_called_once_with(_Sub, "on_ping", _Ping)
    assert [(cls, m.__func__) for cls, m in yields] == [
        (_Sub, _Sub.__dict__["_on_ping"]),
        (_Base, _Base.__dict__["_on_ping"]),
    ]
    # And each yielded method is bound to the second instance, not the first.
    for _cls, method in yields:
        assert method.__self__ is b


# ---------------------------------------------------------------------------
# Patch is idempotent.
# ---------------------------------------------------------------------------


def test_apply_runtime_patch_is_idempotent() -> None:
    first = MessagePump._get_dispatch_methods
    apply_runtime_patch()
    second = MessagePump._get_dispatch_methods
    assert first is second


# ---------------------------------------------------------------------------
# Strongest regression test: for a family of representative class+message
# combinations, the patched generator yields exactly what the original would.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("self_cls", "handler_name", "msg_cls"),
    [
        (_Sub, "on_ping", _Ping),
        (_Base, "on_ping", _Ping),
        (_DecoratedNoSelector, "on_ping", _Ping),
        (_DecoratedAndNamed, "on_ping", _Ping),
        (_DecoratedWithSelector, "on_control_msg", _ControlMsg),
        (_IdleHandler, "on_idle", Message),
    ],
)
def test_patched_output_matches_original_for(self_cls, handler_name, msg_cls) -> None:
    instance = self_cls()
    patched = list(instance._get_dispatch_methods(handler_name, msg_cls()))
    original = _original_dispatch(instance, handler_name, msg_cls())
    # Compare structurally: same classes in same order, same underlying funcs.
    patched_keys = [(cls, m.__func__) for cls, m in patched]
    original_keys = [(cls, m.__func__) for cls, m in original]
    assert patched_keys == original_keys


def test_plan_steps_omit_empty_classes() -> None:
    """``MessagePump`` and ``object`` define no ``_on_ping`` — they should not appear."""
    plan = _build_dispatch_plan(_Sub, "on_ping", _Ping)
    classes_in_plan = [step[0] for step in plan.steps]
    assert _Sub in classes_in_plan
    assert _Base in classes_in_plan
    assert MessagePump not in classes_in_plan
    assert object not in classes_in_plan
