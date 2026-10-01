# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: Textual widgets keep their unchanged defaults on the class.

Problem
-------
Every mounted widget is a message pump, a DOM node and a widget, and each ``__init__`` stores
about a hundred attributes, most of them constants every instance starts with (``False``,
``None``, ``(-1, False)``, a fresh ``VerticalLayout``). A large transcript mounts thousands of
widgets, so these fill a 3.6 KB instance dict per widget (CPython doubles the dict at 86 keys).
Each pump also builds containers most never use: a queue made of a ``deque`` and an
``asyncio.Event`` (with its own ``deque``), a mounted ``asyncio.Event``, a ``WeakSet`` of timers,
a ``WeakKeyDictionary`` of signal subscribers, a set of disabled messages, a set of CSS type
names and an ``RLock``. They cost memory, and each is a container the cyclic collector traverses
on every full collection.

Solution
--------
Constant defaults move to class attributes, so an instance stores only what it changes; ``_id``
stays in ``__init__``, since a missing instance ``_id`` is how Textual detects a subclass that
skipped ``super().__init__()``. The pump's queue is a list read from a head index, woken by one
future, and the mounted event a ``Flag`` whose waiter list is made on the first wait. The timer
set, signal subscribers and widget lock are made on first use, disabled messages are replaced
rather than mutated, and the CSS type names are one frozenset per class.

Each file is patched on its own, so no file's patches may need another's: ``Flag`` is defined in
``message_pump.py``, the one module that uses it, and a ``_queue.py`` its patch no longer matches
leaves the upstream queue under a patched pump. A ``_queue.py`` patched while ``Flag`` still lived
there is kept as it is, since its pump imports ``Flag`` from it until the pump's upgrade patches
apply.

The runtime install must precede the first message pump, as bootstrap does: a queue built by
the upstream ``__init__`` keeps a ``deque`` the new methods cannot read.
"""

from __future__ import annotations

import importlib
import logging

from chrys.foundation.patches.patcher import FilePatch, register
from chrys.foundation.patches.staged_members import (
    StagedMembers,
    StagedSourceDriftError,
    members_installed,
    stage_members,
)

_RUNTIME_PATCH_MARKER = "_chrys_node_diet"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
_RUNTIME_MEMBERS: dict[str, dict[str, tuple[str, ...]]] = {
    "textual._queue": {"Queue": ("__init__", "put_nowait", "qsize", "empty", "get", "get_nowait", "_pop")},
    "textual.message_pump": {
        "MessagePump": (
            "_running",
            "_closing",
            "_closed",
            "_disabled_messages",
            "_pending_message",
            "_task",
            "_timers",
            "_max_idle",
            "_is_mounted",
            "__init__",
            "_mounted_event",
            "disable_messages",
            "enable_messages",
            "set_timer",
            "set_interval",
            "_add_timer",
        ),
    },
    "textual.dom": {
        "DOMNode": (
            "_auto_refresh",
            "_auto_refresh_timer",
            "_has_hover_style",
            "_has_focus_within",
            "_has_order_style",
            "_has_odd_or_even",
            "_reactive_connect",
            "_pruning",
            "_trap_focus",
            "__init__",
        ),
    },
    "textual.widget": {
        "Widget": (
            "_layout_required",
            "_layout_updates",
            "_repaint_required",
            "_scroll_required",
            "_recompose_required",
            "_refresh_styles_required",
            "_default_layout",
            "_animate",
            "highlight_style",
            "_vertical_scrollbar",
            "_horizontal_scrollbar",
            "_scrollbar_corner",
            "_border_title",
            "_border_subtitle",
            "_visual_style",
            "_visual_style_cache_key",
            "_render_cache",
            "_content_width_cache",
            "_content_height_cache",
            "_tooltip",
            "_cover_widget",
            "_first_of_type",
            "_last_of_type",
            "_first_child",
            "_last_child",
            "_odd",
            "_extrema",
            "__init__",
            "lock",
        ),
    },
    "textual.signal": {"Signal": ("__init__", "__rich_repr__", "subscribe", "unsubscribe", "publish")},
}
_RUNTIME_CLASSES: dict[str, tuple[str, ...]] = {"textual.message_pump": ("Flag",)}
logger = logging.getLogger(__name__)

# _queue.py
_QUEUE_OLD = """\
import asyncio
from asyncio import Event
from collections import deque
from typing import Generic, TypeVar

QueueType = TypeVar("QueueType")


class Queue(Generic[QueueType]):
    \"\"\"A cut-down version of asyncio.Queue

    This has just enough functionality to run the message pumps.

    \"\"\"

    def __init__(self) -> None:
        self.values: deque[QueueType] = deque()
        self.ready_event = Event()

    def put_nowait(self, value: QueueType) -> None:
        self.values.append(value)
        self.ready_event.set()

    def qsize(self) -> int:
        return len(self.values)

    def empty(self) -> bool:
        return not self.values

    def task_done(self) -> None:
        pass

    async def get(self) -> QueueType:
        if not self.ready_event.is_set():
            await self.ready_event.wait()
        value = self.values.popleft()
        if not self.values:
            self.ready_event.clear()
        return value

    def get_nowait(self) -> QueueType:
        if not self.values:
            raise asyncio.QueueEmpty()
        value = self.values.popleft()
        if not self.values:
            self.ready_event.clear()
        return value
"""

_QUEUE_HEADER_NEW = """\
import asyncio
from typing import Generic, TypeVar

QueueType = TypeVar("QueueType")


"""

# message_pump.py defines it for its mounted event.
_FLAG = """\
class Flag:
    \"\"\"A cut-down version of asyncio.Event.

    Every message pump holds one, so it keeps no waiter container until something waits.
    \"\"\"

    def __init__(self) -> None:
        self._value = False
        self._waiters: list[asyncio.Future[bool]] | None = None

    def is_set(self) -> bool:
        return self._value

    def set(self) -> None:
        if not self._value:
            self._value = True
            waiters = self._waiters
            if waiters:
                self._waiters = None
                for waiter in waiters:
                    if not waiter.done():
                        waiter.set_result(True)

    async def wait(self) -> bool:
        if self._value:
            return True
        waiter = asyncio.get_running_loop().create_future()
        if self._waiters is None:
            self._waiters = []
        self._waiters.append(waiter)
        try:
            await waiter
            return True
        finally:
            waiters = self._waiters
            if waiters is not None and waiter in waiters:
                waiters.remove(waiter)
"""

_QUEUE_CLASS_NEW = """\
class Queue(Generic[QueueType]):
    \"\"\"A cut-down version of asyncio.Queue

    This has just enough functionality to run the message pumps: its one consumer waits on a
    future, and values are a list read from a head index, since a deque or an event allocates
    a block that most pumps never fill.

    \"\"\"

    def __init__(self) -> None:
        self._values: list[QueueType | None] = []
        self._head = 0
        self._waiter: asyncio.Future[None] | None = None

    def put_nowait(self, value: QueueType) -> None:
        self._values.append(value)
        waiter = self._waiter
        if waiter is not None:
            self._waiter = None
            if not waiter.done():
                waiter.set_result(None)

    def qsize(self) -> int:
        return len(self._values) - self._head

    def empty(self) -> bool:
        return self._head >= len(self._values)

    def task_done(self) -> None:
        pass

    async def get(self) -> QueueType:
        while self._head >= len(self._values):
            waiter = self._waiter = asyncio.get_running_loop().create_future()
            try:
                await waiter
            finally:
                if self._waiter is waiter:
                    self._waiter = None
        return self._pop()

    def get_nowait(self) -> QueueType:
        if self._head >= len(self._values):
            raise asyncio.QueueEmpty()
        return self._pop()

    def _pop(self) -> QueueType:
        values = self._values
        head = self._head
        value = values[head]
        head += 1
        if head == len(values):
            values.clear()
            head = 0
        elif head >= 64 and head * 2 >= len(values):
            # Drop the consumed prefix once it is at least half the list.
            del values[:head]
            head = 0
        else:
            # Release the consumed message now rather than at the next compaction.
            values[head - 1] = None
        self._head = head
        return value  # type: ignore[return-value]
"""

_QUEUE_NEW = _QUEUE_HEADER_NEW + _QUEUE_CLASS_NEW

# An earlier version of this patch defined ``Flag`` in ``_queue.py``, where ``message_pump.py``
# imported it. That queue is kept, because a pump whose upgrade patches no longer match imports it.
_QUEUE_WITH_FLAG_NEW = _QUEUE_HEADER_NEW + _FLAG + "\n\n" + _QUEUE_CLASS_NEW

# message_pump.py
_PUMP_IMPORT_WITH_FLAG_OLD = """\
from textual._queue import Flag, Queue
"""

_PUMP_IMPORT_WITH_FLAG_NEW = """\
from textual._queue import Queue
"""

_PUMP_FLAG_OLD = """\
class MessagePumpClosed(Exception):
    pass
"""

_PUMP_FLAG_NEW = _PUMP_FLAG_OLD + "\n\n" + _FLAG

_PUMP_INIT_OLD = """\
class MessagePump(metaclass=_MessagePumpMeta):
    \"\"\"Base class which supplies a message pump.\"\"\"

    def __init__(self, parent: MessagePump | None = None) -> None:
        self._parent = parent
        self._running: bool = False
        self._closing: bool = False
        self._closed: bool = False
        self._disabled_messages: set[type[Message]] = set()
        self._pending_message: Message | None = None
        self._task: Task | None = None
        self._timers: WeakSet[Timer] = WeakSet()
        self._last_idle: float = time()
        self._max_idle: float | None = None
        self._is_mounted = False
        \"\"\"Having this explicit Boolean is an optimization.

        The same information could be retrieved from `self._mounted_event.is_set()`, but
        we need to access this frequently in the compositor and the attribute with the
        explicit Boolean value is faster than the two lookups and the function call.
        \"\"\"
        self._next_callbacks: list[events.Callback] = []
"""

_PUMP_INIT_NEW = """\
class MessagePump(metaclass=_MessagePumpMeta):
    \"\"\"Base class which supplies a message pump.\"\"\"

    # Defaults every pump starts with live on the class, so an instance only stores what it
    # changes: a large transcript mounts thousands of pumps.
    _running: bool = False
    _closing: bool = False
    _closed: bool = False
    _disabled_messages: frozenset[type[Message]] | set[type[Message]] = frozenset()
    _pending_message: Message | None = None
    _task: Task | None = None
    _timers: WeakSet[Timer] | frozenset[Timer] = frozenset()
    \"\"\"Replaced by a `WeakSet` when the first timer starts.\"\"\"
    _max_idle: float | None = None
    _is_mounted = False
    \"\"\"Having this explicit Boolean is an optimization.

    The same information could be retrieved from `self._mounted_event.is_set()`, but
    we need to access this frequently in the compositor and the attribute with the
    explicit Boolean value is faster than the two lookups and the function call.
    \"\"\"

    def __init__(self, parent: MessagePump | None = None) -> None:
        self._parent = parent
        self._last_idle: float = time()
        self._next_callbacks: list[events.Callback] = []
"""

_PUMP_MOUNTED_OLD = """\
    @cached_property
    def _mounted_event(self) -> asyncio.Event:
        return asyncio.Event()
"""

_PUMP_MOUNTED_NEW = """\
    @cached_property
    def _mounted_event(self) -> Flag:
        return Flag()
"""

_PUMP_DISABLED_OLD = """\
    def disable_messages(self, *messages: type[Message]) -> None:
        \"\"\"Disable message types from being processed.\"\"\"
        self._disabled_messages.update(messages)

    def enable_messages(self, *messages: type[Message]) -> None:
        \"\"\"Enable processing of messages types.\"\"\"
        self._disabled_messages.difference_update(messages)
"""

_PUMP_DISABLED_NEW = """\
    def disable_messages(self, *messages: type[Message]) -> None:
        \"\"\"Disable message types from being processed.\"\"\"
        self._disabled_messages = {*self._disabled_messages, *messages}

    def enable_messages(self, *messages: type[Message]) -> None:
        \"\"\"Enable processing of messages types.\"\"\"
        self._disabled_messages = self._disabled_messages.difference(messages)
"""

_PUMP_TIMER_OLD = """\
            repeat=0,
            pause=pause,
        )
        timer._start()
        self._timers.add(timer)
        return timer
"""

_PUMP_TIMER_NEW = """\
            repeat=0,
            pause=pause,
        )
        timer._start()
        self._add_timer(timer)
        return timer
"""

_PUMP_INTERVAL_OLD = """\
            repeat=repeat or None,
            pause=pause,
        )
        timer._start()
        self._timers.add(timer)
        return timer
"""

_PUMP_INTERVAL_NEW = """\
            repeat=repeat or None,
            pause=pause,
        )
        timer._start()
        self._add_timer(timer)
        return timer

    def _add_timer(self, timer: Timer) -> None:
        timers = self._timers
        if not isinstance(timers, WeakSet):
            timers = self._timers = WeakSet()
        timers.add(timer)
"""

# dom.py
_DOM_DEFAULTS_OLD = """\
    # Names of potential computed reactives
    _computes: ClassVar[frozenset[str]]
"""

_DOM_DEFAULTS_NEW = """\
    # Names of potential computed reactives
    _computes: ClassVar[frozenset[str]]

    # Defaults every node starts with live on the class, so an instance only stores what it
    # changes: a large transcript mounts thousands of nodes. `_id` stays an instance attribute:
    # its absence is how a missing `super().__init__()` is detected.
    _auto_refresh: float | None = None
    _auto_refresh_timer: Timer | None = None
    _has_hover_style: bool = False
    _has_focus_within: bool = False
    _has_order_style: bool = False
    \"\"\"The node has an ordered dependent pseudo-style (`:odd`, `:even`, `:first-of-type`, `:last-of-type`, `:first-child`, `:last-child`)\"\"\"
    _has_odd_or_even: bool = False
    \"\"\"The node has the pseudo class `odd` or `even`.\"\"\"
    _reactive_connect: dict[str, tuple[MessagePump, Reactive[object] | object]] | None = None
    _pruning: bool = False
    _trap_focus: bool = False
"""

_DOM_INIT_OLD = """\
        self._auto_refresh: float | None = None
        self._auto_refresh_timer: Timer | None = None
        self._css_types = {cls.__name__ for cls in self._css_bases(self.__class__)}
        self._bindings = (
            BindingsMap()
            if self._merged_bindings is None
            else self._merged_bindings.copy()
        )
        self._has_hover_style: bool = False
        self._has_focus_within: bool = False
        self._has_order_style: bool = False
        \"\"\"The node has an ordered dependent pseudo-style (`:odd`, `:even`, `:first-of-type`, `:last-of-type`, `:first-child`, `:last-child`)\"\"\"
        self._has_odd_or_even: bool = False
        \"\"\"The node has the pseudo class `odd` or `even`.\"\"\"
        self._reactive_connect: (
            dict[str, tuple[MessagePump, Reactive[object] | object]] | None
        ) = None
        self._pruning = False
        self._query_one_cache: LRUCache[QueryOneCacheKey, DOMNode] = LRUCache(1024)
        self._trap_focus = False
"""

_DOM_INIT_NEW = """\
        node_class = self.__class__
        css_types = node_class.__dict__.get("_css_types_shared")
        if css_types is None:
            # The same for every instance of a class: computed once, shared by all.
            css_types = frozenset(cls.__name__ for cls in self._css_bases(node_class))
            node_class._css_types_shared = css_types
        self._css_types: frozenset[str] = css_types
        self._bindings = (
            BindingsMap()
            if self._merged_bindings is None
            else self._merged_bindings.copy()
        )
        self._query_one_cache: LRUCache[QueryOneCacheKey, DOMNode] = LRUCache(1024)
"""

# widget.py
_WIDGET_IMPORT_OLD = """\
from textual._arrange import DockArrangeResult, arrange
"""

_WIDGET_IMPORT_NEW = """\
from textual._arrange import DockArrangeResult, arrange
from textual._compat import cached_property
"""

_WIDGET_DEFAULTS_OLD = """\
    }  # type: ignore[assignment]

    def __init__(
        self,
        *children: Widget,
"""

_WIDGET_DEFAULTS_NEW = """\
    }  # type: ignore[assignment]

    # Defaults every widget starts with live on the class, so an instance only stores what it
    # changes: a large transcript mounts thousands of widgets. Immutable values only.
    _layout_required: bool = False
    _layout_updates: int = 0
    _repaint_required: bool = False
    _scroll_required: bool = False
    _recompose_required: bool = False
    _refresh_styles_required: bool = False
    _default_layout: Layout = VerticalLayout()
    _animate: BoundAnimator | None = None
    highlight_style: Style | None = None
    _vertical_scrollbar: ScrollBar | None = None
    _horizontal_scrollbar: ScrollBar | None = None
    _scrollbar_corner: ScrollBarCorner | None = None
    _border_title: Content | None = None
    _border_subtitle: Content | None = None
    _visual_style: VisualStyle | None = None
    \"\"\"Cached style of visual.\"\"\"
    _visual_style_cache_key: int = -1
    \"\"\"Cache busting integer.\"\"\"
    _render_cache: _RenderCache = _RenderCache(NULL_SIZE, [])
    _content_width_cache: tuple[object, int] = (None, 0)
    _content_height_cache: tuple[object, int] = (None, 0)
    _tooltip: VisualType | None = None
    \"\"\"The tooltip content.\"\"\"
    _cover_widget: Widget | None = None
    \"\"\"Widget to render over this widget (used by loading indicator).\"\"\"
    _first_of_type: tuple[int, bool] = (-1, False)
    \"\"\"Used to cache :first-of-type pseudoclass state.\"\"\"
    _last_of_type: tuple[int, bool] = (-1, False)
    \"\"\"Used to cache :last-of-type pseudoclass state.\"\"\"
    _first_child: tuple[int, bool] = (-1, False)
    \"\"\"Used to cache :first-child pseudoclass state.\"\"\"
    _last_child: tuple[int, bool] = (-1, False)
    \"\"\"Used to cache :last-child pseudoclass state.\"\"\"
    _odd: tuple[int, bool] = (-1, False)
    \"\"\"Used to cache :odd pseudoclass state.\"\"\"
    _extrema: Extrema = Extrema()
    \"\"\"Optional minimum and maximum values for width and height.\"\"\"

    def __init__(
        self,
        *children: Widget,
"""

_WIDGET_INIT_OLD = """\
        self._render_markup = markup
        _null_size = NULL_SIZE
        self._size = _null_size
        self._container_size = _null_size
        self._layout_required = False
        self._layout_updates = 0
        self._repaint_required = False
        self._scroll_required = False
        self._recompose_required = False
        self._refresh_styles_required = False
        self._default_layout = VerticalLayout()
        self._animate: BoundAnimator | None = None
        Widget._sort_order += 1
        self.sort_order = Widget._sort_order
        self.highlight_style: Style | None = None

        self._vertical_scrollbar: ScrollBar | None = None
        self._horizontal_scrollbar: ScrollBar | None = None
        self._scrollbar_corner: ScrollBarCorner | None = None

        self._border_title: Content | None = None
        self._border_subtitle: Content | None = None

        self._layout_cache: dict[str, object] = {}
        \"\"\"A dict that is refreshed when the widget is resized / refreshed.\"\"\"

        self._visual_style: VisualStyle | None = None
        \"\"\"Cached style of visual.\"\"\"
        self._visual_style_cache_key: int = -1
        \"\"\"Cache busting integer.\"\"\"

        self._render_cache = _RenderCache(_null_size, [])
        # Regions which need to be updated (in Widget)
"""

_WIDGET_INIT_NEW = """\
        self._render_markup = markup
        _null_size = NULL_SIZE
        self._size = _null_size
        self._container_size = _null_size
        Widget._sort_order += 1
        self.sort_order = Widget._sort_order

        self._layout_cache: dict[str, object] = {}
        \"\"\"A dict that is refreshed when the widget is resized / refreshed.\"\"\"

        # Regions which need to be updated (in Widget)
"""

_WIDGET_CONTENT_CACHE_OLD = """\
        self._box_model_cache: LRUCache[object, BoxModel] = LRUCache(16)

        # Cache the auto content dimensions
        self._content_width_cache: tuple[object, int] = (None, 0)
        self._content_height_cache: tuple[object, int] = (None, 0)

        self._arrangement_cache: FIFOCache[
"""

_WIDGET_CONTENT_CACHE_NEW = """\
        self._box_model_cache: LRUCache[object, BoxModel] = LRUCache(16)

        self._arrangement_cache: FIFOCache[
"""

_WIDGET_TOOLTIP_OLD = """\
        self._visual_style_cache: dict[tuple[str, ...], VisualStyle] = {}

        self._tooltip: VisualType | None = None
        \"\"\"The tooltip content.\"\"\"
        self.absolute_offset: Offset | None = None
"""

_WIDGET_TOOLTIP_NEW = """\
        self._visual_style_cache: dict[tuple[str, ...], VisualStyle] = {}

        self.absolute_offset: Offset | None = None
"""

_WIDGET_LOCK_OLD = """\
        self.lock = RLock()
        \"\"\"`asyncio` lock to be used to synchronize the state of the widget.

        Two different tasks might call methods on a widget at the same time, which
        might result in a race condition.
        This can be fixed by adding `async with widget.lock:` around the method calls.
        \"\"\"
        self._anchored: bool = False
        \"\"\"Has this widget been anchored?\"\"\"
        self._anchor_released: bool = False
        \"\"\"Has the anchor been released?\"\"\"

        \"\"\"Flag to enable animation when scrolling anchored widgets.\"\"\"
        self._cover_widget: Widget | None = None
        \"\"\"Widget to render over this widget (used by loading indicator).\"\"\"

        self._first_of_type: tuple[int, bool] = (-1, False)
        \"\"\"Used to cache :first-of-type pseudoclass state.\"\"\"
        self._last_of_type: tuple[int, bool] = (-1, False)
        \"\"\"Used to cache :last-of-type pseudoclass state.\"\"\"
        self._first_child: tuple[int, bool] = (-1, False)
        \"\"\"Used to cache :first-child pseudoclass state.\"\"\"
        self._last_child: tuple[int, bool] = (-1, False)
        \"\"\"Used to cache :last-child pseudoclass state.\"\"\"
        self._odd: tuple[int, bool] = (-1, False)
        \"\"\"Used to cache :odd pseudoclass state.\"\"\"
        self._last_scroll_time = monotonic()
        \"\"\"Time of last scroll.\"\"\"
        self._extrema = Extrema()
        \"\"\"Optional minimum and maximum values for width and height.\"\"\"
"""

_WIDGET_LOCK_NEW = """\
        self._anchored: bool = False
        \"\"\"Has this widget been anchored?\"\"\"
        self._anchor_released: bool = False
        \"\"\"Has the anchor been released?\"\"\"
        self._last_scroll_time = monotonic()
        \"\"\"Time of last scroll.\"\"\"

    @cached_property
    def lock(self) -> RLock:
        \"\"\"`asyncio` lock to be used to synchronize the state of the widget.

        Two different tasks might call methods on a widget at the same time, which
        might result in a race condition.
        This can be fixed by adding `async with widget.lock:` around the method calls.
        \"\"\"
        return RLock()
"""

# signal.py
_SIGNAL_INIT_OLD = """\
        self._subscriptions: WeakKeyDictionary[
            DOMNode, list[SignalCallbackType[SignalT]]
        ] = WeakKeyDictionary()
"""

_SIGNAL_INIT_NEW = """\
        # Created by the first subscription: every message pump owns a signal most never publish.
        self._subscriptions: (
            WeakKeyDictionary[DOMNode, list[SignalCallbackType[SignalT]]] | None
        ) = None
"""

_SIGNAL_REPR_OLD = """\
        yield "subscriptions", list(self._subscriptions.keys())
"""

_SIGNAL_REPR_NEW = """\
        yield "subscriptions", list(self._subscriptions or ())
"""

_SIGNAL_SUBSCRIBE_OLD = """\
        callbacks = self._subscriptions.setdefault(node, [])
        callbacks.append(signal_callback)
"""

_SIGNAL_SUBSCRIBE_NEW = """\
        subscriptions = self._subscriptions
        if subscriptions is None:
            subscriptions = self._subscriptions = WeakKeyDictionary()
        callbacks = subscriptions.setdefault(node, [])
        callbacks.append(signal_callback)
"""

_SIGNAL_UNSUBSCRIBE_OLD = """\
        self._subscriptions.pop(node, None)

    def publish"""

_SIGNAL_UNSUBSCRIBE_NEW = """\
        if self._subscriptions is not None:
            self._subscriptions.pop(node, None)

    def publish"""

_SIGNAL_PUBLISH_OLD = """\
        if not self._subscriptions:
            return
"""

_SIGNAL_PUBLISH_NEW = """\
        subscriptions = self._subscriptions
        if not subscriptions:
            return
"""

_SIGNAL_PUBLISH_LOOP_OLD = """\
        for node, callbacks in list(self._subscriptions.items()):
            if not (node.is_running and node.is_attached) or node._pruning:
                # Removed nodes that are no longer running
                self._subscriptions.pop(node)
"""

_SIGNAL_PUBLISH_LOOP_NEW = """\
        for node, callbacks in list(subscriptions.items()):
            if not (node.is_running and node.is_attached) or node._pruning:
                # Removed nodes that are no longer running
                subscriptions.pop(node)
"""


def _patch(module_file: str, old: str, new: str, description: str, equivalent: tuple[str, ...] = ()) -> FilePatch:
    return FilePatch(
        package="textual",
        module_file=module_file,
        old_fragment=old,
        new_fragment=new,
        description=description,
        equivalent_fragments=equivalent,
    )


_QUEUE_PATCH = _patch("_queue.py", _QUEUE_OLD, _QUEUE_NEW, "Message pump queue reads a list", (_QUEUE_WITH_FLAG_NEW,))

_PUMP_PATCHES = (
    _patch(
        "message_pump.py",
        _PUMP_IMPORT_WITH_FLAG_OLD,
        _PUMP_IMPORT_WITH_FLAG_NEW,
        "Upgrade message_pump: Flag is no longer imported",
    ),
    _patch("message_pump.py", _PUMP_FLAG_OLD, _PUMP_FLAG_NEW, "message_pump defines Flag"),
    _patch("message_pump.py", _PUMP_INIT_OLD, _PUMP_INIT_NEW, "MessagePump defaults live on the class"),
    _patch("message_pump.py", _PUMP_MOUNTED_OLD, _PUMP_MOUNTED_NEW, "MessagePump._mounted_event is a Flag"),
    _patch("message_pump.py", _PUMP_DISABLED_OLD, _PUMP_DISABLED_NEW, "MessagePump replaces its disabled messages"),
    _patch("message_pump.py", _PUMP_TIMER_OLD, _PUMP_TIMER_NEW, "MessagePump.set_timer makes the timer set"),
    _patch("message_pump.py", _PUMP_INTERVAL_OLD, _PUMP_INTERVAL_NEW, "MessagePump.set_interval makes the timer set"),
)

_DOM_PATCHES = (
    _patch("dom.py", _DOM_DEFAULTS_OLD, _DOM_DEFAULTS_NEW, "DOMNode defaults live on the class"),
    _patch("dom.py", _DOM_INIT_OLD, _DOM_INIT_NEW, "DOMNode.__init__ stores changed state only"),
)

_WIDGET_PATCHES = (
    _patch("widget.py", _WIDGET_IMPORT_OLD, _WIDGET_IMPORT_NEW, "widget imports cached_property"),
    _patch("widget.py", _WIDGET_DEFAULTS_OLD, _WIDGET_DEFAULTS_NEW, "Widget defaults live on the class"),
    _patch("widget.py", _WIDGET_INIT_OLD, _WIDGET_INIT_NEW, "Widget.__init__ stores changed state only"),
    _patch("widget.py", _WIDGET_CONTENT_CACHE_OLD, _WIDGET_CONTENT_CACHE_NEW, "Widget content caches start empty"),
    _patch("widget.py", _WIDGET_TOOLTIP_OLD, _WIDGET_TOOLTIP_NEW, "Widget tooltip starts unset"),
    _patch("widget.py", _WIDGET_LOCK_OLD, _WIDGET_LOCK_NEW, "Widget.lock is made on first use"),
)

_SIGNAL_PATCHES = (
    _patch("signal.py", _SIGNAL_INIT_OLD, _SIGNAL_INIT_NEW, "Signal subscriptions are made on first subscribe"),
    _patch("signal.py", _SIGNAL_REPR_OLD, _SIGNAL_REPR_NEW, "Signal repr reads no subscriptions"),
    _patch("signal.py", _SIGNAL_SUBSCRIBE_OLD, _SIGNAL_SUBSCRIBE_NEW, "Signal.subscribe makes the subscriptions"),
    _patch("signal.py", _SIGNAL_UNSUBSCRIBE_OLD, _SIGNAL_UNSUBSCRIBE_NEW, "Signal.unsubscribe without subscriptions"),
    _patch("signal.py", _SIGNAL_PUBLISH_OLD, _SIGNAL_PUBLISH_NEW, "Signal.publish without subscriptions"),
    _patch("signal.py", _SIGNAL_PUBLISH_LOOP_OLD, _SIGNAL_PUBLISH_LOOP_NEW, "Signal.publish prunes its subscriptions"),
)

_MODULE_PATCHES: dict[str, tuple[FilePatch, ...]] = {
    "textual._queue": (_QUEUE_PATCH,),
    "textual.message_pump": _PUMP_PATCHES,
    "textual.dom": _DOM_PATCHES,
    "textual.widget": _WIDGET_PATCHES,
    "textual.signal": _SIGNAL_PATCHES,
}

register(*(patch for patches in _MODULE_PATCHES.values() for patch in patches))


def apply_runtime_patch() -> None:
    """Install the class defaults and lazy containers in the current process, before any pump exists."""
    try:
        import textual
        from textual._compat import cached_property
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual node diet runtime patch: loaded Textual is not the pinned %s that the patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return
    modules = {name: importlib.import_module(name) for name in _MODULE_PATCHES}
    if all(
        members_installed(modules[name], members, _RUNTIME_PATCH_MARKER) for name, members in _RUNTIME_MEMBERS.items()
    ):
        return
    # ``Widget.lock`` is decorated while it is staged, so the alias must exist first.
    vars(modules["textual.widget"]).setdefault("cached_property", cached_property)
    staged: dict[str, StagedMembers] = {}
    try:
        for name, patches in _MODULE_PATCHES.items():
            staged[name] = stage_members(
                modules[name],
                patches,
                _RUNTIME_MEMBERS[name],
                label="node diet",
                classes=_RUNTIME_CLASSES.get(name, ()),
            )
    except StagedSourceDriftError as exc:
        logger.warning("Skipping Textual node diet runtime patch: %s", exc)
        return
    for members in staged.values():
        members.install(_RUNTIME_PATCH_MARKER)
