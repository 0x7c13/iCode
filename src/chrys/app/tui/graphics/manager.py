# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Draw chat images with the terminal's own graphics, where it has a protocol for them.

One ``TerminalGraphics`` per app holds the protocol chosen at start, what the terminal has been sent,
the background jobs that build it, and, for Sixel, the overlay drawn after each frame. Image widgets
find it with ``graphics_for(app)``; where there is none (tests, the web host, unknown terminals) they
draw half-blocks.

Kitty: each image is transmitted once for its cell box and placed with Unicode placeholder cells
that the widget renders (``kitty.py``). The terminal keeps the pixels, so an image costs nothing per
frame; ids the terminal holds are kept within a budget below its storage limit, the least recently
drawn deleted first, and all are deleted on suspend and exit, while the app's screen is still the
terminal's: kitty and Ghostty keep each screen's images apart, so a delete sent after the app has
left the alternate screen frees nothing.

Sixel: the widget renders half-blocks and the overlay draws each image row over them inside the
frame Textual writes (``ChrysApp._display``). Windows Terminal erases a cell's image whenever text is
printed into that cell, and only there, so every repaint that prints into an image row takes the
image away there with no bookkeeping, and the overlay draws that row again: at once while the image
stays put, or once it has stopped moving, since redrawing every frame of a scroll would flood the
terminal.
"""

from __future__ import annotations

import asyncio
import binascii
import logging
import os
import secrets
import sys
import weakref
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from PIL import Image, UnidentifiedImageError
from textual import __version__ as TEXTUAL_VERSION
from textual._compositor import ChopsUpdate, CompositorUpdate, LayoutUpdate
from textual.geometry import Region

from chrys.app.tui.graphics import kitty
from chrys.app.tui.graphics.detect import GraphicsProtocol, detect_graphics_protocol, terminal_cell_pixels
from chrys.app.tui.graphics.payloads import kitty_png, sixel_rows
from chrys.app.tui.graphics.sixel import EncodingStopped
from chrys.app.tui.terminal.title import app_terminal_title_writer, running_under_textual_web_driver

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
    from textual.app import App
    from textual.screen import Screen

    from chrys.app.tui.widgets.chat.image_preview import ChatImagePreview, ImageBox
    from chrys.app.tui.widgets.chat.image_view import TerminalImages

logger = logging.getLogger(__name__)

KITTY_BUDGET_BYTES: Final = 192 * 1024 * 1024
"""Decoded bytes kept in the terminal; Ghostty and kitty start evicting at 320 MB."""
SIXEL_BUDGET_CHARS: Final = 48 * 1024 * 1024
"""Encoded rows kept for images not on screen."""
FRAME_BUDGET_CHARS: Final = 192 * 1024
"""Sixel written per frame: Textual's writer queues 30 writes and Windows splits a frame into 8 KB
writes, so a larger frame would make the event loop wait for the terminal."""
SETTLE_SECONDS: Final = 0.12
"""How long a moving image waits, after it last moved, before Sixel is drawn again."""
SIXEL_TEXTUAL_VERSION: Final = "8.2.7"
"""The Textual whose private frame path (``App._display``, compositor updates and visible-widget map)
the overlay was written against; any other draws half-blocks instead of Sixel."""
_DEFAULT_CELL_PIXELS: Final = (10, 20)
_DECODE_ERRORS = (binascii.Error, Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError)
_registry: weakref.WeakKeyDictionary[App, TerminalGraphics] = weakref.WeakKeyDictionary()


@dataclass(eq=False)
class ImageSlot:
    """What the terminal has been sent, or is being sent, for one image of one widget."""

    preview: ChatImagePreview
    kitty_id: int = 0
    """The transmitted image's id, or 0."""
    kitty_box: ImageBox | None = None
    sixel: tuple[str, ...] = ()
    sixel_box: ImageBox | None = None
    pending: ImageBox | None = None
    """The box a job is building a payload for."""
    failed: ImageBox | None = None
    """A box whose payload could not be built; not asked for again."""


def graphics_for(app: App) -> TerminalGraphics | None:
    """The app's graphics, or None where images are drawn as half-blocks."""
    return _registry.get(app)


def start_terminal_graphics(app: App, *, enabled: bool) -> TerminalGraphics | None:
    """Choose a protocol for ``app``'s terminal and install the graphics that draw it, if any."""
    write = app_terminal_title_writer(app)
    if not enabled or write is None or app.is_headless or app.is_inline or running_under_textual_web_driver():
        return None
    protocol = detect_graphics_protocol(os.environ, truecolor=app.console.color_system == "truecolor")
    if protocol is GraphicsProtocol.BLOCKS:
        return None
    if protocol is GraphicsProtocol.SIXEL and TEXTUAL_VERSION != SIXEL_TEXTUAL_VERSION:
        return None
    graphics = TerminalGraphics(app, protocol, write)
    logger.debug("Terminal images use %s", protocol.value)
    return graphics


class TerminalGraphics:
    """The app's terminal images: payload jobs, kitty ids, and the Sixel overlay."""

    def __init__(
        self,
        app: App,
        protocol: GraphicsProtocol,
        write: Callable[[str], object],
        *,
        cell_pixels: Callable[[], tuple[int, int] | None] | None = None,
    ) -> None:
        self.protocol = protocol
        self._app = app
        self._write = write
        self._cell_pixels = cell_pixels or _tty_cell_pixels
        self._views: weakref.WeakSet[TerminalImages] = weakref.WeakSet()
        self._slots: OrderedDict[ImageSlot, int] = OrderedDict()
        """Slots holding a payload, least recently drawn first, with what each costs."""
        self._spent = 0
        self._doomed: set[int] = set()
        """Ids to delete once the frame being rendered is written."""
        self._jobs: deque[_Job] = deque()
        self._runner: asyncio.Task[None] | None = None
        self._closed = False
        self.overlay = SixelOverlay(app, self) if protocol is GraphicsProtocol.SIXEL else None
        _registry[app] = self
        # Never unsubscribed: unsubscribing is per node and would drop the app's other subscriptions.
        # Immediate: a covered screen must stop placing images before the modal's first frame, and
        # images must be freed before the terminal is handed over.
        app.screen_change_signal.subscribe(app, self._on_screen_change, immediate=True)
        app.app_suspend_signal.subscribe(app, self._on_suspend, immediate=True)
        app.app_resume_signal.subscribe(app, self._on_resume, immediate=True)

    # ── Views ────────────────────────────────────────────────────

    @property
    def views(self) -> tuple[TerminalImages, ...]:
        """The mounted image widgets."""
        return tuple(self._views)

    def add_view(self, view: TerminalImages) -> None:
        """Track an image widget being drawn."""
        if view in self._views:
            return
        self._views.add(view)
        if self.overlay is not None:
            self.overlay.request_frame()

    def remove_view(self, view: TerminalImages, slots: Sequence[ImageSlot]) -> None:
        """Forget an unmounted widget and free what the terminal holds for it."""
        self._views.discard(view)
        if self.overlay is not None:
            self.overlay.forget(view)
        for slot in slots:
            self.release(slot)

    def release(self, slot: ImageSlot) -> None:
        """Free one image's payload; a job building one is dropped when it lands."""
        slot.pending = None
        self._drop(slot)

    def frame(self, renderable: RenderableType | None) -> RenderableType | None:
        """What ``App._display`` should write for ``renderable``: a frame is followed by its Sixel."""
        if self.overlay is None or not isinstance(renderable, CompositorUpdate):
            return renderable
        return self.overlay.wrap(renderable)

    # ── Payloads ─────────────────────────────────────────────────

    def kitty_image(self, view: TerminalImages, slot: ImageSlot, box: ImageBox) -> int:
        """The id to place ``slot`` in ``box`` with, or 0 while it is being sent (draw half-blocks)."""
        if slot.kitty_id and slot.kitty_box == box:
            self._slots.move_to_end(slot)
            return slot.kitty_id
        if slot.kitty_id:
            # The old placement is already gone from the frame being rendered; delete after it is written.
            self._drop_after_frame(slot)
        if max(box.columns, box.rows) <= kitty.MAX_PLACEMENT_CELLS:
            self._request(view, slot, box)
        return 0

    def sixel_rows(self, view: TerminalImages, slot: ImageSlot, box: ImageBox) -> tuple[str, ...]:
        """One Sixel string per cell row of ``slot`` in ``box``, or () while they are being built."""
        if slot.sixel and slot.sixel_box == box:
            self._slots.move_to_end(slot)
            return slot.sixel
        if slot.sixel:
            self._drop(slot)
        self._request(view, slot, box)
        return ()

    def _request(self, view: TerminalImages, slot: ImageSlot, box: ImageBox) -> None:
        if self._closed or slot.pending == box or slot.failed == box:
            return
        slot.pending = box
        self._jobs.append(_Job(weakref.ref(view), slot, box))
        if self._runner is None or self._runner.done():
            self._runner = asyncio.get_running_loop().create_task(self._run_jobs(), name="terminal-image-payloads")

    async def _run_jobs(self) -> None:
        while self._jobs and not self._closed:
            job = self._jobs.popleft()
            view = job.view()
            if view is None or job.slot.pending != job.box:
                continue
            build = self._builder(job)
            try:
                payload = await asyncio.to_thread(build)
            except EncodingStopped:
                continue
            except _DECODE_ERRORS:
                logger.debug("Could not build a terminal image payload", exc_info=True)
                if job.slot.pending == job.box:
                    job.slot.pending = None
                    job.slot.failed = job.box
                continue
            view = job.view()
            if self._closed or view is None or job.slot.pending != job.box:
                continue
            job.slot.pending = None
            self._land(view, job.slot, job.box, payload)

    def _builder(self, job: _Job) -> Callable[[], bytes | tuple[str, ...]]:
        data_uri, box, slot = job.slot.preview.data_uri, job.box, job.slot
        if self.protocol is GraphicsProtocol.KITTY:
            cells = self._cell_pixels() or _DEFAULT_CELL_PIXELS
            return lambda: kitty_png(data_uri, columns=box.columns, rows=box.rows, cell_pixels=cells)

        # Asked from the worker thread: plain attributes only. Unmounting or a new box releases the
        # slot, which clears ``pending``; the view is never dereferenced there, where its last
        # reference could be dropped off the event loop.
        def wanted() -> bool:
            return not self._closed and slot.pending == box

        return lambda: sixel_rows(
            data_uri, columns=box.columns, rows=box.rows, pixel_height=box.pixel_height, wanted=wanted
        )

    def _land(self, view: TerminalImages, slot: ImageSlot, box: ImageBox, payload: bytes | tuple[str, ...]) -> None:
        self._drop(slot)
        if isinstance(payload, bytes):
            image_id = self._new_kitty_id()
            self._write("".join(kitty.transmit_commands(image_id, payload, columns=box.columns, rows=box.rows)))
            slot.kitty_id, slot.kitty_box = image_id, box
            cost = _kitty_cost(box, self._cell_pixels() or _DEFAULT_CELL_PIXELS)
            budget = KITTY_BUDGET_BYTES
        else:
            slot.sixel, slot.sixel_box = payload, box
            cost = sum(len(row) for row in payload)
            budget = SIXEL_BUDGET_CHARS
        self._slots[slot] = cost
        self._spent += cost
        self._evict(budget, keep=slot)
        if self.overlay is not None:
            self.overlay.request_frame()
        else:
            view.refresh()

    def _new_kitty_id(self) -> int:
        # Random ids rather than a counter: a placeholder color nudged by one by a stray filter would
        # otherwise name a neighbouring image of ours instead of none.
        held = {slot.kitty_id for slot in self._slots} | self._doomed
        while True:
            image_id = 1 + secrets.randbelow(kitty.MAX_IMAGE_ID)
            if image_id not in held:
                return image_id

    def _evict(self, budget: int, *, keep: ImageSlot) -> None:
        for slot in list(self._slots):
            if self._spent <= budget:
                return
            if slot is keep:
                continue
            had_placement = bool(slot.kitty_id)
            self._drop(slot)
            if had_placement:
                # Its cells would turn blank: draw half-blocks until it is sent again.
                for view in self._views:
                    if slot in view.slots:
                        view.refresh()

    def _drop(self, slot: ImageSlot) -> None:
        cost = self._slots.pop(slot, None)
        if cost is not None:
            self._spent -= cost
        if slot.kitty_id:
            self._write(kitty.delete_command(slot.kitty_id))
        slot.kitty_id, slot.kitty_box = 0, None
        slot.sixel, slot.sixel_box = (), None

    def _drop_after_frame(self, slot: ImageSlot) -> None:
        image_id = slot.kitty_id
        cost = self._slots.pop(slot, None)
        if cost is not None:
            self._spent -= cost
        slot.kitty_id, slot.kitty_box = 0, None
        self._doomed.add(image_id)
        asyncio.get_running_loop().call_soon(self._write_delete, image_id)

    def _write_delete(self, image_id: int) -> None:
        if image_id in self._doomed:
            self._doomed.discard(image_id)
            self._write(kitty.delete_command(image_id))

    def _delete_everything(self) -> None:
        for slot in list(self._slots):
            if slot.kitty_id:
                self._drop(slot)
        for image_id in list(self._doomed):
            self._write_delete(image_id)

    # ── Lifecycle ────────────────────────────────────────────────

    def _on_screen_change(self, _screen: Screen) -> None:
        if self._closed:
            return
        # Placeholders are drawn only on the active screen: a screen under a modal is tinted, which
        # would change the color naming each image.
        if self.protocol is GraphicsProtocol.KITTY:
            for view in self._views:
                view.refresh()

    def _on_suspend(self, _app: App) -> None:
        if self._closed:
            return
        # The terminal is handed to another program: free its memory; images are sent again on resume.
        self._delete_everything()

    def _on_resume(self, _app: App) -> None:
        if self._closed:
            return
        for view in self._views:
            view.refresh()
        if self.overlay is not None:
            self.overlay.reset()

    def stop(self) -> None:
        """Delete every image the terminal holds, and send or draw none from now on.

        Call it while the app's screen is still the terminal's, before the app exits: images deleted
        later stay in the terminal's memory.
        """
        if self._closed:
            return
        self._delete_everything()
        for slot in list(self._slots):
            self._drop(slot)
        self._closed = True
        self._jobs.clear()
        if self.overlay is not None:
            self.overlay.close()
        _registry.pop(self._app, None)

    async def close(self) -> None:
        """Stop, and wait for the payload being built, if any, to be dropped."""
        self.stop()
        runner, self._runner = self._runner, None
        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                if (task := asyncio.current_task()) is not None and task.cancelling():
                    raise


@dataclass(frozen=True, eq=False)
class _Job:
    view: weakref.ref[TerminalImages]
    slot: ImageSlot
    box: ImageBox


@dataclass(eq=False)
class _ViewState:
    region: Region | None = None
    """The view's content region in the last frame."""
    drawn: set[tuple[int, int, ImageBox]] = field(default_factory=set)
    """Rows on screen now, as (image index, row, box)."""


_Painted = dict[int, list[tuple[int, int]]] | None
"""The cells a frame prints into, as column ranges by screen row; None for every cell."""


class SixelOverlay:
    """Draws Sixel image rows over the half-blocks of the frames Textual writes."""

    def __init__(self, app: App, graphics: TerminalGraphics) -> None:
        self._app = app
        self._graphics = graphics
        self._states: weakref.WeakKeyDictionary[TerminalImages, _ViewState] = weakref.WeakKeyDictionary()
        self._frame_handle: asyncio.Handle | None = None
        self._settle_handle: asyncio.TimerHandle | None = None
        self._closed = False

    def wrap(self, update: CompositorUpdate) -> _OverlaidUpdate:
        """``update`` followed by the image rows it erased, and any still owed."""
        return _OverlaidUpdate(update, self)

    def forget(self, view: TerminalImages) -> None:
        """Stop drawing ``view``; its cells are repainted as it is removed."""
        self._states.pop(view, None)

    def reset(self) -> None:
        """Assume nothing is drawn: after a resume the screen is painted from scratch."""
        for state in self._states.values():
            state.drawn.clear()
            state.region = None
        self.request_frame()

    def close(self) -> None:
        """Stop drawing."""
        self._closed = True
        for handle in (self._frame_handle, self._settle_handle):
            if handle is not None:
                handle.cancel()
        self._frame_handle = self._settle_handle = None

    def request_frame(self) -> None:
        """Draw what is owed in a frame of its own, soon."""
        if not self._closed and self._frame_handle is None:
            self._frame_handle = asyncio.get_running_loop().call_soon(self._overlay_frame)

    def _request_settled_frame(self) -> None:
        """Draw once nothing has moved for ``SETTLE_SECONDS``."""
        if self._closed:
            return
        if self._settle_handle is not None:
            self._settle_handle.cancel()
        self._settle_handle = asyncio.get_running_loop().call_later(SETTLE_SECONDS, self._settled)

    def _settled(self) -> None:
        self._settle_handle = None
        self._overlay_frame()

    def _overlay_frame(self) -> None:
        self._frame_handle = None
        app = self._app
        if self._closed or not app.is_running:
            return
        if app._batch_count:
            # Inside a batch the compositor can be ahead of the screen; draw once it is painted.
            self._request_settled_frame()
            return
        app._display(app.screen, _OverlayFrame())

    def frame_suffix(self, update: CompositorUpdate) -> str:
        """The Sixel to write after ``update``: rows it erased, and rows not drawn yet."""
        if self._closed:
            return ""
        screen = self._app.screen
        height = screen.size.height
        painted = _painted_cells(update)
        visible = screen._compositor.visible_widgets
        order = list(visible)
        out: list[str] = []
        spent = 0
        moving = owed = False
        for view in list(self._graphics.views):
            state = self._states.setdefault(view, _ViewState())
            if state.region is not None and painted != {}:
                left, top = state.region.x, state.region.y
                state.drawn = {
                    (index, row, box)
                    for index, row, box in state.drawn
                    if not _prints_into(painted, top + row, left + box.x, left + box.x + box.columns)
                }
            geometry = visible.get(view) if view.is_attached and view.screen is screen else None
            if geometry is None:
                state.region = None
                state.drawn.clear()
                continue
            region, clip = geometry
            content = region.shrink(view.styles.gutter)
            if content != state.region:
                # Moving: drawing each step of a scroll would flood the terminal.
                state.region = content
                state.drawn.clear()
                moving = True
                continue
            if owed:
                continue
            covers = [
                other_region.intersection(other_clip)
                for other_region, other_clip in (visible[other] for other in order[: order.index(view)])
                if other_region.overlaps(content)
            ]
            for index, (slot, box) in enumerate(view.image_cells(content.width)):
                for row, text in enumerate(self._graphics.sixel_rows(view, slot, box)):
                    key = (index, row, box)
                    if key in state.drawn:
                        continue
                    cells = Region(content.x + box.x, content.y + row, box.columns, 1)
                    # A row's last band reaches into the next row: on the last screen row it would scroll.
                    if (
                        cells.bottom >= height
                        or not clip.contains_region(cells)
                        or any(cover.overlaps(cells) for cover in covers)
                    ):
                        continue
                    if spent and spent + len(text) > FRAME_BUDGET_CHARS:
                        owed = True
                        break
                    out.append(f"\x1b[{cells.y + 1};{cells.x + 1}H{text}")
                    spent += len(text)
                    state.drawn.add(key)
                if owed:
                    break
        if moving:
            self._request_settled_frame()
        if owed:
            self.request_frame()
        return "".join(out)


def _painted_cells(update: CompositorUpdate) -> _Painted:
    """The cells ``update`` prints into.

    Textual prints a changed span from the start of each chop (one widget's part of a line) it
    overlaps, and may run past the span for a wide character, so the span itself undercounts: the
    whole chop is counted, up to the strip's own width.
    """
    if isinstance(update, _OverlayFrame):
        return {}
    if isinstance(update, ChopsUpdate):
        painted: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for y, x1, x2 in update.spans:
            for x, strip in update.chops[y].items():
                if strip is not None and x <= x2 and x + strip.cell_length > x1:
                    painted[y].append((x, x + strip.cell_length))
        return painted
    if isinstance(update, LayoutUpdate):
        region = update.region
        return {y: [(region.x, region.right)] for y in range(region.y, region.bottom)}
    return None


def _prints_into(painted: _Painted, y: int, x1: int, x2: int) -> bool:
    """Whether ``painted`` covers any cell of row ``y`` from column ``x1`` up to ``x2``."""
    if painted is None:
        return True
    return any(start < x2 and end > x1 for start, end in painted.get(y, ()))


class _OverlaidUpdate(CompositorUpdate):
    """A compositor update followed by the Sixel rows to draw over it."""

    def __init__(self, update: CompositorUpdate, overlay: SixelOverlay) -> None:
        self._update = update
        self._overlay = overlay

    def render_segments(self, console: Console) -> str:
        frame = self._update.render_segments(console)
        try:
            suffix = self._overlay.frame_suffix(self._update)
        except Exception:
            # Inside Textual's frame write an error would end the app; the half-blocks stay instead.
            logger.exception("Could not draw terminal images")
            return frame
        return frame + suffix

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        _ = console, options
        yield self._update


class _OverlayFrame(CompositorUpdate):
    """A frame with nothing of Textual's to paint, written to draw Sixel that is owed."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        _ = console, options
        yield from ()


def _kitty_cost(box: ImageBox, cell_pixels: tuple[int, int]) -> int:
    """Decoded bytes the terminal holds: at most the box's pixels, as RGBA."""
    width, height = cell_pixels
    return min(box.columns * width, 4096) * min(box.rows * height, 4096) * 4


def _tty_cell_pixels() -> tuple[int, int] | None:
    """The cell size from whichever standard stream is the terminal."""
    for stream in (sys.__stdout__, sys.__stderr__, sys.__stdin__):
        if stream is None:
            continue
        try:
            fd = stream.fileno()
        except OSError, ValueError:
            continue
        if (cells := terminal_cell_pixels(fd)) is not None:
            return cells
    return None
