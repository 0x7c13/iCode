# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: a Textual reflow replays the arrangement of every subtree that has not changed.

Problem
-------
Each layout refresh (``Screen._refresh_layout`` → ``Compositor.reflow``) arranges the whole
widget tree from scratch: box models, placements and map entries for every mounted transcript
widget, and then compares every widget's old and new geometry. A streaming reply, a sidebar tab
switch or a panel toggle changes a handful of widgets, yet each frame's layout costs time and
garbage in proportion to the mounted transcript.

Solution
--------
Nodes carry the layout epoch in which they, or a node beneath them, last changed in a way the
compositor reads (``DOMNode._mark_layout_dirty``): a mount, a child-list update, any
``refresh()``, a layout or scroll change, an absolute offset, anchoring, a raw style-rule
write (``Styles.set_rule``/``clear_rule``, which chrome widgets use to skip layout escalation)
or an anchored widget's container-size change. The compositor pins an anchored widget from its
container size, which ``_size_updated`` stores only after the arrangement that pinned it, and
nothing else stamps the widget when its scrollbar is hidden. A reactive that refreshes stamps
before its watchers run rather than at that refresh: a watcher may look up geometry, which can
rebuild the full map (``TextArea`` sizes its wrapping when its scrollbar shows).
A full arrangement records what arranging each widget produced, and the next reflow replays the
record of every subtree unchanged since then, translating it when only its origin moved.
``Widget.layers`` takes the outermost ``layers`` declaration among a widget and its ancestors, so
a declaration above a subtree restacks it without stamping it: a record also keeps the
declaration its widget inherits, ``None`` when none does, which an explicit ``default`` must not
match.
Replayed placements are the recorded objects, so the reflow diff settles most widgets by
identity.

Only a reflow from ``Screen._refresh_layout``, ``full_map`` or ``reflow_reusing_records`` reuses
records. The last serves code that writes raw style rules outside a layout pass (chrome visibility
and width flips) and then resynchronizes the compositor: the rule writes stamp their widgets, so
only those widgets' ancestor paths are arranged again. ``_arrange_root`` keeps its signature and
takes the choice from ``_reuse_next_arrangement``. As upstream's, a reflow leaves a pending
full-map rebuild pending: a repaint outside the visible map requests one, and a widget shown after
the reflow then reports the region its next layout gives it. That rebuild must not decide which
widgets are new, though: a reflow reports Show against the map the previous reflow produced, so a
widget shown in between still gets Show and its first Resize after a lookup mapped it. Tests set
``textual._compositor._VERIFY_REUSE`` to check every reusing arrangement against a from-scratch
one that starts from the same scroll and anchor state: anchoring and ``arrange`` overrides such
as the chat panel's write that state, so arranging twice in a row can differ. The check compares
the state each leaves as well as the maps, and restores the layout epochs the from-scratch
arrangement's watchers stamp, which would otherwise hide a change that was not stamped.
``Widget._set_dirty`` also stops marking the lines of a styles cache it has just cleared:
reading ``self.size`` for that could rebuild the full map.

The patch is runtime only. Its fragments span nine modules, and file patches apply all or
nothing per file: after a Textual change a compositor could be rewritten to reuse records while
another module kept the upstream code that does not stamp, painting stale placements from then
on. The runtime install stages every module before installing any, and must precede the first
widget: ``absolute_offset``, ``_anchored`` and ``_anchor_released`` become properties, which
hide the values an existing instance keeps in its ``__dict__``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, TypeGuard

from chrys.foundation.patches.patcher import FilePatch
from chrys.foundation.patches.staged_members import StagedSourceDriftError, members_installed, stage_members

if TYPE_CHECKING:
    from textual._compositor import Compositor, ReflowResult
    from textual.geometry import Size
    from textual.widget import Widget

_RUNTIME_PATCH_MARKER = "_chrys_reflow_reuse"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
_RUNTIME_MEMBERS: dict[str, dict[str, tuple[str, ...]]] = {
    "textual._compositor": {
        "Compositor": (
            "__init__",
            "clear",
            "reflow",
            "full_map",
            "_arrange_root",
            "_arrange",
            "_arrangement_state",
            "_restore_arrangement_state",
            "_verify_reuse",
        ),
    },
    "textual.css.styles": {"Styles": ("clear_rule", "set_rule")},
    "textual._node_list": {"NodeList": ("updated",)},
    "textual.dom": {"DOMNode": ("_layout_epoch", "_layout_dirty_epoch", "_mark_layout_dirty", "_on_mounted_state")},
    "textual.message_pump": {"MessagePump": ("_pre_process", "_on_mounted_state", "_process_messages_loop")},
    "textual.reactive": {"Reactive": ("_set",)},
    "textual.screen": {"Screen": ("_refresh_layout",)},
    "textual.scroll_view": {"ScrollView": ("_size_updated",)},
    "textual.widget": {
        "Widget": (
            "_absolute_offset",
            "_anchored_flag",
            "_anchor_released_flag",
            "absolute_offset",
            "_anchored",
            "_anchor_released",
            "_set_dirty",
            "set_scroll",
            "_refresh_scroll",
            "refresh",
            "_check_refresh",
            "_size_updated",
        ),
    },
}
logger = logging.getLogger(__name__)

# _compositor.py: <import>
_REFLOW_1_OLD = """\
from textual._context import visible_screen_stack
from textual._loop import loop_last
from textual.geometry import NULL_SPACING, Offset, Region, Size, Spacing
from textual.map_geometry import MapGeometry"""

_REFLOW_1_NEW = """\
from textual._context import visible_screen_stack
from textual._loop import loop_last
from textual.dom import DOMNode
from textual.geometry import NULL_SPACING, Offset, Region, Size, Spacing
from textual.map_geometry import MapGeometry"""

# _compositor.py: <if>, _ArrangeRecord=, _VERIFY_REUSE=
_REFLOW_2_OLD = """\
    from typing_extensions import TypeAlias

    from textual.screen import Screen

"""

_REFLOW_2_NEW = """\
    from typing_extensions import TypeAlias

    from textual.layout import WidgetPlacement
    from textual.screen import Screen


_VERIFY_REUSE = False
\"\"\"Check every reusing arrangement against a from-scratch one (a debug aid for tests).\"\"\"

_ArrangeRecord: TypeAlias = "tuple[int, tuple[Region, Region, tuple[tuple[int, int, int], ...], int, Region, bool, Spacing, tuple[str, ...] | None], bool, set[Widget] | None, tuple[Widget, MapGeometry] | None, list[Widget], list[tuple[Widget, MapGeometry]], Region | None, frozenset[Widget], bool]"
\"\"\"What arranging one widget produced: (epoch, arguments, visible, arranged widgets, cover entry, placed children,
map entries, children's clipping region, overlay children, whether the subtree may be translated).\"\"\"

"""

# _compositor.py: Compositor.__init__, Compositor.clear
_REFLOW_3_OLD = """\
        self._layers_visible: list[list[tuple[Widget, Region, Region]]] | None = None

    def clear(self) -> None:
        \"\"\"Remove all references to widgets (used when the screen closes).\"\"\""""

_REFLOW_3_NEW = """\
        self._layers_visible: list[list[tuple[Widget, Region, Region]]] | None = None

        # What the last full arrangement produced per widget, for replaying unchanged subtrees
        self._arrange_records: dict[Widget, _ArrangeRecord] = {}
        self._arrange_records_size: Size | None = None
        self._arrange_records_root: Widget | None = None
        # Set by a caller for the next `_arrange_root` call, which consumes it
        self._reuse_next_arrangement = False
        # The map the last reflow produced, which Show is reported against
        self._reflowed_map: CompositorMap = {}

    def clear(self) -> None:
        \"\"\"Remove all references to widgets (used when the screen closes).\"\"\""""

# _compositor.py: Compositor._regions_to_spans, Compositor.clear
_REFLOW_4_OLD = """\
        self._visible_widgets = None
        self._layers_visible = None

    @classmethod"""

_REFLOW_4_NEW = """\
        self._visible_widgets = None
        self._layers_visible = None
        self._arrange_records = {}
        self._arrange_records_root = None
        self._reflowed_map = {}

    @classmethod"""

# _compositor.py: Compositor.__rich_repr__, Compositor.reflow
_REFLOW_5_OLD = """\
        yield "widgets", self.widgets

    def reflow(self, parent: Widget, size: Size) -> ReflowResult:
        \"\"\"Reflow (layout) widget and its children.
"""

_REFLOW_5_NEW = """\
        yield "widgets", self.widgets

    def reflow(self, parent: Widget, size: Size, reuse: bool = False) -> ReflowResult:
        \"\"\"Reflow (layout) widget and its children.
"""

# _compositor.py: Compositor.reflow
_REFLOW_6_OLD = """\
        Args:
            parent: The root widget.
            size: Size of the area to be filled.

        Returns:
            Hidden, shown, and resized widgets."""

_REFLOW_6_NEW = """\
        Args:
            parent: The root widget.
            size: Size of the area to be filled.
            reuse: Replay the placements of subtrees unchanged since the last full arrangement.

        Returns:
            Hidden, shown, and resized widgets."""

# _compositor.py: Compositor.reflow
_REFLOW_7_OLD = """\
        old_widgets = old_map.keys()

        map, widgets = self._arrange_root(parent, size, visible_only=False)

        new_widgets = map.keys()

        # Newly visible widgets
        shown_widgets = new_widgets - old_widgets
"""

_REFLOW_7_NEW = """\
        old_widgets = old_map.keys()

        self._reuse_next_arrangement = reuse
        state = self._arrangement_state(parent) if reuse and _VERIFY_REUSE else None
        map, widgets = self._arrange_root(parent, size, visible_only=False)
        if state is not None:
            self._verify_reuse(parent, size, map, widgets, state)

        new_widgets = map.keys()

        # Newly visible widgets: those the last reflow did not place. A geometry lookup since
        # then may have rebuilt the full map with them already in it, and compared with that
        # map they would never get Show, or the Resize a shown widget gets.
        shown_widgets = new_widgets - self._reflowed_map.keys()
        self._reflowed_map = map
"""

# _compositor.py: Compositor.reflow
_REFLOW_8_OLD = """\
        # Replace map and widgets
        self._full_map = map
        self.widgets = widgets

        # Contains widgets + geometry for every widget that changed (added, removed, or updated)
        changes = map.items() ^ old_map.items()

        # Widgets in both new and old
        common_widgets = old_widgets & new_widgets

        # Mark dirty regions.
        screen_region = size.region
        if screen_region not in self._dirty_regions:
            regions = {
                region
                for region in (
                    map_geometry.clip.intersection(map_geometry.region)
                    for _, map_geometry in changes
                )
                if region
            }
            self._dirty_regions.update(regions)

        resized_widgets = {
            widget
            for widget, (region, *_) in changes
            if (widget in common_widgets and old_map[widget].region.size != region.size)
        }
        return ReflowResult(
            hidden=hidden_widgets,"""

_REFLOW_8_NEW = """\
        # Replace map and widgets
        self._full_map = map
        self.widgets = widgets

        # Compare each widget's geometry with its previous one (the symmetric difference of
        # the two maps' items). Replayed placements are shared objects, so most unchanged
        # widgets are settled by identity.
        screen_region = size.region
        mark_dirty = screen_region not in self._dirty_regions
        dirty_regions: set[Region] = set()
        add_dirty_region = dirty_regions.add
        resized_widgets: set[Widget] = set()
        add_resized_widget = resized_widgets.add
        get_old_geometry = old_map.get
        intersection = Region.intersection
        for widget, geometry in map.items():
            old_geometry = get_old_geometry(widget)
            if old_geometry is geometry:
                continue
            if old_geometry is not None:
                if old_geometry == geometry:
                    continue
                old_region = old_geometry[0]
                region = geometry[0]
                if old_region[2] != region[2] or old_region[3] != region[3]:
                    add_resized_widget(widget)
                if mark_dirty:
                    add_dirty_region(intersection(old_geometry[2], old_region))
            if mark_dirty:
                add_dirty_region(intersection(geometry[2], geometry[0]))
        if mark_dirty:
            for widget in old_widgets - new_widgets:
                old_geometry = old_map[widget]
                add_dirty_region(intersection(old_geometry[2], old_geometry[0]))
            self._dirty_regions.update(region for region in dirty_regions if region)

        return ReflowResult(
            hidden=hidden_widgets,"""

# _compositor.py: Compositor.full_map
_REFLOW_9_OLD = """\
        if self._full_map_invalidated:
            self._full_map_invalidated = False
            map, _widgets = self._arrange_root(self.root, self.size, visible_only=False)
            # Update any widgets which became visible in the interim
            self._full_map = map"""

_REFLOW_9_NEW = """\
        if self._full_map_invalidated:
            self._full_map_invalidated = False
            self._reuse_next_arrangement = True
            state = self._arrangement_state(self.root) if _VERIFY_REUSE else None
            map, _widgets = self._arrange_root(self.root, self.size, visible_only=False)
            if state is not None:
                self._verify_reuse(self.root, self.size, map, _widgets, state)
            # Update any widgets which became visible in the interim
            self._full_map = map"""

# _compositor.py: Compositor._arrange, Compositor._arrange_root
_REFLOW_10_OLD = """\
        \"\"\"Arrange a widget's children based on its layout attribute.

        Args:
            root: Top level widget.
            size: Size of visible area (screen).
            visible_only: Only update visible widgets (used in scrolling).

        Returns:"""

_REFLOW_10_NEW = """\
        \"\"\"Arrange a widget's children based on its layout attribute.

        A full arrangement records its placements. It replays those of subtrees
        unchanged since the last one only if its caller set `_reuse_next_arrangement`,
        which this call consumes.

        Args:
            root: Top level widget.
            size: Size of visible area (screen).
            visible_only: Only update visible widgets (used in scrolling).

        Returns:
            Compositor map and set of widgets.
        \"\"\"
        reuse = self._reuse_next_arrangement
        self._reuse_next_arrangement = False
        return self._arrange(
            root, size, visible_only, reuse=reuse, record=not visible_only
        )

    def _arrange(
        self,
        root: Widget,
        size: Size,
        visible_only: bool,
        *,
        reuse: bool,
        record: bool,
    ) -> tuple[CompositorMap, set[Widget]]:
        \"\"\"Arrange a widget's children (see `_arrange_root`).

        Args:
            root: Top level widget.
            size: Size of visible area (screen).
            visible_only: Only update visible widgets (used in scrolling).
            reuse: Replay the recorded placements of subtrees that have not changed
                since they were recorded (full arrangements only).
            record: Record placements so a later arrangement may reuse them
                (full arrangements only).

        Returns:"""

# _compositor.py: Compositor._arrange
_REFLOW_11_OLD = """\
        no_clip = size.region

        def add_widget(
            widget: Widget,"""

_REFLOW_11_NEW = """\
        no_clip = size.region

        records: dict[Widget, _ArrangeRecord] | None = None
        old_records: dict[Widget, _ArrangeRecord] | None = None
        epoch = 0
        if record:
            DOMNode._layout_epoch += 1
            epoch = DOMNode._layout_epoch
            records = {}
            if (
                reuse
                and self._arrange_records_size == size
                and self._arrange_records_root is root
            ):
                old_records = self._arrange_records

        def replay(widget: Widget, arrange_record: _ArrangeRecord) -> None:
            \"\"\"Re-emit the recorded placements of an unchanged subtree.\"\"\"
            records[widget] = arrange_record
            _, _, visible, arranged_widgets, cover_entry, children, entries, *_ = (
                arrange_record
            )
            if visible:
                add_new_widget(widget)
            else:
                add_new_invisible_widget(widget)
            if arranged_widgets is not None:
                widgets.update(arranged_widgets)
            if cover_entry is not None:
                map[cover_entry[0]] = cover_entry[1]
            for child in children:
                replay(child, old_records[child])
            for chrome_widget, geometry in entries:
                map[chrome_widget] = geometry

        def replay_moved(
            widget: Widget,
            arrange_record: _ArrangeRecord,
            dx: int,
            dy: int,
            arrange_args: tuple,
            _MapGeometry: type[MapGeometry] = MapGeometry,
            _Region: type[Region] = Region,
            _new: Callable[..., tuple] = tuple.__new__,
        ) -> None:
            \"\"\"Re-emit an unchanged subtree whose position or clip changed.

            Every region within the subtree moves by `(dx, dy)`; clips are re-derived
            from the new clip and the (moved) regions that clip children.
            \"\"\"
            # Regions and geometries are built with `tuple.__new__`, which skips the
            # named tuples' Python-level constructors: this runs for every moved widget.
            (
                record_epoch,
                _,
                visible,
                arranged_widgets,
                cover_entry,
                children,
                entries,
                child_region,
                overlay_children,
                translatable,
            ) = arrange_record
            clip = arrange_args[4]
            if visible:
                add_new_widget(widget)
            else:
                add_new_invisible_widget(widget)
            if arranged_widgets is not None:
                widgets.update(arranged_widgets)
            if cover_entry is not None:
                cover_widget, (
                    (x, y, width, height),
                    order,
                    _,
                    virtual_size,
                    container_size,
                    virtual_region,
                    dock_gutter,
                ) = cover_entry
                geometry = _new(
                    _MapGeometry,
                    (
                        _new(_Region, (x + dx, y + dy, width, height)),
                        order,
                        clip,
                        virtual_size,
                        container_size,
                        virtual_region,
                        dock_gutter,
                    ),
                )
                cover_entry = (cover_widget, geometry)
                map[cover_widget] = geometry
            if child_region is not None:
                x, y, width, height = child_region
                child_region = _new(_Region, (x + dx, y + dy, width, height))
                sub_clip = clip.intersection(child_region)
                for child in children:
                    child_record = old_records[child]
                    (
                        child_virtual_region,
                        (x, y, width, height),
                        child_order,
                        child_layer_order,
                        _,
                        child_visible,
                        child_dock_gutter,
                        child_inherited_layers,
                    ) = child_record[1]
                    replay_moved(
                        child,
                        child_record,
                        dx,
                        dy,
                        (
                            child_virtual_region,
                            _new(_Region, (x + dx, y + dy, width, height)),
                            child_order,
                            child_layer_order,
                            no_clip if child in overlay_children else sub_clip,
                            child_visible,
                            child_dock_gutter,
                            child_inherited_layers,
                        ),
                    )
            if entries:
                # Chrome (scrollbars) first, the widget itself last; chrome regions are
                # absolute, the widget's virtual region is relative to its container.
                last = len(entries) - 1
                moved_entries: list[tuple[Widget, MapGeometry]] = []
                for index, (
                    chrome_widget,
                    (
                        (x, y, width, height),
                        order,
                        _,
                        virtual_size,
                        container_size,
                        virtual_region,
                        dock_gutter,
                    ),
                ) in enumerate(entries):
                    if index != last:
                        vx, vy, vwidth, vheight = virtual_region
                        virtual_region = _new(
                            _Region, (vx + dx, vy + dy, vwidth, vheight)
                        )
                    moved = _new(
                        _MapGeometry,
                        (
                            _new(_Region, (x + dx, y + dy, width, height)),
                            order,
                            clip,
                            virtual_size,
                            container_size,
                            virtual_region,
                            dock_gutter,
                        ),
                    )
                    map[chrome_widget] = moved
                    moved_entries.append((chrome_widget, moved))
                entries = moved_entries
            records[widget] = (
                record_epoch,
                arrange_args,
                visible,
                arranged_widgets,
                cover_entry,
                children,
                entries,
                child_region,
                overlay_children,
                translatable,
            )

        def add_widget(
            widget: Widget,"""

# _compositor.py: Compositor._arrange
_REFLOW_12_OLD = """\
            dock_gutter: Spacing,
            _MapGeometry: type[MapGeometry] = MapGeometry,
        ) -> None:
            \"\"\"Called recursively to place a widget and its children in the map.
"""

_REFLOW_12_NEW = """\
            dock_gutter: Spacing,
            inherited_layers: tuple[str, ...] | None,
            _MapGeometry: type[MapGeometry] = MapGeometry,
        ) -> bool:
            \"\"\"Called recursively to place a widget and its children in the map.
"""

# _compositor.py: Compositor._arrange
_REFLOW_13_OLD = """\
                visible: Whether the widget should be visible by default.
                    This may be overridden by the CSS rule `visibility`.
            \"\"\"
            if not widget._is_mounted:
                return
            styles = widget.styles
"""

_REFLOW_13_NEW = """\
                visible: Whether the widget should be visible by default.
                    This may be overridden by the CSS rule `visibility`.
                inherited_layers: The outermost `layers` declaration among the widget's
                    ancestors, or `None` if none declares any.

            Returns:
                `True` if the widget was placed, `False` if it isn't mounted.
            \"\"\"
            if not widget._is_mounted:
                return False

            if records is not None:
                arrange_args = (
                    virtual_region,
                    region,
                    order,
                    layer_order,
                    clip,
                    visible,
                    dock_gutter,
                    inherited_layers,
                )
                if old_records is not None:
                    arrange_record = old_records.get(widget)
                    if (
                        arrange_record is not None
                        and widget._layout_dirty_epoch < arrange_record[0]
                    ):
                        old_args = arrange_record[1]
                        if old_args == arrange_args:
                            replay(widget, arrange_record)
                            return True
                        old_region = old_args[1]
                        if (
                            arrange_record[9]
                            and old_region.size == region.size
                            and old_args[0] == virtual_region
                            and old_args[2] == order
                            and old_args[3] == layer_order
                            and old_args[5] == visible
                            and old_args[6] == dock_gutter
                            and old_args[7] == inherited_layers
                        ):
                            replay_moved(
                                widget,
                                arrange_record,
                                region[0] - old_region[0],
                                region[1] - old_region[1],
                                arrange_args,
                            )
                            return True

            styles = widget.styles
"""

# _compositor.py: Compositor._arrange
_REFLOW_14_OLD = """\
            else:
                add_new_invisible_widget(widget)

            # Container region is minus border"""

_REFLOW_14_NEW = """\
            else:
                add_new_invisible_widget(widget)

            arranged_widgets: set[Widget] | None = None
            cover_entry: tuple[Widget, MapGeometry] | None = None
            children: list[Widget] = []
            entries: list[tuple[Widget, MapGeometry]] = []
            clip_region: Region | None = None
            overlay_children: frozenset[Widget] = frozenset()
            translatable = True

            # Container region is minus border"""

# _compositor.py: Compositor._arrange
_REFLOW_15_OLD = """\
                            )
                        )
                        widget.set_reactive(Widget.scroll_y, new_scroll_y)
                        widget.set_reactive(Widget.scroll_target_y, new_scroll_y)"""

_REFLOW_15_NEW = """\
                            )
                        )
                        if new_scroll_y != widget.scroll_y:
                            # Placements beneath this widget move with its scroll offset.
                            widget._mark_layout_dirty()
                        widget.set_reactive(Widget.scroll_y, new_scroll_y)
                        widget.set_reactive(Widget.scroll_target_y, new_scroll_y)"""

# _compositor.py: Compositor._arrange
_REFLOW_16_OLD = """\
                    placement_scroll_offset = placement_offset - widget.scroll_offset

                    placements = [
                        placement.process_offset(size.region, placement_scroll_offset)
                        for placement in placements
                    ]

                    layers_to_index = {"""

_REFLOW_16_NEW = """\
                    placement_scroll_offset = placement_offset - widget.scroll_offset

                    if records is None:
                        placements = [
                            placement.process_offset(
                                size.region, placement_scroll_offset
                            )
                            for placement in placements
                        ]
                    else:
                        clip_region = child_region
                        processed_placements: list[WidgetPlacement] = []
                        for placement in placements:
                            placement_widget = placement.widget
                            if (
                                placement_widget.absolute_offset is not None
                                or placement_widget.styles.has_any_rules(
                                    "constrain_x", "constrain_y"
                                )
                            ):
                                # Placed relative to the screen: moving this subtree
                                # would not simply move the placement.
                                translatable = False
                                placement = placement.process_offset(
                                    size.region, placement_scroll_offset
                                )
                            processed_placements.append(placement)
                        placements = processed_placements

                    # `Widget.layers` takes the outermost declaration, which a replayed
                    # subtree inherits without being stamped when it changes.
                    children_inherited_layers = inherited_layers
                    if children_inherited_layers is None and styles.has_rule("layers"):
                        children_inherited_layers = styles.layers

                    layers_to_index = {"""

# _compositor.py: Compositor._arrange
_REFLOW_17_OLD = """\

                    if widget._cover_widget is not None:
                        map[widget._cover_widget] = _MapGeometry(
                            region.shrink(widget.styles.gutter),
                            order,
                            clip,
                            region.size,
                            container_size,
                            virtual_region,
                            dock_gutter,
                        )

                    # Add all the widgets"""

_REFLOW_17_NEW = """\

                    if widget._cover_widget is not None:
                        cover_entry = (
                            widget._cover_widget,
                            _MapGeometry(
                                region.shrink(widget.styles.gutter),
                                order,
                                clip,
                                region.size,
                                container_size,
                                virtual_region,
                                dock_gutter,
                            ),
                        )
                        map[cover_entry[0]] = cover_entry[1]

                    # Add all the widgets"""

# _compositor.py: Compositor._arrange
_REFLOW_18_OLD = """\

                        if widget._cover_widget is None:
                            add_widget(
                                sub_widget,
                                sub_region,"""

_REFLOW_18_NEW = """\

                        if widget._cover_widget is None:
                            if add_widget(
                                sub_widget,
                                sub_region,"""

# _compositor.py: Compositor._arrange
_REFLOW_19_OLD = """\
                                visible,
                                arrange_result.scroll_spacing,
                            )
                        layer_order -= 1
                else:"""

_REFLOW_19_NEW = """\
                                visible,
                                arrange_result.scroll_spacing,
                                children_inherited_layers,
                            ):
                                children.append(sub_widget)
                                if overlay:
                                    overlay_children = overlay_children | {sub_widget}
                        layer_order -= 1
                else:"""

# _compositor.py: Compositor._arrange
_REFLOW_20_OLD = """\
                            container_region
                        ):
                            map[chrome_widget] = _MapGeometry(
                                chrome_region,
                                order,
                                clip,
                                container_size,
                                container_size,
                                chrome_region,
                                dock_gutter,
                            )

                    map[widget._render_widget] = _MapGeometry(
                        region,
                        order,
                        clip,
                        total_region.size,
                        container_size,
                        virtual_region,
                        dock_gutter,
                    )

            elif visible:
                # Add the widget to the map
                map[widget._render_widget] = _MapGeometry(
                    region,
                    order,
                    clip,
                    region.size,
                    container_size,
                    virtual_region,
                    dock_gutter,
                )

        # Add top level (root) widget"""

_REFLOW_20_NEW = """\
                            container_region
                        ):
                            entries.append(
                                (
                                    chrome_widget,
                                    _MapGeometry(
                                        chrome_region,
                                        order,
                                        clip,
                                        container_size,
                                        container_size,
                                        chrome_region,
                                        dock_gutter,
                                    ),
                                )
                            )

                    entries.append(
                        (
                            widget._render_widget,
                            _MapGeometry(
                                region,
                                order,
                                clip,
                                total_region.size,
                                container_size,
                                virtual_region,
                                dock_gutter,
                            ),
                        )
                    )

            elif visible:
                # Add the widget to the map
                entries.append(
                    (
                        widget._render_widget,
                        _MapGeometry(
                            region,
                            order,
                            clip,
                            region.size,
                            container_size,
                            virtual_region,
                            dock_gutter,
                        ),
                    )
                )

            for chrome_widget, geometry in entries:
                map[chrome_widget] = geometry

            if records is not None:
                if translatable:
                    for child in children:
                        if not records[child][9]:
                            translatable = False
                            break
                records[widget] = (
                    epoch,
                    arrange_args,
                    visible,
                    arranged_widgets,
                    cover_entry,
                    children,
                    entries,
                    clip_region,
                    overlay_children,
                    translatable,
                )
            return True

        root_inherited_layers: tuple[str, ...] | None = None
        ancestor = root._parent
        while isinstance(ancestor, Widget):
            if ancestor.styles.has_rule("layers"):
                root_inherited_layers = ancestor.styles.layers
            ancestor = ancestor._parent

        # Add top level (root) widget"""

# _compositor.py: Compositor._arrange, Compositor._arrangement_state, Compositor._restore_arrangement_state, Compositor._verify_reuse, Compositor.layers
_REFLOW_21_OLD = """\
            NULL_SPACING,
        )
        widgets -= invisible_widgets
        return map, widgets

    @property"""

_REFLOW_21_NEW = """\
            NULL_SPACING,
            root_inherited_layers,
        )
        # The nested functions refer to themselves through closure cells; clearing the
        # cells breaks that cycle, so this arrangement's maps and records are freed by
        # reference counting instead of waiting for the cyclic garbage collector.
        del add_widget, replay, replay_moved
        widgets -= invisible_widgets
        if records is not None:
            self._arrange_records = records
            self._arrange_records_size = size
            self._arrange_records_root = root
        return map, widgets

    @staticmethod
    def _arrangement_state(
        root: Widget,
    ) -> list[tuple[DOMNode, tuple[str, ...], dict[str, object]]]:
        \"\"\"Snapshot the state an arrangement writes (debug aid).

        Scroll offsets, anchoring and container sizes: anchoring and `arrange` overrides
        change them, so a second arrangement only matches the first when it starts from the
        same state. And every node's layout epoch, which the watchers of the offsets an
        arrangement sets stamp, from the node they refresh up to the App.
        \"\"\"
        widget_names = (
            "_reactive_scroll_x",
            "_reactive_scroll_y",
            "_reactive_scroll_target_x",
            "_reactive_scroll_target_y",
            "_anchored_flag",
            "_anchor_released_flag",
            "_container_size",
            "_layout_dirty_epoch",
        )
        scrollbar_names = ("_reactive_position", "_layout_dirty_epoch")
        epoch_names = ("_layout_dirty_epoch",)
        nodes: list[tuple[DOMNode, tuple[str, ...]]] = [
            (ancestor, epoch_names) for ancestor in root.ancestors
        ]
        for node in root.walk_children(with_self=True):
            if not isinstance(node, Widget):
                nodes.append((node, epoch_names))
                continue
            nodes.append((node, widget_names if node.is_scrollable else epoch_names))
            nodes.extend(
                (scrollbar, scrollbar_names)
                for scrollbar in (node._vertical_scrollbar, node._horizontal_scrollbar)
                if scrollbar is not None
            )
            if node._scrollbar_corner is not None:
                nodes.append((node._scrollbar_corner, epoch_names))
        return [
            (
                node,
                names,
                {name: vars(node)[name] for name in names if name in vars(node)},
            )
            for node, names in nodes
        ]

    @staticmethod
    def _restore_arrangement_state(
        state: list[tuple[DOMNode, tuple[str, ...], dict[str, object]]],
    ) -> None:
        \"\"\"Restore a snapshot taken by `_arrangement_state` (debug aid).\"\"\"
        for node, names, values in state:
            namespace = vars(node)
            for name in names:
                if name in values:
                    namespace[name] = values[name]
                else:
                    namespace.pop(name, None)

    def _verify_reuse(
        self,
        root: Widget,
        size: Size,
        map: CompositorMap,
        widgets: set[Widget],
        state: list[tuple[DOMNode, tuple[str, ...], dict[str, object]]],
    ) -> None:
        \"\"\"Check a reusing arrangement against a from-scratch one (debug aid).

        The from-scratch arrangement starts from the state the reusing one started from. Both
        must place the same widgets and leave the same scroll, anchor and container state,
        which the map does not show for a widget that scrolls its own content. The state the
        reusing one left is restored afterwards, layout epochs included: the from-scratch
        arrangement's watchers stamp the widgets they refresh, which would invalidate the
        records under check and hide a change that was not stamped.
        \"\"\"
        after = self._arrangement_state(root)
        self._restore_arrangement_state(state)
        try:
            full_map, full_widgets = self._arrange(
                root, size, False, reuse=False, record=False
            )
            full_after = self._arrangement_state(root)
        finally:
            self._restore_arrangement_state(after)
        unset = object()
        full_values = {
            (node, name): value
            for node, _, values in full_after
            for name, value in values.items()
        }
        state_changed = [
            (node, name, values.get(name, unset), full_values.get((node, name), unset))
            for node, names, values in after
            for name in names
            if name != "_layout_dirty_epoch"
            and values.get(name, unset) != full_values.get((node, name), unset)
        ]
        if (
            list(full_map.items()) != list(map.items())
            or full_widgets != widgets
            or state_changed
        ):
            missing = full_map.keys() - map.keys()
            extra = map.keys() - full_map.keys()
            changed = [
                (widget, map[widget], geometry)
                for widget, geometry in full_map.items()
                if widget in map and map[widget] != geometry
            ]
            raise AssertionError(
                "compositor reuse diverged: "
                f"missing={list(missing)[:5]} extra={list(extra)[:5]} "
                f"changed={changed[:3]} "
                f"order_equal={list(full_map) == list(map)} "
                f"widgets_missing={list(full_widgets - widgets)[:5]} "
                f"widgets_extra={list(widgets - full_widgets)[:5]} "
                f"state_changed={state_changed[:3]}"
            )

    @property"""

# _node_list.py: NodeList.updated
_REFLOW_22_OLD = """\
        self._updates += 1
        node = None if self._parent is None else self._parent()
        while node is not None and (node := node._parent) is not None:
            node._nodes._updates += 1"""

_REFLOW_22_NEW = """\
        self._updates += 1
        node = None if self._parent is None else self._parent()
        if node is not None:
            node._mark_layout_dirty()
        while node is not None and (node := node._parent) is not None:
            node._nodes._updates += 1"""

# dom.py: DOMNode._PSEUDO_CLASSES, DOMNode.__init__, DOMNode._layout_dirty_epoch, DOMNode._layout_epoch, DOMNode._mark_layout_dirty, DOMNode._on_mounted_state
_REFLOW_23_OLD = """\
    _PSEUDO_CLASSES: ClassVar[dict[str, Callable[[App[Any]], bool]]] = {}
    \"\"\"Pseudo class checks.\"\"\"

    def __init__("""

_REFLOW_23_NEW = """\
    _PSEUDO_CLASSES: ClassVar[dict[str, Callable[[App[Any]], bool]]] = {}
    \"\"\"Pseudo class checks.\"\"\"

    _layout_epoch: ClassVar[int] = 1
    \"\"\"Global arrangement epoch, advanced by every full compositor arrangement.\"\"\"

    _layout_dirty_epoch: int = 0
    \"\"\"The epoch in which this node, or a node beneath it, last changed in a way the compositor reads.\"\"\"

    def _mark_layout_dirty(self) -> None:
        \"\"\"Record that this node's compositor placement may have changed.

        Stamps the node and its ancestors with the current epoch, stopping at the first
        ancestor already stamped (whose own ancestors are then stamped too).
        \"\"\"
        epoch = DOMNode._layout_epoch
        node: DOMNode | None = self
        while node is not None and node._layout_dirty_epoch != epoch:
            node._layout_dirty_epoch = epoch
            node = node._parent

    def _on_mounted_state(self) -> None:
        # The compositor skips unmounted widgets, so mounting changes placement.
        self._mark_layout_dirty()

    def __init__("""

# message_pump.py: MessagePump._on_mounted_state, MessagePump._post_mount, MessagePump._pre_process
_REFLOW_24_OLD = """\
            self._mounted_event.set()
            self._is_mounted = True
        return True

    def _post_mount(self):"""

_REFLOW_24_NEW = """\
            self._mounted_event.set()
            self._is_mounted = True
            self._on_mounted_state()
        return True

    def _on_mounted_state(self) -> None:
        \"\"\"Called once `_is_mounted` becomes `True`.\"\"\"

    def _post_mount(self):"""

# message_pump.py: MessagePump._process_messages_loop
_REFLOW_25_OLD = """\
                self._mounted_event.set()
                self._is_mounted = True
                self.app._handle_exception(error)
                break"""

_REFLOW_25_NEW = """\
                self._mounted_event.set()
                self._is_mounted = True
                self._on_mounted_state()
                self.app._handle_exception(error)
                break"""

# screen.py: Screen._refresh_layout
_REFLOW_26_OLD = """\

            else:
                hidden, shown, resized = self._compositor.reflow(self, size)
                self._layout_widgets.clear()
                Hide = events.Hide"""

_REFLOW_26_NEW = """\

            else:
                hidden, shown, resized = self._compositor.reflow(
                    self, size, reuse=True
                )
                self._layout_widgets.clear()
                Hide = events.Hide"""

# widget.py: Widget._absolute_offset, Widget._anchor_released, Widget._anchor_released_flag, Widget._anchored, Widget._anchored_flag, Widget._cover, Widget.absolute_offset, Widget.pre_render
_REFLOW_27_OLD = """\
        \"\"\"
        self._visual_style = None

    def _cover(self, widget: Widget) -> None:"""

_REFLOW_27_NEW = """\
        \"\"\"
        self._visual_style = None

    # Backing fields of the properties below, which stamp the layout epoch on change.
    _absolute_offset: Offset | None = None
    _anchored_flag: bool = False
    _anchor_released_flag: bool = False

    @property
    def absolute_offset(self) -> Offset | None:
        \"\"\"Force an absolute offset for the widget (used by tooltips).\"\"\"
        return self._absolute_offset

    @absolute_offset.setter
    def absolute_offset(self, offset: Offset | None) -> None:
        if offset != self._absolute_offset:
            self._absolute_offset = offset
            self._mark_layout_dirty()

    @property
    def _anchored(self) -> bool:
        \"\"\"Has this widget been anchored?\"\"\"
        return self._anchored_flag

    @_anchored.setter
    def _anchored(self, anchored: bool) -> None:
        if anchored != self._anchored_flag:
            self._anchored_flag = anchored
            self._mark_layout_dirty()

    @property
    def _anchor_released(self) -> bool:
        \"\"\"Has the anchor been released?\"\"\"
        return self._anchor_released_flag

    @_anchor_released.setter
    def _anchor_released(self, released: bool) -> None:
        if released != self._anchor_released_flag:
            self._anchor_released_flag = released
            self._mark_layout_dirty()

    def _cover(self, widget: Widget) -> None:"""

# widget.py: Widget._set_dirty
_REFLOW_28_OLD = """\
            self._dirty_regions.clear()
            self._repaint_regions.clear()
            self._styles_cache.clear()
            self._styles_cache.set_dirty(self.size.region)
            outer_size = self.outer_size
            self._dirty_regions.add(outer_size.region)"""

_REFLOW_28_NEW = """\
            self._dirty_regions.clear()
            self._repaint_regions.clear()
            # A cleared cache re-renders every line; marking lines dirty as well would
            # only cost a compositor lookup (`self.size`), which may rebuild the full map.
            self._styles_cache.clear()
            outer_size = self.outer_size
            self._dirty_regions.add(outer_size.region)"""

# widget.py: Widget.scroll_to, Widget.set_scroll
_REFLOW_29_OLD = """\
        if y is not None:
            self.set_reactive(Widget.scroll_y, round(y))

    def scroll_to("""

_REFLOW_29_NEW = """\
        if y is not None:
            self.set_reactive(Widget.scroll_y, round(y))
        self._mark_layout_dirty()

    def scroll_to("""

# widget.py: Widget._refresh_scroll
_REFLOW_30_OLD = """\
        \"\"\"Refreshes the scroll position.\"\"\"
        self._scroll_required = True
        self.check_idle()
"""

_REFLOW_30_NEW = """\
        \"\"\"Refreshes the scroll position.\"\"\"
        self._scroll_required = True
        self._mark_layout_dirty()
        self.check_idle()
"""

# widget.py: Widget.refresh
_REFLOW_31_OLD = """\
        \"\"\"

        if layout and not self._layout_required:
            self._layout_required = True"""

_REFLOW_31_NEW = """\
        \"\"\"

        self._mark_layout_dirty()
        if layout and not self._layout_required:
            self._layout_required = True"""

# widget.py: Widget._check_refresh
_REFLOW_32_OLD = """\
                if self._layout_required:
                    self._layout_required = False
                    for ancestor in self.ancestors:
                        if not isinstance(ancestor, Widget):"""

_REFLOW_32_NEW = """\
                if self._layout_required:
                    self._layout_required = False
                    self._mark_layout_dirty()
                    for ancestor in self.ancestors:
                        if not isinstance(ancestor, Widget):"""

# css/styles.py: Styles.clear_rule, Styles.get_rules
_REFLOW_33_OLD = """\
        changed = self._rules.pop(rule_name, None) is not None  # type: ignore
        if changed:
            self._updates += 1
        return changed

    def get_rules(self) -> RulesMap:"""

_REFLOW_33_NEW = """\
        changed = self._rules.pop(rule_name, None) is not None  # type: ignore
        if changed:
            self._updates += 1
            if self.node is not None:
                # A raw rule write skips the refresh a style property makes.
                self.node._mark_layout_dirty()
        return changed

    def get_rules(self) -> RulesMap:"""

# css/styles.py: Styles.set_rule
_REFLOW_34_OLD = """\
        if value is None:
            changed = self._rules.pop(rule, None) is not None  # type: ignore
            if changed:
                self._updates += 1
            return changed
        current = self._rules.get(rule)
        self._rules[rule] = value  # type: ignore
        changed = current != value
        if changed:
            self._updates += 1
        return changed
"""

_REFLOW_34_NEW = """\
        if value is None:
            changed = self._rules.pop(rule, None) is not None  # type: ignore
        else:
            current = self._rules.get(rule)
            self._rules[rule] = value  # type: ignore
            changed = current != value
        if changed:
            self._updates += 1
            if self.node is not None:
                # A raw rule write skips the refresh a style property makes.
                self.node._mark_layout_dirty()
        return changed
"""

# widget.py: Widget._size_updated
_REFLOW_35_OLD = """\
            else:
                self.set_reactive(Widget.virtual_size, virtual_size)
            self._container_size = container_size"""

_REFLOW_35_NEW = """\
            else:
                self.set_reactive(Widget.virtual_size, virtual_size)
            if (
                self._container_size != container_size
                and self._anchored
                and not self._anchor_released
            ):
                # An anchored widget is pinned from its container size, which is only
                # known after the arrangement that pinned it.
                self._mark_layout_dirty()
            self._container_size = container_size"""

# scroll_view.py: ScrollView._size_updated
_REFLOW_36_OLD = """\
            virtual_size = self.virtual_size
            self._container_size = size - self.styles.gutter.totals"""

_REFLOW_36_NEW = """\
            virtual_size = self.virtual_size
            new_container_size = size - self.styles.gutter.totals
            if (
                self._container_size != new_container_size
                and self._anchored
                and not self._anchor_released
            ):
                # An anchored widget is pinned from its container size, which is only
                # known after the arrangement that pinned it.
                self._mark_layout_dirty()
            self._container_size = new_container_size"""

# reactive.py: Reactive._set
_REFLOW_37_OLD = """\
            # Store the internal value
            setattr(obj, self.internal_name, value)

            # Check all watchers
            self._check_watchers(obj, name, current_value)"""

_REFLOW_37_NEW = """\
            # Store the internal value
            setattr(obj, self.internal_name, value)

            if self._layout or self._repaint or self._recompose:
                # The refresh below would stamp the change only after the watchers, which
                # may look up geometry and so rebuild the full map from records.
                obj._mark_layout_dirty()

            # Check all watchers
            self._check_watchers(obj, name, current_value)"""

_PATCHES = (
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_1_OLD,
        new_fragment=_REFLOW_1_NEW,
        description="Reuse unchanged arrangements: <import>",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_2_OLD,
        new_fragment=_REFLOW_2_NEW,
        description="Reuse unchanged arrangements: <if>, _ArrangeRecord=, _VERIFY_REUSE=",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_3_OLD,
        new_fragment=_REFLOW_3_NEW,
        description="Reuse unchanged arrangements: Compositor.__init__, Compositor.clear",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_4_OLD,
        new_fragment=_REFLOW_4_NEW,
        description="Reuse unchanged arrangements: Compositor._regions_to_spans, Compositor.clear",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_5_OLD,
        new_fragment=_REFLOW_5_NEW,
        description="Reuse unchanged arrangements: Compositor.__rich_repr__, Compositor.reflow",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_6_OLD,
        new_fragment=_REFLOW_6_NEW,
        description="Reuse unchanged arrangements: Compositor.reflow",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_7_OLD,
        new_fragment=_REFLOW_7_NEW,
        description="Reuse unchanged arrangements: Compositor.reflow",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_8_OLD,
        new_fragment=_REFLOW_8_NEW,
        description="Reuse unchanged arrangements: Compositor.reflow",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_9_OLD,
        new_fragment=_REFLOW_9_NEW,
        description="Reuse unchanged arrangements: Compositor.full_map",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_10_OLD,
        new_fragment=_REFLOW_10_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange, Compositor._arrange_root",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_11_OLD,
        new_fragment=_REFLOW_11_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_12_OLD,
        new_fragment=_REFLOW_12_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_13_OLD,
        new_fragment=_REFLOW_13_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_14_OLD,
        new_fragment=_REFLOW_14_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_15_OLD,
        new_fragment=_REFLOW_15_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_16_OLD,
        new_fragment=_REFLOW_16_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_17_OLD,
        new_fragment=_REFLOW_17_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_18_OLD,
        new_fragment=_REFLOW_18_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_19_OLD,
        new_fragment=_REFLOW_19_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_20_OLD,
        new_fragment=_REFLOW_20_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_REFLOW_21_OLD,
        new_fragment=_REFLOW_21_NEW,
        description="Reuse unchanged arrangements: Compositor._arrange, Compositor._arrangement_state, Compositor._restore_arrangement_state, Compositor._verify_reuse, Compositor.layers",
    ),
    FilePatch(
        package="textual",
        module_file="_node_list.py",
        old_fragment=_REFLOW_22_OLD,
        new_fragment=_REFLOW_22_NEW,
        description="Reuse unchanged arrangements: NodeList.updated",
    ),
    FilePatch(
        package="textual",
        module_file="dom.py",
        old_fragment=_REFLOW_23_OLD,
        new_fragment=_REFLOW_23_NEW,
        description="Reuse unchanged arrangements: DOMNode._PSEUDO_CLASSES, DOMNode.__init__, DOMNode._layout_dirty_epoch, DOMNode._layout_epoch, DOMNode._mark_layout_dirty, DOMNode._on_mounted_state",
    ),
    FilePatch(
        package="textual",
        module_file="message_pump.py",
        old_fragment=_REFLOW_24_OLD,
        new_fragment=_REFLOW_24_NEW,
        description="Reuse unchanged arrangements: MessagePump._on_mounted_state, MessagePump._post_mount, MessagePump._pre_process",
    ),
    FilePatch(
        package="textual",
        module_file="message_pump.py",
        old_fragment=_REFLOW_25_OLD,
        new_fragment=_REFLOW_25_NEW,
        description="Reuse unchanged arrangements: MessagePump._process_messages_loop",
    ),
    FilePatch(
        package="textual",
        module_file="screen.py",
        old_fragment=_REFLOW_26_OLD,
        new_fragment=_REFLOW_26_NEW,
        description="Reuse unchanged arrangements: Screen._refresh_layout",
    ),
    FilePatch(
        package="textual",
        module_file="widget.py",
        old_fragment=_REFLOW_27_OLD,
        new_fragment=_REFLOW_27_NEW,
        description="Reuse unchanged arrangements: Widget._absolute_offset, Widget._anchor_released, Widget._anchor_released_flag, Widget._anchored, Widget._anchored_flag, Widget._cover, Widget.absolute_offset, Widget.pre_render",
    ),
    FilePatch(
        package="textual",
        module_file="widget.py",
        old_fragment=_REFLOW_28_OLD,
        new_fragment=_REFLOW_28_NEW,
        description="Reuse unchanged arrangements: Widget._set_dirty",
    ),
    FilePatch(
        package="textual",
        module_file="widget.py",
        old_fragment=_REFLOW_29_OLD,
        new_fragment=_REFLOW_29_NEW,
        description="Reuse unchanged arrangements: Widget.scroll_to, Widget.set_scroll",
    ),
    FilePatch(
        package="textual",
        module_file="widget.py",
        old_fragment=_REFLOW_30_OLD,
        new_fragment=_REFLOW_30_NEW,
        description="Reuse unchanged arrangements: Widget._refresh_scroll",
    ),
    FilePatch(
        package="textual",
        module_file="widget.py",
        old_fragment=_REFLOW_31_OLD,
        new_fragment=_REFLOW_31_NEW,
        description="Reuse unchanged arrangements: Widget.refresh",
    ),
    FilePatch(
        package="textual",
        module_file="widget.py",
        old_fragment=_REFLOW_32_OLD,
        new_fragment=_REFLOW_32_NEW,
        description="Reuse unchanged arrangements: Widget._check_refresh",
    ),
    FilePatch(
        package="textual",
        module_file="css/styles.py",
        old_fragment=_REFLOW_33_OLD,
        new_fragment=_REFLOW_33_NEW,
        description="Reuse unchanged arrangements: Styles.clear_rule, Styles.get_rules",
    ),
    FilePatch(
        package="textual",
        module_file="css/styles.py",
        old_fragment=_REFLOW_34_OLD,
        new_fragment=_REFLOW_34_NEW,
        description="Reuse unchanged arrangements: Styles.set_rule",
    ),
    FilePatch(
        package="textual",
        module_file="widget.py",
        old_fragment=_REFLOW_35_OLD,
        new_fragment=_REFLOW_35_NEW,
        description="Reuse unchanged arrangements: Widget._size_updated",
    ),
    FilePatch(
        package="textual",
        module_file="scroll_view.py",
        old_fragment=_REFLOW_36_OLD,
        new_fragment=_REFLOW_36_NEW,
        description="Reuse unchanged arrangements: ScrollView._size_updated",
    ),
    FilePatch(
        package="textual",
        module_file="reactive.py",
        old_fragment=_REFLOW_37_OLD,
        new_fragment=_REFLOW_37_NEW,
        description="Reuse unchanged arrangements: Reactive._set",
    ),
)


def apply_runtime_patch() -> None:
    """Install the reusing reflow on classes already imported; on drift in any module, in none."""
    try:
        import importlib

        import textual
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual reflow reuse runtime patch: loaded Textual is not the pinned %s that the patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return
    modules = {name: importlib.import_module(name) for name in _RUNTIME_MEMBERS}
    if all(
        members_installed(modules[name], members, _RUNTIME_PATCH_MARKER) for name, members in _RUNTIME_MEMBERS.items()
    ):
        return
    try:
        # Stage every module before installing any: a compositor that reuses records while
        # widgets do not stamp their changes would paint stale placements.
        staged = [
            stage_members(
                modules[name],
                [patch for patch in _PATCHES if _module_name(patch) == name],
                members,
                label="reflow reuse",
            )
            for name, members in _RUNTIME_MEMBERS.items()
        ]
    except StagedSourceDriftError as exc:
        logger.warning("Skipping Textual reflow reuse runtime patch: %s", exc)
        return
    compositor_globals = vars(modules["textual._compositor"])
    compositor_globals.setdefault("DOMNode", vars(modules["textual.dom"])["DOMNode"])
    compositor_globals.setdefault("_VERIFY_REUSE", False)
    for members in staged:
        members.install(_RUNTIME_PATCH_MARKER)


def reflow_reusing_records(compositor: Compositor, root: Widget, size: Size) -> ReflowResult:
    """``compositor.reflow(root, size)`` that replays every subtree unchanged since the last full arrangement.

    For a caller that resynchronizes the compositor outside a layout pass after changes that stamp
    their widgets (a raw style-rule write does): only the stamped widgets' ancestor paths are
    arranged again, instead of every displayed widget. Without the installed patch (another
    Textual version) this is the stock reflow, which arranges the whole tree.
    """
    if _reflow_reuses_records(compositor):
        return compositor.reflow(root, size, reuse=True)
    return compositor.reflow(root, size)


class _ReusingCompositor(Protocol):
    """A compositor whose ``reflow`` is the installed one, which Textual's own signature does not describe."""

    def reflow(self, parent: Widget, size: Size, reuse: bool = False) -> ReflowResult: ...


def _reflow_reuses_records(compositor: Compositor) -> TypeGuard[_ReusingCompositor]:
    """Whether ``compositor.reflow`` is the installed reusing reflow, which takes ``reuse``."""
    return bool(vars(type(compositor).reflow).get(_RUNTIME_PATCH_MARKER, False))


def _module_name(patch: FilePatch) -> str:
    return "textual." + patch.module_file.removesuffix(".py").replace("/", ".")
