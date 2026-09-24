# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Theme editing UI backed by detached documents, never a mutable registered Theme."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalGroup, Vertical, VerticalGroup, VerticalScroll
from textual.message import Message
from textual.theme import Theme
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Label

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog
from chrys.app.tui.theme_loader import theme_is_read_only
from chrys.app.tui.themes.document import SURFACES, EditToken, ThemeDocument, copy_theme
from chrys.app.tui.themes.store import ThemeFileRevision, ThemeStoreError, UserThemeStore
from chrys.app.tui.widgets.select import LazySelect, Select
from chrys.foundation.config.settings import DEFAULT_THEME
from chrys.foundation.i18n import MessageRef

from . import messages as M
from .dialogs import _PickerModal
from .save import SaveThemeDialog

if TYPE_CHECKING:
    from uuid import UUID

    from chrys.app.tui.app import ChrysApp
from .palette import (
    _ANSI_THEME_COLOR_NAMES,
    _CANONICAL_VARIABLE_NAMES,
    _THEME_COLOR_NAMES,
    _FlatThemeColorButton,
    _group_variables,
    _ResetButton,
    _VariableSwatchButton,
)


class ResettableThemeEditor(Widget):
    """Sidebar controller; one confirmed dialog is one history entry."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+z", "undo", show=False),
        Binding("ctrl+y", "redo", show=False),
    ]

    class CloseRequested(Message):
        pass

    def __init__(self, theme: Theme, store: UserThemeStore, revision: ThemeFileRevision | None = None) -> None:
        super().__init__()
        self.store = store
        self.revision = revision
        self._document = ThemeDocument(theme, is_new=revision is None and not theme_is_read_only(theme.name))
        self._color_buttons: dict[str, _FlatThemeColorButton] = {}
        self._variable_swatches: dict[str, _VariableSwatchButton] = {}
        self._reset_buttons: dict[str, _ResetButton] = {}
        self._preview_timer: Timer | None = None
        self._pending: EditToken | None = None
        self._switching = False
        self._recompose_pending = False
        self._confirming_switch = False
        self._confirming_delete = False
        self._deleting = False
        self._selection_error: MessageRef | None = None
        self._listed_themes: tuple[str, ...] = ()

    @property
    def host(self) -> ChrysApp:
        return cast("ChrysApp", self.app)

    @property
    def document(self) -> ThemeDocument:
        return self._document

    @property
    def _theme(self) -> Theme:
        return self.document.draft

    @property
    def _color_mapping(self) -> dict[str, str]:
        return self._theme.to_color_system().generate()

    @property
    def _pruned(self) -> bool:
        """The dock is being torn down; the controls may be gone or, after a prune that landed
        before compose (mount no-ops while pruning), may never have existed although Mount and
        its deferred refresh still ran."""
        return self._pruning or self._closing

    def compose(self) -> ComposeResult:
        self._color_buttons.clear()
        self._variable_swatches.clear()
        self._reset_buttons.clear()
        with Vertical(id="theme-editor-body") as body:
            body.border_title = M.text(self, M.TITLE)
            with HorizontalGroup(id="theme-document-controls"):
                self._listed_themes = self._theme_names()
                yield LazySelect(
                    [(Text(name), name) for name in self._listed_themes],
                    separators_before=[name for name in self._listed_themes if theme_is_read_only(name)][:1],
                    value=self._theme.name,
                    allow_blank=False,
                    prompt=M.text(self, M.CHOOSE_THEME),
                    id="theme-select",
                )
            yield Label(Text(self._meta()), id="base-theme", classes="--meta")
            with VerticalScroll(can_focus=False, id="theme-fields"):
                yield from self._compose_fields()
        yield Label(Text(""), id="editor-error")
        with HorizontalGroup(classes="theme-actions"):
            yield Button(Text(M.text(self, M.UNDO)), id="theme-undo", flat=True, variant="primary")
            yield Button(Text(M.text(self, M.REDO)), id="theme-redo", flat=True, variant="primary")
        with HorizontalGroup(classes="theme-actions"):
            yield Button(Text(M.text(self, M.RESET)), id="theme-reset", flat=True, variant="warning")
            yield Button(Text(M.text(self, M.SAVE)), id="theme-save", flat=True, variant="success")
        with HorizontalGroup(id="theme-document-actions"):
            delete = Button(Text(M.text(self, M.DELETE)), id="theme-delete", flat=True, variant="error")
            delete.display = not theme_is_read_only(self._theme.name)
            delete.disabled = self.revision is None
            yield delete
            yield Button(Text(M.text(self, M.CLOSE)), id="theme-close", flat=True, variant="warning")

    def _compose_fields(self) -> ComposeResult:
        yield Label(Text(M.COLORS), id="theme-colors-heading", classes="--group-header")
        with VerticalGroup(classes="--rows-grid"):
            for name in _ANSI_THEME_COLOR_NAMES if self._theme.ansi else _THEME_COLOR_NAMES:
                yield Label(Text(name))
                button = _FlatThemeColorButton(getattr(self._theme, name), name)
                self._color_buttons[name] = button
                yield button
                yield self._reset_button(f"color:{name}")
        for group, variables in _group_variables(list(_CANONICAL_VARIABLE_NAMES), ansi=self._theme.ansi):
            yield Label(Text(M.GROUPS[group]), id=f"theme-group-{group}", classes="--group-header")
            with VerticalGroup(classes="--rows-grid"):
                for name in variables:
                    label = Label(
                        Text(M.BUTTON_FIELDS[name] if group == "button" else name),
                        id=f"theme-label-{name}" if group == "button" else None,
                    )
                    if group == "button":
                        label.tooltip = Text(name)
                    yield label
                    swatch = _VariableSwatchButton(name, self._variable_swatch_value(name))
                    self._variable_swatches[name] = swatch
                    yield swatch
                    yield self._reset_button(f"var:{name}")

    def _reset_button(self, target: str) -> _ResetButton:
        button = _ResetButton(target)
        self._reset_buttons[target] = button
        return button

    def _meta(self) -> str:
        return M.text(
            self,
            M.META,
            name=M.text(self, M.BUILTIN_SOURCE if theme_is_read_only(self._theme.name) else M.USER_THEME),
            mode=M.text(self, M.DARK if self._theme.dark else M.LIGHT),
        )

    def _theme_names(self) -> tuple[str, ...]:
        return tuple(
            sorted({*self.app.available_themes, self._theme.name}, key=lambda name: (theme_is_read_only(name), name))
        )

    def _refresh_selector(self) -> None:
        selector = self.query_one("#theme-select", Select)
        names = self._theme_names()
        with self.prevent(Select.Changed):
            if names != self._listed_themes:
                self._listed_themes = names
                selector.set_options(
                    [(Text(name), name) for name in names],
                    separators_before=[name for name in names if theme_is_read_only(name)][:1],
                )
            selector.value = self._theme.name

    @on(Select.Changed, "#theme-select")
    def _theme_selected(self, event: Select.Changed) -> None:
        event.stop()
        if self.delete_pending or self._switching or self._confirming_switch or not isinstance(event.value, str):
            return
        if event.select is not self.query_one("#theme-select", Select) or event.value != event.select.value:
            return
        name = event.value
        if name == self._theme.name:
            return
        self._refresh_selector()
        if not self.document.unsaved:
            self.call_after_refresh(self._load_theme, name)
            return
        self._confirming_switch = True

        def decided(discard: bool | None) -> None:
            self._confirming_switch = False
            if discard and self.is_mounted:
                self.call_after_refresh(self._load_theme, name)

        self.app.push_screen(
            ConfirmDialog(
                title=M.DISCARD_TITLE.bind(),
                message=M.SWITCH_BODY.bind(),
                confirm_label=M.DISCARD.bind(),
                cancel_label=M.CANCEL.bind(),
                confirm_variant="error",
                locale_controller=self.host.locale_controller,
            ),
            decided,
        )

    async def _load_theme(self, name: str, *, allow_covered: bool = False) -> None:
        if self._switching or self._confirming_switch or not self.is_mounted:
            return
        self._switching = True
        self._selection_error = None
        try:
            self.cancel_active_dialog()
            if theme_is_read_only(name):
                source = self.app.get_theme(name)
                assert source is not None
                theme, revision = copy_theme(source), None
            else:
                theme, revision = await asyncio.to_thread(self.store.load, name)
            if not self.is_mounted or (not allow_covered and self.screen is not self.app.screen):
                return
            if not self._show(theme):
                return
            self._document.cancel()
            self._document = ThemeDocument(theme)
            self.revision = revision
            self._recompose_pending = True
            if self.screen is self.app.screen:
                await self.recompose()
                self._recompose_pending = False
        except ThemeStoreError as error:
            self._selection_error = error.display
        finally:
            self._switching = False
            if self.is_mounted and not self._recompose_pending and self.screen is self.app.screen:
                self.refresh_localization()
                self._refresh_state()
                self._display_error()
                self.focus_first_swatch()

    def resume(self) -> None:
        if self._recompose_pending:
            self.call_after_refresh(self._resume_document)
        else:
            self._refresh_state()
            self._display_error()

    async def _resume_document(self) -> None:
        """Project an already adopted document once its dock is visible again."""
        if not self._recompose_pending or self._switching or not self.is_mounted or self.screen is not self.app.screen:
            return
        self._switching = True
        try:
            await self.recompose()
            self._recompose_pending = False
        finally:
            self._switching = False
            self.disabled = self._deleting or self._recompose_pending
        self.refresh_localization()
        self._refresh_state()
        self._display_error()
        self.focus_first_swatch()

    def on_mount(self) -> None:
        self.host.locale_controller.register_surface(self)
        self.host.begin_theme_preview(self._theme)
        self.app.theme_changed_signal.subscribe(self, self._effective_theme_changed)
        self.call_later(self._refresh_state)
        self.call_after_refresh(self.focus_first_swatch)

    def refresh_localization(self) -> None:
        """Translate the existing controls without replacing the draft or focus."""
        if not self.is_mounted or self._pruned or self._switching or self._recompose_pending:
            return
        self.query_one("#theme-editor-body").border_title = Text(M.text(self, M.TITLE))
        self.query_one("#theme-select", Select).prompt = M.text(self, M.CHOOSE_THEME)
        for name, definition in (
            ("undo", M.UNDO),
            ("redo", M.REDO),
            ("reset", M.RESET),
            ("save", M.SAVE),
            ("delete", M.DELETE),
            ("close", M.CLOSE),
        ):
            self.query_one(f"#theme-{name}", Button).label = Text(M.text(self, definition))
        self.query_one("#base-theme", Label).update(Text(self._meta()))
        self._display_error()

    def _effective_theme_changed(self, _theme: Theme) -> None:
        self._display_error()

    def _show(self, theme: Theme, field: str = "") -> bool:
        preview = self.host.theme_preview
        assert preview is not None
        success = preview.show(theme, field)
        self._display_error()
        return success

    def _display_error(self) -> None:
        if not self.app.is_running:
            return
        preview = self.host.theme_preview
        if preview is None:
            return
        if (
            self.is_mounted
            and not self._pruned
            and not self._switching
            and not self._recompose_pending
            and self.screen is self.app.screen
        ):
            error = self.query_one("#editor-error", Label)
            message = (
                render_str(widget_localizer(self), self._selection_error)
                if self._selection_error is not None
                else preview.error
            )
            error.update(Text(message))
            error.display = bool(message)
        for screen in self.app.screen_stack:
            if (
                isinstance(screen, _PickerModal)
                and screen._editor is self
                and screen.is_mounted
                and not screen._transaction_closed
            ):
                screen.show_preview_error(preview.error)

    def preview_edit(self, token: EditToken, value: str | None) -> None:
        if self.document.stage(token, value) is None:
            return
        self._pending = token
        if self._preview_timer is None:
            self._preview_timer = self.set_timer(0.1, self._flush_preview)

    def _flush_preview(self) -> None:
        self._preview_timer = None
        token, self._pending = self._pending, None
        if token is not None and self.document.owns(token):
            assert self.document.transaction is not None
            self._show(self.document.transaction.candidate, token.field)

    def _clear_pending(self) -> None:
        self._pending = None
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None

    def restore_transaction(self, token: EditToken) -> None:
        if not self.document.owns(token):
            return
        assert self.document.transaction is not None
        self._clear_pending()
        self.document.transaction.candidate = copy_theme(self._theme)
        self._show(self._theme)

    def commit_edit(self, token: EditToken) -> bool:
        if not self.document.owns(token):
            return False
        self._clear_pending()
        assert self.document.transaction is not None
        if not self._show(self.document.transaction.candidate, token.field):
            return False
        self.document.commit(token)
        self._refresh_state()
        return True

    def cancel_edit(self, token: EditToken) -> None:
        if self.document.owns(token):
            self._clear_pending()
            self.document.cancel()
            if self.app.is_running:
                self._show(self._theme)

    def cancel_active_dialog(self, *, restore: bool = True) -> None:
        self._clear_pending()
        if not restore:
            self.document.cancel()
        for screen in tuple(self.app.screen_stack):
            if (isinstance(screen, _PickerModal) and screen._editor is self) or (
                not restore and isinstance(screen, SaveThemeDialog)
            ):
                screen.action_close()

    def _refresh_state(self) -> None:
        if (
            self._switching
            or self._recompose_pending
            or self._pruned
            or not self.is_mounted
            or self.screen is not self.app.screen
        ):
            return
        self._refresh_selector()
        for name, button in self._color_buttons.items():
            button.set_value(getattr(self._theme, name))
        for name, button in self._variable_swatches.items():
            button.set_value(self._variable_swatch_value(name))
        original = self.document.original
        for target, button in self._reset_buttons.items():
            kind, name = target.split(":", 1)
            dirty = (
                getattr(self._theme, name) != getattr(original, name)
                if kind == "color"
                else self._theme.variables.get(name) != original.variables.get(name)
            )
            button.set_class(not dirty, "--clean")
        self.query_one("#theme-undo", Button).disabled = not self.document.undo_stack
        self.query_one("#theme-redo", Button).disabled = not self.document.redo_stack
        self.query_one("#theme-reset", Button).disabled = not self.document.changed
        self.query_one("#theme-save", Button).disabled = not self.document.unsaved
        delete = self.query_one("#theme-delete", Button)
        delete.display = not theme_is_read_only(self._theme.name)
        delete.disabled = self.revision is None
        self.query_one("#base-theme", Label).update(Text(self._meta()))

    @property
    def delete_pending(self) -> bool:
        return self._confirming_delete or self._deleting

    def _confirm_delete(self) -> None:
        if self.delete_pending or self._switching or theme_is_read_only(self._theme.name) or self.revision is None:
            return
        self.cancel_active_dialog()
        self._confirming_delete = True
        document_id, name, revision = self.document.id, self._theme.name, self.revision

        def decided(confirmed: bool | None) -> None:
            if confirmed:
                self.call_after_refresh(self._delete_theme, document_id, name, revision)
            else:
                self._confirming_delete = False

        self.app.push_screen(
            ConfirmDialog(
                title=M.DELETE_TITLE.bind(),
                message=M.DELETE_BODY.bind(name=name),
                confirm_label=M.DELETE.bind(),
                cancel_label=M.CANCEL.bind(),
                confirm_variant="error",
                locale_controller=self.host.locale_controller,
            ),
            decided,
        )

    async def _delete_theme(self, document_id: UUID, name: str, revision: ThemeFileRevision) -> None:
        self._confirming_delete = False
        if not self.is_mounted or self.document.id != document_id or self.revision != revision:
            return
        self._deleting = True
        self.disabled = True
        self._selection_error = None
        try:
            await asyncio.to_thread(self.store.delete, name, revision)
            self.host.unregister_user_theme(name)
            if self.is_mounted:
                await self._load_theme(self.host.theme, allow_covered=True)
                # The applied user theme may have changed externally as well.
                # Never leave the deleted document available to save or undo.
                if self._theme.name == name:
                    await self._load_theme(DEFAULT_THEME, allow_covered=True)
                preview = self.host.theme_preview
                assert preview is not None
                preview.history.clear()
                preview.checkpoint()
                self.notify(M.text(self, M.DELETED, name=name), title=M.text(self, M.TITLE), markup=False, timeout=3)
        except ThemeStoreError as error:
            self._selection_error = error.display
        finally:
            self._deleting = False
            self.disabled = self._recompose_pending
            if self.is_mounted:
                self._refresh_state()
                self._display_error()
                self.focus_first_swatch()

    def action_undo(self) -> None:
        if self.delete_pending or self._recompose_pending:
            return
        self.cancel_active_dialog()
        if self.document.undo_stack and self._show(self.document.undo_stack[-1]):
            self.document.undo()
            self._refresh_state()

    def action_redo(self) -> None:
        if self.delete_pending or self._recompose_pending:
            return
        self.cancel_active_dialog()
        if self.document.redo_stack and self._show(self.document.redo_stack[-1]):
            self.document.redo()
            self._refresh_state()

    def _reset_target(self, target: str) -> None:
        self.cancel_active_dialog()
        token = self.document.begin(target)
        kind, name = target.split(":", 1)
        value = getattr(self.document.original, name) if kind == "color" else self.document.original.variables.get(name)
        self.document.stage(token, value)
        self.commit_edit(token)

    def _reset_all(self) -> None:
        self.cancel_active_dialog()
        if self._show(self.document.original):
            self.document.replace(self.document.original)
            self._refresh_state()

    def _variable_swatch_value(self, name: str) -> str | None:
        value = self._theme.variables.get(name)
        if value is not None and "%" in value:
            return self.app.theme_variables.get(name, value)
        return value

    def _variable_picker_initial(self, name: str) -> str | None:
        return self.app.theme_variables.get(name) or self._theme.variables.get(name) or self._color_mapping.get(name)

    @on(Button.Pressed)
    async def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.delete_pending:
            return
        if event.button.id == "theme-close":
            self.post_message(self.CloseRequested())
            return
        button = event.button
        if isinstance(button, _ResetButton):
            self._reset_target(button.target)
            button.action_focus_swatch()
        elif isinstance(button, _FlatThemeColorButton):
            name = button.color_name
            self.app.push_screen(
                _PickerModal(
                    self,
                    target=f"color:{name}",
                    initial=getattr(self._theme, name) or self._color_mapping.get(name),
                    allow_transparent=name not in SURFACES,
                )
            )
        elif isinstance(button, _VariableSwatchButton):
            self.app.push_screen(
                _PickerModal(
                    self, target=f"var:{button.var_name}", initial=self._variable_picker_initial(button.var_name)
                )
            )
        elif button.id == "theme-undo":
            self.action_undo()
        elif button.id == "theme-redo":
            self.action_redo()
        elif button.id == "theme-reset":
            self._reset_all()
        elif button.id == "theme-save":
            await self.open_save()
        elif button.id == "theme-delete":
            self._confirm_delete()

    async def open_save(self) -> None:
        self.cancel_active_dialog()
        target = copy_theme(self._theme)
        if theme_is_read_only(target.name):
            try:
                target.name = await asyncio.to_thread(self.store.suggest_name, target.name)
            except ThemeStoreError as error:
                self._selection_error = error.display
                self._display_error()
                return
        if not self.is_mounted or self.screen is not self.app.screen:
            return

        def show(theme: Theme) -> str | None:
            return None if self._show(theme) else self.host.theme_preview.error if self.host.theme_preview else ""

        def saved(theme: Theme, revision: ThemeFileRevision) -> None:
            self.revision = revision
            document = self.document
            document.original = copy_theme(document.original, name=theme.name)
            document.undo_stack = [copy_theme(item, name=theme.name) for item in document.undo_stack]
            document.redo_stack = [copy_theme(item, name=theme.name) for item in document.redo_stack]
            document.draft = copy_theme(theme)
            document.mark_saved()
            self.host.apply_saved_theme(copy_theme(theme))

        self.app.push_screen(
            SaveThemeDialog(
                target,
                self.store,
                self.revision,
                preview=show,
                saved=saved,
                restore=lambda: self._show(self._theme),
                locale_controller=self.host.locale_controller,
                preview_error=lambda: self.host.theme_preview.error if self.host.theme_preview else "",
            )
        )

    def focus_first_swatch(self) -> None:
        if self._recompose_pending or self._pruned or not self.is_mounted or self.screen is not self.app.screen:
            return
        if self._color_buttons:
            next(iter(self._color_buttons.values())).focus()
        else:
            self.query_one("#theme-select", Select).focus()

    def dismiss_dropdown(self) -> bool:
        """Escape closes the expanded selector before closing the editor."""
        selector = self.query_one("#theme-select", Select)
        if not selector.expanded:
            return False
        selector.expanded = False
        selector.focus()
        return True

    def on_unmount(self) -> None:
        self.host.locale_controller.unregister_surface(self)
        self._clear_pending()
        self.app.theme_changed_signal.unsubscribe(self)
        self._document.cancel()
        if self.app.is_running:
            self.host.end_theme_preview()
