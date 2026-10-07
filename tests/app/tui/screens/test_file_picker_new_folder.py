# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for creating a folder from the folder picker."""

from __future__ import annotations

import asyncio
import errno
import sys
import threading
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from types import ModuleType

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widgets import Button, OptionList, Static
from textual.widgets import _directory_tree as directory_tree_module
from textual.widgets._directory_tree import DirEntry
from textual.widgets.option_list import Option
from textual.widgets.tree import TreeNode
from textual.worker import Worker

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.dialogs import new_folder
from chrys.app.tui.screens.dialogs.file_picker import FilePicker, FilePickerMode, _FilteredDirectoryTree
from chrys.app.tui.screens.dialogs.new_folder import (
    NewFolderDialog,
    NewFolderError,
    create_folder,
    validate_folder_name,
)
from chrys.app.tui.widgets import EnhancedInput
from chrys.foundation.config.settings import Settings
from chrys.foundation.i18n import MessageRef
from tests.support.symlinks import symlink_or_skip
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for, wait_until


class _PickerApp(App):
    def compose(self) -> ComposeResult:
        yield Static("placeholder")


class _LocalizedPickerApp(_PickerApp):
    def __init__(self, locale: str) -> None:
        self.locale_controller = LocaleController(Settings(locale=locale))
        super().__init__()


def _key(reference: MessageRef | None) -> str | None:
    return None if reference is None else reference.definition.key


# ──────────── name rules ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "windows", "name", "error"),
    [
        ("", False, "", None),
        ("   ", False, "", None),
        ("  notes  ", False, "notes", None),
        (".hidden", False, ".hidden", None),
        ("笔记", False, "笔记", None),
        ("a\\b", False, "a\\b", None),
        ("a:b", False, "a:b", None),
        ("con", False, "con", None),
        ("notes.", False, "notes.", None),
        (".", False, ".", "dot_name"),
        ("..", True, "..", "dot_name"),
        ("a/b", False, "a/b", "invalid_char"),
        ("a/b", True, "a/b", "invalid_char"),
        ("a\tb", False, "a\tb", "control_char"),
        ("a\x00b", False, "a\x00b", "control_char"),
        ("a\x1bb", True, "a\x1bb", "control_char"),
        ("a\x7fb", False, "a\x7fb", "control_char"),
        ("a\\b", True, "a\\b", "invalid_char"),
        ("a:b", True, "a:b", "invalid_char"),
        ("a*b", True, "a*b", "invalid_char"),
        ("a?", True, "a?", "invalid_char"),
        ("a<b>", True, "a<b>", "invalid_char"),
        ('a"b', True, 'a"b', "invalid_char"),
        ("a|b", True, "a|b", "invalid_char"),
        ("notes.", True, "notes.", "trailing_dot"),
        ("CON", True, "CON", "reserved"),
        ("aux.md", True, "aux.md", "reserved"),
        ("com1", True, "com1", "reserved"),
        ("notes", True, "notes", None),
        ("v1.2", True, "v1.2", None),
    ],
)
def test_validate_folder_name(raw: str, windows: bool, name: str, error: str | None) -> None:
    trimmed, reference = validate_folder_name(raw, windows=windows)

    assert trimmed == name
    assert _key(reference) == (None if error is None else f"tui.file_picker.new_folder.error.{error}")


def test_invalid_char_error_names_the_character() -> None:
    _name, reference = validate_folder_name("a/b", windows=False)

    assert reference is not None
    assert dict(reference.args) == {"char": "/"}


# ──────────── create errors ────────────────────────────────────────────


def test_create_folder_creates_the_folder(tmp_path: Path) -> None:
    created = create_folder(tmp_path, "notes")

    assert created == tmp_path / "notes"
    assert created.is_dir()


@pytest.mark.parametrize("existing", ["folder", "file"])
def test_create_folder_reports_a_taken_name(tmp_path: Path, existing: str) -> None:
    taken = tmp_path / "taken"
    if existing == "folder":
        taken.mkdir()
    else:
        taken.write_text("", encoding="utf-8")

    with pytest.raises(NewFolderError) as caught:
        create_folder(tmp_path, "taken")

    assert caught.value.message.definition.key == "tui.file_picker.new_folder.error.exists"
    assert dict(caught.value.message.args) == {"name": "taken"}


def test_create_folder_reports_a_missing_parent(tmp_path: Path) -> None:
    with pytest.raises(NewFolderError) as caught:
        create_folder(tmp_path / "gone", "child")

    assert caught.value.message.definition.key == "tui.file_picker.new_folder.error.parent_missing"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FileExistsError(errno.EEXIST, "exists"), "exists"),
        (PermissionError(errno.EACCES, "denied"), "permission"),
        (OSError(errno.EROFS, "read-only"), "read_only"),
        (OSError(errno.ENAMETOOLONG, "too long"), "too_long"),
        (FileNotFoundError(errno.ENOENT, "missing"), "parent_missing"),
        (NotADirectoryError(errno.ENOTDIR, "not a folder"), "parent_missing"),
        (OSError(errno.EINVAL, "invalid"), "not_allowed"),
        (OSError(errno.ENOSPC, "No space left on device"), "failed"),
    ],
)
def test_create_folder_describes_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError, expected: str
) -> None:
    def fail(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        raise error

    monkeypatch.setattr(Path, "mkdir", fail)

    with pytest.raises(NewFolderError) as caught:
        create_folder(tmp_path, "child")

    assert caught.value.message.definition.key == f"tui.file_picker.new_folder.error.{expected}"


def test_create_folder_reports_the_os_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "mkdir", fail)

    with pytest.raises(NewFolderError) as caught:
        create_folder(tmp_path, "child")

    assert dict(caught.value.message.args) == {"reason": "No space left on device"}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows error codes exist only on Windows")
@pytest.mark.parametrize(("winerror", "expected"), [(19, "read_only"), (206, "too_long")])
def test_create_folder_reads_windows_error_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int, expected: str
) -> None:
    # CPython turns these into PermissionError and FileNotFoundError.
    error = OSError(0, "Windows error", None, winerror)

    def fail(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        raise error

    monkeypatch.setattr(Path, "mkdir", fail)

    with pytest.raises(NewFolderError) as caught:
        create_folder(tmp_path, "child")

    assert caught.value.message.definition.key == f"tui.file_picker.new_folder.error.{expected}"


# ──────────── picker flow ──────────────────────────────────────────────


async def _open_picker(pilot: Pilot, path: Path, mode: FilePickerMode = FilePickerMode.FOLDER) -> FilePicker:
    screen = FilePicker(mode=mode, initial_path=str(path))
    await pilot.app.push_screen(screen)
    await _tree(screen).reload()
    await pilot.pause()
    return screen


def _tree(screen: FilePicker) -> _FilteredDirectoryTree:
    return screen.query_one("#fsd-tree", _FilteredDirectoryTree)


def _child_node(tree: _FilteredDirectoryTree, path: Path):
    return next(node for node in tree.root.children if node.data is not None and node.data.path == path)


async def _open_new_folder(pilot: Pilot) -> NewFolderDialog:
    await click_when_settled(pilot, "#fsd-new-folder")
    await wait_for(
        lambda: isinstance(pilot.app.screen, NewFolderDialog) and pilot.app.screen.is_mounted,
        pilot=pilot,
        description="new folder dialog",
    )
    dialog = pilot.app.screen
    assert isinstance(dialog, NewFolderDialog)
    return dialog


async def _type_name(pilot: Pilot, dialog: NewFolderDialog, name: str) -> None:
    dialog.query_one("#new-folder-input", EnhancedInput).value = name
    await pilot.pause()


def _error_text(dialog: NewFolderDialog) -> str:
    return dialog.query_one("#new-folder-error", Static).content.plain


@pytest.mark.asyncio
async def test_new_folder_button_only_in_folder_mode(tmp_path: Path) -> None:
    app = _PickerApp()
    async with app.run_test() as pilot:
        folder_picker = await _open_picker(pilot, tmp_path)
        assert folder_picker.query("#fsd-new-folder")
        folder_picker.dismiss(None)
        await wait_for(lambda: app.screen is not folder_picker, pilot=pilot, description="folder picker closed")

        file_picker = await _open_picker(pilot, tmp_path, FilePickerMode.FILE)
        assert not file_picker.query("#fsd-new-folder")


@pytest.mark.asyncio
async def test_new_folder_is_created_in_highlighted_folder_and_selected(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, tmp_path)
        tree = _tree(picker)
        tree.move_cursor(_child_node(tree, child))
        await wait_for(lambda: picker._selected_path == str(child), pilot=pilot, description="child highlighted")

        dialog = await _open_new_folder(pilot)
        assert str(child) in dialog.query_one("#new-folder-hint", Static).content.plain
        field = dialog.query_one("#new-folder-input", EnhancedInput)
        await wait_for(lambda: app.focused is field, pilot=pilot, description="name field focused")
        await _type_name(pilot, dialog, "  notes ")
        await pilot.press("enter")

        created = child / "notes"
        await wait_for(
            lambda: app.screen is picker and picker._selected_path == str(created) and app.focused is tree,
            pilot=pilot,
            description="created folder selected and tree focused",
        )
        assert created.is_dir()
        assert tree.cursor_node is not None and tree.cursor_node.data is not None
        assert tree.cursor_node.data.path == created
        assert not picker.query_one("#fsd-select", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["root", "parent_entry"])
async def test_new_folder_from_root_or_parent_entry_goes_in_tree_root(tmp_path: Path, cursor: str) -> None:
    root = tmp_path / "root"
    (root / "existing").mkdir(parents=True)
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, root)
        tree = _tree(picker)
        tree.move_cursor(tree.root if cursor == "root" else tree.root.children[0])
        await pilot.pause()

        dialog = await _open_new_folder(pilot)
        assert str(root) in dialog.query_one("#new-folder-hint", Static).content.plain
        await _type_name(pilot, dialog, "fresh")
        await pilot.press("enter")

        created = root / "fresh"
        await wait_for(
            lambda: app.screen is picker and picker._selected_path == str(created),
            pilot=pilot,
            description="created folder selected",
        )
        assert created.is_dir()
        assert tree.path == root
        assert [node for node in tree.root.children if tree.is_parent_navigation_node(node)] == [tree.root.children[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("link", ["highlighted_folder", "picker_root"])
async def test_new_folder_inside_a_symlinked_folder_is_selected(tmp_path: Path, link: str) -> None:
    # The tree lists a symlinked folder's children under the link target's path.
    base = tmp_path / "base"
    real = base / "real"
    real.mkdir(parents=True)
    alias = base / "alias"
    symlink_or_skip(alias, real, target_is_directory=True)
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, base if link == "highlighted_folder" else alias)
        tree = _tree(picker)
        tree.move_cursor(_child_node(tree, alias) if link == "highlighted_folder" else tree.root)
        await pilot.pause()

        dialog = await _open_new_folder(pilot)
        await _type_name(pilot, dialog, "fresh")
        await pilot.press("enter")

        await wait_for(
            lambda: (
                app.screen is picker
                and picker._selected_path is not None
                and Path(picker._selected_path).name == "fresh"
            ),
            pilot=pilot,
            description="created folder selected",
        )
        assert picker._selected_path is not None
        assert Path(picker._selected_path).resolve() == (real / "fresh").resolve()
        assert (alias / "fresh").is_dir()


@pytest.mark.asyncio
async def test_new_folder_named_like_the_parent_entry_selects_the_new_folder(tmp_path: Path) -> None:
    # The root-level ".." entry lists the parent folder, whose name is "outer" here.
    root = tmp_path / "outer" / "root"
    root.mkdir(parents=True)
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, root)
        tree = _tree(picker)
        tree.move_cursor(tree.root)
        await pilot.pause()

        dialog = await _open_new_folder(pilot)
        await _type_name(pilot, dialog, "outer")
        await pilot.press("enter")

        created = root / "outer"
        await wait_for(
            lambda: app.screen is picker and picker._selected_path == str(created),
            pilot=pilot,
            description="created folder selected",
        )
        assert tree.cursor_node is not None
        assert not tree.is_parent_navigation_node(tree.cursor_node)


@pytest.mark.asyncio
async def test_existing_name_keeps_dialog_open_until_renamed(tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, tmp_path)
        tree = _tree(picker)
        tree.move_cursor(tree.root)
        await pilot.pause()
        dialog = await _open_new_folder(pilot)
        field = dialog.query_one("#new-folder-input", EnhancedInput)

        await _type_name(pilot, dialog, "taken")
        await pilot.press("enter")
        await wait_for(
            lambda: "taken" in _error_text(dialog) and app.focused is field,
            pilot=pilot,
            description="exists error shown and name field refocused",
        )
        assert app.screen is dialog
        assert not field.disabled

        await _type_name(pilot, dialog, "taken2")
        assert _error_text(dialog) == ""
        assert not dialog.query_one("#new-folder-create", Button).disabled

        # The failure belongs to the name that caused it: typing it again shows it again.
        await _type_name(pilot, dialog, "taken")
        assert "taken" in _error_text(dialog)


@pytest.mark.asyncio
async def test_invalid_and_empty_names_disable_create(tmp_path: Path) -> None:
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, tmp_path)
        before = sorted(tmp_path.iterdir())
        dialog = await _open_new_folder(pilot)
        create = dialog.query_one("#new-folder-create", Button)

        assert create.disabled
        await pilot.press("enter")
        assert app.screen is dialog

        await _type_name(pilot, dialog, "a/b")
        assert create.disabled
        assert "/" in _error_text(dialog)
        await pilot.press("enter")
        assert app.screen is dialog

        await _type_name(pilot, dialog, "ab")
        assert not create.disabled
        assert _error_text(dialog) == ""
        assert picker is not app.screen
    assert sorted(tmp_path.iterdir()) == before


@pytest.mark.asyncio
async def test_escape_closes_only_the_new_folder_dialog(tmp_path: Path) -> None:
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, tmp_path)
        before = picker._selected_path
        dialog = await _open_new_folder(pilot)
        await _type_name(pilot, dialog, "unused")

        await pilot.press("escape")

        await wait_for(lambda: app.screen is picker, pilot=pilot, description="back to picker")
        assert picker._selected_path == before
    assert not (tmp_path / "unused").exists()


@pytest.mark.asyncio
async def test_create_failure_is_shown_in_dialog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(parent: Path, name: str) -> Path:
        raise NewFolderError(new_folder._PERMISSION.bind())

    monkeypatch.setattr(new_folder, "create_folder", deny)
    app = _PickerApp()
    async with app.run_test() as pilot:
        await _open_picker(pilot, tmp_path)
        dialog = await _open_new_folder(pilot)
        await _type_name(pilot, dialog, "blocked")
        await pilot.press("enter")

        await wait_for(lambda: "permission" in _error_text(dialog), pilot=pilot, description="permission error")
        assert app.screen is dialog
        assert not dialog.query_one("#new-folder-input", EnhancedInput).disabled
        assert not dialog.query_one("#new-folder-cancel", Button).disabled


@pytest.mark.asyncio
async def test_create_in_flight_ignores_resubmit_and_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    real_create = new_folder.create_folder

    def held_create(parent: Path, name: str) -> Path:
        calls.append(name)
        started.set()
        release.wait(timeout=30)
        return real_create(parent, name)

    monkeypatch.setattr(new_folder, "create_folder", held_create)
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, tmp_path)
        dialog = await _open_new_folder(pilot)
        await _type_name(pilot, dialog, "slow")
        try:
            await pilot.press("enter")
            await wait_for(started.is_set, pilot=pilot, description="create started")
            assert dialog.query_one("#new-folder-input", EnhancedInput).disabled
            assert dialog.query_one("#new-folder-cancel", Button).disabled

            await pilot.press("escape")
            await pilot.click(offset=(0, 0))
            dialog._start_create()
            await pilot.pause()
            assert app.screen is dialog
            assert not dialog.dismiss_requested
        finally:
            release.set()

        await wait_for(
            lambda: app.screen is picker and picker._selected_path == str(tmp_path / "slow"),
            pilot=pilot,
            description="created after release",
        )
    assert calls == ["slow"]


class _HeldListing:
    """A directory listing whose result is handed back only once the test releases it."""

    def __init__(self, worker: Worker[list[Path]]) -> None:
        self._worker = worker
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self.returned = asyncio.Event()

    async def wait(self) -> list[Path]:
        content = await self._worker.wait()
        self.held.set()
        await self.release.wait()
        self.returned.set()
        return content


@pytest.mark.asyncio
async def test_created_folder_stays_selected_when_a_slow_expansion_lists_the_folder_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a slow drive the tree is still checking the root it just expanded when
    # the folder is created. The listing that check then queues puts back the
    # cursor it saw when it started, after the reveal has moved it.
    root = tmp_path / "root"
    (root / "existing").mkdir(parents=True)
    created = root / "fresh"
    stat_held, release_stat = asyncio.Event(), asyncio.Event()
    real_to_thread = asyncio.to_thread

    async def to_thread(func: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        if args == (root,) and not stat_held.is_set():
            stat_held.set()
            await release_stat.wait()
        return await real_to_thread(func, *args, **kwargs)

    shadow = ModuleType("asyncio")
    shadow.to_thread = to_thread  # type: ignore[attr-defined]
    monkeypatch.setattr(directory_tree_module, "asyncio", shadow)

    listings: list[_HeldListing] = []
    hold_listings = False
    real_load = _FilteredDirectoryTree._load_directory

    def load_directory(tree: _FilteredDirectoryTree, node: TreeNode[DirEntry]) -> Worker[list[Path]] | _HeldListing:
        worker = real_load(tree, node)
        if not hold_listings or len(listings) == 2:
            return worker
        listings.append(_HeldListing(worker))
        return listings[-1]

    monkeypatch.setattr(_FilteredDirectoryTree, "_load_directory", load_directory)
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = FilePicker(mode=FilePickerMode.FOLDER, initial_path=str(root))
        await app.push_screen(picker)
        tree = _tree(picker)
        # Pilot waits drain every pump, and the tree's stays inside the held
        # handler, so the steps below poll without the pilot until it returns.
        try:
            await wait_for(stat_held.is_set, description="root expansion checking the root")
            picker.query_one("#fsd-new-folder", Button).press()
            await wait_for(
                lambda: isinstance(app.screen, NewFolderDialog) and app.screen.is_mounted,
                description="new folder dialog",
            )
            dialog = app.screen
            assert isinstance(dialog, NewFolderDialog)
            dialog.query_one("#new-folder-input", EnhancedInput).value = created.name
            create = dialog.query_one("#new-folder-create", Button)
            await wait_for(lambda: not create.disabled, description="create enabled")
            hold_listings = True
            create.press()

            await wait_for(lambda: listings and listings[0].held.is_set(), description="reveal listing")
            assert created.is_dir()
            release_stat.set()
            await wait_for(
                lambda: tree.root.data is not None and tree.root.data.loaded,
                description="expansion queued another root listing",
            )
            listings[0].release.set()
            await wait_for(lambda: len(listings) == 2 and listings[1].held.is_set(), description="queued root listing")
            listings[1].release.set()
            await wait_for(listings[1].returned.is_set, description="queued listing applied")

            await wait_for(
                lambda: picker._selected_path == str(created) and app.focused is tree,
                pilot=pilot,
                description="created folder selected after the queued listing",
            )
            assert tree.cursor_node is not None and tree.cursor_node.data is not None
            assert tree.cursor_node.data.path == created
        finally:
            release_stat.set()
            for listing in listings:
                listing.release.set()


@pytest.mark.asyncio
async def test_moving_elsewhere_while_the_reveal_lists_keeps_the_new_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The picker reuses its root node for a new location, which here holds a
    # folder with the created folder's name.
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    (second / "docs").mkdir(parents=True)
    listings: list[_HeldListing] = []
    hold_listing = False
    real_load = _FilteredDirectoryTree._load_directory

    def load_directory(tree: _FilteredDirectoryTree, node: TreeNode[DirEntry]) -> Worker[list[Path]] | _HeldListing:
        worker = real_load(tree, node)
        if not hold_listing or listings:
            return worker
        listings.append(_HeldListing(worker))
        return listings[-1]

    monkeypatch.setattr(_FilteredDirectoryTree, "_load_directory", load_directory)
    app = _PickerApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(pilot, first)
        tree = _tree(picker)
        tree.move_cursor(tree.root)
        await pilot.pause()
        favorites = picker.query_one("#fsd-favorites", OptionList)
        favorites.add_option(Option("second", id=str(second)))
        dialog = await _open_new_folder(pilot)
        await _type_name(pilot, dialog, "docs")
        create = dialog.query_one("#new-folder-create", Button)
        await wait_for(lambda: not create.disabled, pilot=pilot, description="create enabled")
        hold_listing = True
        # The held listing keeps the tree's lock, which the tree's idle handler
        # waits for, so the steps below poll without the pilot until it's released.
        try:
            create.press()
            await wait_for(lambda: listings and listings[0].held.is_set(), description="reveal listing")
            assert (first / "docs").is_dir()
            await wait_for(lambda: app.screen is picker, description="dialog closed")

            favorites.highlighted = favorites.option_count - 1
            favorites.action_select()
            await wait_for(lambda: tree.path == second, description="picker moved to second")
        finally:
            for listing in listings:
                listing.release.set()

        await wait_for(
            lambda: (
                tree.root.data is not None
                and tree.root.data.path == second
                and any(node.data is not None and node.data.path == second / "docs" for node in tree.root.children)
            ),
            pilot=pilot,
            description="second listed",
        )
        await wait_for(lambda: picker._selected_path == str(second), pilot=pilot, description="second selected")
        assert not await wait_until(lambda: picker._selected_path != str(second), pilot=pilot, timeout=0.5)


# ──────────── picker layout ────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["en", "zh-Hans"])
@pytest.mark.parametrize("width", [60, 80, 99, 100, 140])
async def test_picker_buttons_fit_beside_the_path(tmp_path: Path, locale: str, width: int) -> None:
    app = _LocalizedPickerApp(locale)
    async with app.run_test(size=(width, 30)) as pilot:
        picker = await _open_picker(pilot, tmp_path)
        assert picker.has_class("-narrow") is (width < 100)

        row = picker.query_one("#fsd-buttons").region
        path_bar = picker.query_one("#fsd-path-bar", Static).region
        buttons = [picker.query_one(f"#{name}", Button) for name in ("fsd-new-folder", "fsd-select", "fsd-cancel")]
        regions = [button.region for button in buttons]

        assert path_bar.width >= 5
        assert all(row.contains_region(region) for region in regions)
        assert path_bar.right <= regions[0].x
        assert all(left.right <= right.x for left, right in pairwise(regions))
        for button in buttons:
            assert button.content_region.width >= cell_len(button.label.plain)
