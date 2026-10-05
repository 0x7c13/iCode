# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for pasted/dropped image path conversion in the chat input."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
from textual.app import App, ComposeResult
from textual.events import Paste

from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.widgets.chrome import image_paste
from chrys.app.tui.widgets.chrome.image_paste import (
    convert_pasted_image_paths_to_mentions,
    save_clipboard_image_to_file,
)
from chrys.app.tui.widgets.chrome.input_bar import InputBar, _ChatTextArea
from chrys.foundation.config.settings import SESSION_ROOT_DIR_ENV_VAR
from chrys.foundation.events.bus import EventBus
from chrys.foundation.text.mentions import format_file_mention
from chrys.foundation.util.session_ids import session_short_id
from chrys.orchestration.engine.run.attachments import parse_image_mentions
from chrys.service.state.store import JsonFileStateStore
from tests.support.images import image_bytes
from tests.support.tui_helpers import png_bytes
from tests.support.waiting import wait_for


class _InputBarApp(App):
    def compose(self) -> ComposeResult:
        yield InputBar()


class _MainScreenPasteApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.main_screen = MainScreen(EventBus(), engine_provider=None)

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        await self.push_screen(self.main_screen)


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes((200, 40, 40)))
    return path


def _expected_mention(path: Path) -> str:
    return format_file_mention(path.resolve()) + " "


def test_clipboard_image_dir_for_session_uses_session_attachment_subdir() -> None:
    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / f"short-{session_id}"

    state_store = _FakeStateStore()

    assert image_paste.clipboard_image_dir_for_session(state_store, "abc123") == Path(
        "/sessions/short-abc123/attachments/clipboard"
    )
    assert image_paste.clipboard_image_dir_for_session(None, "abc123") is None
    assert image_paste.clipboard_image_dir_for_session(state_store, None) is None


def test_clipboard_image_dir_for_session_uses_configured_session_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(custom_root))
    state_store = JsonFileStateStore()

    assert image_paste.clipboard_image_dir_for_session(state_store, "abc123") == (
        custom_root / "sessions" / session_short_id("abc123") / "attachments" / "clipboard"
    )


def test_pasted_image_outside_cwd_becomes_absolute_mention(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = _write_image(tmp_path / "outside image.png")

    converted = convert_pasted_image_paths_to_mentions(str(image), cwd=workspace)

    assert converted == _expected_mention(image)
    parsed = parse_image_mentions(converted.strip(), cwd=workspace)
    assert len(parsed.attachments) == 1
    assert parsed.attachments[0].path == image.resolve()


def test_pasted_relative_image_path_uses_active_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _write_image(workspace / "shot.png")

    assert convert_pasted_image_paths_to_mentions("shot.png", cwd=workspace) == _expected_mention(image)


@pytest.mark.skipif(os.name != "nt", reason="Windows backslash paths require Windows path semantics")
def test_pasted_windows_backslash_image_path_becomes_mention(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "Users" / "me" / "shot.png")

    assert convert_pasted_image_paths_to_mentions("Users\\me\\shot.png", cwd=tmp_path) == _expected_mention(image)


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell escaping is not used for Windows file drops")
def test_pasted_posix_shell_escaped_image_path_becomes_mention(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "screen shot.png")
    payload = str(image).replace(" ", "\\ ")

    assert convert_pasted_image_paths_to_mentions(payload, cwd=tmp_path) == _expected_mention(image)


def test_pasted_file_uri_image_path_becomes_mention(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "screen shot.webp")

    assert convert_pasted_image_paths_to_mentions(image.as_uri(), cwd=tmp_path) == _expected_mention(image)


def test_pasted_unsupported_file_uri_is_left_unchanged(tmp_path: Path) -> None:
    text = "file://server/share/shot.png"

    assert convert_pasted_image_paths_to_mentions(text, cwd=tmp_path) == text


def test_pasted_multiple_image_paths_become_mentions(tmp_path: Path) -> None:
    first = _write_image(tmp_path / "first.png")
    second = _write_image(tmp_path / "second.jpg")

    converted = convert_pasted_image_paths_to_mentions(f"{first}\n{second}\n", cwd=tmp_path)

    assert converted == f"{_expected_mention(first).strip()} {_expected_mention(second)}"


def test_pasted_prose_is_not_rewritten(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "shot.png")
    text = f"please inspect {image}"

    assert convert_pasted_image_paths_to_mentions(text, cwd=tmp_path) == text


def test_pasted_malformed_path_is_left_unchanged(tmp_path: Path) -> None:
    text = "a\0.png"

    assert convert_pasted_image_paths_to_mentions(text, cwd=tmp_path) == text


def test_pasted_invalid_urlish_text_is_left_unchanged(tmp_path: Path) -> None:
    text = "// a\uff1a"

    assert convert_pasted_image_paths_to_mentions(text, cwd=tmp_path) == text


def test_pasted_unknown_user_home_path_is_left_unchanged(tmp_path: Path) -> None:
    text = "~nosuch_chrys_user_xyz/shot.png"

    assert convert_pasted_image_paths_to_mentions(text, cwd=tmp_path) == text


def test_pasted_prose_does_not_stat_each_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        calls.append(str(self))
        raise AssertionError("prose paste should fail before filesystem resolution")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    text = " ".join(f"token-{i}" for i in range(500))

    assert convert_pasted_image_paths_to_mentions(text, cwd=tmp_path) == text
    assert calls == []


def test_pasted_mixed_path_group_stops_before_resolving_later_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original_resolve = Path.resolve
    image = _write_image(tmp_path / "shot.png")

    def record_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        calls.append(str(self))
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", record_resolve)

    text = f"not-an-image {image}"

    assert convert_pasted_image_paths_to_mentions(text, cwd=tmp_path) == text
    assert calls == []


@pytest.mark.parametrize("os_name", ["macos", "windows"])
def test_platform_clipboard_bitmap_is_saved_to_session_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
) -> None:
    image = Image.new("RGB", (2, 2), color=(80, 120, 160))
    platform = SimpleNamespace(is_macos=os_name == "macos", is_windows=os_name == "windows")

    monkeypatch.setattr(image_paste, "get_platform", lambda: platform)
    monkeypatch.setattr(image_paste, "_grab_clipboard", lambda: image)
    directory = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR

    path = save_clipboard_image_to_file(directory)

    assert path is not None
    assert path.parent == directory
    assert path.suffix == ".png"
    with Image.open(path) as saved:
        assert saved.size == (2, 2)


def test_clipboard_bitmap_without_session_directory_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = SimpleNamespace(is_macos=False, is_windows=True)

    monkeypatch.setattr(image_paste, "get_platform", lambda: platform)
    monkeypatch.setattr(image_paste, "_grab_clipboard", lambda: pytest.fail("clipboard should not be read"))

    assert save_clipboard_image_to_file(None) is None


def test_windows_clipboard_grab_retries_transient_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (2, 2), color=(80, 120, 160))
    calls = 0

    def grabclipboard() -> Image.Image:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("clipboard temporarily unavailable")
        return image

    monkeypatch.setattr(image_paste, "get_platform", lambda: SimpleNamespace(is_macos=False, is_windows=True))
    monkeypatch.setattr("PIL.ImageGrab.grabclipboard", grabclipboard)
    monkeypatch.setattr(image_paste.time, "sleep", lambda _delay: None)

    assert image_paste._grab_clipboard() is image
    assert calls == 2


def test_windows_clipboard_file_list_uses_first_supported_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = tmp_path / "notes.txt"
    doc.write_text("not an image")
    image = _write_image(tmp_path / "from file list.png")

    monkeypatch.setattr(image_paste, "get_platform", lambda: SimpleNamespace(is_macos=False, is_windows=True))
    monkeypatch.setattr(image_paste, "_grab_clipboard", lambda: [str(doc), str(image)])

    assert (
        save_clipboard_image_to_file(tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR)
        == image.resolve()
    )


def test_macos_clipboard_reader_tries_non_png_clipboard_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = SimpleNamespace(is_macos=True, is_windows=False)
    calls: list[str] = []

    def read_clipboard_class(class_code: str) -> bytes | None:
        calls.append(class_code)
        return png_bytes((80, 120, 160), size=(2, 2)) if class_code == "TIFF" else None

    monkeypatch.setattr(image_paste, "get_platform", lambda: platform)
    monkeypatch.setattr(image_paste, "_read_macos_clipboard_class", read_clipboard_class)
    directory = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR

    path = save_clipboard_image_to_file(directory)

    assert calls == ["PNGf", "TIFF"]
    assert path is not None
    assert path.suffix == ".png"


def test_macos_clipboard_reader_decodes_only_the_clipboard_image_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {"PNGf": image_bytes("PCX"), "TIFF": image_bytes("TIFF")}
    calls: list[str] = []

    def read_clipboard_class(class_code: str) -> bytes | None:
        calls.append(class_code)
        return payloads.get(class_code)

    monkeypatch.setattr(image_paste, "get_platform", lambda: SimpleNamespace(is_macos=True, is_windows=False))
    monkeypatch.setattr(image_paste, "_read_macos_clipboard_class", read_clipboard_class)

    path = save_clipboard_image_to_file(tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR)

    assert calls == ["PNGf", "TIFF"]
    assert path is not None
    assert path.suffix == ".png"


def test_decode_macos_clipboard_data_extracts_hex_payload() -> None:
    assert image_paste._decode_macos_clipboard_data(b"\xc2\xabdata PNGf616263\xc2\xbb\n") == b"abc"


def test_clipboard_image_save_failure_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (2, 2), color=(80, 120, 160))

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise ValueError("cannot save")

    monkeypatch.setattr(image, "save", fail_save)

    assert image_paste._save_clipboard_image(image, directory=tmp_path / "session") is None


def test_clipboard_image_save_prunes_old_saved_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = tmp_path / "clipboard-image-old.png"
    newer = tmp_path / "clipboard-image-newer.png"
    old.write_bytes(b"old")
    newer.write_bytes(b"newer")
    os.utime(old, ns=(1, 1))
    os.utime(newer, ns=(2, 2))
    image = Image.new("RGB", (2, 2), color=(80, 120, 160))

    monkeypatch.setattr(image_paste, "_CLIPBOARD_IMAGE_KEEP_COUNT", 2)
    monkeypatch.setattr(image_paste, "_CLIPBOARD_IMAGE_KEEP_BYTES", 1024 * 1024)

    path = image_paste._save_clipboard_image(image, directory=tmp_path)

    assert path is not None
    remaining = {candidate.name for candidate in tmp_path.glob("clipboard-image-*.png")}
    assert path.name in remaining
    assert "clipboard-image-newer.png" in remaining
    assert "clipboard-image-old.png" not in remaining


@pytest.mark.asyncio
async def test_chat_input_paste_converts_image_path_to_mention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _write_image(tmp_path / "dropped image.png")
    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.safe_getcwd", lambda: str(tmp_path))

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        await text_area._on_paste(Paste(str(image)))
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)


@pytest.mark.asyncio
async def test_chat_input_paste_inserts_invalid_urlish_text_unchanged(tmp_path: Path) -> None:
    text = "// a\uff1a"

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        await text_area._on_paste(Paste(text))
        await pilot.pause()

        assert input_bar.value == text


@pytest.mark.asyncio
async def test_chat_input_paste_inserts_text_when_image_detection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "not an image"

    def fail_convert(_text: str, *, cwd: Path) -> str:
        raise ValueError("invalid path payload")

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.convert_pasted_image_paths_to_mentions", fail_convert)

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        await text_area._on_paste(Paste(text))
        await pilot.pause()

        assert input_bar.value == text


@pytest.mark.asyncio
async def test_chat_input_paste_uses_configured_workspace_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    image = _write_image(workspace / "relative.png")

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.set_paste_cwd(str(workspace))
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        await text_area._on_paste(Paste("relative.png"))
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)


@pytest.mark.asyncio
async def test_chat_input_paste_inserts_space_before_image_mention_when_needed(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "shot.png")

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        input_bar.value = "look at"
        text_area.focus()
        await pilot.pause()

        await text_area._on_paste(Paste(str(image)))
        await pilot.pause()

        assert input_bar.value == f"look at {_expected_mention(image)}"


@pytest.mark.asyncio
@pytest.mark.parametrize("paste_key", [None, "shift+insert"], ids=["action", "shift-insert"])
async def test_chat_input_paste_action_and_shift_insert_convert_os_clipboard_image_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paste_key: str | None,
) -> None:
    image = _write_image(tmp_path / "clipboard.png")
    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.safe_getcwd", lambda: str(tmp_path))

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await wait_for(lambda: text_area.has_focus, pilot=pilot, description="control focus before interaction")
        await pilot.pause()

        monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.save_clipboard_image_to_file", lambda _dir: None)
        with patch("chrys.app.tui.clipboard.platform_helpers.clipboard_paste", return_value=str(image)):
            if paste_key is None:
                text_area.action_paste()
            else:
                await pilot.press(paste_key)
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)


@pytest.mark.asyncio
async def test_chat_input_shift_insert_in_browser_uses_only_session_clipboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser shortcuts must not read the server machine's text or image clipboard."""
    clipboard_dir = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR

    def fail_server_clipboard_read() -> str:
        raise AssertionError("server clipboard must stay isolated from browser sessions")

    def fail_server_image_read() -> object:
        raise AssertionError("server image clipboard must stay isolated from browser sessions")

    monkeypatch.setenv("TEXTUAL_DRIVER", "textual.drivers.web_driver:WebDriver")
    monkeypatch.setattr("chrys.app.tui.clipboard.platform_helpers.clipboard_paste", fail_server_clipboard_read)
    monkeypatch.setattr("chrys.app.tui.widgets.chrome.image_paste._grab_clipboard", fail_server_image_read)

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.set_clipboard_image_dir(clipboard_dir)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await wait_for(lambda: text_area.has_focus, pilot=pilot, description="control focus before interaction")
        pilot.app.copy_to_clipboard("browser-session-text")
        await pilot.pause()

        await pilot.press("shift+insert")
        await pilot.pause()

        assert input_bar.value == "browser-session-text"
        assert not clipboard_dir.exists()


@pytest.mark.asyncio
async def test_chat_input_action_paste_saves_clipboard_bitmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _write_image(tmp_path / "clipboard bitmap.png")
    clipboard_dir = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR
    seen_dirs: list[Path | None] = []

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.set_clipboard_image_dir(clipboard_dir)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        pilot.app.copy_to_clipboard("stale-app-clipboard")
        await pilot.pause()

        monkeypatch.setattr(
            "chrys.app.tui.widgets.chrome.input_bar.save_clipboard_image_to_file",
            lambda directory: seen_dirs.append(directory) or image,
        )
        text_area.action_paste()
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)
        assert seen_dirs == [clipboard_dir]


@pytest.mark.asyncio
async def test_chat_input_action_paste_inserts_clipboard_path_when_image_detection_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _write_image(tmp_path / "clipboard fallback.png")
    clipboard_dir = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR

    def fail_convert(_text: str, *, cwd: Path) -> str:
        raise ValueError("invalid path payload")

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.set_clipboard_image_dir(clipboard_dir)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.save_clipboard_image_to_file", lambda _dir: image)
        monkeypatch.setattr(
            "chrys.app.tui.widgets.chrome.input_bar.convert_pasted_image_paths_to_mentions", fail_convert
        )
        text_area.action_paste()
        await pilot.pause()

        assert input_bar.value == str(image)


@pytest.mark.asyncio
async def test_chat_input_empty_paste_event_saves_clipboard_bitmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _write_image(tmp_path / "empty paste bitmap.png")
    clipboard_dir = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR
    seen_dirs: list[Path | None] = []

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.set_clipboard_image_dir(clipboard_dir)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        monkeypatch.setattr(
            "chrys.app.tui.widgets.chrome.input_bar.save_clipboard_image_to_file",
            lambda directory: seen_dirs.append(directory) or image,
        )
        await text_area._on_paste(Paste(""))
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)
        assert seen_dirs == [clipboard_dir]


@pytest.mark.asyncio
async def test_chat_input_empty_paste_without_bitmap_does_not_reprobe_in_screen_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clipboard_dir = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR
    calls: list[Path | None] = []
    app = _MainScreenPasteApp()

    async with app.run_test() as pilot:
        input_bar = app.main_screen.query_one(InputBar)
        input_bar.set_clipboard_image_dir(clipboard_dir)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        monkeypatch.setattr(
            "chrys.app.tui.widgets.chrome.input_bar.save_clipboard_image_to_file",
            lambda directory: calls.append(directory) or None,
        )
        event = Paste("")
        await text_area._on_paste(event)
        app.main_screen.on_paste(event)
        await pilot.pause()

        assert input_bar.value == ""
        assert calls == [clipboard_dir]
        assert event._no_default_action is True
        assert event._stop_propagation is True


@pytest.mark.asyncio
async def test_input_bar_fallback_inserts_image_drop_payload(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "fallback.png")

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.value = "look at"
        await pilot.pause()

        assert input_bar.insert_image_paste_payload(str(image)) is True

        assert input_bar.value == f"look at {_expected_mention(image)}"


@pytest.mark.asyncio
async def test_input_bar_fallback_ignores_normal_prose(tmp_path: Path) -> None:
    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.value = "existing"
        await pilot.pause()

        assert input_bar.insert_image_paste_payload("please inspect /tmp/shot.png") is False

        assert input_bar.value == "existing"


@pytest.mark.asyncio
async def test_input_bar_fallback_ignores_image_detection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_convert(_text: str, *, cwd: Path) -> str:
        raise ValueError("invalid path payload")

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.convert_pasted_image_paths_to_mentions", fail_convert)

    async with _InputBarApp().run_test() as pilot:
        input_bar = pilot.app.query_one(InputBar)
        input_bar.value = "existing"
        await pilot.pause()

        assert input_bar.insert_image_paste_payload("not an image") is False
        assert input_bar.value == "existing"


@pytest.mark.asyncio
async def test_main_screen_paste_fallback_routes_image_drop_to_input(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "screen fallback.png")
    app = _MainScreenPasteApp()

    async with app.run_test() as pilot:
        input_bar = app.main_screen.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        app.set_focus(None)
        await pilot.pause()
        assert not text_area.has_focus

        event = Paste(str(image))
        app.main_screen.on_paste(event)
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)
        assert event._no_default_action is True
        assert event._stop_propagation is True


@pytest.mark.asyncio
async def test_main_screen_empty_paste_fallback_routes_clipboard_bitmap_to_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _write_image(tmp_path / "screen bitmap.png")
    clipboard_dir = tmp_path / "session" / image_paste.CLIPBOARD_IMAGE_SESSION_SUBDIR
    app = _MainScreenPasteApp()

    async with app.run_test() as pilot:
        input_bar = app.main_screen.query_one(InputBar)
        input_bar.set_clipboard_image_dir(clipboard_dir)
        app.set_focus(None)
        await pilot.pause()

        monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.save_clipboard_image_to_file", lambda _dir: image)
        event = Paste("")
        app.main_screen.on_paste(event)
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)
        assert event._no_default_action is True
        assert event._stop_propagation is True


@pytest.mark.asyncio
async def test_main_screen_paste_fallback_ignores_prose_payload(tmp_path: Path) -> None:
    app = _MainScreenPasteApp()

    async with app.run_test() as pilot:
        input_bar = app.main_screen.query_one(InputBar)
        await pilot.pause()

        event = Paste("please inspect /tmp/shot.png")
        app.main_screen.on_paste(event)
        await pilot.pause()

        assert input_bar.value == ""
        assert event._no_default_action is False
        assert event._stop_propagation is False


@pytest.mark.asyncio
async def test_main_screen_paste_fallback_skips_shell_mode(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "shell.png")
    app = _MainScreenPasteApp()

    async with app.run_test() as pilot:
        input_bar = app.main_screen.query_one(InputBar)
        app.main_screen._state.shell.active = True
        await pilot.pause()

        event = Paste(str(image))
        app.main_screen.on_paste(event)
        await pilot.pause()

        assert input_bar.value == ""
        assert event._no_default_action is False
        assert event._stop_propagation is False


@pytest.mark.asyncio
async def test_focused_chat_input_image_paste_does_not_double_insert_when_bubbled(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "focused.png")
    app = _MainScreenPasteApp()

    async with app.run_test() as pilot:
        input_bar = app.main_screen.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        event = Paste(str(image))
        await text_area._on_paste(event)
        app.main_screen.on_paste(event)
        await pilot.pause()

        assert input_bar.value == _expected_mention(image)
