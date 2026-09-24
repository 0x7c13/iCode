# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the clean-room in-place Markdown message-editor highlighter."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.content import Content
from textual.events import Paste
from textual.widgets.text_area import Selection

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.widgets.editor import (
    MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF,
    MARKDOWN_HIGHLIGHT_LINE_CUTOFF,
    MARKDOWN_HIGHLIGHT_PLAIN_STATUS,
    MESSAGE_EDITOR_CHARACTER_LIMIT_STATUS,
    MESSAGE_EDITOR_MAX_CHARACTERS,
    MarkdownHighlightedTextArea,
    MessageEditor,
)
from chrys.app.tui.widgets.text_area import MESSAGE_EDITOR_PASTE_MAX_TOKENS, EnhancedTextArea
from chrys.foundation.config.settings import Settings
from chrys.foundation.i18n import Localizer, MessageRef
from chrys.foundation.i18n.formatting import format_message
from tests.support.waiting import wait_for

pytestmark = pytest.mark.asyncio


def _render_status(status: MessageRef | str) -> str:
    return format_message(status) if isinstance(status, MessageRef) else status


class _CountingMessageEditor(MessageEditor):
    def __init__(self, text: str) -> None:
        self.highlight_calls = 0
        super().__init__(text=text, id="editor", soft_wrap=True)

    def _highlight_markdown_document(self, source: str) -> Content:
        self.highlight_calls += 1
        return super()._highlight_markdown_document(source)


class _EditorApp(App[None]):
    locale_controller = LocaleController(Settings(locale="en"))

    def __init__(self, editor: MessageEditor) -> None:
        self.editor = editor
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.editor


def _span_signature(line: Text) -> list[tuple[int, int, str]]:
    return [(span.start, span.end, str(span.style)) for span in line.spans]


async def test_message_editor_inherits_enhanced_text_area_and_shared_paste_cap() -> None:
    editor = MessageEditor()

    assert isinstance(editor, MarkdownHighlightedTextArea)
    assert isinstance(editor, EnhancedTextArea)
    assert editor._max_paste_tokens == MESSAGE_EDITOR_PASTE_MAX_TOKENS == 30_000


async def test_message_editor_base_unmount_handler_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Dispatch plans cache unbound handlers by widget class. A fresh subclass
    # isolates the spy from earlier editors and keeps it out of their plans.
    class _UnmountProbeEditor(MessageEditor):
        pass

    editor = _UnmountProbeEditor(text="# Heading")
    app = _EditorApp(editor)
    base_unmount = MarkdownHighlightedTextArea.on_unmount
    unmount_calls: list[MarkdownHighlightedTextArea] = []

    def record_base_unmount(text_area: MarkdownHighlightedTextArea) -> None:
        unmount_calls.append(text_area)
        base_unmount(text_area)

    monkeypatch.setattr(MarkdownHighlightedTextArea, "on_unmount", record_base_unmount)

    async with app.run_test():
        editor.get_line(0)
        assert editor._markdown_content_lines is not None
        assert editor._markdown_rendered_lines

        await editor.remove()

        assert unmount_calls == [editor]
        assert editor._markdown_content_lines is None
        assert editor._markdown_rendered_lines == {}


async def test_markdown_styles_do_not_change_source_or_selection() -> None:
    source = "# Heading\n*emphasis*\n`inline`\n```python\nx = 1\n```"
    editor = MessageEditor(text=source)
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 24)):
        editor.selection = Selection((1, 1), (2, 3))
        lines = [editor.get_line(index) for index in range(editor.document.line_count)]

        assert editor.text == source
        assert editor.cursor_location == (2, 3)
        assert editor.selected_text == "emphasis*\n`in"
        assert lines[0].spans
        assert lines[1].spans
        assert lines[2].spans
        assert lines[4].spans


async def test_document_edit_invalidates_rendered_lines_synchronously() -> None:
    editor = MessageEditor(text="before")
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 24)):
        assert editor.get_line(0).plain == "before"

        editor.replace("after", (0, 0), (0, len("before")))

        assert editor.get_line(0).plain == "after"


async def test_undo_and_redo_invalidate_rendered_lines_synchronously() -> None:
    editor = MessageEditor(text="hello")
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 24)):
        editor.insert(" world", (0, len("hello")))
        assert editor.get_line(0).plain == "hello world"

        editor.undo()
        assert editor.text == "hello"
        assert editor.get_line(0).plain == "hello"

        editor.redo()
        assert editor.text == "hello world"
        assert editor.get_line(0).plain == "hello world"


async def test_unclosed_fence_matches_same_source_lines_when_closed() -> None:
    open_editor = MessageEditor(text="```python\nvalue = 1")
    closed_editor = MessageEditor(text="```python\nvalue = 1\n```")

    class _PairApp(App[None]):
        def compose(self) -> ComposeResult:
            yield open_editor
            yield closed_editor

    async with _PairApp().run_test(size=(100, 24)):
        for line_index in range(open_editor.document.line_count):
            open_line = open_editor.get_line(line_index)
            closed_line = closed_editor.get_line(line_index)
            assert open_line.plain == closed_line.plain
            assert _span_signature(open_line) == _span_signature(closed_line)
        assert open_editor.text == "```python\nvalue = 1"


async def test_whole_document_and_rendered_line_caches_are_lazy_and_return_copies() -> None:
    editor = _CountingMessageEditor("# Heading\nbody")
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 24)):
        assert editor.highlight_calls == 1

        first = editor.get_line(0)
        editor.get_line(1)
        again = editor.get_line(0)

        assert editor.highlight_calls == 1
        assert first is not again
        first.append(" mutated")
        assert editor.get_line(0).plain == "# Heading"


async def test_changed_and_style_updates_invalidate_both_cache_levels() -> None:
    editor = _CountingMessageEditor("# Heading")
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 24)) as pilot:
        assert editor.highlight_calls == 1

        editor.move_cursor((0, len(editor.text)))
        editor.insert("!")
        await pilot.pause()
        editor.get_line(0)
        assert editor.highlight_calls == 2

        editor.notify_style_update()
        editor.get_line(0)
        assert editor.highlight_calls == 3


@pytest.mark.parametrize(
    ("length", "degraded"),
    [
        (MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF - 1, False),
        (MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF, True),
        (MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF + 1, True),
    ],
)
async def test_character_cutoff_boundaries(length: int, degraded: bool) -> None:
    editor = MessageEditor(text="x" * length)
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 12)):
        assert editor.markdown_highlighting_degraded is degraded
        expected_status = MARKDOWN_HIGHLIGHT_PLAIN_STATUS.bind() if degraded else ""
        assert editor.markdown_highlight_status == expected_status
        if degraded:
            assert editor.get_line(0).spans == []


@pytest.mark.parametrize(
    ("line_count", "degraded"),
    [
        (MARKDOWN_HIGHLIGHT_LINE_CUTOFF - 1, False),
        (MARKDOWN_HIGHLIGHT_LINE_CUTOFF, False),
        (MARKDOWN_HIGHLIGHT_LINE_CUTOFF + 1, True),
    ],
)
async def test_line_cutoff_boundaries(line_count: int, degraded: bool) -> None:
    editor = MessageEditor(text="\n".join(["x"] * line_count))
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 12)):
        assert editor.document.line_count == line_count
        assert editor.markdown_highlighting_degraded is degraded


async def test_highlighting_automatically_resumes_below_both_cutoffs() -> None:
    editor = MessageEditor(text="x" * MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF)
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 12)) as pilot:
        assert editor.markdown_highlighting_degraded is True

        editor.text = "# resumed"
        await pilot.pause()

        assert editor.markdown_highlighting_degraded is False
        assert _render_status(editor.status_text) != format_message(MARKDOWN_HIGHLIGHT_PLAIN_STATUS.bind())
        assert editor.get_line(0).spans


async def test_editor_paste_normalizes_crlf_and_uses_shared_cap() -> None:
    editor = MessageEditor()
    app = _EditorApp(editor)

    async with app.run_test() as pilot:
        await editor._on_paste(Paste("a\r\nb\rc"))
        await pilot.pause()
        assert editor.text == "a\nb\nc"

        editor.clear()
        with patch(
            "chrys.app.tui.widgets.text_area._truncate_to_tokens",
            return_value=("truncated", MESSAGE_EDITOR_PASTE_MAX_TOKENS + 1),
        ) as truncate:
            await editor._on_paste(Paste("huge"))
        assert editor.text == "truncated"
        truncate.assert_called_once_with("huge", MESSAGE_EDITOR_PASTE_MAX_TOKENS)


async def test_large_paste_enters_plain_mode_and_clear_resumes_highlighting() -> None:
    editor = MessageEditor()
    app = _EditorApp(editor)

    async with app.run_test(size=(100, 12)) as pilot:
        await editor._on_paste(Paste("x" * MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF))
        await pilot.pause()
        assert editor.markdown_highlighting_degraded is True
        assert _render_status(editor.status_text) == format_message(MARKDOWN_HIGHLIGHT_PLAIN_STATUS.bind())

        editor.clear()
        editor.insert("# small")
        await pilot.pause()
        assert editor.markdown_highlighting_degraded is False
        assert editor.get_line(0).spans


async def test_editor_enforces_hard_character_limit_and_accounts_for_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "x" * (MESSAGE_EDITOR_MAX_CHARACTERS - 2)
    editor = MessageEditor(text=source)
    app = _EditorApp(editor)
    notifications: list[str] = []
    monkeypatch.setattr(
        editor,
        "notify",
        lambda message, **_kwargs: notifications.append(str(message)),
    )

    async with app.run_test(size=(100, 12)) as pilot:
        editor.move_cursor(editor.document.end)
        editor.insert("abcd")
        await pilot.pause()

        assert len(editor.text) == MESSAGE_EDITOR_MAX_CHARACTERS
        assert editor.text.endswith("ab")
        assert _render_status(editor.status_text) == format_message(MESSAGE_EDITOR_CHARACTER_LIMIT_STATUS.bind())
        assert notifications == ["Editor content is limited to 500,000 characters."]

        editor.replace("replacement", (0, 0), editor.document.end)
        await pilot.pause()

        assert editor.text == "replacement"
        assert _render_status(editor.status_text) == "Ready"


async def test_editor_status_definitions_keep_english_and_render_chinese() -> None:
    chinese = Localizer("zh-Hans")

    assert format_message(MESSAGE_EDITOR_CHARACTER_LIMIT_STATUS.bind()) == ("Character limit reached · 500,000 maximum")
    assert format_message(MARKDOWN_HIGHLIGHT_PLAIN_STATUS.bind()) == (
        "PLAIN · Markdown highlighting paused for large draft"
    )
    assert chinese.render(MESSAGE_EDITOR_CHARACTER_LIMIT_STATUS.bind()) == "已达到字符限制 · 最大 500,000"
    assert chinese.render(MARKDOWN_HIGHLIGHT_PLAIN_STATUS.bind()) == "纯文本 · 大型草稿已暂停 Markdown 高亮"


async def test_editor_rejects_oversize_initial_and_reloaded_documents() -> None:
    oversize = "x" * (MESSAGE_EDITOR_MAX_CHARACTERS + 1)

    with pytest.raises(ValueError, match="exceeds 500,000 characters"):
        MessageEditor(text=oversize)

    editor = MessageEditor()
    with pytest.raises(ValueError, match="exceeds 500,000 characters"):
        editor.text = oversize


async def test_standard_printable_key_inserts_once() -> None:
    editor = MessageEditor()
    app = _EditorApp(editor)

    async with app.run_test() as pilot:
        editor.focus()
        await wait_for(lambda: editor.has_focus, pilot=pilot, description="control focus before interaction")
        await pilot.press("x")
        assert editor.text == "x"


async def test_editor_sources_do_not_reference_textual_symbols_added_by_chrys_patches() -> None:
    """Keep editor imports independent of Chrys-only Textual extensions.

    CI's freshly synchronized environment is the real stock-Textual guarantee;
    this process may already have Chrys's on-disk patches, so it must not claim
    to exercise an unpatched installation.
    """
    from chrys.foundation import patches as patches_package
    from chrys.foundation.patches import (
        patcher,
        textual_block_border,
        textual_button_ansi,
        textual_callback_cache,
        textual_compositor_cjk,
        textual_ime_cursor_anchor,
        textual_kitty_keyboard,
        textual_message_pump,
        textual_option_list,
        textual_tab_selection,
        textual_utf8_decoder,
    )

    registration_modules = (
        textual_block_border,
        textual_button_ansi,
        textual_callback_cache,
        textual_compositor_cjk,
        textual_ime_cursor_anchor,
        textual_kitty_keyboard,
        textual_message_pump,
        textual_option_list,
        textual_tab_selection,
        textual_utf8_decoder,
    )
    assert registration_modules

    definition_pattern = re.compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")
    patch_added_symbols: set[str] = set()
    for file_patch in patcher._patches:
        if file_patch.package != "textual":
            continue
        old_definitions = set(definition_pattern.findall(file_patch.old_fragment))
        patch_added_symbols.update(set(definition_pattern.findall(file_patch.new_fragment)) - old_definitions)

    patch_dir = Path(patches_package.__file__).parent
    for patch_source in patch_dir.glob("textual_*.py"):
        tree = ast.parse(patch_source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id[:1].isupper() or target.value.id.endswith("_mod"):
                            patch_added_symbols.add(target.attr)
                    elif (
                        isinstance(target, ast.Name)
                        and target.id.endswith("PATCH_MARKER")
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                    ):
                        patch_added_symbols.add(node.value.value)

    source_root = Path(__file__).parents[5] / "src" / "chrys" / "app" / "tui"
    editor_sources = [
        *sorted((source_root / "widgets" / "editor").glob("*.py")),
        source_root / "screens" / "dialogs" / "editor.py",
    ]
    referenced_symbols: set[str] = set()
    for editor_source in editor_sources:
        tree = ast.parse(editor_source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None and node.module.startswith("textual"):
                referenced_symbols.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Attribute):
                referenced_symbols.add(node.attr)

    assert patch_added_symbols
    assert referenced_symbols.isdisjoint(patch_added_symbols), sorted(referenced_symbols & patch_added_symbols)
