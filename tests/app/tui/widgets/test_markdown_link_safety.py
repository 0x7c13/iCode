# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Markdown external links retain their encoding while unsafe targets are blocked."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from markdown_it import MarkdownIt
from markdown_it.rules_core.state_core import StateCore
from textual.app import App

from chrys.app.tui.widgets.markdown.links import external_link_target, terminal_link_target
from chrys.app.tui.widgets.markdown.parser import _create_markdown_parser
from chrys.app.tui.widgets.markdown.viewer import VirtualizedMarkdownViewer
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.waiting import wait_for


@pytest.mark.parametrize(
    "href",
    [
        "%66ile:///tmp/payload",
        "%66ile:///C:/Users/me/payload.bat",
        "%6aavascript:alert(1)",
        "%64ata:text/html,payload",
        "%76bscript:payload",
        "%2566ile:///tmp/payload",
        "//server/share/payload.bat",
        "C:/payload.bat",
        "https://example.com/%0D%0Afile:///etc/passwd",
        "%09https://example.com",
        "ht%09tps://example.com",
        "%0Bhttps://example.com",
        "%0Chttps://example.com",
        "%00https://example.com",
        "https://example.com/%1F",
        "https://example.com/%7F",
        "%20https://example.com",
        "ht%20tps://example.com",
        "%68ttps://example.com",
        "vscode://file/tmp/payload",
        "https://example.com/%C2%9C",
        "https://",
        "https:///path",
        "https://?q=value",
        "https://user@",
        "https://example.com:invalid/",
        "https://example.com:65536/",
        "https://%20/",
    ],
)
async def test_decoded_unsafe_markdown_link_never_reaches_platform_opener(href: str) -> None:
    tokens = _create_markdown_parser().parse(f"[click]({href})")
    link = next(child for token in tokens for child in token.children or [] if child.type == "link_open")
    app = App()
    async with app.run_test():
        widget = VirtualizedMarkdown(f"[click]({href})")
        await app.mount(widget)
        await wait_for(lambda: widget.region.width > 0 and bool(widget._blocks))
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert "\x1b]8;" not in output
        with patch.object(app, "open_url", autospec=True) as opened:
            event = VirtualizedMarkdown.LinkClicked(widget, link.attrGet("href"))
            widget.on_virtualized_markdown_link_clicked(event)
        opened.assert_not_called()


@pytest.mark.parametrize(
    "href",
    [
        "https://example.com/a%20b?c=1",
        "https://example.com/My%20Report.pdf",
        "HTTP://example.com",
        "mailto:me@example.com",
        "mailto:me@example.com?subject=Hello%20world",
        "https://example.com/a%2Fb?value=%23%26%3F",
        "https://example.com/a%2520b",
        "https://example.com/%C2%A0",
        "https://example.com/%E2%80%A8",
        "http://localhost:3000/path",
        "https://127.0.0.1:443/path",
        "http://[::1]:8080/path",
        "https://user:pass@[2001:db8::1]:443/a%5Bb%5D?q=%5Bvalue%5D#end",
        "https://xn--r8jz45g.jp/path",
    ],
)
async def test_allowed_markdown_links_open_with_original_encoding(href: str) -> None:
    tokens = _create_markdown_parser().parse(f"[click]({href})")
    link = next(child for token in tokens for child in token.children or [] if child.type == "link_open")
    app = App()
    async with app.run_test():
        widget = VirtualizedMarkdown(f"[click]({href})")
        await app.mount(widget)
        await wait_for(lambda: widget.region.width > 0 and bool(widget._blocks))
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert f";{href}\x1b\\" in output
        with patch.object(app, "open_url", autospec=True) as opened:
            event = VirtualizedMarkdown.LinkClicked(widget, link.attrGet("href"))
            widget.on_virtualized_markdown_link_clicked(event)
        opened.assert_called_once_with(href)


@pytest.mark.parametrize(
    "href",
    [
        " https://example.com",
        "https://example.com/My Report.pdf",
        "https://example.com/\t",
        "https://example.com/\r\nfile:///etc/passwd",
        "https://example.com/\x00",
        "https://example.com/\x7f",
        "https://example.com/\u00a0",
        "https://example.com/\u2028",
        "https://[invalid/path",
    ],
)
async def test_literal_whitespace_and_controls_never_reach_platform_opener(href: str) -> None:
    app = App()
    async with app.run_test():
        widget = VirtualizedMarkdown()
        await app.mount(widget)
        with patch.object(app, "open_url", autospec=True) as opened:
            widget.on_virtualized_markdown_link_clicked(VirtualizedMarkdown.LinkClicked(widget, href))
        opened.assert_not_called()


@pytest.mark.parametrize("size", [2047, 2048, 2049])
@pytest.mark.parametrize("non_ascii", [False, True])
def test_terminal_limit_counts_utf8_bytes_without_limiting_ordinary_clicks(size: int, non_ascii: bool) -> None:
    prefix = "https://example.com/" + ("中文" if non_ascii else "")
    href = prefix + "a" * (size - len(prefix.encode("utf-8")))
    assert len(href.encode("utf-8")) == size
    assert external_link_target(href) == href
    assert terminal_link_target(href) == (href if size <= 2048 else None)


def test_terminal_target_rejects_unencodable_custom_parser_input() -> None:
    assert terminal_link_target("https://example.com/\udcff") is None


@pytest.mark.parametrize("href", ["./notes.md", "foo/bar.md"])
async def test_relative_chat_links_do_not_launch_external_apps(href: str) -> None:
    app = App()
    async with app.run_test():
        widget = VirtualizedMarkdown(f"[local]({href})")
        await app.mount(widget)
        await wait_for(lambda: widget.region.width > 0 and bool(widget._blocks))
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert "\x1b]8;" not in output
        with patch.object(app, "open_url", autospec=True) as opened:
            widget.on_virtualized_markdown_link_clicked(VirtualizedMarkdown.LinkClicked(widget, href))
        opened.assert_not_called()


@pytest.mark.parametrize(
    ("href", "target_path"),
    [("./notes.md", "notes.md"), ("foo/bar.md", "foo/bar.md"), ("./My%20Notes.md", "My Notes.md")],
)
async def test_relative_links_still_navigate_inside_document_viewer(tmp_path, href: str, target_path: str) -> None:
    start = tmp_path / "index.md"
    start.write_text(f"[next]({href})", encoding="utf-8")
    target = (tmp_path / target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# Local notes", encoding="utf-8")
    app = App()
    async with app.run_test():
        viewer = VirtualizedMarkdownViewer()
        await app.mount(viewer)
        await viewer.go(start)
        with patch.object(app, "open_url", autospec=True) as opened:
            viewer.document.post_message(VirtualizedMarkdown.LinkClicked(viewer.document, href))
            await wait_for(lambda: viewer.navigator.location == target and viewer.document.source == "# Local notes")
        opened.assert_not_called()


async def test_local_anchor_can_contain_decoded_spaces() -> None:
    app = App()
    async with app.run_test():
        widget = VirtualizedMarkdown("[local](#Local%20notes)")
        await app.mount(widget)
        await wait_for(lambda: widget.region.width > 0 and bool(widget._blocks))
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert "\x1b]8;" not in output
        with patch.object(widget, "goto_anchor", autospec=True) as goto_anchor:
            widget.on_virtualized_markdown_link_clicked(VirtualizedMarkdown.LinkClicked(widget, "#Local%20notes"))
        goto_anchor.assert_called_once_with("Local notes")


@pytest.mark.parametrize("token_type", ["link_open", "image"])
@pytest.mark.parametrize(
    "href", ["https://example.com/\x1b\\payload", "https://example.com/\x07payload", "https://example.com/\x9cpayload"]
)
async def test_custom_parser_control_targets_never_escape_into_terminal(token_type: str, href: str) -> None:
    """A parser factory can supply raw destinations that bypass markdown-it normalization."""
    attribute = "href" if token_type == "link_open" else "src"

    def replace_target(state: StateCore) -> None:
        for token in state.tokens:
            for child in token.children or []:
                if child.type == token_type:
                    child.attrSet(attribute, href)

    def parser_factory() -> MarkdownIt:
        parser = _create_markdown_parser()
        parser.core.ruler.after("inline", "test_raw_target", replace_target)
        return parser

    app = App()
    async with app.run_test() as pilot:
        prefix = "!" if token_type == "image" else ""
        widget = VirtualizedMarkdown(f"{prefix}[click](https://example.com/)", parser_factory=parser_factory)
        await app.mount(widget)
        await wait_for(lambda: widget.region.width > 0 and bool(widget._blocks), pilot=pilot)
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert "\x1b]8;" not in output
        assert "\x9c" not in output
        assert "click" in output
        with patch.object(app, "open_url", autospec=True) as opened:
            widget.on_virtualized_markdown_link_clicked(VirtualizedMarkdown.LinkClicked(widget, href))
        opened.assert_not_called()
