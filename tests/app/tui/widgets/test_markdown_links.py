# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Markdown link-boundary regression tests."""

from __future__ import annotations

import pytest
from markdown_it import MarkdownIt
from markdown_it.rules_core.linkify import linkify as _stock_core_linkify
from markdown_it.rules_inline.linkify import linkify as _stock_inline_linkify
from textual.content import Content

from chrys.app.tui.widgets.markdown.parser import (
    _configure_markdown_parser,
    _create_markdown_parser,
    _token_to_content,
)


def _parse_inline(markdown: str, *, parser: MarkdownIt | None = None) -> Content:
    parser = parser if parser is not None else _create_markdown_parser()
    inline = next(token for token in parser.parse(markdown) if token.type == "inline")
    return _token_to_content(inline)


@pytest.mark.parametrize(
    "text",
    ["README.md", "script.sh", "foo.rs", "example.com", "www.example.com", "me@example.com", "v1.2.3", "requests.get"],
)
def test_bare_filenames_identifiers_and_domains_stay_plain(text: str) -> None:
    content = _parse_inline(f"{text} https://example.com/\uff0c正文")
    assert content.plain == f"{text} https://example.com/\uff0c正文"
    assert len(content.spans) == 1
    span = content.spans[0]
    assert (span.start, span.end) == (len(text) + 1, len(text) + 1 + len("https://example.com/"))
    assert span.style.link == "https://example.com/"


@pytest.mark.parametrize("text", ["//example.com/a", "ftp://example.com/a", "mailto:me@example.com"])
def test_bare_non_http_schemes_are_not_automatically_linked(text: str) -> None:
    content = _parse_inline(text)
    assert content.plain == text
    assert not content.spans


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("[文档](https://example.com/)", "https://example.com/"),
        ("<https://example.com/>", "https://example.com/"),
        ("[文档][ref]\n\n[ref]: https://example.com/", "https://example.com/"),
        ("[邮箱](mailto:me@example.com)", "mailto:me@example.com"),
        ("<me@example.com>", "mailto:me@example.com"),
        ("<http://[::1]:8080/path>", "http://[::1]:8080/path"),
    ],
)
def test_explicit_links_remain_available_without_fuzzy_matching(source: str, target: str) -> None:
    content = _parse_inline(source)
    assert len(content.spans) == 1
    assert content.spans[0].style.link == target
    assert content.spans[0].style.meta == {"@click": f"link({target!r})"}


@pytest.mark.parametrize(
    "boundary",
    "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\u2026\u2025\uff0e\uff61\uff64"
    "\uff08\uff09\u3010\u3011\u300a\u300b\u300c\u300d\u300e\u300f"
    "\uff3b\uff3d\uff5b\uff5d\u3008\u3009\u3014\u3015\u3016\u3017"
    "\u3018\u3019\u301a\u301b\uff62\uff63\u301d\u301e\u301f\uff5f\uff60"
    "\u201c\u201d\u2018\u2019\uff02\uff07\uff1c\uff1e"
    "\ufe10\ufe11\ufe12\ufe13\ufe14\ufe15\ufe16\ufe17\ufe18\ufe19\ufe30"
    "\ufe35\ufe36\ufe37\ufe38\ufe39\ufe3a\ufe3b\ufe3c\ufe3d\ufe3e"
    "\ufe3f\ufe40\ufe41\ufe42\ufe43\ufe44\ufe47\ufe48"
    "\ufe50\ufe51\ufe52\ufe54\ufe55\ufe56\ufe57"
    "\ufe59\ufe5a\ufe5b\ufe5c\ufe5d\ufe5e\ufe64\ufe65",
)
def test_linkified_url_stops_before_cjk_punctuation(boundary: str) -> None:
    """CJK prose punctuation immediately after a bare URL is not clickable."""
    url = "http://127.0.0.1:45237/"
    content = _parse_inline(f"{url}{boundary}记录在这里")

    assert content.plain == f"{url}{boundary}记录在这里"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len(url))
    assert link_span.style.meta == {"@click": f"link({url!r})"}


def test_paren_wrapped_url_stops_at_closing_delimiter() -> None:
    """A URL wrapped in full-width parentheses is clickable without the closing one."""
    prefix = "访问\uff08"
    url = "http://example.com/路径"
    content = _parse_inline(f"{prefix}{url}\uff09查看")

    assert content.plain == f"{prefix}{url}\uff09查看"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (len(prefix), len(prefix) + len(url))
    assert link_span.style.meta == {"@click": "link('http://example.com/%E8%B7%AF%E5%BE%84')"}


def test_linkified_url_keeps_percent_encoded_punctuation() -> None:
    """Punctuation explicitly percent-encoded into a bare URL stays clickable."""
    source = "http://example.com/search?q=%E4%B8%AD%E6%96%87%EF%BC%8Ctest"
    visible = "http://example.com/search?q=中文\uff0ctest"
    content = _parse_inline(source)

    assert content.plain == visible
    assert len(content.spans) == 1
    assert (content.spans[0].start, content.spans[0].end) == (0, len(visible))
    assert content.spans[0].style.meta == {"@click": f"link({source!r})"}


def test_encoded_punctuation_before_a_literal_boundary_stays_in_the_href() -> None:
    """A literal boundary still splits while earlier encoded punctuation is preserved."""
    content = _parse_inline("http://example.com/%EF%BC%8Cx\uff0c后文")

    assert content.plain == "http://example.com/\uff0cx\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("http://example.com/\uff0cx"))
    assert link_span.style.meta == {"@click": "link('http://example.com/%EF%BC%8Cx')"}


def test_percent_decoded_ascii_does_not_hide_a_literal_boundary() -> None:
    """An earlier decoded ASCII escape does not prevent the later literal split."""
    content = _parse_inline("http://e.com/a%28b\uff0c后文")

    assert content.plain == "http://e.com/a(b\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("http://e.com/a(b"))
    assert link_span.style.meta == {"@click": "link('http://e.com/a%28b')"}


def test_verbatim_percent_escape_does_not_derail_boundary_alignment() -> None:
    """A visible ``%25`` sequence is aligned literally before the CJK split."""
    content = _parse_inline("http://e.com/a%25b\uff0c后文")

    assert content.plain == "http://e.com/a%25b\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("http://e.com/a%25b"))
    assert link_span.style.meta == {"@click": "link('http://e.com/a%25b')"}


def test_custom_parser_fuzzy_link_uses_its_exact_percent_encoded_source() -> None:
    """A custom parser may still opt into fuzzy links without losing provenance."""
    content = _parse_inline(
        "www.example.com/a%28b\uff0c后文", parser=_configure_markdown_parser(MarkdownIt("gfm-like"))
    )

    assert content.plain == "www.example.com/a(b\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("www.example.com/a(b"))
    assert link_span.style.meta == {"@click": "link('http://www.example.com/a%28b')"}


def test_url_quoted_in_code_span_does_not_shadow_a_later_bare_link() -> None:
    """Raw-source anchoring skips an identical URL inside an earlier code span."""
    url = "http://example.com/"
    content = _parse_inline(f"`{url}` {url}\uff0c后文")

    assert content.plain == f"{url} {url}\uff0c后文"
    assert len(content.spans) == 2
    assert content.spans[0].style == ".code_inline"
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (len(url) + 1, len(url) + 1 + len(url))
    assert link_span.style.meta == {"@click": f"link({url!r})"}


def test_prior_identical_bare_link_does_not_break_the_split() -> None:
    """An earlier identical bare URL without punctuation does not defeat alignment."""
    url = "http://a.com/x"
    content = _parse_inline(f"{url} {url}\uff0c后文")

    assert content.plain == f"{url} {url}\uff0c后文"
    assert len(content.spans) == 2
    assert (content.spans[0].start, content.spans[0].end) == (0, len(url))
    second = content.spans[1]
    assert (second.start, second.end) == (len(url) + 1, len(url) + 1 + len(url))
    assert second.style.meta == {"@click": f"link({url!r})"}


def test_prior_explicit_link_does_not_break_the_split() -> None:
    """An earlier explicit link to the same URL does not defeat alignment."""
    url = "http://example.com/"
    content = _parse_inline(f"[说明]({url}) {url}\uff0c后文")

    assert content.plain == f"说明 {url}\uff0c后文"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (3, 3 + len(url))
    assert link_span.style.meta == {"@click": f"link({url!r})"}


def test_alignment_rejects_prior_construct_with_literal_punctuation() -> None:
    """Full-display alignment keeps encoded URL content even when an earlier
    construct spells the same prefix with the punctuation written literally."""
    content = _parse_inline("[x](http://e.com/\uff0cy) http://e.com/%EF%BC%8Cz后有\u3002终")

    assert content.plain == "x http://e.com/\uff0cz后有\u3002终"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    visible = "http://e.com/\uff0cz后有"
    assert (link_span.start, link_span.end) == (2, 2 + len(visible))
    assert link_span.style.meta == {"@click": "link('http://e.com/%EF%BC%8Cz%E5%90%8E%E6%9C%89')"}


def test_encoded_explicit_destination_does_not_mask_a_literal_boundary() -> None:
    """An encoded twin in a prior explicit-link destination cannot suppress the split."""
    content = _parse_inline("[x](http://e.com/%EF%BC%8Cz) http://e.com/\uff0cz")

    assert content.plain == "x http://e.com/\uff0cz"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (2, 2 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_literal_explicit_destination_does_not_truncate_an_encoded_link() -> None:
    """A literal twin in a prior explicit-link destination cannot force a split."""
    content = _parse_inline("[x](http://e.com/\uff0cz) http://e.com/%EF%BC%8Cz")

    assert content.plain == "x http://e.com/\uff0cz"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (2, 2 + len("http://e.com/\uff0cz"))
    assert link_span.style.meta == {"@click": "link('http://e.com/%EF%BC%8Cz')"}


def test_autolink_destination_does_not_shadow_a_later_bare_link() -> None:
    """An encoded twin inside an autolink cannot suppress the split."""
    content = _parse_inline("<http://e.com/%EF%BC%8Cz> http://e.com/\uff0cz")

    assert content.plain == "http://e.com/\uff0cz http://e.com/\uff0cz"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (16, 16 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_image_destination_does_not_shadow_a_later_bare_link() -> None:
    """An encoded twin in an image destination cannot suppress the split."""
    content = _parse_inline("![x](http://e.com/%EF%BC%8Cz) http://e.com/\uff0cz")

    assert content.plain == "\U0001f5bc  x http://e.com/\uff0cz"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (5, 5 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_reference_link_destination_does_not_swallow_a_later_bare_link() -> None:
    """A reference link's out-of-line destination never claims the bare link's raw text."""
    content = _parse_inline("[x][ref] http://e.com/\uff0cz\n\n[ref]: http://e.com/%EF%BC%8Cz")

    assert content.plain == "x http://e.com/\uff0cz"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (2, 2 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_literal_angle_bracket_before_a_bare_link_still_splits() -> None:
    """A lone literal < before a bare URL is prose, not an autolink opener."""
    content = _parse_inline("<http://e.com/\uff0c后文")

    assert content.plain == "<http://e.com/\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (1, 1 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_literal_bracket_paren_before_a_bare_link_still_splits() -> None:
    """A literal ]( before a bare URL is prose, not an inline destination opener."""
    content = _parse_inline("](http://e.com/\uff0c后文")

    assert content.plain == "](http://e.com/\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (2, 2 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_unrelated_angle_text_does_not_hide_the_real_destination() -> None:
    """A prose ``<`` cannot make an explicit destination shadow a later bare URL."""
    content = _parse_inline("<junk [x](http://e.com/%EF%BC%8Cz) http://e.com/\uff0cz")

    assert content.plain == "<junk x http://e.com/\uff0cz"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (8, 8 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_escaped_backtick_does_not_hide_the_real_code_span() -> None:
    """An escaped backtick cannot make code content shadow a later bare URL."""
    content = _parse_inline("\\` `http://e.com/%EF%BC%8Cz` http://e.com/\uff0cz")

    code_text = "http://e.com/%EF%BC%8Cz"
    prefix = f"` {code_text} "
    assert content.plain == f"{prefix}http://e.com/\uff0cz"
    assert len(content.spans) == 2
    assert (content.spans[0].start, content.spans[0].end) == (2, 2 + len(code_text))
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (len(prefix), len(prefix) + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_reference_link_title_does_not_swallow_a_later_bare_link() -> None:
    """A reference link's out-of-line title never claims a bare link's raw text."""
    content = _parse_inline('[x][ref] "http://e.com/\uff0cz\n\n[ref]: /dest "http://e.com/\uff0cz"')

    assert content.plain == 'x "http://e.com/\uff0cz'
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (3, 3 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_reference_destination_cannot_claim_an_angle_prefixed_bare_link() -> None:
    """A reference destination twin does not advance over a <-prefixed bare link."""
    content = _parse_inline("[a][r] <http://e.com/\uff0cz\n\n[r]: http://e.com/\uff0cz")

    assert content.plain == "a <http://e.com/\uff0cz"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (3, 3 + len("http://e.com/"))
    assert link_span.style.meta == {"@click": "link('http://e.com/')"}


def test_entity_normalized_destination_is_bounded_to_its_slot() -> None:
    """An entity-normalized destination cannot claim a later twin's slot."""
    source = "[x](http://e.com/a&amp;b) http://target.com/\uff0c后文 [y](http://e.com/a&b)"
    content = _parse_inline(source)

    assert content.plain == "x http://target.com/\uff0c后文 y"
    assert len(content.spans) == 3
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (2, 2 + len("http://target.com/"))
    assert link_span.style.meta == {"@click": "link('http://target.com/')"}


def test_backslash_escaped_destination_is_bounded_to_its_slot() -> None:
    """A backslash-escaped destination cannot claim a later twin's slot."""
    source = "[x](http://e.com/a\\)b) http://target.com/\uff0c后文 [y](http://e.com/a)b)"
    content = _parse_inline(source)

    assert content.plain == "x http://target.com/\uff0c后文 yb)"
    assert len(content.spans) == 3
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (2, 2 + len("http://target.com/"))
    assert link_span.style.meta == {"@click": "link('http://target.com/')"}


def test_code_span_normalization_does_not_derail_the_cursor() -> None:
    """A code span whose rendered text differs from its raw source is skipped structurally."""
    content = _parse_inline("`abc\ndef` http://target.com/\uff0c后文 abc def")

    assert content.plain == "abc def http://target.com/\uff0c后文 abc def"
    assert len(content.spans) == 2
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (8, 8 + len("http://target.com/"))
    assert link_span.style.meta == {"@click": "link('http://target.com/')"}


def test_bare_link_keeps_entities_verbatim() -> None:
    """Linkify renders ``&...;`` verbatim; alignment must not treat it as decoded."""
    content = _parse_inline("http://e.com/a&amp;b\uff0c后文")

    assert content.plain == "http://e.com/a&amp;b\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("http://e.com/a&amp;b"))
    assert link_span.style.meta == {"@click": "link('http://e.com/a&amp;b')"}


def test_bare_link_does_not_consume_a_later_twin_destination_slot() -> None:
    """A bare link has no destination slot and must not advance the cursor into one."""
    source = "http://e.com/x\uff0c中文 http://e.com/y\uff0c后 [z](http://e.com/x)"
    content = _parse_inline(source)

    assert content.plain == "http://e.com/x\uff0c中文 http://e.com/y\uff0c后 z"
    assert len(content.spans) == 3
    link_span = content.spans[1]
    assert (link_span.start, link_span.end) == (18, 18 + len("http://e.com/y"))
    assert link_span.style.meta == {"@click": "link('http://e.com/y')"}


def test_lowercase_kept_escape_does_not_derail_boundary_alignment() -> None:
    """A verbatim-kept escape re-emitted in uppercase still aligns with its source."""
    content = _parse_inline("http://e.com/a%2cb\uff0c后文")

    assert content.plain == "http://e.com/a%2Cb\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("http://e.com/a%2Cb"))
    assert link_span.style.meta == {"@click": "link('http://e.com/a%2cb')"}


def test_punycode_host_still_splits_at_a_literal_boundary() -> None:
    """A source-punycode host renders decoded; the path is still aligned and split."""
    content = _parse_inline("http://xn--r8jz45g.jp/x\uff0c后文")

    assert content.plain == "http://例え.jp/x\uff0c后文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("http://例え.jp/x"))
    assert link_span.style.meta == {"@click": "link('http://xn--r8jz45g.jp/x')"}


def test_reference_twin_chain_does_not_swallow_a_bare_link() -> None:
    """Out-of-line twins of later destinations cannot displace a bare link's source."""
    source = "[a][ref] [y](http://e.com/x) http://e.com/x\uff0c后 [z](http://e.com/x)\n\n[ref]: http://e.com/x"
    content = _parse_inline(source)

    assert content.plain == "a y http://e.com/x\uff0c后 z"
    assert len(content.spans) == 4
    link_span = content.spans[2]
    assert (link_span.start, link_span.end) == (4, 4 + len("http://e.com/x"))
    assert link_span.style.meta == {"@click": "link('http://e.com/x')"}


def test_fullwidth_bracket_wrapped_url_stops_at_closing_delimiter() -> None:
    """A URL wrapped in fullwidth square brackets is clickable without the closing one."""
    prefix = "访问\uff3b"
    url = "http://127.0.0.1:45237/"
    content = _parse_inline(f"{prefix}{url}\uff3d查看")

    assert content.plain == f"{prefix}{url}\uff3d查看"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (len(prefix), len(prefix) + len(url))
    assert link_span.style.meta == {"@click": f"link({url!r})"}


def test_custom_parser_protocol_relative_punycode_host_still_splits() -> None:
    """A protocol-relative punycode URL recovers alignment from where its path begins."""
    content = _parse_inline("//xn--r8jz45g.jp/x\uff0c後文", parser=_configure_markdown_parser(MarkdownIt("gfm-like")))

    assert content.plain == "//例え.jp/x\uff0c後文"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("//例え.jp/x"))
    assert link_span.style.meta == {"@click": "link('//xn--r8jz45g.jp/x')"}


def test_userinfo_boundary_survives_a_punycode_host() -> None:
    """A literal boundary in userinfo splits even when the host defeats alignment."""
    content = _parse_inline("http://user\uff0cname@xn--r8jz45g.jp/x\uff0c后")

    assert content.plain == "http://user\uff0cname@例え.jp/x\uff0c后"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len("http://user"))
    assert link_span.style.meta == {"@click": "link('http://user')"}


def test_factory_supplied_linkify_rules_are_preserved() -> None:
    """Custom linkify rules from a parser factory run unreplaced; splitting degrades."""
    calls: list[str] = []

    def custom_inline(state, silent):
        calls.append("inline")
        return _stock_inline_linkify(state, silent)

    def custom_core(state):
        calls.append("core")
        _stock_core_linkify(state)

    parser = MarkdownIt("gfm-like")
    parser.inline.ruler.at("linkify", custom_inline)
    parser.core.ruler.at("linkify", custom_core)
    assert _configure_markdown_parser(parser) is parser

    inline = next(token for token in parser.parse("http://e.com/x\uff0c后") if token.type == "inline")
    content = _token_to_content(inline)

    assert "inline" in calls
    assert "core" in calls
    assert content.plain == "http://e.com/x\uff0c后"
    assert len(content.spans) == 1
    link_span = content.spans[0]
    assert (link_span.start, link_span.end) == (0, len(content.plain))


def test_linkified_url_keeps_a_chinese_path() -> None:
    """Ordinary Chinese path characters remain part of a bare URL."""
    url = "http://example.com/中文路径"
    content = _parse_inline(url)

    assert len(content.spans) == 1
    assert (content.spans[0].start, content.spans[0].end) == (0, len(url))


@pytest.mark.parametrize(
    ("markdown", "visible"),
    (
        ("<http://example.com/中文路径\uff08版本\uff09>", "http://example.com/中文路径\uff08版本\uff09"),
        ("[说明](http://example.com/中文路径\uff08版本\uff09)", "说明"),
    ),
)
def test_explicit_links_keep_cjk_punctuation(markdown: str, visible: str) -> None:
    """Explicit link syntax remains authoritative for intentional CJK URLs."""
    content = _parse_inline(markdown)

    assert content.plain == visible
    assert len(content.spans) == 1
    assert (content.spans[0].start, content.spans[0].end) == (0, len(visible))


def test_linkified_url_keeps_balanced_ascii_parentheses() -> None:
    """The upstream ASCII-parenthesis balancing behavior is unchanged."""
    url = "http://example.com/reference_(topic)"
    content = _parse_inline(url)

    assert len(content.spans) == 1
    assert (content.spans[0].start, content.spans[0].end) == (0, len(url))
