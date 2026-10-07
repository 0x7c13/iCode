# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Math integration: source fidelity, streaming, selection and narrow viewports."""

from __future__ import annotations

import pytest
from markdown_it.rules_block import StateBlock
from textual.app import App, ComposeResult
from textual.selection import SELECT_ALL

from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.markdown.parser import (
    _create_markdown_parser,
    _parse_tokens,
    create_user_text_markdown_parser,
)
from tests.support.tui_helpers import resize_when_settled


@pytest.mark.parametrize(
    "source",
    [
        "```math\n\\frac{a}{b}\n```",
        "$$\\frac{a}{b}$$",
        "$$\n\\frac{a}{b}\n$$",
        r"\[\frac{a}{b}\]",
        "\\[\n\\frac{a}{b}\n\\]",
    ],
)
def test_all_display_delimiters(source: str) -> None:
    blocks = _parse_tokens(_create_markdown_parser().parse(source))
    assert len(blocks) == 1
    block = blocks[0]
    assert block.block_type == "math"
    assert block.math is not None
    assert block.math.rows == (" a ", "---", " b ")


@pytest.mark.parametrize(
    "source",
    [
        "$5 and $10",
        "US$5 / US$10",
        "$HOME/$USER",
        "${HOME}/${USER}",
        "$PATH:$HOME",
        "$var_name/$file_name",
        "$HOME/.config",
        "echo $1 $2 $? $$",
        "$HOME and $USER",
        "$20 per person, $30 total",
        "$ not math $",
        "$hello world$",
        r"\$5 and \$10",
        "$x+1$2",
        "$foo_bar$",
        "$NAME=$VALUE",
        "${VAR:-default}",
        "${VAR:=default}/$HOME",
        "${VAR:+other}/$USER",
        "echo $((x + 1)) and $HOME",
        "$USER$",
        "$x + $HOME",
        "$x$PATH",
        "echo $x/${y}",
        "echo $?/$#",
        "echo $x/$?",
        "echo $x/$@",
        "echo $x/$*",
        "echo $x/$!",
        "echo $x$?",
        "echo $x$*",
        "echo $x$!",
        "echo $x_1$?",
        "echo $x_1$*",
        "echo $x_1$!",
        "echo prefix_$x$",
        "echo $x/$(pwd)",
        "$x + $",
        "$[0, 1]$",
        "$(a, b)$",
        "$|x|$",
        "$n!$",
        "`$x^2$`",
    ],
)
def test_prices_shell_and_code_keep_commonmark_text(source: str) -> None:
    from markdown_it import MarkdownIt

    expected = _parse_tokens(MarkdownIt("gfm-like", {"html": False}).parse(source))[0].content.plain
    actual = _parse_tokens(_create_markdown_parser().parse(source))[0].content.plain
    assert actual == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"With $x^2$ and \(\alpha_1\).", "With x² and α₁."),
        (r"中文公式\(\frac{a}{b}\)继续", "中文公式a/b继续"),
        (r"$\unknown{x}$", r"$\unknown{x}$"),
        (r"\(\unknown{x}\)", r"\(\unknown{x}\)"),
        (r"\(x^2", "(x^2"),
        (r"\[x^2", "[x^2"),
        (r"**Result:** $x^2$.", "Result: x²."),
        (r"**$x^2$**", "x²"),
        (r"*$x^2$*", "x²"),
        (r"***$x^2$***", "x²"),
        (r"_$x^2$_", "x²"),
        (r"__$x^2$__", "x²"),
        (r"_$x^2$ is squared_", "x² is squared"),
        (r"__$x^2$ is squared__", "x² is squared"),
        (r"**$x$**", "x"),
        (r"**$x_1$**", "x₁"),
        (r"_$x$_", "x"),
        (r"答案是 **$x = 42$**。", "答案是 x = 42。"),
        (r"**Answer: $x^2$**", "Answer: x²"),
        (r"Is $x^2$?", "Is x²?"),
        (r"Found $x=1$!", "Found x = 1!"),
        (r"$a/b$", "a/b"),
        (r"$x_1$", "x₁"),
        (r"其中$x^2$表示平方。", "其中x²表示平方。"),
        (r"其中 $x^2$表示平方。", "其中 x²表示平方。"),
        ("$α$", "α"),  # noqa: RUF001 - literal Greek math symbol
        ("$f(x)$", "f(x)"),
        ("$E = mc^2$", "E = mc²"),
        ("$ax^2 + bx + c = 0$", "ax² + bx + c = 0"),
        ("$x_{ij}$", "xᵢⱼ"),
        ("$e^{ix}$", "e^{ix}"),
        ("$xy = 1$", "xy = 1"),
        ("the $n$-th and $k$-means", "the n-th and k-means"),
        (r"Inline \[x\] and $$x^2$$.", "Inline x and x²."),
        (r"set $\{1, 2, 3\}$", "set {1, 2, 3}"),
        (r"norm $\|v\|$", "norm ‖v‖"),
        (r"$O(n \log n)$", "O(n log n)"),
        (r"$\{x : x > 0\}$", "{x : x > 0}"),
        (r"$P(A\mid B)$", "P(A | B)"),
        (r"$x:=5$", "x := 5"),
        (r"$90^\circ$", "90°"),
        (r"$f\circ g$", "f ∘ g"),
        (r"$n!\log n$", "n! log n"),
        (r"$\boxed{x=1}$", "x = 1"),
        (r"The answer is $\boxed{42}$", "The answer is 42"),
        (r"$\left.x^2\right|_0^1$", "x²|₀¹"),
        (r"$\left.\frac{x^3}{3}\right|_0^1$", "x³/3|₀¹"),
        (r"$(\frac12)^n$", "(1/2)ⁿ"),
        (r"$\frac{n!}{k!(n-k)!}$", "n!/(k!(n - k)!)"),
        (r"$\frac1{n!}$", "1/n!"),
        (r"$\frac{n!+1}{k!}$", "(n! + 1)/k!"),
        (r"$\frac{n!}{(n-k)!}$", "n!/(n - k)!"),
        (r"$\frac{(n+1)!}{n!}$", "(n + 1)!/n!"),
        (r"$\frac{(2n)!}{(n!)^2}$", "(2n)!/(n!)²"),
        (r"$\frac{(a+b)}{c}$", "(a + b)/c"),
        (r"$\frac{(a)+(b)!}{c}$", "((a) + (b)!)/c"),
        (r"$\frac1{(a{)+(}b)}$", "1/((a) + (b))"),
        (r"$\frac1{(a\text{)+(}b)}$", "1/((a)+(b))"),
        (r"$\hat{\beta}_1$", "hat(β)₁"),
        (r"$\hat\sigma^2$", "hat(σ)²"),  # noqa: RUF001 - Greek sigma
        (r"$\frac{\sqrt{3}}{2}$", "√(3)/2"),
        (r"$\frac{1}{\sqrt{2}}$", "1/√(2)"),
        (r"$\binom{n}{k}^2$", "binom(n, k)²"),
        (r"$\frac{y^{(k)}}{k!}$", "y^{(k)}/k!"),
        (r"$\frac{(a+b)^{(k)}}{c}$", "(a + b)^{(k)}/c"),
        (r"$\frac1{(x^{\text{)a(}}+y)}$", "1/(x^{)a(} + y)"),
        (r"$\frac{(2n-1)!!}{(2n)!!}$", "(2n - 1)!!/(2n)!!"),
        (r"$\frac{ab!!}{k}$", "(ab!!)/k"),
        (r"$f\colon A\to B$", "f: A → B"),
        (r"$\not=0$", "≠ 0"),
        (r"$-\boxed{x+y}$", "-(x + y)"),
        (r"$x-\boxed{y+z}$", "x - (y + z)"),
        (r"$\boxed{x+y}/z$", "(x + y)/z"),
        (r"$\boxed{x+y}\times z$", "(x + y) × z"),  # noqa: RUF001 - multiplication sign
        (r"Result: $\boxed{\not=0}$", "Result: ≠ 0"),
        (r"Result: ${\not=0}$", "Result: ≠ 0"),
        (r"By definition $a\stackrel{\text{def}}{=}b$.", "By definition a =^{def} b."),
        (r"求 $x=?$", "求 x = ?"),
        (r"$f(2)=?$", "f(2) = ?"),
        (r"$x=\mathbf{?}$", "x = ?"),
        (r"$a{+}b$", "a+b"),
        (r"$a{-}b$", "a-b"),
        (r"$x\mathbf{+}y$", "x+y"),
        (r"$a{+}\sin x$", "a+ sin x"),
        (r"$a\mathbf{+}-b$", "a+ - b"),
        (r"$x^e$", "x^{e}"),
        (r"$f:\mathbb{R}^d\to\mathbb{R}^m$", "f : ℝ^{d} → ℝ^{m}"),  # noqa: RUF001 - blackboard bold R
        (r"We require $f(x)\overset{!}{=}0$ here.", "We require f(x) =^{!} 0 here."),
        (r"$x\overset{?}{=}y$", "x =^{?} y"),
        (r"Minimize $\underset{\theta}{\arg\min}\,L(\theta)$.", "Minimize arg min_{θ} L(θ)."),
        (r"$|-3|=3$", "|-3| = 3"),
        (r"$\lvert-x\rvert$", "|-x|"),
        (r"$\|-v\|$", "‖-v‖"),
        (r"$\lvert\frac12\rvert^n$", "|1/2|ⁿ"),
        (r"$|\frac12|^n$", "|1/2|ⁿ"),
        (r"$(=x)$", "(= x)"),
        (r"$\sqrt{\le x}$", "√(≤ x)"),
        (r"$\begin{cases}\ge0&x\end{cases}$", "cases[≥ 0, x]"),
        (r"$P\bigl(A\bigm|B\bigr)$", "P(A | B)"),
        (r"$x\not\in S$", "x ∉ S"),
        (r"$\big x$", r"$\big x$"),
        (r"$\boxed{\unknown x}$", r"$\boxed{\unknown x}$"),
    ],
)
def test_inline_and_literal_source_rules(source: str, expected: str) -> None:
    assert _parse_tokens(_create_markdown_parser().parse(source))[0].content.plain == expected


@pytest.mark.parametrize(("marker", "style"), [("*", ".em"), ("_", ".em"), ("**", ".strong"), ("__", ".strong")])
def test_rendered_math_keeps_its_markdown_emphasis(marker: str, style: str) -> None:
    content = _parse_tokens(_create_markdown_parser().parse(marker + "$x^2$" + marker))[0].content
    assert content.plain == "x²"
    assert [(span.start, span.end, span.style) for span in content.spans] == [(0, 2, style)]


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (r"\(\Vert v\Vert\)", "‖v‖"),
        (r"$\Vert v\Vert$", "‖v‖"),
        (r"\(\\|v\\|\)", "‖v‖"),
        (r"$\\|v\\|$", "‖v‖"),
        (r"\(\lvert x\rvert\)", "|x|"),
        (r"\(\|x\|\)", "|x|"),
    ],
)
def test_table_math_respects_gfm_pipe_escaping(cell: str, expected: str) -> None:
    # GFM consumes one backslash before a pipe, before parsing inline syntax.
    blocks = _parse_tokens(_create_markdown_parser().parse(f"| expression |\n|---|\n| {cell} |"))
    assert len(blocks) == 1
    rows = blocks[0].table_rows
    assert rows is not None
    assert [[content.plain for content in row] for row in rows] == [[expected]]


@pytest.mark.parametrize("cell", [r"`\|v\|`", r"**\|v\|**"])
def test_table_pipe_escaping_keeps_code_and_emphasis(cell: str) -> None:
    from markdown_it import MarkdownIt

    source = f"| expression |\n|---|\n| {cell} |"
    ordinary = _parse_tokens(MarkdownIt("gfm-like", {"html": False}).parse(source))
    actual = _parse_tokens(_create_markdown_parser().parse(source))
    assert actual[0].table_headers == ordinary[0].table_headers
    assert actual[0].table_rows == ordinary[0].table_rows


def test_escaped_dollar_does_not_hide_adjacent_display_closure() -> None:
    block = VirtualizedMarkdown()._build_blocks(r"$$x\$$$")[0]
    assert block.block_type == "math"
    assert block.math is not None and block.math.linear == "x$"


@pytest.mark.parametrize(
    "source",
    [
        r"\[1\]",
        r"\[not a link\]",
        r"matches \(foo\)",
        r"\( **bold** and `code`",
        r"\(in progress\)",
        r"\(see note 2\)",
        r"\(TODO: fix\)",
        r"\[in progress\]",
        r"Inline \[see note 2\] here.",
        r"\[TODO: fix\]",
        r"\(**Note**\)",
        r"\(**Note:**\)",
        r"\(**Note**.\)",
        r"\(~~**_Note:_**~~\)",
        r"\(**Note:**...\)",
        r"\(*optional*\)",
        r"\(_note_\)",
        r"\(~~Note~~\)",
        r"\(~~**_Note_**~~\)",
        r"\([note](https://example.com)\)",
        r"\([note](url(foo))\)",
        r"\([note](https://en.wikipedia.org/wiki/Limit_(mathematics))\)",
        r"\(`note`\)",
        r"\[**Note**\]",
        r"\(see *note*2\)",
        r"\(some*thing* here\)",
        r"\(see **note** 2\)",
        r"\(see `note` 2\)",
        r"\(see _note_ 2\)",
        r"\(see ~~note~~ 2\)",
        r"\(see [note](guide) 2\)",
        r"\([note](https://[::1]) carefully\)",
        r"\([note](https://example.com/?a[]=1) carefully\)",
        r"\(see [note](url(foo)) 2\)",
        r"\(see [note](https://en.wikipedia.org/wiki/Limit_(mathematics)) here\)",
        r"\(see ***note*** 2\)",
        r"\(see **_note_** 2\)",
        r"\(see _**note**_ 2\)",
        r"\(see ~~**_note_**~~ 2\)",
    ],
)
def test_non_math_escapes_keep_commonmark_formatting(source: str) -> None:
    from markdown_it import MarkdownIt

    ordinary = _parse_tokens(MarkdownIt("gfm-like", {"html": False}).parse(source))
    actual = _parse_tokens(_create_markdown_parser().parse(source))
    assert [(block.block_type, block.content) for block in actual] == [
        (block.block_type, block.content) for block in ordinary
    ]


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    ("repeated", "suffix", "expected"), [("[", "x", True), ("a", "", False), ("[a](", "", True), ("[a](", ")", True)]
)
def test_explicit_math_detection_handles_long_incomplete_prose(repeated: str, suffix: str, expected: bool) -> None:
    from chrys.app.tui.widgets.markdown.math.markdown import bracket_is_math

    # Failed link and word matches must not restart a scan of every suffix.
    assert bracket_is_math(repeated * 100_000 + suffix) is expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("AB", "AB"),
        ("|x|", "|x|"),
        ("n!", "n!"),
        ("3x", "3x"),
        ("a,b", "a, b"),
        ("f'(x)", "f'(x)"),
        ("AB * CD", "AB*CD"),
        ("AB*CD*EF", "AB*CD*EF"),
        ("a*b*c", "a*b*c"),
    ],
)
@pytest.mark.parametrize(("opening", "closing"), [(r"\(", r"\)"), (r"\[", r"\]")])
def test_explicit_delimiters_accept_notation_without_dollar_heuristics(
    body: str, expected: str, opening: str, closing: str
) -> None:
    blocks = _parse_tokens(_create_markdown_parser().parse(f"Formula: {opening} {body} {closing}."))
    assert blocks[0].content.plain == "Formula: " + expected + "."


@pytest.mark.parametrize(
    ("body", "expected"),
    [("AB", "AB"), ("|x|", "|x|"), ("n!", "n!"), ("3x", "3x"), ("a,b", "a, b"), ("f'(x)", "f'(x)")],
)
def test_same_line_display_brackets_accept_explicit_notation(body: str, expected: str) -> None:
    blocks = _parse_tokens(_create_markdown_parser().parse(r"\[ " + body + r" \]"))
    assert len(blocks) == 1
    assert blocks[0].block_type == "math"
    assert blocks[0].math is not None
    assert blocks[0].math.rows == (expected,)


@pytest.mark.parametrize("gap", ["\n", "\n\n"])
def test_unclosed_display_does_not_consume_following_code_fence(gap: str) -> None:
    source = "$$ is the PID" + gap + "```bash\necho $$\n```\n\nend"
    blocks = _parse_tokens(_create_markdown_parser().parse(source))
    assert [block.block_type for block in blocks] == ["paragraph", "fence", "paragraph"]
    assert blocks[-1].content.plain == "end"
    assert blocks[1].content.plain == "echo $$"


def test_blank_line_ends_display_candidate() -> None:
    blocks = _parse_tokens(_create_markdown_parser().parse("$$x\n\ny$$"))
    assert all(block.math is None for block in blocks)


@pytest.mark.parametrize("source", [r"$x^2$", r"\(\alpha_1\)", r"\[x^2\]", "$$x^2$$", "```math\nx^2\n```"])
def test_user_messages_do_not_enable_math(source: str) -> None:
    widget = VirtualizedMarkdown(parser_factory=create_user_text_markdown_parser)
    blocks = widget._build_blocks(source)
    assert all(block.math is None for block in blocks)
    if not source.startswith("```"):
        assert blocks[0].content.plain == source


@pytest.mark.parametrize("source", ["Prices $5 **per user** and $10 per team", "echo $HOME with **care** and $USER"])
def test_user_dollars_do_not_swallow_markdown_between_prices_or_variables(source: str) -> None:
    widget = VirtualizedMarkdown(parser_factory=create_user_text_markdown_parser)
    blocks = widget._build_blocks(source)
    assert blocks[0].content.plain == source.replace("**", "")
    assert blocks[0].content.spans


@pytest.mark.parametrize("prefix", ["> ", "- "])
def test_display_math_respects_container(prefix: str) -> None:
    source = prefix + r"\[\frac{a}{b}\]"
    blocks = _parse_tokens(_create_markdown_parser().parse(source))
    formula = next(block for block in blocks if block.block_type == "math")
    assert formula.math is not None and formula.math.rows
    assert formula.bq_depth == (1 if prefix == "> " else 0)


def test_repeated_unclosed_display_openers_scan_bounded_line_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "\\[\nx\n" * 1000
    scanned = 0
    original = StateBlock.getLines

    def measured(self: StateBlock, begin: int, end: int, indent: int, keep_last_lf: bool) -> str:
        nonlocal scanned
        value = original(self, begin, end, indent, keep_last_lf)
        scanned += len(value)
        return value

    monkeypatch.setattr(StateBlock, "getLines", measured)
    tokens = _create_markdown_parser().parse(source)
    assert not any(token.type == "math_block" for token in tokens)
    # Rebuilding every growing candidate used to scan millions of characters
    # for this small streamed paragraph. Count work instead of wall-clock time.
    assert scanned < 20 * len(source)


@pytest.mark.parametrize(("opening", "closing"), [(r"\[", r"\]"), ("$$", "$$"), (r"\(", r"\)"), ("$", "$")])
@pytest.mark.parametrize("prefix", ["", "Before: "])
def test_oversized_closed_formulas_retain_complete_source(opening: str, closing: str, prefix: str) -> None:
    from chrys.app.tui.widgets.markdown.math.parser import MAX_SOURCE

    source = prefix + opening + r"\{" + "x+" * (MAX_SOURCE // 2) + r"y\}" + closing
    widget = VirtualizedMarkdown()
    blocks = widget._build_blocks(source)
    assert len(blocks) == 1
    assert blocks[0].content.plain == source
    assert blocks[0].math is None or not blocks[0].math.rows
    assert not widget._math_compile_cache


@pytest.mark.parametrize(("opening", "closing"), [(r"\[", r"\]"), ("$$", "$$")])
def test_oversized_multiline_display_preserves_source_and_following_markdown(opening: str, closing: str) -> None:
    body = "\n".join([r"\{", *["x+" * 20] * 300, r"y\}"])
    source = opening + "\n" + body + "\n" + closing
    blocks = VirtualizedMarkdown()._build_blocks(source + "\n\n**After**")
    assert [block.block_type for block in blocks] == ["math", "paragraph"]
    assert blocks[0].content.plain == source
    assert blocks[0].math is not None and not blocks[0].math.rows
    assert blocks[1].content.plain == "After"
    assert blocks[1].content.spans


@pytest.mark.parametrize("opening", [r"\[", r"\(", "$$", "$"])
def test_oversized_unclosed_formula_keeps_following_inline_markdown(opening: str) -> None:
    from markdown_it import MarkdownIt

    from chrys.app.tui.widgets.markdown.math.parser import MAX_SOURCE

    source = opening + "x+" * MAX_SOURCE + " **bold** and `code`"
    expected = _parse_tokens(MarkdownIt("gfm-like", {"html": False}).parse(source))
    actual = _parse_tokens(_create_markdown_parser().parse(source))
    assert [(block.block_type, block.content) for block in actual] == [
        (block.block_type, block.content) for block in expected
    ]


@pytest.mark.parametrize(("opening", "closing"), [(r"\[", r"\]"), ("$$", "$$")])
@pytest.mark.parametrize("boundary", ["\n\n", "\n```text\n"])
def test_oversized_display_does_not_cross_blank_lines_or_fences(opening: str, closing: str, boundary: str) -> None:
    from chrys.app.tui.widgets.markdown.math.parser import MAX_SOURCE

    source = opening + "\n" + "x+" * MAX_SOURCE + boundary + closing
    if "```" in boundary:
        source += "\n```"
    blocks = VirtualizedMarkdown()._build_blocks(source)
    assert all(block.math is None for block in blocks)
    if "```" in boundary:
        assert blocks[-1].block_type == "fence"
        assert blocks[-1].content.plain == closing


@pytest.mark.parametrize("opening", [r"\[", r"\("])
def test_unclosed_marker_index_work_is_linear_and_parse_local(opening: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.app.tui.widgets.markdown.math import markdown as math_markdown

    source = (opening + "x\n") * 2000 + "**After**"
    scanned = 0
    indexed: list[str] = []
    original = math_markdown._index_delimiters

    def measured(text: str) -> dict[str, list[int]]:
        nonlocal scanned
        scanned += len(text)
        indexed.append(text)
        return original(text)

    monkeypatch.setattr(math_markdown, "_index_delimiters", measured)
    env = {"caller_value": 42}
    blocks = _parse_tokens(_create_markdown_parser().parse(source, env))
    assert all(block.math is None for block in blocks)
    assert blocks[-1].content.plain.endswith("After")
    assert blocks[-1].content.spans
    assert len(indexed) == len(set(indexed))
    assert scanned <= 3 * len(source)
    assert env == {"caller_value": 42}


class MathApp(App):
    CSS = "VirtualizedMarkdown { padding: 0; height: 1fr; }"

    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source

    def compose(self) -> ComposeResult:
        yield VirtualizedMarkdown(self.source)


def rendered_rows(widget: VirtualizedMarkdown) -> list[str]:
    return [
        "".join(segment.text for segment in widget.render_line(y)._segments).rstrip()
        for y in range(widget.virtual_size.height)
    ]


@pytest.mark.parametrize(("opening", "closing"), [("```math\n", "\n```"), ("$$\n", "\n$$"), ("\\[\n", "\n\\]")])
async def test_streaming_waits_for_closure_and_reuses_compilation(opening: str, closing: str) -> None:
    source = opening + r"\frac{a}{b}"
    app = MathApp(source)
    async with app.run_test(size=(50, 20)) as pilot:
        await pilot.pause()
        widget = app.query_one(VirtualizedMarkdown)
        assert not any(block.math is not None for block in widget._blocks)
        await widget.append(closing)
        result = widget._blocks[0].math
        assert result is not None and result.rows
        await widget.append("\n\nNext paragraph")
        assert widget._blocks[0].math is result
        assert any("---" in row for row in rendered_rows(widget))


@pytest.mark.parametrize(
    "source",
    [r"\[\frac{abcdefghijklm}{nopqrstuvwxyz}=123456789\]", r"\[\boxed{x:=\frac{abcdefghijklm}{nopqrstuvwxyz}}\]"],
)
async def test_narrow_resize_preserves_entire_equation_and_recovers_layout(source: str) -> None:
    app = MathApp(source)
    async with app.run_test(size=(70, 30)) as pilot:
        await pilot.pause()
        widget = app.query_one(VirtualizedMarkdown)
        assert any("---" in row for row in rendered_rows(widget))
        await resize_when_settled(pilot, 14, 30)
        rows = rendered_rows(widget)
        assert "".join(rows).replace(" ", "") == source
        selected = widget.get_selection(SELECT_ALL)
        assert selected is not None and selected[0].strip() == source
        await resize_when_settled(pilot, 70, 30)
        assert any("---" in row for row in rendered_rows(widget))


async def test_source_fallback_copy_preserves_hard_newlines_but_joins_soft_wraps() -> None:
    source = "\\[\n\\unknown{with spaces}\n+\\frac{abcdef}{ghijkl}\n\\]"
    app = MathApp(source)
    async with app.run_test(size=(14, 30)) as pilot:
        await pilot.pause()
        widget = app.query_one(VirtualizedMarkdown)
        selected = widget.get_selection(SELECT_ALL)
        assert selected is not None
        # Source line boundaries remain; visual left padding is not formula text.
        assert "\n".join(line.removeprefix("  ") for line in selected[0].splitlines()) == source


async def test_formula_rows_have_inset_padding() -> None:
    app = MathApp("$$x^2+y^2=z^2$$")
    async with app.run_test(size=(40, 10)) as pilot:
        await pilot.pause()
        assert rendered_rows(app.query_one(VirtualizedMarkdown)) == ["  x² + y² = z²"]


async def test_visual_copy_and_original_source_have_explicit_separate_contracts() -> None:
    source = r"Inline: \(\alpha^2\)." + "\n\n" + r"\[\frac{a}{b}\]"
    app = MathApp(source)
    async with app.run_test(size=(50, 20)) as pilot:
        await pilot.pause()
        widget = app.query_one(VirtualizedMarkdown)
        selected = widget.get_selection(SELECT_ALL)
        assert selected is not None
        assert "α²" in selected[0]
        assert "---" in selected[0]
        assert widget.source == source


async def test_unknown_block_notation_is_visible_and_never_partially_rendered() -> None:
    source = r"\[x+\unsupported{secret}+y\]"
    app = MathApp(source)
    async with app.run_test(size=(60, 10)) as pilot:
        await pilot.pause()
        assert source in [row.strip() for row in rendered_rows(app.query_one(VirtualizedMarkdown))]
