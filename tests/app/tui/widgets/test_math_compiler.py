# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Notation correctness, resource bounds and generated terminal formula corpus."""

from __future__ import annotations

import random
import unicodedata

import pytest
from rich.cells import cell_len

from chrys.app.tui.widgets.markdown.math import compile_math
from chrys.app.tui.widgets.markdown.math.layout import MAX_AREA, MAX_HEIGHT, MAX_WIDTH

# Model-authored examples cover engineering, probability and linear algebra;
# these are literal expected notation, independent of the layout algorithm.
FORMULAS = (
    r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
    r"\int_0^\infty e^{-x^2}\,dx=\frac{\sqrt{\pi}}{2}",
    r"\sum_{k=0}^{n}\binom{n}{k}p^k(1-p)^{n-k}=1",
    r"\frac{\partial u}{\partial t}=\alpha\nabla^2u",
    r"H(s)=\frac{\omega_n^2}{s^2+2\zeta\omega_ns+\omega_n^2}",
    r"P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}",
    r"\lim_{x\to0}\frac{\sin x}{x}=1",
    r"\hat{\theta}=\underset{\theta}{\operatorname{argmin}}\sum_{i=1}^{n}(y_i-f_\theta(x_i))^2",
    r"A=\begin{bmatrix}1&\frac{1}{2}\\\sqrt{3}&4\end{bmatrix}",
    r"f(x)=\begin{cases}x^2&x\ge0\\-x&x<0\end{cases}",
    r"\begin{aligned}a+b&=c\\a&=c-b\end{aligned}",
    r"\left\langle\frac{x}{y},z\right\rangle=\alpha",
    r"\sqrt[3]{\frac{a+b}{c+d}}",
    r"\left(\frac{1}{x}\right)^2+\left[\frac{1}{y}\right]^2",
    r"\vec{F}=m\vec{a},\quad E=mc^2",
    r"\mathbb{R}^n\to\mathbb{R},\quad\mathbf{x}\mapsto\|x\|",
    r"\begin{aligned}f(x)&=a+b\\&=c\end{aligned}",
    r"\begin{pmatrix}a&&0\\&\ddots&\\0&&a\end{pmatrix}",
    r"\boxed{\theta=90^\circ-\arctan\frac{y}{x}}",
    r"P\Bigl(A\Bigm|B\Bigr):=\frac{P(A\cap B)}{P(B)}",
    r"\boxed{\begin{aligned}T(n)&:=n!\log n\\f&:=g\circ h\end{aligned}}",
    r"\{x\mid x\not\in S\}\Longrightarrow x\not=0",
    r"\angle ABC\cong\angle DEF",
    r"\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)",
    r"\int_0^1x^2\,dx=\left.\frac{x^3}{3}\right|_0^1=\frac13",
    r"\Pr(X=k)=\binom{n}{k}(\frac12)^k(\frac12)^{n-k}",
    r"J(\theta)\stackrel{\text{def}}{=}\frac1n\sum_{i=1}^n\lvert y_i-f_\theta(x_i)\rvert^2",
    r"\theta^*=\underset{\theta}{\arg\min}\,L(\theta)",
    r"\nabla L(\theta)\overset{!}{=}0",
    r"\lVert-\mathbf{v}\rVert=\lVert\mathbf{v}\rVert,\quad\lvert\frac12\rvert^n=2^{-n}",
    r"\underbrace{x+y}_{z}",  # unsupported notation must remain whole
)


@pytest.mark.parametrize("source", FORMULAS[:-1])
def test_engineering_formula_corpus_compiles_without_losing_source(source: str) -> None:
    result = compile_math(source)
    assert result.source == source
    assert result.rows, source
    assert all(cell_len(row) == result.width for row in result.rows)
    assert result.width <= MAX_WIDTH
    assert len(result.rows) <= MAX_HEIGHT
    assert result.width * len(result.rows) <= MAX_AREA


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\frac{a}{b}", (" a ", "---", " b ")),
        (r"x^2", ("x²",)),
        (r"x_i", ("xᵢ",)),
        (r"\frac{1}{\frac{2}{3}}", ("  1  ", "-----", "  2  ", " --- ", "  3  ")),
        (r"\begin{matrix}a&b\\c&d\end{matrix}", ("a  b", "c  d")),
        (r"\sum_{i=1}^{n}x_i", (" n    ", " ∑  xᵢ", "i=1   ")),
    ],
)
def test_exact_spatial_notation(source: str, expected: tuple[str, ...]) -> None:
    assert compile_math(source).rows == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\alpha^2+\beta_1", "α² + β₁"),
        (r"x^{a+b}", "x^{a+b}"),
        (r"\frac{a+b}{c+d}", "(a + b)/(c + d)"),
        (r"\sqrt[3]{x}", "root[3](x)"),
        (r"\sin x", "sin x"),
        (r"\sin^2 x", "sin² x"),
        (r"\operatorname{rank} A", "rank A"),
        (r"\max a_i", "max aᵢ"),
        (r"\sup A", "sup A"),
        (r"\lim f(x)", "lim f(x)"),
        (r"\limsup a_n", "limsup aₙ"),
        (r"\Re z+\Im z", "Re z + Im z"),
        (r"\max\nolimits_i a_i", "maxᵢ aᵢ"),
        (r"\mathbb{E}[X]", "𝔼[X]"),  # noqa: RUF001 - mathematical alphabet
        (r"\mathbb{x}", "𝕩"),  # noqa: RUF001 - mathematical alphabet
        (r"x\equiv1\pmod{2}", "x ≡ 1(mod 2)"),
        (r"a\bmod b", "a mod b"),
        (r"\textbf{v}", "v"),
        (r"\textit{v}", "v"),
        (r"\text{\{value\}}", "{value}"),
        (r"\left(x\rightarrow y\right)", "(x → y)"),
        (r"\text{total cost}=\$5", "total cost = $5"),
        (r"\mathbf{x}+\mathbb{R}", "𝐱 + ℝ"),  # noqa: RUF001 - mathematical blackboard R
    ],
)
def test_linear_notation_preserves_grouping(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"O(n\log n)", "O(n log n)"),
        (r"2\sin x", "2 sin x"),
        (r"a\cos\theta", "a cos θ"),
        (r"\sin(x)", "sin(x)"),
        (r"\operatorname{tr}(A)", "tr(A)"),
        (r"\sin^2(x)", "sin²(x)"),
        (r"\sin[x]", "sin[x]"),
        (r"\sin\{x\}", "sin{x}"),
        (r"\sin\left(x\right)", "sin(x)"),
        (r"\sin x+\cos y", "sin x + cos y"),
        (r"\sin=\cos", "sin = cos"),
        (r"\sin,\cos", "sin, cos"),
        (r"\sin;\cos", "sin; cos"),
        (r"(\sin)", "(sin)"),
        (r"\sin", "sin"),
        (r"\sum", "∑"),
        (r"n{\log n}", "n log n"),
        (r"\sin{}x", "sin x"),
        (r"n\,\log n", "n log n"),
        (r"\sin\quad x", "sin  x"),
        (r"\sin\,(x)", "sin (x)"),
        (r"\sin\!x", "sinx"),
        (r"\sin^2\!x", "sin²x"),
        (r"n\!\log n", "nlog n"),
        (r"x^2\log x", "x² log x"),
        (r"(x)\sin x", "(x) sin x"),
        (r"\sin\cos x", "sin cos x"),
        (r"k\sum\nolimits_i x_i", "k ∑ᵢ xᵢ"),
    ],
)
def test_operator_spacing_is_owned_by_adjacent_atoms(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == (expected,)


def test_spatial_operators_are_separated_from_fraction_bars_and_factors() -> None:
    fraction = compile_math(r"\frac12\log n")
    assert fraction.linear == "(1/2) log n"
    assert fraction.rows[1] == "--- log n"
    summation = compile_math(r"k\sum_i x_i")
    assert summation.linear == "k ∑ᵢ xᵢ"
    assert summation.rows[0] == "k ∑ xᵢ"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"f(x,y)", "f(x, y)"),
        (r"x\in\{1,2,\dots,n\}", "x ∈ {1, 2, …, n}"),
        (r"(a_1,\ldots,a_n)", "(a₁, …, aₙ)"),
        (r"(x;y)", "(x; y)"),
        (r"f(x,\,y)", "f(x, y)"),
        (r"f(x,\quad y)", "f(x,  y)"),
        (r"a,\!b", "a,b"),
    ],
)
def test_body_punctuation_has_one_gap_without_changing_explicit_glue(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == (expected,)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\{x : x > 0\}", "{x : x > 0}"),
        (r"f: A\to B", "f : A → B"),
        (r"f\colon A\to B", "f: A → B"),
        (r"\{x\mid x>0\}", "{x | x > 0}"),
        (r"P(A\mid B)", "P(A | B)"),
        (r"|x|+\vert y\vert", "|x| + |y|"),
        (r"x:=5", "x := 5"),
        (r"x=:5", "x =: 5"),
        (r"x::=5", "x ::= 5"),
        (r"f\circ g", "f ∘ g"),
        (r"n!\log n", "n! log n"),
        (r"n?\log n", "n? log n"),
        (r"n!+1", "n! + 1"),
        (r"A\cong B\Longrightarrow x\not=0", "A ≅ B ⟹ x ≠ 0"),
        (r"x\not\in S", "x ∉ S"),
        (r"a\odot b", "a ⊙ b"),
        (r"a\dagger b", "a † b"),
        (r"a\Longleftarrow b\Longleftrightarrow c", "a ⟸ b ⟺ c"),
        (r"45^\circ+45^{\circ}=90^\circ", "45° + 45° = 90°"),
    ],
)
def test_relation_runs_angles_and_common_symbols(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == (expected,)


@pytest.mark.parametrize("body", ["i:=0", r"i\mid j", r"f\circ g", r"a\odot b", r"i\not=0"])
def test_new_relations_and_binary_symbols_stay_compact_in_scripts(body: str) -> None:
    result = compile_math("x^{" + body + "}")
    assert result.rows
    assert " " not in result.linear


def test_relation_runs_keep_aligned_column_positions() -> None:
    result = compile_math(r"\begin{aligned}x&:=5\\y&=6\end{aligned}")
    assert result.rows == ("x := 5", "y = 6 ")


@pytest.mark.parametrize("size", ["big", "Big", "bigg", "Bigg"])
def test_delimiter_size_hints_preserve_brackets_and_atom_classes(size: str) -> None:
    result = compile_math(rf"P\{size}l(A\{size}m|B\{size}r)")
    assert result.linear == "P(A | B)"
    assert result.rows == ("P(A | B)",)
    assert compile_math(rf"\{size}[x\{size}]").rows == ("[x]",)
    assert compile_math(rf"\{size}l\vert\sin x\{size}r\vert\log n").rows == ("|sin x| log n",)


def test_boxed_answers_keep_borders_only_in_spatial_layout() -> None:
    result = compile_math(r"\boxed{x=\frac12}")
    assert result.linear == "x = 1/2"
    assert result.rows == ("+---------+", "|      1  |", "| x = --- |", "|      2  |", "+---------+")
    adjacent = compile_math(r"y=\boxed{x}\log n")
    assert adjacent.linear == "y = x log n"
    assert adjacent.rows[1] == "y = | x | log n"


def test_boxed_scripts_are_outside_the_border() -> None:
    result = compile_math(r"\boxed{x}^2")
    assert result.linear == "x²"
    assert result.rows == ("     2", "+---+ ", "| x | ", "+---+ ")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\left.x^2\right|_0^1", "x²|₀¹"),
        (r"\left.f\right|_{x=0}", "f|ₓ₌₀"),
        (r"\left.\frac{x^3}{3}\right|_0^1", "x³/3|₀¹"),
        (r"\frac1{\left.a+b\right|}", "1/(a + b|)"),
        (r"\frac1{\left.a+b\right|_0^1}", "1/(a + b|₀¹)"),
        (r"(\frac12)^n", "(1/2)ⁿ"),
        (r"(\frac{a}{b})^2", "(a/b)²"),
        (r"[\frac12]_i^n", "[1/2]ᵢⁿ"),
        (r"\langle\frac12\rangle^n", "⟨1/2⟩ⁿ"),
        (r"x^2\frac12", "x²(1/2)"),
        (r"\frac12x^2", "(1/2)x²"),
        (r"\boxed{42}", "42"),
        (r"\boxed{42}^2", "42²"),
        (r"\boxed{x+y}^2", "(x + y)²"),
        (r"\boxed{\frac12}x", "(1/2)x"),
        (r"\boxed{x+y}z", "(x + y)z"),
        (r"z\boxed{x+y}", "z(x + y)"),
        (r"-\boxed{x+y}", "-(x + y)"),
        (r"a-\boxed{x+y}", "a - (x + y)"),
        (r"\boxed{x+y}\times z", "(x + y) × z"),  # noqa: RUF001 - multiplication sign
        (r"z\times\boxed{x+y}", "z × (x + y)"),  # noqa: RUF001 - multiplication sign
        (r"\boxed{x+y}/z", "(x + y)/z"),
        (r"z/\boxed{x+y}", "z/(x + y)"),
        (r"\left.\boxed{x+y}\right.z", "(x + y)z"),
        (r"z\left.\boxed{x+y}\right.", "z(x + y)"),
        (r"z\boxed{\left.x+y\right|}", "z(x + y|)"),
        (r"\boxed{\left.x+y\right|}/z", "(x + y|)/z"),
        (r"\frac1{\boxed{x+y}}", "1/(x + y)"),
        (r"\boxed{\sin}x", "sin x"),
        (r"\boxed{\boxed{x+y}}^2", "(x + y)²"),
    ],
)
def test_linear_grouping_preserves_evaluation_bars_and_unboxed_contents(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.rows
    assert result.linear == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\left.x^2\right|_0^1", "x²|₀¹"),
        (r"\left.f\right|_{x=0}", "f|ₓ₌₀"),
        (r"\not=0", "≠ 0"),
        (r"\bigm|x", "| x"),
        (":=5", ":= 5"),
        (r"\left.\not=0\right|^1", "≠ 0|¹"),
        (r"{\not=0}^2", "(≠ 0)²"),
    ],
)
def test_standalone_compact_math_has_no_extra_parentheses_or_leading_glue(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.rows == (expected,)
    assert result.linear == expected


def test_relation_edge_cleanup_preserves_explicit_spacing_and_alignment() -> None:
    assert compile_math(r"\quad\not=0").linear.startswith("  ")
    result = compile_math(r"\begin{aligned}a&=b\\&\not=0\end{aligned}")
    assert result.rows == ("a = b", "  ≠ 0")


@pytest.mark.parametrize("template", ["{{{}}}", r"\mathrm{{{}}}", r"\mathbf{{{}}}", r"\left.{}\right."])
def test_root_relation_spacing_passes_through_transparent_groups(template: str) -> None:
    source = template.format(r"\not=0")
    result = compile_math(source)
    assert not result.linear.startswith(" ")
    assert not result.rows[0].startswith(" ")
    spaced = compile_math(template.format(r"\quad\not=0"))
    assert spaced.linear.startswith("  ")
    assert spaced.rows[0].startswith("  ")


def test_boxed_leading_relation_has_only_border_padding() -> None:
    result = compile_math(r"\boxed{\not=0}")
    assert result.linear == "≠ 0"
    assert result.rows == ("+-----+", "| ≠ 0 |", "+-----+")


def test_colon_command_is_punctuation_in_body_and_compact_in_scripts() -> None:
    assert compile_math(r"f\colon\sin x").rows == ("f: sin x",)
    assert compile_math(r"x_{i\colon j}").linear == "x_{i:j}"
    assert compile_math(r"f\colon\quad A").rows == ("f:  A",)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x=?", "x = ?"),
        ("f(2)=?", "f(2) = ?"),
        ("x=!", "x = !"),
        ("x={?}", "x = ?"),
        (r"x=\mathbf{?}", "x = ?"),
        (r"\left(x=?\right)", "(x = ?)"),
        (r"n!\log n", "n! log n"),
        (r"n?\log n", "n? log n"),
        ("n!-1", "n! - 1"),
        (r"x_{n!}", "x_{n!}"),
        (r"x_{=?}", "x_{=?}"),
        ("(x=)", "(x =)"),
    ],
)
def test_postfix_punctuation_preserves_relation_spacing(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    if "_{" not in expected:
        assert result.rows == (expected,)


@pytest.mark.parametrize("template", ["{{{}}}", "{{{{{}}}}}", r"\mathbf{{{}}}", r"\mathrm{{{}}}", r"\mathsf{{{}}}"])
@pytest.mark.parametrize("operator", ["+", "-"])
def test_grouped_signs_remain_ordinary_atoms(template: str, operator: str) -> None:
    result = compile_math("a" + template.format(operator) + "b")
    assert result.linear == "a" + operator + "b"
    assert result.rows == ("a" + operator + "b",)


@pytest.mark.parametrize("group", ["{+}", r"\mathbf{+}", r"\mathrm{+}"])
@pytest.mark.parametrize(("tail", "expected"), [(r"\sin x", "a+ sin x"), ("-b", "a+ - b")])
def test_grouped_signs_expose_an_ordinary_boundary_to_their_neighbours(group: str, tail: str, expected: str) -> None:
    result = compile_math("a" + group + tail)
    assert result.linear == expected
    assert result.rows == (expected,)


@pytest.mark.parametrize(
    ("source", "expected", "rows"),
    [
        (r"a\overset{!}{\mathbf{+}}b", "a +^{!} b", ("  !  ", "a + b")),
        (r"a\underset{n}{\mathrm{-}}b", "a -ₙ b", ("a - b", "  n  ")),
        (r"a{\overset{!}{+}}b", "a+^{!}b", (" ! ", "a+b")),
        (r"a\mathbf{\overset{!}{+}}b", "a+^{!}b", (" ! ", "a+b")),
        (r"a{\overset{!}{+}}-b", "a+^{!} - b", (" !    ", "a+ - b")),
        (r"a\overset{!}{+}-b", "a +^{!} -b", ("  !   ", "a + -b")),
    ],
)
def test_only_outer_annotations_inherit_binary_spacing(source: str, expected: str, rows: tuple[str, ...]) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == rows


@pytest.mark.parametrize("letter", ["d", "e", "f", "m", "k"])
def test_unmapped_superscript_letters_use_the_same_fallback(letter: str) -> None:
    result = compile_math("x^" + letter)
    assert result.linear == "x^{" + letter + "}"
    assert result.rows == (" " + letter, "x ")


def test_dimension_and_parenthesized_letter_superscripts_remain_explicit() -> None:
    assert compile_math(r"f:\mathbb{R}^d\to\mathbb{R}^m").linear == "f : ℝ^{d} → ℝ^{m}"  # noqa: RUF001 - blackboard bold R
    result = compile_math("x^{(d)}")
    assert result.linear == "x^{(d)}"
    assert result.rows == (" (d)", "x   ")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"a\stackrel{\text{def}}{=}b", "a =^{def} b"),
        (r"f(x)\overset{!}{=}0", "f(x) =^{!} 0"),
        (r"x\overset{?}{=}y", "x =^{?} y"),
        (r"x\overset{!}=y", "x =^{!} y"),
        (r"\underset{\theta}{\arg\min}\,L(\theta)", "arg min_{θ} L(θ)"),
        (r"\underset{n}{\lim}x_n", "limₙ xₙ"),
        (r"x\underset{n}{\to}y", "x →ₙ y"),
        (r"\overset{n}{a+b}", "(a + b)ⁿ"),
        (r"\overset{n}{\frac12}", "(1/2)ⁿ"),
        (r"\overset{!}{=}", "=^{!}"),
        (r"(\overset{!}{=})", "(=^{!})"),
        (r"x^{\overset{!}{=}}", "x^{=^{!}}"),
        (r"a\stackrel{\text{def}}{=}\!b", "a =^{def}b"),
        (r"a\quad\overset{!}{=}\quad b", "a  =^{!}  b"),
        (r"a\overset{!}{+}b", "a +^{!} b"),
        (r"a\overset{!}{\times}b", "a ×^{!} b"),  # noqa: RUF001 - multiplication sign
        (r"a\underset{n}{+}b", "a +ₙ b"),
        (r"\overset{!}{-}x", "-^{!}x"),
        (r"x-\overset{!}{-}y", "x - -^{!}y"),
        (r"x^{a\overset{!}{+}b}", "x^{a+^{!}b}"),
    ],
)
def test_linear_annotations_use_scripts_and_keep_base_spacing(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows


@pytest.mark.parametrize(
    ("source", "rows"),
    [
        (r"a\stackrel{\text{def}}{=}b", (" def ", "a = b")),
        (r"x\overset{!}{=}y", ("  !  ", "x = y")),
        (r"x\underset{!}{=}y", ("x = y", "  !  ")),
        (r"x\overset{!}=y", ("  !  ", "x = y")),
        (r"a\overset{!}{+}b", ("  !  ", "a + b")),
        (r"a\underset{n}{\times}b", ("a × b", "  n  ")),  # noqa: RUF001 - multiplication sign
        (r"x-\overset{!}{-}y", ("    ! ", "x - -y")),
        (r"\underset{\theta}{\arg\min}\,L(\theta)", ("arg min L(θ)", "   θ        ")),
    ],
)
def test_spatial_annotations_keep_baselines_and_single_relation_gaps(source: str, rows: tuple[str, ...]) -> None:
    assert compile_math(source).rows == rows


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"|-3|=3", "|-3| = 3"),
        (r"\lvert-x\rvert", "|-x|"),
        (r"\lVert-v\rVert", "‖-v‖"),
        (r"\|-v\|", "‖-v‖"),
        (r"|x|-|y|", "|x| - |y|"),
        (r"\|x\|-\|y\|", "‖x‖ - ‖y‖"),
        (r"x+|-y|", "x + |-y|"),
        (r"x=|-y|", "x = |-y|"),
        (r"f(|-x|,|-y|)", "f(|-x|, |-y|)"),
        (r"P(A\mid B)", "P(A | B)"),
        (r"2|-x|", "2|-x|"),
        (r"|x||-y|", "|x||-y|"),
        (r"2\|-v\|", "2‖-v‖"),
        (r"|x+|-y||", "|x + |-y||"),
    ],
)
def test_absolute_and_norm_bars_resolve_unary_signs(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == (expected,)


@pytest.mark.parametrize(
    ("left", "right", "glyph"),
    [
        (r"\lvert", r"\rvert", "|"),
        (r"\lVert", r"\rVert", "‖"),
        ("|", "|", "|"),
        (r"\|", r"\|", "‖"),
        (r"\bigl|", r"\bigr|", "|"),
    ],
)
def test_bar_boundaries_do_not_add_parentheses_to_scripted_fractions(left: str, right: str, glyph: str) -> None:
    result = compile_math(left + r"\frac12" + right + "^n")
    assert result.linear == glyph + "1/2" + glyph + "ⁿ"
    assert result.rows[1] == glyph + "---" + glyph + "ⁿ"


def test_implicit_bar_products_and_unpaired_evaluation_bar_keep_fraction_grouping() -> None:
    result = compile_math(r"2|\frac12|^2")
    assert result.linear == "2|1/2|²"
    assert result.rows[1] == "2|---|²"
    result = compile_math(r"\frac13|_0^1")
    assert result.linear == "1/3|₀¹"
    assert result.rows[1] == "---|₀¹"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("(=x)", "(= x)"),
        (r"\left(=x\right)", "(= x)"),
        (r"[\le1]", "[≤ 1]"),
        ("(x=)", "(x =)"),
        (r"\left(x=\right)", "(x =)"),
        (r"\lvert=x\rvert", "|= x|"),
        (r"x\boxed{=0}", "x(= 0)"),
        (r"\sqrt{\le x}", "√(≤ x)"),
        (r"\frac{\le x}{\ge y}", "(≤ x)/(≥ y)"),
        (r"\begin{cases}\ge0&x\end{cases}", "cases[≥ 0, x]"),
        (r"\begin{aligned}a&=b+c\\&=d\end{aligned}", "aligned[a, = b + c; , = d]"),
        (r"(\quad=x\quad)", "(  = x  )"),
        (r"(\! =x)", "(= x)"),
    ],
)
def test_subformula_relations_have_no_implicit_outer_padding(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


def test_independent_boxes_root_and_matrix_cells_reset_relation_edges() -> None:
    assert compile_math(r"x\boxed{=0}").rows == (" +-----+", "x| = 0 |", " +-----+")
    assert compile_math(r"\sqrt{\le x}").rows == (" ---", "√≤ x")
    assert compile_math(r"\frac{\le x}{\ge y}").rows == (" ≤ x ", "-----", " ≥ y ")
    assert compile_math(r"\begin{matrix}=x&y=\end{matrix}").rows == ("= x  y =",)
    assert compile_math(r"\begin{aligned}a&=b+c\\&=d\end{aligned}").rows == ("a = b + c", "  = d    ")


@pytest.mark.parametrize(
    "source",
    [
        r"\boxed",
        r"\boxed{\unknown x}",
        r"\big",
        r"\big x",
        r"\Bigl\unknown",
        r"\not",
        r"\not+",
        r"\not\unknown",
        r"\not\subset",
        r"\boxed{" * 60 + "x" + "}" * 60,
    ],
)
def test_new_commands_retain_whole_source_when_incomplete_or_unsupported(source: str) -> None:
    result = compile_math(source)
    assert result.source == source
    assert not result.rows
    assert not result.linear


def test_script_punctuation_remains_compact() -> None:
    result = compile_math(r"a_{i,j}")
    assert result.linear == "a_{i,j}"
    assert result.rows[-1].strip() == "i,j"


@pytest.mark.parametrize("prefix", ["-", "a-", "a+"])
def test_script_sign_never_merges_with_a_fraction_bar(prefix: str) -> None:
    result = compile_math(r"x^{" + prefix + r"\frac12}")
    assert result.rows[1].strip() == prefix + " ---"
    assert result.linear == "x^{" + prefix + "1/2}"


@pytest.mark.parametrize(
    ("source", "expected"),
    [(r"{}^{14}_{6}\mathrm{C}", "₆¹⁴C"), (r"{}^n x", "ⁿx"), (r"{\quad}^n x", "  ⁿx"), (r"{\!}^n x", "ⁿx")],
)
def test_empty_base_scripts_do_not_invent_parentheses(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == (expected,)


def test_empty_base_with_non_unicode_scripts_keeps_spatial_positions() -> None:
    result = compile_math(r"{}_{ab}^{cd}x")
    assert result.linear == "_{ab}^{cd}x"
    assert result.rows == ("cd ", "  x", "ab ")


@pytest.mark.parametrize(
    "command", ["mathrm", "mathit", "mathsf", "mathtt", "text", "textrm", "textsf", "texttt", "textbf", "textit"]
)
def test_font_only_commands_show_their_content(command: str) -> None:
    result = compile_math(rf"\{command}{{d}}x")
    assert result.linear == "dx"
    assert result.rows == ("dx",)


@pytest.mark.parametrize(
    ("source", "expected"),
    [(r"\mathbb{E}", "𝔼"), (r"\mathcal{L}", "ℒ"), (r"\mathbf{v}", "𝐯"), (r"\mathfrak{g}", "𝔤")],  # noqa: RUF001 - mathematical alphabets
)
def test_distinct_math_alphabets_use_unicode(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == (expected,)


def test_aligned_relations_have_consistent_spacing_at_cell_start() -> None:
    rows = compile_math(r"\begin{aligned}a&=b+c\\&=d\end{aligned}").rows
    assert tuple(row.rstrip() for row in rows) == ("a = b + c", "  = d")
    assert rows[0].index("=") == rows[1].index("=")


@pytest.mark.parametrize("environment", ["aligned", "split"])
def test_aligned_column_pairs_have_only_one_inter_pair_gap(environment: str) -> None:
    source = rf"\begin{{{environment}}}a&=b&c&=d\\aa&=e&cc&=f\end{{{environment}}}"
    assert compile_math(source).rows == (" a = b   c = d", "aa = e  cc = f")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"x^{n+1}", "xⁿ⁺¹"),
        (r"2^{n-1}", "2ⁿ⁻¹"),
        (r"a_{i-1}+a_{i+1}", "aᵢ₋₁ + aᵢ₊₁"),
        (r"x_{n+1}=x_n-1", "xₙ₊₁ = xₙ - 1"),
        (r"x^{\mathrm{n+1}}", "xⁿ⁺¹"),
        (r"x^{\left(n+1\right)}", "x⁽ⁿ⁺¹⁾"),
    ],
)
def test_script_operators_are_compact_without_changing_body_spacing(source: str, expected: str) -> None:
    result = compile_math(source)
    assert result.linear == expected
    assert result.rows == (expected,)


def test_script_spacing_propagates_through_nested_fractions() -> None:
    result = compile_math(r"x^{\frac{n+1}{2}}")
    assert result.linear == "x^{(n+1)/2}"
    assert "n+1" in result.rows[0]
    assert "n + 1" not in "\n".join(result.rows)


@pytest.mark.parametrize(("source", "script"), [(r"x^{\text{n + 1}}", "n + 1"), (r"x^{n\quad+1}", "n  +1")])
def test_script_spacing_preserves_explicit_text_and_space(source: str, script: str) -> None:
    result = compile_math(source)
    assert result.linear == "x^{" + script + "}"
    assert result.rows[0].strip() == script


@pytest.mark.parametrize(("command", "symbol"), [("int", "∫"), ("iint", "∬"), ("iiint", "∭"), ("oint", "∮")])
@pytest.mark.parametrize("modifier", ["", r"\nolimits", r"\limits"])
def test_integral_limits_default_to_the_side_and_respect_explicit_limits(
    command: str, symbol: str, modifier: str
) -> None:
    result = compile_math(rf"\{command}{modifier}_0^1 f(x)\,dx")
    assert result.linear == symbol + "₀¹ f(x) dx"
    if modifier == r"\limits":
        assert tuple(row.rstrip() for row in result.rows) == ("1", symbol + " f(x) dx", "0")
    else:
        assert result.rows == (symbol + "₀¹ f(x) dx",)


@pytest.mark.parametrize(("command", "symbol"), [("sum", "∑"), ("prod", "∏"), ("lim", "lim")])
def test_large_operator_limits_remain_stacked_by_default(command: str, symbol: str) -> None:
    for modifier in ("", r"\limits"):
        result = compile_math(rf"\{command}{modifier}_{{i=1}}^n x_i")
        assert result.linear == symbol + "ᵢ₌₁ⁿ xᵢ"
        assert len(result.rows) == 3
        assert "i=1" in result.rows[-1]
    assert compile_math(rf"\{command}\nolimits_{{i=1}}^n x_i").rows == (symbol + "ᵢ₌₁ⁿ xᵢ",)


@pytest.mark.parametrize(
    "template",
    [r"x^{{{}}}", r"x_{{{}}}", r"\sqrt[{}]{{x}}", r"\overset{{{}}}{{x}}", r"\underset{{{}}}{{x}}"],
)
def test_script_context_uses_side_limits_unless_explicitly_stacked(template: str) -> None:
    default = compile_math(template.format(r"\sum_{i=1}^{n}a_i"))
    side = compile_math(template.format(r"\sum\nolimits_{i=1}^{n}a_i"))
    above = compile_math(template.format(r"\sum\limits_{i=1}^{n}a_i"))
    assert default.rows == side.rows
    assert len(above.rows) > len(default.rows)
    assert default.linear == side.linear == above.linear


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"f(x)=-\frac12", "f(x) = -1/2"),
        (r"a+-b", "a + -b"),
        (r"\frac{1}{2}", "1/2"),
        (r"\frac{a+b}{c}", "(a + b)/c"),
        (r"x^2+y^2=z^2", "x² + y² = z²"),
    ],
)
def test_readable_operators_and_compact_atomic_notation(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


def test_unary_minus_does_not_merge_with_fraction_bar() -> None:
    rows = compile_math(r"f(x)=-\frac12").rows
    assert "f(x) = - ---" in rows


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\frac{n!}{k!(n-k)!}", "n!/(k!(n - k)!)"),
        (r"\frac1{n!}", "1/n!"),
        (r"\frac{10!}{3!}", "10!/3!"),
        (r"\frac{\alpha!}{k!}", "α!/k!"),  # noqa: RUF001 - Greek alpha
        (r"\frac{\mathbf{n}!}{k!}", "𝐧!/k!"),  # noqa: RUF001 - mathematical bold n
        (r"\frac{{n!}}{{k!}}", "n!/k!"),
        (r"\frac{ab!}{k}", "(ab!)/k"),
        (r"\frac{n!+1}{k!}", "(n! + 1)/k!"),
        (r"\frac{(n-k)!}{k!}", "(n - k)!/k!"),
        (r"\frac{n!!}{k}", "n!!/k"),
        (r"\frac{\sin x!}{k}", "(sin x!)/k"),
        (r"x^{\frac{n!}{k!(n-k)!}}", "x^{n!/(k!(n-k)!)}"),
        (r"\frac{n!}{k!}x", "(n!/k!)x"),
    ],
)
def test_simple_factorial_fraction_operands_omit_only_redundant_parentheses(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


def test_factorial_fraction_keeps_its_spatial_layout() -> None:
    assert compile_math(r"\frac{n!}{k!(n-k)!}").rows == ("     n!     ", "------------", " k!(n - k)! ")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\hat{\beta}_1", "hat(β)₁"),
        (r"\hat\sigma^2", "hat(σ)²"),  # noqa: RUF001 - Greek sigma
        (r"\frac{\sqrt{3}}{2}", "√(3)/2"),
        (r"\frac{1}{\sqrt{2}}", "1/√(2)"),
        (r"\sqrt{x}^n", "√(x)ⁿ"),
        (r"\frac{\hat\beta}{\bar\sigma}", "hat(β)/bar(σ)"),  # noqa: RUF001 - Greek sigma
        (r"\binom{n}{k}^2", "binom(n, k)²"),
        (r"\frac{\binom{n}{k}}{2}", "binom(n, k)/2"),
        (r"\frac{1}{\binom{n}{k}}", "1/binom(n, k)"),
        (r"\frac{\sqrt[3]{x}}{2}", "root[3](x)/2"),
        (r"\frac{y^{(k)}}{k!}", "y^{(k)}/k!"),
        (r"\frac{(a+b)^{(k)}}{c}", "(a + b)^{(k)}/c"),
        (r"\frac{x_{(q)}}{y}", "x_{(q)}/y"),
        (r"\frac{(x^{\text{)a(}}+y)}{z}", "(x^{)a(} + y)/z"),
        (r"\frac1{(x^{\text{)a(}}+y)}", "1/(x^{)a(} + y)"),
        (r"\frac{x_{\text{)q(}}}{y}", "x_{)q(}/y"),
        (r"\frac{(2n-1)!!}{(2n)!!}", "(2n - 1)!!/(2n)!!"),
        (r"\frac{n!!}{2}", "n!!/2"),
        (r"\frac{1}{n!!}", "1/n!!"),
        (r"\frac{10!!}{3!!}", "10!!/3!!"),
        (r"\frac{ab!!}{k}", "(ab!!)/k"),
        (r"\frac{(a)(b)!!}{c}", "((a)(b)!!)/c"),
        (r"\frac{!!}{2}", "!!/2"),
    ],
)
def test_self_contained_inline_operands_keep_only_necessary_parentheses(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


def test_compact_inline_atoms_do_not_change_spatial_notation() -> None:
    assert compile_math(r"\hat{\beta}_1").rows == ("^ ", "β ", " 1")
    assert compile_math(r"\frac{\sqrt{3}}2").rows == ("  - ", " √3 ", "----", " 2  ")
    assert compile_math(r"\frac{(2n-1)!!}{(2n)!!}").rows == (" (2n - 1)!! ", "------------", "   (2n)!!   ")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\frac{n!}{(n-k)!}", "n!/(n - k)!"),
        (r"\frac{(n+1)!}{n!}", "(n + 1)!/n!"),
        (r"\frac{(n-1)!}{2}", "(n - 1)!/2"),
        (r"\frac{(2n)!}{(n!)^2}", "(2n)!/(n!)²"),
        (r"\frac{(2n)!}{n!\,n!}", "(2n)!/(n! n!)"),
        (r"\frac{(a+b)}{c}", "(a + b)/c"),
        (r"\frac{(a+[b])^2!}{c}", "(a + [b])²!/c"),
        (r"\frac{((n-1)!)!}{2}", "((n - 1)!)!/2"),
        (r"\frac{[n+1]!}{n!}", "[n + 1]!/n!"),
        (r"\frac{\{n+1\}!}{n!}", "{n + 1}!/n!"),
        (r"\frac{\bigl(n+1\bigr)!}{n!}", "(n + 1)!/n!"),
        (r"\frac{\left(n+1\right)!}{n!}", "(n + 1)!/n!"),
        (r"\frac{\left(n+1\right)^2!}{n!}", "(n + 1)²!/n!"),
        (r"\frac{\left(a+(b+c)\right)!}{c}", "(a + (b + c))!/c"),
        (r"\frac{|n+1|!}{n!}", "|n + 1|!/n!"),
        (r"\frac{(a+b)_i}{n!}", "(a + b)ᵢ/n!"),
        (r"\frac{{(a+b)}}{c}", "(a + b)/c"),
        (r"\frac{\mathrm{(a+b)!}}{c}", "(a + b)!/c"),
        (r"x^{\frac{(n+1)!}{n!}}", "x^{(n+1)!/n!}"),
        (r"\frac{\sin^2}{x}", "sin²/x"),
        (r"\frac1{\sin^2}", "1/sin²"),
        (r"\frac{\lim_n}{x}", "limₙ/x"),
        (r"\frac{{\arg\min}_x}{y}", "arg minₓ/y"),
        (r"\frac{\operatorname{rank}^2}{x}", "rank²/x"),
        (r"\frac{x^{(n)}}{y}", "x⁽ⁿ⁾/y"),
        (r"\frac{(n!)^{(n)}}{y}", "(n!)⁽ⁿ⁾/y"),
        (r"\frac{(x^{(n)}+y)}{z}", "(x⁽ⁿ⁾ + y)/z"),
        (r"\frac{{x^{(n)}+y}^2}{z}", "(x⁽ⁿ⁾ + y)²/z"),
        (r"\frac{\left(x^{(n)}+y\right)^2}{z}", "(x⁽ⁿ⁾ + y)²/z"),
    ],
)
def test_fraction_operands_already_enclosed_by_delimiters_need_no_extra_pair(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\frac{(a)+(b)}{c}", "((a) + (b))/c"),
        (r"\frac{(a)(b)}{c}", "((a)(b))/c"),
        (r"\frac{(a+b)(c+d)!}{x}", "((a + b)(c + d)!)/x"),
        (r"\frac{|a|+|b|}{c}", "(|a| + |b|)/c"),
        (r"\frac{(a+b]!}{c}", "((a + b]!)/c"),
        (r"\frac{((a+b)}{c}", "(((a + b))/c"),
        (r"\frac{(a)+(b)!}{c}", "((a) + (b)!)/c"),
        (r"\frac{\left.a+b\right|!}{c}", "(a + b|!)/c"),
        (r"\frac{\operatorname{a+b}}{c}", "(a+b)/c"),
        (r"\frac{\text{(a)+(b)}}{c}", "((a)+(b))/c"),
        (r"\frac1{(a{)+(}b)}", "1/((a) + (b))"),
        (r"\frac1{(a{)}+{(}b)}", "1/((a) + (b))"),
        (r"\frac1{(a\mathrm{)+(}b)}", "1/((a) + (b))"),
        (r"\frac1{(a\left.b\right)+\left(c\right.d)}", "1/((ab) + (cd))"),
        (r"\frac1{(a\left.b\right)^2+\left(c\right.d)}", "1/((ab)² + (cd))"),
        (r"\frac1{(a\text{)+(}b)}", "1/((a)+(b))"),
        (r"\frac1{(a\overset{n}{)}+\underset{i}{(}b)}", "1/((a)ⁿ + (ᵢb))"),
        (r"\frac1{\left(a{)+(}b\right)!}", "1/((a) + (b)!)"),
        (r"\frac1{\left(a{)+(}b\right)^2!}", "1/((a) + (b)²!)"),
        (r"\frac{\operatorname{a+b}^2!}{c}", "(a+b²!)/c"),
    ],
)
def test_fraction_operand_delimiters_must_enclose_the_whole_expression(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


def test_compact_scripts_still_keep_large_operator_limits_and_nested_grouping() -> None:
    assert compile_math("x^2+y^2=z^2").rows == ("x² + y² = z²",)
    assert len(compile_math(r"\sum_{i=1}^n x_i").rows) == 3
    assert compile_math(r"{x^2}^3").rows == ("(x²)³",)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\frac{a}{\left.b+c\right.}", "a/(b + c)"),
        (r"\frac{a}{\left(b+c\right)}", "a/(b + c)"),
        (r"\frac{1}{2}mv^2", "(1/2)mv²"),
        (r"{\!\frac12}mv^2", "(1/2)mv²"),
        (r"\frac12\!", "1/2"),
        (r"{\frac12}mv^2", "(1/2)mv²"),
        (r"\mathrm{\frac12}mv^2", "(1/2)mv²"),
        (r"\mathbf{\frac12}mv^2", "(𝟏/𝟐)mv²"),  # noqa: RUF001 - mathematical bold digits
        (r"\left.\frac12\right.mv^2", "(1/2)mv²"),
        (r"a\frac{1}{2}", "a(1/2)"),
        (r"\frac{a}{\frac{b}{c}}", "a/(b/c)"),
        (r"a/\frac{b}{c}", "a/(b/c)"),
        (r"1/{\frac12}", "1/(1/2)"),
        (r"\frac{b}{c}/a", "b/c/a"),
        (r"\left.a+b\right.^2", "(a + b)²"),
        (r"\mathbf{x}^2", "𝐱²"),
        (r"\mathbb{R}^n", "ℝⁿ"),
        (r"\sin -x", "sin -x"),
        (r"\sum_i -x_i", "∑ᵢ -xᵢ"),
        (r"\int -f(x)\,dx", "∫ -f(x) dx"),
    ],
)
def test_compact_notation_preserves_grouping_and_unary_signs(source: str, expected: str) -> None:
    assert compile_math(source).linear == expected


def test_bold_variant_symbols_have_distinct_unicode_glyphs() -> None:
    result = compile_math(r"\mathbf{\epsilon\phi\vartheta\varpi\varrho\varkappa\varsigma\partial\nabla}")
    assert [unicodedata.name(char) for char in result.linear] == [
        "MATHEMATICAL BOLD " + name
        for name in [
            "EPSILON SYMBOL",
            "PHI SYMBOL",
            "THETA SYMBOL",
            "PI SYMBOL",
            "RHO SYMBOL",
            "KAPPA SYMBOL",
            "SMALL FINAL SIGMA",
            "PARTIAL DIFFERENTIAL",
            "NABLA",
        ]
    ]


@pytest.mark.parametrize(
    "source",
    [
        "",
        "{x",
        "x}",
        "x^^2",
        "x_1_2",
        r"\frac{x}",
        r"\sqrt",
        r"\left(x",
        r"\left(x\rightward y",
        r"\unknown{x}",
        r"\input{secret}",
        FORMULAS[-1],
        r"\begin{matrix}a&b\\c\end{matrix}",
        r"\begin{cases}a\end{cases}",
        r"\begin{matrix}a\end{pmatrix}",
        r"\begin{array}{cc}a&b\end{array}",
        "\x1b[31mx",
        "x\x00y",
        "x" * 9000,
        "{" * 60 + "x" + "}" * 60,
        r"\sqrt" * 60 + "x",
        r"\frac{1}{" * 60 + "x" + "}" * 60,
    ],
)
def test_unsupported_or_unbounded_formula_fails_closed(source: str) -> None:
    result = compile_math(source)
    assert result.source == source
    assert not result.rows
    assert not result.linear


def test_seeded_combinations_keep_all_operand_occurrences() -> None:
    rng = random.Random(73019)
    atoms = ("a", "b", "c", "d")

    def expression(depth: int) -> tuple[str, list[str]]:
        if depth == 0:
            atom = rng.choice(atoms)
            return atom, [atom]
        left, left_atoms = expression(depth - 1)
        right, right_atoms = expression(depth - 1)
        operation = rng.randrange(9)
        if operation == 0:
            return rf"\frac{{{left}}}{{{right}}}", left_atoms + right_atoms
        if operation == 1:
            return rf"\sqrt{{{left}+{right}}}", left_atoms + right_atoms
        if operation == 2:
            return rf"{{{left}}}^{{{right}}}", left_atoms + right_atoms
        if operation == 4:
            return rf"\boxed{{{left}:={right}}}", left_atoms + right_atoms
        if operation == 5:
            return rf"\Bigl({left}\Bigm|{right}\Bigr)", left_atoms + right_atoms
        if operation == 6:
            return rf"\left.{left}+{right}\right|_0^1", left_atoms + right_atoms
        if operation == 7:
            return rf"\overset{{{left}}}{{{right}}}", left_atoms + right_atoms
        if operation == 8:
            return rf"\lvert {left}-{right}\rvert", left_atoms + right_atoms
        return rf"\begin{{matrix}}{left}\\{right}\end{{matrix}}", left_atoms + right_atoms

    for _ in range(100):
        source, operands = expression(3)
        result = compile_math(source)
        assert result.rows, source
        # Superscript glyphs still represent the same operand occurrences.
        text = unicodedata.normalize("NFKD", "".join(result.rows))
        for atom in atoms:
            assert text.count(atom) == operands.count(atom), source
        assert all(cell_len(row) == result.width for row in result.rows)


def test_wide_unicode_text_boxes_align_by_terminal_cells() -> None:
    result = compile_math(r"\frac{\text{总量}}{\text{人数}}=\mu")
    assert result.rows
    assert "总量" in result.rows[0]
    assert "人数" in result.rows[-1]
    assert all(cell_len(row) == result.width for row in result.rows)


def test_seeded_malformed_input_never_raises_or_loses_source() -> None:
    rng = random.Random(931)
    alphabet = "abc012{}_^&$%\\[]()+-= \n"
    for _ in range(300):
        source = "".join(rng.choices(alphabet, k=rng.randrange(1, 100)))
        assert compile_math(source).source == source
