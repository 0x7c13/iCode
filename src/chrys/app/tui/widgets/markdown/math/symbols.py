# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit terminal spellings for the supported TeX math vocabulary."""

# Mathematical symbols intentionally differ from their Latin lookalikes.
# ruff: noqa: RUF001

from __future__ import annotations

import string
import unicodedata

SYMBOLS: dict[str, str] = dict(
    zip(
        [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "varepsilon",
            "zeta",
            "eta",
            "theta",
            "vartheta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "omicron",
            "pi",
            "varpi",
            "rho",
            "varrho",
            "sigma",
            "varsigma",
            "tau",
            "upsilon",
            "phi",
            "varphi",
            "chi",
            "psi",
            "omega",
        ],
        [
            "α",
            "β",
            "γ",
            "δ",
            "ϵ",
            "ε",
            "ζ",
            "η",
            "θ",
            "ϑ",
            "ι",
            "κ",
            "λ",
            "μ",
            "ν",
            "ξ",
            "ο",
            "π",
            "ϖ",
            "ρ",
            "ϱ",
            "σ",
            "ς",
            "τ",
            "υ",
            "ϕ",
            "φ",
            "χ",
            "ψ",
            "ω",
        ],
        strict=True,
    )
)
SYMBOLS.update(
    zip(
        ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"],
        ["Γ", "Δ", "Θ", "Λ", "Ξ", "Π", "Σ", "Υ", "Φ", "Ψ", "Ω"],
        strict=True,
    )
)
SYMBOLS.update(
    {
        "infty": "∞",
        "partial": "∂",
        "nabla": "∇",
        "ell": "ℓ",
        "hbar": "ℏ",
        "imath": "ı",
        "jmath": "ȷ",
        "Re": "Re",
        "Im": "Im",
        "emptyset": "∅",
        "varnothing": "∅",
        "varkappa": "ϰ",
        "forall": "∀",
        "exists": "∃",
        "neg": "¬",
        "lnot": "¬",
        "sum": "∑",
        "prod": "∏",
        "coprod": "∐",
        "int": "∫",
        "iint": "∬",
        "iiint": "∭",
        "oint": "∮",
        "pm": "±",
        "mp": "∓",
        "times": "×",
        "div": "÷",
        "cdot": "·",
        "ast": "∗",
        "le": "≤",
        "leq": "≤",
        "ge": "≥",
        "geq": "≥",
        "ne": "≠",
        "neq": "≠",
        "approx": "≈",
        "equiv": "≡",
        "sim": "∼",
        "simeq": "≃",
        "cong": "≅",
        "colon": ":",
        "propto": "∝",
        "ll": "≪",
        "gg": "≫",
        "in": "∈",
        "notin": "∉",
        "ni": "∋",
        "subset": "⊂",
        "supset": "⊃",
        "subseteq": "⊆",
        "supseteq": "⊇",
        "cup": "∪",
        "cap": "∩",
        "bigcup": "⋃",
        "bigcap": "⋂",
        "setminus": "∖",
        "land": "∧",
        "wedge": "∧",
        "lor": "∨",
        "vee": "∨",
        "oplus": "⊕",
        "otimes": "⊗",
        "odot": "⊙",
        "dagger": "†",
        "to": "→",
        "rightarrow": "→",
        "leftarrow": "←",
        "leftrightarrow": "↔",
        "Rightarrow": "⇒",
        "Leftarrow": "⇐",
        "Leftrightarrow": "⇔",
        "implies": "⇒",
        "iff": "⇔",
        "mapsto": "↦",
        "longrightarrow": "⟶",
        "Longrightarrow": "⟹",
        "Longleftarrow": "⟸",
        "Longleftrightarrow": "⟺",
        "uparrow": "↑",
        "downarrow": "↓",
        "perp": "⊥",
        "parallel": "∥",
        "mid": "|",
        "vert": "|",
        "Vert": "‖",
        "lvert": "|",
        "rvert": "|",
        "lVert": "‖",
        "rVert": "‖",
        "langle": "⟨",
        "rangle": "⟩",
        "lfloor": "⌊",
        "rfloor": "⌋",
        "lceil": "⌈",
        "rceil": "⌉",
        "lbrace": "{",
        "rbrace": "}",
        "ldots": "…",
        "dots": "…",
        "cdots": "⋯",
        "vdots": "⋮",
        "ddots": "⋱",
        "angle": "∠",
        "degree": "°",
        "circ": "∘",
        "bullet": "•",
        "prime": "′",
        "top": "⊤",
        "bot": "⊥",
        "therefore": "∴",
        "because": "∵",
        "quad": "  ",
        "qquad": "    ",
        "enspace": " ",
        " ": " ",
        ",": " ",
        ";": " ",
        ":": " ",
        "!": "",
        "{": "{",
        "}": "}",
        "%": "%",
        "_": "_",
        "#": "#",
        "&": "&",
        "$": "$",
        "|": "‖",
    }
)

FUNCTIONS = frozenset(
    [
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "log",
        "ln",
        "exp",
        "lim",
        "limsup",
        "liminf",
        "max",
        "min",
        "sup",
        "inf",
        "det",
        "gcd",
        "dim",
        "ker",
        "Pr",
        "arg",
        "deg",
        "hom",
        "rank",
        "tr",
        "mod",
        "bmod",
        "pmod",
    ]
)
LARGE_OPERATORS = frozenset(
    [
        "sum",
        "prod",
        "coprod",
        "bigcup",
        "bigcap",
        "lim",
        "limsup",
        "liminf",
        "max",
        "min",
        "sup",
        "inf",
    ]
)
INTEGRAL_OPERATORS = frozenset({"int", "iint", "iiint", "oint"})
FONTS = frozenset(
    ["mathrm", "mathit", "mathbf", "mathsf", "mathtt", "mathcal", "mathbb", "mathfrak", "boldsymbol", "bm"]
)
TEXT_COMMANDS = frozenset({"text", "textrm", "textsf", "texttt", "textbf", "textit", "operatorname"})


def _alphabet(style: str, exceptions: dict[str, str]) -> dict[str, str]:
    alphabet = exceptions.copy()
    for char in string.ascii_letters:
        if char not in alphabet:
            case = "CAPITAL" if char.isupper() else "SMALL"
            alphabet[char] = unicodedata.lookup(f"MATHEMATICAL {style} {case} {char.upper()}")
    if style in {"BOLD", "DOUBLE-STRUCK"}:
        for char, name in zip(
            string.digits, ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"], strict=True
        ):
            alphabet[char] = unicodedata.lookup(f"MATHEMATICAL {style} DIGIT {name}")
    return alphabet


BLACKBOARD = _alphabet("DOUBLE-STRUCK", dict(zip("CHNPQRZ", "ℂℍℕℙℚℝℤ", strict=True)))
SCRIPT = _alphabet("SCRIPT", dict(zip("BEFHILMRego", "ℬℰℱℋℐℒℳℛℯℊℴ", strict=True)))
FRAKTUR = _alphabet("FRAKTUR", dict(zip("CHIRZ", "ℭℌℑℜℨ", strict=True)))
BOLD = _alphabet("BOLD", {})
for _char in "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω":
    _name = unicodedata.name(_char).replace("GREEK ", "").replace("LETTER ", "")
    BOLD[_char] = unicodedata.lookup("MATHEMATICAL BOLD " + _name)
for _char, _name in {
    "ϵ": "EPSILON SYMBOL",
    "ϕ": "PHI SYMBOL",
    "ϑ": "THETA SYMBOL",
    "ϖ": "PI SYMBOL",
    "ϱ": "RHO SYMBOL",
    "ϰ": "KAPPA SYMBOL",
    "ς": "SMALL FINAL SIGMA",
    "∂": "PARTIAL DIFFERENTIAL",
    "∇": "NABLA",
}.items():
    BOLD[_char] = unicodedata.lookup("MATHEMATICAL BOLD " + _name)
FONT_ALPHABETS = {
    "mathbb": BLACKBOARD,
    "mathcal": SCRIPT,
    "mathfrak": FRAKTUR,
    "mathbf": BOLD,
    "boldsymbol": BOLD,
    "bm": BOLD,
}
SUPERSCRIPT = dict(zip("0123456789+-=()in∘", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ°", strict=True))
SUBSCRIPT = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ", strict=True))
