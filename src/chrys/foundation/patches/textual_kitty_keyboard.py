# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: keep Textual's Kitty keyboard input compatible with IMEs.

Problem
-------
Some terminals send IME commits through the Kitty keyboard protocol as an
associated-text field with colon-separated Unicode codepoints, for example
``ESC [ 32 ; ; 24403:21069 u`` for ``当前``. Textual 8.2.7 only accepts a
single numeric associated-text codepoint, so it treats the sequence as unknown
and reissues the printable tail into focused inputs. Longer commits also exceed
Textual's 32-character unknown-sequence search cap before the parser sees the
final ``u``.

Solution
--------
Accept colons in Kitty keyboard parameters and decode the associated-text field
as one or more Unicode codepoints. The resulting ``Key.character`` is the
committed text, so widgets receive the same text they would have received from a
plain UTF-8 terminal input path. Raise the search cap enough for normal IME
commit chunks to reach the final sequence byte.

Some Chrys TUI modules import Textual before ``bootstrap_runtime()`` applies
site-package patches, so this module also exposes an in-process monkey patch
for the already-imported ``XTermParser`` class.

iTerm 3.6.x has a separate upstream composition bug when Textual requests both
``REPORT_ALL_KEYS`` and ``REPORT_ASSOCIATED_TEXT``. In that terminal/version
combination an IME commit can be lost while its candidate-confirmation space is
still reported. The runtime patch therefore leaves Kitty disambiguation enabled
but removes those two progressive-enhancement flags for affected iTerm builds.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from chrys.foundation.patches.patcher import FilePatch, register

_OLD_SEARCH_THRESHOLD = "_MAX_SEQUENCE_SEARCH_THRESHOLD = 32"
_NEW_SEARCH_THRESHOLD = "_MAX_SEQUENCE_SEARCH_THRESHOLD = 2048"

_OLD_REGEX = """\
_re_extended_key: Final[re.Pattern[str]] = re.compile(
    r"\\x1b\\[((?:\\d*;?){2,3})([u~ABCDEFHPQRS])"
)"""

_NEW_REGEX = """\
_re_extended_key: Final[re.Pattern[str]] = re.compile(
    r"\\x1b\\[((?:\\d+(?::\\d+)*)?(?:;(?:\\d+(?::\\d+)*)?){0,2})([u~ABCDEFHPQRS])"
)"""

_OLD_PARSE = """\
        codepoint = int(codepoint_str or "1")
        modifiers = int(modifiers_str or "0")
        text = chr(int(text_str)) if text_str else None"""

_NEW_PARSE = """\
        try:
            codepoint = int(codepoint_str.partition(":")[0] or "1")
            modifiers = int(modifiers_str.partition(":")[0] or "0")
            text = "".join(chr(int(part)) for part in text_str.split(":")) if text_str else None
        except ValueError:
            return None"""

_NEW_PARSE_WITH_RELEASE_GUARD = """\
        try:
            codepoint = int(codepoint_str.partition(":")[0] or "1")
            modifiers = int(modifiers_str.partition(":")[0] or "0")
            text = "".join(chr(int(part)) for part in text_str.split(":")) if text_str else None
        except ValueError:
            return None

        event_type = modifiers_str.partition(":")[2].partition(":")[0]
        if event_type == "3":
            return events.Key(Keys.Ignore, sequence)"""

_OLD_KEY = """\
        if not (key := FUNCTIONAL_KEYS.get(f"{codepoint}{end}", "")):
            key = _character_to_key(text if text else chr(codepoint))"""

_EQUIVALENT_KEY = """\
        if not (key := FUNCTIONAL_KEYS.get(f"{codepoint}{end}", "")):
            if text and len(text) == 1:
                key = _character_to_key(text)
            elif text:
                key = text
            else:
                key = _character_to_key(chr(codepoint))"""

_NEW_KEY = """\
        if not (key := FUNCTIONAL_KEYS.get(f"{codepoint}{end}", "")):
            if text and len(text) == 1:
                key = _character_to_key(text)
            elif text:
                # Multi-codepoint IME commit: use text as the key name; character drives input.
                key = text
            else:
                key = _character_to_key(chr(codepoint))"""

_RUNTIME_PATCH_MARKER = "_chrys_kitty_multi_codepoint_text"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
_RUNTIME_EXTENDED_KEY_RE = re.compile(r"\x1b\[((?:\d+(?::\d+)*)?(?:;(?:\d+(?::\d+)*)?){0,2})([u~ABCDEFHPQRS])")
_AFFECTED_ITERM_VERSION_RE = re.compile(r"^3\.6(?:\D|$)")
logger = logging.getLogger(__name__)


def _is_affected_iterm(env: Mapping[str, str]) -> bool:
    """Return whether the terminal has iTerm's Kitty composition bug.

    Remote and multiplexer sessions may omit or replace ``TERM_PROGRAM``. We
    intentionally avoid guessing from ``TERM`` or process ancestry: failing to
    detect iTerm only preserves Textual's normal flags, while a false positive
    would unnecessarily degrade keyboard reporting in another terminal.
    """
    return env.get("TERM_PROGRAM") == "iTerm.app" and bool(
        _AFFECTED_ITERM_VERSION_RE.match(env.get("TERM_PROGRAM_VERSION", ""))
    )


def _apply_iterm_kitty_flag_workaround(env: Mapping[str, str]) -> None:
    """Keep only Kitty key disambiguation on affected iTerm versions."""
    if not _is_affected_iterm(env):
        return
    try:
        import textual.drivers.linux_driver as linux_driver
    except ImportError:
        return

    linux_driver.KITTY_REPORT_ALL_KEYS = 0
    linux_driver.KITTY_REPORT_ASSOCIATED_TEXT = 0


def _decode_codepoint_text(text: str) -> str | None:
    if not text:
        return None
    try:
        return "".join(chr(int(part)) for part in text.split(":"))
    except ValueError:
        return None


def apply_runtime_patch() -> None:
    """Patch ``XTermParser._parse_extended_key`` in the current process."""
    _apply_iterm_kitty_flag_workaround(os.environ)
    try:
        import textual
        import textual._xterm_parser as xterm_parser
        from textual import events
        from textual._keyboard_protocol import FUNCTIONAL_KEYS, MODIFIER_FUNCTIONAL_KEYS
        from textual.keys import _character_to_key
    except ImportError:
        return

    if getattr(textual, "__version__", None) != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual Kitty keyboard runtime patch: loaded Textual is not the pinned %s that the patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return

    XTermParser = xterm_parser.XTermParser
    if getattr(XTermParser._parse_extended_key, _RUNTIME_PATCH_MARKER, False):
        return

    @lru_cache(maxsize=1024)
    def _parse_extended_key(self: Any, sequence: str) -> Any:
        """Parse a Kitty sequence."""
        if (match := _RUNTIME_EXTENDED_KEY_RE.fullmatch(sequence)) is None:
            return None

        codes, end = match.groups(default="")
        codepoint_str, modifiers_str, text_str, *_ = [*codes.split(";"), "", "", ""]
        text = _decode_codepoint_text(text_str)
        if text_str and text is None:
            return None
        try:
            codepoint = int(codepoint_str.partition(":")[0] or "1")
            modifiers = int(modifiers_str.partition(":")[0] or "0")
        except ValueError:
            return None

        event_type = modifiers_str.partition(":")[2].partition(":")[0]
        if event_type == "3":
            from textual.keys import Keys

            return events.Key(Keys.Ignore, sequence)

        if not (key := FUNCTIONAL_KEYS.get(f"{codepoint}{end}", "")):
            if text and len(text) == 1:
                key = _character_to_key(text)
            elif text:
                # Multi-codepoint IME commit: use text as the key name; character drives input.
                key = text
            else:
                key = _character_to_key(chr(codepoint))

        key_tokens: list[str] = []
        if modifiers and key not in MODIFIER_FUNCTIONAL_KEYS and text_str is not None:
            modifier_bits = int(modifiers) - 1
            modifiers_names = ("alt", "ctrl", "super", "hyper", "meta")
            if modifier_bits & 1 and (text is None or text.isspace()):
                key_tokens.append("shift")
            for bit, modifier in enumerate(modifiers_names, 1):
                if modifier == "alt" and text is not None:
                    continue
                if modifier_bits & (1 << bit):
                    key_tokens.append(modifier)

        key_tokens.sort()
        if key is not None:
            key_tokens.append(key)
        return events.Key(
            "+".join(key_tokens),
            text or (None if modifiers else xterm_parser.SPECIAL_KEY_TO_CHARACTER.get(key, None)),
        )

    setattr(_parse_extended_key, _RUNTIME_PATCH_MARKER, True)
    xterm_parser._MAX_SEQUENCE_SEARCH_THRESHOLD = 2048
    xterm_parser._re_extended_key = _RUNTIME_EXTENDED_KEY_RE
    XTermParser._parse_extended_key = _parse_extended_key


register(
    FilePatch(
        package="textual",
        module_file="_xterm_parser.py",
        old_fragment=_OLD_SEARCH_THRESHOLD,
        new_fragment=_NEW_SEARCH_THRESHOLD,
        description="Allow longer Kitty keyboard associated-text sequences",
    ),
    FilePatch(
        package="textual",
        module_file="_xterm_parser.py",
        old_fragment=_OLD_REGEX,
        new_fragment=_NEW_REGEX,
        description="Accept colon-separated Kitty keyboard text parameters",
    ),
    FilePatch(
        package="textual",
        module_file="_xterm_parser.py",
        old_fragment=_OLD_PARSE,
        new_fragment=_NEW_PARSE,
        description="Decode multi-codepoint Kitty keyboard associated text",
    ),
    FilePatch(
        package="textual",
        module_file="_xterm_parser.py",
        old_fragment=_NEW_PARSE,
        new_fragment=_NEW_PARSE_WITH_RELEASE_GUARD,
        description="Ignore Kitty keyboard release events after associated text decode",
    ),
    FilePatch(
        package="textual",
        module_file="_xterm_parser.py",
        old_fragment=_OLD_KEY,
        new_fragment=_NEW_KEY,
        description="Guard multi-character Kitty keyboard text key conversion",
        equivalent_fragments=(_EQUIVALENT_KEY,),
    ),
)
