# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: use ``errors="replace"`` for UTF-8 incremental decoders in Textual drivers.

Problem
-------
The input threads in ``LinuxDriver``, ``LinuxInlineDriver``, and
``WebDriver`` create a strict UTF-8 decoder (the default).  If any
invalid byte sequence arrives from the terminal (e.g. a locale mismatch,
raw binary leaking into stdin, or a broken paste), the decoder raises
``UnicodeDecodeError`` and crashes the input thread, killing all
keyboard/mouse handling.

Solution
--------
Pass ``errors="replace"`` so invalid bytes are replaced with U+FFFD
instead of raising.  The application stays responsive; garbled input is
visibly wrong but non-fatal.

If a launcher has already imported a driver, its input method still has the old
code. Update that module's decoder factory too. This also reaches an input
method captured by a Thread before startup without replacing driver classes.
"""

from __future__ import annotations

import logging
import sys
from codecs import IncrementalDecoder, getincrementaldecoder
from encodings.utf_8 import IncrementalDecoder as Utf8IncrementalDecoder
from typing import Protocol

from chrys.foundation.patches.patcher import FilePatch, register

_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
_DRIVER_MODULES = (
    "textual.drivers.linux_driver",
    "textual.drivers.linux_inline_driver",
    "textual.drivers.web_driver",
)
logger = logging.getLogger(__name__)


class _IncrementalDecoderFactory(Protocol):
    """Codec factory accepting the stdlib's optional error policy."""

    def __call__(self, errors: str = ...) -> IncrementalDecoder: ...


class _TerminalUTF8Decoder(Utf8IncrementalDecoder):
    """Use replacement by default while retaining explicit error policies."""

    def __init__(self, errors: str = "replace") -> None:
        super().__init__(errors)


def _terminal_decoder_factory(encoding: str) -> _IncrementalDecoderFactory:
    decoder = getincrementaldecoder(encoding)
    return _TerminalUTF8Decoder if decoder is Utf8IncrementalDecoder else decoder


def apply_runtime_patch() -> None:
    """Repair preloaded drivers without importing POSIX modules on Windows."""
    try:
        import textual
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning("Skipping UTF-8 decoder patch for unsupported Textual %s", textual.__version__)
        return
    # Future imports use the file patch; preloaded functions resolve this live
    # module global. The stdlib codec registry and other encodings stay intact.
    for module_name in _DRIVER_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            module.getincrementaldecoder = _terminal_decoder_factory


_OLD = 'utf8_decoder = getincrementaldecoder("utf-8")().decode'
_NEW = 'utf8_decoder = getincrementaldecoder("utf-8")(errors="replace").decode'

register(
    FilePatch(
        package="textual",
        module_file="drivers/linux_driver.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use errors='replace' for UTF-8 decoder in LinuxDriver",
    ),
    FilePatch(
        package="textual",
        module_file="drivers/linux_inline_driver.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use errors='replace' for UTF-8 decoder in LinuxInlineDriver",
    ),
    FilePatch(
        package="textual",
        module_file="drivers/web_driver.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use errors='replace' for UTF-8 decoder in WebDriver",
    ),
)
