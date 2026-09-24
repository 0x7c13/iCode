# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Publish mdurl's lookup tables only once they are complete.

mdurl 0.1.2 builds each percent-encoding table lazily and stores the empty list
in its module-level cache before filling it. Markdown blocks are built in
executor threads, so a second thread normalizing a link at the same moment can
take the half-filled list and index past its end (``IndexError`` from
``encode``) or decode with a table that is missing entries. Build each table
first and publish it with one ``setdefault``, so every thread either builds its
own copy or reads a finished one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

_RUNTIME_PATCH_MDURL_VERSION = "0.1.2"
_RUNTIME_PATCH_MARKER = "_chrys_publishes_complete_tables"
logger = logging.getLogger(__name__)


def apply_runtime_patch() -> None:
    """Replace the two table builders of the pinned mdurl."""
    try:
        import mdurl
        from mdurl import _decode, _encode
    except ImportError:
        return
    if mdurl.__version__ != _RUNTIME_PATCH_MDURL_VERSION:
        logger.warning("Skipping mdurl cache patch for unsupported mdurl %s", mdurl.__version__)
        return
    if getattr(_encode.get_encode_cache, _RUNTIME_PATCH_MARKER, False):
        return

    # The tables mdurl 0.1.2 builds; encode() and decode() look the builders up at call time.
    def get_encode_cache(exclude: str) -> Sequence[str]:
        cache = _encode.encode_cache.get(exclude)
        if cache is None:
            table = [chr(i) if chr(i) in _encode.ASCII_LETTERS_AND_DIGITS else f"%{i:02X}" for i in range(128)]
            for ch in exclude:
                table[ord(ch)] = ch
            cache = _encode.encode_cache.setdefault(exclude, table)
        return cache

    def get_decode_cache(exclude: str) -> Sequence[str]:
        cache = _decode.decode_cache.get(exclude)
        if cache is None:
            table = [chr(i) for i in range(128)]
            for ch in exclude:
                table[ord(ch)] = f"%{ord(ch):02X}"
            cache = _decode.decode_cache.setdefault(exclude, table)
        return cache

    setattr(get_encode_cache, _RUNTIME_PATCH_MARKER, True)
    setattr(get_decode_cache, _RUNTIME_PATCH_MARKER, True)
    _encode.get_encode_cache = get_encode_cache
    _decode.get_decode_cache = get_decode_cache
