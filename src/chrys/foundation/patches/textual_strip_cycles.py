# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: a Textual strip never caches itself.

Problem
-------
``Strip.crop_extend`` and ``Strip.divide`` cache each result on the strip. When the result is
the strip itself (a line that already has the width it is cropped or divided to, the
compositor's common case on every painted line), the strip's own cache holds it: a reference
cycle. Every such line then outlives its last reference until a cyclic collection, so painting
fills the young generation with strips and triggers collections, and pauses, in proportion to
the lines painted.

Solution
--------
Return the strip itself without caching it: that answer costs nothing to recompute. Every other
result is still cached.
"""

from __future__ import annotations

import logging

from chrys.foundation.patches.patcher import FilePatch, register
from chrys.foundation.patches.staged_members import install_patched_members

_RUNTIME_PATCH_MARKER = "_chrys_strip_skips_self_cache"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
logger = logging.getLogger(__name__)

_CROP_EXTEND_OLD = """\
        strip = self.extend_cell_length(end, style).crop(start, end)
        self._crop_extend_cache[cache_key] = strip
        return strip"""

_CROP_EXTEND_NEW = """\
        strip = self.extend_cell_length(end, style).crop(start, end)
        if strip is not self:
            # A cache holding the strip itself would be a reference cycle.
            self._crop_extend_cache[cache_key] = strip
        return strip"""

_DIVIDE_OLD = """\
        cuts = [cut for cut in cuts if cut <= cell_length]
        cache_key = tuple(cuts)
        if (cached := self._divide_cache.get(cache_key)) is not None:
            return cached

        strips: list[Strip]
        if cuts == [cell_length]:
            strips = [self]
        else:
            strips = []
            add_strip = strips.append
            for segments, cut in zip(Segment.divide(self._segments, cuts), cuts):
                add_strip(Strip(segments, cut - pos))
                pos = cut

        self._divide_cache[cache_key] = strips"""

_DIVIDE_NEW = """\
        cuts = [cut for cut in cuts if cut <= cell_length]
        if cuts == [cell_length]:
            # Not cached: a cache holding the strip itself would be a reference cycle.
            return [self]
        cache_key = tuple(cuts)
        if (cached := self._divide_cache.get(cache_key)) is not None:
            return cached

        strips: list[Strip] = []
        add_strip = strips.append
        for segments, cut in zip(Segment.divide(self._segments, cuts), cuts):
            add_strip(Strip(segments, cut - pos))
            pos = cut

        self._divide_cache[cache_key] = strips"""

_PATCHES = (
    FilePatch(
        package="textual",
        module_file="strip.py",
        old_fragment=_CROP_EXTEND_OLD,
        new_fragment=_CROP_EXTEND_NEW,
        description="Strip.crop_extend does not cache the strip itself",
    ),
    FilePatch(
        package="textual",
        module_file="strip.py",
        old_fragment=_DIVIDE_OLD,
        new_fragment=_DIVIDE_NEW,
        description="Strip.divide does not cache the strip itself",
    ),
)

register(*_PATCHES)


def apply_runtime_patch() -> None:
    """Install the patched ``Strip`` methods in the current process."""
    try:
        import textual
        import textual.strip as strip_mod
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual strip runtime patch: loaded Textual is not the pinned %s that the patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return
    install_patched_members(
        strip_mod,
        _PATCHES,
        {"Strip": ["crop_extend", "divide"]},
        marker=_RUNTIME_PATCH_MARKER,
        label="strip",
    )
