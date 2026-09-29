# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reconstruct text within caller-selected segments without inventing whitespace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import groupby
from operator import itemgetter

from chrys.kernel import OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY


def text_block_id(additional_properties: object) -> str | None:
    """Read a stable Responses item identity from live or persisted properties."""
    if not isinstance(additional_properties, Mapping):
        return None
    envelope = additional_properties.get(OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY)
    if not isinstance(envelope, Mapping):
        return None
    block_id = envelope.get("id")
    return block_id if isinstance(block_id, str) and block_id else None


def reconstruct_text_blocks(parts: Iterable[tuple[str, str | None]]) -> list[str]:
    """Join consecutive fragments of one block, retaining distinct block boundaries.

    Callers select a segment before extracting (text, block_id) pairs, then
    choose the separator between the returned blocks. Missing identities form
    one consecutive block; this preserves the existing live convention for
    providers without Responses item IDs. Only identity matters, not mutable
    status or phase. An A/B/A sequence remains three blocks.

    Whitespace and empty blocks are retained. In particular, an empty identified
    block still separates its neighbors. Use ``join_text_blocks`` after
    reconstruction to omit empty blocks without losing those boundaries.
    """
    return ["".join(text for text, _ in fragments) for _, fragments in groupby(parts, key=itemgetter(1))]


def join_text_blocks(blocks: Iterable[str]) -> str:
    """Separate reconstructed blocks, omitting only empty strings, not whitespace.

    Reconstruct each segment before calling this function so empty fragments
    still mark item boundaries. Presentation-only trimming belongs to callers.
    """
    return "\n".join(block for block in blocks if block)
