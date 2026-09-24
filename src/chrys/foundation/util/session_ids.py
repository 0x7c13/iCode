# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session identifier helpers shared across Chrys layers."""

from __future__ import annotations

# Number of hex characters used when projecting a full session UUID
# down to a short identifier for display and on-disk folder names.
# The canonical session id is the full ``uuid4().hex`` (32 chars);
# ``session_id[:SESSION_SHORT_ID_LEN]`` is the user-facing / folder
# form.  Pre-refactor sessions were themselves 12 hex chars — slicing
# a legacy id by ``[:SESSION_SHORT_ID_LEN]`` returns the same string,
# so the same code paths handle both.
SESSION_SHORT_ID_LEN = 12


def session_short_id(session_id: str) -> str:
    """Return the short projection of *session_id* for display / folder.

    The canonical session id is the RFC 4122 UUID string with dashes
    (36 chars, e.g. ``"4201eebc-ca45-4328-8882-272f3d7c41cb"``).  The
    short form strips dashes then truncates to
    :data:`SESSION_SHORT_ID_LEN` hex chars — yielding a clean prefix of
    the dashless form (e.g. ``"4201eebcca45"``).  Pre-refactor sessions
    whose id was already 12 hex chars slice to themselves, so this
    helper handles both shapes uniformly.
    """
    safe_id = session_id.replace("/", "_").replace("\\", "_").replace("-", "")
    return safe_id[:SESSION_SHORT_ID_LEN]
