# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared YAML dump helper — readable output for human-edited profile files.

``yaml.dump``'s default behaviour wraps long / multi-line strings into
double-quoted scalars with ``\\n`` escapes.  Agent and model profile YAMLs
are meant to be opened and hand-edited by users, so we override string
serialisation to use the literal block style (``|``) whenever a string
contains a newline.  The result is natural, line-broken text with no
escape noise.

Scoped to a custom ``Dumper`` subclass so the representer does not leak
into other ``yaml.dump`` call sites in the process.
"""

from __future__ import annotations

from typing import Any

import yaml


class _BlockStyleDumper(yaml.SafeDumper):
    """SafeDumper variant that emits multi-line strings in block style."""


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Use ``|`` for multi-line strings; leave single-line strings to the default."""
    if "\n" in data:
        # ``|`` preserves line breaks verbatim and is trivial for users to
        # edit.  Strip trailing whitespace per line to avoid YAML's
        # "chomping"-style surprises where trailing spaces change meaning.
        cleaned = "\n".join(line.rstrip() for line in data.splitlines())
        # Preserve a final trailing newline if the original had one so
        # round-tripping stays faithful.
        if data.endswith("\n"):
            cleaned += "\n"
        return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockStyleDumper.add_representer(str, _str_representer)


def dump_yaml(data: Any) -> str:
    """Serialise *data* to YAML using human-friendly block-style strings.

    Args:
        data: Any ``yaml.dump``-compatible mapping / sequence / scalar.

    Returns:
        YAML text with ``default_flow_style=False``, keys in insertion
        order, unicode preserved, and multi-line strings rendered with
        ``|`` block scalars.
    """
    return yaml.dump(
        data,
        Dumper=_BlockStyleDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
