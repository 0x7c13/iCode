# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runtime environment-variable template resolution.

Configuration files may carry literal placeholders such as ``{{API_KEY}}``.
Those placeholders are intentionally resolved only at runtime so saved YAML
keeps the user-authored value and never receives resolved secrets.
"""

from __future__ import annotations

import os
import re
from typing import Any

_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ESCAPED_OPEN = "\0CHRYS_ENV_TEMPLATE_OPEN\0"
_ESCAPED_CLOSE = "\0CHRYS_ENV_TEMPLATE_CLOSE\0"


class EnvVarResolutionError(ValueError):
    """Raised when an environment-variable placeholder cannot be resolved."""


def _invalid_placeholder_message(raw: str, location: str) -> str:
    return (
        f"Invalid environment variable placeholder {raw!r} in {location}. "
        "Use {{ENV_VAR}} with letters, digits, and underscores only, and no spaces."
    )


def resolve_env_templates(value: str, *, location: str) -> str:
    """Resolve ``{{ENV_VAR}}`` placeholders in a string.

    Resolution is single-pass: if the environment value itself contains
    ``{{...}}``, that text is preserved literally.  Missing or empty
    environment variables are treated as errors because an explicit
    placeholder means the user expects that runtime value to exist.
    Only double-brace blocks containing a valid environment variable
    identifier are resolved; other non-empty blocks, such as
    ``{{"kind":"json"}}``, are preserved literally.
    """
    value = value.replace("{{{{", _ESCAPED_OPEN).replace("}}}}", _ESCAPED_CLOSE)
    if "{{" not in value and "}}" not in value:
        return value.replace(_ESCAPED_OPEN, "{{").replace(_ESCAPED_CLOSE, "}}")

    parts: list[str] = []
    pos = 0
    while True:
        start = value.find("{{", pos)
        stray_end = value.find("}}", pos)
        if start == -1:
            if stray_end != -1:
                raise EnvVarResolutionError(_invalid_placeholder_message("}}", location))
            parts.append(value[pos:])
            break
        if stray_end != -1 and stray_end < start:
            raise EnvVarResolutionError(_invalid_placeholder_message("}}", location))

        end = value.find("}}", start + 2)
        if end == -1:
            raise EnvVarResolutionError(_invalid_placeholder_message(value[start:], location))

        parts.append(value[pos:start])
        name = value[start + 2 : end]
        raw_placeholder = value[start : end + 2]
        if name == "":
            raise EnvVarResolutionError(_invalid_placeholder_message(raw_placeholder, location))
        if _ENV_NAME_RE.fullmatch(name) is None:
            stripped = name.strip()
            if stripped != name and _ENV_NAME_RE.fullmatch(stripped) is not None:
                raise EnvVarResolutionError(_invalid_placeholder_message(raw_placeholder, location))
            parts.append(raw_placeholder)
            pos = end + 2
            continue

        resolved = os.environ.get(name)
        if resolved is None or resolved == "":
            raise EnvVarResolutionError(
                f"Environment variable {name!r} is required by {location}, but it is not set. "
                f"Set {name} in the environment or remove the {{{{{name}}}}} placeholder."
            )
        parts.append(resolved)
        pos = end + 2

    return "".join(parts).replace(_ESCAPED_OPEN, "{{").replace(_ESCAPED_CLOSE, "}}")


def resolve_env_template_mapping(values: dict[str, Any], *, location: str) -> dict[str, str]:
    """Resolve placeholders in mapping values, coercing keys and values to strings."""
    return {
        str(key): resolve_env_templates(str(value), location=f"{location}[{key!r}]") for key, value in values.items()
    }


def resolve_env_template_values(value: Any, *, location: str) -> Any:
    """Recursively resolve placeholders in string values within JSON-like data."""
    if isinstance(value, str):
        return resolve_env_templates(value, location=location)
    if isinstance(value, list):
        return [resolve_env_template_values(item, location=f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: resolve_env_template_values(item, location=f"{location}[{key!r}]") for key, item in value.items()}
    return value
