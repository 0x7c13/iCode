# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Helpers for leading slash references to loaded runtime skills."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from chrys.foundation.events.types import RuntimeSkillDetails

_SKILL_REFERENCE_RE = re.compile(r"^[/\uff0f]([A-Za-z][A-Za-z0-9_-]*)(?:[ \t]|$)")
_WHITESPACE_RE = re.compile(r"\s+")
_DESCRIPTION_LIMIT = 240


@dataclass(frozen=True)
class SkillReference:
    """A resolved leading slash reference to a loaded runtime skill."""

    skill: RuntimeSkillDetails
    token: str


def parse_skill_reference(text: str, skills: Sequence[RuntimeSkillDetails]) -> SkillReference | None:
    """Return the loaded skill referenced by the leading slash token, if any."""
    match = _SKILL_REFERENCE_RE.match(text)
    if match is None:
        return None
    token = match.group(1)
    for skill in skills:
        if skill.name == token:
            return SkillReference(skill=skill, token=token)
    return None


def format_skill_reference_reminder(reference: SkillReference) -> str:
    """Format the short system reminder for a resolved skill reference."""
    skill = reference.skill
    description = _compact_description(skill.description)
    description_text = f" ({description})" if description else ""
    return (
        f"[Skill Reference] User explicitly invoked a skill: {skill.name}{description_text}. "
        "Apply this skill to the user's message; call load_skill with "
        f'skill_name="{skill.name}" if its full instructions are not yet loaded.'
    )


def _compact_description(description: str) -> str:
    """Collapse skill descriptions to a bounded single line for reminders."""
    compact = _WHITESPACE_RE.sub(" ", description).strip()
    if len(compact) > _DESCRIPTION_LIMIT:
        compact = compact[: _DESCRIPTION_LIMIT - 3].rstrip() + "..."
    return _escape_reminder_text(compact)


def _escape_reminder_text(text: str) -> str:
    """Escape tag-like content before embedding authored text in a reminder."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
