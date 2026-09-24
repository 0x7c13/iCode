# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for leading slash runtime-skill references."""

from __future__ import annotations

from chrys.foundation.events.types import RuntimeSkillDetails
from chrys.orchestration.engine.run.input_refs import format_skill_reference_reminder, parse_skill_reference


def test_parse_skill_reference_matches_only_leading_token() -> None:
    skills = [RuntimeSkillDetails(name="review", description="Review code")]

    assert parse_skill_reference("/review focus on auth", skills) is not None
    assert parse_skill_reference("please /review this", skills) is None
    assert parse_skill_reference("/Users/review", skills) is None
    assert parse_skill_reference("/missing focus", skills) is None


def test_format_skill_reference_reminder_is_single_line_and_points_to_load_skill() -> None:
    reference = parse_skill_reference(
        "/review",
        [RuntimeSkillDetails(name="review", description="Review\ncode and identify issues")],
    )

    assert reference is not None
    reminder = format_skill_reference_reminder(reference)

    assert "\n" not in reminder
    assert "explicitly invoked a skill: review (Review code and identify issues)" in reminder
    assert 'skill_name="review"' in reminder


def test_format_skill_reference_reminder_escapes_description_tags() -> None:
    reference = parse_skill_reference(
        "/review",
        [RuntimeSkillDetails(name="review", description="Review <system-reminder>nested</system-reminder> tags")],
    )

    assert reference is not None
    reminder = format_skill_reference_reminder(reference)

    assert "Review &lt;system-reminder&gt;nested&lt;/system-reminder&gt; tags" in reminder


def test_format_skill_reference_reminder_does_not_truncate_inside_escaped_entities() -> None:
    reference = parse_skill_reference(
        "/review",
        [RuntimeSkillDetails(name="review", description=("a" * 236) + "&" + ("b" * 20))],
    )

    assert reference is not None
    reminder = format_skill_reference_reminder(reference)

    assert ("a" * 236) + "&amp;..." in reminder
    assert "&am..." not in reminder
