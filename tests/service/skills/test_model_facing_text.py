# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Skill text the model sees stays sendable, whatever Unicode a SKILL.md or a file name holds."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from chrys.orchestration.engine.run.input_refs import format_skill_reference_reminder, parse_skill_reference
from chrys.service.skills.model import Skill, SkillResource
from chrys.service.skills.provider import ChrysSkillsProvider, _render_catalog_block, _skill_revision, _tokenizer

if TYPE_CHECKING:
    from pathlib import Path


def _assert_sendable(text: str) -> None:
    # How a request body is encoded.
    json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")


async def _provider(*skills: Skill) -> ChrysSkillsProvider:
    skill_list = list(skills)

    async def load() -> list[Skill]:
        return skill_list

    provider = ChrysSkillsProvider(load)
    await provider.initialize()
    return provider


def _surrogate_skill() -> Skill:
    return Skill(name="convert", description="Converts files \ud800 \udcff", content="Body \udcff")


async def test_lone_surrogates_in_a_skill_keep_the_provider_and_its_text_sendable() -> None:
    provider = await _provider(_surrogate_skill())

    catalog = _render_catalog_block(provider._skills)
    loaded = await provider._load_skill(provider._skills, "convert")

    assert _skill_revision(provider._skills[0]) == _skill_revision(_surrogate_skill())
    _assert_sendable(catalog)
    _assert_sendable(loaded)
    assert r"Converts files \ud800 \udcff" in catalog
    assert loaded.startswith(r"Body \udcff")


async def test_a_slash_reference_to_a_skill_with_lone_surrogates_stays_sendable() -> None:
    provider = await _provider(_surrogate_skill())

    reference = parse_skill_reference("/convert this file", provider.skill_details())

    assert reference is not None
    reminder = format_skill_reference_reminder(reference)
    _assert_sendable(reminder)
    assert r"(Converts files \ud800 \udcff)" in reminder


async def test_a_resource_with_lone_surrogates_stays_within_its_token_budget() -> None:
    skill = Skill(
        name="convert",
        description="d",
        content="Body.",
        resources=[SkillResource(name="ref.md", content="\udcff " * 400)],
    )
    provider = await _provider(skill)

    read = await provider._read_skill_resource(provider._skills, "convert", "ref.md", 100)

    _assert_sendable(read)
    assert _tokenizer.count_tokens(read) <= 100


@pytest.mark.parametrize("literal_twin", [False, True])
async def test_a_resource_with_an_undecodable_name_is_shown_escaped_and_read_by_that_name(
    tmp_path: Path, literal_twin: bool
) -> None:
    # How a file name holding the byte 0xFF loads on POSIX.
    raw_name = "ref_\udcff.md"
    shown_name = r"ref_\udcff.md"
    resources = [SkillResource(name=raw_name, content="RAW BYTES")]
    if literal_twin:
        resources.append(SkillResource(name=shown_name, content="LITERAL NAME"))
    skill = Skill(name="raw", description="d", content="Body.", path=str(tmp_path), resources=resources)
    provider = await _provider(skill)

    loaded = await provider._load_skill(provider._skills, "raw")
    read = await provider._read_skill_resource(provider._skills, "raw", shown_name)
    context = provider._read_skill_resource_call_context({"skill_name": "raw", "resource_name": shown_name})

    _assert_sendable(loaded)
    assert f'name="{shown_name}"' in loaded
    # A name that matches as given wins over one that only shows that way.
    assert read == ("LITERAL NAME" if literal_twin else "RAW BYTES")
    assert context is not None and context["resource_name"] == (shown_name if literal_twin else raw_name)
