# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The skill tools reject a top-level argument they do not define, instead of dropping it."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY
from chrys.service.skills.model import Skill, SkillResource, SkillScript
from chrys.service.skills.provider import ChrysSkillsProvider
from tests.kernel._fakes import _call_response, _result_contents, _stack, _text_response, _user


async def _provider() -> ChrysSkillsProvider:
    skill = Skill(
        name="pdf",
        description="Render PDFs",
        content="body",
        resources=[SkillResource(name="ref.md", content="ref")],
        scripts=[SkillScript(name="scripts/render.py", full_path="/nonexistent/scripts/render.py")],
    )

    async def load() -> list[Skill]:
        return [skill]

    provider = ChrysSkillsProvider(load)
    await provider.initialize()
    return provider


@pytest.mark.parametrize("unknown", [True, False])
@pytest.mark.parametrize(
    ("tool_name", "arguments", "unknown_argument"),
    [
        ("load_skill", {"skill_name": "pdf"}, {"version": "2"}),
        ("read_skill_resource", {"skill_name": "pdf", "resource_name": "ref.md"}, {"maxTokens": 500}),
        ("run_skill_script", {"skill_name": "pdf", "script_name": "scripts/render.py"}, {"argv": ["a.pdf"]}),
    ],
)
async def test_an_unknown_top_level_argument_is_an_argument_error(
    tool_name: str, arguments: dict[str, Any], unknown_argument: dict[str, Any], unknown: bool
) -> None:
    provider = await _provider()
    call_arguments = {**arguments, **unknown_argument} if unknown else arguments
    layer, _wire = _stack([_call_response(("c1", tool_name, call_arguments)), _text_response()])

    response = await layer.get_response([_user()], options={"tools": provider._chrys_tools})

    (result,) = _result_contents(response)
    rejected = str(result.result).startswith(f"Error: Invalid arguments for '{tool_name}'")
    assert rejected is unknown
    if unknown:
        assert next(iter(unknown_argument)) in str(result.result)
        assert result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
