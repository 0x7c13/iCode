# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for renaming agent profiles on the configuration screen: retargeting refs, rollback on failure."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import (
    Button,
    Input,
    Select,
)

from chrys.service.profiles.agents.loader import load_profile_from_yaml
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    SubAgentRef,
    SubAgentsConfig,
)
from tests.app.tui.screens._agent_config_support import (
    _DEFAULT_WAIT_TIMEOUT,
    _activate_agent_config_tab,
    _draft_for_original,
    _wait_for_hydrated,
    _wait_for_selected_profile_name,
    _wait_for_selectors,
    make_profile,
    open_agent_config,
    registry_with,
)
from tests.support.waiting import wait_for

# The polling helpers in _agent_config_support share one 45s ceiling, below
# the repository's 60s default per-test timeout. This module retains a 120s
# override for expensive screen mounting before the first wait; once polling
# starts, the helper deadline still produces a clean AssertionError before the
# thread timeout can kill the xdist worker and surface only "worker gwN crashed".
pytestmark = [pytest.mark.timeout(120), pytest.mark.usefixtures("isolated_chrys_config_dir")]


async def test_agent_config_saves_cross_rename_without_clobbering_profiles(tmp_path: Path) -> None:
    registry = registry_with(
        make_profile(
            "A",
            display_name="Agent A",
            description="First agent",
        ),
        make_profile(
            "B",
            display_name="Agent B",
            description="Second agent",
        ),
    )

    async with open_agent_config(registry, current_profile="Agent A") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "B"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        b_key = _draft_for_original(screen, "B").key
        screen._load_profile(b_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "C"
        await pilot.pause()

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    renamed_a = registry.get("B")
    renamed_b = registry.get("C")
    assert renamed_a is not None
    assert renamed_a.display_name == "Agent A"
    assert renamed_b is not None
    assert renamed_b.display_name == "Agent B"
    assert registry.get("A") is None
    assert (tmp_path / "agents" / "B.yaml").is_file()
    assert (tmp_path / "agents" / "C.yaml").is_file()


async def test_agent_config_cross_rename_retargets_refs_by_profile_identity() -> None:
    registry = registry_with(
        make_profile(
            "Parent",
            display_name="Parent Agent",
            description="Uses both agents",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="A"), SubAgentRef(profile="B")]),
        ),
        make_profile(
            "A",
            display_name="Agent A",
            description="First agent",
        ),
        make_profile(
            "B",
            display_name="Agent B",
            description="Second agent",
        ),
    )

    async with open_agent_config(registry, current_profile="Agent A") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "B"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        b_key = _draft_for_original(screen, "B").key
        screen._load_profile(b_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "C"
        await pilot.pause()

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    renamed_a = registry.get("B")
    renamed_b = registry.get("C")
    assert parent is not None
    assert renamed_a is not None
    assert renamed_a.display_name == "Agent A"
    assert renamed_b is not None
    assert renamed_b.display_name == "Agent B"
    assert [ref.profile for ref in parent.sub_agents.agents] == ["B", "C"]


async def test_agent_config_save_retargets_subagent_refs_after_rename() -> None:
    registry = registry_with(
        make_profile(
            "Parent",
            display_name="Parent Agent",
            description="Uses a child agent",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
        ),
        make_profile(
            "Child",
            display_name="Child Agent",
            description="Child agent",
        ),
    )

    async with open_agent_config(registry, current_profile="Child Agent") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        save = screen.query_one("#ac-save", Button)
        await wait_for(
            lambda: not save.disabled,
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="save enabled after profile rename",
        )
        save.press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Helper"]
    assert registry.get("Helper") is not None
    assert registry.get("Child") is None


async def test_agent_config_staged_rename_keeps_original_subagent_ref_selectable_until_save() -> None:
    registry = registry_with(
        make_profile(
            "Parent",
            display_name="Parent Agent",
            description="Uses a child agent",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
        ),
        make_profile(
            "Child",
            display_name="Child Agent",
            description="Child agent",
        ),
    )

    async with open_agent_config(registry, current_profile="Child Agent") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        parent_key = _draft_for_original(screen, "Parent").key
        screen._load_profile(parent_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        await _activate_agent_config_tab(screen, pilot, "sub-agents")
        await _wait_for_selectors(screen, pilot, "#sa-profile-0")
        select = screen.query_one("#sa-profile-0", Select)
        # SubAgentCard.get_config() falls back to its seed profile name when
        # the Select reactive hasn't settled, so _wait_for_hydrated returns
        # while the widget itself is still on Select.NULL under xdist load on
        # Windows CI.  Poll the widget directly before asserting.
        await wait_for(
            lambda: select.value == "Child",
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="sub-agent select reflects staged-rename original name",
        )
        assert select.value == "Child"

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Helper"]
    assert registry.get("Helper") is not None
    assert registry.get("Child") is None


async def test_agent_config_failed_rename_validation_rolls_back_retargeted_draft_refs() -> None:
    registry = registry_with(
        make_profile(
            "Parent",
            display_name="Parent Agent",
            description="Uses a child agent",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
        ),
        make_profile(
            "Child",
            display_name="Child Agent",
            description="Child agent",
        ),
        make_profile(
            "Helper",
            display_name="Existing Helper",
            description="Existing helper agent",
        ),
    )
    async with open_agent_config(registry, current_profile="Child Agent") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

        parent_draft = _draft_for_original(screen, "Parent")
        assert [ref.profile for ref in parent_draft.profile.sub_agents.agents] == ["Child"]
        assert registry.get("Child") is not None
        assert registry.get("Helper") is not None

        screen.query_one("#bc-name", Input).value = "Other"
        await pilot.pause()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Other"]
    assert registry.get("Other") is not None
    assert registry.get("Child") is None


async def test_agent_config_save_persists_retargeted_subagent_refs_after_rename(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    child = make_profile(
        "Child",
        display_name="Child Agent",
        description="Child agent",
    )
    fallback = make_profile(
        "Fallback",
        display_name="Fallback Agent",
        description="Fallback agent",
    )
    parent = make_profile(
        "parent",
        display_name="Parent",
        description="Parent profile",
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
    )
    for profile in (child, fallback, parent):
        save_profile(profile)
        registry.register(profile)

    async with open_agent_config(registry, current_profile="Child Agent") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        save = screen.query_one("#ac-save", Button)
        await wait_for(
            lambda: not save.disabled,
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="save enabled after retarget rename",
        )
        save.press()
        await pilot.pause()
        await wait_for(
            lambda: (
                (saved_parent := registry.get("parent")) is not None
                and [ref.profile for ref in saved_parent.sub_agents.agents] == ["Helper"]
            ),
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="renamed sub-agent refs persisted",
        )

    parent = registry.get("parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Helper"]
    saved_parent = load_profile_from_yaml(tmp_path / "agents" / "parent.yaml")
    assert [ref.profile for ref in saved_parent.sub_agents.agents] == ["Helper"]


async def test_agent_config_save_validates_retargeted_profiles_before_writing(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    child = make_profile(
        "Child",
        display_name="Child Agent",
        description="Child agent",
    )
    sibling = make_profile(
        "Sibling",
        display_name="Sibling Agent",
        description="Sibling agent",
    )
    parent = make_profile(
        "parent",
        display_name="Parent",
        description="Parent profile",
        sub_agents=SubAgentsConfig(
            agents=[
                SubAgentRef(profile="Child"),
                SubAgentRef(profile="Sibling", tool_name="Helper"),
            ]
        ),
    )
    for profile in (child, sibling, parent):
        save_profile(profile)
        registry.register(profile)
    original_parent_yaml = (tmp_path / "agents" / "parent.yaml").read_bytes()

    async with open_agent_config(registry, current_profile="Child Agent") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

    assert registry.get("Child") is not None
    assert registry.get("Helper") is None
    parent = registry.get("parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Child", "Sibling"]
    assert (tmp_path / "agents" / "parent.yaml").read_bytes() == original_parent_yaml


async def test_agent_config_save_persists_retargeted_builtin_shadow(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    registry.load_builtins()
    child = make_profile(
        "Child",
        display_name="Child Agent",
        description="Child agent",
    )
    code_shadow = make_profile(
        "Code",
        display_name="Code Agent",
        description="User shadow of built-in Code",
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
    )
    save_profile(code_shadow)
    registry.register(child)
    registry.register(code_shadow)
    assert registry.is_builtin("Code") is True

    async with open_agent_config(registry, current_profile="Child Agent") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        # Wait until Input.Changed has been harvested into the draft.
        # On slower Windows CI, pressing Save before the rename lands on
        # the draft causes the retarget pass to find an empty rename_map.
        await _wait_for_selected_profile_name(screen, pilot, "Helper")
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    shadow = registry.get("Code")
    assert shadow is not None
    assert [ref.profile for ref in shadow.sub_agents.agents] == ["Helper"]
    saved_shadow = load_profile_from_yaml(tmp_path / "agents" / "Code.yaml")
    assert [ref.profile for ref in saved_shadow.sub_agents.agents] == ["Helper"]


async def test_agent_config_case_only_rename_keeps_case_alias_target_file(tmp_path: Path) -> None:
    """Case-only rename keeps the target file; failure rollback is covered by _apply_all snapshots."""
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    profile = make_profile(
        "foo",
        display_name="Foo Agent",
        description="Custom profile",
    )
    save_profile(profile)
    registry.register(profile)

    old_path = tmp_path / "agents" / "foo.yaml"
    alias_path = tmp_path / "agents" / "Foo.yaml"
    if not alias_path.exists():
        # A case-only rename is only a real scenario on a case-insensitive
        # filesystem, where ``Foo.yaml`` already resolves to ``foo.yaml``. The
        # former symlink stand-in no longer models this: owner-only writes are
        # ``O_NOFOLLOW`` and would replace the link with a distinct file, so the
        # case-sensitive runner (Linux) is not a valid host for this assertion.
        pytest.skip("case-only rename requires a case-insensitive filesystem")

    async with open_agent_config(registry, current_profile="Foo Agent") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Foo"
        save = screen.query_one("#ac-save", Button)
        await _wait_for_selected_profile_name(screen, pilot, "Foo")
        await wait_for(
            lambda: not save.disabled,
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="save enabled after case-only rename",
        )
        save.press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    renamed = registry.get("Foo")
    assert renamed is not None
    assert registry.get("foo") is None
    assert old_path.is_file()
    saved = load_profile_from_yaml(alias_path)
    assert saved.name == "Foo"


async def test_agent_config_save_failure_rolls_back_cross_rename_files_and_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    profile_a = make_profile(
        "A",
        display_name="Agent A",
        description="First agent",
    )
    profile_b = make_profile(
        "B",
        display_name="Agent B",
        description="Second agent",
    )
    save_profile(profile_a)
    save_profile(profile_b)
    registry.register(profile_a)
    registry.register(profile_b)
    path_a = tmp_path / "agents" / "A.yaml"
    path_b = tmp_path / "agents" / "B.yaml"
    original_a = path_a.read_bytes()
    original_b = path_b.read_bytes()

    original_save_profile = save_profile

    def fail_on_c(profile, *args, **kwargs):
        if profile.name == "C":
            raise OSError("simulated save failure")
        return original_save_profile(profile, *args, **kwargs)

    monkeypatch.setattr("chrys.service.profiles.agents.serializer.save_profile", fail_on_c)

    async with open_agent_config(registry, current_profile="Agent A") as (screen, pilot):
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "B"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        b_key = _draft_for_original(screen, "B").key
        screen._load_profile(b_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "C"
        await pilot.pause()

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

    restored_a = registry.get("A")
    restored_b = registry.get("B")
    assert restored_a is not None
    assert restored_a.display_name == "Agent A"
    assert restored_b is not None
    assert restored_b.display_name == "Agent B"
    assert registry.get("C") is None
    assert path_a.read_bytes() == original_a
    assert path_b.read_bytes() == original_b
    assert not (tmp_path / "agents" / "C.yaml").exists()
