# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Basic configuration panel — composable widget for the Basic tab."""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.text import Text
from textual import on
from textual.containers import HorizontalGroup, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Label, Static

from chrys.app.tui.i18n import render_str, render_text, widget_localizer
from chrys.app.tui.screens.agents.validation_messages import (
    DESCRIPTION as _DESCRIPTION,
)
from chrys.app.tui.screens.agents.validation_messages import (
    DISPLAY_NAME as _DISPLAY_NAME,
)
from chrys.app.tui.screens.agents.validation_messages import (
    DISPLAY_NAME_FIELD,
    FIELD_REQUIRED,
    PROFILE_NAME_FIELD,
    PROFILE_NAME_FORMAT,
    SELECT_MODEL_PROFILE,
)
from chrys.app.tui.widgets import Checkbox, EnhancedTextArea, Select
from chrys.app.tui.widgets import EnhancedInput as Input
from chrys.foundation.i18n import msg

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile


_DEFAULT_PROFILE = msg("tui.agent_config.basic.default_profile", fallback="(default)")
_USE_ACTIVE_MODEL_PROFILE = msg(
    "tui.agent_config.basic.use_active_model_profile",
    fallback="Use active model profile (Current: {name})",
)
_USE_PROFILE = msg("tui.agent_config.basic.use_profile", fallback="Use profile:")
_AGENT_PROFILE = msg("tui.agent_config.basic.agent_profile", fallback="Agent Profile")
_AGENT_PROFILE_DESCRIPTION = msg(
    "tui.agent_config.basic.agent_profile_description",
    fallback="Configure profile name, description, and metadata",
)
_NAME = msg("tui.agent_config.basic.name", fallback="Name")
_PROFILE_NAME_PLACEHOLDER = msg("tui.agent_config.basic.profile_name_placeholder", fallback="profile-name")
_DISPLAY_NAME_PLACEHOLDER = msg("tui.agent_config.basic.display_name_placeholder", fallback="User-friendly name")
_AGENT_TYPE = msg("tui.agent_config.basic.agent_type", fallback="Agent Type")
_BUILT_IN = msg("tui.agent_config.basic.built_in", fallback="Built-in")
_EXTERNAL_ACP = msg("tui.agent_config.basic.external_acp", fallback="External ACP")
_AGENT_TYPE_NOTE = msg(
    "tui.agent_config.basic.agent_type_note",
    fallback=(
        "Switching an existing profile to External ACP clears its model-driven instructions, tools, sub-agents, "
        "skills, MCP, memory, compaction, model, and approval settings."
    ),
)
_SUB_AGENT_ONLY = msg("tui.agent_config.basic.sub_agent_only", fallback="Sub-Agent only")
_SUB_AGENT_ONLY_DESCRIPTION = msg(
    "tui.agent_config.basic.sub_agent_only_description",
    fallback="Focused helper agent called by other agents; cannot be selected as the main agent",
)
_MODEL_PROFILE = msg("tui.agent_config.basic.model_profile", fallback="Model Profile")
_MODEL_PROFILE_DESCRIPTION = msg(
    "tui.agent_config.basic.model_profile_description",
    fallback="Pick the model profile this agent runs with",
)


class BasicConfigPanel(VerticalScroll):
    """Composable widget for the Basic profile configuration tab.

    Exposes core profile fields: name, display_name, description,
    instructions, and sub_agent_only.
    """

    DEFAULT_CSS = """
    BasicConfigPanel {
        height: 1fr;
        /* Left inset only: the scrollbar is carved out inside the padding,
           so right padding would sit right of the scrollbar. The right inset
           rides child margins below instead. */
        padding: 0 0 0 2;
        scrollbar-size-vertical: 1;
        scrollbar-gutter: stable;
    }
    BasicConfigPanel > * {
        margin-right: 1;
    }
    BasicConfigPanel .bc-section-title {
        color: $secondary;
        text-style: bold;
        height: 1;
    }
    BasicConfigPanel .bc-section-desc {
        color: $text-muted;
        height: 1;
        margin: 0 1 1 0;
    }
    BasicConfigPanel .bc-label {
        height: 1;
        color: $text-muted;
        margin: 1 0 0 0;
    }
    BasicConfigPanel .bc-label-first {
        height: 1;
        color: $text-muted;
        margin: 0;
    }
    BasicConfigPanel Input {
        height: 1;
        border: none;
        background: $foreground 8%;
        padding: 0 1;
    }
    BasicConfigPanel Input:focus {
        border: none;
        background: $foreground 12%;
    }
    BasicConfigPanel Input.-readonly {
        color: $text-muted;
    }
    BasicConfigPanel EnhancedTextArea {
        border: none;
        background: $foreground 8%;
        padding: 0 0;
        scrollbar-size-vertical: 1;
    }
    BasicConfigPanel EnhancedTextArea:focus {
        border: none;
        background: $foreground 12%;
    }
    BasicConfigPanel EnhancedTextArea .text-area--cursor-line {
        background: $primary 15%;
    }
    BasicConfigPanel #bc-description {
        height: 5;
    }
    BasicConfigPanel Checkbox {
        height: 1;
        padding: 0;
        border: none;
        background: transparent;
        margin: 1 0 0 0;
    }
    BasicConfigPanel Checkbox > .toggle--button {
        color: $foreground 35%;
        background: $foreground 6%;
    }
    BasicConfigPanel Checkbox.-on > .toggle--button {
        color: $success;
        background: $secondary 12%;
    }
    BasicConfigPanel .bc-option-desc {
        color: $text-muted;
        height: auto;
        width: 1fr;
        margin: 0 1 1 4;
        text-wrap: wrap;
        text-overflow: fold;
    }
    BasicConfigPanel .bc-type-note {
        color: $text-muted;
        height: auto;
        width: 1fr;
        margin: 0 1 1 4;
        text-wrap: wrap;
        text-overflow: fold;
    }
    BasicConfigPanel Select {
        height: auto;
    }
    BasicConfigPanel SelectCurrent {
        height: 1;
        border: none;
        background: $foreground 8%;
        padding: 0 1;
    }
    BasicConfigPanel SelectOverlay {
        height: auto;
        max-height: 12;
        border: round $tui-border-primary $border-opacity;
        background: $surface;
    }
    BasicConfigPanel #bc-model-row {
        height: auto;
        width: 1fr;
        margin: 1 1 0 0;
    }
    BasicConfigPanel #bc-model-row Checkbox {
        margin: 0;
        width: auto;
    }
    BasicConfigPanel #bc-model-profile {
        height: 1;
        width: 1fr;
        margin: 0 2 0 1;
    }
    BasicConfigPanel #bc-model-profile.-hidden {
        display: none;
    }
    BasicConfigPanel .bc-section-separator {
        height: auto;
        max-height: 1;
        margin: 1 1 1 0;
        border-top: solid $tui-border-foreground 15%;
    }
    """

    def __init__(
        self,
        profile: AgentProfile,
        is_builtin: bool = False,
        *,
        model_registry: ModelProfileRegistry | None = None,
        active_profile_id: str = "",
        read_only: bool = False,
    ) -> None:
        self._profile = profile
        self._is_builtin = is_builtin
        self._model_registry = model_registry
        self._active_profile_id = active_profile_id
        self._read_only = read_only
        super().__init__()

    # ── Model override helpers ───────────────────────────────────────

    def _available_model_profiles(self) -> list[ModelProfile]:
        if self._model_registry is None:
            return []
        return list(self._model_registry.list_profiles())

    def _active_profile_name(self) -> str:
        """Human-readable name of the session-active model profile, or a placeholder."""
        if self._model_registry is None or not self._active_profile_id:
            return render_str(widget_localizer(self), _DEFAULT_PROFILE.bind())
        profile = self._model_registry.get(self._active_profile_id)
        if profile is None:
            return render_str(widget_localizer(self), _DEFAULT_PROFILE.bind())
        return profile.name

    def _override_is_resolvable(self) -> bool:
        """True iff the profile's saved override id exists in the registry."""
        pid = self._profile.model.profile_id if self._profile.model else ""
        if not pid or self._model_registry is None:
            return False
        return self._model_registry.get(pid) is not None

    def _checkbox_label(self, use_active: bool) -> Text:
        """Label for the 'Use active model profile' checkbox.

        Checked  → "Use active model profile ({name})"
        Unchecked → "Use profile:" (paired with the inline Select)
        """
        localizer = widget_localizer(self)
        if use_active:
            return render_text(
                localizer,
                _USE_ACTIVE_MODEL_PROFILE.bind(name=self._active_profile_name()),
            )
        return render_text(localizer, _USE_PROFILE.bind())

    def compose(self) -> ComposeResult:
        p = self._profile
        is_acp = p.acp is not None
        localizer = widget_localizer(self)

        yield Label(render_str(localizer, _AGENT_PROFILE.bind()), classes="bc-section-title")
        yield Label(render_str(localizer, _AGENT_PROFILE_DESCRIPTION.bind()), classes="bc-section-desc")

        yield Label(f"[red]*[/red] {escape(render_str(localizer, _NAME.bind()))}", classes="bc-label-first")
        name_input = Input(
            value=p.name,
            placeholder=render_str(localizer, _PROFILE_NAME_PLACEHOLDER.bind()),
            id="bc-name",
        )
        if self._is_builtin:
            name_input.disabled = True
            name_input.add_class("-readonly")
        yield name_input

        yield Label(f"[red]*[/red] {escape(render_str(localizer, _DISPLAY_NAME.bind()))}", classes="bc-label")
        yield Input(
            value=p.display_name,
            placeholder=render_str(localizer, _DISPLAY_NAME_PLACEHOLDER.bind()),
            id="bc-display-name",
        )

        yield Label(f"[red]*[/red] {escape(render_str(localizer, _DESCRIPTION.bind()))}", classes="bc-label")
        yield EnhancedTextArea(
            p.description,
            id="bc-description",
        )

        yield Label(render_str(localizer, _AGENT_TYPE.bind()), classes="bc-label")
        agent_type = Select(
            [
                (Text(render_str(localizer, _BUILT_IN.bind())), "builtin"),
                (Text(render_str(localizer, _EXTERNAL_ACP.bind())), "acp"),
            ],
            id="bc-agent-type",
            allow_blank=False,
            value="acp" if is_acp else "builtin",
        )
        agent_type.disabled = self._is_builtin
        yield agent_type
        yield Label(
            render_str(localizer, _AGENT_TYPE_NOTE.bind()),
            classes="bc-type-note",
            id="bc-agent-type-note",
        )

        sub_agent_only = Checkbox(
            render_str(localizer, _SUB_AGENT_ONLY.bind()),
            value=True if is_acp else p.sub_agent_only,
            id="bc-sub-agent-only",
        )
        sub_agent_only.disabled = is_acp
        yield sub_agent_only
        yield Label(
            render_str(localizer, _SUB_AGENT_ONLY_DESCRIPTION.bind()),
            classes="bc-option-desc",
        )

        # ── Model profile override ────────────────────────────────
        # Editable for builtins too — saving creates a user-side shadow
        # copy in ``~/.chrys/agents/{name}.yaml`` that overrides the
        # baked-in definition on the next load.
        for header in (
            Static("", classes="bc-section-separator bc-model-section"),
            Label(render_str(localizer, _MODEL_PROFILE.bind()), classes="bc-section-title bc-model-section"),
            Label(
                render_str(localizer, _MODEL_PROFILE_DESCRIPTION.bind()),
                classes="bc-section-desc bc-model-section",
            ),
        ):
            header.display = not is_acp
            yield header

        mp_list = self._available_model_profiles()
        options = [(Text(mp.name), mp.id) for mp in mp_list]
        # No model profiles configured → nothing to override with; lock
        # the checkbox on "Use active" so validation never fails.
        no_profiles = not options
        use_active = no_profiles or not self._override_is_resolvable()
        initial_id = self._profile.model.profile_id if self._override_is_resolvable() else None
        select_classes = "-hidden" if use_active else ""
        with HorizontalGroup(id="bc-model-row", classes="bc-model-section") as model_row:
            model_row.display = not is_acp
            checkbox = Checkbox(
                self._checkbox_label(use_active),
                value=use_active,
                id="bc-model-use-active",
            )
            if no_profiles:
                checkbox.disabled = True
            yield checkbox
            yield Select(
                options,
                id="bc-model-profile",
                allow_blank=True,
                value=initial_id or Select.NULL,
                classes=select_classes,
            )

    @on(Select.Changed, "#bc-agent-type")
    def _on_agent_type_changed(self, event: Select.Changed) -> None:
        self._apply_agent_type_state(event.value == "acp")

    def _apply_agent_type_state(self, is_acp: bool) -> None:
        sub_agent_only = self.query_one("#bc-sub-agent-only", Checkbox)
        if is_acp:
            sub_agent_only.value = True
            sub_agent_only.disabled = True
        else:
            # Never re-enable on a read-only screen: the Select posts its
            # initial Changed after the screen's read-only sweep, and an
            # unconditional enable here would undo that sweep.
            sub_agent_only.disabled = self._read_only
        # An ACP agent runs on its external process, not a chrys model
        # profile — hide the whole section, headers included.
        for widget in self.query(".bc-model-section"):
            widget.display = not is_acp

    @on(Checkbox.Changed, "#bc-model-use-active")
    def _on_use_active_toggled(self, event: Checkbox.Changed) -> None:
        """Show/hide the Select and update the checkbox label."""
        select = self.query_one("#bc-model-profile", Select)
        checkbox = self.query_one("#bc-model-use-active", Checkbox)
        checkbox.label = self._checkbox_label(event.value)
        if event.value:
            select.add_class("-hidden")
            # Returning to "active" wipes any previous override choice so
            # the user re-picks intentionally next time.
            select.value = Select.NULL
        else:
            select.remove_class("-hidden")

    def get_config(self) -> dict:
        """Read current widget state into a dict of basic profile fields."""
        name = self._profile.name.strip()
        with contextlib.suppress(NoMatches):
            name = self.query_one("#bc-name", Input).value.strip()

        display_name = self._profile.display_name.strip()
        with contextlib.suppress(NoMatches):
            display_name = self.query_one("#bc-display-name", Input).value.strip()

        description = self._profile.description.strip()
        with contextlib.suppress(NoMatches):
            description = self.query_one("#bc-description", EnhancedTextArea).text.strip()

        sub_agent_only = self._profile.sub_agent_only
        with contextlib.suppress(NoMatches):
            sub_agent_only = bool(self.query_one("#bc-sub-agent-only", Checkbox).value)

        agent_type = "acp" if self._profile.acp is not None else "builtin"
        with contextlib.suppress(NoMatches):
            selected_type = self.query_one("#bc-agent-type", Select).value
            if selected_type in {"builtin", "acp"}:
                agent_type = str(selected_type)
        if agent_type == "acp":
            sub_agent_only = True

        seed_model_profile_id = self._profile.model.profile_id if self._override_is_resolvable() else ""
        use_active: bool | None = None
        with contextlib.suppress(NoMatches):
            use_active = bool(self.query_one("#bc-model-use-active", Checkbox).value)
        selected: object | None = None
        with contextlib.suppress(NoMatches):
            selected = self.query_one("#bc-model-profile", Select).value
        if use_active is True:
            model_profile_id = ""
        elif selected is not None:
            model_profile_id = "" if selected is Select.NULL else str(selected)
        else:
            model_profile_id = seed_model_profile_id
        return {
            "name": name,
            "display_name": display_name,
            "description": description,
            "sub_agent_only": sub_agent_only,
            "agent_type": agent_type,
            "model_profile_id": model_profile_id,
        }

    def validate(self) -> list[str]:
        """Validate basic configuration fields."""
        localizer = widget_localizer(self)
        errors: list[str] = []
        name = self.query_one("#bc-name", Input).value.strip()
        if not name:
            errors.append(
                render_str(
                    localizer,
                    FIELD_REQUIRED.bind(field=render_str(localizer, PROFILE_NAME_FIELD.bind())),
                )
            )
        elif not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", name):
            errors.append(
                render_str(
                    localizer,
                    PROFILE_NAME_FORMAT.bind(field=render_str(localizer, PROFILE_NAME_FIELD.bind())),
                )
            )
        if not self.query_one("#bc-display-name", Input).value.strip():
            errors.append(
                render_str(
                    localizer,
                    FIELD_REQUIRED.bind(field=render_str(localizer, DISPLAY_NAME_FIELD.bind())),
                )
            )
        if not self.query_one("#bc-description", EnhancedTextArea).text.strip():
            errors.append(
                render_str(
                    localizer,
                    FIELD_REQUIRED.bind(field=render_str(localizer, _DESCRIPTION.bind())),
                )
            )
        is_acp = self.query_one("#bc-agent-type", Select).value == "acp"
        # ACP profiles have no model override. For model-driven profiles, an
        # unchecked override requires a concrete model selection.
        if not is_acp and not self.query_one("#bc-model-use-active", Checkbox).value:
            selected = self.query_one("#bc-model-profile", Select).value
            if selected is Select.NULL:
                errors.append(render_str(localizer, SELECT_MODEL_PROFILE.bind()))
        return errors
