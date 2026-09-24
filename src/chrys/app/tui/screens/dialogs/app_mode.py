# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""App mode picker, matching the approval mode menu."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.widgets import OptionList

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets.option_menu import MenuOption, MenuOptionList, OptionMenu
from chrys.app.tui.widgets.workflow import text

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController


class AppModeDialog(BaseDialog[bool | None]):
    DEFAULT_CSS = "AppModeDialog { align: center middle; }"

    BINDINGS: ClassVar[list] = [localized_binding("escape", "dismiss", CLOSE_BINDING)]

    def __init__(self, workflow: bool, *, locale_controller: LocaleController | None = None) -> None:
        super().__init__()
        self.workflow = workflow
        self.locale_controller = locale_controller

    def compose(self) -> ComposeResult:
        with OptionMenu(id="container") as container:
            container.border_title = Text(text.render(text.APP_MODE.bind(), self.locale_controller))
            options: list[MenuOption] = []
            for workflow, label, description in (
                (False, text.MODE_CHAT, text.CHAT_DESCRIPTION),
                (True, text.MODE_WORKFLOW, text.WORKFLOW_DESCRIPTION),
            ):
                options.append(
                    MenuOption(
                        text.render(label.bind(), self.locale_controller),
                        text.render(description.bind(), self.locale_controller),
                        id="workflow" if workflow else "chat",
                        current=workflow == self.workflow,
                    )
                )
            yield MenuOptionList(options)

    @on(OptionList.OptionSelected)
    def mode_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id == "workflow")
