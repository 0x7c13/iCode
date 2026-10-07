# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Simplified-Chinese coverage for modal main-flow presentation boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from chrys.app.tui.i18n import LocaleController, render_str
from chrys.app.tui.screens.dialogs.agent_load import AgentLoadDialog, map_load_progress_prose
from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog
from chrys.app.tui.screens.dialogs.image_compression import ImageCompressionDialog
from chrys.app.tui.screens.main.dialog_controllers import (
    AgentLoadDialogController,
    ImageCompressionDialogController,
)
from chrys.app.tui.screens.main.navigation import MainNavigationController
from chrys.app.tui.screens.main.session_handlers import _FINISH_SESSION_RESTORED
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.app.tui.widgets.chrome.input_bar import InputBar
from chrys.app.tui.widgets.chrome.status_bar import StatusBar
from chrys.app.tui.widgets.trajectory.panel import _NO_ACTIVE_SESSION
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import AgentLoadProgress, ImageAttachmentCompressionStarted
from chrys.foundation.i18n import Localizer
from chrys.foundation.i18n.formatting import format_message


def _plain_content(widget: Static) -> str:
    content = widget.content
    return content.plain if isinstance(content, Text) else str(content)


def _zh_controller() -> LocaleController:
    return LocaleController(Settings(locale="zh-Hans"))


def test_session_json_empty_state_keeps_english_and_localizes_chinese() -> None:
    assert format_message(_NO_ACTIVE_SESSION.bind()) == "No active session."
    assert Localizer("zh-Hans").render(_NO_ACTIVE_SESSION.bind()) == "当前没有活动会话。"


class _StatusHostApp(App[None]):
    def __init__(self, controller: LocaleController) -> None:
        self.status = StatusBar(locale_controller=controller)
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.status


def _view_adapter(app: _StatusHostApp, controller: LocaleController) -> MainScreenViewAdapter:
    def query_one(selector: object) -> object:
        if selector is InputBar:
            return SimpleNamespace(set_clipboard_image_dir=lambda _directory: None)
        return app.query_one(cast(Any, selector))

    screen = SimpleNamespace(
        app=app,
        query_one=query_one,
    )
    return MainScreenViewAdapter(cast(Any, screen), state=MainScreenState(), locale_controller=controller)


@pytest.mark.asyncio
async def test_agent_load_titles_reach_chinese_dialog_and_status_boundaries() -> None:
    controller = _zh_controller()
    assert render_str(controller.localizer, AgentLoadDialogController.load_title("startup")) == "正在初始化智能体"
    assert render_str(controller.localizer, AgentLoadDialogController.load_title("restore")) == "正在恢复会话"

    app = _StatusHostApp(controller)
    async with app.run_test() as pilot:
        adapter = _view_adapter(app, controller)
        title = AgentLoadDialogController.load_title("switch")
        dialog = adapter.create_agent_load_dialog(title=title, subtitle="Code")
        adapter.prepare_agent_load_ui(
            title=title,
            session_id=None,
            update_clipboard_dir=False,
            capture_status_snapshot=False,
        )
        await app.push_screen(cast(Any, dialog))
        await pilot.pause()

        assert str(cast(AgentLoadDialog, dialog).query_one("#agent-load-container").border_title) == "正在切换智能体"
        assert _plain_content(app.status.query_one("#status-text", Static)) == "正在切换智能体"


@pytest.mark.asyncio
async def test_restore_percentage_renders_chinese_in_dialog_and_status() -> None:
    controller = _zh_controller()
    app = _StatusHostApp(controller)
    async with app.run_test() as pilot:
        adapter = _view_adapter(app, controller)
        dialog = cast(
            AgentLoadDialog,
            adapter.create_agent_load_dialog(
                title=AgentLoadDialogController.load_title("restore"),
                subtitle="session",
            ),
        )
        await app.push_screen(dialog)
        flow = AgentLoadDialogController(cast(Any, adapter))
        flow.dialog = dialog

        flow.update_session_history_progress(32, 40)
        await pilot.pause()

        assert dialog._messages[-1] == "正在恢复会话历史（80%）"  # noqa: RUF001
        assert _plain_content(app.status.query_one("#status-text", Static)) == "正在恢复会话历史（80%）"  # noqa: RUF001


def test_finish_message_ref_localizes_while_raw_text_passes_through() -> None:
    controller = _zh_controller()
    localized = AgentLoadDialog(locale_controller=controller)
    localized.finish(_FINISH_SESSION_RESTORED.bind(session_id="abc12345"))
    assert localized._messages[-1] == "会话已恢复：abc12345"  # noqa: RUF001

    raw = AgentLoadDialog(locale_controller=controller)
    raw.finish("provider supplied completion")
    assert raw._messages[-1] == "provider supplied completion"


def test_agent_load_semantic_sentinels_survive_chinese_rendering() -> None:
    dialog = AgentLoadDialog(locale_controller=_zh_controller())

    dialog.update_progress("Checking session availability", phase="session")
    dialog.update_progress("Session availability checked", phase="session")
    assert dialog._messages == ["会话可用性检查完成"]
    assert dialog._progress_entries[0].status == "done"

    dialog.update_progress("Resolving model profile", phase="model")
    dialog.update_progress("Loading built-in tools", phase="tools")
    assert dialog._messages[-2:] == ["模型配置已解析", "正在加载内置工具"]


def test_agent_load_count_and_failed_suffixes_localize() -> None:
    dialog = AgentLoadDialog(locale_controller=_zh_controller())
    dialog.update_progress("Connecting MCP servers", phase="mcp", current=1, total=2, failed=1)
    assert dialog._messages == ["MCP 服务器已连接：1/2，失败：1"]  # noqa: RUF001

    english = AgentLoadDialog()
    english.update_progress("Connecting MCP servers", phase="mcp", current=1, total=2, failed=1)
    assert english._messages == ["MCP servers connected: 1/2, failed: 1"]


def test_load_progress_prose_maps_to_localized_definitions() -> None:
    controller = _zh_controller()

    def zh(prose: str) -> str:
        reference = map_load_progress_prose(prose)
        assert reference is not None
        return render_str(controller.localizer, reference)

    assert zh("Resolving model profile") == "正在解析模型配置"
    assert zh("Capturing workspace context") == "正在采集工作区上下文"
    assert zh("Finalizing agent") == "正在完成智能体初始化"
    assert zh("Loading sub-agent tools") == "正在加载子智能体"
    assert zh("Loading sub-agent review") == "正在加载子智能体 review"
    assert zh("Skipped sub-agent review: cycle detected") == "已跳过子智能体 review：cycle detected"  # noqa: RUF001
    assert zh("Connecting MCP server fs") == "正在连接 MCP 服务器 fs"
    assert map_load_progress_prose("Some future prose") is None


def test_status_bar_receives_localized_load_progress_in_chinese() -> None:
    controller = _zh_controller()

    def _render(message: object) -> str:
        return message if isinstance(message, str) else render_str(controller.localizer, cast(Any, message))

    class _ZhStatusPort:
        def __init__(self) -> None:
            self.status_messages: list[str] = []

        def show_load_status(self, message: object) -> None:
            self.status_messages.append(_render(message))

        def render_status_message(self, message: object) -> str:
            return _render(message)

        def debug(self, _key: str, _message: str = "") -> None:
            return

    port = _ZhStatusPort()
    load_controller = AgentLoadDialogController(cast(Any, port))
    asyncio.run(load_controller.on_progress(AgentLoadProgress(phase="model", message="Resolving model profile")))
    asyncio.run(
        load_controller.on_progress(
            AgentLoadProgress(phase="mcp", message="Connecting MCP server fs", current=0, total=2)
        )
    )
    asyncio.run(
        load_controller.on_progress(
            AgentLoadProgress(phase="mcp", message="Connected MCP server fs", current=1, total=2, failed=1)
        )
    )

    assert port.status_messages == [
        "正在解析模型配置",
        "正在连接 MCP 服务器 fs (0/2)",
        "MCP 服务器 fs 已连接 (1/2，失败 1)",  # noqa: RUF001
    ]


class _CompressionPort:
    def __init__(self, controller: LocaleController | None) -> None:
        self.controller = controller
        self.dialog: ImageCompressionDialog | None = None

    def create_image_compression_dialog(self, *, title: object) -> ImageCompressionDialog:
        self.dialog = ImageCompressionDialog(cast(Any, title), locale_controller=self.controller)
        return self.dialog

    async def push_image_compression_dialog(self, _dialog: object) -> None:
        return

    def debug(self, _key: str, _message: str = "") -> None:
        return


@pytest.mark.asyncio
async def test_image_compression_plural_keeps_english_and_chinese_forms() -> None:
    english_port = _CompressionPort(None)
    await ImageCompressionDialogController(cast(Any, english_port)).on_started(
        ImageAttachmentCompressionStarted(image_count=1)
    )
    assert english_port.dialog is not None
    assert english_port.dialog._title == "Preparing Image"

    chinese_port = _CompressionPort(_zh_controller())
    await ImageCompressionDialogController(cast(Any, chinese_port)).on_started(
        ImageAttachmentCompressionStarted(image_count=2)
    )
    assert chinese_port.dialog is not None
    assert chinese_port.dialog._title == "正在准备图片"


class _ExitDialogView:
    def __init__(self, controller: LocaleController) -> None:
        self.controller = controller
        self.dialog: ConfirmDialog | None = None

    def open_confirm_dialog(self, **kwargs: Any) -> None:
        kwargs.pop("on_result")
        self.dialog = ConfirmDialog(**kwargs, locale_controller=self.controller)


@pytest.mark.asyncio
async def test_exit_confirmation_path_renders_chinese_title_and_labels() -> None:
    controller = _zh_controller()
    view = _ExitDialogView(controller)
    navigation = MainNavigationController(
        services=MainScreenServices(bus=EventBus()),
        view=cast(Any, view),
        is_agent_loading=lambda: False,
        is_agent_running=lambda: False,
        is_submit_pending=lambda: False,
        is_workflow_mode=lambda: False,
        has_messages=lambda: True,
        is_dashboard_visible=lambda: False,
        set_interrupt_confirm_active=lambda _active: None,
        publish_interrupt=lambda: None,
        dismiss_suggestions=lambda: False,
        cancel_pending_injection=lambda: False,
        delete_current_and_new=cast(Any, lambda _session_id: None),
        restore_session=cast(Any, lambda _session_id: None),
        flush_notifications=cast(Any, lambda: None),
        start_worker=lambda _work: None,
        debug=lambda _key, _message: None,
        locale_controller=controller,
    )

    navigation.confirm_exit()
    assert view.dialog is not None

    class _ConfirmHost(App[None]):
        def compose(self) -> ComposeResult:
            yield Static("host")

    app = _ConfirmHost()
    async with app.run_test() as pilot:
        await app.push_screen(view.dialog)
        await pilot.pause()

        assert view.dialog._title == "退出"
        assert view.dialog._message == f"是否退出 {APP_DISPLAY_NAME}？"  # noqa: RUF001
        assert view.dialog.query_one("#confirm-yes", Button).label.plain == "退出"
        assert view.dialog.query_one("#confirm-no", Button).label.plain == "取消"
