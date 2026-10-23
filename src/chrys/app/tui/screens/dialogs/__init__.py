# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reusable modal dialogs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chrys.app.tui.screens.dialogs.agent_load import AgentLoadDialog
    from chrys.app.tui.screens.dialogs.approval import ApprovalDialog
    from chrys.app.tui.screens.dialogs.approval.mode import ApprovalModeScreen
    from chrys.app.tui.screens.dialogs.ask_user import AskUserDialog
    from chrys.app.tui.screens.dialogs.base import BaseDialog
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog
    from chrys.app.tui.screens.dialogs.connection_test import ConnectionTestDialog
    from chrys.app.tui.screens.dialogs.editor import EditorDialog, EditorDialogResult
    from chrys.app.tui.screens.dialogs.file_picker import FilePicker, FilePickerMode
    from chrys.app.tui.screens.dialogs.image_compression import ImageCompressionDialog
    from chrys.app.tui.screens.dialogs.man_page import ManPageDialog
    from chrys.app.tui.screens.dialogs.session_title import SessionTitleDialog
    from chrys.app.tui.screens.dialogs.vision_unsupported import VisionUnsupportedDialog

_EXPORTS = {
    "AgentLoadDialog": "chrys.app.tui.screens.dialogs.agent_load",
    "ApprovalDialog": "chrys.app.tui.screens.dialogs.approval",
    "ApprovalModeScreen": "chrys.app.tui.screens.dialogs.approval.mode",
    "AskUserDialog": "chrys.app.tui.screens.dialogs.ask_user",
    "BaseDialog": "chrys.app.tui.screens.dialogs.base",
    "ConfirmDialog": "chrys.app.tui.screens.dialogs.confirm",
    "ConnectionTestDialog": "chrys.app.tui.screens.dialogs.connection_test",
    "EditorDialog": "chrys.app.tui.screens.dialogs.editor",
    "EditorDialogResult": "chrys.app.tui.screens.dialogs.editor",
    "FilePicker": "chrys.app.tui.screens.dialogs.file_picker",
    "FilePickerMode": "chrys.app.tui.screens.dialogs.file_picker",
    "ImageCompressionDialog": "chrys.app.tui.screens.dialogs.image_compression",
    "ManPageDialog": "chrys.app.tui.screens.dialogs.man_page",
    "SessionTitleDialog": "chrys.app.tui.screens.dialogs.session_title",
    "VisionUnsupportedDialog": "chrys.app.tui.screens.dialogs.vision_unsupported",
}


__all__ = [
    "AgentLoadDialog",
    "ApprovalDialog",
    "ApprovalModeScreen",
    "AskUserDialog",
    "BaseDialog",
    "ConfirmDialog",
    "ConnectionTestDialog",
    "EditorDialog",
    "EditorDialogResult",
    "FilePicker",
    "FilePickerMode",
    "ImageCompressionDialog",
    "ManPageDialog",
    "SessionTitleDialog",
    "VisionUnsupportedDialog",
]


def __getattr__(name: str) -> Any:
    """Resolve public feature exports only when requested."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = vars(import_module(module_name))[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include public exports without importing their implementations."""
    return sorted({*globals(), *__all__})
