# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Public editor widget and keymap extension surface."""

from chrys.app.tui.widgets.editor.highlighter import (
    MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF,
    MARKDOWN_HIGHLIGHT_LINE_CUTOFF,
    MARKDOWN_HIGHLIGHT_PLAIN_STATUS,
    MarkdownHighlightedTextArea,
    MarkdownHighlightTheme,
)
from chrys.app.tui.widgets.editor.keymaps import (
    DEFAULT_EDITOR_KEYMAP_REGISTRY,
    EditorKeymapRegistry,
    EmacsKeymap,
    StandardKeymap,
)
from chrys.app.tui.widgets.editor.types import (
    MESSAGE_EDITOR_CHARACTER_LIMIT_STATUS,
    MESSAGE_EDITOR_MAX_CHARACTERS,
    EditorBufferSnapshot,
    EditorCommand,
    EditorCommandHost,
    EditorIntent,
    EditorKeymap,
    EditorKeyResult,
    EditorKeyStroke,
    EditorLocation,
    EditorMode,
    EditorRegister,
    EditorRegisterKind,
    EditorScrollDirection,
    EditorStatus,
    EditorTextRange,
)
from chrys.app.tui.widgets.editor.vim import (
    MAX_VIM_COUNT,
    VimKeymap,
    VimMotion,
    VimOperator,
    VimState,
    VimTextObject,
)
from chrys.app.tui.widgets.editor.widget import MessageEditor

__all__ = [
    "DEFAULT_EDITOR_KEYMAP_REGISTRY",
    "MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF",
    "MARKDOWN_HIGHLIGHT_LINE_CUTOFF",
    "MARKDOWN_HIGHLIGHT_PLAIN_STATUS",
    "MAX_VIM_COUNT",
    "MESSAGE_EDITOR_CHARACTER_LIMIT_STATUS",
    "MESSAGE_EDITOR_MAX_CHARACTERS",
    "EditorBufferSnapshot",
    "EditorCommand",
    "EditorCommandHost",
    "EditorIntent",
    "EditorKeyResult",
    "EditorKeyStroke",
    "EditorKeymap",
    "EditorKeymapRegistry",
    "EditorLocation",
    "EditorMode",
    "EditorRegister",
    "EditorRegisterKind",
    "EditorScrollDirection",
    "EditorStatus",
    "EditorTextRange",
    "EmacsKeymap",
    "MarkdownHighlightTheme",
    "MarkdownHighlightedTextArea",
    "MessageEditor",
    "StandardKeymap",
    "VimKeymap",
    "VimMotion",
    "VimOperator",
    "VimState",
    "VimTextObject",
]
