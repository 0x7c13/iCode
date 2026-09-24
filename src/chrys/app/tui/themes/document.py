# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Theme drafts and field-scoped, one-shot edit transactions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from textual.theme import Theme

SURFACES = ("background", "surface", "panel", "boost")


def copy_theme(theme: Theme, *, name: str | None = None) -> Theme:
    """Textual Theme is mutable; its variables must never be shared with drafts."""
    return replace(theme, name=theme.name if name is None else name, variables=dict(theme.variables))


@dataclass(frozen=True)
class EditToken:
    document: UUID
    field: str
    transaction: UUID


@dataclass
class EditTransaction:
    token: EditToken
    candidate: Theme


class ThemeDocument:
    """Opening baseline, saved baseline and history are separate concepts."""

    def __init__(self, theme: Theme, *, is_new: bool = False) -> None:
        self.is_new = is_new
        self.id = uuid4()
        self.original = copy_theme(theme)
        self.saved = copy_theme(theme)
        self.draft = copy_theme(theme)
        self.undo_stack: list[Theme] = []
        self.redo_stack: list[Theme] = []
        self.transaction: EditTransaction | None = None

    @property
    def changed(self) -> bool:
        return self.draft != self.original

    @property
    def unsaved(self) -> bool:
        return self.is_new or self.draft != self.saved

    def begin(self, field: str) -> EditToken:
        self.cancel()
        token = EditToken(self.id, field, uuid4())
        self.transaction = EditTransaction(token, copy_theme(self.draft))
        return token

    def owns(self, token: EditToken) -> bool:
        return self.transaction is not None and self.transaction.token == token

    def require_transaction(self) -> EditTransaction:
        if self.transaction is None:
            raise RuntimeError("The theme document has no active edit transaction.")
        return self.transaction

    def stage(self, token: EditToken, value: str | None) -> Theme | None:
        if not self.owns(token):
            return None
        candidate = copy_theme(self.draft)
        kind, name = token.field.split(":", 1)
        if kind == "color":
            setattr(candidate, name, value)
        elif kind == "var":
            if value is None:
                candidate.variables.pop(name, None)
            else:
                candidate.variables[name] = value
        else:
            raise ValueError(f"Unknown theme field: {token.field}")
        self.transaction = EditTransaction(token, candidate)
        return candidate

    def commit(self, token: EditToken) -> bool:
        if not self.owns(token):
            return False
        candidate = self.require_transaction().candidate
        self.transaction = None
        return self.replace(candidate)

    def cancel(self) -> None:
        self.transaction = None

    def replace(self, candidate: Theme) -> bool:
        self.cancel()
        if candidate == self.draft:
            return False
        self.undo_stack.append(copy_theme(self.draft))
        del self.undo_stack[:-100]
        self.redo_stack.clear()
        self.draft = copy_theme(candidate)
        return True

    def undo(self) -> None:
        self.cancel()
        if self.undo_stack:
            self.redo_stack.append(self.draft)
            self.draft = self.undo_stack.pop()

    def redo(self) -> None:
        self.cancel()
        if self.redo_stack:
            self.undo_stack.append(self.draft)
            self.draft = self.redo_stack.pop()

    def mark_saved(self) -> None:
        """A future writer calls this only after a successful atomic write."""
        self.is_new = False
        self.saved = copy_theme(self.draft)
