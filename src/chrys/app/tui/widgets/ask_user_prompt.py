# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared multi-question prompt used by ask-user modal and inline surfaces."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.cells import cell_len
from rich.console import Group
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.content import Content
from textual.message import Message
from textual.widgets import Static, TabbedContent, TabPane

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.util.source_text import sanitize_source_text
from chrys.app.tui.widgets.ask_user_controls import (
    AskUserContentResized,
    AskUserDraftChanged,
    AskUserFooterAction,
    AskUserInlineRequested,
    AskUserOptionChosen,
    AskUserOptions,
    AskUserResponseFooter,
    AskUserSubmitted,
    FooterPosition,
    ask_user_hanging_answer,
    ask_user_hanging_grid,
)
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.models.ask_user import AskUserAnswer, AskUserQuestion

if TYPE_CHECKING:
    from rich.console import RenderableType
    from textual.app import ComposeResult


_TAB_SUBMIT = msg("tui.ask_user.tab.submit", fallback="Submit")
_TITLE_SINGLE = msg("tui.ask_user.title", fallback="Question")
_TAB_FALLBACK_HEADER = msg("tui.ask_user.tab.fallback_header", fallback="Q{n}")
ASK_USER_REVIEW_TITLE_REF = msg("tui.ask_user.review.title", fallback="Review your answers")
_REVIEW_UNANSWERED = msg("tui.ask_user.review.unanswered", fallback="(not answered)")
_REVIEW_INCOMPLETE = msg(
    "tui.ask_user.review.incomplete_warning",
    fallback="{count} question is still unanswered.",
    plural_fallback="{count} questions are still unanswered.",
)
_TITLE_NUMBERED = msg(
    "tui.ask_user.title.numbered",
    fallback="Question {index}/{total} · {header}",
)
_TITLE_REVIEW = msg("tui.ask_user.title.review", fallback="Question · Review")
ASK_USER_NOT_ANSWERED_REF = msg("tui.ask_user.answer.not_answered", fallback="(not answered)")

_DISPLAY_HEADER_CELLS = 12
# A header is one tab-label row: whitespace that would break the row displays as a space.
_ROW_BREAKS = str.maketrans("\t\n\r", "   ")
ASK_USER_REVIEW_PANE_ID = "askuser-review"


def ask_user_pane_id(index: int, question_count: int) -> str:
    """Return the pane id for question ``index`` or the review pane."""
    return ASK_USER_REVIEW_PANE_ID if index == question_count else f"askuser-q{index}-pane"


@dataclass(frozen=True, slots=True)
class PromptDraft:
    """Immutable modal-to-inline snapshot of every prompt-owned state field."""

    active: int = 0
    selected: tuple[tuple[int, ...], ...] = ()
    drafts: tuple[str, ...] = ()
    armed_empty_submit: bool = False


class AskUserActiveQuestionChanged(Message):
    """The prompt switched its active question or entered review."""

    def __init__(self, request_id: str, index: int) -> None:
        super().__init__()
        self.request_id = request_id
        self.index = index


def _ellipsize_cells(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    if width == 1:
        return "…"
    result = ""
    for character in value:
        if cell_len(result + character) > width - 1:
            break
        result += character
    return f"{result}…"


class AskUserTabbedContent(TabbedContent, can_focus=False):
    """Question tabs docked above one shared scroll region.

    The tab strip stays fixed while the active pane scrolls beneath it; a
    single-question prompt hides the strip but keeps the same pane structure.
    """

    DEFAULT_CSS = """
    AskUserTabbedContent {
        width: 100%;
        height: auto;
        overflow-y: auto;
        scrollbar-size: 1 1;
        scrollbar-gutter: stable;
        &.-single > ContentTabs {
            display: none;
        }
        ContentTab.-active {
            text-style: bold;
        }
        ContentTab.-answered {
            color: $success;
        }
        ContentTab.-dim {
            opacity: 50%;
        }
    }
    """

    def __init__(self, *, initial: str, single: bool) -> None:
        super().__init__(initial=initial, id="askuser-inner", classes="-single" if single else None)

    def refresh_state(self, labels: tuple[tuple[str, Content], ...], answered: tuple[bool, ...]) -> None:
        """Relabel the tabs and mirror answered state onto their classes."""
        any_answered = any(answered)
        for index, (pane_id, label) in enumerate(labels):
            with suppress(Exception):
                tab = self.get_tab(pane_id)
                if tab.label.plain != label.plain:
                    tab.label = label
                is_review = index == len(answered)
                tab.set_class(any_answered if is_review else answered[index], "-answered")
                tab.set_class(is_review and not any_answered, "-dim")

    def _on_tab_pane_focused(self, event: TabPane.Focused) -> None:
        # The prompt focuses only inside the active pane, so focus-driven
        # activation can only replay a stale focus over a newer switch.
        event.stop()
        event.prevent_default()


class _AskUserPane(TabPane):
    """A pane whose laid-out height changes relayout the inline card."""

    def on_resize(self, event: events.Resize) -> None:
        self.post_message(AskUserContentResized())


class AskUserQuestionPane(_AskUserPane):
    """One permanently mounted question view: Markdown question plus options."""

    DEFAULT_CSS = """
    AskUserQuestionPane {
        width: 100%;
        height: auto;
        VirtualizedMarkdown {
            width: 100%;
            height: auto;
            overflow-y: hidden;
            text-align: left;
            text-style: bold;
            margin: 1 2 0 2;
            padding: 0;
        }
        AskUserOptions {
            margin: 1 1 0 1;
        }
    }
    """

    def __init__(
        self,
        request_id: str,
        index: int,
        question: AskUserQuestion,
        *,
        title: Content,
        initial_selection: tuple[int, ...],
    ) -> None:
        super().__init__(title, id=f"askuser-q{index}-pane", classes="askuser-question-pane")
        self._request_id = request_id
        self._index = index
        self._question = question
        self._initial_selection = initial_selection

    def compose(self) -> ComposeResult:
        from chrys.app.tui.widgets.ask_user_markdown import AskUserQuestionMarkdown

        yield AskUserQuestionMarkdown(
            sanitize_source_text(self._question.question), id=f"askuser-q{self._index}-question"
        )
        if self._question.options:
            yield AskUserOptions(
                self._request_id,
                self._index,
                self._question.options,
                multi_select=self._question.multi_select,
                initial_selection=self._initial_selection,
            )


class AskUserReviewPane(_AskUserPane):
    """Shared answer-region review list and incomplete warning."""

    COMPONENT_CLASSES: ClassVar[set[str]] = {"askuser-review--unanswered", "askuser-review--answer"}

    DEFAULT_CSS = """
    AskUserReviewPane {
        width: 100%;
        height: auto;
        margin: 1 2 0 2;
        #askuser-review-list, #askuser-review-warning {
            width: 100%;
            height: auto;
        }
        #askuser-review-warning {
            color: $warning;
            margin-top: 1;
        }
        /* An unanswered question is flagged in the same colour as the
           incomplete warning below the list. */
        & > .askuser-review--unanswered {
            color: $warning;
        }
        /* Answers are muted like the completed tool card's answer block, so
           the questions stay the eye-catching part of the summary. */
        & > .askuser-review--answer {
            color: $text-muted;
        }
    }
    """

    def __init__(self, *, title: Content) -> None:
        super().__init__(title, id=ASK_USER_REVIEW_PANE_ID, classes="askuser-review-pane")

    def compose(self) -> ComposeResult:
        yield Static("", id="askuser-review-list")
        yield Static("", id="askuser-review-warning")

    def update_review(
        self,
        questions: tuple[AskUserQuestion, ...],
        answers: tuple[AskUserAnswer, ...],
    ) -> None:
        localizer = widget_localizer(self)
        unanswered_style = self.get_component_rich_style("askuser-review--unanswered", partial=True)
        # ``$text-muted`` is "auto 60%": the partial style drops that alpha and
        # comes back pure white, so take the colour Textual already blended
        # over the pane background (the same blend the tool card's CSS gets).
        answer_style = Style(color=self.get_component_rich_style("askuser-review--answer").color)
        unanswered_label = render_str(localizer, _REVIEW_UNANSWERED.bind())
        # Questions and answers sit in a gutter/body grid so a soft-wrapped
        # line continues under its own first character, never at the margin.
        rows: list[tuple[str, RenderableType]] = []
        for question, answer in zip(questions, answers, strict=True):
            question_text = sanitize_source_text(question.question)
            rows.append((" • ", Text(question_text, style="" if answer.answered else unanswered_style)))
            rows.append(("", ask_user_hanging_answer(answer, unanswered=unanswered_label, style=answer_style)))
        review = Group(Text(render_str(localizer, ASK_USER_REVIEW_TITLE_REF.bind())), ask_user_hanging_grid(rows))
        unanswered = sum(not answer.answered for answer in answers)
        warning = Text()
        if unanswered:
            warning.append("⚠ ")
            warning.append(render_str(localizer, _REVIEW_INCOMPLETE.bind(count=unanswered)))
        self.query_one("#askuser-review-list", Static).update(review)
        warning_widget = self.query_one("#askuser-review-warning", Static)
        warning_widget.update(warning)
        warning_widget.display = bool(unanswered)


class AskUserPrompt(VerticalGroup):
    """Single owner of selections, drafts, active pane and submission state."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+pagedown", "next_question", show=False, priority=True),
        Binding("ctrl+pageup", "previous_question", show=False, priority=True),
    ]

    def __init__(
        self,
        request_id: str,
        questions: tuple[AskUserQuestion, ...],
        *,
        inline: bool,
        allow_inline: bool,
        draft: PromptDraft | None = None,
    ) -> None:
        super().__init__(id="ask-inline" if inline else "askuser-prompt")
        self.request_id = request_id
        self.questions = questions
        self.inline = inline
        self.allow_inline = allow_inline
        state = draft or PromptDraft()
        self._selected = [
            list(state.selected[index]) if index < len(state.selected) else [] for index in range(len(questions))
        ]
        for index, question in enumerate(questions):
            self._selected[index] = [
                option_index for option_index in self._selected[index] if 0 <= option_index < len(question.options)
            ]
        self._drafts = [state.drafts[index] if index < len(state.drafts) else "" for index in range(len(questions))]
        self._active = min(max(0, state.active), len(questions))
        self._armed_empty_submit = state.armed_empty_submit
        self._submitted = False
        self._generation = 0
        self._focus_pending = False
        self._pane_index = {ask_user_pane_id(index, len(questions)): index for index in range(len(questions) + 1)}

    def compose(self) -> ComposeResult:
        answered = tuple(answer.answered for answer in self._composed_answers())
        labels = self._tab_labels(answered)
        with AskUserTabbedContent(
            initial=ask_user_pane_id(self._active, len(self.questions)),
            single=len(self.questions) == 1,
        ):
            for index, question in enumerate(self.questions):
                yield AskUserQuestionPane(
                    self.request_id,
                    index,
                    question,
                    title=labels[index][1],
                    initial_selection=tuple(self._selected[index]),
                )
            yield AskUserReviewPane(title=labels[-1][1])
        initial = self._drafts[self._active] if self._active < len(self.questions) else ""
        yield AskUserResponseFooter(
            self.request_id,
            allow_inline=self.allow_inline,
            initial_response=initial,
            defer_layout_to_parent=self.inline,
        )

    def on_mount(self) -> None:
        self._apply_active(initial=True)

    @property
    def active_index(self) -> int:
        return self._active

    @property
    def generation(self) -> int:
        return self._generation

    def snapshot(self) -> PromptDraft:
        """Capture the exact prompt state for modal-to-inline handoff."""
        self._capture_active_draft()
        return PromptDraft(
            active=self._active,
            selected=tuple(tuple(selection) for selection in self._selected),
            drafts=tuple(self._drafts),
            armed_empty_submit=self._armed_empty_submit,
        )

    def answers(self) -> tuple[AskUserAnswer, ...]:
        """Compose fixed-length answers from prompt-owned selection and draft state."""
        self._capture_active_draft()
        return self._composed_answers()

    def _composed_answers(self) -> tuple[AskUserAnswer, ...]:
        answers: list[AskUserAnswer] = []
        for index, question in enumerate(self.questions):
            values = tuple(question.options[item].label for item in self._selected[index])
            draft = self._drafts[index].strip()
            if values:
                answers.append(AskUserAnswer(values=values, note=draft))
            elif draft:
                answers.append(AskUserAnswer(values=(draft,)))
            else:
                answers.append(AskUserAnswer())
        return tuple(answers)

    def disable_controls(self) -> None:
        for options in self.query(AskUserOptions):
            options.disable_controls()
        self.query_one(AskUserResponseFooter).disable_controls()

    def _capture_active_draft(self) -> None:
        if self._active >= len(self.questions):
            return
        with suppress(Exception):
            self._drafts[self._active] = self.query_one(AskUserResponseFooter).draft_text

    def _footer_position(self) -> FooterPosition:
        if self._active == len(self.questions):
            return "review"
        if len(self.questions) == 1:
            return "single"
        if self._active == len(self.questions) - 1:
            return "last"
        return "next"

    def _display_header(self, index: int) -> str:
        localizer = widget_localizer(self)
        header = self.questions[index].header or render_str(localizer, _TAB_FALLBACK_HEADER.bind(n=index + 1))
        return _ellipsize_cells(sanitize_legacy_scalar(header.translate(_ROW_BREAKS)), _DISPLAY_HEADER_CELLS)

    def _tab_labels(self, answered: tuple[bool, ...]) -> tuple[tuple[str, Content], ...]:
        labels: list[tuple[str, Content]] = []
        for index in range(len(self.questions)):
            prefix = "☑ " if answered[index] else "☐ "
            labels.append(
                (
                    ask_user_pane_id(index, len(self.questions)),
                    Content.from_text(f"{prefix}{self._display_header(index)}", markup=False),
                )
            )
        submit = render_str(widget_localizer(self), _TAB_SUBMIT.bind())
        labels.append((ASK_USER_REVIEW_PANE_ID, Content.from_text(f"✓ {submit}", markup=False)))
        return tuple(labels)

    def _refresh_views(self) -> None:
        answers = self._composed_answers()
        answered = tuple(answer.answered for answer in answers)
        with suppress(Exception):
            self.query_one(AskUserTabbedContent).refresh_state(self._tab_labels(answered), answered)
        self.query_one(AskUserReviewPane).update_review(self.questions, answers)
        if self._active < len(self.questions):
            with suppress(Exception):
                self.query_one(f"#askuser-q{self._active}-options", AskUserOptions).set_selection(
                    tuple(self._selected[self._active])
                )
        footer = self.query_one(AskUserResponseFooter)
        footer.set_context(
            self._active,
            self._drafts[self._active] if self._active < len(self.questions) else "",
            has_selection=self._active < len(self.questions) and bool(self._selected[self._active]),
            position=self._footer_position(),
            generation=self._generation,
            answered_count=sum(answered),
            question_count=len(self.questions),
            armed_empty_submit=self._armed_empty_submit,
        )

    def _apply_active(self, *, initial: bool = False) -> None:
        # The active watcher echoes a TabActivated for programmatic switches;
        # handling that echo later would replay a stale switch over a newer one.
        with suppress(Exception), self.prevent(TabbedContent.TabActivated):
            self.query_one(AskUserTabbedContent).active = ask_user_pane_id(self._active, len(self.questions))
        self._refresh_views()
        if not initial:
            self.post_message(AskUserActiveQuestionChanged(self.request_id, self._active))
        # One refresh may cover several switches. Focus the latest context
        # once, so older requests cannot reclaim focus after user navigation.
        if not self._focus_pending:
            self._focus_pending = self.call_after_refresh(self._focus_active)

    def _switch(self, index: int) -> None:
        if not 0 <= index <= len(self.questions) or index == self._active:
            return
        self._capture_active_draft()
        self._active = index
        self._armed_empty_submit = False
        self._generation += 1
        self._apply_active()

    def _focus_active(self) -> None:
        self._focus_pending = False
        # A workflow can finish and dismiss its question before this frame callback runs.
        if not self.is_attached or self.screen is not self.app.screen:
            return
        footer = self.query_one(AskUserResponseFooter)
        if self._active == len(self.questions):
            footer.focus_submit()
            return
        selection = self._selected[self._active]
        option_index = selection[0] if selection else 0
        if self.questions[self._active].options:
            with suppress(Exception):
                self.query_one(f"#askuser-q{self._active}-options", AskUserOptions).focus_option(option_index)
                return
        footer.focus_input()

    def _next_unanswered(self, current: int) -> int:
        answers = self.answers()
        for offset in range(1, len(self.questions) + 1):
            index = (current + offset) % len(self.questions)
            if not answers[index].answered:
                return index
        return len(self.questions)

    def _submit(self) -> None:
        if self._submitted:
            return
        answers = self.answers()
        if not any(answer.answered for answer in answers) and not self._armed_empty_submit:
            self._armed_empty_submit = True
            self._generation += 1
            self._refresh_views()
            return
        self._submitted = True
        self.disable_controls()
        self.post_message(AskUserSubmitted(self.request_id, answers))

    @on(TabbedContent.TabActivated)
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tabbed_content is not self.query_one(AskUserTabbedContent):
            return
        event.stop()
        index = self._pane_index.get(event.pane.id or "")
        if index is not None:
            self._switch(index)

    @on(AskUserOptionChosen)
    def _on_option_chosen(self, event: AskUserOptionChosen) -> None:
        if event.request_id != self.request_id or self._submitted:
            return
        event.stop()
        selection = self._selected[event.question_index]
        question = self.questions[event.question_index]
        if event.option_index in selection:
            selection.remove(event.option_index)
        elif question.multi_select:
            selection.append(event.option_index)
        else:
            selection[:] = [event.option_index]
        self._armed_empty_submit = False
        self._generation += 1
        self._refresh_views()
        if not question.multi_select and selection:
            if len(self.questions) == 1:
                self._submit()
            else:
                self._switch(self._next_unanswered(event.question_index))

    @on(AskUserDraftChanged)
    def _on_draft_changed(self, event: AskUserDraftChanged) -> None:
        if event.request_id != self.request_id or not 0 <= event.question_index < len(self.questions):
            return
        event.stop()
        self._drafts[event.question_index] = event.text
        if self._armed_empty_submit:
            self._armed_empty_submit = False
        self._generation += 1
        self._refresh_views()

    @on(AskUserFooterAction)
    def _on_footer_action(self, event: AskUserFooterAction) -> None:
        if event.request_id != self.request_id or self._submitted:
            return
        event.stop()
        self._capture_active_draft()
        if event.inline:
            self.disable_controls()
            self.post_message(AskUserInlineRequested(self.request_id, self.snapshot()))
            return
        if self._active == len(self.questions) or len(self.questions) == 1:
            self._submit()
            return
        if self.answers()[self._active].answered:
            self._switch(self._next_unanswered(self._active))

    def action_next_question(self) -> None:
        self._switch((self._active + 1) % (len(self.questions) + 1))

    def action_previous_question(self) -> None:
        self._switch((self._active - 1) % (len(self.questions) + 1))

    def active_title(self) -> Text:
        """Render the localized, markup-safe title for the active pane."""
        localizer = widget_localizer(self)
        if self._active == len(self.questions):
            return Text(render_str(localizer, _TITLE_REVIEW.bind()))
        if len(self.questions) == 1:
            return Text(render_str(localizer, _TITLE_SINGLE.bind()))
        return Text(
            render_str(
                localizer,
                _TITLE_NUMBERED.bind(
                    index=self._active + 1,
                    total=len(self.questions),
                    header=self._display_header(self._active),
                ),
            )
        )


__all__ = [
    "AskUserActiveQuestionChanged",
    "AskUserPrompt",
    "AskUserQuestionPane",
    "AskUserReviewPane",
    "AskUserTabbedContent",
    "PromptDraft",
    "ask_user_pane_id",
]
