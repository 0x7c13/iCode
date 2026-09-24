# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Event contract tests."""

from chrys.foundation.events.types import AgentLoadFailed, AgentLoadProgress, Error, InvocationRetryAttempt, Warning
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.models.invocations import InvocationOrigin

_BOUND_EVENT_MESSAGE = msg("test_types.bound_event_message", fallback="Bound {value}")


def test_events_with_display_messages_default_to_none() -> None:
    assert Warning().display_message is None
    assert Error().display_message is None
    assert InvocationRetryAttempt(origin=InvocationOrigin("turn", "", "turn-test", None)).display_message is None
    assert AgentLoadFailed().display_message is None


def test_events_accept_bound_display_messages() -> None:
    display_message = _BOUND_EVENT_MESSAGE.bind(value="detail")

    warning = Warning(display_message=display_message)
    error = Error(display_message=display_message)
    retry = InvocationRetryAttempt(
        display_message=display_message, origin=InvocationOrigin("turn", "", "turn-test", None)
    )
    load_failed = AgentLoadFailed(display_message=display_message)

    assert isinstance(warning.display_message, MessageRef)
    assert warning.display_message is display_message
    assert isinstance(error.display_message, MessageRef)
    assert error.display_message is display_message
    assert isinstance(retry.display_message, MessageRef)
    assert retry.display_message is display_message
    assert isinstance(load_failed.display_message, MessageRef)
    assert load_failed.display_message is display_message


def test_agent_load_progress_semantic_fields_default_to_empty_strings() -> None:
    progress = AgentLoadProgress()

    assert progress.status == ""
    assert progress.subject == ""
    assert progress.detail == ""


def test_warning_and_error_keep_legacy_fields_with_keyword_construction() -> None:
    warning = Warning(code="warning_code", message="warning message")
    error = Error(code="error_code", message="error message", recoverable=False)

    assert (warning.code, warning.message) == ("warning_code", "warning message")
    assert (error.code, error.message, error.recoverable) == ("error_code", "error message", False)


def test_warning_and_error_remain_constructible_without_arguments() -> None:
    assert isinstance(Warning(), Warning)
    assert isinstance(Error(), Error)
