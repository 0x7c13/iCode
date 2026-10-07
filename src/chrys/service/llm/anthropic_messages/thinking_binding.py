# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""How one request binds its thinking to the conversation, and which betas that needs.

Claude Opus 5.5, Fable 5.1 and Sonnet 5.5 bind each signed thinking block to
the request prefix before it: the system prompt, the tool set and every
earlier message. Compaction, a tool set change or a fork rewrites that prefix,
and an account that enforces the binding then rejects the replayed blocks
unless ``thinking.block_binding.prefix_mismatch_behavior`` says what to do
with them: ``drop_block`` drops the first block that no longer matches and
the thinking after it, ``error`` rejects the request.

:func:`resolve_thinking_binding` reads the request's final thinking and model
(an ``extra_body`` key wins, even a null) and the profile's settings, and
returns the request's :class:`ThinkingBindingPolicy`:

- ``auto`` (the default) adds ``drop_block`` on Anthropic's own endpoint when
  one of those models runs adaptive thinking;
- ``drop_block`` and ``error`` are added to enabled or adaptive thinking on
  any endpoint and model;
- ``off`` adds nothing.

A ``block_binding`` the options write themselves is sent as written. Any
mapping ``block_binding`` the request sends needs the binding-controls beta.
Budgeted (``enabled``) thinking on Anthropic's own endpoint also gets the
interleaved-thinking beta, so the model keeps thinking between tool calls,
unless the profile turns that off or the model has no interleaved thinking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from chrys.foundation.errors.route import origin_of
from chrys.service.llm.providers import PROVIDERS
from chrys.service.profiles.models.options import AUTO_INTERLEAVED_THINKING_OPTION, THINKING_BLOCK_BINDING_OPTION

BINDING_MODELS: Final = frozenset({"claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5"})
"""Models whose replayed thinking ``auto`` binds with ``drop_block`` (exact ids)."""

NO_AUTO_INTERLEAVED_MODELS: Final = frozenset({"claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-opus-4-6"})
"""Models that take budgeted thinking but do not interleave it (exact ids)."""

BINDING_CONTROLS_BETA: Final = "thinking-binding-controls-2026-08-01"
INTERLEAVED_THINKING_BETA: Final = "interleaved-thinking-2025-05-14"

_ANTHROPIC_ORIGIN: Final = origin_of(PROVIDERS["anthropic"].default_base_url)
_EXPLICIT_BINDINGS: Final = ("drop_block", "error")


@dataclass(frozen=True, slots=True)
class ThinkingBindingPolicy:
    """What one request sends about its thinking."""

    thinking: Mapping[str, Any] | None
    """The thinking to send in place of the request's own, None to send it unchanged."""
    thinking_in_extra_body: bool
    """The request's final thinking is the one in ``extra_body``."""
    mismatch_behavior: object
    """The prefix mismatch behavior in effect.

    The sent ``block_binding``'s choice, else an explicit ``drop_block`` or
    ``error`` setting even when the request sends no ``block_binding``; None
    when neither applies.
    """
    mismatch_behavior_sent: bool
    """The sent ``block_binding`` names :attr:`mismatch_behavior` itself."""
    controls_beta: bool
    """The request sends a ``block_binding``, which needs the binding-controls beta."""
    block_binding_written: bool
    """The request's own final thinking holds a ``block_binding``, sent as written: the setting adds none."""
    interleaved_beta: bool
    """The request gets the interleaved-thinking beta."""
    binding_model: bool
    """The final model binds its thinking to the conversation."""
    thinking_type: object
    """The ``type`` of the request's final thinking; None when that is no mapping or has no ``type``."""


def resolve_thinking_binding(
    request: Mapping[str, Any], options: Mapping[str, Any], *, base_url: object
) -> ThinkingBindingPolicy:
    """The policy of *request*, built from *options*, going to *base_url*.

    *request* holds the request fields with the call keywords merged in;
    *options* holds the profile's settings. Neither is changed.
    """
    thinking, thinking_in_extra_body = _final_field(request, "thinking")
    model, _ = _final_field(request, "model")
    setting = options.get(THINKING_BLOCK_BINDING_OPTION)
    default_endpoint = origin_of(base_url) == _ANTHROPIC_ORIGIN
    binding_model = isinstance(model, str) and model in BINDING_MODELS
    thinking_type = thinking.get("type") if isinstance(thinking, Mapping) else None

    added: str | None = None
    if isinstance(thinking, Mapping) and "block_binding" not in thinking:
        if setting in _EXPLICIT_BINDINGS and thinking_type in ("enabled", "adaptive"):
            added = setting
        elif (
            setting not in (*_EXPLICIT_BINDINGS, "off")
            and binding_model
            and default_endpoint
            and thinking_type == "adaptive"
        ):
            added = "drop_block"
    new_thinking = (
        {**thinking, "block_binding": {"prefix_mismatch_behavior": added}}
        if added is not None and isinstance(thinking, Mapping)
        else None
    )

    final_thinking = new_thinking if new_thinking is not None else thinking
    block_binding = final_thinking.get("block_binding") if isinstance(final_thinking, Mapping) else None
    mismatch_behavior: object = None
    mismatch_behavior_sent = isinstance(block_binding, Mapping) and "prefix_mismatch_behavior" in block_binding
    if mismatch_behavior_sent:
        mismatch_behavior = block_binding["prefix_mismatch_behavior"]
    elif setting in _EXPLICIT_BINDINGS:
        mismatch_behavior = setting

    return ThinkingBindingPolicy(
        thinking=new_thinking,
        thinking_in_extra_body=thinking_in_extra_body,
        mismatch_behavior=mismatch_behavior,
        mismatch_behavior_sent=mismatch_behavior_sent,
        controls_beta=isinstance(block_binding, Mapping),
        block_binding_written=isinstance(thinking, Mapping) and "block_binding" in thinking,
        interleaved_beta=(
            options.get(AUTO_INTERLEAVED_THINKING_OPTION) is not False
            and default_endpoint
            and thinking_type == "enabled"
            and not (isinstance(model, str) and model in NO_AUTO_INTERLEAVED_MODELS)
        ),
        binding_model=binding_model,
        thinking_type=thinking_type,
    )


def apply_thinking_binding(request: dict[str, Any], policy: ThinkingBindingPolicy) -> None:
    """Write *policy*'s thinking where *request* keeps its final thinking, copying ``extra_body``."""
    if policy.thinking is None:
        return
    if policy.thinking_in_extra_body:
        request["extra_body"] = {**request["extra_body"], "thinking": policy.thinking}
    else:
        request["thinking"] = policy.thinking


def _final_field(request: Mapping[str, Any], name: str) -> tuple[Any, bool]:
    """The value the service gets for *name*, and whether it is the ``extra_body`` one."""
    extra_body = request.get("extra_body")
    if isinstance(extra_body, Mapping) and name in extra_body:
        return extra_body[name], True
    return request.get(name), False
