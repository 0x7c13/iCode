# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared helper for the tool argument guidance tests: validate, capture, format."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from chrys.kernel._tool_arg_errors import _argument_validation_message


def _message_for(
    model: type[BaseModel],
    arguments: Mapping[str, Any],
    *,
    tool_name: str,
    **overrides: Any,
) -> str:
    """Validate ``arguments`` against ``model`` and format the failure the way the loop does.

    The common defaults (``arguments_unparseable=False``, ``reject_unexpected=True`` and the
    model's own JSON schema) can be replaced through ``overrides``. ``input_model`` is only
    forwarded when a caller passes it, so every test keeps exercising exactly the argument
    set it passed to the formatter before this helper existed.
    """
    with pytest.raises(ValidationError) as caught:
        model.model_validate(arguments)
    kwargs: dict[str, Any] = {
        "arguments_unparseable": False,
        "schema": model.model_json_schema(),
        "reject_unexpected": True,
    }
    kwargs.update(overrides)
    return _argument_validation_message(
        tool_name=tool_name,
        arguments=arguments,
        exception=caught.value,
        **kwargs,
    )
