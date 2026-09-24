# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""GLM (Zhipu AI / z.ai) chat-completion client.

GLM's API is OpenAI-compatible at the wire level and speaks the same
``reasoning_content`` dialect as DeepSeek, so the base
``RawOpenAIChatCompletionClient`` already captures and replays its
reasoning; this subclass only encodes GLM endpoint policy:

- ``TOKEN_LIMIT_PARAM``: GLM's Chat Completions API documents only the
  OpenAI-legacy ``max_tokens`` output cap, not ``max_completion_tokens``.
- Reasoning replay keeps the base's unconditional policy: z.ai's
  thinking-mode contract (preserved thinking) requires historical
  ``reasoning_content`` to be replayed on every multi-turn request,
  including turns without tool calls — see
  https://docs.z.ai/guides/llm/glm-4.7 (interleaved/preserved thinking)
  and https://docs.bigmodel.cn/api-reference (Chat Completions schema).
"""

from __future__ import annotations

from typing import ClassVar

from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient

GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class GLMChatCompletionClient(RawOpenAIChatCompletionClient):
    """OpenAI-compatible Chat Completions client for GLM thinking mode.

    Subclasses the loop-free ``Raw`` wire client: the tool loop and
    chat middleware are chrys-owned layers composed around it by
    ``chrys.service.llm.instrumented``.  Everything else — reasoning
    capture/replay, message assembly, transport — is inherited from the
    base client unchanged.
    """

    TOKEN_LIMIT_PARAM: ClassVar[str] = "max_tokens"
