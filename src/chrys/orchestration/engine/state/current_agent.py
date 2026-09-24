# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The current build resources and the last installed configuration."""

from __future__ import annotations

from chrys.orchestration.engine.build.loaded import AgentManifest, LoadedAgent


class CurrentAgent:
    """Keep resource lifetime separate from manifest lifetime."""

    __slots__ = ("loaded", "manifest")

    loaded: LoadedAgent | None
    manifest: AgentManifest

    def __init__(self) -> None:
        self.loaded = None
        self.manifest = AgentManifest.empty()
