# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrys — A general-purpose extensible agent platform for rapidly building and evolving custom agents.

Chrys gives agents the ability to perceive their own context window usage
and autonomously compress (fold) completed work into concise summaries, freeing
up token space for new tasks. Agents are configured through profiles — customizable
combinations of prompts, tools, skills, and hooks.
"""

from importlib.metadata import version as _version

DISTRIBUTION_NAME = "iCode-TUI"
"""The name the package is published and installed under; the import package stays ``chrys``."""

__version__ = _version(DISTRIBUTION_NAME)
