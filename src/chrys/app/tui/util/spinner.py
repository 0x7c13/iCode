# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The braille spinner every running indicator shares.

Tool cards, the auto-review label, workflow nodes, the input bar and the
terminal tab all draw these frames.
"""

from __future__ import annotations

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# The workflow graph and the terminal tab show this instead when animations are off.
STATIC_SPINNER = "●"
