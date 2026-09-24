# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Evaluation context: what the loader needs to know that is not a setting.

Some project-layer checks cannot be decided from the stored value alone. The
transient-retry budget is the case that forces this: a stored ``None`` means
"use the frontend's policy default", which is 7 in the TUI and 15 in headless,
so the *same* project value of 10 is a loosening for one frontend and a
tightening for the other. Comparing raw fields would silently apply the wrong
verdict in headless.

This lives in its own module so both :mod:`chrys.foundation.config.spec` (which
types the comparator) and the loader can import it without a cycle through
``settings``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Frontend policy in force for one ``load_settings`` call.

    Must be passed explicitly at load time, not patched in afterwards: the
    project layer is evaluated *during* the load, so a value substituted later
    would arrive after the verdict it was meant to inform. The initial load,
    ``SettingsReload``, a workspace change and a session restore all have to
    pass the same context, or one project file yields different verdicts at
    different points in a single process's life.
    """

    frontend_default_max_transient_retries: int
