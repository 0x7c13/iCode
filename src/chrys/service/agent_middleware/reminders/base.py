# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What the reminder middleware asks of its content sources."""

from __future__ import annotations

from typing import Protocol


class CatalogSource(Protocol):
    """Standing context, sent again whenever the latest version in view differs."""

    @property
    def name(self) -> str:
        """The catalog's record name; persisted, so it never changes."""
        ...

    @property
    def withdrawn(self) -> str | None:
        """What replaces the catalog once it is no longer offered; None leaves its last version standing."""
        ...

    def snapshot(self) -> str | None:
        """The catalog's current text, or None when it offers none."""
        ...
