# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Build-scoped web secrets from the pre-bootstrap process and user dotenv.

Unlike legacy SDK credential interpolation this excludes project dotenv and
live environment mutations after bootstrap. Destination grants are checked by
the caller before requesting any values here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from chrys.foundation.config.env_file import config_env_path
from chrys.foundation.config.env_layers import canonical_env_name, process_env_snapshot, read_dotenv_layer


@dataclass(frozen=True)
class ResolvedSearchCredentials:
    """Secrets excluded from repr and never passed to profile serialization."""

    values: dict[str, str] = field(repr=False)


def resolve_search_credentials(names: set[str]) -> ResolvedSearchCredentials:
    snapshot = process_env_snapshot()
    process = {canonical_env_name(k): v for k, v in (snapshot.values if snapshot else os.environ).items()}
    user = {canonical_env_name(k): v for k, v in read_dotenv_layer(config_env_path(), base=process).items()}
    result: dict[str, str] = {}
    for name in names:
        key = canonical_env_name(name)
        value = process[key] if key in process else user.get(key, "")
        if not value:
            raise ValueError(f"Missing web search credential {name}")
        result[name] = value
    return ResolvedSearchCredentials(result)
