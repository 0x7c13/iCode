# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The shipped workflow templates and their pre-generated manifests; ``python -m`` regenerates the manifests."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

from chrys.service.workflows.discovery import BUILTIN_DIR, builtin_manifest_path


def builtin_templates() -> list[Path]:
    """Every template discovery would list, in id order."""
    return sorted(path for path in BUILTIN_DIR.glob("*.py") if not path.name.startswith("_"))


def manifest_path(template: Path) -> Path:
    return builtin_manifest_path(template.stem)


def builtin_manifest(template: Path) -> dict[str, Any]:
    """The manifest *template* builds when executed here; the SDK facade stands in for the injected SDK."""
    namespace = runpy.run_path(str(template), run_name="chrys_workflow_template")
    return namespace["workflow"].manifest()


def regenerate() -> list[Path]:
    """Rewrite every template's manifest file from the template itself."""
    written: list[Path] = []
    for template in builtin_templates():
        path = manifest_path(template)
        path.write_text(json.dumps(builtin_manifest(template), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for written_path in regenerate():
        sys.stdout.write(f"wrote {written_path}\n")
