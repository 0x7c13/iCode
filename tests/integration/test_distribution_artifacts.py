# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline distribution artifact guards."""

from __future__ import annotations

import gettext
import io
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repository = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("wheel-build") / "artifacts"
    completed = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert completed.returncode == 0, completed.stderr
    wheels = sorted(output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_contains_a_loadable_simplified_chinese_catalog(built_wheel: Path) -> None:
    catalog_member = "chrys/foundation/i18n/_catalogs/zh-Hans/LC_MESSAGES/chrys.mo"
    with ZipFile(built_wheel) as wheel:
        assert catalog_member in wheel.namelist()
        catalog_bytes = wheel.read(catalog_member)

    translations = gettext.GNUTranslations(io.BytesIO(catalog_bytes))
    assert translations.gettext("missing.key") == "missing.key"
    assert translations.info()["plural-forms"] == "nplurals=1; plural=0;"
