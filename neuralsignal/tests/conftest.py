from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_v2_foundation_cwd(request, monkeypatch, tmp_path):
    if request.path.name == "test_v2_foundation.py":
        monkeypatch.chdir(tmp_path)

