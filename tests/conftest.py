from pathlib import Path

import pytest

from nssim import paths


@pytest.fixture
def cv_dir() -> Path:
    path = paths.cv_dir()
    if not (path / "src" / "uart").is_dir():
        pytest.skip(paths.missing_hint(path, "Northstar-CV", "NORTHSTAR_CV_DIR"))
    return path
