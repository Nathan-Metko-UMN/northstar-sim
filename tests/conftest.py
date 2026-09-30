import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def northstar_cv_dir() -> Path:
    """Northstar-CV checkout; defaults to a sibling of this repo."""
    return Path(os.environ.get("NORTHSTAR_CV_DIR", REPO_ROOT.parent / "Northstar-CV"))


@pytest.fixture
def cv_dir() -> Path:
    path = northstar_cv_dir()
    if not (path / "src" / "uart").is_dir():
        pytest.skip(f"Northstar-CV not found at {path} (set NORTHSTAR_CV_DIR)")
    return path
