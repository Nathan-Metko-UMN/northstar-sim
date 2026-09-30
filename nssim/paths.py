"""Where the harness finds Northstar-CV and the TR simulator.

Both are git submodules under ``external/``. Point ``NORTHSTAR_CV_DIR`` (or ``--cv-dir``) at another
Northstar-CV checkout to test your working copy instead of the pinned one; ``TR_SIM_DIR`` does the
same for the TR assets.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = REPO_ROOT / "external"


def cv_dir(override: str | os.PathLike | None = None) -> Path:
    return Path(override or os.environ.get("NORTHSTAR_CV_DIR") or EXTERNAL / "Northstar-CV")


def tr_dir(override: str | os.PathLike | None = None) -> Path:
    return Path(override or os.environ.get("TR_SIM_DIR") or EXTERNAL / "TR-Simulation-ARCTIC-2026")


def missing_hint(path: Path, what: str, env: str) -> str:
    if path.parent == EXTERNAL:
        return f"{what} not found at {path}; run `git submodule update --init --recursive` (or set {env})"
    return f"{what} not found at {path} (check {env} / --cv-dir)"
