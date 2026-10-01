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
        if path.is_dir() and (path / ".git").exists() and all(p.name == ".git" for p in path.iterdir()):
            # Cloned, but writing out its files failed or was cut short. The submodule is already at
            # its commit, so a plain update does nothing; --force writes the files (or shows why not).
            rel = path.relative_to(REPO_ROOT).as_posix()
            return (f"{what}: the submodule at {path} was cloned but its files were never checked out; "
                    f"run `git submodule update --init --force {rel}`")
        return f"{what} not found at {path}; run `git submodule update --init --recursive` (or set {env})"
    return f"{what} not found at {path} (check {env} / --cv-dir)"
