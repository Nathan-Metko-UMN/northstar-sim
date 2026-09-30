"""Builds Northstar-CV's simulator binary in Docker, the local stand-in for the Jetson.

Two images: ``northstar-cv:dev`` from Northstar-CV's own ``.devcontainer/Dockerfile``, and
``northstar-cv:sim`` on top of it (adds socat, see ``docker/cv-sim.Dockerfile``). The binary is
built inside the image into ``<Northstar-CV>/build/sim-x86``, which ``nssim run`` mounts and starts.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from nssim.paths import REPO_ROOT

DEV_IMAGE = "northstar-cv:dev"
SIM_IMAGE = "northstar-cv:sim"
BUILD_DIR = "build/sim-x86"


def image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return result.returncode == 0


def host_cuda_arch() -> str | None:
    """Compute capability of this machine's first GPU as CMake wants it ("89" for an RTX 4060 Ti)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    out = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    lines = out.stdout.split() if out.returncode == 0 else []
    return lines[0].replace(".", "") if lines else None


def binary_path(cv_dir: Path, build_dir: str = BUILD_DIR) -> Path:
    return Path(cv_dir) / build_dir / "NorthstarCV2"


def _run(cmd: list[str]) -> None:
    print("[nssim] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def missing_submodules(cv_dir: Path) -> list[str]:
    """Northstar-CV's own submodules (the particle filter library) that aren't checked out."""
    status = subprocess.run(
        ["git", "-C", str(cv_dir), "submodule", "status", "--recursive"], capture_output=True, text=True
    )
    return [line.split()[1] for line in status.stdout.splitlines() if line.startswith("-")]


def build(cv_dir: Path, cuda_arch: str | None = None, rebuild_images: bool = False, build_dir: str = BUILD_DIR) -> Path:
    cv_dir = Path(cv_dir).resolve()
    missing = missing_submodules(cv_dir)
    if missing:
        raise SystemExit(
            f"Northstar-CV's submodules aren't checked out ({', '.join(missing)}); "
            "run `git submodule update --init --recursive`"
        )
    if rebuild_images or not image_exists(DEV_IMAGE):
        # The dev Dockerfile copies nothing in, so its own folder is enough context; the repo root
        # would upload build/ and .ccache/ to the daemon for nothing.
        devcontainer = cv_dir / ".devcontainer"
        _run(["docker", "build", "-t", DEV_IMAGE, "-f", str(devcontainer / "Dockerfile"), str(devcontainer)])
    if rebuild_images or not image_exists(SIM_IMAGE):
        docker_dir = REPO_ROOT / "docker"
        _run(["docker", "build", "-t", SIM_IMAGE, "--build-arg", f"BASE={DEV_IMAGE}",
              "-f", str(docker_dir / "cv-sim.Dockerfile"), str(docker_dir)])

    arch = cuda_arch or host_cuda_arch()
    if arch is None:
        raise SystemExit("couldn't read the GPU's compute capability from nvidia-smi; pass --cuda-arch (e.g. 89)")
    script = (
        f"cmake -S . -B {build_dir} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES={arch}"
        f" && cmake --build {build_dir}"
    )
    _run(["docker", "run", "--rm", "-v", f"{cv_dir}:/ws", "-w", "/ws", SIM_IMAGE, "bash", "-lc", script])
    return binary_path(cv_dir, build_dir)
