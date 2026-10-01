"""Builds Northstar-CV's simulator binary in Docker, the local stand-in for the Jetson.

There is a toolchain per JetPack, since the robot may run either:

- JetPack 7 (CUDA 13): ``northstar-cv:dev`` from Northstar-CV's own ``.devcontainer/Dockerfile``,
  and ``northstar-cv:sim`` on top of it (adds socat, see ``docker/cv-sim.Dockerfile``). Builds into
  ``build/sim-x86``.
- JetPack 6 (CUDA 12.6, Ubuntu 22.04): ``northstar-cv:jp6`` from ``docker/cv-jp6.Dockerfile``.
  Builds into ``build/sim-jp6``.

Both compile for this machine's GPU (to run in the sim) and the Orin's sm_87 (so the Jetson's code
is checked too). ``nssim run`` mounts the build and starts it.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nssim.paths import REPO_ROOT

DEV_IMAGE = "northstar-cv:dev"
ORIN_ARCH = "87"


@dataclass(frozen=True)
class Toolchain:
    image: str  # what the binary is built and run in
    build_dir: str  # inside the Northstar-CV checkout
    what: str


TOOLCHAINS = {
    7: Toolchain("northstar-cv:sim", "build/sim-x86", "CUDA 13 (Northstar-CV's dev image), as on JetPack 7"),
    6: Toolchain("northstar-cv:jp6", "build/sim-jp6", "CUDA 12.6 on Ubuntu 22.04, as on JetPack 6"),
}
BUILD_DIR = TOOLCHAINS[7].build_dir


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


def _ensure_images(jetpack: int, cv_dir: Path, rebuild: bool) -> None:
    docker_dir = REPO_ROOT / "docker"
    if jetpack == 6:
        if rebuild or not image_exists(TOOLCHAINS[6].image):
            _run(["docker", "build", "-t", TOOLCHAINS[6].image, "-f", str(docker_dir / "cv-jp6.Dockerfile"), str(docker_dir)])
        return
    if rebuild or not image_exists(DEV_IMAGE):
        # The dev Dockerfile copies nothing in, so its own folder is enough context; the repo root
        # would upload build/ and .ccache/ to the daemon for nothing.
        devcontainer = cv_dir / ".devcontainer"
        _run(["docker", "build", "-t", DEV_IMAGE, "-f", str(devcontainer / "Dockerfile"), str(devcontainer)])
    if rebuild or not image_exists(TOOLCHAINS[7].image):
        _run(["docker", "build", "-t", TOOLCHAINS[7].image, "--build-arg", f"BASE={DEV_IMAGE}",
              "-f", str(docker_dir / "cv-sim.Dockerfile"), str(docker_dir)])


def build(cv_dir: Path, jetpack: int = 7, cuda_arch: str | None = None, rebuild_images: bool = False) -> Path:
    cv_dir = Path(cv_dir).resolve()
    toolchain = TOOLCHAINS[jetpack]
    missing = missing_submodules(cv_dir)
    if missing:
        raise SystemExit(
            f"Northstar-CV's submodules aren't checked out ({', '.join(missing)}); "
            "run `git submodule update --init --recursive`"
        )
    _ensure_images(jetpack, cv_dir, rebuild_images)

    if cuda_arch is None:
        host = host_cuda_arch()
        if host is None:
            raise SystemExit("couldn't read the GPU's compute capability from nvidia-smi; pass --cuda-arch (e.g. 89)")
        cuda_arch = ";".join(sorted({host, ORIN_ARCH}))
    print(f"[nssim] building for JetPack {jetpack}: {toolchain.what}; CUDA archs {cuda_arch}", flush=True)
    script = (
        f"cmake -S . -B {toolchain.build_dir} -G Ninja -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}'"
        f" && cmake --build {toolchain.build_dir}"
    )
    _run(["docker", "run", "--rm", "-v", f"{cv_dir}:/ws", "-w", "/ws", toolchain.image, "bash", "-lc", script])
    return binary_path(cv_dir, toolchain.build_dir)
