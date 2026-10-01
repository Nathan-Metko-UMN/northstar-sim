"""Builds Northstar-CV's simulator binary in Docker, the local stand-in for the Jetson.

The robot may run JetPack 6 or 7. Northstar-CV's own dev container (``.devcontainer/Dockerfile``)
builds for either with its ``JETPACK`` build argument, into the same images ``scripts/dev.sh``
uses:

- JetPack 7: ``northstar-cv:jetpack7``, CUDA 13.2 on Ubuntu 24.04. Builds into ``build/sim-jetpack7``.
- JetPack 6: ``northstar-cv:jetpack6``, CUDA 12.6 on Ubuntu 22.04. Builds into ``build/sim-jetpack6``.

On each goes socat (``docker/cv-sim.Dockerfile``), as ``northstar-cv:sim-jetpack7`` and
``northstar-cv:sim-jetpack6``. Both compile for this machine's GPU (to run in the sim) and the
Orin's sm_87 (so the Jetson's code is checked too). ``nssim run`` mounts the build and starts it.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nssim.paths import REPO_ROOT

ORIN_ARCH = "87"


@dataclass(frozen=True)
class Toolchain:
    jetpack: int
    dev_image: str  # Northstar-CV's dev container; `scripts/dev.sh --jetpack N` makes the same
    image: str  # dev_image plus socat: what the binary is built and run in
    build_dir: str  # inside the Northstar-CV checkout
    what: str


TOOLCHAINS = {
    7: Toolchain(7, "northstar-cv:jetpack7", "northstar-cv:sim-jetpack7", "build/sim-jetpack7", "CUDA 13.2 on Ubuntu 24.04"),
    6: Toolchain(6, "northstar-cv:jetpack6", "northstar-cv:sim-jetpack6", "build/sim-jetpack6", "CUDA 12.6 on Ubuntu 22.04"),
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


def _build_images(toolchain: Toolchain, cv_dir: Path, rebuild: bool) -> None:
    """Brings both images up to date. Docker's cache makes this a few seconds when nothing changed,
    and it picks up a changed dev Dockerfile, or a dev image `scripts/dev.sh` rebuilt, by itself."""
    devcontainer = cv_dir / ".devcontainer"
    dockerfile = devcontainer / "Dockerfile"
    if toolchain.jetpack != 7 and "ARG JETPACK" not in dockerfile.read_text():
        raise SystemExit(
            f"{dockerfile} only builds JetPack 7's toolchain; check out a later sim-harness commit "
            f"of Northstar-CV to build for JetPack {toolchain.jetpack}"
        )
    no_cache = ["--no-cache"] if rebuild else []
    # A rebuild also takes NVIDIA's latest base image (--pull). The dev Dockerfile copies nothing
    # in, so its own folder is enough context; the repo root would upload build/ and .ccache/ to
    # the daemon for nothing.
    _run(["docker", "build", *no_cache, *(["--pull"] if rebuild else []), "-t", toolchain.dev_image,
          "--build-arg", f"JETPACK={toolchain.jetpack}", "-f", str(dockerfile), str(devcontainer)])
    docker_dir = REPO_ROOT / "docker"
    _run(["docker", "build", *no_cache, "-t", toolchain.image, "--build-arg", f"BASE={toolchain.dev_image}",
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
    _build_images(toolchain, cv_dir, rebuild_images)

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
