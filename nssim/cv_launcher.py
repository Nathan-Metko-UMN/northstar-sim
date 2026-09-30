"""Starts Northstar-CV (sim build) in a container wired to the harness.

Inside the container, socat exposes the harness's TCP UART as a PTY, and NorthstarCV2 runs with
``--sim`` (frames), ``--uart`` (that PTY) and ``--telemetry``. The same command works against a
remote Docker host (e.g. the Jetson via ``docker --context``) with ``harness_host`` set to this
machine's address.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CvContainer:
    cv_dir: Path
    image: str = "northstar-cv:sim"
    build_dir: str = "build/sim-x86"
    harness_host: str = "host.docker.internal"
    frame_port: int = 5600
    uart_port: int = 5760
    telemetry_port: int = 5800
    name: str = "nssim-cv"
    docker_context: str | None = None
    gpus: bool = True

    def command(self) -> list[str]:
        h = self.harness_host
        inner = (
            # nodelay: without it Nagle holds back the CV's small aim writes until the harness
            # ACKs the previous segment, which Windows delays by up to 200 ms.
            f"socat PTY,link=/tmp/ttySIM,raw,echo=0 TCP:{h}:{self.uart_port},forever,interval=0.5,nodelay & "
            "for i in $(seq 100); do [ -e /tmp/ttySIM ] && break; sleep 0.1; done; "
            f"exec ./{self.build_dir}/NorthstarCV2 --sim {h}:{self.frame_port} --uart /tmp/ttySIM "
            f"--telemetry {h}:{self.telemetry_port}"
        )
        cmd = ["docker"]
        if self.docker_context:
            cmd += ["--context", self.docker_context]
        cmd += ["run", "--rm", "--name", self.name]
        if self.gpus:
            cmd += ["--gpus", "all"]
        cmd += ["-v", f"{self.cv_dir}:/ws", "-w", "/ws", self.image, "bash", "-lc", inner]
        return cmd

    def start(self, log_path: Path) -> subprocess.Popen:
        self.stop()  # a leftover container from an earlier run would hold the name
        log = open(log_path, "w")
        return subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT)

    def stop(self) -> None:
        cmd = ["docker"] + (["--context", self.docker_context] if self.docker_context else [])
        subprocess.run(cmd + ["rm", "-f", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
