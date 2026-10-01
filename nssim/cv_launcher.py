"""Starts Northstar-CV (sim build) wired to the harness: in a local container, or on the Jetson.

Either way socat exposes the harness's TCP UART as a PTY, and NorthstarCV2 runs with ``--sim``
(frames), ``--uart`` (that PTY) and ``--telemetry``, connecting back to this machine.
"""

from __future__ import annotations

import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nssim.cv_build import TOOLCHAINS


def cv_command(harness_host: str, build_dir: str, frame_port: int, uart_port: int, telemetry_port: int) -> str:
    """The shell command that runs Northstar-CV against the harness, from its repo root."""
    h = harness_host
    return (
        # nodelay: without it Nagle holds back the CV's small aim writes until the harness ACKs the
        # previous segment, which Windows delays by up to 200 ms.
        f"socat PTY,link=/tmp/ttySIM,raw,echo=0 TCP:{h}:{uart_port},forever,interval=0.5,nodelay & "
        "for i in $(seq 100); do [ -e /tmp/ttySIM ] && break; sleep 0.1; done; "
        f"exec ./{build_dir}/NorthstarCV2 --sim {h}:{frame_port} --uart /tmp/ttySIM --telemetry {h}:{telemetry_port}"
    )


def local_address_towards(host: str) -> str:
    """This machine's address on the network that reaches ``host`` (what ``host`` should connect to)."""
    target = socket.gethostbyname(host)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((target, 9))  # UDP connect sends nothing; it only picks the route
        return s.getsockname()[0]


def ssh_hostname(destination: str) -> str:
    """The machine an ssh destination reaches: the HostName of an alias in ~/.ssh/config (ssh -G
    resolves it the way ssh would), else the host part of user@host."""
    try:
        out = subprocess.run(["ssh", "-G", destination], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    for line in out.splitlines():
        key, _, value = line.partition(" ")
        if key == "hostname" and value.strip():
            return value.strip()
    return destination.rsplit("@", 1)[-1]


@dataclass
class CvContainer:
    """Northstar-CV in Docker on this machine (Docker Desktop stands in for the Jetson)."""

    cv_dir: Path
    image: str = TOOLCHAINS[7].image
    build_dir: str = TOOLCHAINS[7].build_dir
    harness_host: str = "host.docker.internal"
    frame_port: int = 5600
    uart_port: int = 5760
    telemetry_port: int = 5800
    name: str = "nssim-cv"
    docker_context: str | None = None
    gpus: bool = True

    def command(self) -> list[str]:
        inner = cv_command(self.harness_host, self.build_dir, self.frame_port, self.uart_port, self.telemetry_port)
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


@dataclass
class CvRemote:
    """Northstar-CV on another machine (the Jetson), started over SSH: natively, or in the Docker
    image its scripts/dev.sh builds there (``image``).

    That machine needs Northstar-CV built from a branch with the simulator mode, socat (in the
    image, or installed natively), and this machine's frame, UART and telemetry ports reachable
    (see the README's firewall note). SSH must log in without a password prompt (a key).
    """

    host: str  # ssh destination: "nvidia@jetson.local", or a Host alias from ~/.ssh/config
    cv_dir: str = "~/Northstar-CV"
    build_dir: str = "build"
    harness_host: str | None = None  # default: this machine's address on the route to the Jetson
    image: str | None = None  # e.g. "northstar-cv:jetpack6"; None runs it natively
    frame_port: int = 5600
    uart_port: int = 5760
    telemetry_port: int = 5800
    name: str = "nssim-cv"  # the container's, so stop() can find it

    def __post_init__(self):
        if self.harness_host is None:
            self.harness_host = local_address_towards(ssh_hostname(self.host))

    def _ssh(self, remote: str) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", self.host, remote]

    def command(self) -> list[str]:
        inner = cv_command(self.harness_host, self.build_dir, self.frame_port, self.uart_port, self.telemetry_port)
        if self.image is None:
            return self._ssh(f"cd {self.cv_dir} && {inner}")
        # As scripts/dev.sh runs it: the checkout at /ws and the GPU. Host networking spares the
        # CV's connections back to this machine Docker's NAT.
        return self._ssh(
            f'cd {self.cv_dir} && exec docker run --rm --name {self.name} --gpus all --network host '
            f'-v "$PWD":/ws -w /ws {self.image} bash -lc {shlex.quote(inner)}'
        )

    def start(self, log_path: Path) -> subprocess.Popen:
        self.stop()  # a leftover run (e.g. after a crash here) would hold the PTY
        log = open(log_path, "w")
        return subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def stop(self) -> None:
        if self.image is None:
            # Patterns written so they don't match this pkill's own command line.
            remote = "pkill -f '[N]orthstarCV2 --sim'; pkill -f '[s]ocat PTY,link=/tmp/ttySIM'; true"
        else:
            remote = f"docker rm -f {self.name} >/dev/null 2>&1; true"
        subprocess.run(
            self._ssh(remote), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        )
