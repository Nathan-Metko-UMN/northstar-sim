import argparse
import shlex
import subprocess
from pathlib import Path

from nssim import cli, cv_launcher
from nssim.cv_launcher import CvContainer, CvRemote, cv_command, local_address_towards, ssh_hostname


def test_cv_command_wires_every_port_to_the_harness():
    cmd = cv_command("10.0.0.5", "build", 5600, 5760, 5800)
    assert "TCP:10.0.0.5:5760" in cmd  # socat's UART link
    assert "--sim 10.0.0.5:5600" in cmd
    assert "--telemetry 10.0.0.5:5800" in cmd
    assert "--uart /tmp/ttySIM" in cmd
    assert cmd.rstrip().split()[-1] == "10.0.0.5:5800"


def test_remote_runs_from_the_checkout_over_ssh_without_prompts():
    remote = CvRemote(host="nvidia@jetson", cv_dir="~/Northstar-CV", build_dir="build", harness_host="10.0.0.5")
    cmd = remote.command()
    assert cmd[0] == "ssh" and "BatchMode=yes" in cmd and cmd[-2] == "nvidia@jetson"
    assert cmd[-1].startswith("cd ~/Northstar-CV && socat ")
    assert "./build/NorthstarCV2 --sim 10.0.0.5:5600" in cmd[-1]


def test_remote_in_the_dev_container_runs_the_same_command_inside_docker():
    remote = CvRemote(host="jetson1", cv_dir="~/CV/Northstar-CV", build_dir="build/jetpack6",
                      harness_host="10.0.0.5", image="northstar-cv:jetpack6")
    cmd = remote.command()
    assert cmd[0] == "ssh" and cmd[-2] == "jetson1"
    words = shlex.split(cmd[-1])  # as the Jetson's shell splits it
    assert words[:4] == ["cd", "~/CV/Northstar-CV", "&&", "exec"]
    run = words[4:]
    assert run[:2] == ["docker", "run"] and "--gpus" in run and "--network" in run
    assert run[run.index("-v") + 1] == "$PWD:/ws"  # expanded by that shell, after the cd
    assert run[-4:-1] == ["northstar-cv:jetpack6", "bash", "-lc"]
    assert run[-1] == cv_command("10.0.0.5", "build/jetpack6", 5600, 5760, 5800)  # reaches bash intact


def test_remote_stop_removes_its_container_or_kills_the_native_processes(monkeypatch):
    sent = []
    monkeypatch.setattr(cv_launcher.subprocess, "run", lambda cmd, **kw: sent.append(cmd[-1]))
    CvRemote(host="jetson1", harness_host="10.0.0.5", image="northstar-cv:jetpack6").stop()
    CvRemote(host="jetson1", harness_host="10.0.0.5").stop()
    assert sent[0].startswith("docker rm -f nssim-cv")
    assert "pkill" in sent[1]


def test_ssh_hostname_follows_an_ssh_config_alias(monkeypatch):
    def fake_run(cmd, **kw):
        assert cmd == ["ssh", "-G", "jetson1"]
        return subprocess.CompletedProcess(cmd, 0, stdout="user northstar\nhostname 192.168.1.50\nport 22\n")

    monkeypatch.setattr(cv_launcher.subprocess, "run", fake_run)
    assert ssh_hostname("jetson1") == "192.168.1.50"


def test_ssh_hostname_without_ssh_takes_the_host_part(monkeypatch):
    def no_ssh(cmd, **kw):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(cv_launcher.subprocess, "run", no_ssh)
    assert ssh_hostname("nvidia@10.0.0.9") == "10.0.0.9"


def test_jetpack6_dev_container_defaults_to_dev_shs_build_directory():
    args = argparse.Namespace(jetson="jetson1", jetson_dir="~/CV/Northstar-CV", jetson_build=None,
                              jetson_image="northstar-cv:jetpack6", harness_ip="10.0.0.5")
    assert cli._launcher(args, Path("unused")).build_dir == "build/jetpack6"
    args.jetson_image = None
    assert cli._launcher(args, Path("unused")).build_dir == "build"


def test_container_uses_docker_desktops_host_name():
    cmd = CvContainer(cv_dir=Path("/cv")).command()
    assert cmd[:2] == ["docker", "run"] and "--gpus" in cmd
    assert "--sim host.docker.internal:5600" in cmd[-1]


def test_local_address_towards_loopback_is_loopback():
    assert local_address_towards("127.0.0.1") == "127.0.0.1"
