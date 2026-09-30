from pathlib import Path

from nssim.cv_launcher import CvContainer, CvRemote, cv_command, local_address_towards


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


def test_container_uses_docker_desktops_host_name():
    cmd = CvContainer(cv_dir=Path("/cv")).command()
    assert cmd[:2] == ["docker", "run"] and "--gpus" in cmd
    assert "--sim host.docker.internal:5600" in cmd[-1]


def test_local_address_towards_loopback_is_loopback():
    assert local_address_towards("127.0.0.1") == "127.0.0.1"
