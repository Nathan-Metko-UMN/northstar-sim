"""The paced loop against a fake Northstar-CV, with or without pipelining."""

import json
import socket
import threading
import time

import numpy as np
import pytest

import nssim.runner as runner_mod
from nssim.camera.stream import FrameClient
from nssim.protocol.messages import TurretAimData, encode
from nssim.runner import RunConfig, Runner
from nssim.sim.assets import TrAssets
from nssim.sim.robot_constants import load_robot_constants


class _StubScene:
    """Stands in for SAPIEN: a black image, after ``delay_s`` of 'rendering'."""

    delay_s = 0.0

    def __init__(self, assets, camera, appearance=None):
        self.shape = (camera.height, camera.width, 3)

    def add_target(self, spec) -> None:
        pass

    def render(self, camera_pose, targets) -> np.ndarray:
        time.sleep(self.delay_s)
        return np.zeros(self.shape, np.uint8)


def _fake_cv(uart_port: int, frame_port: int, processing_us: int, work_s: float) -> None:
    """Acks every frame with ``processing_us`` after ``work_s`` of wall time, aim first."""
    uart = socket.create_connection(("127.0.0.1", uart_port))
    frames = FrameClient("127.0.0.1", frame_port, timeout=None)
    try:
        while True:
            header, _, _ = frames.read_frame()
            time.sleep(work_s)
            uart.sendall(encode(TurretAimData(yaw=0.1, pitch=0.05, distance=3.0, target_id=3), seq=header.seq))
            frames.ack(header.seq, processing_us)
    except (ConnectionError, OSError):
        pass
    finally:
        uart.close()
        frames.close()


def _run(tmp_path, monkeypatch, *, pipeline: bool, processing_us: int, work_s: float = 0.0, render_s: float = 0.0):
    try:
        assets = TrAssets()
    except (OSError, FileNotFoundError) as e:
        pytest.skip(f"TR assets not available: {e}")
    monkeypatch.setattr(runner_mod, "ArenaScene", _StubScene)
    monkeypatch.setattr(_StubScene, "delay_s", render_s)
    cfg = RunConfig(duration_s=0.2, fps=50.0, frame_port=0, uart_port=0, telemetry_port=0,
                    pipeline=pipeline, connect_timeout_s=10.0, ack_timeout_s=10.0)
    runner = Runner(cfg, load_robot_constants(None), tmp_path, assets)
    cv = threading.Thread(target=_fake_cv, args=(runner.link.port, runner.frames.port, processing_us, work_s),
                          daemon=True)
    cv.start()
    wall0 = time.perf_counter()
    runner.run()
    wall = time.perf_counter() - wall0
    cv.join(timeout=5)
    frames = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    return frames, runner, wall


def test_pipelined_aims_land_on_time_when_cv_takes_longer_than_a_frame(tmp_path, monkeypatch):
    frames, runner, _ = _run(tmp_path, monkeypatch, pipeline=True, processing_us=30_000)
    # Every frame that arrives within the run is acked and logged, the last one too (the 10th is
    # rendered but would arrive after the end).
    assert [f["seq"] for f in frames] == list(range(9))
    assert all(f["aim_received"] for f in frames)  # so the CV decoded every frame
    assert all({"optics", "pack", "develop_wait"} <= f["wall_ms"].keys() for f in frames)
    assert runner._late_aims == 0
    for f in frames:
        assert f["aim_late_us"] == 0
        assert f["aim_us"] >= f["arrival_us"] + 30_000


def test_pipelined_quick_aims_are_applied_late_and_counted(tmp_path, monkeypatch):
    frames, runner, _ = _run(tmp_path, monkeypatch, pipeline=True, processing_us=2_000)
    assert runner._late_aims > 0
    late = [f for f in frames if f["aim_late_us"] > 0]
    assert late
    for f in late:
        due_us = f["aim_us"] - f["aim_late_us"]
        assert 0 < due_us - (f["arrival_us"] + 2_000) < 2_000  # processing + the aim's UART bytes


def test_lockstep_never_applies_an_aim_late(tmp_path, monkeypatch):
    frames, runner, _ = _run(tmp_path, monkeypatch, pipeline=False, processing_us=2_000)
    assert [f["seq"] for f in frames] == list(range(9))
    assert runner._late_aims == 0
    assert all(f["aim_late_us"] == 0 for f in frames)


def test_pipelining_overlaps_rendering_with_the_cv(tmp_path, monkeypatch):
    slow = dict(processing_us=30_000, work_s=0.05, render_s=0.05)
    _, _, lockstep = _run(tmp_path / "lockstep", monkeypatch, pipeline=False, **slow)
    _, _, pipelined = _run(tmp_path / "pipelined", monkeypatch, pipeline=True, **slow)
    assert pipelined < 0.8 * lockstep, (pipelined, lockstep)
