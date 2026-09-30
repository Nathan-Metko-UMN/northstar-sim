"""Paced hardware-in-the-loop run: the sim renders one frame at a time and waits for Northstar-CV.

Time is the MCB clock in integer microseconds, advanced in fixed ticks. Per frame:

1. Render at mid-exposure and record ground truth for that instant.
2. At the frame's arrival time (end of exposure + GigE transfer), deliver every UART byte that
   has arrived by then, send the frame, and wait for Northstar-CV's ack.
3. The ack reports how long Northstar-CV really took. Its aim command reaches the MCB at
   arrival + processing + UART time, and the turret reacts from then on.

Before the first frame (while Northstar-CV starts up and waits for the robot ID) the virtual MCB
runs in real time.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from nssim.camera import CameraModel, FrameHeader, FrameServer, LinkModel, mosaic_rggb
from nssim.mcb import McbConfig, TurretState, VirtualMcb
from nssim.mcb.link import UartLink
from nssim.mcb.turret import TurretConfig, TurretModel
from nssim.protocol import FrameDecoder, MsgType, encode_frame
from nssim.sim.assets import FLOOR_TOP_Z, TrAssets
from nssim.sim.projection import project
from nssim.sim.robot_constants import RobotConstants
from nssim.sim.scene import ArenaScene, Lighting
from nssim.sim.shooter import Shooter
from nssim.sim.targets import Motion, Target, TargetSpec, light_bar_corners
from nssim.telemetry import TelemetryReceiver


@dataclass
class TargetConfig:
    spec: TargetSpec = field(default_factory=TargetSpec)
    motion: Motion = field(default_factory=Motion)


@dataclass
class RunConfig:
    name: str = "run"
    duration_s: float = 5.0
    fps: float = 1e6 / 6000  # camera frame rate in sim time
    exposure_us: int = 2000
    link: LinkModel = field(default_factory=LinkModel)
    camera: CameraModel = field(default_factory=CameraModel)
    lighting: Lighting = field(default_factory=Lighting)
    tick_us: int = 1000
    shooter_base_xy: list[float] = field(default_factory=lambda: [-1.5, 0.0])
    initial_aim: str | list[float] | None = None  # target name to face at t=0, or [yaw, pitch]
    turret: TurretConfig = field(default_factory=TurretConfig)
    mcb: McbConfig = field(default_factory=McbConfig)
    targets: list[TargetConfig] = field(default_factory=lambda: [TargetConfig()])
    frame_port: int = 5600
    uart_port: int = 5760
    telemetry_port: int = 5800
    # Lossless zlib on the frame stream. Worth it where the link is slow (Docker Desktop's
    # host->container path manages ~15 MB/s); off for a real GigE link to the Jetson.
    compress_frames: bool = True
    connect_timeout_s: float = 300.0
    ack_timeout_s: float = 30.0


@dataclass
class _PendingFrame:
    seq: int
    capture_us: int
    render_us: int
    arrival_us: int
    bayer: np.ndarray
    gt: dict


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


class Runner:
    def __init__(self, config: RunConfig, constants: RobotConstants, out_dir: Path, assets: TrAssets | None = None):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.assets = assets or TrAssets()
        self.shooter = Shooter(constants, [*config.shooter_base_xy, FLOOR_TOP_Z])
        self.targets = {tc.spec.name: Target(tc.spec, tc.motion, FLOOR_TOP_Z) for tc in config.targets}
        self.scene = ArenaScene(self.assets, config.camera, config.lighting)
        for tc in config.targets:
            self.scene.add_target(tc.spec)

        yaw, pitch = self._initial_aim()
        self.turret = TurretModel(yaw, pitch, config.turret)
        self.mcb = VirtualMcb(config.mcb)
        self.link = UartLink(port=config.uart_port)
        self.frames = FrameServer(port=config.frame_port)
        self.telemetry = TelemetryReceiver(port=config.telemetry_port)
        self._rx_decoder = FrameDecoder()
        self._pending_rx: list[tuple[int, bytes]] = []  # (apply at MCB us, raw frame bytes)
        self._aims_received = 0
        self.mcb_us = config.mcb.boot_offset_us
        self._files = {
            name: open(self.out_dir / f"{name}.jsonl", "w")
            for name in ("frames", "telemetry", "events")
        }

    # --- setup -----------------------------------------------------------------------------

    def _initial_aim(self) -> tuple[float, float]:
        aim = self.config.initial_aim
        if isinstance(aim, str):
            return self.shooter.aim_at(self.targets[aim].state(0.0).center)
        if aim is not None:
            return float(aim[0]), float(aim[1])
        return 0.0, 0.0

    def _hello(self) -> dict:
        cam = self.config.camera
        return {
            "mode": "paced",
            "width": cam.width,
            "height": cam.height,
            "pixel_format": "BayerRG8",
            "K": cam.K,
            "dist": cam.dist,
            "exposure_us": self.config.exposure_us,
            "fps": self.config.fps,
            "scenario": self.config.name,
        }

    # --- helpers -----------------------------------------------------------------------------

    def _turret_state(self) -> TurretState:
        return TurretState(yaw=self.turret.yaw, pitch=self.turret.pitch, yaw_rate=self.turret.yaw_rate)

    def _mcb_tx(self) -> None:
        for queued_us, data in self.mcb.poll_tx(self.mcb_us, self._turret_state()):
            odometry_ts = queued_us if data[5] == MsgType.ODOMETRY else None
            self.link.queue(queued_us, data, odometry_ts)

    def _log(self, name: str, record: dict) -> None:
        self._files[name].write(json.dumps(record, default=_json_default) + "\n")

    def _event(self, what: str, **fields) -> None:
        self._log("events", {"mcb_us": self.mcb_us, "event": what, **fields})
        print(f"[nssim] {what} {fields if fields else ''}", flush=True)

    def _drain_telemetry(self) -> None:
        now = time.time()
        for msg in self.telemetry.poll():
            msg["_rx_wall"] = now
            self._log("telemetry", msg)

    def _collect_rx(self) -> list[tuple[int, bytes]]:
        """Complete frames the CV has written so far, as (msg_type, frame bytes) for the MCB."""
        data = self.link.recv()
        frames = []
        for frame in self._rx_decoder.feed(data):
            if frame.msg_type == MsgType.TURRET_AIM_DATA:
                self._aims_received += 1
            frames.append((frame.msg_type, encode_frame(frame.msg_type, frame.payload, frame.seq)))
        return frames

    # --- phases ------------------------------------------------------------------------------

    def _startup(self) -> None:
        """Real-time MCB until Northstar-CV has connected its frame stream."""
        self._event("waiting for Northstar-CV (UART)", port=self.link.port)
        deadline = time.monotonic() + self.config.connect_timeout_s
        while not self.link.accept(timeout=0.5):
            if time.monotonic() > deadline:
                raise TimeoutError("Northstar-CV never connected to the UART link")
        self._event("UART connected; waiting for the frame stream", port=self.frames.port)

        last_wall = time.monotonic()
        while not self.frames.connected:
            now_wall = time.monotonic()
            self.mcb_us += int((now_wall - last_wall) * 1e6)
            last_wall = now_wall
            self._mcb_tx()
            self.link.flush(self.mcb_us)
            for _, data in self._collect_rx():
                for queued_us, reply in self.mcb.feed_rx(data, self.mcb_us):
                    self.link.queue(queued_us, reply)
            self._drain_telemetry()
            try:
                self.frames.accept(self._hello(), timeout=0)  # non-blocking poll
            except (BlockingIOError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Northstar-CV never connected its frame stream")
            time.sleep(0.0005)
        self._event("frame stream connected; starting paced run")

    def _ground_truth(self, t: float):
        """(ground-truth record, camera pose, target states) at scenario time t."""
        cam = self.config.camera
        cam_pose = self.shooter.camera(self.turret.yaw, self.turret.pitch)
        base = self.shooter.base_xyz
        targets = []
        states = {}
        for name, target in self.targets.items():
            state = target.state(t)
            states[name] = state
            spec = target.spec
            bars = self.assets.plate_geometry(spec.panel).bars
            plates = []
            for i, plate in enumerate(state.plates):
                corners = light_bar_corners(plate, bars.half_spacing, bars.half_height)
                facing = float(plate.R[:, 0] @ (cam_pose.p - plate.p)) > 0
                plates.append(
                    {
                        "index": i,
                        "center_base": plate.p - base,
                        "normal": plate.R[:, 0],
                        "facing": facing,
                        "corners_px": project(cam, cam_pose, corners),
                    }
                )
            targets.append(
                {
                    "name": name,
                    "number": spec.number,
                    "center_base": state.center - base,
                    "velocity": state.velocity,
                    "spin": state.spin,
                    "omega": state.omega,
                    "radius_high": spec.radius_high,
                    "radius_low": spec.radius_low,
                    "z_offset": spec.z_offset,
                    "plates": plates,
                }
            )
        gt = {
            "t": t,
            "turret": {"yaw": self.turret.yaw, "pitch": self.turret.pitch, "yaw_rate": self.turret.yaw_rate},
            "base_world": base,
            "camera_world": {"p": cam_pose.p, "R": cam_pose.R},
            "targets": targets,
        }
        return gt, cam_pose, states

    def _deliver(self, frame: _PendingFrame) -> None:
        self.link.flush(frame.arrival_us)
        watermark = self.link.last_delivered_odometry_ts
        cam = self.config.camera
        header = FrameHeader(
            seq=frame.seq,
            capture_mcb_us=frame.capture_us,
            exposure_us=self.config.exposure_us,
            arrival_delay_us=frame.arrival_us - frame.capture_us,
            width=cam.width,
            height=cam.height,
            odom_watermark_mcb_us=-1 if watermark is None else watermark,
        )
        aims_before = self._aims_received
        wall0 = time.perf_counter()
        self.frames.send_frame(header, frame.bayer, compress=self.config.compress_frames)
        wall_sent = time.perf_counter()
        seq, processing_us = self.frames.recv_ack(timeout=self.config.ack_timeout_s)
        wall_ack = time.perf_counter()
        if seq != frame.seq:
            raise RuntimeError(f"ack for frame {seq}, expected {frame.seq}")

        # The aim for this frame was written to the UART just before the ack; give socat a moment.
        rx = self._collect_rx()
        deadline = time.perf_counter() + 0.5
        while self._aims_received == aims_before and time.perf_counter() < deadline:
            time.sleep(0.0002)
            rx += self._collect_rx()
        wall_aim = time.perf_counter()
        apply_base = frame.arrival_us + processing_us
        for msg_type, data in rx:
            self._pending_rx.append((apply_base + int(len(data) * self.link.byte_us), data))
        self._drain_telemetry()
        self._log(
            "frames",
            {
                "seq": frame.seq,
                "capture_us": frame.capture_us,
                "render_us": frame.render_us,
                "arrival_us": frame.arrival_us,
                "processing_us": processing_us,
                "wall_roundtrip_ms": (wall_ack - wall0) * 1e3,
                "wall_ms": {
                    "send": (wall_sent - wall0) * 1e3,
                    "ack": (wall_ack - wall_sent) * 1e3,
                    "aim": (wall_aim - wall_ack) * 1e3,
                },
                "aim_received": self._aims_received > aims_before,
                "gt": frame.gt,
            },
        )

    def run(self) -> Path:
        cfg = self.config
        try:
            self._startup()
            t0_us = self.mcb_us
            period_us = int(round(1e6 / cfg.fps))
            half_exposure = cfg.exposure_us // 2
            link_free_us = 0
            seq = 0
            next_render_us = t0_us + period_us - half_exposure
            pending: list[_PendingFrame] = []
            frame_bytes = cfg.camera.width * cfg.camera.height
            end_us = t0_us + int(cfg.duration_s * 1e6)
            wall_start = time.perf_counter()

            while self.mcb_us < end_us:
                self.mcb_us += cfg.tick_us
                t = (self.mcb_us - t0_us) / 1e6

                # CV -> MCB messages that have arrived by now (aims, ALIVE, ...).
                due = [p for p in self._pending_rx if p[0] <= self.mcb_us]
                if due:
                    self._pending_rx = [p for p in self._pending_rx if p[0] > self.mcb_us]
                    for apply_us, data in due:
                        for queued_us, reply in self.mcb.feed_rx(data, apply_us):
                            self.link.queue(queued_us, reply)

                aim = self.mcb.aim_state
                self.turret.step(cfg.tick_us / 1e6, (aim.aim.yaw, aim.aim.pitch) if aim.updated else None)
                self._mcb_tx()

                if self.mcb_us >= next_render_us:
                    gt, cam_pose, states = self._ground_truth(t)
                    rgb = self.scene.render(cam_pose, states)
                    capture_us = next_render_us + half_exposure
                    start = max(capture_us, link_free_us)
                    link_free_us = start + cfg.link.transfer_us(frame_bytes)
                    pending.append(
                        _PendingFrame(seq, capture_us, self.mcb_us, link_free_us, mosaic_rggb(rgb), gt)
                    )
                    seq += 1
                    next_render_us += period_us

                while pending and pending[0].arrival_us <= self.mcb_us:
                    self._deliver(pending.pop(0))

                self.link.flush(self.mcb_us)

            wall = time.perf_counter() - wall_start
            self._event("done", frames=seq, sim_s=cfg.duration_s, wall_s=round(wall, 2))
        finally:
            for f in self._files.values():
                f.close()
            self.frames.close()
            self.link.close()
            self.telemetry.close()
            (self.out_dir / "config.json").write_text(json.dumps(asdict(cfg), default=_json_default, indent=2))
        return self.out_dir
