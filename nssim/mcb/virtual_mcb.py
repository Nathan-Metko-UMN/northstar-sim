"""Virtual MCB: the firmware side of the vision UART link, driven by simulation time.

Behavior mirrors northstar-robomaster ``main`` (``communication/serial/vision_comms.cpp``,
``robot/standard/standard_uart_constants.hpp`` and ``control/turret/cv``):

- Nothing is sent until 1 s after MCB boot, then odometry every 4 ms, health every 30 ms and
  the robot ID every 1030 ms. A ROBOT_ID request only sends if that 1030 ms timer is due.
- ALIVE keeps the CV "online" for 1 s.
- TURRET_AIM_DATA sets absolute yaw/pitch setpoints; (0, 0) means no target. The fire gate
  tolerance is ``1.5 * atan(plate_half_size / distance)``, looked up by target ID; IDs the
  firmware does not know keep the previous tolerance.

All times are integer microseconds on the MCB clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from nssim.protocol import (
    Empty,
    Frame,
    FrameDecoder,
    Health,
    MsgType,
    Odometry,
    RobotId,
    RobotIdMsg,
    TurretAimData,
    decode,
    encode,
)

TWO_PI = 2.0 * math.pi

# vision_comms.hpp plateLookup: robot ID -> (width, height) in meters.
PLATE_LOOKUP = {1: (0.2, 0.15), 7: (0.15, 0.15), 3: (0.15, 0.15)}


@dataclass
class McbConfig:
    robot_id: int = RobotId.RED_SOLDIER_1
    # MCB clock value when the simulation starts; the MCB boots well before the Jetson.
    boot_offset_us: int = 5_000_000
    start_delay_us: int = 1_000_000  # TIME_BEFORE_UART_START
    odometry_period_us: int = 4_000  # Standard; Hero/Sentry use 20 ms
    health_period_us: int = 30_000
    robot_id_period_us: int = 1_030_000
    cv_offline_timeout_us: int = 1_000_000
    aim_tolerance_scale: float = 1.5
    hp: int = 400


@dataclass
class TurretState:
    """What the MCB measures, in the conventions it sends to the CV."""

    yaw: float = 0.0  # world yaw from the turret IMU, CCW positive (not yet wrapped)
    pitch: float = 0.0  # positive looking down
    roll: float = 0.0
    yaw_rate: float = 0.0  # gyro z, rad/s
    chassis_vel_x: float = 0.0
    chassis_vel_y: float = 0.0


@dataclass
class AimState:
    aim: TurretAimData = field(default_factory=TurretAimData)
    updated: bool = False  # firmware ``aimDataUpdated``: a target is being tracked
    max_error_yaw: float = 0.0
    max_error_pitch: float = 0.0
    received_us: int | None = None


class _PeriodicTimer:
    """taproot PeriodicMilliTimer: fires once per period, catching up without drift."""

    def __init__(self, period_us: int):
        self.period_us = period_us
        self.next_us: int | None = None

    def start(self, now_us: int) -> None:
        self.next_us = now_us + self.period_us

    def execute(self, now_us: int) -> bool:
        if self.next_us is None or now_us < self.next_us:
            return False
        self.next_us += self.period_us
        if self.next_us <= now_us:
            self.next_us = now_us + self.period_us
        return True


def wrap_2pi(angle: float) -> float:
    """Mahony ``getYaw()``: fmod(yaw + 2*pi, 2*pi), i.e. [0, 2*pi) for yaw >= -2*pi."""
    wrapped = math.fmod(angle, TWO_PI)
    return wrapped + TWO_PI if wrapped < 0 else wrapped


class VirtualMcb:
    def __init__(self, config: McbConfig | None = None):
        self.config = config or McbConfig()
        self.decoder = FrameDecoder()
        self.aim_state = AimState()
        self._cv_seen_until_us = -1
        self._seq = 0
        c = self.config
        self._odometry = _PeriodicTimer(c.odometry_period_us)
        self._health = _PeriodicTimer(c.health_period_us)
        self._robot_id = _PeriodicTimer(c.robot_id_period_us)
        # All three timers are armed at boot; the first sends happen once the start delay has passed.
        for timer in (self._odometry, self._health, self._robot_id):
            timer.start(0)
        self.rx_counts: dict[int, int] = {}
        self.tx_counts: dict[int, int] = {}

    # --- clock -----------------------------------------------------------------------------

    def mcb_time_us(self, sim_time_us: int) -> int:
        return self.config.boot_offset_us + sim_time_us

    # --- MCB -> CV -----------------------------------------------------------------------

    def poll_tx(self, now_us: int, turret: TurretState) -> list[tuple[int, bytes]]:
        """Frames the firmware writes at MCB time ``now_us``, as (queue_time_us, bytes)."""
        if now_us < self.config.start_delay_us:
            return []
        out: list[tuple[int, bytes]] = []
        if self._odometry.execute(now_us):
            odom = Odometry(
                timestamp_us=now_us,
                vel_x=turret.chassis_vel_x,
                vel_y=turret.chassis_vel_y,
                pitch=turret.pitch,
                yaw=wrap_2pi(turret.yaw),
                roll=turret.roll,
                yaw_vel=turret.yaw_rate,
            )
            out.append((now_us, self._encode(odom)))
        out += self._send_robot_id(now_us)
        if self._health.execute(now_us):
            out.append((now_us, self._encode(Health(hp=self.config.hp))))
        return out

    def _send_robot_id(self, now_us: int) -> list[tuple[int, bytes]]:
        if not self._robot_id.execute(now_us):
            return []
        return [(now_us, self._encode(RobotIdMsg(robot_id=int(self.config.robot_id))))]

    def _encode(self, msg) -> bytes:
        frame = encode(msg, seq=self._seq)
        self._seq = (self._seq + 1) & 0xFF
        msg_type = int(msg.TYPE)
        self.tx_counts[msg_type] = self.tx_counts.get(msg_type, 0) + 1
        return frame

    # --- CV -> MCB -----------------------------------------------------------------------

    def feed_rx(self, data: bytes, now_us: int) -> list[tuple[int, bytes]]:
        """Handle bytes from the CV; returns any immediate replies (ROBOT_ID)."""
        replies: list[tuple[int, bytes]] = []
        for frame in self.decoder.feed(data):
            replies += self._handle(frame, now_us)
        return replies

    def _handle(self, frame: Frame, now_us: int) -> list[tuple[int, bytes]]:
        self.rx_counts[frame.msg_type] = self.rx_counts.get(frame.msg_type, 0) + 1
        msg = decode(frame)
        if isinstance(msg, TurretAimData):
            self._on_aim(msg, now_us)
        elif isinstance(msg, Empty) and msg.msg_type == MsgType.ALIVE:
            self._cv_seen_until_us = now_us + self.config.cv_offline_timeout_us
        elif isinstance(msg, Empty) and msg.msg_type == MsgType.ROBOT_ID:
            if now_us >= self.config.start_delay_us:
                return self._send_robot_id(now_us)
        return []

    def _on_aim(self, aim: TurretAimData, now_us: int) -> None:
        state = self.aim_state
        state.aim = aim
        state.updated = aim.has_target
        state.received_us = now_us
        if aim.distance != 0.0:
            dims = PLATE_LOOKUP.get(aim.target_id & 0xFF)
            if dims is not None:
                scale = self.config.aim_tolerance_scale
                state.max_error_yaw = math.atan((dims[0] / 2.0) / aim.distance) * scale
                state.max_error_pitch = math.atan((dims[1] / 2.0) / aim.distance) * scale

    # --- firmware state ------------------------------------------------------------------

    def cv_online(self, now_us: int) -> bool:
        return now_us < self._cv_seen_until_us

    def within_aiming_tolerance(self, yaw_measured: float, pitch_measured: float) -> bool:
        """TurretCVControlCommand::execute: wrapped yaw/pitch errors within the plate tolerance."""
        state = self.aim_state
        if not state.updated:
            return False
        yaw_err = abs(_min_difference(state.aim.yaw - yaw_measured))
        pitch_err = abs(_min_difference(state.aim.pitch - pitch_measured))
        return yaw_err < state.max_error_yaw and pitch_err < state.max_error_pitch


def _min_difference(angle: float) -> float:
    """Angle(x).minDifference(0): the equivalent angle in [-pi, pi)."""
    return (angle + math.pi) % TWO_PI - math.pi
