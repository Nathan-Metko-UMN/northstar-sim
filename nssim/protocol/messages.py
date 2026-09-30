"""Message payloads on the MCB <-> Northstar-CV link.

Layouts follow the MCB firmware's ``vision_comms.hpp`` (northstar-robomaster ``main``) and
Northstar-CV's ``src/uart/messages``. All fields are packed little-endian.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from .framing import Frame, encode_frame


class MsgType(IntEnum):
    TURRET_AIM_DATA = 1  # CV -> MCB
    ROBOT_ID = 2  # CV -> MCB: empty request. MCB -> CV: 1-byte robot ID
    ALIVE = 3  # CV -> MCB, empty
    ODOMETRY = 4  # MCB -> CV
    AUTO_PATH = 5
    HEALTH = 6  # MCB -> CV
    REF_TURRET_DATA = 7  # disabled in MCB main
    VISION_LOCALIZATION = 8
    FLY_SKY_DATA = 9
    VT_DATA = 10
    RESTART_DETECTOR = 11  # MCB -> CV, empty


class RobotId(IntEnum):
    """Referee-system robot IDs; ``id // 100`` is the team color (0 red, 1 blue)."""

    RED_HERO = 1
    RED_ENGINEER = 2
    RED_SOLDIER_1 = 3
    RED_SOLDIER_2 = 4
    RED_SOLDIER_3 = 5
    RED_DRONE = 6
    RED_SENTINEL = 7
    BLUE_HERO = 101
    BLUE_ENGINEER = 102
    BLUE_SOLDIER_1 = 103
    BLUE_SOLDIER_2 = 104
    BLUE_SOLDIER_3 = 105
    BLUE_DRONE = 106
    BLUE_SENTINEL = 107


@dataclass
class TurretAimData:
    """Aim command. yaw/pitch are absolute turret setpoints; (0, 0) means no target."""

    yaw: float = 0.0
    pitch: float = 0.0
    distance: float = 0.0
    target_id: int = 0

    TYPE: ClassVar[MsgType] = MsgType.TURRET_AIM_DATA
    FORMAT: ClassVar[struct.Struct] = struct.Struct("<fffH")

    @property
    def has_target(self) -> bool:
        return not (self.yaw == 0.0 and self.pitch == 0.0)

    def pack(self) -> bytes:
        return self.FORMAT.pack(self.yaw, self.pitch, self.distance, self.target_id)

    @classmethod
    def unpack(cls, payload: bytes) -> "TurretAimData":
        return cls(*cls.FORMAT.unpack_from(payload))


@dataclass
class Odometry:
    """Turret/chassis state at ``timestamp_us`` on the MCB clock (a u32 that wraps every ~71.6 min).

    Angles come from the turret BMI088: yaw is wrapped to [0, 2*pi), pitch is positive
    looking down, ``yaw_vel`` is gyro z in rad/s. Velocities are chassis m/s.
    """

    timestamp_us: int = 0
    vel_x: float = 0.0
    vel_y: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    yaw_vel: float = 0.0

    TYPE: ClassVar[MsgType] = MsgType.ODOMETRY
    FORMAT: ClassVar[struct.Struct] = struct.Struct("<I6f")

    def pack(self) -> bytes:
        return self.FORMAT.pack(
            self.timestamp_us & 0xFFFFFFFF,
            self.vel_x,
            self.vel_y,
            self.pitch,
            self.yaw,
            self.roll,
            self.yaw_vel,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "Odometry":
        return cls(*cls.FORMAT.unpack_from(payload))


@dataclass
class RobotIdMsg:
    robot_id: int = 0

    TYPE: ClassVar[MsgType] = MsgType.ROBOT_ID
    FORMAT: ClassVar[struct.Struct] = struct.Struct("<B")

    def pack(self) -> bytes:
        return self.FORMAT.pack(self.robot_id)

    @classmethod
    def unpack(cls, payload: bytes) -> "RobotIdMsg":
        # The CV sends an empty payload as a request; treat that as ID 0.
        return cls(payload[0] if payload else 0)


@dataclass
class Health:
    hp: int = 0

    TYPE: ClassVar[MsgType] = MsgType.HEALTH
    FORMAT: ClassVar[struct.Struct] = struct.Struct("<H")

    def pack(self) -> bytes:
        return self.FORMAT.pack(self.hp)

    @classmethod
    def unpack(cls, payload: bytes) -> "Health":
        return cls(*cls.FORMAT.unpack_from(payload))


@dataclass
class Empty:
    """Payload-less message (ALIVE, ROBOT_ID request, RESTART_DETECTOR)."""

    msg_type: MsgType

    def pack(self) -> bytes:
        return b""


_DECODERS = {
    MsgType.TURRET_AIM_DATA: TurretAimData.unpack,
    MsgType.ODOMETRY: Odometry.unpack,
    MsgType.HEALTH: Health.unpack,
}


def encode(msg, seq: int = 0) -> bytes:
    msg_type = msg.msg_type if isinstance(msg, Empty) else msg.TYPE
    return encode_frame(int(msg_type), msg.pack(), seq)


def decode(frame: Frame):
    """Decode a frame into a message object; unknown types come back as the raw ``Frame``."""
    try:
        msg_type = MsgType(frame.msg_type)
    except ValueError:
        return frame
    if msg_type == MsgType.ROBOT_ID:
        return RobotIdMsg.unpack(frame.payload) if frame.payload else Empty(msg_type)
    if msg_type in (MsgType.ALIVE, MsgType.RESTART_DETECTOR):
        return Empty(msg_type)
    decoder = _DECODERS.get(msg_type)
    return decoder(frame.payload) if decoder else frame
