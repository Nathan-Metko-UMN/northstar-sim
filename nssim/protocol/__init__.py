"""DJI-framed UART protocol spoken between the MCB and Northstar-CV."""

from .crc import crc8, crc16
from .framing import Frame, FrameDecoder, encode_frame
from .messages import (
    Empty,
    Health,
    MsgType,
    Odometry,
    RobotId,
    RobotIdMsg,
    TurretAimData,
    decode,
    describe,
    encode,
)

__all__ = [
    "Empty",
    "Frame",
    "FrameDecoder",
    "Health",
    "MsgType",
    "Odometry",
    "RobotId",
    "RobotIdMsg",
    "TurretAimData",
    "crc8",
    "crc16",
    "decode",
    "describe",
    "encode",
    "encode_frame",
]
