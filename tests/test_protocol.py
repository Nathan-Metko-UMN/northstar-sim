import random
import re

import pytest

from nssim.protocol import (
    Empty,
    FrameDecoder,
    Health,
    MsgType,
    Odometry,
    RobotIdMsg,
    TurretAimData,
    crc8,
    crc16,
    decode,
    encode,
    encode_frame,
)
from nssim.protocol.crc import CRC8_TABLE, CRC16_TABLE
from nssim.protocol.framing import MAX_FRAME_SIZE, OVERHEAD


def _bitwise_crc8(data: bytes) -> int:
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
    return crc


def _bitwise_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


def _cpp_table(source: str, name: str) -> list[int]:
    body = re.search(name + r"\s*=\s*\{(.*?)\};", source, re.S).group(1)
    return [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)]


def test_crc_tables_match_northstar_cv(cv_dir):
    source = (cv_dir / "src" / "uart" / "crc" / "crc.cpp").read_text()
    assert list(CRC8_TABLE) == _cpp_table(source, "CRC8_TABLE")
    assert list(CRC16_TABLE) == _cpp_table(source, "CRC16_TABLE")


def test_crc_matches_bitwise_reference():
    rng = random.Random(1)
    for _ in range(200):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
        assert crc8(data) == _bitwise_crc8(data)
        assert crc16(data) == _bitwise_crc16(data)


def test_payload_sizes_match_firmware_structs():
    # vision_comms.hpp: OdometryData is u32 + 2 chassis floats + 4 turret floats (incl. yaw_vel).
    assert len(Odometry().pack()) == 28
    # CV AutoAimMessage::serialize: 3 floats + u16 target id.
    assert len(TurretAimData().pack()) == 14
    assert len(Health().pack()) == 2
    assert len(RobotIdMsg().pack()) == 1


def test_frame_layout():
    frame = encode_frame(MsgType.ODOMETRY, bytes(28), seq=7)
    assert frame[0] == 0xA5
    assert frame[1:3] == (28).to_bytes(2, "little")
    assert frame[3] == 7
    assert frame[4] == crc8(frame[:4])
    assert frame[5:7] == int(MsgType.ODOMETRY).to_bytes(2, "little")
    assert frame[-2:] == crc16(frame[:-2]).to_bytes(2, "little")
    assert len(frame) == 28 + OVERHEAD


@pytest.mark.parametrize(
    "msg",
    [
        TurretAimData(yaw=1.25, pitch=-0.125, distance=4.5, target_id=3),
        Odometry(timestamp_us=123456, vel_x=0.5, pitch=0.25, yaw=6.0, yaw_vel=-2.0),
        Health(hp=400),
        RobotIdMsg(robot_id=3),
        Empty(MsgType.ALIVE),
        Empty(MsgType.RESTART_DETECTOR),
    ],
)
def test_roundtrip(msg):
    frames = FrameDecoder().feed(encode(msg, seq=3))
    assert len(frames) == 1
    assert decode(frames[0]) == msg


def test_empty_robot_id_is_a_request():
    [frame] = FrameDecoder().feed(encode(Empty(MsgType.ROBOT_ID)))
    assert decode(frame) == Empty(MsgType.ROBOT_ID)


def test_odometry_timestamp_wraps_to_u32():
    msg = Odometry(timestamp_us=(1 << 32) + 5)
    assert Odometry.unpack(msg.pack()).timestamp_us == 5


def test_decoder_handles_split_and_garbage():
    stream = b"\x00\x13\xa5" + encode(Health(hp=1)) + b"\xff" + encode(TurretAimData(1, 2, 3, 4))
    decoder = FrameDecoder()
    frames = []
    for i in range(len(stream)):
        frames += decoder.feed(stream[i : i + 1])
    assert [decode(f) for f in frames] == [Health(hp=1), TurretAimData(1, 2, 3, 4)]


def test_decoder_resyncs_after_corruption():
    good = encode(Health(hp=7))
    bad = bytearray(encode(Health(hp=9)))
    bad[-1] ^= 0xFF  # break CRC16
    decoder = FrameDecoder()
    frames = decoder.feed(bytes(bad) + good)
    assert [decode(f) for f in frames] == [Health(hp=7)]
    assert decoder.stats.crc16_errors == 1


def test_decoder_rejects_oversize_length():
    # A head byte followed by an implausible length must not stall the stream (uart.cpp rule).
    length = 0x01FF
    assert length + OVERHEAD > MAX_FRAME_SIZE
    bogus = bytes([0xA5]) + length.to_bytes(2, "little") + b"\x00"
    bogus += bytes([crc8(bogus)])
    assert bogus.count(0xA5) == 1
    decoder = FrameDecoder()
    frames = decoder.feed(bogus + encode(Health(hp=3)))
    assert [decode(f) for f in frames] == [Health(hp=3)]
    assert decoder.stats.oversize == 1
