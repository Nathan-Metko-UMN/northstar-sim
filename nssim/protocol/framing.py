"""DJI serial framing as used by taproot's ``DJISerial`` and Northstar-CV's ``Uart``.

Frame layout (little-endian)::

    0xA5 | data_length u16 | seq u8 | CRC8(bytes 0..3) | msg_type u16 | payload | CRC16(all before)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .crc import crc8, crc16

HEAD_BYTE = 0xA5
HEADER_SIZE = 5
TYPE_SIZE = 2
CRC16_SIZE = 2
OVERHEAD = HEADER_SIZE + TYPE_SIZE + CRC16_SIZE

# Northstar-CV treats anything longer as a misframed header and resyncs (uart.cpp).
MAX_FRAME_SIZE = 512

_HEADER = struct.Struct("<BHB")
_U16 = struct.Struct("<H")


@dataclass(frozen=True)
class Frame:
    msg_type: int
    payload: bytes
    seq: int = 0


def encode_frame(msg_type: int, payload: bytes = b"", seq: int = 0) -> bytes:
    header = _HEADER.pack(HEAD_BYTE, len(payload), seq & 0xFF)
    header += bytes((crc8(header),))
    body = header + _U16.pack(msg_type) + bytes(payload)
    return body + _U16.pack(crc16(body))


def frame_size(payload_len: int) -> int:
    return payload_len + OVERHEAD


@dataclass
class DecoderStats:
    frames: int = 0
    dropped_bytes: int = 0
    crc8_errors: int = 0
    crc16_errors: int = 0
    oversize: int = 0


@dataclass
class FrameDecoder:
    """Streaming frame decoder.

    Garbage, oversize lengths and CRC failures drop only the offending head byte, so a valid
    frame that follows is never lost (the MCB's taproot parser resyncs the same way).
    """

    check_crc: bool = True
    max_frame_size: int = MAX_FRAME_SIZE
    stats: DecoderStats = field(default_factory=DecoderStats)
    _buf: bytearray = field(default_factory=bytearray, repr=False)

    def feed(self, data: bytes | bytearray | memoryview) -> list[Frame]:
        self._buf += data
        buf = self._buf
        frames: list[Frame] = []
        while True:
            start = buf.find(HEAD_BYTE)
            if start < 0:
                self.stats.dropped_bytes += len(buf)
                buf.clear()
                break
            if start:
                self.stats.dropped_bytes += start
                del buf[:start]
            if len(buf) < HEADER_SIZE:
                break
            length = buf[1] | (buf[2] << 8)
            total = length + OVERHEAD
            if total > self.max_frame_size:
                self.stats.oversize += 1
                self._drop_head()
                continue
            if self.check_crc and crc8(buf[:4]) != buf[4]:
                self.stats.crc8_errors += 1
                self._drop_head()
                continue
            if len(buf) < total:
                break
            if self.check_crc:
                expected = buf[total - 2] | (buf[total - 1] << 8)
                if crc16(buf[: total - 2]) != expected:
                    self.stats.crc16_errors += 1
                    self._drop_head()
                    continue
            msg_type = buf[5] | (buf[6] << 8)
            frames.append(Frame(msg_type, bytes(buf[7 : total - 2]), buf[3]))
            self.stats.frames += 1
            del buf[:total]
        return frames

    def _drop_head(self) -> None:
        self.stats.dropped_bytes += 1
        del self._buf[:1]
