"""Frame stream between the Triton2 twin (server) and Northstar-CV's SimCamera (client).

Every message is ``magic[4] | body_length u32 | body`` (little-endian):

- ``NSHL`` hello, server -> client on connect: UTF-8 JSON describing the camera and run mode.
- ``NSFR`` frame, server -> client: ``FRAME_HEADER`` then ``image_bytes`` of Bayer data
  (``width*height`` bytes, or zlib-compressed if ``FLAG_ZLIB``), then (if ``FLAG_ORACLE``)
  ``json_length u32 | JSON`` with ground-truth plate corners.
- ``NSAK`` ack, client -> server (paced mode): ``seq u64 | processing_us u32``, sent when the
  client asks for the next frame, i.e. after Northstar-CV has finished with ``seq``.

``capture_mcb_us`` is the end of exposure on the MCB clock; ``arrival_delay_us`` is how long
after that the frame would reach the Jetson (readout + GigE transfer). In paced mode
``odom_watermark_mcb_us`` is the timestamp of the last odometry message sent before this frame;
SimCamera holds the frame until Northstar-CV's odometry buffer has caught up to it, because the
UART and the frame stream are separate connections.
"""

from __future__ import annotations

import json
import socket
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

MAGIC_HELLO = b"NSHL"
MAGIC_FRAME = b"NSFR"
MAGIC_ACK = b"NSAK"
PROTOCOL_VERSION = 1

PIXEL_BAYER_RG8 = 1
FLAG_ORACLE = 1
FLAG_ZLIB = 2  # image bytes are zlib-compressed (lossless); used where the link is slow

_PREFIX = struct.Struct("<4sI")
# seq, capture_mcb_us, odom_watermark_mcb_us, exposure_us, arrival_delay_us, width, height,
# pixel_format, flags, reserved, image_bytes
FRAME_HEADER = struct.Struct("<QqqIIHHBBHI")
_ACK = struct.Struct("<QI")
_U32 = struct.Struct("<I")


@dataclass
class FrameHeader:
    seq: int
    capture_mcb_us: int
    exposure_us: int
    arrival_delay_us: int
    width: int
    height: int
    pixel_format: int = PIXEL_BAYER_RG8
    flags: int = 0
    odom_watermark_mcb_us: int = -1  # -1: don't wait for odometry
    image_bytes: int = 0  # filled in by pack_frame

    def pack(self) -> bytes:
        return FRAME_HEADER.pack(
            self.seq,
            self.capture_mcb_us,
            self.odom_watermark_mcb_us,
            self.exposure_us,
            self.arrival_delay_us,
            self.width,
            self.height,
            self.pixel_format,
            self.flags,
            0,
            self.image_bytes,
        )

    @classmethod
    def unpack(cls, body: bytes) -> "FrameHeader":
        seq, capture, watermark, exposure, arrival, width, height, fmt, flags, _, image_bytes = (
            FRAME_HEADER.unpack_from(body)
        )
        return cls(seq, capture, exposure, arrival, width, height, fmt, flags, watermark, image_bytes)


def pack_message(magic: bytes, body: bytes) -> bytes:
    return _PREFIX.pack(magic, len(body)) + body


def pack_frame(
    header: FrameHeader,
    image: bytes | memoryview,
    oracle: dict | None = None,
    compress: bool = False,
) -> bytes:
    if compress:
        data = deflate_parallel(image)
        header.flags |= FLAG_ZLIB
    else:
        data = bytes(image)
    if oracle is not None:
        header.flags |= FLAG_ORACLE
    header.image_bytes = len(data)
    parts = [header.pack(), data]
    if oracle is not None:
        payload = json.dumps(oracle, separators=(",", ":")).encode()
        parts += [_U32.pack(len(payload)), payload]
    return pack_message(MAGIC_FRAME, b"".join(parts))


_DEFLATE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="deflate")


def deflate_parallel(data, chunks: int = 4, level: int = 1) -> bytes:
    """One standard zlib stream, compressed on several threads (the way pigz does it).

    Each chunk is raw deflate on its own, ended with a full flush so it stops byte-aligned and
    unfinished; concatenated behind a zlib header, with the adler32 of all the data at the end, they
    are an ordinary zlib stream that ``uncompress()`` reads in one go.
    """
    view = memoryview(data).cast("B")
    step = -(-len(view) // chunks)
    pieces = [view[i:i + step] for i in range(0, len(view), step)] or [view]

    def deflate(index: int) -> bytes:
        c = zlib.compressobj(level, zlib.DEFLATED, -15)
        last = index == len(pieces) - 1
        return c.compress(pieces[index]) + c.flush(zlib.Z_FINISH if last else zlib.Z_FULL_FLUSH)

    body = b"".join(_DEFLATE_POOL.map(deflate, range(len(pieces))))
    return bytes([0x78, 0x01]) + body + zlib.adler32(view).to_bytes(4, "big")  # zlib header: deflate, fastest


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("frame stream closed")
        got += k
    return bytes(buf)


def read_message(sock: socket.socket) -> tuple[bytes, bytes]:
    magic, length = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    return magic, _recv_exact(sock, length)


def parse_frame(body: bytes) -> tuple[FrameHeader, bytes, dict | None]:
    header = FrameHeader.unpack(body)
    start = FRAME_HEADER.size
    end = start + header.image_bytes
    image = body[start:end]
    if header.flags & FLAG_ZLIB:
        image = zlib.decompress(image)
    oracle = None
    if header.flags & FLAG_ORACLE:
        (length,) = _U32.unpack_from(body, end)
        oracle = json.loads(body[end + 4 : end + 4 + length])
    return header, image, oracle


class FrameServer:
    """Serves one SimCamera client at a time."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5600):
        self._listener = socket.create_server((host, port))
        self.port = self._listener.getsockname()[1]
        self._conn: socket.socket | None = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def accept(self, hello: dict, timeout: float | None = None) -> None:
        self._listener.settimeout(timeout)
        conn, _ = self._listener.accept()
        conn.settimeout(None)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = {"protocol": PROTOCOL_VERSION, **hello}
        conn.sendall(pack_message(MAGIC_HELLO, json.dumps(hello).encode()))
        self._conn = conn

    def send_frame(
        self,
        header: FrameHeader,
        image: bytes | memoryview,
        oracle: dict | None = None,
        compress: bool = False,
    ) -> None:
        self.send_packed(pack_frame(header, image, oracle, compress))

    def send_packed(self, message: bytes) -> None:
        """Send a frame message already built with pack_frame()."""
        self._conn.sendall(message)

    def recv_ack(self, timeout: float | None = None) -> tuple[int, int]:
        """(seq, processing_us) of the next ack from the client."""
        self._conn.settimeout(timeout)
        try:
            magic, body = read_message(self._conn)
        finally:
            self._conn.settimeout(None)
        if magic != MAGIC_ACK:
            raise ConnectionError(f"expected ack, got {magic!r}")
        return _ACK.unpack(body)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._listener.close()


class FrameClient:
    """Python stand-in for SimCamera (tests and debugging)."""

    def __init__(self, host: str, port: int, timeout: float | None = 5.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        magic, body = read_message(self._sock)
        if magic != MAGIC_HELLO:
            raise ConnectionError(f"expected hello, got {magic!r}")
        self.hello = json.loads(body)

    def read_frame(self) -> tuple[FrameHeader, bytes, dict | None]:
        magic, body = read_message(self._sock)
        if magic != MAGIC_FRAME:
            raise ConnectionError(f"expected frame, got {magic!r}")
        return parse_frame(body)

    def ack(self, seq: int, processing_us: int) -> None:
        self._sock.sendall(pack_message(MAGIC_ACK, _ACK.pack(seq, processing_us)))

    def close(self) -> None:
        self._sock.close()
