"""UART link to Northstar-CV: a TCP server that the CV container's ``socat`` PTY connects to.

Bytes from the virtual MCB are released on the MCB clock at the speed a 115200-baud 8N1 UART
would deliver them, so odometry reaches the CV with realistic serialization delay and queueing.
"""

from __future__ import annotations

import socket
from collections import deque
from dataclasses import dataclass


@dataclass
class _Pending:
    done_us: float  # MCB time when the last byte has arrived at the CV
    data: bytes
    odometry_ts: int | None  # MCB timestamp if this frame is odometry


class UartLink:
    def __init__(self, host: str = "0.0.0.0", port: int = 5760, baud: int = 115_200):
        self._listener = socket.create_server((host, port))
        self.port = self._listener.getsockname()[1]
        self._conn: socket.socket | None = None
        self.byte_us = 10 * 1e6 / baud  # 8N1: start + 8 data + stop bits
        self._line_free_us = 0.0
        self._queue: deque[_Pending] = deque()
        self.last_delivered_odometry_ts: int | None = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def accept(self, timeout: float | None) -> bool:
        self._listener.settimeout(timeout)
        try:
            conn, _ = self._listener.accept()
        except (TimeoutError, socket.timeout):
            return False
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setblocking(False)
        self._conn = conn
        return True

    def queue(self, queued_us: int, data: bytes, odometry_ts: int | None = None) -> float:
        """Put a frame on the wire at MCB time ``queued_us``; returns when it fully arrives."""
        start = max(float(queued_us), self._line_free_us)
        done = start + len(data) * self.byte_us
        self._line_free_us = done
        self._queue.append(_Pending(done, data, odometry_ts))
        return done

    def flush(self, now_us: int) -> None:
        """Send everything that has fully arrived by MCB time ``now_us``."""
        out = bytearray()
        while self._queue and self._queue[0].done_us <= now_us:
            item = self._queue.popleft()
            out += item.data
            if item.odometry_ts is not None:
                self.last_delivered_odometry_ts = item.odometry_ts
        if out and self._conn is not None:
            self._conn.setblocking(True)
            try:
                self._conn.sendall(out)
            finally:
                self._conn.setblocking(False)

    def recv(self) -> bytes:
        """Whatever the CV has written so far (non-blocking)."""
        if self._conn is None:
            return b""
        chunks = []
        while True:
            try:
                chunk = self._conn.recv(65536)
            except (BlockingIOError, InterruptedError):
                break
            if not chunk:
                raise ConnectionError("UART link closed by the CV side")
            chunks.append(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._listener.close()
