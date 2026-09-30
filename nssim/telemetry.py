"""Receives Northstar-CV's UDP telemetry (``--telemetry host:port``): one JSON object per datagram."""

from __future__ import annotations

import json
import socket


class TelemetryReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 5800):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self.port = self._sock.getsockname()[1]
        self.bad_datagrams = 0

    def poll(self) -> list[dict]:
        messages = []
        while True:
            try:
                data = self._sock.recv(65536)
            except (BlockingIOError, InterruptedError):
                break
            except ConnectionResetError:  # Windows reports ICMP port-unreachable here
                continue
            try:
                messages.append(json.loads(data))
            except json.JSONDecodeError:
                self.bad_datagrams += 1
        return messages

    def close(self) -> None:
        self._sock.close()
