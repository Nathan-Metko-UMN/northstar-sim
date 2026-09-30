"""The run's event stream: everything that crosses between the sim and Northstar-CV, as it happens.

An event is a plain dict ``{"topic": str, "t": float, "data": dict}``: ``t`` is seconds on the
source's clock (the MCB clock in the sim; Northstar-CV's own clock when its telemetry comes straight
from the robot). Topics:

- ``run``: what is running (scenario, frame rate, the enemies)
- ``uart``: one serial message, ``{"dir": "mcb_to_cv" | "cv_to_mcb", "type", "seq", "fields", "bytes"}``
  (the same shape Northstar-CV will use to report its own UART traffic from the robot)
- ``frame``: a camera frame's timing: capture, arrival, processing, when its aim reached the MCB
- ``truth``: where the robots really are (sim only)
- ``metrics``: Northstar-CV's errors on a frame against the truth (sim only; see metrics.py)
- ``cv.det``, ``cv.track``, ...: Northstar-CV's telemetry, as it sent it

Sinks must return quickly: the paced loop calls them inline.
"""

from __future__ import annotations

from typing import Callable

Sink = Callable[[dict], None]


class Bus:
    def __init__(self):
        self._sinks: list[Sink] = []

    @property
    def active(self) -> bool:
        """Anyone listening? (Producers skip work that only listeners need.)"""
        return bool(self._sinks)

    def subscribe(self, sink: Sink) -> None:
        self._sinks.append(sink)

    def publish(self, topic: str, t: float, data: dict) -> None:
        if not self._sinks:
            return
        event = {"topic": topic, "t": t, "data": data}
        for sink in self._sinks:
            sink(event)
