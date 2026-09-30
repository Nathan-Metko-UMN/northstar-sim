"""Triton2 TRT016S-CC (Sony IMX273) with a 6 mm C-mount lens: the camera the sim imitates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CameraModel:
    width: int = 1440
    height: int = 1080
    pixel_size_m: float = 3.45e-6
    focal_length_m: float = 6.0e-3
    # OpenCV (k1, k2, p1, p2, k3). Zero until the real lens is calibrated.
    dist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # Optional calibrated override: [fx, fy, cx, cy].
    calibrated: list[float] | None = None

    @property
    def fx(self) -> float:
        return self.calibrated[0] if self.calibrated else self.focal_length_m / self.pixel_size_m

    @property
    def fy(self) -> float:
        return self.calibrated[1] if self.calibrated else self.focal_length_m / self.pixel_size_m

    @property
    def cx(self) -> float:
        # OpenCV convention: pixel centers sit on integer coordinates.
        return self.calibrated[2] if self.calibrated else (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return self.calibrated[3] if self.calibrated else (self.height - 1) / 2.0

    @property
    def K(self) -> list[float]:
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / 2 / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / 2 / self.fy))


@dataclass
class LinkModel:
    """GigE link between camera and host: sets how long a frame takes to arrive after exposure."""

    gbps: float = 2.5  # 2.5GBASE-T; the Orin Nano devkit port is 1.0
    efficiency: float = 0.95  # GigE Vision payload efficiency
    readout_us: int = 0  # sensor readout; the IMX273 overlaps readout with transfer

    def transfer_us(self, payload_bytes: int) -> int:
        return int(round(payload_bytes * 8 / (self.gbps * 1e9 * self.efficiency) * 1e6)) + self.readout_us
