"""Turret dynamics driven by the virtual MCB's setpoints.

Placeholder until the firmware's world-frame turret-IMU cascade PID
(``WorldFrameYawTurretImuCascadePidTurretController``) is ported: each axis tracks its setpoint
as a rate- and acceleration-limited second-order system. ``hold`` ignores setpoints and
``ideal`` snaps to them, for isolating perception from control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class TurretConfig:
    mode: str = "second_order"  # hold | ideal | second_order
    natural_hz: float = 8.0
    damping: float = 0.9
    max_rate: float = 12.0  # rad/s
    max_accel: float = 150.0  # rad/s^2


class _Axis:
    def __init__(self, angle: float, config: TurretConfig, wrap: bool):
        self.angle = angle
        self.rate = 0.0
        self.config = config
        self.wrap = wrap

    def step(self, dt: float, setpoint: float) -> None:
        c = self.config
        error = _wrap(setpoint - self.angle) if self.wrap else setpoint - self.angle
        wn = 2 * math.pi * c.natural_hz
        accel = wn * wn * error - 2 * c.damping * wn * self.rate
        accel = max(-c.max_accel, min(c.max_accel, accel))
        self.rate = max(-c.max_rate, min(c.max_rate, self.rate + accel * dt))
        self.angle += self.rate * dt


class TurretModel:
    """World-frame turret. yaw is continuous (CCW positive), pitch is positive looking down."""

    def __init__(self, yaw: float, pitch: float, config: TurretConfig | None = None):
        self.config = config or TurretConfig()
        self._yaw = _Axis(yaw, self.config, wrap=True)
        self._pitch = _Axis(pitch, self.config, wrap=False)

    @property
    def yaw(self) -> float:
        return self._yaw.angle

    @property
    def pitch(self) -> float:
        return self._pitch.angle

    @property
    def yaw_rate(self) -> float:
        return self._yaw.rate

    def step(self, dt: float, setpoint: tuple[float, float] | None) -> None:
        if setpoint is None or self.config.mode == "hold":
            self._yaw.rate = self._pitch.rate = 0.0
            return
        yaw_sp, pitch_sp = setpoint
        if self.config.mode == "ideal":
            self._yaw.angle += _wrap(yaw_sp - self._yaw.angle)
            self._pitch.angle = pitch_sp
            self._yaw.rate = self._pitch.rate = 0.0
            return
        self._yaw.step(dt, yaw_sp)
        self._pitch.step(dt, pitch_sp)
