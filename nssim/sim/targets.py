"""Enemy robots: plate layout and scripted motion.

Plate layout follows the particle filter's model (fast_plate_orbit): four plates 90 degrees
apart around the center; plates 0 and 2 are the high pair (+z_offset), 1 and 3 the low pair.
So the filter's ``orientation`` is this model's ``spin`` modulo pi. Radii are horizontal
distances from the center to each plate's light-bar center, which is what PnP measures.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field

import numpy as np

from .geometry import Transform, rot_y, rot_z

# Panel mesh axes (x width, y up, z out) -> plate frame axes (x outward normal, y, z up).
R_PLATE_FROM_MESH = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class TargetSpec:
    name: str = "enemy"
    color: str = "blue"  # light bar color
    panel: str = "infantry"  # infantry | sentry | hero
    number: str = "3"  # what the classifier should read
    radius_high: float = 0.25  # plates 0 and 2
    radius_low: float = 0.25  # plates 1 and 3
    z_offset: float = 0.03  # high pair sits this far above the center, low pair below
    center_height: float = 0.20  # plate-ring center above the floor
    tilt_deg: float = 15.0  # plate normals point up by this much


def plate_transforms(spec: TargetSpec, center: np.ndarray, spin: float) -> list[Transform]:
    """World transforms of the four plate frames (origin at the light-bar center)."""
    tilt = rot_y(-np.radians(spec.tilt_deg))  # rotates the outward normal upward
    plates = []
    for i in range(4):
        radius = spec.radius_high if i % 2 == 0 else spec.radius_low
        height = spec.z_offset if i % 2 == 0 else -spec.z_offset
        T = (
            Transform.translation(center)
            @ Transform.rotation(rot_z(spin + i * np.pi / 2))
            @ Transform.translation([radius, 0.0, height])
            @ Transform.rotation(tilt)
        )
        plates.append(T)
    return plates


def light_bar_corners(plate: Transform, half_spacing: float, half_height: float) -> np.ndarray:
    """(4, 3) world points: both ends of both light bars (unordered)."""
    local = np.array(
        [
            [0.0, half_spacing, -half_height],
            [0.0, half_spacing, half_height],
            [0.0, -half_spacing, half_height],
            [0.0, -half_spacing, -half_height],
        ]
    )
    return plate.apply(local)


@dataclass
class Motion:
    """Scripted target motion: center moves linearly or sinusoidally while spinning at a constant rate.

    center(t) = start + velocity * t + strafe_amplitude * sin(2*pi*strafe_hz*t) * strafe_dir
    spin(t) = spin0 + omega * t
    """

    start: list[float] = field(default_factory=lambda: [0.0, 0.0])  # x, y on the field
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])
    strafe_amplitude: float = 0.0
    strafe_hz: float = 0.0
    strafe_dir: list[float] = field(default_factory=lambda: [0.0, 1.0])
    spin0: float = 0.0
    omega: float = 0.0  # rad/s, CCW positive

    def center_xy(self, t: float) -> np.ndarray:
        xy = np.asarray(self.start, float) + np.asarray(self.velocity, float) * t
        if self.strafe_amplitude:
            d = np.asarray(self.strafe_dir, float)
            xy = xy + self.strafe_amplitude * np.sin(2 * np.pi * self.strafe_hz * t) * d / np.linalg.norm(d)
        return xy

    def velocity_xy(self, t: float) -> np.ndarray:
        v = np.asarray(self.velocity, float)
        if self.strafe_amplitude:
            d = np.asarray(self.strafe_dir, float)
            w = 2 * np.pi * self.strafe_hz
            v = v + self.strafe_amplitude * w * np.cos(w * t) * d / np.linalg.norm(d)
        return v

    def spin(self, t: float) -> float:
        return self.spin0 + self.omega * t


class DrivenMotion:
    """Motion steered live (``nssim drive``), with the same interface as :class:`Motion`.

    The velocity follows the commanded velocity with a first-order lag and the spin rate ramps
    toward the commanded rate, so what the filter sees is something a robot could do. The state is
    "now": ``center_xy(t)`` and friends ignore t and return the current integrated state.
    """

    def __init__(self, start_xy, spin0: float = 0.0, velocity_tau_s: float = 0.25, spin_accel: float = 12.0,
                 bounds=None):
        self.start = np.array(start_xy, dtype=float)
        self.spin0 = float(spin0)
        self.velocity_tau_s = velocity_tau_s
        self.spin_accel = spin_accel  # rad/s^2
        self.bounds = bounds  # ((xmin, ymin), (xmax, ymax)) or None
        self.command_velocity = np.zeros(2)
        self.command_omega = 0.0
        self.reset()

    def reset(self) -> None:
        self.xy = self.start.copy()
        self.v = np.zeros(2)
        self.angle = self.spin0
        self.omega = 0.0

    def step(self, dt: float) -> None:
        self.v += (self.command_velocity - self.v) * (1.0 - math.exp(-dt / self.velocity_tau_s))
        self.xy += self.v * dt
        if self.bounds is not None:
            clipped = np.clip(self.xy, self.bounds[0], self.bounds[1])
            self.v[clipped != self.xy] = 0.0
            self.xy = clipped
        max_change = self.spin_accel * dt
        self.omega += float(np.clip(self.command_omega - self.omega, -max_change, max_change))
        self.angle += self.omega * dt

    def center_xy(self, t: float) -> np.ndarray:
        return self.xy.copy()

    def velocity_xy(self, t: float) -> np.ndarray:
        return self.v.copy()

    def spin(self, t: float) -> float:
        return self.angle


@dataclass
class TargetState:
    center: np.ndarray  # world, plate-ring center
    velocity: np.ndarray  # world xy velocity
    spin: float
    omega: float
    plates: list[Transform]


class Target:
    def __init__(self, spec: TargetSpec, motion: Motion, floor_z: float):
        self.spec = spec
        self.motion = motion
        self.floor_z = floor_z

    def state(self, t: float) -> TargetState:
        xy = self.motion.center_xy(t)
        center = np.array([xy[0], xy[1], self.floor_z + self.spec.center_height])
        spin = self.motion.spin(t)
        return TargetState(
            center=center,
            velocity=self.motion.velocity_xy(t),
            spin=spin,
            omega=self.motion.omega,
            plates=plate_transforms(self.spec, center, spin),
        )
