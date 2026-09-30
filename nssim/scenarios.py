"""Built-in scenarios. Each returns a RunConfig; override fields afterwards as needed."""

from __future__ import annotations

import math

from nssim.mcb.turret import TurretConfig
from nssim.runner import RunConfig, TargetConfig
from nssim.sim.targets import Motion, TargetSpec


def static_plate(distance: float = 2.5) -> RunConfig:
    """Frame check: a still target in front of a turret that doesn't move."""
    return RunConfig(
        name="static_plate",
        duration_s=3.0,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="hold"),
        targets=[TargetConfig(TargetSpec(), Motion(start=[-1.5 + distance, 0.0], spin0=math.radians(20)))],
    )


def spinning(omega: float = 6.0, distance: float = 3.0, duration_s: float = 6.0) -> RunConfig:
    """Target spinning in place; the turret follows the CV's aim."""
    return RunConfig(
        name=f"spin_{omega:g}",
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="second_order"),
        targets=[TargetConfig(TargetSpec(), Motion(start=[-1.5 + distance, 0.0], omega=omega))],
    )


def strafing(omega: float = 4.0, amplitude: float = 0.6, hz: float = 0.5) -> RunConfig:
    """Spinning while strafing side to side."""
    return RunConfig(
        name=f"strafe_{omega:g}",
        duration_s=8.0,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="second_order"),
        targets=[
            TargetConfig(
                TargetSpec(),
                Motion(start=[1.5, 0.0], omega=omega, strafe_amplitude=amplitude, strafe_hz=hz),
            )
        ],
    )


SCENARIOS = {
    "static_plate": static_plate,
    "spinning": spinning,
    "strafing": strafing,
}
