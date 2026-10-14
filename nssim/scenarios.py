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


def _drive_lineup() -> list[tuple[TargetSpec, tuple[float, float]]]:
    """Opponents like TR's, which carry infantry, hero or sentry panels: those are the stickers TR
    modeled (the digit 3, the digit 1 on large plates, the sentry icon). Offsets from the first."""
    return [
        (TargetSpec(name="infantry", panel="infantry", number="3", radius_high=0.25, radius_low=0.4), (0.0, 0.0)),
        (TargetSpec(name="hero", panel="hero", number="1", radius_high=0.3, radius_low=0.3), (0.9, -1.3)),
        (TargetSpec(name="sentry", panel="sentry", number="guard"), (0.6, 1.3)),
    ]


def drive(distance: float = 3.0, fps: float = 50.0, enemies: int = 1) -> RunConfig:
    """``nssim drive``: open-ended; up to three enemies start still in front of us and are steered live.

    50 fps by default, about what the paced loop manages in real time on a desktop with Northstar-CV
    in Docker Desktop; at the camera's 166 fps it runs in slow motion.
    """
    lineup = _drive_lineup()
    if not 1 <= enemies <= len(lineup):
        raise ValueError(f"enemies must be 1 to {len(lineup)}")
    x0 = -1.5 + distance
    targets = [TargetConfig(spec, Motion(start=[x0 + dx, dy])) for spec, (dx, dy) in lineup[:enemies]]
    return RunConfig(
        name="drive",
        duration_s=math.inf,
        fps=fps,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim=targets[0].spec.name,
        turret=TurretConfig(mode="second_order"),
        targets=targets,
    )


# Particle-filter tests (plate height offset and per-pair radii). Every one is an infantry with a
# 0.03 m height offset and radii 0.25 / 0.35 m unless --plate-height / --radius / --radius-low say
# otherwise; the filter's radius prior is 0.30.


def _pf_target(**motion) -> TargetConfig:
    return TargetConfig(TargetSpec(radius_high=0.25, radius_low=0.35, z_offset=0.03), Motion(start=[1.5, 0.0], **motion))


def still_long(duration_s: float = 15.0) -> RunConfig:
    """A still robot showing one plate square on: nothing tells the filter which pair is high or
    what the radius is, so this shows what it settles on without that (and whether it drifts)."""
    return RunConfig(
        name="still_long",
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="hold"),
        targets=[_pf_target(spin0=0.0)],
    )


def spin_up(omega: float = 6.0, start_s: float = 4.0, duration_s: float = 12.0) -> RunConfig:
    """Still with one plate in view, then spinning up: the radius and height offset the filter held
    while it couldn't see them, then how fast it finds them."""
    return RunConfig(
        name="spin_up",
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="second_order"),
        targets=[_pf_target(spin0=0.0, omega=omega, spin_start_s=start_s)],
    )


def two_plate_start(omega: float = 6.0, duration_s: float = 10.0) -> RunConfig:
    """Spinning, starting turned 45 degrees so the first frame shows two plates: the filter starts
    from their heights, so it can start with the pairs the wrong way round."""
    return RunConfig(
        name="two_plate_start",
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="second_order"),
        targets=[_pf_target(spin0=math.pi / 4, omega=omega)],
    )


def height_flip(omega: float = 6.0, flip_s: float = 5.0, duration_s: float = 15.0) -> RunConfig:
    """Spinning; at flip_s the pairs trade heights. Not something a robot does: it tests whether the
    filter can change which pair it calls high once it has settled."""
    cfg = RunConfig(
        name="height_flip",
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="second_order"),
        targets=[_pf_target(omega=omega)],
    )
    cfg.targets[0].spec.z_offset_flip_s = flip_s
    return cfg


def spin_long(omega: float = 6.0, duration_s: float = 10.0) -> RunConfig:
    """Spinning in place from the start: steady-state height offset and radii."""
    return RunConfig(
        name="spin_long",
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="second_order"),
        targets=[_pf_target(omega=omega)],
    )


def strafe_long(omega: float = 4.0, duration_s: float = 10.0) -> RunConfig:
    """Spinning while strafing, with the test robot's unequal radii."""
    return RunConfig(
        name="strafe_long",
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim="enemy",
        turret=TurretConfig(mode="second_order"),
        targets=[_pf_target(omega=omega, strafe_amplitude=0.6, strafe_hz=0.5)],
    )


SCENARIOS = {
    "static_plate": static_plate,
    "spinning": spinning,
    "strafing": strafing,
    "still_long": still_long,
    "spin_up": spin_up,
    "two_plate_start": two_plate_start,
    "height_flip": height_flip,
    "spin_long": spin_long,
    "strafe_long": strafe_long,
}
