"""Built-in scenarios. Each returns a RunConfig; override fields afterwards as needed."""

from __future__ import annotations

import math

import numpy as np

from nssim.mcb.turret import TurretConfig
from nssim.runner import RunConfig, TargetConfig
from nssim.sim.targets import Motion, ProfileMotion, TargetSpec


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


# Recordings for replay (tools/replay.py): what the camera sees mustn't depend on the tracker, so
# the turret holds its first aim or follows a script, never the CV's aim. Robots are ovals (the
# pairs at different radii) with different plate height offsets, as real robots are.

REC_ROBOTS = {
    # name: (panel, number, radius of plates 0/2 (the high pair), radius of 1/3, height offset)
    "oval3": ("infantry", "3", 0.22, 0.32, 0.03),
    "oval3b": ("infantry", "3", 0.32, 0.22, 0.015),
    "round3": ("infantry", "3", 0.27, 0.27, 0.0),
    "hero": ("hero", "1", 0.30, 0.36, 0.05),
    "sentry": ("sentry", "guard", 0.28, 0.28, 0.0),
    "drive_oval": ("infantry", "3", 0.25, 0.40, 0.03),  # nssim drive's infantry
    "drive_oval_b": ("infantry", "3", 0.36, 0.24, 0.015),
}


def _rec_spec(robot: str, name: str | None = None) -> TargetSpec:
    panel, number, r_high, r_low, dz = REC_ROBOTS[robot]
    return TargetSpec(name=name or robot, panel=panel, number=number, radius_high=r_high, radius_low=r_low, z_offset=dz)


def _rec(name: str, duration_s: float, targets: list[TargetConfig], aim_script=None) -> RunConfig:
    return RunConfig(
        name=name,
        duration_s=duration_s,
        shooter_base_xy=[-1.5, 0.0],
        initial_aim=targets[0].spec.name,
        turret=TurretConfig(mode="second_order" if aim_script else "hold"),
        targets=targets,
        aim_script=aim_script or [],
    )


def _rec_spin(robot: str, omega: float, duration_s: float = 20.0, name: str | None = None, **motion) -> RunConfig:
    return _rec(name or f"rec_{robot}", duration_s, [TargetConfig(_rec_spec(robot), Motion(start=[1.5, 0.0], omega=omega, **motion))])


def rec_switch(duration_s: float = 26.0) -> RunConfig:
    """Three different robots side by side, all spinning; the turret looks at one, then another,
    and back, so the tracker drops one robot and picks up another (and meets each again)."""
    lineup = [("oval3", [1.5, -1.0], 6.0), ("hero", [1.7, 0.1], -4.0), ("sentry", [1.5, 1.1], 5.0)]
    targets = [TargetConfig(_rec_spec(robot), Motion(start=xy, omega=omega)) for robot, xy, omega in lineup]
    script = [(0.0, "oval3"), (5.0, "hero"), (9.0, "oval3"), (13.0, "sentry"), (17.0, "hero"), (21.0, "oval3")]
    return _rec("rec_switch", duration_s, targets, script)


# Like a driven session: still, slow spin, faster, full spin, stop, spin again, with some driving.
DRIVE_PROFILE = [
    [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.5], [5.0, 0.0, 0.0, -0.5], [7.0, 0.0, 0.0, 0.0],
    [9.0, 1.1, 0.0, 0.0], [19.0, 2.25, 0.0, 0.0], [26.0, 4.5, 0.0, 0.0], [38.0, 0.0, 0.0, 0.0],
    [44.0, 4.5, 0.0, 0.0], [52.0, 4.5, 0.0, 0.5], [54.0, 4.5, 0.0, -0.5], [56.0, 4.5, 0.0, 0.0],
    [60.0, 1.1, 0.0, 0.0], [66.0, 0.0, 0.0, 0.0],
]


def rec_drive(robot: str = "drive_oval", fps: float = 50.0, duration_s: float = 70.0) -> RunConfig:
    """A driven session (DRIVE_PROFILE) at nssim drive's frame rate, turret holding still."""
    spec = _rec_spec(robot)
    cfg = _rec(f"rec_drive_{robot}", duration_s, [TargetConfig(spec, ProfileMotion(start=[1.5, 0.0], segments=DRIVE_PROFILE, duration_s=duration_s + 1))])
    cfg.fps = fps
    return cfg


# --- Tracker test (nssim pftest): a robot driving around while it changes how it spins ---

# Where we stand for the tracker test, and the clear line of sight across the field from there
# (between the side box, the barriers and the ramps): the robot under test drives along it.
PFTEST_SHOOTER = (-2.5, -0.8)
_ALONG = np.array([4.4, 3.8]) / np.hypot(4.4, 3.8)
_ACROSS = np.array([-_ALONG[1], _ALONG[0]])

# [t_start s, spin rad/s, speed away from us m/s, speed across m/s]: fast, slow, the other way and
# stopped, while driving out to ~5.5 m, across, and back in to ~2 m. Starts 3 m away.
TOUR_LOCAL = [
    [0.0, 6.0, 0.0, 0.0], [4.0, 6.0, 0.5, 0.0], [7.0, 1.5, 0.4, 0.0], [9.0, 1.5, 0.0, 0.3],
    [11.0, -4.0, 0.0, -0.3], [14.0, -4.0, -0.8, 0.0], [17.0, 0.0, -0.4, 0.0], [19.0, 0.0, 0.0, 0.0],
    [21.0, 3.0, 0.0, 0.3], [24.0, 6.0, 0.0, -0.3], [27.0, -6.0, 0.5, 0.15], [31.0, -1.5, 0.4, 0.1],
    [34.0, 4.5, -0.6, -0.2], [38.0, 0.0, -0.3, 0.1], [40.0, 6.0, 0.0, 0.0], [43.0, 0.0, 0.0, 0.0],
]
TOUR = [[t, w, *(along * _ALONG + across * _ACROSS)] for t, w, along, across in TOUR_LOCAL]
TOUR_START = (np.asarray(PFTEST_SHOOTER) + 3.0 * _ALONG).tolist()
TOUR_S = 45.0

# Plate layouts every robot type is tested with: (radius of plates 0/2 (the high pair), of 1/3,
# height offset). Hero plates sit a little further out.
PFTEST_GEOMETRIES = {
    "oval_offset": (0.24, 0.34, 0.03),
    "round_offset": (0.29, 0.29, 0.03),
    "oval_flat": (0.24, 0.34, 0.0),
    "round_flat": (0.29, 0.29, 0.0),
}
PFTEST_ROBOTS = {"infantry": ("infantry", "3", 0.0), "hero": ("hero", "1", 0.04), "sentry": ("sentry", "guard", 0.0)}


def _scaled_tour(scale: float, shift_s: float) -> list[list[float]]:
    """The tour with smaller moves, started shift_s later (for the robots we're not tracking)."""
    tour = [[t, w, vx * scale, vy * scale] for t, w, vx, vy in TOUR]
    shifted = [[(t + shift_s) % TOUR_S, w, vx, vy] for t, w, vx, vy in tour]
    return sorted(shifted)


def pftest(robot: str = "infantry", geometry: str = "oval_offset", others: bool = True, follow: str = "truth",
           fps: float = 1e6 / 6000) -> RunConfig:
    """The tracker test: `robot` with `geometry` drives TOUR; with `others`, the two other robot
    types drive smaller loops beside it (their own default layouts). The turret follows the robot
    under test (follow="truth": what the camera sees doesn't depend on the tracker) or the CV's aim
    (follow="cv")."""
    panel, number, extra = PFTEST_ROBOTS[robot]
    r_high, r_low, dz = PFTEST_GEOMETRIES[geometry]
    spec = TargetSpec(name=robot, panel=panel, number=number, radius_high=r_high + extra, radius_low=r_low + extra, z_offset=dz)
    targets = [TargetConfig(spec, ProfileMotion(start=TOUR_START, segments=TOUR, duration_s=TOUR_S + 1))]
    if others:
        spots = [[0.9, -1.4], [1.2, 0.3]]
        for i, other in enumerate(k for k in PFTEST_ROBOTS if k != robot):
            o_panel, o_number, o_extra = PFTEST_ROBOTS[other]
            o_spec = TargetSpec(name=other, panel=o_panel, number=o_number, radius_high=0.26 + o_extra,
                                radius_low=0.30 + o_extra, z_offset=0.02)
            targets.append(TargetConfig(o_spec, ProfileMotion(start=spots[i], segments=_scaled_tour(0.2, 11.0 * (i + 1)),
                                                              duration_s=TOUR_S + 1)))
    cfg = RunConfig(
        name=f"pftest_{robot}_{geometry}",
        duration_s=TOUR_S,
        fps=fps,
        shooter_base_xy=list(PFTEST_SHOOTER),
        initial_aim=robot,
        turret=TurretConfig(mode="second_order"),
        targets=targets,
        aim_script=[(0.0, robot)] if follow == "truth" else [],
    )
    return cfg


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
    "rec_oval3": lambda: _rec_spin("oval3", 6.0),
    "rec_oval3b": lambda: _rec_spin("oval3b", -5.0),
    "rec_round3": lambda: _rec_spin("round3", 6.0),
    "rec_hero": lambda: _rec_spin("hero", 4.0),
    "rec_sentry": lambda: _rec_spin("sentry", 6.0),
    "rec_strafe_oval3": lambda: _rec_spin("oval3", 4.0, name="rec_strafe_oval3", strafe_amplitude=0.6, strafe_hz=0.5),
    "rec_strafe_hero": lambda: _rec_spin("hero", -3.0, name="rec_strafe_hero", strafe_amplitude=0.5, strafe_hz=0.4),
    "rec_spinup_oval3": lambda: _rec_spin("oval3", 6.0, duration_s=14.0, name="rec_spinup_oval3", spin_start_s=4.0),
    "rec_switch": rec_switch,
    "rec_drive": rec_drive,
    "rec_drive_b": lambda: rec_drive("drive_oval_b"),
    "rec_drive_166": lambda: rec_drive(fps=1e6 / 6000),
    **{f"pftest_{robot}_{geometry}": (lambda robot=robot, geometry=geometry: pftest(robot, geometry))
       for robot in PFTEST_ROBOTS for geometry in PFTEST_GEOMETRIES},
}
