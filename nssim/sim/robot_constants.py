"""Our robot's turret geometry, read from Northstar-CV's ``src/constants.hpp``.

Reading it (instead of copying numbers) keeps the simulated robot identical to what the CV's
kinematics assume, so any position error the harness reports comes from the pipeline itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Standard values as of Northstar-CV master, used when constants.hpp can't be read.
_STANDARD_DEFAULTS = {
    "YAW_OFFSET": [0.0, 0.0, 0.2735],
    "PITCH_OFFSET": [0.0, 0.0, 0.2087250],
    "CAMERA_OFFSET": [0.1450980, 0.0, 0.0855],
    "LAUNCH_OFFSET": [0.0728, 0.0, 0.0],
    "YAW_AIM_OFFSET": -0.0385,
    "PITCH_AIM_OFFSET": 0.0,
    "PROJECTILE_VELOCITY": 24.0,
    "FIRE_LATENCY_S": 0.050,
}


@dataclass
class RobotConstants:
    name: str
    yaw_offset: np.ndarray  # base -> yaw pivot
    pitch_offset: np.ndarray  # yaw pivot -> pitch pivot (after yaw rotation)
    camera_offset: np.ndarray  # pitch pivot -> camera (after pitch rotation)
    launch_offset: np.ndarray  # pitch pivot -> muzzle
    yaw_aim_offset: float
    pitch_aim_offset: float
    projectile_velocity: float
    fire_latency_s: float


def _parse_namespace(source: str, robot: str) -> dict:
    match = re.search(r"namespace\s+" + robot + r"\s*\{(.*?)\}\s*//\s*namespace\s+" + robot, source, re.S)
    if not match:
        raise ValueError(f"namespace {robot} not found")
    body = match.group(1)
    values = {}
    for name, args in re.findall(r"Eigen::Vector3f\s+(\w+)\s*\{([^}]*)\}", body):
        values[name] = [float(v.strip().rstrip("fF")) for v in args.split(",")]
    for name, val in re.findall(r"float\s+(\w+)\s*=\s*([-+0-9.eE]+)f?\s*;", body):
        values[name] = float(val)
    return values


def load_plate_dims(cv_dir: Path | None) -> dict[str, tuple[float, float]]:
    """Armor plate size the CV's PnP assumes (light-bar spacing, light-bar length) in meters.

    Parsed from ``src/pnp_solver.hpp``. The sim scales TR's panel meshes to these, so a range
    error in a run is the pipeline's, not a mismatch between two plate models.
    """
    dims = {"small": (0.132, 0.057), "large": (0.223, 0.057)}
    path = None if cv_dir is None else Path(cv_dir) / "src" / "pnp_solver.hpp"
    if path is not None and path.is_file():
        values = dict(re.findall(r"(\w+_ARMOR_(?:WIDTH|HEIGHT))\s*=\s*([0-9.]+)", path.read_text()))
        for kind in ("small", "large"):
            w, h = values.get(f"{kind.upper()}_ARMOR_WIDTH"), values.get(f"{kind.upper()}_ARMOR_HEIGHT")
            if w and h:
                dims[kind] = (float(w) / 1000.0, float(h) / 1000.0)
    return dims


def load_robot_constants(cv_dir: Path | None, robot: str = "Standard") -> RobotConstants:
    values = dict(_STANDARD_DEFAULTS)
    source_path = None if cv_dir is None else Path(cv_dir) / "src" / "constants.hpp"
    if source_path is not None and source_path.is_file():
        values.update(_parse_namespace(source_path.read_text(), robot))
    return RobotConstants(
        name=robot,
        yaw_offset=np.array(values["YAW_OFFSET"]),
        pitch_offset=np.array(values["PITCH_OFFSET"]),
        camera_offset=np.array(values["CAMERA_OFFSET"]),
        launch_offset=np.array(values["LAUNCH_OFFSET"]),
        yaw_aim_offset=values["YAW_AIM_OFFSET"],
        pitch_aim_offset=values["PITCH_AIM_OFFSET"],
        projectile_velocity=values["PROJECTILE_VELOCITY"],
        fire_latency_s=values["FIRE_LATENCY_S"],
    )
