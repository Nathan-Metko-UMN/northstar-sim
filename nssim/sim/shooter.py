"""Our robot: turret kinematics identical to Northstar-CV's ``kinematics.hpp``.

The CV's base frame has its origin at our robot's base and does not turn with the turret; its
heading is the turret IMU's yaw zero. Here that zero is the world +x axis, so base-frame axes are
world axes and only the origin moves with the robot.
"""

from __future__ import annotations

import numpy as np

from .geometry import Transform, rot_y, rot_z
from .robot_constants import RobotConstants


class Shooter:
    def __init__(self, constants: RobotConstants, base_xyz):
        self.constants = constants
        self.base_xyz = np.asarray(base_xyz, dtype=float)

    def base(self) -> Transform:
        return Transform.translation(self.base_xyz)

    def pitch_joint(self, yaw: float, pitch: float) -> Transform:
        """World transform of the pitch pivot. yaw is CCW positive, pitch positive looks down."""
        c = self.constants
        return (
            self.base()
            @ Transform.translation(c.yaw_offset)
            @ Transform.rotation(rot_z(yaw))
            @ Transform.translation(c.pitch_offset)
            @ Transform.rotation(rot_y(pitch))
        )

    def camera(self, yaw: float, pitch: float) -> Transform:
        """Physical camera frame (x forward, y left, z up), which is also SAPIEN's camera convention."""
        return self.pitch_joint(yaw, pitch) @ Transform.translation(self.constants.camera_offset)

    def muzzle(self, yaw: float, pitch: float) -> Transform:
        return self.pitch_joint(yaw, pitch) @ Transform.translation(self.constants.launch_offset)

    def aim_at(self, point) -> tuple[float, float]:
        """Turret yaw/pitch that points the camera's optical axis at a world point."""
        c = self.constants
        pivot_yaw = self.base_xyz + c.yaw_offset
        d = np.asarray(point, float) - pivot_yaw
        yaw = float(np.arctan2(d[1], d[0]))
        # Pitch pivot and camera offsets are small; one refinement pass is plenty.
        pitch = 0.0
        for _ in range(3):
            cam = self.camera(yaw, pitch)
            rel = cam.inverse().apply(np.asarray(point, float))
            pitch += float(np.arctan2(-rel[2], rel[0]))
        return yaw, pitch
