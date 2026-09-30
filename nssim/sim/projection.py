"""Pinhole projection matching OpenCV's model, from our physical camera frame."""

from __future__ import annotations

import numpy as np

from nssim.camera import CameraModel

from .geometry import Transform

# Physical camera frame (x forward, y left, z up) -> OpenCV optical frame (x right, y down, z forward).
R_OPTICAL_FROM_CAMERA = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def project(camera: CameraModel, camera_pose: Transform, points_world: np.ndarray) -> np.ndarray:
    """(N, 3) world points -> (N, 2) pixels (NaN behind the camera). Distortion is not applied."""
    pts = np.atleast_2d(points_world)
    optical = camera_pose.inverse().apply(pts) @ R_OPTICAL_FROM_CAMERA.T
    z = optical[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera.fx * optical[:, 0] / z + camera.cx
        v = camera.fy * optical[:, 1] / z + camera.cy
    uv = np.stack([u, v], axis=1)
    uv[z <= 1e-6] = np.nan
    return uv
