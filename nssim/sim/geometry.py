"""Small rigid-transform helpers (numpy). Quaternions are (w, x, y, z), as in SAPIEN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


@dataclass
class Transform:
    """Rigid transform: x_parent = R @ x_child + p."""

    R: np.ndarray
    p: np.ndarray

    @staticmethod
    def identity() -> "Transform":
        return Transform(np.eye(3), np.zeros(3))

    @staticmethod
    def translation(xyz) -> "Transform":
        return Transform(np.eye(3), np.asarray(xyz, dtype=float))

    @staticmethod
    def rotation(R: np.ndarray) -> "Transform":
        return Transform(np.asarray(R, dtype=float), np.zeros(3))

    def __matmul__(self, other: "Transform") -> "Transform":
        return Transform(self.R @ other.R, self.R @ other.p + self.p)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform (3,) or (N, 3) points."""
        return np.asarray(points) @ self.R.T + self.p

    def inverse(self) -> "Transform":
        return Transform(self.R.T, -self.R.T @ self.p)

    def to_sapien(self):
        import sapien

        return sapien.Pose(self.p.tolist(), mat_to_quat(self.R).tolist())


def wrap_pi(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi
