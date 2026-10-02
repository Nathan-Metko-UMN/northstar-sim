"""TR ARCTIC 2026 simulator assets (field and armor panel models) used by the harness scene."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from nssim import paths

FIELD_FLOOR = "2026_ARC_3v3_floor.gltf"
FIELD_ELEMENTS = [
    "2026_ARC_3v3_side_box.gltf",
    "2026_ARC_3v3_long_ramp_and_platform.gltf",
    "2026_ARC_3v3_platform_short_wall.gltf",
    "2026_ARC_3v3_outer_wall_top.gltf",
    "2026_ARC_3v3_outer_wall_right.gltf",
    "2026_ARC_3v3_outer_wall_bottom.gltf",
    "2026_ARC_3v3_outer_wall_left.gltf",
    "2026_ARC_3v3_large_barrier_right.gltf",
    "2026_ARC_3v3_small_barrier_right.gltf",
    "2026_ARC_3v3_small_ramp_right.gltf",
    "2026_ARC_3v3_large_barrier_left.gltf",
    "2026_ARC_3v3_small_barrier_left.gltf",
    "2026_ARC_3v3_small_ramp_left.gltf",
]
FLOOR_TOP_Z = 0.019  # top surface of the floor mesh


@dataclass(frozen=True)
class PanelModel:
    """One armor panel mesh and which of its parts are the light bars, symbol and body."""

    file: str
    large: bool
    bars: int
    symbol: int
    body: int


# Part order differs between the meshes (TR's ellipse_robot.py swaps the hero's for the same reason).
PANEL_MODELS = {
    "infantry": PanelModel("infantry_armor_panel.gltf", large=False, bars=0, symbol=1, body=2),
    "sentry": PanelModel("sentry_armor_panel.gltf", large=False, bars=0, symbol=1, body=2),
    "hero": PanelModel("hero_armor_panel.gltf", large=True, bars=2, symbol=0, body=1),
}


@dataclass(frozen=True)
class LightBarGeometry:
    """Light bars in panel-mesh coordinates (x width, y up, z out of the plate), meters."""

    half_spacing: float  # |x| of each bar's centerline
    y_center: float
    half_height: float
    half_width: float
    z_front: float  # front surface of the bars


@dataclass(frozen=True)
class MeshPart:
    """One part (material group) of a mesh: arrays as SAPIEN loaded them."""

    vertices: np.ndarray  # (n, 3) float32
    triangles: np.ndarray  # (m, 3) uint32
    normals: np.ndarray  # (n, 3) float32
    uvs: np.ndarray  # (n, 2) float32


@dataclass(frozen=True)
class PlateGeometry:
    """A panel as rendered: mesh scale and light bars in the (scaled) mesh."""

    scale: tuple[float, float, float]
    bars: LightBarGeometry


class TrAssets:
    def __init__(self, tr_dir: Path | None = None, plate_dims: dict[str, tuple[float, float]] | None = None):
        """``plate_dims``: {"small"|"large": (bar spacing, bar length)} to scale panels to; None keeps TR's."""
        self.plate_dims = plate_dims
        self._part_cache: dict[str, list[MeshPart]] = {}
        root = paths.tr_dir(tr_dir)
        self.models_dir = root / "src" / "tr-simulation-maniskill" / "resource" / "models"
        if not (self.models_dir / "field" / FIELD_FLOOR).is_file():
            raise FileNotFoundError(paths.missing_hint(root, "TR simulator assets", "TR_SIM_DIR"))

    def field_file(self, name: str) -> Path:
        return self.models_dir / "field" / name

    def panel_file(self, kind: str) -> Path:
        return self.models_dir / "individual_armor_panels" / PANEL_MODELS[kind].file

    @cached_property
    def light_bars(self) -> dict[str, LightBarGeometry]:
        """Light bars as TR modeled them (unscaled meshes)."""
        return {kind: self._measure_light_bars(kind) for kind in PANEL_MODELS}

    def plate_geometry(self, kind: str) -> PlateGeometry:
        bars = self.light_bars[kind]
        if self.plate_dims is None:
            return PlateGeometry((1.0, 1.0, 1.0), bars)
        spacing, length = self.plate_dims["large" if PANEL_MODELS[kind].large else "small"]
        sx = spacing / 2 / bars.half_spacing
        sy = length / 2 / bars.half_height
        scaled = LightBarGeometry(
            half_spacing=spacing / 2,
            y_center=bars.y_center * sy,
            half_height=length / 2,
            half_width=bars.half_width * sx,
            z_front=bars.z_front,
        )
        return PlateGeometry((sx, sy, 1.0), scaled)

    def panel_front_z(self, kind: str) -> float:
        """Front-most z (out of the plate) of the whole panel mesh."""
        return float(max(part.vertices[:, 2].max() for part in self._parts(kind)))

    def symbol_mesh(self, kind: str) -> MeshPart:
        """The number/icon sticker of a panel (unscaled mesh coordinates)."""
        return self._parts(kind)[PANEL_MODELS[kind].symbol]

    def panel_face(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """x-y bounds (min, max) of the panel body's front face, where the sticker goes (unscaled
        mesh coordinates). Its outline is the sticker's: 1.085 wide per high on small panels, as
        on DJI's reference stickers."""
        body = self._parts(kind)[PANEL_MODELS[kind].body].vertices
        face = body[body[:, 2] > body[:, 2].max() - 0.0005]
        return face[:, :2].min(axis=0).astype(np.float64), face[:, :2].max(axis=0).astype(np.float64)

    def _parts(self, kind: str) -> list[MeshPart]:
        if kind not in self._part_cache:
            import sapien

            shape = sapien.render.RenderShapeTriangleMesh(str(self.panel_file(kind)), np.ones(3, np.float32), None)
            self._part_cache[kind] = [
                MeshPart(
                    vertices=np.asarray(part.vertices, np.float32),
                    triangles=np.asarray(part.triangles, np.uint32),
                    normals=np.asarray(part.get_vertex_normal(), np.float32),
                    uvs=np.asarray(part.get_vertex_uv(), np.float32),
                )
                for part in shape.parts
            ]
        return self._part_cache[kind]

    def _measure_light_bars(self, kind: str) -> LightBarGeometry:
        verts = self._parts(kind)[PANEL_MODELS[kind].bars].vertices
        left, right = verts[verts[:, 0] < 0], verts[verts[:, 0] > 0]
        centers = [(v[:, 0].min() + v[:, 0].max()) / 2 for v in (left, right)]
        widths = [v[:, 0].max() - v[:, 0].min() for v in (left, right)]
        return LightBarGeometry(
            half_spacing=float((centers[1] - centers[0]) / 2),
            y_center=float((verts[:, 1].min() + verts[:, 1].max()) / 2),
            half_height=float((verts[:, 1].max() - verts[:, 1].min()) / 2),
            half_width=float(np.mean(widths) / 2),
            z_front=float(verts[:, 2].max()),
        )
