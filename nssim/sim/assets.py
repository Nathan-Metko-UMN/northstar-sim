"""TR ARCTIC 2026 simulator assets (field and armor panel models) used by the harness scene."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

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
    z_front: float  # front surface of the bars


@dataclass(frozen=True)
class PlateGeometry:
    """A panel as rendered: mesh scale and light bars in the (scaled) mesh."""

    scale: tuple[float, float, float]
    bars: LightBarGeometry


def default_tr_dir() -> Path:
    here = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("TR_SIM_DIR", here.parent / "TR-Simulation-ARCTIC-2026"))


class TrAssets:
    def __init__(self, tr_dir: Path | None = None, plate_dims: dict[str, tuple[float, float]] | None = None):
        """``plate_dims``: {"small"|"large": (bar spacing, bar length)} to scale panels to; None keeps TR's."""
        self.plate_dims = plate_dims
        root = Path(tr_dir) if tr_dir else default_tr_dir()
        self.models_dir = root / "src" / "tr-simulation-maniskill" / "resource" / "models"
        if not (self.models_dir / "field" / FIELD_FLOOR).is_file():
            raise FileNotFoundError(
                f"TR simulator assets not found under {self.models_dir}; set TR_SIM_DIR to your "
                "TR-Simulation-ARCTIC-2026 checkout"
            )

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
            z_front=bars.z_front,
        )
        return PlateGeometry((sx, sy, 1.0), scaled)

    def _measure_light_bars(self, kind: str) -> LightBarGeometry:
        import sapien

        scene = sapien.Scene()
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(str(self.panel_file(kind)))
        actor = builder.build_kinematic(name=kind)
        body = actor.find_component_by_type(sapien.render.RenderBodyComponent)
        verts = np.asarray(body.render_shapes[0].parts[PANEL_MODELS[kind].bars].vertices)
        left, right = verts[verts[:, 0] < 0], verts[verts[:, 0] > 0]
        centers = [(v[:, 0].min() + v[:, 0].max()) / 2 for v in (left, right)]
        return LightBarGeometry(
            half_spacing=float((centers[1] - centers[0]) / 2),
            y_center=float((verts[:, 1].min() + verts[:, 1].max()) / 2),
            half_height=float((verts[:, 1].max() - verts[:, 1].min()) / 2),
            z_front=float(verts[:, 2].max()),
        )
