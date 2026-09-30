"""SAPIEN scene for the harness: TR's ARC 2026 field, enemy robots and the Triton2 camera.

Everything is kinematic: poses are set from the scripted motion and the turret model right
before a frame is rendered, so ground truth and pixels come from the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien

from nssim.camera import CameraModel

from .assets import FIELD_ELEMENTS, FIELD_FLOOR, PANEL_MODELS, TrAssets
from .geometry import Transform
from .targets import R_PLATE_FROM_MESH, TargetSpec, TargetState

BAR_COLORS = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0]}


@dataclass
class Lighting:
    """Dim scene light imitates the low exposure the CV runs at; only emissive light bars are bright."""

    ambient: float = 0.04
    sun: float = 0.25
    sun_direction: tuple = (0.3, 0.2, -1.0)
    bar_emission: float = 40.0  # emissive strength of the light bars


class _TargetVisual:
    def __init__(self, scene: sapien.Scene, assets: TrAssets, spec: TargetSpec, lighting: Lighting):
        self.spec = spec
        model = PANEL_MODELS[spec.panel]
        geometry = assets.plate_geometry(spec.panel)
        bars = geometry.bars
        # (Scaled) panel mesh origin -> plate frame (origin at the light-bar center, x out of the plate).
        offset = np.array([0.0, bars.y_center, bars.z_front])
        self._plate_from_mesh = Transform(R_PLATE_FROM_MESH, -R_PLATE_FROM_MESH @ offset)

        self.panels = []
        for i in range(4):
            builder = scene.create_actor_builder()
            builder.add_visual_from_file(str(assets.panel_file(spec.panel)), scale=list(geometry.scale))
            actor = builder.build_kinematic(name=f"{spec.name}_plate_{i}")
            self._style_panel(actor, model, lighting)
            self.panels.append(actor)

        # Dark chassis inside the plate ring, so plates facing away don't show through.
        r = 0.75 * min(spec.radius_high, spec.radius_low)
        material = sapien.render.RenderMaterial(base_color=[0.08, 0.08, 0.08, 1.0], roughness=0.9)
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[r, r, 0.07], material=material)
        self.chassis = builder.build_kinematic(name=f"{spec.name}_chassis")

    def _style_panel(self, actor, model, lighting: Lighting) -> None:
        body = actor.find_component_by_type(sapien.render.RenderBodyComponent)
        parts = body.render_shapes[0].parts
        color = BAR_COLORS[self.spec.color]
        bar = parts[model.bars].material
        bar.set_base_color([*color, 1.0])
        bar.set_emission([*color, lighting.bar_emission])
        parts[model.symbol].material.set_base_color([1.0, 1.0, 1.0, 1.0])
        parts[model.body].material.set_base_color([0.02, 0.02, 0.02, 1.0])

    def update(self, state: TargetState) -> None:
        for actor, plate in zip(self.panels, state.plates):
            actor.set_pose((plate @ self._plate_from_mesh).to_sapien())
        self.chassis.set_pose(sapien.Pose(state.center.tolist(), _yaw_quat(state.spin)))


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


class ArenaScene:
    def __init__(self, assets: TrAssets, camera: CameraModel, lighting: Lighting | None = None):
        self.assets = assets
        self.camera_model = camera
        self.lighting = lighting or Lighting()
        # 8-bit color straight from the GPU: same values as clamping the float output, ~10x cheaper
        # to read back. Process-wide, so it must be set before any camera exists.
        sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
        self.scene = sapien.Scene()
        self._setup_lights()
        self._load_field()
        self.targets: dict[str, _TargetVisual] = {}

        self.camera = self.scene.add_camera("cv_camera", camera.width, camera.height, 0.6, 0.01, 60.0)
        self.camera.set_perspective_parameters(0.01, 60.0, camera.fx, camera.fy, camera.cx, camera.cy, 0.0)

    def _setup_lights(self) -> None:
        a = self.lighting.ambient
        self.scene.set_ambient_light([a, a, a])
        s = self.lighting.sun
        self.scene.add_directional_light(list(self.lighting.sun_direction), [s, s, s], shadow=False)

    def _load_field(self) -> None:
        for name in [FIELD_FLOOR, *FIELD_ELEMENTS]:
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(str(self.assets.field_file(name)))
            builder.build_kinematic(name=name)

    def add_target(self, spec: TargetSpec) -> None:
        self.targets[spec.name] = _TargetVisual(self.scene, self.assets, spec, self.lighting)

    def render(self, camera_pose: Transform, targets: dict[str, TargetState]) -> np.ndarray:
        """RGB uint8 (H, W, 3) from the given camera pose (x forward, y left, z up)."""
        for name, state in targets.items():
            self.targets[name].update(state)
        self.camera.entity.set_pose(camera_pose.to_sapien())
        self.scene.update_render()
        self.camera.take_picture()
        return self.camera.get_picture("Color")[..., :3]  # uint8, see set_picture_format
