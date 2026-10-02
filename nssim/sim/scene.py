"""SAPIEN scene for the harness: TR's ARC 2026 field, enemy robots and the Triton2 camera.

Everything is kinematic: poses are set from the scripted motion and the turret model right
before a frame is rendered, so ground truth and pixels come from the same numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import sapien

from nssim.camera import CameraModel

from .assets import FIELD_ELEMENTS, FIELD_FLOOR, PANEL_MODELS, TrAssets
from .geometry import Transform
from .targets import R_PLATE_FROM_MESH, TargetSpec, TargetState



@dataclass
class Appearance:
    """How the scene looks to the camera.

    The scene light is dim, to imitate the low exposure the CV runs at, so only the emissive light
    bars are bright.
    """

    ambient: float = 0.04
    sun: float = 0.25
    sun_direction: tuple = (0.3, 0.2, -1.0)
    # Light bars as a color camera records the LEDs: blue ones bleed into the green pixels and look
    # cyan with a near-white core, red ones look orange-red (compare the HSV bounds Northstar-CV's
    # detector was tuned with: blue hue 160-230 deg, red 0-60 deg, both very bright). The strength
    # is set so the dominant channel just saturates, leaving the others to carry the hue.
    bar_colors: dict = field(
        default_factory=lambda: {"blue": [0.3, 0.75, 1.0], "red": [1.0, 0.3, 0.18]}
    )
    bar_emission: float = 1.2
    # The sticker glyphs are sized and placed as on DJI's reference stickers (STICKER_GLYPHS);
    # this scales them from there, to see how the classifier copes with other sizes.
    glyph_scale: float = 1.0


@dataclass(frozen=True)
class StickerGlyph:
    """A glyph on DJI's reference sticker (RoboMaster 2026 rules, Appendix II), which covers the
    panel's front face: its bounding box's height as a fraction of the sticker's, and its center's
    offset from the sticker's center as fractions of the sticker's width and height (x right, y up)."""

    height: float
    offset: tuple[float, float]


# Measured from DJI's drawings. TR's glyphs have the same shapes (width per height within 2%), so
# only their size and place change: TR's are 0.71-0.78 of the sticker's height, against these.
STICKER_GLYPHS = {
    "infantry": StickerGlyph(0.841, (-0.001, -0.002)),  # the 3
    "sentry": StickerGlyph(0.700, (-0.001, 0.001)),
    "hero": StickerGlyph(0.815, (-0.028, -0.006)),  # the 1, its flag off to the left
}


# Light bars as boxes in the plate frame. TR's panel meshes recess the bars behind the panel rim,
# which hides the far bar beyond ~40 degrees; real light bars stand proud and stay visible, so the
# harness draws its own (TR's width and length), centered on the ground-truth bar line and in front
# of the panel face.
BAR_HALF_DEPTH = 0.002
PANEL_BEHIND_BARS = 0.001  # gap between the panel face and the back of the bars
SYMBOL_LIFT = 0.0002  # the rescaled sticker sits this far in front of TR's (hidden) one
CHASSIS_ABOVE_CENTER = 0.07  # the chassis's top above the plate-ring center
CHASSIS_FLOOR_GAP = 0.03  # and its bottom above the floor

PANEL_GRAY = [0.02, 0.02, 0.02, 1.0]


class _TargetVisual:
    def __init__(self, scene: sapien.Scene, assets: TrAssets, spec: TargetSpec, look: Appearance):
        self.spec = spec
        model = PANEL_MODELS[spec.panel]
        geometry = assets.plate_geometry(spec.panel)
        bars = geometry.bars
        # (Scaled) panel mesh -> plate frame (origin at the light-bar center, x out of the plate),
        # with the panel face just behind the bars.
        face_x = assets.panel_front_z(spec.panel) + BAR_HALF_DEPTH + PANEL_BEHIND_BARS
        offset = np.array([0.0, bars.y_center, face_x])
        self._plate_from_mesh = Transform(R_PLATE_FROM_MESH, -R_PLATE_FROM_MESH @ offset)
        mesh_from_plate = self._plate_from_mesh.inverse()

        color = look.bar_colors[spec.color]
        bar_material = sapien.render.RenderMaterial(base_color=[*color, 1.0], emission=[*color, look.bar_emission])
        symbol_material = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        symbol = _scaled_symbol(assets, spec.panel, geometry.scale, look.glyph_scale)

        self.panels = []
        for i in range(4):
            body = sapien.render.RenderBodyComponent()
            panel = sapien.render.RenderShapeTriangleMesh(
                str(assets.panel_file(spec.panel)), np.asarray(geometry.scale, np.float32), None
            )
            self._style_panel(panel, model)
            body.attach(panel)
            for side in (1.0, -1.0):
                # Box axes in mesh terms: x = plate width, y = plate up, z = out of the plate.
                box = sapien.render.RenderShapeBox(
                    np.array([bars.half_width, bars.half_height, BAR_HALF_DEPTH], np.float32), bar_material
                )
                box.local_pose = sapien.Pose(mesh_from_plate.apply([0.0, side * bars.half_spacing, 0.0]).tolist())
                body.attach(box)
            body.attach(sapien.render.RenderShapeTriangleMesh(*symbol, symbol_material))
            entity = sapien.Entity()
            entity.name = f"{spec.name}_plate_{i}"
            entity.add_component(body)
            scene.add_entity(entity)
            self.panels.append(entity)

        # Dark chassis inside the plate ring, from just off the floor to above the plates: plates
        # facing away don't show through, and up close, where the camera looks down past a plate's
        # bottom edge, it sees the robot as on a real one rather than the floor. The number
        # classifier's window reaches a little below the plate, and the gray floor there reads as
        # part of the glyph ("negative" for the 1 and the sentry icon within ~1.5 m).
        r = 0.6 * min(spec.radius_high, spec.radius_low)
        top, bottom = CHASSIS_ABOVE_CENTER, spec.center_height - CHASSIS_FLOOR_GAP  # from the ring center
        material = sapien.render.RenderMaterial(base_color=[0.08, 0.08, 0.08, 1.0], roughness=0.9)
        builder = scene.create_actor_builder()
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, (top - bottom) / 2]),
                               half_size=[r, r, (top + bottom) / 2], material=material)
        self.chassis = builder.build_kinematic(name=f"{spec.name}_chassis")

    @staticmethod
    def _style_panel(panel, model) -> None:
        parts = panel.parts
        # TR's recessed bars stay dark (the boxes are the light source) and TR's sticker is hidden
        # under the rescaled copy.
        recessed = parts[model.bars].material
        recessed.set_base_color(PANEL_GRAY)
        recessed.set_emission([0.0, 0.0, 0.0, 0.0])
        parts[model.symbol].material.set_base_color(PANEL_GRAY)
        parts[model.body].material.set_base_color(PANEL_GRAY)

    def update(self, state: TargetState) -> None:
        for entity, plate in zip(self.panels, state.plates):
            entity.set_pose((plate @ self._plate_from_mesh).to_sapien())
        self.chassis.set_pose(sapien.Pose(state.center.tolist(), _yaw_quat(state.spin)))


def _scaled_symbol(assets: TrAssets, kind: str, panel_scale, glyph_scale: float = 1.0):
    """TR's sticker mesh at the panel's scale, resized and moved to where DJI's reference sticker
    has the glyph (STICKER_GLYPHS), and lifted off the panel.

    Returns (vertices, triangles, normals, uvs) for RenderShapeTriangleMesh.
    """
    part = assets.symbol_mesh(kind)
    scale = np.asarray(panel_scale, np.float64)
    v = part.vertices.astype(np.float64) * scale
    face_min, face_max = (b * scale[:2] for b in assets.panel_face(kind))
    face_size = face_max - face_min
    glyph = STICKER_GLYPHS[kind]
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    s = glyph_scale * glyph.height * face_size[1] / (hi[1] - lo[1])
    target = (face_min + face_max) / 2 + np.asarray(glyph.offset) * face_size
    v[:, :2] = target + (v[:, :2] - (lo + hi) / 2) * s
    v[:, 2] += SYMBOL_LIFT
    k = np.array([s, s, 1.0])
    n = part.normals.astype(np.float64) / (scale * k)  # normals transform by the inverse scale
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return v.astype(np.float32), part.triangles, n.astype(np.float32), part.uvs


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


RENDER_DEVICE_ENV = "NSSIM_RENDER_DEVICE"


def render_system() -> sapien.render.RenderSystem:
    """SAPIEN's renderer, on the GPU NSSIM_RENDER_DEVICE names (cuda for the NVIDIA one, or
    pci:<bus address>), else the one SAPIEN picks."""
    device = os.environ.get(RENDER_DEVICE_ENV) or None
    try:
        return sapien.render.RenderSystem(device) if device else sapien.render.RenderSystem()
    except RuntimeError as e:
        raise RuntimeError(
            f"SAPIEN couldn't start its Vulkan renderer on {device or 'the GPU it picked'}: {e}\n"
            f"The GPUs it sees:\n{sapien.render.get_device_summary()}\n"
            f"Set {RENDER_DEVICE_ENV} to choose another (cuda picks the NVIDIA GPU), or update the GPU's driver."
        ) from e


class ArenaScene:
    def __init__(self, assets: TrAssets, camera: CameraModel, appearance: Appearance | None = None):
        self.assets = assets
        self.camera_model = camera
        self.appearance = appearance or Appearance()
        # 8-bit color straight from the GPU: same values as clamping the float output, ~10x cheaper
        # to read back. Process-wide, so it must be set before any camera exists.
        sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
        renderer = render_system()
        self.gpu = renderer.device.name
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), renderer])
        self._setup_lights()
        self._load_field()
        self.targets: dict[str, _TargetVisual] = {}

        self.camera = self.scene.add_camera("cv_camera", camera.width, camera.height, 0.6, 0.01, 60.0)
        self.camera.set_perspective_parameters(0.01, 60.0, camera.fx, camera.fy, camera.cx, camera.cy, 0.0)

    def _setup_lights(self) -> None:
        a = self.appearance.ambient
        self.scene.set_ambient_light([a, a, a])
        s = self.appearance.sun
        self.scene.add_directional_light(list(self.appearance.sun_direction), [s, s, s], shadow=False)

    def _load_field(self) -> None:
        for name in [FIELD_FLOOR, *FIELD_ELEMENTS]:
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(str(self.assets.field_file(name)))
            entity = builder.build_kinematic(name=name)
            if name == FIELD_FLOOR:
                body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
                aabb = body.compute_global_aabb_tight()
                self.floor_bounds = (aabb[0, :2].astype(float), aabb[1, :2].astype(float))  # xy min, max

    def add_target(self, spec: TargetSpec) -> None:
        self.targets[spec.name] = _TargetVisual(self.scene, self.assets, spec, self.appearance)

    def set_targets(self, targets: dict[str, TargetState]) -> None:
        """Pose the robots (render() does this too; the live viewer calls it between frames)."""
        for name, state in targets.items():
            self.targets[name].update(state)

    def render(self, camera_pose: Transform, targets: dict[str, TargetState]) -> np.ndarray:
        """RGB uint8 (H, W, 3) from the given camera pose (x forward, y left, z up)."""
        self.set_targets(targets)
        self.camera.entity.set_pose(camera_pose.to_sapien())
        self.scene.update_render()
        try:
            self.camera.take_picture()
        except RuntimeError as e:
            # e.g. ErrorOutOfPoolMemory on Intel's UHD 770, which starts the renderer fine
            raise RuntimeError(
                f"SAPIEN couldn't render on {self.gpu}: {e}\n"
                f"Set {RENDER_DEVICE_ENV}=cuda to use an NVIDIA GPU, if this machine has one."
            ) from e
        # uint8 RGBA (see set_picture_format); cvtColor drops alpha ~10x faster than a numpy copy.
        return cv2.cvtColor(self.camera.get_picture("Color"), cv2.COLOR_RGBA2RGB)
