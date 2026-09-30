"""Watch a run live in SAPIEN's viewer and, in ``nssim drive``, steer the robots from the keyboard.

The viewer shows the scene the camera renders for Northstar-CV, plus wireframes that exist only in
the viewer (never in the CV's image):

- our robot: chassis, barrel, where the turret points (red) and the aim Northstar-CV sent (yellow)
- Northstar-CV's particle filter: its four plates and center (magenta) and its ballistic aim
  point (yellow), next to the enemy's true center (green)

A second window shows the frame Northstar-CV received, with its detections.

The run is paced to the wall clock. When Northstar-CV and the renderer can't keep up (a high camera
frame rate, a slow machine) it runs in slow motion instead; the panel shows the real-time factor.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
from sapien import internal_renderer as R
from sapien.utils import Viewer
from sapien.utils.viewer.plugin import Plugin

from nssim.camera import RGGB_TO_BGR
from nssim.sim.geometry import mat_to_quat
from nssim.sim.targets import DrivenMotion

CONTROLS = [
    "arrows    drive the enemy (relative to the view)",
    "Q / E     enemy spin slower / faster",
    "space     stop / restart the spin",
    "I J K L   drive our robot",
    "R         reset positions",
    "1 / 2     free view / ride on our camera",
    "mouse, W A S D   move the view",
    "(keys go to this window when it has focus)",
]

ENEMY_SPEED = 2.0  # m/s at full stick
BASE_SPEED = 1.5
SPIN_STEP = 1.0  # rad/s per Q/E press
FIRST_SPIN = 6.0  # what space starts if the enemy has never spun

MAGENTA = (1.0, 0.2, 1.0, 1.0)
YELLOW = (1.0, 0.85, 0.0, 1.0)
RED = (1.0, 0.2, 0.1, 1.0)
GREEN = (0.2, 1.0, 0.3, 1.0)
WHITE = (0.85, 0.85, 0.85, 1.0)

PLATE_HALF = (0.005, 0.0675, 0.0625)  # thickness, width, height of a small armor panel, halved
CV_WINDOW = "Northstar-CV camera"

_CUBE = np.array([[1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
                  [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]], dtype=np.float32)
_CUBE_EDGES = _CUBE[[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]]
_SEGMENT = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)  # unit length along x
_IDENTITY = [1.0, 0.0, 0.0, 0.0]


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class _Wire:
    """A wireframe for the viewer: a unit shape that is posed and scaled every view frame.

    SAPIEN's viewer draws into the same render scene as the CV's camera, so wires are only shown
    while the viewer renders and are parked out of sight (tiny, far below the floor) otherwise.
    """

    PARKED = np.array([0.0, 0.0, -100.0], np.float32)

    def __init__(self, viewer: Viewer, vertices: np.ndarray, color, width: float = 2.0):
        colors = np.tile(np.asarray(color, np.float32), (len(vertices), 1))
        self.node = viewer.render_scene.add_line_set(viewer.renderer_context.create_line_set(vertices, colors))
        self.node.line_width = width
        self.hide()

    def show(self, position, quat_wxyz, scale) -> None:
        self.node.set_position(np.asarray(position, np.float32))
        self.node.set_rotation(np.asarray(quat_wxyz, np.float32))
        self.node.set_scale(np.asarray(scale, np.float32))

    def hide(self) -> None:
        self.node.set_position(self.PARKED)
        self.node.set_scale(np.full(3, 1e-6, np.float32))


class _Panel(Plugin):
    """Status and key help, as an ImGui window in the viewer."""

    def __init__(self):
        self.lines: list[str] = []
        self._window = None
        self._count = -1

    def get_ui_windows(self):
        if self._window is None or len(self.lines) != self._count:
            self._count = len(self.lines)
            self._window = R.UIWindow().Label("nssim").Pos(10, 460).Size(430, 22 * self._count + 40)
            for i in range(self._count):
                self._window.append(R.UIDisplayText().Bind(lambda i=i: self.lines[i] if i < len(self.lines) else ""))
        return [self._window]


class LiveSession:
    """Run hooks (see ``Runner.run``) that show the run and, with ``drive``, take keyboard control."""

    view_hz = 30.0
    cv_view_hz = 15.0

    def __init__(self, runner, drive: bool = False, show_cv: bool = True):
        self.runner = runner
        self.drive = drive
        self.show_cv = show_cv
        self.stop_requested = False

        self.viewer = Viewer(resolutions=(1600, 900))
        self.viewer.set_scene(runner.scene.scene)
        self.panel = _Panel()
        self.panel.init(self.viewer)
        self.viewer.plugins.append(self.panel)
        # Both would leak into the CV's image, which shares the viewer's render scene: camera
        # frustum lines, and the half-transparent highlight SAPIEN gives a clicked entity.
        self.viewer.control_window.show_camera_linesets = False
        self.viewer.register_click_handler(lambda viewer, x, y: True)

        self.enemy_name = next(iter(runner.targets))
        self.enemy: DrivenMotion | None = None
        self.base: DrivenMotion | None = None
        if drive:
            lo, hi = runner.scene.floor_bounds
            bounds = (lo + 0.4, hi - 0.4)
            target = runner.targets[self.enemy_name]
            self.enemy = DrivenMotion(target.motion.center_xy(0.0), spin0=target.motion.spin(0.0), bounds=bounds)
            target.motion = self.enemy
            self.base = DrivenMotion(runner.shooter.base_xyz[:2], bounds=bounds)
        self._saved_omega = FIRST_SPIN

        self._make_wires()
        self._place_view()
        self._riding = False
        self._free_view = None
        self._wall0 = None
        self._last_view = self._last_cv = -math.inf
        self._rtf = 1.0
        self._rtf_ref = None  # (sim t, wall) at the last real-time factor update
        self._t = 0.0

    # --- hooks -----------------------------------------------------------------------------

    def tick(self, runner, t: float, dt: float) -> None:
        self._t = t
        if self.drive:
            self.enemy.step(dt)
            self.base.step(dt)
            runner.shooter.base_xyz[:2] = self.base.xy
            runner.base_velocity[:] = self.base.v

        now = time.perf_counter()
        # Pace sim time to the wall clock. After falling behind (slow motion), re-anchor instead of
        # racing to catch up.
        if self._wall0 is None or t - (now - self._wall0) < -0.2:
            self._wall0 = now - t
        ahead = t - (now - self._wall0)
        if ahead > 0.001:
            time.sleep(ahead)

        if now - self._last_view >= 1.0 / self.view_hz:
            self._last_view = now
            self._update_rtf(t, now)
            self._render_view(t)
            if self.viewer.closed:
                self.stop_requested = True

    def frame_delivered(self, runner, frame) -> None:
        if not self.show_cv:
            return
        now = time.perf_counter()
        if now - self._last_cv < 1.0 / self.cv_view_hz:
            return
        self._last_cv = now
        cv2.imshow(CV_WINDOW, self._cv_image(runner, frame))
        cv2.waitKey(1)

    def close(self) -> None:
        if self.show_cv:
            cv2.destroyAllWindows()
        if not self.viewer.closed:
            self.viewer.close()

    # --- viewer ------------------------------------------------------------------------------

    def _make_wires(self) -> None:
        v = self.viewer
        self.wires = {
            "chassis": _Wire(v, _CUBE_EDGES, WHITE),
            "barrel": _Wire(v, _SEGMENT, WHITE, width=4.0),
            "pointing": _Wire(v, _SEGMENT, RED),
            "aim": _Wire(v, _SEGMENT, YELLOW),
            "pf_center": _Wire(v, _CUBE_EDGES, MAGENTA),
            "pf_target": _Wire(v, _CUBE_EDGES, YELLOW),
            "true_center": _Wire(v, _CUBE_EDGES, GREEN),
        }
        self.pf_plates = [_Wire(v, _CUBE_EDGES, MAGENTA) for _ in range(4)]

    def _place_view(self) -> None:
        """Start behind and above our robot, looking past it at the enemy."""
        ours = self.runner.shooter.base_xyz
        enemy = self.runner.targets[self.enemy_name].state(0.0).center
        ahead = enemy[:2] - ours[:2]
        ahead /= max(np.linalg.norm(ahead), 1e-6)
        eye = np.array([*(ours[:2] - 2.2 * ahead + 1.2 * np.array([ahead[1], -ahead[0]])), ours[2] + 1.8])
        look = (ours + enemy) / 2
        d = look - eye
        heading = math.atan2(d[1], d[0])
        down = math.atan2(-d[2], math.hypot(d[0], d[1]))
        self.viewer.set_camera_xyz(*eye)
        self.viewer.set_camera_rpy(0.0, -down, -heading)  # SAPIEN's viewer: +pitch looks up, +yaw turns right

    def _render_view(self, t: float) -> None:
        runner = self.runner
        states = {name: target.state(t) for name, target in runner.targets.items()}
        runner.scene.set_targets(states)
        self._update_wires(states[self.enemy_name])
        self.panel.lines = self._status(states[self.enemy_name]) + [""] + CONTROLS
        if self._riding:
            # (SAPIEN's own "focus camera" doesn't follow in this version, so do it here.)
            eye = runner.shooter.camera(runner.turret.yaw, runner.turret.pitch)
            self.viewer.set_camera_pose(eye.to_sapien())
        self.viewer.render()
        self._hide_wires()  # before the CV's camera renders again
        if not self.viewer.closed:
            self._read_keys()

    def _hide_wires(self) -> None:
        for wire in [*self.wires.values(), *self.pf_plates]:
            wire.hide()

    def _update_wires(self, enemy) -> None:
        runner = self.runner
        shooter = runner.shooter
        base = shooter.base_xyz
        self.wires["chassis"].show(base + [0.0, 0.0, 0.12], _IDENTITY, [0.3, 0.25, 0.1])

        joint = shooter.pitch_joint(runner.turret.yaw, runner.turret.pitch)
        muzzle = shooter.muzzle(runner.turret.yaw, runner.turret.pitch)
        ray = np.linalg.norm(enemy.center - muzzle.p) + 0.5
        q = mat_to_quat(joint.R)
        self.wires["barrel"].show(joint.p, q, [np.linalg.norm(muzzle.p - joint.p) + 0.05, 1, 1])
        self.wires["pointing"].show(muzzle.p, q, [ray, 1, 1])
        aim = runner.mcb.aim_state
        if aim.updated:
            commanded = shooter.muzzle(aim.aim.yaw, aim.aim.pitch)
            self.wires["aim"].show(commanded.p, mat_to_quat(commanded.R), [ray, 1, 1])
        else:
            self.wires["aim"].hide()

        self.wires["true_center"].show(enemy.center, _IDENTITY, [0.03] * 3)
        track = runner.latest.get("track")
        if track and track.get("pf_alive") and "state" in track:
            state = track["state"]
            center = np.asarray(state["center"]) + base
            self.wires["pf_center"].show(center, _IDENTITY, [0.03] * 3)
            for wire, plate in zip(self.pf_plates, state["plates"]):
                p = np.asarray(plate) + base
                wire.show(p, _yaw_quat(math.atan2(p[1] - center[1], p[0] - center[0])), PLATE_HALF)
            ballistics = track.get("ballistics", {})
            if ballistics.get("success"):
                self.wires["pf_target"].show(np.asarray(ballistics["target"]) + base, _IDENTITY, [0.02] * 3)
            else:
                self.wires["pf_target"].hide()
        else:
            for wire in [self.wires["pf_center"], self.wires["pf_target"], *self.pf_plates]:
                wire.hide()

    def _status(self, enemy) -> list[str]:
        runner = self.runner
        base = runner.shooter.base_xyz
        lines = [
            f"sim {self._t:7.2f} s    real time x{self._rtf:.2f}    camera {runner.config.fps:.0f} fps",
            f"enemy  {np.linalg.norm(enemy.velocity):.1f} m/s   spin {enemy.omega:+.1f} rad/s"
            + (f" (cmd {self.enemy.command_omega:+.0f})" if self.drive else ""),
        ]
        if self.drive:
            lines.append(f"ours   {np.linalg.norm(self.base.v):.1f} m/s")
        det = runner.latest.get("det")
        if det:
            lines.append(f"CV     frame {det['seq']}: {len(det['armors'])} plate(s), {det.get('n_lights', '?')} lights")
        track = runner.latest.get("track")
        if track and track.get("pf_alive") and "state" in track:
            state = track["state"]
            center_err = np.linalg.norm(np.asarray(state["center"][:2]) + base[:2] - enemy.center[:2])
            radius = (runner.targets[self.enemy_name].spec.radius_high + runner.targets[self.enemy_name].spec.radius_low) / 2
            lines += [
                f"filter spin {state['omega']:+.1f} rad/s (true {enemy.omega:+.1f})   radius {state['radius']:.2f} m (true {radius:.2f})",
                f"       center off by {100 * center_err:.0f} cm",
            ]
        else:
            lines.append("filter not tracking")
        return lines

    def _update_rtf(self, t: float, now: float) -> None:
        if self._rtf_ref is None:
            self._rtf_ref = (t, now)
            return
        t0, w0 = self._rtf_ref
        if now - w0 >= 0.5:
            self._rtf = 0.5 * self._rtf + 0.5 * (t - t0) / (now - w0)
            self._rtf_ref = (t, now)

    # --- input -------------------------------------------------------------------------------

    def _read_keys(self) -> None:
        w = self.viewer.window
        if w.key_press("1"):
            self._ride(False)
        if w.key_press("2"):
            self._ride(True)
        if not self.drive:
            return
        forward, left = self._view_axes()
        self.enemy.command_velocity = ENEMY_SPEED * _stick(w, forward, left, "up", "down", "left", "right")
        self.base.command_velocity = BASE_SPEED * _stick(w, forward, left, "i", "k", "j", "l")
        if w.key_press("e"):
            self.enemy.command_omega += SPIN_STEP
        if w.key_press("q"):
            self.enemy.command_omega -= SPIN_STEP
        if w.key_press("space"):
            if self.enemy.command_omega:
                self._saved_omega, self.enemy.command_omega = self.enemy.command_omega, 0.0
            else:
                self.enemy.command_omega = self._saved_omega
        if w.key_press("r"):
            self.enemy.reset()
            self.base.reset()

    def _view_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """The view's forward and left directions, flattened onto the floor."""
        m = self.viewer.window.get_camera_pose().to_transformation_matrix()
        forward = m[:2, 0] if np.linalg.norm(m[:2, 0]) > 0.2 else m[:2, 2]  # looking straight down: use "up"
        forward = forward / np.linalg.norm(forward)
        return forward, np.array([-forward[1], forward[0]])

    def _ride(self, on: bool) -> None:
        """Put the view on our turret camera (with its field of view), or give it back."""
        if on == self._riding:
            return
        w = self.viewer.window
        if on:
            self._free_view = (w.get_camera_pose(), w.fovy)
            cam = self.runner.config.camera
            w.set_camera_parameters(w.near, w.far, 2 * math.atan(cam.height / 2 / cam.fy))
        else:
            pose, fovy = self._free_view
            w.set_camera_parameters(w.near, w.far, fovy)
            self.viewer.set_camera_pose(pose)
            self.viewer.control_window._sync_fps_camera_controller()
        self._riding = on

    # --- the CV's view -----------------------------------------------------------------------

    def _cv_image(self, runner, frame) -> np.ndarray:
        bgr = cv2.cvtColor(frame.bayer, RGGB_TO_BGR)  # what Northstar-CV debayers
        for target in frame.gt["targets"]:
            for plate in target["plates"]:
                if plate["facing"]:
                    for u, v in np.asarray(plate["corners_px"], float):
                        if np.isfinite(u):
                            cv2.circle(bgr, (int(round(u)), int(round(v))), 5, (0, 255, 0), 2)
        det = runner.latest.get("det")
        if det is not None and det.get("seq") == frame.seq:
            for key, color in (("armors", (255, 255, 0)), ("rejected", (255, 0, 255))):
                for armor in det.get(key, []):
                    pts = np.round(np.asarray(armor["corners"])).astype(np.int32)
                    cv2.polylines(bgr, [pts.reshape(-1, 1, 2)], True, color, 2)
                    label = f"{armor['number']} {armor['confidence']:.2f}"
                    cv2.putText(bgr, label, (int(pts[1][0]), int(pts[1][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(bgr, f"frame {frame.seq}   green: true light bars   cyan: detected   magenta: rejected",
                    (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def _stick(window, forward, left, up_key, down_key, left_key, right_key) -> np.ndarray:
    d = np.zeros(2)
    if window.key_down(up_key):
        d += forward
    if window.key_down(down_key):
        d -= forward
    if window.key_down(left_key):
        d += left
    if window.key_down(right_key):
        d -= left
    n = np.linalg.norm(d)
    return d / n if n > 0 else d
