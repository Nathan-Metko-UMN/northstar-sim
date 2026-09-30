"""Watch a run live in SAPIEN's viewer and, in ``nssim drive``, steer the robots from the keyboard.

The viewer shows the scene the camera renders for Northstar-CV, plus wireframes that exist only in
the viewer (never in the CV's image):

- our robot: chassis, barrel, where the turret points (red) and the aim Northstar-CV sent (yellow)
- Northstar-CV's particle filter: its four plates and center (magenta) and its ballistic aim
  point (yellow), next to each enemy's true center (green)

A second window shows the frame Northstar-CV received, with its detections.

The keys are the TR simulator's (``util_nodes/keyboard_controls.py`` in TR-Simulation-ARCTIC-2026),
so the same fingers work in both: its secondary robot (the target it uses for aiming practice) is
our enemy, its primary robot is ours. Like TR, WASD is left to the viewer's camera.

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

# TR's teleop constants (sim_node/constants.py).
TR_TELEOP_SPEED = 0.6  # m/s while a move key is held
TR_TELEOP_TURN = 1.0  # rad/s, our chassis (R / Y)
TR_MAX_SPIN = 4.5  # rad/s; the number keys pick fractions of it
SPIN_PRESETS = {"5": 0.0, "4": 0.25, "3": 0.5, "2": 0.75, "1": 1.0, "6": -0.25, "7": -0.5, "8": -0.75, "9": -1.0}

CONTROLS = [
    "enemy     I / K forward / back, J / L left / right (field axes)",
    "          5 stop spin, 4 3 2 1 = 25-100% CCW, 6 7 8 9 = 25-100% CW",
    "          hold Shift / Ctrl for the 2nd / 3rd enemy",
    "our robot T / G forward / back, F / H left / right (turret heading)",
    "          R / Y turn the chassis",
    "V         ride on our turret camera / free view",
    "0         reset positions",
    "mouse, W A S D   move the view",
    "(keys go to this window when it has focus)",
]

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
            self._window = R.UIWindow().Label("nssim").Pos(10, 460).Size(470, 22 * self._count + 40)
            for i in range(self._count):
                self._window.append(R.UIDisplayText().Bind(lambda i=i: self.lines[i] if i < len(self.lines) else ""))
        return [self._window]


class LiveSession:
    """Run hooks (see ``Runner.run``) that show the run and, with ``drive``, take keyboard control."""

    view_hz = 30.0
    cv_view_hz = 15.0

    def __init__(self, runner, drive: bool = False, show_cv: bool = True,
                 speed: float = TR_TELEOP_SPEED, max_spin: float = TR_MAX_SPIN):
        self.runner = runner
        self.drive = drive
        self.show_cv = show_cv
        self.speed = speed
        self.max_spin = max_spin
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

        self.names = list(runner.targets)
        self.enemies: list[DrivenMotion] = []
        self.base: DrivenMotion | None = None
        self.base_heading = 0.0  # our chassis (the turret is IMU-stabilized, so only the wireframe turns)
        self._base_turn = 0.0
        if drive:
            lo, hi = runner.scene.floor_bounds
            bounds = (lo + 0.4, hi - 0.4)
            for target in runner.targets.values():
                motion = DrivenMotion(target.motion.center_xy(0.0), spin0=target.motion.spin(0.0), bounds=bounds)
                target.motion = motion
                self.enemies.append(motion)
            self.base = DrivenMotion(runner.shooter.base_xyz[:2], bounds=bounds)

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
            for enemy in self.enemies:
                enemy.step(dt)
            self.base.step(dt)
            self.base_heading += self._base_turn * dt
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
        }
        self.pf_plates = [_Wire(v, _CUBE_EDGES, MAGENTA) for _ in range(4)]
        self.true_centers = [_Wire(v, _CUBE_EDGES, GREEN) for _ in self.names]

    def _all_wires(self) -> list[_Wire]:
        return [*self.wires.values(), *self.pf_plates, *self.true_centers]

    def _place_view(self) -> None:
        """Start behind and above our robot, looking past it at the enemies."""
        ours = self.runner.shooter.base_xyz
        enemy = np.mean([t.state(0.0).center for t in self.runner.targets.values()], axis=0)
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
        tracked = self._tracked(states)
        self._update_wires(states, tracked)
        self.panel.lines = self._status(states, tracked) + [""] + CONTROLS
        if self._riding:
            # (SAPIEN's own "focus camera" doesn't follow in this version, so do it here.)
            eye = runner.shooter.camera(runner.turret.yaw, runner.turret.pitch)
            self.viewer.set_camera_pose(eye.to_sapien())
        self.viewer.render()
        for wire in self._all_wires():  # out of sight before the CV's camera renders again
            wire.hide()
        if not self.viewer.closed:
            self._read_keys()

    def _tracked(self, states) -> str | None:
        """The enemy the filter is on: the one whose true center is nearest its estimate."""
        track = self.runner.latest.get("track")
        if not (track and track.get("pf_alive") and "state" in track):
            return None
        estimate = np.asarray(track["state"]["center"][:2]) + self.runner.shooter.base_xyz[:2]
        return min(states, key=lambda name: np.linalg.norm(states[name].center[:2] - estimate))

    def _update_wires(self, states, tracked) -> None:
        runner = self.runner
        shooter = runner.shooter
        base = shooter.base_xyz
        self.wires["chassis"].show(base + [0.0, 0.0, 0.12], _yaw_quat(self.base_heading), [0.3, 0.25, 0.1])

        joint = shooter.pitch_joint(runner.turret.yaw, runner.turret.pitch)
        muzzle = shooter.muzzle(runner.turret.yaw, runner.turret.pitch)
        focus = states[tracked or self.names[0]].center
        ray = np.linalg.norm(focus - muzzle.p) + 0.5
        q = mat_to_quat(joint.R)
        self.wires["barrel"].show(joint.p, q, [np.linalg.norm(muzzle.p - joint.p) + 0.05, 1, 1])
        self.wires["pointing"].show(muzzle.p, q, [ray, 1, 1])
        aim = runner.mcb.aim_state
        if aim.updated:
            commanded = shooter.muzzle(aim.aim.yaw, aim.aim.pitch)
            self.wires["aim"].show(commanded.p, mat_to_quat(commanded.R), [ray, 1, 1])

        for wire, name in zip(self.true_centers, self.names):
            wire.show(states[name].center, _IDENTITY, [0.03] * 3)
        track = runner.latest.get("track")
        if tracked is not None:
            state = track["state"]
            center = np.asarray(state["center"]) + base
            self.wires["pf_center"].show(center, _IDENTITY, [0.03] * 3)
            for wire, plate in zip(self.pf_plates, state["plates"]):
                p = np.asarray(plate) + base
                wire.show(p, _yaw_quat(math.atan2(p[1] - center[1], p[0] - center[0])), PLATE_HALF)
            ballistics = track.get("ballistics", {})
            if ballistics.get("success"):
                self.wires["pf_target"].show(np.asarray(ballistics["target"]) + base, _IDENTITY, [0.02] * 3)

    def _status(self, states, tracked) -> list[str]:
        runner = self.runner
        lines = [f"sim {self._t:7.2f} s    real time x{self._rtf:.2f}    camera {runner.config.fps:.0f} fps"]
        for i, name in enumerate(self.names):
            state = states[name]
            keys = ("", "Shift+", "Ctrl+")[i] if i < 3 else ""
            lines.append(
                f"{keys + name:14s} {np.linalg.norm(state.velocity):.1f} m/s  spin {state.omega:+.1f} rad/s"
                + ("   <- tracked" if name == tracked else "")
            )
        if self.drive:
            lines.append(f"{'our robot':14s} {np.linalg.norm(self.base.v):.1f} m/s")
        det = runner.latest.get("det")
        if det:
            found = ", ".join(a["number"] for a in det["armors"]) or "none"
            lines.append(f"CV     frame {det['seq']}: plates {found}")
        if tracked is not None:
            state = runner.latest["track"]["state"]
            spec = runner.targets[tracked].spec
            true = states[tracked]
            off = np.linalg.norm(np.asarray(state["center"][:2]) + runner.shooter.base_xyz[:2] - true.center[:2])
            lines += [
                f"filter spin {state['omega']:+.1f} rad/s (true {true.omega:+.1f})"
                f"   radius {state['radius']:.2f} m (true {(spec.radius_high + spec.radius_low) / 2:.2f})",
                f"       center off by {100 * off:.0f} cm",
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

    def _read_keys(self, window=None) -> None:
        """Keys held / pressed since the last view frame (``window`` stands in for tests)."""
        w = window or self.viewer.window
        if w.key_press("v"):
            self._ride(not self._riding)
        if not self.drive:
            return
        if w.key_press("0"):
            for motion in (*self.enemies, self.base):
                motion.reset()
            self.base_heading = 0.0

        # Our robot, TR's primary-robot keys: moves relative to where the turret points.
        forward, left = _axes(w, "t", "g", "f", "h")
        c, s = math.cos(self.runner.turret.yaw), math.sin(self.runner.turret.yaw)
        self.base.command_velocity = self.speed * np.array([c * forward - s * left, s * forward + c * left])
        self._base_turn = TR_TELEOP_TURN * (w.key_down("r") - w.key_down("y"))

        # Enemies, TR's secondary-robot keys along the field axes; Shift / Ctrl pick the 2nd / 3rd.
        pick = 2 if w.ctrl else 1 if w.shift else 0
        for i, enemy in enumerate(self.enemies):
            enemy.command_velocity = self.speed * np.array(_axes(w, "i", "k", "j", "l")) if i == pick else np.zeros(2)
        if pick < len(self.enemies):
            for key, fraction in SPIN_PRESETS.items():
                if w.key_press(key):
                    self.enemies[pick].command_omega = fraction * self.max_spin

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


def _axes(window, plus_x: str, minus_x: str, plus_y: str, minus_y: str) -> tuple[float, float]:
    """Held keys as (x, y) in {-1, 0, 1}; like TR, both axes at once make a faster diagonal."""
    return (
        float(window.key_down(plus_x)) - float(window.key_down(minus_x)),
        float(window.key_down(plus_y)) - float(window.key_down(minus_y)),
    )
