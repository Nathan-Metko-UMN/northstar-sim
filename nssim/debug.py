"""Re-render a logged frame with ground truth and Northstar-CV's detections drawn on top."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import dacite
import numpy as np

from nssim.camera import apply_optics
from nssim.runner import RunConfig
from nssim.sim.assets import TrAssets
from nssim.sim.geometry import Transform
from nssim.sim.robot_constants import load_plate_dims
from nssim.sim.scene import ArenaScene
from nssim.sim.targets import TargetState, plate_transforms


def load_config(run_dir: Path) -> RunConfig:
    data = json.loads((Path(run_dir) / "config.json").read_text())
    return dacite.from_dict(RunConfig, data, config=dacite.Config(check_types=False))


def render_logged_frame(run_dir: Path, seq: int, cv_dir: Path, out: Path) -> Path:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    frame = next(f for f in map(json.loads, open(run_dir / "frames.jsonl")) if f["seq"] == seq)
    telemetry = [json.loads(line) for line in open(run_dir / "telemetry.jsonl")]
    det = next((m for m in telemetry if m.get("type") == "det" and m["seq"] == seq), None)

    assets = TrAssets(plate_dims=load_plate_dims(cv_dir))
    scene = ArenaScene(assets, cfg.camera, cfg.appearance)
    specs = {tc.spec.name: tc.spec for tc in cfg.targets}
    for spec in specs.values():
        scene.add_target(spec)

    # Everything from the logged ground truth, so driven (nssim drive) runs re-render correctly too.
    gt = frame["gt"]
    base = np.asarray(gt["base_world"], float)
    cam_pose = Transform(np.asarray(gt["camera_world"]["R"], float), np.asarray(gt["camera_world"]["p"], float))
    states = {}
    for target in gt["targets"]:
        center = np.asarray(target["center_base"], float) + base
        spec = specs[target["name"]]
        states[target["name"]] = TargetState(
            center=center,
            velocity=np.asarray(target["velocity"], float),
            spin=target["spin"],
            omega=target["omega"],
            plates=plate_transforms(spec, center, target["spin"]),
        )
    bgr = cv2.cvtColor(apply_optics(scene.render(cam_pose, states), cfg.optics), cv2.COLOR_RGB2BGR)

    for target in gt["targets"]:
        for plate in target["plates"]:
            for u, v in np.asarray(plate["corners_px"], float):
                if np.isfinite(u):
                    color = (0, 255, 0) if plate["facing"] else (0, 0, 160)
                    cv2.circle(bgr, (int(round(u)), int(round(v))), 4, color, 1)
    if det is not None:
        for key, color in (("armors", (255, 255, 0)), ("rejected", (255, 0, 255))):
            for armor in det.get(key, []):
                pts = np.round(np.asarray(armor["corners"])).astype(int)
                cv2.polylines(bgr, [pts.reshape(-1, 1, 2)], True, color, 1)
                label = f"{armor['number']} {armor['confidence']:.2f}"
                cv2.putText(bgr, label, tuple(pts[1] + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(bgr, f"seq {seq} lights {det.get('n_lights', '?')}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bgr)
    return out
