"""Northstar-CV's errors on one frame, against the ground truth recorded for it.

eval.py adds these up over a run; the live dashboard plots them frame by frame. All positions are
in the CV's base frame (origin at our robot's base, world axes).
"""

from __future__ import annotations

import math

import numpy as np

MATCH_PX = 30.0  # max centroid distance to call a detection a given plate
MATCH_M = 0.3  # max distance to match a measured plate to a ground-truth plate
MAX_INCIDENCE_DEG = 70.0  # plates seen more edge-on than this don't count against recall
IMAGE_SIZE = (1440, 1080)


def pnp_order(corners: np.ndarray) -> np.ndarray:
    """Order 4 pixels like the detector: left bottom, left top, right top, right bottom."""
    by_x = corners[np.argsort(corners[:, 0])]
    left, right = by_x[:2], by_x[2:]
    left = left[np.argsort(-left[:, 1])]  # bottom (larger y) first
    right = right[np.argsort(right[:, 1])]  # top first
    return np.array([left[0], left[1], right[0], right[1]])


def wrap(a, period):
    return (np.asarray(a) + period / 2) % period - period / 2


def frame_errors(frame: dict, det: dict | None, track: dict | None) -> dict:
    """Errors on one frame. ``frame`` is a frames.jsonl record; ``det`` / ``track`` are the CV's
    telemetry for the same sequence number (None if it sent none)."""
    gt = frame["gt"]
    cam_p = np.asarray(gt["camera_world"]["p"]) - np.asarray(gt["base_world"])
    out = {
        "seq": frame["seq"],
        "visible": 0,  # plates the CV should see: facing, unblocked, in the image, not too edge-on
        "detected": 0,
        "misread": 0,
        "corner_px": [],
        "range_m": [],
        "lateral_m": [],
        "vertical_m": [],
        "filter": None,
    }
    for target in gt["targets"]:
        for plate in target["plates"]:
            px = np.asarray(plate["corners_px"], float)
            if not plate["facing"] or plate.get("occluded") or not np.all(np.isfinite(px)):
                continue
            if np.any(px < 0) or np.any(px[:, 0] > IMAGE_SIZE[0] - 1) or np.any(px[:, 1] > IMAGE_SIZE[1] - 1):
                continue
            to_cam = cam_p - np.asarray(plate["center_base"])
            cos_inc = float(np.dot(plate["normal"], to_cam) / np.linalg.norm(to_cam))
            if math.degrees(math.acos(min(1.0, cos_inc))) > MAX_INCIDENCE_DEG:
                continue
            out["visible"] += 1
            if det is None:
                continue
            gt_px = pnp_order(px)
            best = None
            for armor in det["armors"]:
                c = np.asarray(armor["corners"], float)
                d = np.linalg.norm(c.mean(0) - gt_px.mean(0))
                if d < MATCH_PX and (best is None or d < best[0]):
                    best = (d, armor, c)
            if best is None:
                continue
            out["detected"] += 1
            if best[1]["number"] != target["number"]:
                out["misread"] += 1
            out["corner_px"] += [float(e) for e in np.linalg.norm(best[2] - gt_px, axis=1)]

    if det is not None:
        gt_plates = [np.asarray(p["center_base"]) for t in gt["targets"] for p in t["plates"]]
        for plate in det["plates"]:
            pos = np.asarray(plate["position"])
            dists = [np.linalg.norm(pos - c) for c in gt_plates]
            if not dists or min(dists) > MATCH_M:
                continue
            gt_c = gt_plates[int(np.argmin(dists))]
            err = pos - gt_c
            ray = (gt_c - cam_p) / np.linalg.norm(gt_c - cam_p)
            horiz = np.cross(ray, [0.0, 0.0, 1.0])
            horiz /= np.linalg.norm(horiz)
            out["range_m"].append(float(err @ ray))
            out["lateral_m"].append(float(err @ horiz))
            out["vertical_m"].append(float(err @ np.cross(horiz, ray)))

    if track and track.get("pf_alive") and gt["targets"]:
        state = track["state"]
        center = np.asarray(state["center"])
        # The enemy the filter is on (a run can have several): the true center nearest its estimate.
        target = min(gt["targets"], key=lambda tg: np.linalg.norm(np.asarray(tg["center_base"])[:2] - center[:2]))
        true_center = np.asarray(target["center_base"])
        true_radius_high = target["radius_high"]
        true_radius_low = target["radius_low"]
        out["filter"] = {
            "target": target["name"],
            "center_xy_m": float(np.linalg.norm(center[:2] - true_center[:2])),
            "center_z_m": float(center[2] - true_center[2]),
            "velocity_mps": float(np.linalg.norm(np.asarray(state["center_velocity"]) - np.asarray(target["velocity"]))),
            "radius_high_m": float(state["radius_0"] - true_radius_high),
            "radius_low_m": float(state["radius_1"] - true_radius_low),
            "omega_radps": float(state["omega"] - target["omega"]),
            "orientation_rad": float(wrap(state["orientation"] - target["spin"], math.pi)),
            # What the errors are relative to, for plotting estimate against truth.
            "omega": float(state["omega"]),
            "true_omega": float(target["omega"]),
            "radius_high": float(state["radius_0"]),
            "radius_low": float(state["radius_1"]),
            "true_radius_high": float(true_radius_high),
            "true_radius_low": float(true_radius_low),
        }
    return out
