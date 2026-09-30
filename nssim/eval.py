"""Offline evaluation of a run directory: Northstar-CV's telemetry against ground truth.

Frames are matched by sequence number. All positions are in the CV's base frame (origin at our
robot's base, world axes).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MATCH_PX = 30.0  # max centroid distance to call a detection a given plate
MATCH_M = 0.3  # max distance to match a measured plate to a ground-truth plate
MAX_INCIDENCE_DEG = 70.0  # plates seen more edge-on than this don't count against recall


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pnp_order(corners: np.ndarray) -> np.ndarray:
    """Order 4 pixels like the detector: left bottom, left top, right top, right bottom."""
    by_x = corners[np.argsort(corners[:, 0])]
    left, right = by_x[:2], by_x[2:]
    left = left[np.argsort(-left[:, 1])]  # bottom (larger y) first
    right = right[np.argsort(right[:, 1])]  # top first
    return np.array([left[0], left[1], right[0], right[1]])


def _stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "rms": float(np.sqrt((v**2).mean())),
        "p95_abs": float(np.percentile(np.abs(v), 95)),
        "max_abs": float(np.abs(v).max()),
    }


def _wrap(a, period):
    return (np.asarray(a) + period / 2) % period - period / 2


def evaluate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    frames = {f["seq"]: f for f in _load(run_dir / "frames.jsonl")}
    telemetry = _load(run_dir / "telemetry.jsonl")
    dets = {m["seq"]: m for m in telemetry if m.get("type") == "det"}
    tracks = {m["seq"]: m for m in telemetry if m.get("type") == "track"}

    corner_err, range_err, lateral_err, vertical_err = [], [], [], []
    visible, detected, misread = 0, 0, 0
    center_xy_err, center_z_err, vel_err, radius_err, omega_err, orient_err = [], [], [], [], [], []
    processing_ms = [f["processing_us"] / 1e3 for f in frames.values()]

    for seq, frame in frames.items():
        gt = frame["gt"]
        cam_p = np.asarray(gt["camera_world"]["p"]) - np.asarray(gt["base_world"])
        det = dets.get(seq)
        for target in gt["targets"]:
            for plate in target["plates"]:
                px = np.asarray(plate["corners_px"], float)
                if not plate["facing"] or not np.all(np.isfinite(px)):
                    continue
                if np.any(px[:, 0] < 0) or np.any(px[:, 0] > 1439) or np.any(px[:, 1] < 0) or np.any(px[:, 1] > 1079):
                    continue
                to_cam = cam_p - np.asarray(plate["center_base"])
                cos_inc = float(np.dot(plate["normal"], to_cam) / np.linalg.norm(to_cam))
                if math.degrees(math.acos(min(1.0, cos_inc))) > MAX_INCIDENCE_DEG:
                    continue
                visible += 1
                if det is None:
                    continue
                gt_px = _pnp_order(px)
                best = None
                for armor in det["armors"]:
                    c = np.asarray(armor["corners"], float)
                    d = np.linalg.norm(c.mean(0) - gt_px.mean(0))
                    if d < MATCH_PX and (best is None or d < best[0]):
                        best = (d, armor, c)
                if best is None:
                    continue
                detected += 1
                if best[1]["number"] != target["number"]:
                    misread += 1
                corner_err += list(np.linalg.norm(best[2] - gt_px, axis=1))

        if det is not None:
            gt_plates = [(np.asarray(p["center_base"]), t) for t in gt["targets"] for p in t["plates"]]
            for plate in det["plates"]:
                pos = np.asarray(plate["position"])
                dists = [np.linalg.norm(pos - c) for c, _ in gt_plates]
                if not dists or min(dists) > MATCH_M:
                    continue
                gt_c = gt_plates[int(np.argmin(dists))][0]
                err = pos - gt_c
                ray = (gt_c - cam_p) / np.linalg.norm(gt_c - cam_p)
                range_err.append(float(err @ ray))
                horiz = np.cross(ray, [0.0, 0.0, 1.0])
                horiz /= np.linalg.norm(horiz)
                lateral_err.append(float(err @ horiz))
                vertical_err.append(float(err @ np.cross(horiz, ray)))

        track = tracks.get(seq)
        if track and track.get("pf_alive") and gt["targets"]:
            target = gt["targets"][0]
            state = track["state"]
            center = np.asarray(state["center"])
            gt_center = np.asarray(target["center_base"])
            center_xy_err.append(float(np.linalg.norm(center[:2] - gt_center[:2])))
            center_z_err.append(float(center[2] - gt_center[2]))
            vel_err.append(float(np.linalg.norm(np.asarray(state["center_velocity"]) - np.asarray(target["velocity"]))))
            radius_err.append(float(state["radius"] - 0.5 * (target["radius_high"] + target["radius_low"])))
            omega_err.append(float(state["omega"] - target["omega"]))
            orient_err.append(float(_wrap(state["orientation"] - target["spin"], math.pi)))

    return {
        "frames": len(frames),
        "telemetry": {"det": len(dets), "track": len(tracks)},
        "detection": {
            "visible_plates": visible,
            "recall": detected / visible if visible else None,
            "misread_numbers": misread,
            "corner_error_px": _stats(corner_err),
        },
        "measurement_m": {
            "range": _stats(range_err),
            "lateral": _stats(lateral_err),
            "vertical": _stats(vertical_err),
        },
        "filter": {
            "center_xy_m": _stats(center_xy_err),
            "center_z_m": _stats(center_z_err),
            "velocity_mps": _stats(vel_err),
            "radius_m": _stats(radius_err),
            "omega_radps": _stats(omega_err),
            "orientation_rad": _stats(orient_err),
        },
        "timing": {"processing_ms": _stats(processing_ms)},
    }


def _mean_rms(stat: dict, scale: float, unit: str, digits: int = 1) -> str:
    if not stat.get("n"):
        return "n/a"
    return f"{stat['mean'] * scale:+.{digits}f}{unit} (rms {stat['rms'] * scale:.{digits}f})"


def format_summary(summary: dict) -> str:
    """The headline numbers of evaluate(), one topic per line."""
    d, m, f = summary["detection"], summary["measurement_m"], summary["filter"]
    recall = "n/a" if d["recall"] is None else f"{d['recall']:.2f}"
    corners = d["corner_error_px"]
    cv_ms = summary["timing"]["processing_ms"]
    return "\n".join([
        f"frames     {summary['frames']}",
        f"detection  recall {recall} of {d['visible_plates']} visible plates, {d['misread_numbers']} misread"
        + (f", corners {corners['rms']:.2f} px rms" if corners.get("n") else ""),
        f"plates     range {_mean_rms(m['range'], 100, ' cm')}, lateral {_mean_rms(m['lateral'], 1000, ' mm')},"
        f" vertical {_mean_rms(m['vertical'], 1000, ' mm')}",
        f"filter     center {_mean_rms(f['center_xy_m'], 100, ' cm')}, velocity {_mean_rms(f['velocity_mps'], 1, ' m/s', 2)},"
        f" radius {_mean_rms(f['radius_m'], 100, ' cm')}",
        f"           spin {_mean_rms(f['omega_radps'], 1, ' rad/s', 2)},"
        f" orientation {_mean_rms(f['orientation_rad'], 180 / math.pi, ' deg')}",
        f"CV time    {cv_ms['mean']:.1f} ms mean, {cv_ms['p95_abs']:.1f} ms p95" if cv_ms.get("n") else "CV time    n/a",
    ])


def main(run_dir: str) -> None:
    summary = evaluate(Path(run_dir))
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(format_summary(summary))
