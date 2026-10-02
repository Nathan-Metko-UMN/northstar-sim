"""Offline evaluation of a run directory: Northstar-CV's telemetry against ground truth.

Frames are matched by sequence number; the per-frame errors come from metrics.py (the same numbers
the live dashboard plots). All positions are in the CV's base frame (origin at our robot's base,
world axes).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from nssim.metrics import MATCH_M, MATCH_PX, MAX_INCIDENCE_DEG, frame_errors  # noqa: F401 (re-exported)
from nssim.metrics import pnp_order as _pnp_order  # noqa: F401 (older scripts import it from here)
from nssim.metrics import wrap as _wrap  # noqa: F401

FILTER_KEYS = ["center_xy_m", "center_z_m", "velocity_mps", "radius_high_m", "radius_low_m", "omega_radps", "orientation_rad"]


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def evaluate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    frames = {f["seq"]: f for f in _load(run_dir / "frames.jsonl")}
    telemetry = _load(run_dir / "telemetry.jsonl")
    dets = {m["seq"]: m for m in telemetry if m.get("type") == "det"}
    tracks = {m["seq"]: m for m in telemetry if m.get("type") == "track"}

    visible = detected = misread = 0
    lists = {key: [] for key in ("corner_px", "range_m", "lateral_m", "vertical_m")}
    filter_errors = {key: [] for key in FILTER_KEYS}
    for seq, frame in frames.items():
        errors = frame_errors(frame, dets.get(seq), tracks.get(seq))
        visible += errors["visible"]
        detected += errors["detected"]
        misread += errors["misread"]
        for key in lists:
            lists[key] += errors[key]
        if errors["filter"] is not None:
            for key in FILTER_KEYS:
                filter_errors[key].append(errors["filter"][key])

    return {
        "frames": len(frames),
        "telemetry": {"det": len(dets), "track": len(tracks)},
        "detection": {
            "visible_plates": visible,
            "recall": detected / visible if visible else None,
            "misread_numbers": misread,
            "corner_error_px": _stats(lists["corner_px"]),
        },
        "measurement_m": {
            "range": _stats(lists["range_m"]),
            "lateral": _stats(lists["lateral_m"]),
            "vertical": _stats(lists["vertical_m"]),
        },
        "filter": {key: _stats(values) for key, values in filter_errors.items()},
        "timing": {"processing_ms": _stats([f["processing_us"] / 1e3 for f in frames.values()])},
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
        f" radius high {_mean_rms(f['radius_high_m'], 100, ' cm')}",
        f" radius high {_mean_rms(f['radius_low_m'], 100, ' cm')}",
        f"           spin {_mean_rms(f['omega_radps'], 1, ' rad/s', 2)},"
        f" orientation {_mean_rms(f['orientation_rad'], 180 / math.pi, ' deg')}",
        f"CV time    {cv_ms['mean']:.1f} ms mean, {cv_ms['p95_abs']:.1f} ms p95" if cv_ms.get("n") else "CV time    n/a",
    ])


def main(run_dir: str) -> None:
    summary = evaluate(Path(run_dir))
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(format_summary(summary))
