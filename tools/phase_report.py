"""Filter errors per phase of a driven-session run (rec_drive*), live or replayed.

    python tools/phase_report.py RUN_DIR [RUN_DIR ...]

For each phase of DRIVE_PROFILE (from 1 s into it): mean radius error (both pairs), center error
and error of the plate the filter predicts nearest the one facing the camera.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nssim.metrics import frame_errors  # noqa: E402

PHASES = {4: "slow 1.1", 5: "mid 2.25", 6: "fast 4.5", 7: "stopped", 8: "fast again", 12: "slow again", 13: "stopped again"}


def phase_errors(run: Path) -> dict[str, tuple[float, float, float]]:
    cfg = json.loads((run / "config.json").read_text())
    segments = cfg["targets"][0]["motion"]["segments"]
    tracks = {}
    for line in open(run / "telemetry.jsonl"):
        m = json.loads(line)
        if m.get("type") == "track" and m.get("pf_alive"):
            tracks[m["seq"]] = m
    rows = []
    for line in open(run / "frames.jsonl"):
        f = json.loads(line)
        err = frame_errors(f, None, tracks.get(f["seq"]))["filter"]
        if err is not None:
            rows.append((f["gt"]["t"], (abs(err["radius_high_m"]) + abs(err["radius_low_m"])) / 2, err["center_xy_m"], err["aim_plate_m"]))
    rows = np.asarray(rows)
    ends = [seg[0] for seg in segments[1:]] + [rows[-1, 0]]
    out = {}
    for i, name in PHASES.items():
        w = (rows[:, 0] >= segments[i][0] + 1.0) & (rows[:, 0] < ends[i])
        if w.any():
            out[name] = tuple(float(np.nanmean(rows[w, k]) * 100) for k in (1, 2, 3))
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(Path(arg).name)
        for name, (r, c, a) in phase_errors(Path(arg)).items():
            print(f"  {name:14s} radius err {r:5.1f} cm   center err {c:5.1f} cm   aim plate err {a:5.1f} cm")
