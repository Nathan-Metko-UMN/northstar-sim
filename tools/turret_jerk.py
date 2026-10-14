"""Turret smoothness of closed-loop runs: how hard the turret's yaw rate changes frame to frame.

    python tools/turret_jerk.py RUN_DIR [RUN_DIR ...]

Prints, after the first 2 s: RMS yaw acceleration (rad/s^2), its 99th percentile, and the RMS of
the yaw rate's frame-to-frame change with plate switches (single steps over 1 rad/s) left out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def turret_jerk(run: Path) -> dict:
    t, rate = [], []
    for line in open(run / "frames.jsonl"):
        f = json.loads(line)
        t.append(f["gt"]["t"])
        rate.append(f["gt"]["turret"]["yaw_rate"])
    t, rate = np.asarray(t), np.asarray(rate)
    keep = t >= 2.0
    t, rate = t[keep], rate[keep]
    accel = np.diff(rate) / np.diff(t)
    d = np.abs(np.diff(rate))
    calm = d[d < 1.0]
    return {"accel_rms": float(np.sqrt(np.mean(accel ** 2))), "accel_p99": float(np.percentile(np.abs(accel), 99)),
            "rate_step_rms": float(np.sqrt(np.mean(calm ** 2))), "big_steps": int(np.sum(d >= 1.0))}


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        r = turret_jerk(Path(arg))
        print(f"{Path(arg).name:70s} accel rms {r['accel_rms']:7.1f}  p99 {r['accel_p99']:7.1f} rad/s^2  "
              f"rate step rms {r['rate_step_rms']:.3f} rad/s  big steps {r['big_steps']}")
