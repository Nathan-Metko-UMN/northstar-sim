"""Tracker benchmark over a fixed set of recordings, for tuning (replays only; nothing is rendered).

    python tools/bench.py VARIANT [VARIANT ...] [--seeds 0,1] [--ref next3]

A variant is '+'-joined parts: preset or tools/pf_ab.py variant names, or NAME=VALUE settings
(e.g. next3+PF_YAW_SLOW_FACTOR=8). Every variant is replayed over:
  drive   the 12 tracker-test runs (each robot type and plate layout driving its tour)
  still   still robots showing one plate, turret following the aim (and nssim drive's start)
  switch  three robots, the turret looking at one, then another, and back
  stuck   a driven session with a close pass that used to leave the height offset stuck at 0
and scored on each. Results are cached per (recording, variant, seed); the table compares every
variant with --ref.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import replay  # noqa: E402
from nssim.metrics import frame_errors  # noqa: E402
from pf_ab import variant_env as _named_env  # noqa: E402

RECS = REPO / "runs" / "rec"
BATCH = "bench"
OUT = REPO / "runs" / "replay" / BATCH


def sets() -> dict[str, list[str]]:
    names = sorted(p.name for p in RECS.iterdir() if (p / "replay_recorded.txt").is_file())
    return {
        "drive": [n for n in names if n.startswith("pftest_") and n.endswith("__n3")],
        "still": [n for n in names if n.startswith("still_") and n.endswith("__next3_cl")] + ["drive_start__next3b"],
        "switch": [n for n in names if n.startswith("rec_switch3__")],
        "stuck": ["user_drive_stuck"],
    }


def variant_env(variant: str) -> dict[str, str]:
    env, named = {}, []
    for part in variant.split("+"):
        if "=" in part:
            k, v = part.split("=", 1)
            env[k] = v
        else:
            named.append(part)
    return {**(_named_env("+".join(named)) if named else {}), **env}


def run(variants: list[str], seeds: list[str], parallel: int = 6) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for group in sets().values():
        for rec in group:
            robot_type = replay._robot_type(RECS / rec)
            for variant in variants:
                env = variant_env(variant)
                for seed in seeds:
                    name = f"{rec}--recorded--{variant}--seed{seed}"
                    if (OUT / name / "track.jsonl").is_file():
                        continue
                    (OUT / name).mkdir(exist_ok=True)
                    assigns = " ".join(f"{k}={v}" for k, v in {**env, "PF_SEED": seed}.items())
                    jobs.append(f"env {assigns} ./{replay.BUILD_DIR}/NorthstarReplay /runs/rec/{rec}/replay_recorded.txt "
                                f"/runs/replay/{BATCH}/{name}/track.jsonl {robot_type} "
                                f"> /runs/replay/{BATCH}/{name}/replay.log 2>&1 || echo FAILED {name}")
    if not jobs:
        return
    script = OUT / "jobs.sh"
    script.write_text("\n".join(jobs) + "\n", newline="\n")
    print(f"[bench] {len(jobs)} replays", flush=True)
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{replay.CV_DIR}:/ws", "-v", f"{REPO / 'runs'}:/runs", "-w", "/ws",
           replay.IMAGE, "bash", "-c", f"xargs -P {parallel} -I CMD bash -c CMD < /runs/replay/{BATCH}/jobs.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    failed = [line for line in result.stdout.splitlines() if line.startswith("FAILED")]
    if failed:
        raise SystemExit(f"[bench] {len(failed)} replays failed, e.g. {failed[0]}")


# --- metrics --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _frames(rec: str) -> dict:
    return {f["seq"]: f for f in replay._jsonl(RECS / rec / "frames.jsonl")}


def _tracks(path: Path) -> dict:
    return {m["seq"]: m for m in replay._jsonl(path) if m.get("type") == "track"}


def drive_metrics(rec: str, track: Path) -> dict:
    frames = _frames(rec)
    name = json.loads((RECS / rec / "config.json").read_text())["targets"][0]["spec"]["name"]
    tracks = _tracks(track)
    t, aim, center, om, true_om = [], [], [], [], []
    for seq in sorted(frames):
        f = frames[seq]
        target = next(x for x in f["gt"]["targets"] if x["name"] == name)
        m = tracks.get(seq)
        err = frame_errors({**f, "gt": {**f["gt"], "targets": [target]}}, None, m)["filter"]
        on = False
        if err is not None:
            est = np.asarray(m["state"]["center"])
            nearest = min(f["gt"]["targets"], key=lambda x: np.linalg.norm(np.asarray(x["center_base"])[:2] - est[:2]))
            on = nearest["name"] == name
        t.append(f["gt"]["t"])
        aim.append(err["aim_plate_m"] if on else math.nan)
        center.append(err["center_xy_m"] if on else math.nan)
        om.append(err["omega"] if err else math.nan)
        true_om.append(target["omega"])
    t, aim, center, om, true_om = map(np.asarray, (t, aim, center, om, true_om))
    spiking = np.nan_to_num(aim) > 0.10
    # how long after each change of spin the estimate is within 0.5 rad/s for 0.5 s
    lags = []
    changes = np.nonzero(np.abs(np.diff(true_om)) > 1e-3)[0]
    starts = [changes[0]] + [c for a, c in zip(changes, changes[1:]) if t[c] - t[a] > 1.0] if len(changes) else []
    ok = np.abs(np.nan_to_num(om, nan=99) - true_om) < 0.5
    for c in starts:
        for k in range(c, len(t)):
            w = (t >= t[k]) & (t < t[k] + 0.5)
            if np.all(ok[w]):
                lags.append(t[k] - t[c])
                break
    return {"aim": float(np.nanmean(aim)), "aim_p99": float(np.nanpercentile(aim, 99)), "center": float(np.nanmean(center)),
            "spike_frames": float(spiking.mean()), "spikes": int(np.sum(spiking[1:] & ~spiking[:-1])),
            "spin_lag": float(np.mean(lags)) if lags else math.nan, "spin_err": float(np.nanmean(np.abs(om - true_om)))}


def still_metrics(rec: str, track: Path) -> dict:
    frames = _frames(rec)
    rows = [m for m in replay._jsonl(track) if m.get("type") == "track" and m.get("pf_alive") and m.get("aim")
            and m["seq"] in frames and frames[m["seq"]]["gt"]["t"] > 3.0]
    yaw = np.array([m["aim"]["yaw"] for m in rows])
    om = np.array([m["state"]["omega"] for m in rows])
    return {"still_step": float(np.sqrt(np.mean(np.diff(yaw) ** 2)) * 1e3), "still_sd": float(yaw.std() * 1e3),
            "still_spin_sd": float(om.std())}


def switch_metrics(rec: str, track: Path) -> dict:
    run_dir = track.parent
    s = replay.summarize(run_dir)
    return {"switch_aim": s.get("aim_cm", math.nan) / 100, "reacquire": s.get("acquire_again_s", math.nan),
            "revisit_r_err": s.get("early_r_err_again_cm", math.nan) / 100}


def stuck_metrics(rec: str, track: Path) -> dict:
    frames = _frames(rec)
    rows = [(frames[m["seq"]]["gt"]["t"], m["state"]["z_offset"]) for m in replay._jsonl(track)
            if m.get("type") == "track" and m.get("pf_alive") and m["seq"] in frames]
    r = np.array(rows)
    w = r[:, 0] >= 70
    return {"stuck_dz_err": float(np.mean(np.abs(r[w, 1] - 0.03)))}


METRICS = {"drive": drive_metrics, "still": still_metrics, "switch": switch_metrics, "stuck": stuck_metrics}


def metrics(variant: str, seeds: list[str]) -> dict:
    out: dict[str, list] = {}
    for group, recs in sets().items():
        for rec in recs:
            for seed in seeds:
                run_dir = OUT / f"{rec}--recorded--{variant}--seed{seed}"
                cache = run_dir / "bench.json"
                if cache.is_file():
                    m = json.loads(cache.read_text())
                else:
                    m = METRICS[group](rec, run_dir / "track.jsonl")
                    cache.write_text(json.dumps(m))
                for k, v in m.items():
                    out.setdefault(k, []).append(v)
    return {k: float(np.nanmean(v)) if k != "spikes" else float(np.sum(v) / len(seeds)) for k, v in out.items()}


# Lower is better for all; the score is the mean of each one's ratio to the reference.
SCORED = ["aim", "aim_p99", "center", "spike_frames", "spin_lag", "still_step", "reacquire", "stuck_dz_err"]
COLUMNS = [("aim", "aim cm", 100, "{:.2f}"), ("aim_p99", "aim p99 cm", 100, "{:.1f}"), ("center", "center cm", 100, "{:.2f}"),
           ("spike_frames", "spike %", 100, "{:.2f}"), ("spikes", "spikes", 1, "{:.0f}"), ("spin_lag", "spin lag s", 1, "{:.2f}"),
           ("spin_err", "spin err", 1, "{:.3f}"), ("still_step", "still step mrad", 1, "{:.3f}"),
           ("still_sd", "still sd mrad", 1, "{:.3f}"), ("reacquire", "reacquire s", 1, "{:.2f}"),
           ("revisit_r_err", "revisit r err cm", 100, "{:.1f}"), ("stuck_dz_err", "stuck dz err mm", 1000, "{:.1f}")]


def table(variants: list[str], seeds: list[str], ref: str) -> str:
    results = {v: metrics(v, seeds) for v in variants}
    base = results.get(ref) or metrics(ref, seeds)
    lines = ["| variant | score | " + " | ".join(c[1] for c in COLUMNS) + " |", "|---|---|" + "---|" * len(COLUMNS)]
    for v, m in results.items():
        score = np.mean([m[k] / base[k] for k in SCORED if base.get(k)])
        lines.append(f"| {v} | {score:.3f} | " + " | ".join(f.format(m.get(k, math.nan) * s) for k, _, s, f in COLUMNS) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--ref", default="next3")
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args()
    seeds = args.seeds.split(",")
    variants = list(dict.fromkeys([args.ref, *args.variants]))
    run(variants, seeds, args.parallel)
    report = table(variants, seeds, args.ref)
    print(report)
    with open(OUT / "results.md", "a", encoding="utf-8") as f:
        f.write(f"\n### seeds {args.seeds}\n\n{report}\n")


if __name__ == "__main__":
    main()
