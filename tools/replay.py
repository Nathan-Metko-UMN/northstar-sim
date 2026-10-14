"""Replay recorded observations through tracker variants, without rendering or detecting again.

    python tools/replay.py inputs                       # recordings in runs/rec -> replay inputs
    python tools/replay.py run BATCH --variants base,vis2d --seeds 1,2,3 [--recs rec_oval3] [--input recorded,truth]
    python tools/replay.py report BATCH

Recordings are `nssim run rec_*` runs (turret holding still or following a script, so what the
camera saw doesn't depend on the tracker). For each frame they log what the tracker was given:
plate positions, the tracked robot type, dt and the pipeline delay. `inputs` writes that as
NorthstarReplay input ("recorded"), and again with every plate at its true position ("truth": same
detections, no measurement error), to tell measurement error from tracker behavior.

`run` replays every (recording, input, variant, seed) in one container, several at a time, through
Northstar-CV's NorthstarReplay. A 20 s recording replays in a few seconds.
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

from nssim.metrics import frame_errors  # noqa: E402
from pf_ab import VARIANTS, variant_env  # noqa: E402

RECS = REPO / "runs" / "rec"
OUT = REPO / "runs" / "replay"
CV_DIR = REPO / "external" / "Northstar-CV"
IMAGE = "northstar-cv:sim-jetpack7"
BUILD_DIR = "build/sim-jetpack7"


def _jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --- inputs -------------------------------------------------------------------------------------


def write_inputs(rec: Path) -> None:
    frames = {f["seq"]: f for f in _jsonl(rec / "frames.jsonl")}
    dets, delays = {}, {}
    for m in _jsonl(rec / "telemetry.jsonl"):
        if m.get("type") == "det":
            dets[m["seq"]] = m
        elif m.get("type") == "track" and m.get("pipeline_delay_s") is not None:
            delays[m["seq"]] = m["pipeline_delay_s"]
    if not dets or "closest_type" not in next(iter(dets.values())):
        raise SystemExit(f"{rec.name}: no closest_type in its detections; record it with the current Northstar-CV")
    lines = {"recorded": [], "truth": []}
    delay = 0.02
    for seq in sorted(dets):
        det = dets[seq]
        delay = delays.get(seq, delay)
        head = f"{seq} {det['capture_ns']} {det['dt']:.7f} {delay:.6f} {det['closest_type'] or 'p'} {len(det['plates'])}"
        true_plates = []
        if seq in frames:
            true_plates = [(t["number"], np.asarray(p["center_base"])) for t in frames[seq]["gt"]["targets"] for p in t["plates"]]
        for kind in lines:
            parts = [head]
            for plate in det["plates"]:
                pos = np.asarray(plate["position"], float)
                if kind == "truth" and true_plates:
                    pos = min(true_plates, key=lambda tp: np.linalg.norm(tp[1] - pos))[1]
                q = plate["orientation_xyzw"]
                parts.append(f"{plate['number']} {plate['type']} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                             f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
            lines[kind].append(" ".join(parts))
    for kind, rows in lines.items():
        (rec / f"replay_{kind}.txt").write_text("\n".join(rows) + "\n")


def cmd_inputs(args) -> None:
    for rec in sorted(RECS.iterdir()):
        if (rec / "summary.json").is_file() and (args.force or not (rec / "replay_truth.txt").is_file()):
            write_inputs(rec)
            print(f"[replay] inputs for {rec.name}", flush=True)


# --- run ----------------------------------------------------------------------------------------


def _robot_type(rec: Path) -> int:
    cfg = json.loads((rec / "config.json").read_text())
    return int(cfg.get("mcb", {}).get("robot_id", 3)) % 100


def cmd_run(args) -> None:
    batch = OUT / args.batch
    batch.mkdir(parents=True, exist_ok=True)
    recs = [r for r in sorted(RECS.iterdir()) if (r / "replay_truth.txt").is_file()
            and (not args.recs or any(r.name.startswith(p) for p in args.recs.split(",")))]
    jobs = []
    for rec in recs:
        for kind in args.input.split(","):
            for variant in args.variants.split(","):
                env = variant_env(variant)
                for seed in args.seeds.split(","):
                    name = f"{rec.name}--{kind}--{variant}--seed{seed}"
                    if (batch / name / "track.jsonl").is_file() and not args.force:
                        continue
                    (batch / name).mkdir(exist_ok=True)
                    assigns = " ".join(f"{k}={v}" for k, v in {**env, "PF_SEED": seed}.items())
                    jobs.append(f"env {assigns} ./{BUILD_DIR}/NorthstarReplay /runs/rec/{rec.name}/replay_{kind}.txt "
                                f"/runs/replay/{args.batch}/{name}/track.jsonl {_robot_type(rec)} "
                                f"> /runs/replay/{args.batch}/{name}/replay.log 2>&1 || echo FAILED {name}")
    if not jobs:
        print("[replay] nothing to run")
        return
    script = batch / "jobs.sh"
    script.write_text("\n".join(jobs) + "\n", newline="\n")
    print(f"[replay] {len(jobs)} replays, {args.parallel} at a time", flush=True)
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{CV_DIR}:/ws", "-v", f"{REPO / 'runs'}:/runs", "-w", "/ws",
           IMAGE, "bash", "-c", f"xargs -P {args.parallel} -I CMD bash -c CMD < /runs/replay/{args.batch}/jobs.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    failed = [line for line in result.stdout.splitlines() if line.startswith("FAILED")]
    if result.returncode != 0 or failed:
        print(result.stdout[-3000:], result.stderr[-3000:])
        raise SystemExit(f"[replay] {len(failed)} replays failed")
    print("[replay] done", flush=True)


# --- metrics ------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _frames(rec: Path) -> dict:
    return {f["seq"]: f for f in _jsonl(rec / "frames.jsonl")}


def series(run: Path) -> dict:
    """Per-frame filter state and errors of a replay, against its recording's ground truth."""
    rec = RECS / run.name.split("--")[0]
    frames = _frames(rec)
    first_obs = {}
    kind = run.name.split("--")[1]
    for line in (rec / f"replay_{kind}.txt").read_text().splitlines():
        f = line.split()
        if int(f[5]) > 0:
            first_obs[int(f[0])] = np.array([float(x) for x in f[8:11]])
    rows = []
    for m in _jsonl(run / "track.jsonl"):
        if m.get("type") != "track" or not m.get("pf_alive") or m["seq"] not in frames:
            continue
        frame = frames[m["seq"]]
        err = frame_errors(frame, None, m)["filter"]
        if err is None:
            continue
        target = next(t for t in frame["gt"]["targets"] if t["name"] == err["target"])
        seen = -1
        if m["seq"] in first_obs:
            dists = [np.linalg.norm(np.asarray(p["center_base"]) - first_obs[m["seq"]]) for p in target["plates"]]
            seen = int(np.argmin(dists)) if min(dists) < 0.3 else -1
        st = m["state"]
        rows.append({
            "t": frame["gt"]["t"], "seq": m["seq"], "target": err["target"], "seen": seen,
            "cx": st["center"][0], "cy": st["center"][1], "cz": st["center"][2],
            "vel_err": float(np.linalg.norm(np.asarray(st["center_velocity"]) - np.asarray(target["velocity"]))),
            "yaw": (m.get("aim") or {}).get("yaw", math.nan),
            **{k: err[k] for k in ("center_xy_m", "z_offset_m", "radius_high_m", "radius_low_m", "orientation_rad",
                                   "labeling_ok", "plates_m", "aim_plate_m", "omega", "true_omega", "z_offset",
                                   "radius_high", "radius_low", "true_radius_high", "true_radius_low", "true_z_offset")},
        })
    return {k: np.asarray([r[k] for r in rows]) for k in (rows[0] if rows else {})}


def _segments(s: dict) -> list[np.ndarray]:
    """Index ranges over which the tracker stayed on one robot."""
    if not len(s.get("t", [])):
        return []
    cut = np.nonzero((s["target"][1:] != s["target"][:-1]) | (np.diff(s["seq"]) > 3))[0] + 1
    return [seg for seg in np.split(np.arange(len(s["t"])), cut) if len(seg) > 10]


def jerk_metrics(s: dict, settle_s: float = 2.0) -> dict:
    steps, kicks, ripple, yaw_steps, vel = [], [], [], [], []
    for seg in _segments(s):
        t = s["t"][seg]
        keep = seg[t >= t[0] + settle_s]
        if len(keep) < 10:
            continue
        c = np.c_[s["cx"][keep], s["cy"][keep]]
        step = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)]
        steps.append(step[1:])
        seen = s["seen"][keep]
        sw = np.nonzero((seen[1:] != seen[:-1]) & (seen[1:] >= 0) & (seen[:-1] >= 0))[0] + 1
        kicks += [step[i:i + 5].sum() for i in sw if i + 5 <= len(step)]
        ripple.append(s["omega"][keep] - s["true_omega"][keep])
        dyaw = np.abs(np.diff(s["yaw"][keep]))
        yaw_steps.append(dyaw[np.isfinite(dyaw) & (dyaw < math.radians(2))])
        vel.append(s["vel_err"][keep])
    if not steps:
        return {}
    steps, ripple, yaw_steps, vel = map(np.concatenate, (steps, ripple, yaw_steps, vel))
    return {
        "step_mm": float(steps.mean() * 1000),
        "step_p99_mm": float(np.percentile(steps, 99) * 1000),
        "kick_mm": float(np.mean(kicks) * 1000) if kicks else math.nan,
        "vel_err": float(vel.mean()),
        "spin_ripple": float(ripple.std()),
        "yaw_step_p95_mrad": float(np.percentile(yaw_steps, 95) * 1000) if len(yaw_steps) else math.nan,
    }


def summarize(run: Path) -> dict:
    s = series(run)
    if not len(s.get("t", [])):
        return {"run": run.name, "frames": 0}
    out = {"run": run.name, "frames": int(len(s["t"]))}
    t = s["t"]
    late = t >= t[-1] - 5.0
    out.update({
        "aim_cm": float(np.nanmean(s["aim_plate_m"]) * 100),
        "aim_p95_cm": float(np.nanpercentile(s["aim_plate_m"], 95) * 100),
        "plates_cm": float(np.mean(s["plates_m"]) * 100),
        "center_cm": float(np.mean(s["center_xy_m"][late]) * 100),
        "labeling_ok": float(np.mean(s["labeling_ok"][late])),
        "dz_err_mm": float(np.mean(s["z_offset_m"][late]) * 1000),
        "r_high_err_cm": float(np.mean(s["radius_high_m"][late]) * 100),
        "r_low_err_cm": float(np.mean(s["radius_low_m"][late]) * 100),
        "geom_sd_mm": float(np.mean([np.std(s[k][late]) for k in ("radius_high", "radius_low", "z_offset")]) * 1000),
        **jerk_metrics(s),
    })
    # Switching recordings: each look at a robot, first visit or a return.
    cfg = json.loads((RECS / run.name.split("--")[0] / "config.json").read_text())
    script = cfg.get("aim_script") or []
    if len(script) > 1:
        visits, revisit_err, first_err, acquire = {}, [], [], {"first": [], "again": []}
        ends = [start for start, _ in script[1:]] + [t[-1]]
        for (start, name), end in zip(script, ends):
            visit = visits.get(name, 0)
            visits[name] = visit + 1
            on = (s["target"] == name) & (t >= start) & (t < end)
            if not on.any():
                acquire["first" if visit == 0 else "again"].append(math.inf)
                continue
            ts, err = t[on], s["aim_plate_m"][on]
            good = math.inf
            for i in range(len(ts)):
                w = (ts >= ts[i]) & (ts < ts[i] + 0.5)
                if ts[i] + 0.5 <= ts[-1] and np.all(err[w] < 0.05):
                    good = float(ts[i] - start)
                    break
            acquire["first" if visit == 0 else "again"].append(good)
            early = on & (t < start + 1.5)
            geom = float(np.mean(np.abs(s["radius_high_m"][early])) + np.mean(np.abs(s["radius_low_m"][early]))) * 50 \
                if early.any() else math.nan
            (first_err if visit == 0 else revisit_err).append(geom)
        out["acquire_first_s"] = float(np.mean(acquire["first"])) if acquire["first"] else math.nan
        out["acquire_again_s"] = float(np.mean(acquire["again"])) if acquire["again"] else math.nan
        out["early_r_err_first_cm"] = float(np.nanmean(first_err)) if first_err else math.nan
        out["early_r_err_again_cm"] = float(np.nanmean(revisit_err)) if revisit_err else math.nan
    return out


COLUMNS = [
    ("aim_cm", "aim err cm", "{:.1f}"), ("aim_p95_cm", "… p95", "{:.1f}"),
    ("kick_mm", "switch kick mm", "{:.1f}"), ("step_mm", "center step mm", "{:.2f}"),
    ("step_p99_mm", "… p99", "{:.1f}"), ("vel_err", "vel err m/s", "{:.3f}"),
    ("spin_ripple", "spin ripple rad/s", "{:.2f}"), ("yaw_step_p95_mrad", "aim yaw step p95 mrad", "{:.2f}"),
    ("center_cm", "center err cm", "{:.1f}"), ("labeling_ok", "high pair right", "{:.0%}"),
    ("dz_err_mm", "dz err mm", "{:+.1f}"), ("r_high_err_cm", "r high err cm", "{:+.1f}"),
    ("r_low_err_cm", "r low err cm", "{:+.1f}"), ("geom_sd_mm", "geometry sd mm", "{:.2f}"),
    ("acquire_first_s", "acquire 1st visit s", "{:.2f}"), ("acquire_again_s", "acquire revisit s", "{:.2f}"),
    ("early_r_err_first_cm", "radius err 1st 1.5 s, 1st visit cm", "{:.1f}"),
    ("early_r_err_again_cm", "… revisit cm", "{:.1f}"),
]


def _family(rec_name: str) -> str:
    return rec_name.split("__")[0]


def cmd_report(args) -> None:
    batch = OUT / args.batch
    groups: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for run in sorted(p for p in batch.iterdir() if (p / "track.jsonl").is_file()):
        cache = run / "summary.json"
        if cache.is_file() and cache.stat().st_mtime > (run / "track.jsonl").stat().st_mtime:
            summary = json.loads(cache.read_text())
        else:
            summary = summarize(run)
            cache.write_text(json.dumps(summary, indent=2))
        rec, kind, variant, _ = run.name.split("--")
        groups.setdefault((_family(rec), kind), {}).setdefault(variant, []).append(summary)
    lines = [f"# Replay batch {args.batch}", "", "Rows average a variant over the recording's starting angles and seeds.", ""]
    for (family, kind), variants in sorted(groups.items()):
        cols = [c for c in COLUMNS if any(c[0] in s for runs in variants.values() for s in runs)]
        lines += [f"## {family} ({kind})", "", "| variant | n | " + " | ".join(c[1] for c in cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for variant, runs in variants.items():
            cells = []
            for key, _, fmt in cols:
                vals = [r[key] for r in runs if key in r and r[key] is not None]
                finite = [v for v in vals if math.isfinite(v)]
                cell = fmt.format(float(np.mean(finite))) if finite else ""
                if len(finite) < len(vals):
                    cell += f" ({len(vals) - len(finite)} never)"
                cells.append(cell)
            lines.append(f"| {variant} | {len(runs)} | " + " | ".join(cells) + " |")
        lines.append("")
    report = batch / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def plot(runs: list[Path], out: Path, title: str, t_range=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for i, run in enumerate(runs):
        s = series(run)
        if not len(s.get("t", [])):
            continue
        c = f"C{i}"
        label = run.name.split("--")[2]
        tc = s["t"]
        axes[0].plot(tc, s["center_xy_m"] * 100, c, label=label, lw=0.8)
        axes[1].plot(tc, s["z_offset"] * 100, c, lw=0.8)
        axes[1].plot(tc, s["true_z_offset"] * 100, "k--", lw=0.6)
        axes[2].plot(tc, s["radius_high"] * 100, c, lw=0.8)
        axes[2].plot(tc, s["radius_low"] * 100, c, ls=":", lw=0.8)
        axes[2].plot(tc, s["true_radius_high"] * 100, "k--", lw=0.6)
        axes[2].plot(tc, s["true_radius_low"] * 100, "k:", lw=0.6)
        axes[3].plot(tc, s["aim_plate_m"] * 100, c, lw=0.8)
        axes[4].plot(tc, s["omega"], c, lw=0.8)
        axes[4].plot(tc, s["true_omega"], "k--", lw=0.6)
    for ax, label in zip(axes, ["center err (cm)", "height offset (cm)", "radius (cm)\nsolid high, dotted low",
                                "aim plate err (cm)", "spin (rad/s)"]):
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    if t_range:
        axes[0].set_xlim(*t_range)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("sim time (s); dashed black = truth")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def cmd_plot(args) -> None:
    batch = OUT / args.batch
    runs = [batch / name for name in args.runs]
    plot(runs, batch / args.out, args.title or args.out, tuple(args.t_range) if args.t_range else None)
    print(batch / args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_inputs)
    p = sub.add_parser("run")
    p.add_argument("batch")
    p.add_argument("--variants", default="base")
    p.add_argument("--seeds", default="1")
    p.add_argument("--recs", help="comma-separated recording name prefixes (default: all)")
    p.add_argument("--input", default="recorded", help="recorded,truth")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("batch")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("plot")
    p.add_argument("batch")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="plot.png")
    p.add_argument("--title")
    p.add_argument("--t-range", nargs=2, type=float)
    p.set_defaults(func=cmd_plot)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
