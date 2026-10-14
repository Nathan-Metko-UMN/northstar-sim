"""Particle-filter A/B runs: scenarios x filter variants through `nssim run`, then a comparison.

    python tools/pf_ab.py run BATCH --scenario still_long --variants base,swap+relabel --spin0 0,25,-25
    python tools/pf_ab.py report BATCH

Variants are '+'-joined names from VARIANTS; each sets Northstar-CV environment variables (see its
src/sim/pf_overrides.hpp), so one build serves them all. Runs land in runs/pf_ab/BATCH/ and are
skipped if already there (summary.json), so a batch can be extended or resumed.

The PF seeds its random generator the same way every time, so repeating a run gives nearly the same
result; --spin0 (the enemy's starting angle, degrees) gives real variety instead.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nssim.metrics import frame_errors  # noqa: E402

NSSIM = REPO / ".venv" / ("Scripts/nssim.exe" if sys.platform == "win32" else "bin/nssim")
BATCHES = REPO / "runs" / "pf_ab"

VARIANTS = {
    "base": {},  # the code as it was: radius gate, reflect at 0, 3-way average, z variance 0.002
    "swap": {"PF_SWAP_REDUCTION": "1"},
    "vis2d": {"PF_VISIBILITY_XY": "1"},
    "relabel": {"PF_RELABEL_AT_ZERO": "1"},
    "freeradius": {"PF_RADIUS_MODE": "1"},
    "pull1": {"PF_RADIUS_MODE": "2", "PF_RADIUS_REVERSION_TIME": "1"},
    "pull2": {"PF_RADIUS_MODE": "2", "PF_RADIUS_REVERSION_TIME": "2"},
    "pull4": {"PF_RADIUS_MODE": "2", "PF_RADIUS_REVERSION_TIME": "4"},
    "stillpull2": {"PF_RADIUS_MODE": "3", "PF_RADIUS_REVERSION_TIME": "2"},
    "obsz": {"PF_OBS_Z_VARIANCE": "0.0003"},
    "dzq": {"PF_Z_OFFSET_PROCESS_VARIANCE": "0.0001"},
    "omegaq1": {"PF_ORIENTATION_VELOCITY_PROCESS_VARIANCE": "1"},
    "viscoef3": {"PF_VISIBILITY_LOGIT_COEFFICIENT": "3"},
    "velq3": {"PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "3"},
    "velq10": {"PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "10"},
    "velq1": {"PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "1"},
    "velq03": {"PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "0.3"},
    "obs1": {"PF_OBS_MODEL": "1"},
    "obs1loose": {"PF_OBS_MODEL": "1", "PF_OBS_LATERAL_VARIANCE": "0.0004", "PF_OBS_RANGE_VARIANCE": "0.002",
                  "PF_OBS_RANGE_ANGLE_VARIANCE": "0.01"},
    "lock": {"PF_LOCK_TIME": "2", "PF_LOCK_FLOOR": "0.1"},
    "lockhard": {"PF_LOCK_TIME": "1", "PF_LOCK_FLOOR": "0.02"},
    "mem": {"PF_MEMORY": "1"},
    "memz": {"PF_MEMORY": "1", "PF_MEMORY_CENTER_Z": "1"},
    "mode": {"PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "0.1", "PF_MANEUVER_VELOCITY_VARIANCE": "3", "PF_MANEUVER_SWITCH_RATE": "2"},
    "mode03": {"PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "0.3", "PF_MANEUVER_VELOCITY_VARIANCE": "3", "PF_MANEUVER_SWITCH_RATE": "2"},
    "spinradius": {"PF_RADIUS_MODE": "4", "PF_SETTLE_BY_ANGLE": "1", "PF_RADIUS_REVERSION_TIME": "2"},
    "lockangle": {"PF_LOCK_TIME": "12.6", "PF_LOCK_FLOOR": "0.1", "PF_SETTLE_BY_ANGLE": "1", "PF_MEMORY_SETTLED": "12.6"},
    "lockangle6": {"PF_LOCK_TIME": "6.3", "PF_LOCK_FLOOR": "0.1", "PF_SETTLE_BY_ANGLE": "1", "PF_MEMORY_SETTLED": "12.6"},
    "rq5": {"PF_RADIUS_PROCESS_VARIANCE": "0.001"},
    "rq2": {"PF_RADIUS_PROCESS_VARIANCE": "0.0005"},
    "final2": {"PF_VISIBILITY_XY": "1", "PF_SWAP_REDUCTION": "1", "PF_RELABEL_AT_ZERO": "1", "PF_ASSOC_BEST": "1",
               "PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "0.1", "PF_MANEUVER_VELOCITY_VARIANCE": "3",
               "PF_MANEUVER_SWITCH_RATE": "2", "PF_RADIUS_MODE": "4", "PF_SETTLE_BY_ANGLE": "1",
               "PF_RADIUS_REVERSION_TIME": "2", "PF_RADIUS_PROCESS_VARIANCE": "0.001", "PF_LOCK_TIME": "12.6",
               "PF_LOCK_FLOOR": "0.1", "PF_MEMORY": "1", "PF_MEMORY_CENTER_Z": "1", "PF_MEMORY_SETTLED": "12.6"},
    "mode5": {"PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "0.1", "PF_MANEUVER_VELOCITY_VARIANCE": "5", "PF_MANEUVER_SWITCH_RATE": "1"},
    "assoc": {"PF_ASSOC_BEST": "1"},
    "visflat": {"PF_VISIBILITY_LOGIT_COEFFICIENT": "10", "PF_VISIBILITY_THRESHOLD": "0.3"},
    "visflat20": {"PF_VISIBILITY_LOGIT_COEFFICIENT": "20", "PF_VISIBILITY_THRESHOLD": "0.25"},
    "visweak": {"PF_VISIBILITY_LOGIT_COEFFICIENT": "0.2"},
}

DZ_SETTLED_M = 0.005
FINAL_WINDOW_S = 3.0


def variant_env(name: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for part in name.split("+"):
        if part not in VARIANTS:
            sys.exit(f"unknown variant {part!r}; choose from {', '.join(VARIANTS)}")
        env.update(VARIANTS[part])
    return env


def _tag(extra: list[str]) -> str:
    return "".join(a.lstrip("-").replace("-", "").replace(".", "p") for a in extra) or "default"


def cmd_run(args) -> None:
    batch = BATCHES / args.batch
    extra = args.extra.split() if args.extra else []
    for variant in args.variants.split(","):
        env = variant_env(variant)
        for spin0 in args.spin0.split(","):
            out = batch / f"{args.scenario}__{_tag(extra)}__{variant}__s{spin0}"
            if (out / "summary.json").is_file():
                print(f"[pf_ab] have {out.name}", flush=True)
                continue
            cmd = [str(NSSIM), "run", args.scenario, "--out", str(out), "--spin0", spin0, *extra]
            for name, value in env.items():
                cmd += ["--cv-env", f"{name}={value}"]
            print(f"[pf_ab] {out.name}", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not (out / "summary.json").is_file():
                print(result.stdout[-2000:], result.stderr[-2000:], sep="\n")
                sys.exit(f"[pf_ab] {out.name} failed")


def series(run_dir: Path) -> dict[str, np.ndarray]:
    """Per-frame filter errors of a run, over sim time."""
    frames = [json.loads(line) for line in (run_dir / "frames.jsonl").read_text().splitlines() if line.strip()]
    tracks = {}
    for line in (run_dir / "telemetry.jsonl").read_text().splitlines():
        if '"track"' in line:
            m = json.loads(line)
            if m.get("type") == "track":
                tracks[m["seq"]] = m
    keys = ["z_offset", "true_z_offset", "z_offset_m", "radius_high", "radius_low", "true_radius_high",
            "true_radius_low", "radius_high_m", "radius_low_m", "radius_mean_m", "orientation_rad", "labeling_ok",
            "plates_m", "facing_plates_m", "aim_plate_m", "center_xy_m", "omega", "true_omega"]
    rows = {k: [] for k in ["t", *keys]}
    for frame in frames:
        f = frame_errors(frame, None, tracks.get(frame["seq"]))["filter"]
        if f is None:
            continue
        rows["t"].append(frame["gt"]["t"])
        for k in keys:
            rows[k].append(f[k])
    return {k: np.asarray(v, float) for k, v in rows.items()}


def _settle_time(t: np.ndarray, ok: np.ndarray) -> float:
    """First time from which `ok` holds to the end (inf if it doesn't hold at the end)."""
    if len(ok) == 0 or not ok[-1]:
        return math.inf
    bad = np.nonzero(~ok)[0]
    return float(t[0] if len(bad) == 0 else t[bad[-1] + 1])


def summarize(run_dir: Path) -> dict:
    s = series(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").is_file() else {}
    t = s["t"]
    if len(t) == 0:
        return {"run": run_dir.name, "frames": 0}
    final = t >= t[-1] - FINAL_WINDOW_S
    out = {
        "run": run_dir.name,
        "frames": int(len(t)),
        "final_z_err_mm": float(np.mean(s["z_offset_m"][final]) * 1000),
        "final_r_high_err_cm": float(np.mean(s["radius_high_m"][final]) * 100),
        "final_r_low_err_cm": float(np.mean(s["radius_low_m"][final]) * 100),
        "final_r_mean_err_cm": float(np.mean(s["radius_mean_m"][final]) * 100),
        "final_labeling_ok": float(np.mean(s["labeling_ok"][final])),
        "final_plates_cm": float(np.mean(s["plates_m"][final]) * 100),
        "final_orient_rms_deg": float(np.degrees(np.sqrt(np.mean(s["orientation_rad"][final] ** 2)))),
        "labeling_ok": float(np.mean(s["labeling_ok"])),
        "plates_cm": float(np.mean(s["plates_m"]) * 100),
        "plates_p95_cm": float(np.percentile(s["plates_m"], 95) * 100),
        "aim_cm": float(np.nanmean(s["aim_plate_m"]) * 100),
        "aim_p95_cm": float(np.nanpercentile(s["aim_plate_m"], 95) * 100),
        "final_z_std_mm": float(np.std(s["z_offset"][final]) * 1000),
        "final_center_cm": float(np.mean(s["center_xy_m"][final]) * 100),
        "dz_settle_s": _settle_time(t, np.abs(s["z_offset_m"]) < DZ_SETTLED_M),
        "labeling_settle_s": _settle_time(t, s["labeling_ok"] > 0.5),
        "labeling_flips": int(np.sum(np.abs(np.diff(s["labeling_ok"])) > 0.5)),
    }
    targets = cfg.get("targets") or []
    spec = (targets[0].get("spec") or {}) if targets else {}
    flip = spec.get("z_offset_flip_s")
    if flip is not None and math.isfinite(flip):
        after = t >= flip
        out["flip_recover_s"] = _settle_time(t[after], s["labeling_ok"][after] > 0.5) - flip
    motion = (targets[0].get("motion") or {}) if targets else {}
    start = motion.get("spin_start_s") or 0.0
    if start:
        before = t < start
        if before.any():
            i = np.nonzero(before)[0][-1]
            out["at_spin_start_r_mean_err_cm"] = float(s["radius_mean_m"][i] * 100)
            out["at_spin_start_z_err_mm"] = float(s["z_offset_m"][i] * 1000)
            out["still_labeling_ok"] = float(np.mean(s["labeling_ok"][before]))
            out["still_plates_cm"] = float(np.mean(s["plates_m"][before]) * 100)
            out["still_aim_cm"] = float(np.nanmean(s["aim_plate_m"][before]) * 100)
            settled = t >= start + 2.0
            out["after_spin_2s_plates_cm"] = float(np.mean(s["plates_m"][settled]) * 100) if settled.any() else math.nan
            # First time from which the spin estimate stays within 1 rad/s for a whole second.
            ts, err = t[t >= start], np.abs(s["omega"] - s["true_omega"])[t >= start]
            found = math.inf
            for i in range(len(ts)):
                window = (ts >= ts[i]) & (ts < ts[i] + 1.0)
                if ts[-1] >= ts[i] + 1.0 and np.all(err[window] < 1.0):
                    found = float(ts[i] - start)
                    break
            out["spin_found_s"] = found
    return out


def plot(runs: list[Path], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    for i, run_dir in enumerate(runs):
        s = series(run_dir)
        if len(s["t"]) == 0:
            continue
        label = run_dir.name.split("__")[2]
        color = f"C{i % 10}"
        axes[0].plot(s["t"], s["z_offset"] * 100, color=color, label=label)
        axes[0].plot(s["t"], s["true_z_offset"] * 100, color="k", lw=0.8, ls="--")
        axes[1].plot(s["t"], s["radius_high"] * 100, color=color)
        axes[1].plot(s["t"], s["radius_low"] * 100, color=color, ls=":")
        axes[1].plot(s["t"], s["true_radius_high"] * 100, color="k", lw=0.8, ls="--")
        axes[1].plot(s["t"], s["true_radius_low"] * 100, color="k", lw=0.8, ls=":")
        axes[2].plot(s["t"], np.degrees(s["orientation_rad"]), color=color)
        axes[3].plot(s["t"], s["plates_m"] * 100, color=color)
        axes[4].plot(s["t"], s["omega"], color=color)
        axes[4].plot(s["t"], s["true_omega"], color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("height offset (cm)")
    axes[1].set_ylabel("radius (cm)\nsolid high, dotted low")
    axes[2].set_ylabel("orientation err (deg)\n±90 = pairs swapped")
    axes[2].set_ylim(-95, 95)
    axes[3].set_ylabel("true plate to nearest\nfilter plate (cm)")
    axes[4].set_ylabel("spin (rad/s)")
    axes[4].set_xlabel("sim time (s); dashed black = truth")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


COLUMNS = [
    ("labeling_ok", "high pair right", lambda v: f"{v * 100:.0f}%"),
    ("final_labeling_ok", "… last 3 s", lambda v: f"{v * 100:.0f}%"),
    ("labeling_flips", "flips", lambda v: f"{v:.0f}"),
    ("plates_cm", "plate err cm", lambda v: f"{v:.1f}"),
    ("final_plates_cm", "… last 3 s", lambda v: f"{v:.1f}"),
    ("aim_cm", "aim plate err cm", lambda v: f"{v:.1f}"),
    ("aim_p95_cm", "… p95", lambda v: f"{v:.1f}"),
    ("final_center_cm", "center err cm (last 3 s)", lambda v: f"{v:.1f}"),
    ("final_z_err_mm", "dz err mm", lambda v: f"{v:+.1f}"),
    ("final_z_std_mm", "dz jitter mm (last 3 s)", lambda v: f"{v:.1f}"),
    ("dz_settle_s", "dz settle s", lambda v: "never" if not math.isfinite(v) else f"{v:.1f}"),
    ("final_r_high_err_cm", "r high err cm", lambda v: f"{v:+.1f}"),
    ("final_r_low_err_cm", "r low err cm", lambda v: f"{v:+.1f}"),
    ("final_r_mean_err_cm", "r mean err cm", lambda v: f"{v:+.1f}"),
    ("flip_recover_s", "flip recover s", lambda v: "never" if not math.isfinite(v) else f"{v:.1f}"),
    ("still_labeling_ok", "still: pair right", lambda v: f"{v * 100:.0f}%"),
    ("still_plates_cm", "still: plate err cm", lambda v: f"{v:.1f}"),
    ("at_spin_start_r_mean_err_cm", "r err at spin-up cm", lambda v: f"{v:+.1f}"),
    ("at_spin_start_z_err_mm", "dz err at spin-up mm", lambda v: f"{v:+.1f}"),
    ("spin_found_s", "spin found after s", lambda v: "never" if not math.isfinite(v) else f"{v:.1f}"),
    ("after_spin_2s_plates_cm", "plate err from 2 s after spin-up cm", lambda v: f"{v:.1f}"),
]


def cmd_report(args) -> None:
    batch = BATCHES / args.batch
    runs = sorted(p for p in batch.iterdir() if (p / "summary.json").is_file())
    groups: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for run_dir in runs:
        scenario, tag, variant, _ = run_dir.name.split("__")
        summary = summarize(run_dir)
        (run_dir / "pf_summary.json").write_text(json.dumps(summary, indent=2))
        groups.setdefault((scenario, tag), {}).setdefault(variant, []).append(summary)

    lines = [f"# Particle-filter A/B: {args.batch}", "",
             "Each row averages the runs of one variant over the starting angles (--spin0).", ""]
    for (scenario, tag), variants in groups.items():
        present = [c for c in COLUMNS if any(c[0] in s for runs_ in variants.values() for s in runs_)]
        lines += [f"## {scenario} ({tag})", "",
                  "| variant | runs | " + " | ".join(c[1] for c in present) + " |",
                  "|---|---|" + "---|" * len(present)]
        for variant, summaries in variants.items():
            cells = []
            for key, _, fmt in present:
                values = [s[key] for s in summaries if key in s]
                if not values:
                    cells.append("")
                elif any(not math.isfinite(v) for v in values):
                    finite = [v for v in values if math.isfinite(v)]
                    cells.append(f"{sum(not math.isfinite(v) for v in values)}/{len(values)} never"
                                 + (f", else {fmt(np.mean(finite))}" if finite else ""))
                else:
                    cells.append(fmt(float(np.mean(values))))
            lines.append(f"| {variant} | {len(summaries)} | " + " | ".join(cells) + " |")
        lines.append("")
        # One plot per scenario: the first starting angle of each variant.
        firsts = [batch / summaries[0]["run"] for summaries in variants.values()]
        png = batch / f"{scenario}__{tag}.png"
        plot(firsts, png, f"{scenario} ({tag})")
        lines += [f"![{scenario}]({png.name})", ""]
    report = batch / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[pf_ab] wrote {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("batch")
    run.add_argument("--scenario", required=True)
    run.add_argument("--variants", default="base")
    run.add_argument("--spin0", default="0", help="comma-separated starting-angle offsets, degrees")
    run.add_argument("--extra", help="more `nssim run` options, e.g. \"--plate-height 0.01\"")
    run.set_defaults(func=cmd_run)
    rep = sub.add_parser("report")
    rep.add_argument("batch")
    rep.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
