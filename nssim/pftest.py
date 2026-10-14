"""Tracker test (`nssim pftest`): every robot type, with every plate layout, drives a tour while
changing how it spins (scenarios.pftest). Writes graphs and a report per test, and compares tests
(`nssim pftest-report A B`, e.g. Northstar-CV as it is against a --preset).

Errors are against the robot under test only: if the tracker is on another robot, those frames
count as wrong. Distances are from our camera to the robot's center.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from nssim.metrics import frame_errors

DISTANCE_BINS = [(1.5, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 7.0)]
SPIN_BINS = [(0.0, 0.5), (0.5, 2.0), (2.0, 4.0), (4.0, 99.0)]
SETTLE_S = 5.0  # the summary's geometry errors leave out the first seconds
SPIKE_M = 0.10  # an aimed-plate error this large counts as a spike


def _jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def series(run: Path) -> dict[str, np.ndarray]:
    """Per frame: what the robot under test did, what the tracker made of it, what was detected."""
    cfg = json.loads((run / "config.json").read_text())
    name = cfg["targets"][0]["spec"]["name"]
    dets, tracks = {}, {}
    for m in _jsonl(run / "telemetry.jsonl"):
        if m.get("type") == "det":
            dets[m["seq"]] = m
        elif m.get("type") == "track":
            tracks[m["seq"]] = m
    rows = []
    for frame in _jsonl(run / "frames.jsonl"):
        gt = frame["gt"]
        target = next(t for t in gt["targets"] if t["name"] == name)
        alone = {**frame, "gt": {**gt, "targets": [target]}}
        track = tracks.get(frame["seq"])
        err = frame_errors(alone, dets.get(frame["seq"]), track)
        cam = np.asarray(gt["camera_world"]["p"]) - np.asarray(gt["base_world"])
        center = np.asarray(target["center_base"])
        f = err["filter"]
        on_target = False
        if f is not None:
            est = np.asarray(track["state"]["center"])
            nearest = min(gt["targets"], key=lambda t: np.linalg.norm(np.asarray(t["center_base"])[:2] - est[:2]))
            on_target = nearest["name"] == name
        nan = math.nan
        rows.append({
            "t": gt["t"],
            "distance": float(np.linalg.norm((center - cam)[:2])),
            "true_omega": target["omega"],
            "omega": f["omega"] if f else nan,
            "tracking": float(on_target),
            "center_m": f["center_xy_m"] if on_target else nan,
            "aim_m": f["aim_plate_m"] if on_target else nan,
            "radius_high": f["radius_high"] if on_target else nan,
            "radius_low": f["radius_low"] if on_target else nan,
            "true_radius_high": f["true_radius_high"] if f else nan,
            "true_radius_low": f["true_radius_low"] if f else nan,
            "z_offset": f["z_offset"] if on_target else nan,
            "true_z_offset": abs(target.get("z_offset", 0.0)),
            "visible": err["visible"],
            "detected": err["detected"],
            "range_m": float(np.mean(err["range_m"])) if err["range_m"] else nan,
        })
    return {k: np.asarray([r[k] for r in rows], float) for k in rows[0]}


def _mean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else math.nan


def _p95(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95)) if x.size else math.nan


def summarize(run: Path) -> dict:
    s = series(run)
    cfg = json.loads((run / "config.json").read_text())
    spec = cfg["targets"][0]["spec"]
    settled = s["t"] >= SETTLE_S
    r_err = (np.abs(s["radius_high"] - s["true_radius_high"]) + np.abs(s["radius_low"] - s["true_radius_low"])) / 2
    if spec["z_offset"] == 0:
        # No pair is higher, so which one the tracker calls "high" is arbitrary: take the better match.
        swapped = (np.abs(s["radius_high"] - s["true_radius_low"]) + np.abs(s["radius_low"] - s["true_radius_high"])) / 2
        r_err = np.fmin(r_err, swapped)
    out = {
        "run": run.name,
        "robot": spec["name"],
        "layout": f"r {spec['radius_high']:.2f}/{spec['radius_low']:.2f} m, dz {spec['z_offset'] * 100:.0f} cm",
        "tracking": _mean(s["tracking"]),
        "recall": float(s["detected"].sum() / max(1.0, s["visible"].sum())),
        "aim_cm": _mean(s["aim_m"]) * 100,
        "aim_p95_cm": _p95(s["aim_m"]) * 100,
        "center_cm": _mean(s["center_m"]) * 100,
        "radius_err_cm": _mean(r_err[settled]) * 100,
        "dz_err_mm": _mean(np.abs(s["z_offset"] - s["true_z_offset"])[settled]) * 1000,
        "spin_err": _mean(np.abs(s["omega"] - s["true_omega"])[s["tracking"] > 0]),
    }
    # Spikes: frames where the aimed plate is more than SPIKE_M off, and how many separate times.
    spiking = np.nan_to_num(s["aim_m"]) > SPIKE_M
    out["spike_frames"] = float(spiking.mean())
    out["spikes"] = int(np.sum(spiking[1:] & ~spiking[:-1]) + spiking[0])
    for lo, hi in DISTANCE_BINS:
        w = (s["distance"] >= lo) & (s["distance"] < hi)
        key = f"{lo:g}-{hi:g}m"
        out[f"recall_{key}"] = float(s["detected"][w].sum() / max(1.0, s["visible"][w].sum())) if w.any() else math.nan
        out[f"aim_cm_{key}"] = _mean(s["aim_m"][w]) * 100
        out[f"center_cm_{key}"] = _mean(s["center_m"][w]) * 100
        out[f"range_cm_{key}"] = _mean(s["range_m"][w]) * 100
        out[f"range_sd_cm_{key}"] = float(np.nanstd(s["range_m"][w]) * 100) if np.isfinite(s["range_m"][w]).any() else math.nan
    for lo, hi in SPIN_BINS:
        w = (np.abs(s["true_omega"]) >= lo) & (np.abs(s["true_omega"]) < hi)
        key = f"{lo:g}-{hi:g}" if hi < 99 else f"{lo:g}+"
        out[f"aim_cm_spin_{key}"] = _mean(s["aim_m"][w]) * 100
        out[f"center_cm_spin_{key}"] = _mean(s["center_m"][w]) * 100
    return out


def plot_run(run: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = series(run)
    t = s["t"]
    fig, ax = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
    ax[0].plot(t, s["distance"], "k")
    ax[0].set_ylabel("distance (m)")
    ax[1].plot(t, s["true_omega"], "k--", lw=1, label="true")
    ax[1].plot(t, s["omega"], "C0", lw=0.8, label="tracker")
    ax[1].set_ylabel("spin (rad/s)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, s["center_m"] * 100, "C1", lw=0.8)
    ax[2].set_ylabel("center error (cm)")
    ax[3].plot(t, s["aim_m"] * 100, "C3", lw=0.8)
    ax[3].set_ylabel("aimed-plate error (cm)")
    ax[4].plot(t, s["true_radius_high"] * 100, "k--", lw=1)
    ax[4].plot(t, s["true_radius_low"] * 100, "k:", lw=1)
    ax[4].plot(t, s["radius_high"] * 100, "C0", lw=0.8, label="high pair")
    ax[4].plot(t, s["radius_low"] * 100, "C2", lw=0.8, label="low pair")
    ax[4].set_ylabel("radius (cm)")
    ax[4].legend(loc="upper right", fontsize=8)
    ax[5].plot(t, s["true_z_offset"] * 100, "k--", lw=1)
    ax[5].plot(t, s["z_offset"] * 100, "C4", lw=0.8)
    ax[5].set_ylabel("height offset (cm)")
    k = max(1, int(round(0.25 / max(1e-3, np.median(np.diff(t))))))  # 0.25 s running mean
    smooth = lambda x: np.convolve(x, np.ones(k) / k, mode="same")  # noqa: E731
    ax[6].plot(t, smooth(s["visible"]), "k", lw=1, label="should be seen")
    ax[6].plot(t, smooth(s["detected"]), "C0", lw=1, label="detected")
    ax[6].plot(t, smooth(s["tracking"]), "C3", lw=1, label="tracker on this robot")
    ax[6].set_ylabel("plates / tracking")
    ax[6].legend(loc="upper right", fontsize=8)
    ax[6].set_xlabel("time (s); dashed black = truth")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def _runs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "frames.jsonl").is_file() and (p / "summary.json").is_file())


def _fmt(v, spec="{:.1f}") -> str:
    return "" if v is None or not math.isfinite(v) else spec.format(v)


def report(roots: list[Path], out_dir: Path | None = None) -> Path:
    """Graphs and report.md for one test, or a comparison of several (their names as columns)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or roots[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = {}
    for root in roots:
        summaries = []
        for run in _runs(root):
            cache = run / "pftest_summary.json"
            if cache.is_file() and cache.stat().st_mtime >= (run / "summary.json").stat().st_mtime:
                summary = json.loads(cache.read_text())
            else:
                summary = summarize(run)
                cache.write_text(json.dumps(summary, indent=2))
                plot_run(run, run / "pftest.png")
            summaries.append(summary)
        manifest = root / "pftest.json"
        label = json.loads(manifest.read_text()).get("label", root.name) if manifest.is_file() else root.name
        tests[label] = summaries

    lines = ["# Tracker test", ""]
    for label, root in zip(tests, roots):
        manifest = root / "pftest.json"
        info = json.loads(manifest.read_text()) if manifest.is_file() else {}
        env = info.get("cv_env") or {}
        lines.append(f"- **{label}** ({root}): " + (", ".join(f"{k}={v}" for k, v in env.items()) or "Northstar-CV as built"))
    lines.append("")

    cols = [("tracking", "on robot", "{:.0%}"), ("recall", "recall", "{:.0%}"), ("aim_cm", "aim err cm", "{:.1f}"),
            ("aim_p95_cm", "aim p95 cm", "{:.1f}"), ("center_cm", "center err cm", "{:.1f}"),
            ("radius_err_cm", "radius err cm", "{:.1f}"), ("dz_err_mm", "dz err mm", "{:.1f}"),
            ("spin_err", "spin err rad/s", "{:.2f}"), ("spikes", "spikes (aim >10 cm)", "{:.0f}"),
            ("spike_frames", "spike frames", "{:.1%}")]
    lines += ["## Each run", "", "| robot | layout | test | " + " | ".join(c[1] for c in cols) + " |",
              "|---|---|---|" + "---|" * len(cols)]
    by_run: dict[str, dict[str, dict]] = {}
    for label, summaries in tests.items():
        for s in summaries:
            by_run.setdefault(s["run"].split("__")[0], {})[label] = s
    for run_name, per_test in sorted(by_run.items()):
        for label, s in per_test.items():
            lines.append(f"| {s['robot']} | {s['layout']} | {label} | " +
                         " | ".join(_fmt(s.get(k), f) for k, _, f in cols) + " |")
    lines += ["", "| test | spikes (all runs) | spike frames | aim err cm | center err cm |", "|---|---|---|---|---|"]
    for label, summaries in tests.items():
        lines.append(f"| {label} | {sum(s.get('spikes', 0) for s in summaries)} | "
                     f"{_fmt(_mean([s.get('spike_frames') for s in summaries]), '{:.1%}')} | "
                     f"{_fmt(_mean([s['aim_cm'] for s in summaries]))} | {_fmt(_mean([s['center_cm'] for s in summaries]))} |")
    lines.append("")

    def mean_over(label, key):
        return _mean([s.get(key, math.nan) for s in tests[label]])

    lines += ["## By distance (all runs)", "", "| test | distance | recall | PnP range err cm | PnP range sd cm | aim err cm | center err cm |",
              "|---|---|---|---|---|---|---|"]
    for label in tests:
        for lo, hi in DISTANCE_BINS:
            key = f"{lo:g}-{hi:g}m"
            lines.append(f"| {label} | {key} | {_fmt(mean_over(label, f'recall_{key}'), '{:.0%}')} | "
                         f"{_fmt(mean_over(label, f'range_cm_{key}'), '{:+.1f}')} | {_fmt(mean_over(label, f'range_sd_cm_{key}'))} | "
                         f"{_fmt(mean_over(label, f'aim_cm_{key}'))} | {_fmt(mean_over(label, f'center_cm_{key}'))} |")
    lines += ["", "## By spin rate (all runs)", "", "| test | spin rad/s | aim err cm | center err cm |", "|---|---|---|---|"]
    for label in tests:
        for lo, hi in SPIN_BINS:
            key = f"{lo:g}-{hi:g}" if hi < 99 else f"{lo:g}+"
            lines.append(f"| {label} | {key} | {_fmt(mean_over(label, f'aim_cm_spin_{key}'))} | "
                         f"{_fmt(mean_over(label, f'center_cm_spin_{key}'))} |")
    lines.append("")

    # Summary graphs: errors and recall against distance, and aim error per run.
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    centers = [(lo + min(hi, 6.5)) / 2 for lo, hi in DISTANCE_BINS]
    for i, label in enumerate(tests):
        keys = [f"{lo:g}-{hi:g}m" for lo, hi in DISTANCE_BINS]
        ax[0, 0].plot(centers, [mean_over(label, f"recall_{k}") * 100 for k in keys], "o-", color=f"C{i}", label=label)
        ax[0, 1].plot(centers, [mean_over(label, f"range_cm_{k}") for k in keys], "o-", color=f"C{i}", label=label)
        ax[1, 0].plot(centers, [mean_over(label, f"aim_cm_{k}") for k in keys], "o-", color=f"C{i}", label=f"{label} aim")
        ax[1, 0].plot(centers, [mean_over(label, f"center_cm_{k}") for k in keys], "s--", color=f"C{i}", label=f"{label} center")
    ax[0, 0].set_ylabel("detection recall (%)")
    ax[0, 1].set_ylabel("PnP range error (cm, + = too far)")
    ax[1, 0].set_ylabel("error (cm)")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("distance (m)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    names = sorted(by_run)
    width = 0.8 / max(1, len(tests))
    for i, label in enumerate(tests):
        ax[1, 1].bar(np.arange(len(names)) + i * width, [by_run[n].get(label, {}).get("aim_cm", np.nan) for n in names],
                     width, color=f"C{i}", label=label)
    ax[1, 1].set_xticks(np.arange(len(names)) + width * (len(tests) - 1) / 2)
    ax[1, 1].set_xticklabels([n.replace("pftest_", "") for n in names], rotation=60, ha="right", fontsize=7)
    ax[1, 1].set_ylabel("aimed-plate error (cm)")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pftest_summary.png", dpi=80)
    plt.close(fig)
    lines += ["![summary](pftest_summary.png)", "", "Each run's graphs: `pftest.png` in its folder.", ""]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
