"""Command line: build-cv, run, eval, preview and frame (see ``nssim <command> -h``)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from nssim import paths
from nssim.paths import REPO_ROOT


def _cv_dir(arg: str | None) -> Path:
    path = paths.cv_dir(arg)
    if not (path / "src").is_dir():
        sys.exit(paths.missing_hint(path, "Northstar-CV", "NORTHSTAR_CV_DIR"))
    return path


def _config(args):
    from nssim.scenarios import SCENARIOS

    if args.scenario not in SCENARIOS:
        sys.exit(f"unknown scenario {args.scenario!r}; choose from {', '.join(SCENARIOS)}")
    cfg = SCENARIOS[args.scenario]()
    if getattr(args, "duration", None):
        cfg.duration_s = args.duration
    if getattr(args, "turret", None):
        cfg.turret.mode = args.turret
    if getattr(args, "fps", None):
        cfg.fps = args.fps
    if getattr(args, "no_compress", False):
        cfg.compress_frames = False
    return cfg


def _runner(cfg, cv_dir: Path, out_dir: Path):
    from nssim.runner import Runner
    from nssim.sim.assets import TrAssets
    from nssim.sim.robot_constants import load_plate_dims, load_robot_constants

    assets = TrAssets(plate_dims=load_plate_dims(cv_dir))
    return Runner(cfg, load_robot_constants(cv_dir), out_dir, assets)


def cmd_build_cv(args) -> None:
    from nssim.cv_build import build

    binary = build(_cv_dir(args.cv_dir), cuda_arch=args.cuda_arch, rebuild_images=args.rebuild_images)
    print(f"[nssim] built {binary}")


def cmd_run(args) -> None:
    _execute(_config(args), args, live="view" if args.view else None)


def cmd_drive(args) -> None:
    from nssim.scenarios import drive

    cfg = drive(distance=args.distance, **({"fps": args.fps} if args.fps else {}))
    if args.turret:
        cfg.turret.mode = args.turret
    if args.no_compress:
        cfg.compress_frames = False
    _execute(cfg, args, live="drive")


def _execute(cfg, args, live: str | None) -> None:
    """Start Northstar-CV, run the scenario (optionally in the live viewer), evaluate."""
    from nssim.cv_build import binary_path, image_exists
    from nssim.cv_launcher import CvContainer
    from nssim.eval import evaluate, format_summary

    cv_dir = _cv_dir(args.cv_dir)
    if not args.no_launch:
        if not binary_path(cv_dir, args.build_dir).is_file():
            sys.exit(f"no simulator build of Northstar-CV in {cv_dir / args.build_dir}; run `nssim build-cv` first")
        if not image_exists(args.image):
            sys.exit(f"Docker image {args.image} not found; run `nssim build-cv` first")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / f"{stamp}_{cfg.name}"
    runner = _runner(cfg, cv_dir, out_dir)
    session = None
    if live:
        from nssim.live import LiveSession

        session = LiveSession(runner, drive=live == "drive", show_cv=not args.no_cv_window)

    container = None
    if not args.no_launch:
        container = CvContainer(cv_dir=cv_dir, image=args.image, build_dir=args.build_dir)
        print("[nssim] starting Northstar-CV:", " ".join(container.command()), flush=True)
        container.start(out_dir / "cv.log")
    try:
        runner.run(session)
    except KeyboardInterrupt:
        print("[nssim] interrupted", flush=True)
    finally:
        if session is not None:
            session.close()
        if container is not None:
            container.stop()
    summary = evaluate(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(format_summary(summary))
    print(f"[nssim] run saved to {out_dir} (full metrics in summary.json)")


def cmd_eval(args) -> None:
    from nssim.eval import main

    main(args.run_dir)


def cmd_preview(args) -> None:
    import cv2
    import numpy as np

    from nssim.camera import apply_optics, demosaic_to_bgr, mosaic_rggb

    cfg = _config(args)
    cfg.frame_port = cfg.uart_port = cfg.telemetry_port = 0  # any free port; nothing connects
    runner = _runner(cfg, _cv_dir(args.cv_dir), REPO_ROOT / ".scratch" / "preview_run")
    gt, cam_pose, states = runner._ground_truth(args.time)
    # What Northstar-CV gets: lens effects, the sensor's Bayer mosaic, then its debayer.
    bgr = demosaic_to_bgr(mosaic_rggb(apply_optics(runner.scene.render(cam_pose, states), cfg.optics)))
    for target in gt["targets"]:
        for plate in target["plates"]:
            for u, v in np.asarray(plate["corners_px"]):
                if np.isfinite(u):
                    cv2.circle(bgr, (int(round(u)), int(round(v))), 3, (0, 255, 0) if plate["facing"] else (0, 0, 255), 1)
    out = Path(args.out or REPO_ROOT / ".scratch" / f"preview_{cfg.name}.png")
    cv2.imwrite(str(out), bgr)
    print(f"[nssim] wrote {out}")


def cmd_frame(args) -> None:
    from nssim.debug import render_logged_frame

    out = Path(args.out or REPO_ROOT / ".scratch" / f"frame_{Path(args.run_dir).name}_{args.seq}.png")
    render_logged_frame(Path(args.run_dir), args.seq, _cv_dir(args.cv_dir), out)
    print(f"[nssim] wrote {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="nssim", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    bc = sub.add_parser("build-cv", help="build the Docker images and Northstar-CV's simulator binary")
    bc.add_argument("--cv-dir", help="Northstar-CV checkout (default: the external/Northstar-CV submodule)")
    bc.add_argument("--cuda-arch", help="CUDA compute capability to build for, e.g. 89 (default: this machine's GPU)")
    bc.add_argument("--rebuild-images", action="store_true", help="rebuild the Docker images even if they exist")
    bc.set_defaults(func=cmd_build_cv)

    run = sub.add_parser("run", help="run a scenario with Northstar-CV in the loop")
    run.add_argument("scenario")
    run.add_argument("--duration", type=float)
    run.add_argument("--fps", type=float)
    run.add_argument("--turret", choices=["hold", "ideal", "second_order"])
    run.add_argument("--no-compress", action="store_true", help="send raw Bayer frames (fast links)")
    run.add_argument("--cv-dir", help="Northstar-CV checkout to test (default: the external/Northstar-CV submodule)")
    run.add_argument("--out")
    run.add_argument("--no-launch", action="store_true", help="don't start the container (Northstar-CV started elsewhere)")
    run.add_argument("--image", default="northstar-cv:sim")
    run.add_argument("--build-dir", default="build/sim-x86")
    run.add_argument("--view", action="store_true", help="watch it live in a 3D viewer (paced to real time)")
    run.add_argument("--no-cv-window", action="store_true", help="with --view: skip the camera window")
    run.set_defaults(func=cmd_run)

    dr = sub.add_parser("drive", help="drive the enemy (and our robot) live while Northstar-CV tracks it")
    dr.add_argument("--distance", type=float, default=3.0, help="starting distance to the enemy (m)")
    dr.add_argument("--fps", type=float, help="camera frame rate (default 50; 166 runs in slow motion)")
    dr.add_argument("--turret", choices=["hold", "ideal", "second_order"])
    dr.add_argument("--no-compress", action="store_true", help="send raw Bayer frames (fast links)")
    dr.add_argument("--no-cv-window", action="store_true", help="skip the window with the CV's camera image")
    dr.add_argument("--cv-dir", help="Northstar-CV checkout to test (default: the external/Northstar-CV submodule)")
    dr.add_argument("--out")
    dr.add_argument("--no-launch", action="store_true", help="don't start the container (Northstar-CV started elsewhere)")
    dr.add_argument("--image", default="northstar-cv:sim")
    dr.add_argument("--build-dir", default="build/sim-x86")
    dr.set_defaults(func=cmd_drive)

    ev = sub.add_parser("eval", help="evaluate a run directory")
    ev.add_argument("run_dir")
    ev.set_defaults(func=cmd_eval)

    pv = sub.add_parser("preview", help="render one frame of a scenario to a PNG")
    pv.add_argument("scenario")
    pv.add_argument("--time", type=float, default=0.0)
    pv.add_argument("--cv-dir")
    pv.add_argument("--out")
    pv.set_defaults(func=cmd_preview)

    fr = sub.add_parser("frame", help="re-render a logged frame with ground truth and detections")
    fr.add_argument("run_dir")
    fr.add_argument("seq", type=int)
    fr.add_argument("--cv-dir")
    fr.add_argument("--out")
    fr.set_defaults(func=cmd_frame)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
