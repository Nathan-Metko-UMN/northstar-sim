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
    return _apply_common(cfg, args)


def _apply_common(cfg, args):
    """run / drive options that change the scenario itself."""
    if getattr(args, "no_compress", False):
        cfg.compress_frames = False
    if getattr(args, "plate_height", None) is not None:
        for target in cfg.targets:
            target.spec.z_offset = args.plate_height
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

    cfg = drive(distance=args.distance, enemies=args.enemies, **({"fps": args.fps} if args.fps else {}))
    if args.turret:
        cfg.turret.mode = args.turret
    _execute(_apply_common(cfg, args), args, live="drive", speed=args.speed, max_spin=args.max_spin)


def _execute(cfg, args, live: str | None, **live_options) -> None:
    """Start Northstar-CV, run the scenario (optionally in the live viewer), evaluate."""
    from nssim.eval import evaluate, format_summary

    cv_dir = _cv_dir(args.cv_dir)
    launcher = None if args.no_launch else _launcher(args, cv_dir)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / f"{stamp}_{cfg.name}"
    try:
        runner = _runner(cfg, cv_dir, out_dir)
    except OSError as e:
        if e.errno in (98, 10048):  # address in use (Linux, Windows)
            sys.exit(f"ports {cfg.frame_port}/{cfg.uart_port}/{cfg.telemetry_port} are in use: is another nssim run "
                     "or drive still going? Only one can run at a time.")
        raise
    session = None
    if live:
        from nssim.live import LiveSession

        session = LiveSession(runner, drive=live == "drive", show_cv=not args.no_cv_window, **live_options)
    dashboard = _start_dashboard(args.dashboard_port, runner) if args.dashboard else None

    if launcher is not None:
        print("[nssim] starting Northstar-CV:", " ".join(launcher.command()), flush=True)
        launcher.start(out_dir / "cv.log")
    try:
        runner.run(session)
    except KeyboardInterrupt:
        print("[nssim] interrupted", flush=True)
    finally:
        if session is not None:
            session.close()
        if launcher is not None:
            launcher.stop()
        if dashboard is not None:
            dashboard.stop()
    summary = evaluate(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(format_summary(summary))
    print(f"[nssim] run saved to {out_dir} (full metrics in summary.json)")


def _start_dashboard(port: int, runner):
    from nssim.dashboard import DashboardServer

    server = _serve(DashboardServer(port=port), "--dashboard-port")
    runner.bus.subscribe(server)
    return server


def _serve(server, port_option: str):
    """Start a dashboard server and say where to find it."""
    from nssim.cv_launcher import local_address_towards

    try:
        server.start()
    except OSError as e:
        sys.exit(f"can't serve the dashboard on port {server.port} ({e.strerror}); pick another with {port_option}")
    urls = [f"http://localhost:{server.port}"]
    try:
        urls.append(f"http://{local_address_towards('8.8.8.8')}:{server.port}  (other devices on your network)")
    except OSError:
        pass  # no network: this machine only
    print("[nssim] dashboard: " + "\n                   ".join(urls), flush=True)
    return server


def cmd_dashboard(args) -> None:
    """Serve a recorded run in the dashboard until Ctrl+C."""
    import time

    from nssim.dashboard import DashboardServer
    from nssim.dashboard.replay import run_events
    from nssim.sim.robot_constants import load_robot_constants

    run_dir = Path(args.run_dir)
    if not (run_dir / "frames.jsonl").is_file():
        sys.exit(f"{run_dir} doesn't look like a run (no frames.jsonl)")
    try:
        yaw_offset = load_robot_constants(paths.cv_dir(args.cv_dir)).yaw_offset
    except (OSError, ValueError):
        yaw_offset = (0.0, 0.0, 0.0)  # only moves the robots' bearing a little
    print(f"[nssim] loading {run_dir}", flush=True)
    server = DashboardServer(port=args.port)
    server.load(run_events(run_dir, yaw_offset))
    _serve(server, "--port")
    print("[nssim] serving until Ctrl+C", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def _launcher(args, cv_dir: Path):
    """Where Northstar-CV runs: over SSH on the Jetson, or in Docker here."""
    from nssim.cv_launcher import CvContainer, CvRemote

    if args.jetson:
        remote = CvRemote(host=args.jetson, cv_dir=args.jetson_dir, build_dir=args.jetson_build,
                          harness_host=args.harness_ip)
        print(f"[nssim] Northstar-CV on {args.jetson} will connect back to {remote.harness_host}", flush=True)
        return remote

    from nssim.cv_build import binary_path, image_exists

    if not binary_path(cv_dir, args.build_dir).is_file():
        sys.exit(f"no simulator build of Northstar-CV in {cv_dir / args.build_dir}; run `nssim build-cv` first")
    if not image_exists(args.image):
        sys.exit(f"Docker image {args.image} not found; run `nssim build-cv` first")
    return CvContainer(cv_dir=cv_dir, image=args.image, build_dir=args.build_dir)


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
    Path(out).parent.mkdir(parents=True, exist_ok=True)
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

    # Options for anything that runs Northstar-CV in the loop.
    common = argparse.ArgumentParser(add_help=False)
    loop = common.add_argument_group("Northstar-CV")
    loop.add_argument("--cv-dir", help="Northstar-CV checkout the constants are read from, and that runs in "
                      "Docker (default: the external/Northstar-CV submodule)")
    loop.add_argument("--jetson", metavar="USER@HOST", help="run Northstar-CV on the Jetson over SSH instead of Docker")
    loop.add_argument("--jetson-dir", default="~/Northstar-CV", help="its Northstar-CV checkout on the Jetson")
    loop.add_argument("--jetson-build", default="build", help="its build directory, relative to --jetson-dir")
    loop.add_argument("--harness-ip", help="this machine's address as the Jetson sees it (default: detected)")
    loop.add_argument("--no-launch", action="store_true", help="don't start Northstar-CV (it's started by hand)")
    loop.add_argument("--image", default="northstar-cv:sim", help="Docker image (without --jetson)")
    loop.add_argument("--build-dir", default="build/sim-x86", help="build directory in --cv-dir (without --jetson)")
    loop.add_argument("--no-compress", action="store_true", help="send raw Bayer frames (fast wired links)")
    loop.add_argument("--plate-height", type=float, metavar="M",
                      help="build the enemies with this plate height offset (default 0.03, what Northstar-CV "
                      "assumes for CU); try another team's value to see the filter cope")
    loop.add_argument("--out", help="run directory (default: runs/<time>_<scenario>)")
    loop.add_argument("--dashboard", action="store_true", help="serve a live dashboard to browsers (any device)")
    loop.add_argument("--dashboard-port", type=int, default=8050, metavar="PORT")

    run = sub.add_parser("run", parents=[common], help="run a scenario with Northstar-CV in the loop")
    run.add_argument("scenario")
    run.add_argument("--duration", type=float)
    run.add_argument("--fps", type=float)
    run.add_argument("--turret", choices=["hold", "ideal", "second_order"])
    run.add_argument("--view", action="store_true", help="watch it live in a 3D viewer (paced to real time)")
    run.add_argument("--no-cv-window", action="store_true", help="with --view: skip the camera window")
    run.set_defaults(func=cmd_run)

    dr = sub.add_parser("drive", parents=[common], help="drive the enemy (and our robot) live while Northstar-CV tracks it")
    dr.add_argument("--distance", type=float, default=3.0, help="starting distance to the enemy (m)")
    dr.add_argument("--enemies", type=int, default=1, choices=[1, 2, 3],
                    help="infantry, + hero, + sentry (Shift / Ctrl drive the 2nd / 3rd, as in TR's sim)")
    dr.add_argument("--speed", type=float, default=0.6, help="m/s while a move key is held (TR's default 0.6)")
    dr.add_argument("--max-spin", type=float, default=4.5, help="rad/s for spin key 1 / 9 (TR's default 4.5)")
    dr.add_argument("--fps", type=float, help="camera frame rate (default 50; 166 runs in slow motion)")
    dr.add_argument("--turret", choices=["hold", "ideal", "second_order"])
    dr.add_argument("--no-cv-window", action="store_true", help="skip the window with the CV's camera image")
    dr.set_defaults(func=cmd_drive)

    db = sub.add_parser("dashboard", help="look through a recorded run in the dashboard")
    db.add_argument("run_dir")
    db.add_argument("--port", type=int, default=8050)
    db.add_argument("--cv-dir")
    db.set_defaults(func=cmd_dashboard)

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
