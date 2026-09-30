# northstar-sim

Hardware-in-the-loop test harness for [Northstar-CV](https://github.com/Northstar-Advanced-Robotics/Northstar-CV)
auto-aim, built on the [TR ARCTIC 2026 simulator](https://github.com/Triton-Robotics/TR-Simulation-ARCTIC-2026)'s
field and armor models.

The harness plays everything around Northstar-CV — the camera, the MCB and the enemy robot —
and compares what the pipeline believes (detections, particle filter state, aim) against
ground truth.

```
SIM HOST (Windows/Linux, runs this repo)            NORTHSTAR-CV (Docker here, or the Jetson)
  scripted enemies + our turret (1 ms ticks)
  SAPIEN render of the ARC field --- Bayer8 TCP ----> SimCamera          (--sim host:5600)
  virtual MCB (115200 baud) <------ UART/TCP -------> socat PTY          (--uart /tmp/ttySIM)
  telemetry + ground truth logs <--- UDP ------------ detections/PF/aim  (--telemetry host:5800)
```

## Setup

You need an NVIDIA GPU (the sim renders with Vulkan and Northstar-CV's particle filter runs on
CUDA), Docker (Docker Desktop on Windows), git and Python 3.10–3.12.

Windows (PowerShell):

```powershell
git clone --recursive <this repo's URL>      # already cloned? git submodule update --init --recursive
cd northstar-sim
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[sim,dev]"
.venv\Scripts\nssim build-cv
```

Linux: the same with `python3.11 -m venv .venv` and `.venv/bin/` in place of `.venv\Scripts\`.

`nssim build-cv` builds two Docker images the first time: `northstar-cv:dev` from Northstar-CV's
own `.devcontainer/Dockerfile` (the CUDA dev image, about 15 GB), and `northstar-cv:sim` on top
of it, which adds socat. It then compiles Northstar-CV inside them into
`external/Northstar-CV/build/sim-x86`, detecting your GPU's CUDA architecture. Run it again
whenever Northstar-CV changes.

## Running

```powershell
.venv\Scripts\nssim run spinning                  # 6 rad/s spinner at 3 m, turret follows the CV's aim
.venv\Scripts\nssim run static_plate              # a still plate, turret holds
.venv\Scripts\nssim run strafing
.venv\Scripts\nssim preview spinning --time 1.5   # one frame as the CV sees it, ground truth overlaid
.venv\Scripts\nssim frame runs\<run> <seq>        # re-render a logged frame with the CV's detections
.venv\Scripts\nssim eval runs\<run>               # re-run the evaluation of a saved run
```

`nssim run` starts Northstar-CV in Docker, runs the scenario, stops the container and prints
the evaluation. Options: `--duration`, `--fps`, `--turret hold|ideal|second_order`,
`--out <dir>`. Each run directory has `frames.jsonl` (ground truth per frame),
`telemetry.jsonl` (what Northstar-CV reported), `events.jsonl`, `cv.log`, `config.json` and
`summary.json`.

## Where Northstar-CV and the TR assets come from

`external/Northstar-CV` and `external/TR-Simulation-ARCTIC-2026` are git submodules pinned to
the commits this harness was last checked against. The Northstar-CV one follows the
`sim-harness` branch, where the simulator mode lives until it is merged.

To test another Northstar-CV checkout, for example your working copy with uncommitted changes,
pass it to both commands:

```powershell
.venv\Scripts\nssim build-cv --cv-dir D:\Robomaster\Northstar-CV
.venv\Scripts\nssim run spinning --cv-dir D:\Robomaster\Northstar-CV
```

or set `NORTHSTAR_CV_DIR` once (`TR_SIM_DIR` does the same for the TR assets). You can also
work in the submodule itself: `git -C external/Northstar-CV switch sim-harness`, commit and
push there as usual, then record the new pin here with `git add external/Northstar-CV`.

A clone elsewhere can only fetch a pinned commit that has been pushed, so push Northstar-CV's
`sim-harness` before pushing a new pin.

## Paced mode

The sim renders one frame at a time and waits for Northstar-CV, so any machine can test the
pipeline at the full 166 fps camera rate. Time is the MCB clock:

1. Each frame is rendered at mid-exposure and ground truth is recorded for that instant.
2. It is delivered at its arrival time: end of exposure plus the GigE transfer (5.2 ms for a
   1440x1080 BayerRG8 frame on 2.5 GbE). Odometry that has arrived over the UART by then is
   delivered first.
3. Northstar-CV acks with how long it really took; its aim command reaches the virtual MCB at
   arrival + processing + UART time, and the turret moves from then on.

Northstar-CV runs on the MCB clock in this mode (`src/sim/sim_clock.hpp`), so latency
compensation and the filter's dt are what they would be on the robot.

## What is simulated, and how

- **Scene.** TR's field and armor panel models, rendered with SAPIEN. Robots are kinematic:
  poses come straight from scripted motion and the turret model, so ground truth is exact.
  Panels are scaled to the plate size Northstar-CV's PnP assumes (`src/pnp_solver.hpp`).
- **Armor plates.** Light bars at TR's CAD size, in the colors a color camera records for the
  LEDs (cyan-white for blue, orange-red for red). TR's sticker digits are enlarged 1.25x, the
  middle of the range Northstar-CV's number classifier accepts; real stickers should be
  checked against this.
- **Camera.** The Triton2 TRT016S-CC (1440x1080, BayerRG8) with the 6 mm lens's nominal
  intrinsics (fx = fy = 1739 px) until the real lens is calibrated. Lens blur and glare around
  the light bars are applied before the Bayer mosaic (`nssim/camera/optics.py`), and
  Northstar-CV debayers the frame itself, as on the robot.
- **Our robot.** Turret geometry is read from Northstar-CV's `src/constants.hpp`.
- **Enemies.** Four plates 90 degrees apart, high pair at +z_offset (the particle filter's
  model), 15 degree tilt, scripted translation/strafe/spin.
- **MCB.** The UART protocol, send periods, robot-ID timer, ALIVE timeout and fire-gate
  tolerance of northstar-robomaster `main`. The turret is a placeholder second-order model
  for now; the firmware's cascade PID comes next.

## Layout

```
external/          Northstar-CV and TR simulator submodules
docker/            the sim image (Northstar-CV dev image + socat)
nssim/protocol     DJI serial framing, CRCs, message layouts
nssim/mcb          virtual MCB, UART link, turret model
nssim/camera       Triton2 camera model, lens effects, Bayer mosaic, frame stream
nssim/sim          SAPIEN scene, TR assets, targets, our turret kinematics
nssim/runner.py    paced run loop
nssim/eval.py      metrics
nssim/cv_build.py  Docker images + Northstar-CV build
nssim/debug.py     re-render logged frames
```
