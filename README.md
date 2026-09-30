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

## Watching and driving

```powershell
.venv\Scripts\nssim drive                    # drive the enemy yourself while Northstar-CV tracks it
.venv\Scripts\nssim drive --enemies 3        # infantry, hero and sentry
.venv\Scripts\nssim run spinning --view      # watch any scenario live
```

Both open SAPIEN's 3D viewer on the arena and a second window with the frame Northstar-CV
received (green: true light bars, cyan: its detections, magenta: candidates its classifier
rejected). Wireframes in the viewer show our robot (white), where the turret points (red), the aim
Northstar-CV sent (yellow), and its particle filter: the four plates and center it believes in
(magenta) against the enemy's true center (green), plus its ballistic aim point (yellow). The panel
has live numbers: filter spin and radius against the truth, and the real-time factor.

Keys (with the viewer window focused) are the TR simulator's, so the same fingers work in both:
its secondary robot, the one it uses for aiming practice, is our enemy, and its primary robot is
ours. Like TR, WASD is left to the viewer's camera.

| key | |
|---|---|
| I / K, J / L | enemy forward / back, left / right (field axes) |
| 5 | enemy stops spinning |
| 4 3 2 1 | enemy spins counterclockwise at 25 / 50 / 75 / 100 % |
| 6 7 8 9 | enemy spins clockwise at 25 / 50 / 75 / 100 % |
| Shift / Ctrl + the enemy keys | the 2nd / 3rd enemy (`--enemies 2` or `3`: a hero, then a sentry) |
| T / G, F / H | our robot forward / back, left / right (relative to where the turret points) |
| R / Y | turn our chassis |
| V | ride on our turret camera / back to the free view (not in TR) |
| 0 | reset positions (not in TR) |
| mouse, W A S D | move the view (right-drag turns, scroll zooms) |

Speeds are TR's too: 0.6 m/s while a move key is held (`--speed`) and spin presets up to
4.5 rad/s (`--max-spin`).

The run is paced to the wall clock. With Northstar-CV in Docker Desktop each frame costs about
20 ms of wall time, so `drive` defaults to a 50 fps camera, which runs close to real time;
`--fps 166` gives the real camera rate in slow motion (the panel shows the factor). Close the
viewer to stop; the run is saved and evaluated like any other.

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

## How this relates to TR's simulator

The harness doesn't run TR's simulator and doesn't change it: `external/TR-Simulation-ARCTIC-2026`
is TR's repository exactly as published, used for its field and armor-panel models (and its
keyboard layout). The simulation itself is in this repo: `nssim/sim` builds a SAPIEN scene from
those models and moves the robots, `nssim/camera` models the camera and lens, `nssim/mcb` plays
the MCB.

TR's simulator is a ROS 2 node around ManiSkill physics that runs in real time and publishes a
camera image. Testing auto-aim needed things it isn't built for:

- lockstep with Northstar-CV: render a frame, hand it over, wait for the aim. That is what makes
  166 fps possible on any machine and makes runs repeatable;
- exact ground truth for every frame (plate corners in pixels, robot poses and spin), so errors
  are measured rather than eyeballed;
- the MCB's serial protocol and timing instead of ROS topics, so Northstar-CV runs unmodified apart
  from its camera source;
- no ROS, so it runs natively on Windows.

It also adds what the detector needs to behave as it does on real footage: LED light bars that
stay visible at an angle, lens blur and glare, the Bayer mosaic, and sticker digits at the size
the number classifier expects (see NOTES.md).

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
nssim/live.py      live viewer and keyboard driving
nssim/eval.py      metrics
nssim/cv_build.py  Docker images + Northstar-CV build
nssim/debug.py     re-render logged frames
```
