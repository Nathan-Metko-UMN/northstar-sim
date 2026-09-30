# northstar-sim

Hardware-in-the-loop test harness for [Northstar-CV](../Northstar-CV) auto-aim, built on the
[TR ARCTIC 2026 simulator](https://github.com/Triton-Robotics/TR-Simulation-ARCTIC-2026)'s
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

## Setup (Windows or Linux)

```bash
py -3.11 -m venv .venv            # Python 3.10-3.12 (mani_skill pins numpy<2)
.venv/Scripts/python -m pip install -e ".[sim,viz,dev]"
```

The sim needs a Vulkan GPU; it renders natively (not in Docker). It expects the TR simulator
checked out next to this repo (or set `TR_SIM_DIR`), and Northstar-CV next to it (or set
`NORTHSTAR_CV_DIR`). Northstar-CV must be on a branch with the simulator mode
(`sim-harness`).

Build Northstar-CV and the container image it runs in (local Docker stand-in for the Jetson):

```bash
docker run --rm -v <Northstar-CV>:/ws -w /ws northstar-cv:dev \
  bash -lc "cmake -S . -B build/sim-x86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build/sim-x86"
docker build -t northstar-cv:sim -f docker/cv-sim.Dockerfile docker
```

## Running

```bash
nssim run static_plate            # frame check: a still plate, turret holds
nssim run spinning                # 6 rad/s spinner at 3 m, turret follows the CV's aim
nssim run strafing
nssim preview spinning --time 1.5 # render one frame with ground-truth corners overlaid
nssim eval runs/<run>             # re-run the evaluation of a saved run
```

`nssim run` starts Northstar-CV in Docker, runs the scenario, stops the container and prints
the evaluation. Each run directory has `frames.jsonl` (ground truth per frame),
`telemetry.jsonl` (what Northstar-CV reported), `events.jsonl`, `cv.log`, `config.json` and
`summary.json`.

## What is simulated, and how

- **Scene.** TR's field and armor panel models, rendered with SAPIEN. Robots are kinematic:
  poses come straight from scripted motion and the turret model, so ground truth is exact.
  Panels are scaled to the plate size Northstar-CV's PnP assumes (`src/pnp_solver.hpp`).
- **Our robot.** Turret geometry is read from Northstar-CV's `src/constants.hpp`, and the
  camera is the Triton2 TRT016S-CC (1440x1080) with the 6 mm lens's nominal intrinsics
  (fx = fy = 1739 px) until the real lens is calibrated.
- **Enemies.** Four plates 90 degrees apart, high pair at +z_offset (the particle filter's
  model), 15 degree tilt, scripted translation/strafe/spin.
- **MCB.** The UART protocol, send periods, robot-ID timer, ALIVE timeout and fire-gate
  tolerance of northstar-robomaster `main`. The turret is a placeholder second-order model
  for now; the firmware's cascade PID comes next.

## Layout

```
nssim/protocol   DJI serial framing, CRCs, message layouts
nssim/mcb        virtual MCB, UART link, turret model
nssim/camera     Triton2 camera model, Bayer mosaic, frame stream
nssim/sim        SAPIEN scene, TR assets, targets, our turret kinematics
nssim/runner.py  paced run loop
nssim/eval.py    metrics
```
