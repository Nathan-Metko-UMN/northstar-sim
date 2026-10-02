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

Northstar-CV runs either in Docker on the same machine (needs an NVIDIA GPU) or on the Jetson (see
[Running Northstar-CV on the Jetson](#running-northstar-cv-on-the-jetson)). Either way the sim
renders with SAPIEN on this machine's GPU. NVIDIA GPUs work, and so does the Intel Arc 140V of a
Core Ultra 200V laptop; Intel's older UHD 770 doesn't (SAPIEN fails at the first frame with
`ErrorOutOfPoolMemory`). nssim switches SAPIEN's ray tracing off (`SAPIEN_DISABLE_RAY_TRACING=1`
unless you set it): it isn't used, and on Intel Arc it keeps SAPIEN from starting at all.

With more than one GPU, `NSSIM_RENDER_DEVICE=cuda` renders on the NVIDIA one. If SAPIEN still can't
start (`ErrorFeatureNotPresent`, even from
`python -c "import sapien; print(sapien.render.get_device_summary())"`), a GPU whose Vulkan driver
lacks something SAPIEN needs may be in the way, even if it's not the one you want: try hiding the
others with the Vulkan loader's `VK_LOADER_DRIVERS_SELECT=*nv-vk64*` (NVIDIA's driver on Windows).

Everything below: git and Python 3.10–3.12. Windows (PowerShell):

```powershell
git clone --recursive <this repo's URL>      # already cloned? git submodule update --init --recursive
cd northstar-sim
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[sim,dev]"     # ~330 MB
.venv\Scripts\nssim preview spinning --time 1           # checks the renderer: writes .scratch\preview_spin_6.png
.venv\Scripts\nssim build-cv                            # only to run Northstar-CV here, in Docker
```

Linux: the same with `python3.11 -m venv .venv` and `.venv/bin/` in place of `.venv\Scripts\`.
`--recursive` also fetches Northstar-CV's particle-filter library, which only a machine that builds
Northstar-CV needs; `git submodule update --init` without it is enough otherwise. If a submodule's
folder has only a `.git` file in it, its clone was cut short before the files were written, and a
plain update won't notice; `git submodule update --init --force external/<name>` writes them.

`nssim build-cv` builds Northstar-CV's own dev container (`.devcontainer/Dockerfile`, the image
`scripts/dev.sh` uses, about 15 GB the first time) as `northstar-cv:jetpack7`, and
`northstar-cv:sim-jetpack7` on top of it, which adds socat. It then compiles Northstar-CV inside
them into `external/Northstar-CV/build/sim-jetpack7`, for your GPU and the Orin's (sm_87). Run it
again whenever Northstar-CV changes; the images are only rebuilt when the Dockerfile changes, or
from scratch with `--rebuild-images`, and when an image has changed (another CUDA, say) its build
directory starts afresh.

The robot may run JetPack 6 (CUDA 12.6, Ubuntu 22.04) or JetPack 7 (CUDA 13.2, Ubuntu 24.04), and
the `sim-harness` branch builds on both. The dev container has a build of each, and the default
above is JetPack 7's; to build and run with JetPack 6's instead (images `northstar-cv:jetpack6`
and `northstar-cv:sim-jetpack6`, build in `build/sim-jetpack6`):

```powershell
.venv\Scripts\nssim build-cv --jetpack 6
.venv\Scripts\nssim run spinning --jetpack 6
```

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

## Live dashboard

Add `--dashboard` to `run` or `drive` and open the address it prints in any browser, on this
machine or another device on the network (a phone works):

```powershell
.venv\Scripts\nssim drive --dashboard       # prints http://localhost:8050 and http://<this machine>:8050
```

The Overview tab shows the serial traffic both ways (rate and latest values of every message type,
and a filterable log), Northstar-CV's filter against the truth (spin, radius, center, orientation),
plates visible vs detected, the turret's yaw against the CV's aim and the robot's true bearing, and
the latency from capture to the frame arriving and to its aim reaching the MCB.

**Your own tabs and graphs.** `+ New tab` makes a tab (double-click its name to rename it); `+ Add
graph` puts a graph in it. A graph plots any numbers the run produces: every field of every serial
message (`uart.mcb_to_cv.ODOMETRY.yaw`), everything Northstar-CV reports (`cv.track.state.omega`,
`cv.track.ballistics.time_of_flight`), the truth (`truth.targets.infantry.center.x`), the errors
(`metrics.filter.radius_m`) and frame timing (`frame.processing_ms`). Type in the signal box to
search; `*` matches any part (`truth.targets.*.bearing` is every robot's). Each signal can be
converted (rad → °, m → cm, s → ms), and the y axis is automatic (optionally keeping 0 in view) or a
fixed min/max. Tabs and graphs are saved on the sim machine (`.nssim/dashboard_layout.json`), so every
browser sees the same ones.

**Time.** All graphs share one time range. Live shows the last `window` (type any length: `45s`,
`10m`). Pause holds the view, and from/to set any range. Drag across a graph to zoom in, Shift+drag
to pan, Ctrl+scroll (or pinch) to zoom around the pointer, double-click to go back to live; hover
to read values. Zoomed and long ranges come from the server at full detail, so zooming in shows
every sample.

A browser that connects mid-run gets the whole run so far; one left open picks up the next run by
itself. Everything also lands in the run directory as before, now including `uart.jsonl` (every
serial message, decoded). To look through a run afterwards (the live dashboard stops with the run):

```powershell
.venv\Scripts\nssim dashboard runs\<run>     # serves it until Ctrl+C
```

Other devices need the port open like the Jetson's: allow TCP 8050 in the firewall (see below),
or pick another with `--dashboard-port`. It is view-only and has no login, so keep it to networks
you trust.

## Running Northstar-CV on the Jetson

The sim stays on your machine and Northstar-CV runs on the Jetson, so the processing time (and so
the aim latency) is the Jetson's own. Any of `nssim run`, `run --view` and `drive` take `--jetson`;
if Northstar-CV is built in its dev container there (`scripts/dev.sh`), add that image too:

```powershell
.venv\Scripts\nssim drive --jetson nvidia@192.168.1.50 --jetson-image northstar-cv:jetpack6
```

It logs in over SSH, starts Northstar-CV in a container of that image (as `dev.sh` runs one: the
GPU, and the checkout at `/ws`) pointed back at this machine, and removes it at the end. Without
`--jetson-image` it runs Northstar-CV natively there. `--jetson` also takes a Host from your
`~/.ssh/config` (`--jetson jetson1`). `--jetson-dir` (default `~/Northstar-CV`) is the checkout
and `--jetson-build` its build directory (default `build`, or `build/jetpack6` with JetPack 6's
image, where `dev.sh` builds); `--harness-ip` overrides the address the Jetson is told to connect
back to (found from the route to it).

Once, on the Jetson:

1. Check out Northstar-CV's `sim-harness` branch, the commit `external/Northstar-CV` is pinned to
   (this repo reads the robot constants and plate sizes from the submodule, so keep them the same).
2. Build it. In the dev container, which also builds the image:

   ```bash
   ./scripts/dev.sh bash -c "cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build"
   ```

   Natively instead: build it the way you do for the robot, and `sudo apt install socat` (it
   turns the harness's TCP link into the PTY Northstar-CV opens as its UART; the image has it).

Once, on this machine:

1. SSH without a password prompt (skip if `ssh <jetson>` already logs in with a key):
   `ssh-keygen -t ed25519` (skip if you have a key), then
   `type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh nvidia@<jetson> "cat >> ~/.ssh/authorized_keys"`
   (Linux: `ssh-copy-id nvidia@<jetson>`).
2. Let the Jetson in: it connects to TCP 5600 and 5760 and sends UDP to 5800 on this machine. On
   Windows, allow Python when the firewall asks, and check the connection's network profile: a
   direct Ethernet link often comes up as Public, where that allowance doesn't apply. Either set it
   to Private or add rules (PowerShell as administrator):

   ```powershell
   New-NetFirewallRule -DisplayName nssim-tcp -Direction Inbound -Protocol TCP -LocalPort 5600,5760 -Action Allow
   New-NetFirewallRule -DisplayName nssim-udp -Direction Inbound -Protocol UDP -LocalPort 5800 -Action Allow
   ```

Frames are compressed by default (about 50 KB each), which suits Wi-Fi. On a wired gigabit or
faster link `--no-compress` sends raw frames (1.5 MB) and spares the Jetson the decompression.
Neither counts toward the processing time Northstar-CV reports. To start Northstar-CV by hand
instead, run with `--no-launch` and use the command `nssim` would have printed.

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
  LEDs (cyan-white for blue, orange-red for red). The sticker glyphs (TR's 3, hero 1 and sentry
  icon) are sized and placed as on DJI's reference stickers (RoboMaster 2026 rules, Appendix II;
  `STICKER_GLYPHS` in `nssim/sim/scene.py`).
- **Camera.** The Triton2 TRT016S-CC (1440x1080, BayerRG8) with the 6 mm lens's nominal
  intrinsics (fx = fy = 1739 px) until the real lens is calibrated. Lens blur and glare around
  the light bars are applied before the Bayer mosaic (`nssim/camera/optics.py`), and
  Northstar-CV debayers the frame itself, as on the robot.
- **Our robot.** Turret geometry is read from Northstar-CV's `src/constants.hpp`.
- **Enemies.** Four plates 90 degrees apart, high pair at +z_offset (the particle filter's
  model), 15 degree tilt, scripted translation/strafe/spin. Their size is `TargetSpec` in
  `nssim/sim/targets.py` (per enemy in `nssim/scenarios.py`): `radius_high`/`radius_low` from the
  center to each pair of plates (0.25 m), `z_offset` (0.03 m), `center_height` (0.20 m) and
  `tilt_deg`; the plate size comes from the panel (small for infantry and sentry, large for the
  hero). `--radius`, `--radius-low` and `--plate-height` set them for a run without editing code.
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
nssim/bus.py       the run's event stream (serial traffic, frames, truth, errors)
nssim/metrics.py   per-frame errors against the truth (eval.py sums them up)
nssim/dashboard    the browser dashboard: server and page
nssim/eval.py      metrics
nssim/cv_build.py  Docker images + Northstar-CV build
nssim/debug.py     re-render logged frames
```
