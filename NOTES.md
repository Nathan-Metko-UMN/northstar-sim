# Findings and calibration notes

What the harness has turned up in Northstar-CV so far, and the choices behind the sim's
appearance model. Numbers are from the `spinning` scenario (6 rad/s at 3 m) unless noted.

## Northstar-CV issues found

**Odometry yaw interpolated across the 0/2π wrap** (fixed on `sim-harness`, 8888933). The MCB
sends yaw wrapped to [0, 2π) and `OdometryBuffer::interpolate` interpolated it linearly, so a
frame captured between samples on either side of the wrap (6.28 → 0.01) got a yaw near π: the
camera was modelled facing backwards and the plate landed ~3.9 m away. That happens on the robot
whenever the turret yaw crosses the IMU's zero heading. In the sim, three such outliers pushed the
filter's spin estimate from 7 to 10.4 rad/s, it lost the spin, and it re-converged with the high
and low plate pairs swapped.

| filter error vs ground truth | before the fix | after |
|---|---|---|
| worst plate measurement | 3.9 m | 0.16 m |
| orientation rms | 0.745 rad (pairs swapped) | 0.164 rad, never beyond ±30° |
| omega rms | 2.7 rad/s | 0.9 rad/s |
| center velocity (target is still) | 0.42 m/s | 0.14 m/s |
| radius mean | +0.045 m | −0.018 m |
| center xy mean | 0.11 m | 0.076 m |

The remaining center offset (steady ~7.6 cm) is probably the plates' short range, below.

**The particle filter accepts gross outliers.** Those 3.9 m measurements went straight into the
filter. An innovation gate (reject observations far outside the predicted plates) would have
contained the damage.

**Labels with CRLF line endings** (fixed on `sim-harness`, fd519e1). On a Windows checkout
`model/label.txt` gets CRLF (`.gitattributes * text=auto`), so every label was "3\r": the plate
height lookup, the MCB target ID and the negative/plate-size checks all failed. Linux checkouts
(the robot) were never affected.

**Thin light bars are dropped.** `findLights` skips contours with fewer than 3 points, and a bar
that thresholds to a one-pixel line comes back from `findContours(CHAIN_APPROX_SIMPLE)` as 2
points. Expect long-range detections to drop out once bars are ~1–2 px wide.

**The number classifier is brittle.** On sim crops of a "3", scaling the 20x28 crop about its
center gives P("3"): 0.06 at 0.95x, 0.47 at 1.0x, 0.94 at 1.05x, ≥0.99 from 1.1x to 1.6x.
Shifting it 2 px up or down drops P("3") below 0.1. Anything that changes the detected light
length relative to the sticker (exposure, glare, a different lens) moves plates across that edge.
Worth training the next classifier (or the TensorRT detector) with scale and shift augmentation.

**Range reads short, more so at an angle**: −2 cm head-on to −4…−6 cm at 45–60° (3 m). The light
bars come out of `findLights` 0.4 px (head-on) to 0.8 px (45–60°) longer than they are: their ends
are the bounding box of the thresholded blob, which includes the glow past the bar ends. Bar spacing
is right, so PnP puts the plate 1–2% too close, and the filter's center follows (the center sat
4–7 cm toward the camera). Small far bars go the other way (dim ends fall under the threshold:
−0.17 px per end at 20 px long), so the correction depends on bar length:
`LIGHT_END_SHRINK_PX − LIGHT_END_SHRINK_K / length` (0.68 px, 17.2 px; with sub-pixel ends,
`LIGHT_ENDS_SUBPIXEL`, 0.67 and 19.8). That keeps range within ±2 cm from 2 to 5.5 m. The real
camera's glow differs: set it there against tape-measured plates at a few distances. Whole-pixel
bar ends also make range jump as a plate's image creeps (11–130 mm on a still robot watched by a
moving turret); sub-pixel ends take most of that out.

**The filter ignored the plate yaw PnP measures.** With positions alone, a robot that stops at an
angle drifted toward "plate faces the camera" (~25°, the center 11 cm off). PnP's yaw (outward
normal = −x of its armor frame) has a median error within ±2° but 6–21° spread and 3–5% outliers
beyond 30°; scored with σ = 15° and an outlier floor (`PF_YAW_VARIANCE`, `PF_YAW_INIT`) it takes
the stopped center error from 11 cm to 1 cm and halves the center kick at plate switches.

**Full PnP on a far plate trades rotation against range.** A small plate seen nearly head-on is
PnP's weak case: in the sim its yaw spread is 5° at 2 m and 52° at 5 m (28% beyond 30°). The
plate's 15° tilt and zero roll are known, so solving for yaw and position only
(`PnPSolver::solveFixedTilt`, `PNP_FIXED_TILT`) takes that to 0.4° and 14°, and range spread down a
little. Check the tilt on the real robot.

**The number classifier misses plates turned past ~50°**, the hero's and sentry's most (58–66% of
their candidates read "negative" at 50–70°; light-bar pairing still finds 97% of them). A confident
"1" was also thrown away when the bar spacing said "small plate" (a large plate turned past ~35°);
the number fixes the size instead (`ARMOR_TYPE_FROM_NUMBER`). Tracking-assisted detection
(`DETECT_ASSIST`) takes a rejected pair where the tracker expects one of the tracked robot's plates
(12 cm, facing the same way within 30°): hero and sentry recall 68–80% → 95–98%, no misreads.

**Up close, plates leave the image.** Within 1.2 m most missed plates are below or beside the
frame (the robot is wider and lower than the camera sees), not missed by the detector; the tracker
coasts on its prediction (1 s) and per-robot memory restores the geometry.

## Tracker (particle filter) findings

From `nssim pftest` (every robot type and plate layout driving a tour; replays in
`tools/replay.py` and `tools/bench.py`), on `claude/pf-height-tests`. The switches are environment
variables read by `src/sim/pf_overrides.hpp`; `nssim/pf_presets.py` collects them (`--preset`).

- **3D visibility was spurious evidence.** The observer is our base, ~0.2 m below the enemy's
  center, so tilting plates up or down changed how "visible" they scored: on a still robot the
  filter picked the wrong high pair 100% of the time and shrank a radius to the 10 cm clamp. Score
  visibility in xy (`PF_VISIBILITY_XY`).
- **The particle average blended the two high/low labelings 90° apart.** Average with the ±90°
  relabelings too (`PF_SWAP_REDUCTION`), and let a particle change which pair it calls high by
  relabeling at dz = 0 rather than reflecting (`PF_RELABEL_AT_ZERO`).
- **A lone plate was always matched to the predicted plate nearest the camera**, which near a
  plate switch is often the wrong one (`PF_ASSOC_BEST`).
- **Strafing was lost** with center-velocity noise 0.3; one fixed value can't serve both a dodging
  and a still robot. Each particle is calm (0.1) or maneuvering (3), switching at 0.2/s.
- **Radius**: noise only in proportion to spin, a pull toward the prior only before the robot has
  turned, settling by radians turned (`PF_RADIUS_MODE=4`, `PF_SETTLE_BY_ANGLE`); the old spin-gated
  pull dragged learned radii back whenever the robot slowed.
- **Measurement trust by distance**: position as distance² (`PF_POSITION_REF_DISTANCE`), yaw as
  distance² from 2° at 3 m with the fixed-tilt PnP, an outlier floor on position.
- **Still-robot turret shake** came from the spin estimate wobbling ±0.05 rad/s, which the aim
  multiplies by its lead (~120 ms). Calm particles' spin noise 0.003, maneuvering 12, yaw trusted
  4x less when turning under 0.5 rad/s, and small dead-bands on the aim's lead (0.05 rad/s, 2 cm/s)
  take the turret to 0.001° spread (Northstar-CV as built: 0.013°, its estimate barely moves
  because it has no yaw).
- **Plate numbers aren't stable through extrapolation** (orientation wraps every 180°, swapping
  0↔2): anything that follows "plate i" into the future has to follow it by angle.
- **Per-robot memory** must not inherit settledness: a filter restarted from memory that counted as
  settled wrote its own (bad, close-range) state back, and the height offset stuck at 0
  (`PF_MEMORY_FRESH`).

Tracker test, 12 runs (50 fps, no sensor noise): Northstar-CV as built 595 spikes (aimed plate
>10 cm off), 5.0 cm aim error; `--preset next5a` 1 spike, 1.1 cm.

## Appearance calibration (`nssim/sim/scene.py`, `nssim/camera/optics.py`)

The sim has no real footage to match yet, so these are set from physics and from what
Northstar-CV itself was tuned on:

- **Light bars**: TR's CAD size (7.4 mm wide, scaled to PnP's 57 mm length), drawn proud of the
  panel. TR's own bars are recessed behind the panel rim, which hid the far bar beyond ~40°.
- **LED color**: emission (0.3, 0.75, 1.0) for blue and (1.0, 0.3, 0.18) for red at strength
  1.2, so the dominant channel just saturates. Blue renders as about (160, 243, 255), hue ≈ 99 in
  OpenCV units, inside the 80–115 band of the detector's old HSV thresholds, which were tuned on
  the real camera. SAPIEN's 8-bit output is sRGB-encoded (emission 0.5 renders as 186).
- **Lens**: Gaussian PSF σ = 0.7 px and a glare halo (σ = 2 px, 35% of saturated light, added in
  linear light) before the Bayer mosaic. Without them a 5 mm pure-blue bar could put all of its
  blue in one column of Bayer blue sites and vanish after debayering (the thin-bar drop above).
- **Sticker size**: from DJI's reference stickers (RoboMaster 2026 rules, Appendix II). Measured
  on the drawings, each glyph's bounding box is this fraction of the sticker's height (and width):
  the 3 0.841 (0.480), the hero's 1 0.815 (0.193), the sentry icon 0.700 (0.715), the 2 0.833
  (0.497), the 4 0.820 (0.601), the outpost icon 0.775 (0.474); all centered but the 1, whose
  flag puts its box 0.028 of the width left. The sticker covers the panel's front face (1.085
  wide per high on TR's small panels, as on DJI's), so the harness sizes and places TR's glyphs
  by those fractions of the face. TR's glyphs have DJI's shapes (width per height within 2%) but
  are smaller: the 3 needs 1.135x, the sentry icon 1.03x, the 1 1.02x. The earlier uniform
  1.25x, the middle of the range in which the classifier is confident on the 3 (1.1x to 1.6x;
  it rejects below 1.05x), made all three too big, the 1 and the sentry icon by about 20%. Run
  offline on rendered plates, the classifier reads all three alike from 0.9x to 1.3x of DJI's
  size, so the size isn't critical; TR's glyphs also fill their boxes as DJI's do (the 3 52%,
  the 1 56%, the sentry outline only 26%).
- **Robot body**: a dark box inside the plate ring, from 3 cm off the floor to 7 cm above the
  plate centers. The classifier's window (28 px for 133 mm, centered a little low on the light
  bars) reaches below a plate's bottom edge. When the box stopped 13 cm off the floor, the camera
  looked down past the plate onto the gray floor from closer than about 2 m, the threshold made
  that strip part of the glyph, and the 1 and the sentry icon read as "negative" (recall 0.00 at
  1.5 m; 0.47 and 0.65 with the full box, the 3 0.92 either way).
- **Hero and sentry fall off with angle sooner than the 3** (3 m spinning: the 3 is kept 91% at
  40-60 degrees of incidence, the sentry 49%, the hero 0%). That is the CV, not the sim: it calls
  a pair of bars a large plate from their spacing per length, and beyond about 35 degrees a large
  plate's bars are as close as a small one's head on, so a hero plate is treated as small and its
  1 rejected ("wrong plate size") or warped to the small shape and read as negative. The sentry's
  thin outline blurs away sooner than the 3's strokes once the plate turns.

Results with these settings: recall 100% below 40° incidence, 87–96% at 40–60°, 62% at 60–70°;
no misread numbers; corner error ~1 px.

## Plate height offset

Northstar-CV doesn't estimate how far the high plate pair sits above the low one; the particle
filter takes it as a fixed parameter, looked up by the classified number in
`config/plate_heights.yaml` (active team CU: 0.03 m for "3" and "guard", 0.032 m for "1"; 0.04 m
for numbers not in the file). The sim builds every enemy with 0.03 m, so by default the enemy is
exactly what the filter assumes. `--plate-height` builds them differently, e.g. 0.046 (ARUW's and
UBC's infantry in that file) to see how the filter copes with a wrong assumption.

## Open questions

- Real footage at a known distance and exposure would pin down the glare and bar brightness.
- The MCB's fire-gate plate table only has IDs 1, 3 and 7.
- Pitch is now the IMU's (was the encoder): re-check its sign and zero on the robot.
