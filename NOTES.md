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

**Range reads short by about 2%** (−5 cm at 2.4 m). Glare makes the thresholded bars ~2% longer
than the 57 mm PnP assumes, so plates come out closer. The real camera will have its own version
of this; check against a tape-measured plate.

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
  it rejects below 1.05x), made all three too big, the 1 and the sentry icon by about 20%.

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
