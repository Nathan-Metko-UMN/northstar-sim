"""Named sets of Northstar-CV tracker switches (its PF_* environment variables), for --preset.

"current" is Northstar-CV as built. "next5a" is the one to use: the latest of the tracker settings
worked out in the simulator tests (NOTES.md), each preset building on the one before. "final" was
the first of them: xy visibility, 5-way averaging, relabeling at 0, best-fit plate matching, calm and
maneuvering velocity modes, spin-proportional radius noise and angle-based settling, per-robot
memory with center height, PnP plate yaw (trusted less with distance), and the light-bar end
correction (by bar size).
"""

from __future__ import annotations

PRESETS: dict[str, dict[str, str]] = {
    "current": {},
    "final": {
        "PF_VISIBILITY_XY": "1",
        "PF_SWAP_REDUCTION": "1",
        "PF_RELABEL_AT_ZERO": "1",
        "PF_ASSOC_BEST": "1",
        "PF_CENTER_XY_VELOCITY_PROCESS_VARIANCE": "0.1",
        "PF_MANEUVER_VELOCITY_VARIANCE": "3",
        "PF_MANEUVER_SWITCH_RATE": "2",
        "PF_RADIUS_MODE": "4",
        "PF_SETTLE_BY_ANGLE": "1",
        "PF_RADIUS_REVERSION_TIME": "2",
        "PF_RADIUS_PROCESS_VARIANCE": "0.001",
        "PF_LOCK_TIME": "12.6",
        "PF_LOCK_FLOOR": "0.1",
        "PF_MEMORY": "1",
        "PF_MEMORY_CENTER_Z": "1",
        "PF_MEMORY_SETTLED": "12.6",
        "PF_YAW_VARIANCE": "0.0685",
        "PF_YAW_INIT": "1",
        "PF_YAW_REF_DISTANCE": "3",
        "LIGHT_END_SHRINK_PX": "0.68",
        "LIGHT_END_SHRINK_K": "17.2",
    },
}
# "final" plus what the spike hunt added: position trusted less with distance, an outlier floor on
# it, and plate size from the number where only one size exists.
PRESETS["next"] = {
    **PRESETS["final"],
    "PF_POSITION_REF_DISTANCE": "3",
    "PF_POSITION_SD_REF": "0.06",
    "PF_POSITION_OUTLIER_WEIGHT": "0.05",
    "ARMOR_TYPE_FROM_NUMBER": "1",
}
# "next" with plate poses solved for yaw and position only (tilt and roll known), and yaw trusted
# to match: 2 degrees up to 3 m, growing as distance^4 beyond (15 degrees at 5 m).
PRESETS["next2"] = {
    **PRESETS["next"],
    "PNP_FIXED_TILT": "1",
    "PF_YAW_VARIANCE": "0.0012",
    "PF_YAW_REF_DISTANCE": "3",
    "PF_YAW_DISTANCE_POWER": "4",
}
# "next2" with light bar ends found to a fraction of a pixel (their own end correction), and the
# aim staying on the plate facing us best instead of the nearest (no flipping between two equal ones),
# and calm particles given little spin-rate noise and fewer particles maneuvering at a time (a
# still robot's spin estimate, and the aim's lead with it, doesn't wobble). A filter started from
# memory can still correct it, and only its own settling writes memory back.
PRESETS["next3"] = {
    **PRESETS["next2"],
    "PF_SPIN_CALM_VARIANCE": "0.01",
    "PF_MANEUVER_SWITCH_RATE": "0.2",
    "PF_MEMORY_FRESH": "1",
    "PF_LOCK_FLOOR": "0.3",
    "LIGHT_ENDS_SUBPIXEL": "1",
    "LIGHT_END_SHRINK_PX": "0.67",
    "LIGHT_END_SHRINK_K": "19.8",
    "AIM_SELECT_FACING": "1",
}
# "next3" with each plate's yaw trusted less on a slowly turning robot (4x the variance when still,
# back to normal by 0.5 rad/s), and the aim's lead ignoring spin under 0.05 rad/s and center speed
# under 2 cm/s (a still robot's estimates wobbling around zero no longer move the aim).
PRESETS["next4"] = {
    **PRESETS["next3"],
    "PF_YAW_SLOW_FACTOR": "4",
    "PF_YAW_SLOW_SPIN": "0.5",
    "AIM_SPIN_DEADBAND": "0.05",
    "AIM_VELOCITY_DEADBAND": "0.02",
}
# "next4" tuned on the replay benchmark (tools/bench.py): yaw trust falling as distance^2 (not ^4),
# maneuvering particles' spin noise 12, calm particles' 0.003.
PRESETS["next5"] = {
    **PRESETS["next4"],
    "PF_YAW_DISTANCE_POWER": "2",
    "PF_ORIENTATION_VELOCITY_PROCESS_VARIANCE": "12",
    "PF_SPIN_CALM_VARIANCE": "0.003",
}
# "next5" with tracking-assisted detection: a plate the number classifier rejected is taken where
# the tracker expects one of the tracked robot's plates (Detector::assistDetections).
PRESETS["next5a"] = {**PRESETS["next5"], "DETECT_ASSIST": "1"}
