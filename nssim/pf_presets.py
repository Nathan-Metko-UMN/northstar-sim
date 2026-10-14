"""Named sets of Northstar-CV tracker switches (its PF_* environment variables), for --preset.

"current" is Northstar-CV as built; "final" is the tracker settled on in the simulator tests (see
NOTES.md): xy visibility, 5-way averaging, relabeling at 0, best-fit plate matching, calm and
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
