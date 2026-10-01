"""Northstar sim harness: drives Northstar-CV from the TR ARCTIC 2026 simulator.

The harness plays the parts of the robot that Northstar-CV talks to:

- ``nssim.protocol``: the DJI-framed UART protocol shared with the MCB firmware.
- ``nssim.mcb``: a virtual MCB that speaks that protocol (odometry, robot ID, aim input).
- ``nssim.camera``: a Triton2 twin that streams Bayer frames to Northstar-CV's SimCamera.
"""

import os

__version__ = "0.1.0"

# SAPIEN asks for its ray tracing features on any GPU that offers ray tracing, without checking it
# has them all. Intel Arc graphics (a Core Ultra 200V laptop's 140V) lacks one, and SAPIEN then
# can't start at all (ErrorFeatureNotPresent). The harness only rasterizes, so it's off unless set
# otherwise. This must happen before anything loads sapien.
os.environ.setdefault("SAPIEN_DISABLE_RAY_TRACING", "1")
