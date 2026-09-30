"""Northstar sim harness: drives Northstar-CV from the TR ARCTIC 2026 simulator.

The harness plays the parts of the robot that Northstar-CV talks to:

- ``nssim.protocol``: the DJI-framed UART protocol shared with the MCB firmware.
- ``nssim.mcb``: a virtual MCB that speaks that protocol (odometry, robot ID, aim input).
- ``nssim.camera``: a Triton2 twin that streams Bayer frames to Northstar-CV's SimCamera.
"""

__version__ = "0.1.0"
