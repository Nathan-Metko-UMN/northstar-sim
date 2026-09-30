"""Virtual MCB: emulates the robot's control board on the vision UART link."""

from .virtual_mcb import AimState, McbConfig, TurretState, VirtualMcb, wrap_2pi

__all__ = ["AimState", "McbConfig", "TurretState", "VirtualMcb", "wrap_2pi"]
