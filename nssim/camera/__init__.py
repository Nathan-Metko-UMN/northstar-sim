"""Triton2 twin: renders are mosaiced to BayerRG8 and streamed to Northstar-CV's SimCamera."""

from .bayer import RGGB_TO_BGR, demosaic_to_bgr, mosaic_rggb
from .model import CameraModel, LinkModel
from .optics import Optics, apply_optics
from .stream import FrameClient, FrameHeader, FrameServer

__all__ = [
    "RGGB_TO_BGR",
    "CameraModel",
    "FrameClient",
    "FrameHeader",
    "FrameServer",
    "LinkModel",
    "Optics",
    "apply_optics",
    "demosaic_to_bgr",
    "mosaic_rggb",
]
