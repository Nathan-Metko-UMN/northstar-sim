"""Bayer mosaicing: the Triton2 streams BayerRG8 (RGGB) and the Jetson debayers."""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV names Bayer patterns by the second row's second/third pixels, so an RGGB sensor
# (what LUCID calls BayerRG8) is COLOR_BayerBG2BGR. tests/test_camera.py pins this down.
RGGB_TO_BGR = cv2.COLOR_BayerBG2BGR


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 RGB (uint8) -> HxW RGGB Bayer (uint8). H and W must be even."""
    h, w = rgb.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"Bayer images need even dimensions, got {w}x{h}")
    raw = np.empty((h, w), dtype=np.uint8)
    raw[0::2, 0::2] = rgb[0::2, 0::2, 0]  # R
    raw[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G
    raw[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G
    raw[1::2, 1::2] = rgb[1::2, 1::2, 2]  # B
    return raw


def demosaic_to_bgr(raw: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(raw, RGGB_TO_BGR)
