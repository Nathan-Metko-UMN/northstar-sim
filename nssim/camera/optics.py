"""Lens effects between the renderer and the Bayer mosaic.

SAPIEN draws a pinhole image: edges are perfectly sharp and an emissive light bar is a crisp box
of color. A real lens spreads every point over about a pixel (diffraction and aberrations), and the
LED bars are so much brighter than everything else that some of their light scatters into a halo a
few pixels wide (glare). Both matter to the detector. Without them a bar a few pixels wide can have
all of its blue in a single column of the Bayer's blue sites, and after debayering it is a
one-pixel line that findLights() throws away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Optics:
    psf_sigma_px: float = 0.7  # Gaussian lens blur; 0 disables
    glare_threshold: int = 250  # 8-bit level (any channel) at which a pixel counts as a light source
    glare_sigma_px: float = 2.0
    glare_strength: float = 0.35  # fraction of a light source's (linear) light added as halo; 0 disables
    gamma: float = 2.2  # the renderer writes sRGB-like values; the halo is added in linear light


def apply_optics(rgb: np.ndarray, optics: Optics) -> np.ndarray:
    """HxWx3 uint8 render -> the image the sensor sees (new contiguous array)."""
    out = np.ascontiguousarray(rgb)
    if out is rgb:
        out = rgb.copy()
    if optics.glare_strength > 0:
        _add_glare(out, optics)
    if optics.psf_sigma_px > 0:
        out = cv2.GaussianBlur(out, (0, 0), optics.psf_sigma_px)
    return out


def _add_glare(img: np.ndarray, optics: Optics) -> None:
    t = optics.glare_threshold
    dark = cv2.inRange(img, (0, 0, 0), (t - 1, t - 1, t - 1))
    sources = cv2.bitwise_not(dark)
    x, y, w, h = cv2.boundingRect(sources)
    if w == 0 or h == 0:
        return
    # Only the neighbourhood of the light sources changes, so only that is worked on.
    pad = math.ceil(3 * optics.glare_sigma_px)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    roi = img[y0:y1, x0:x1]
    linear = (roi.astype(np.float32) / 255.0) ** optics.gamma
    lit = linear * (sources[y0:y1, x0:x1, None] > 0)
    halo = cv2.GaussianBlur(lit, (0, 0), optics.glare_sigma_px)
    linear = np.minimum(linear + optics.glare_strength * halo, 1.0)
    roi[...] = np.round(255.0 * linear ** (1.0 / optics.gamma)).astype(np.uint8)
