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
from functools import lru_cache

import cv2
import numpy as np

_LINEAR_LEVELS = 65536  # resolution of the linear -> 8-bit table (fine enough near black)


@dataclass
class Optics:
    psf_sigma_px: float = 0.7  # Gaussian lens blur; 0 disables
    glare_threshold: int = 250  # 8-bit level (any channel) at which a pixel counts as a light source
    glare_sigma_px: float = 2.0
    glare_strength: float = 0.35  # fraction of a light source's (linear) light added as halo; 0 disables
    gamma: float = 2.2  # the renderer writes sRGB-like values; the halo is added in linear light
    # Sensor noise on the raw Bayer image, in 8-bit levels: read noise (sd) and shot noise (variance
    # per level of signal). 0 renders a still scene identically every frame, which no camera does.
    read_noise_dn: float = 0.0
    shot_noise_gain: float = 0.0


def apply_optics(rgb: np.ndarray, optics: Optics) -> np.ndarray:
    """HxWx3 uint8 render -> the image the sensor sees (a new array; the input is not modified)."""
    if optics.psf_sigma_px > 0:
        # 3x3 holds ~97% of a sigma-0.7 kernel's weight at half the cost of OpenCV's default 5x5.
        ksize = 3 if optics.psf_sigma_px <= 0.8 else 0
        out = cv2.GaussianBlur(rgb, (ksize, ksize), optics.psf_sigma_px)
    else:
        out = np.array(rgb, copy=True)
    if optics.glare_strength > 0:
        _add_glare(out, optics)
    return out


def add_sensor_noise(raw: np.ndarray, optics: Optics, seed: int) -> np.ndarray:
    """Raw (Bayer) uint8 image with read and shot noise added (seeded, so a run is repeatable)."""
    if optics.read_noise_dn <= 0 and optics.shot_noise_gain <= 0:
        return raw
    rng = np.random.default_rng(seed)
    signal = raw.astype(np.float32)
    sd = np.sqrt(optics.read_noise_dn ** 2 + optics.shot_noise_gain * signal)
    noisy = signal + sd * rng.standard_normal(raw.shape, dtype=np.float32)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


@lru_cache(maxsize=4)
def _tables(gamma: float) -> tuple[np.ndarray, np.ndarray]:
    to_linear = ((np.arange(256) / 255.0) ** gamma).astype(np.float32)
    to_8bit = np.round(255.0 * np.linspace(0.0, 1.0, _LINEAR_LEVELS) ** (1.0 / gamma)).astype(np.uint8)
    return to_linear, to_8bit


def _add_glare(img: np.ndarray, optics: Optics) -> None:
    t = optics.glare_threshold
    sources = cv2.bitwise_not(cv2.inRange(img, (0, 0, 0), (t - 1, t - 1, t - 1)))
    x, y, w, h = cv2.boundingRect(sources)
    if w == 0 or h == 0:
        return
    # Only the neighbourhood of the light sources changes, so only that is worked on.
    pad = math.ceil(3 * optics.glare_sigma_px)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    roi = img[y0:y1, x0:x1]
    to_linear, to_8bit = _tables(optics.gamma)
    linear = to_linear[roi]
    lit = linear * (sources[y0:y1, x0:x1, None] != 0)
    linear += optics.glare_strength * cv2.GaussianBlur(lit, (0, 0), optics.glare_sigma_px)
    np.minimum(linear, 1.0, out=linear)
    index = (linear * (_LINEAR_LEVELS - 1) + 0.5).astype(np.uint16)
    # Glare only adds light; the max keeps table rounding from darkening untouched pixels.
    np.maximum(roi, to_8bit[index], out=roi)
