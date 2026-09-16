"""
Spectral helpers for ray-bubbles.

We trace 16 discrete wavelengths from 380 nm to 700 nm, then convert the
resulting spectral radiance to sRGB using CIE 1931 2-degree colour-matching
functions. Sixteen bins give much smoother thin-film iridescence than the
previous 8-bin setup while still running fast on the Jetson.
"""
import numpy as np

N_WAVELENGTHS = 16

# 16 wavelength bins in nanometres, spanning visible light.
WAVELENGTHS = np.array(
    [380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 700],
    dtype=np.float32,
)

# CIE 1931 2-degree colour-matching functions sampled at the bins above.
# Values are drawn from the standard CIE 1931 observer tables.
CIE_X = np.array(
    [0.0014, 0.0143, 0.1344, 0.3483, 0.2908, 0.0956, 0.0049,
     0.0633, 0.2904, 0.5945, 0.9163, 1.0622, 0.8544, 0.4479, 0.1649, 0.0114],
    dtype=np.float32,
)
CIE_Y = np.array(
    [0.0000, 0.0004, 0.0040, 0.0230, 0.0600, 0.1390, 0.3230,
     0.7100, 0.9540, 0.9950, 0.8700, 0.6310, 0.3810, 0.1750, 0.0610, 0.0041],
    dtype=np.float32,
)
CIE_Z = np.array(
    [0.0065, 0.0679, 0.6456, 1.7471, 1.6692, 0.8130, 0.2720,
     0.0782, 0.0203, 0.0039, 0.0017, 0.0008, 0.0002, 0.0000, 0.0000, 0.0000],
    dtype=np.float32,
)

# sRGB conversion matrix (XYZ -> linear RGB).
XYZ_TO_RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float32)


def spectrum_to_rgb(spectral_values: np.ndarray) -> np.ndarray:
    """
    Convert an array of spectral samples (one per wavelength bin) to linear sRGB.

    Args:
        spectral_values: shape (... , N_WAVELENGTHS) float32

    Returns:
        RGB array of shape (...) float32, linear (not gamma encoded).
    """
    # Inner product over the wavelength dimension.
    X = np.sum(spectral_values * CIE_X, axis=-1)
    Y = np.sum(spectral_values * CIE_Y, axis=-1)
    Z = np.sum(spectral_values * CIE_Z, axis=-1)
    xyz = np.stack([X, Y, Z], axis=-1)
    rgb = xyz @ XYZ_TO_RGB.T
    return rgb


def gamma_encode(rgb: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma encoding to linear RGB values."""
    # Clip tiny negatives from numerical noise before gamma.
    rgb = np.where(rgb > 0.0031308, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055, 12.92 * rgb)
    return np.clip(rgb, 0.0, 1.0)


def white_spectrum() -> np.ndarray:
    """Return a flat spectrum used for the sky dome."""
    return np.ones_like(WAVELENGTHS)


def aces_filmic(x: np.ndarray) -> np.ndarray:
    """
    ACES-inspired filmic tone-mapping curve (Steve Morrow's fit).
    Works on linear RGB images.
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def reinhard(x: np.ndarray) -> np.ndarray:
    """Simple Reinhard tone-mapping curve."""
    return x / (1.0 + x)


def apply_tone_map(rgb: np.ndarray, mode: str = "linear") -> np.ndarray:
    """
    Apply a tone-mapping curve to linear RGB.

    Modes:
      - "linear": clamp only (default, backward compatible).
      - "aces": ACES filmic curve.
      - "reinhard": Reinhard global operator.
    """
    if mode == "aces":
        return aces_filmic(rgb)
    if mode == "reinhard":
        return reinhard(rgb)
    return np.clip(rgb, 0.0, 1.0)
