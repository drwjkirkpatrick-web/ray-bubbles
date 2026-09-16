"""
Spectral helpers for ray-bubbles.

We trace 8 discrete wavelengths from roughly 380 nm to 720 nm, then convert
the resulting spectral radiance to sRGB using a simple CIE 1931 2-deg
approximation. This is far smaller (and faster) than a full spectral render,
but enough to capture thin-film interference rainbows accurately.
"""
import numpy as np

# 8 wavelength bins in nanometres, spanning visible light.
WAVELENGTHS = np.array([380, 450, 500, 550, 600, 650, 700, 720], dtype=np.float32)

# Crude CIE 1931 2-degree RGB colour-matching functions sampled at those bins.
# These are normalised so that an equal-energy spectrum maps roughly to white.
CIE_X = np.array([0.001_4, 0.336_2, 0.004_9, 0.33_20, 1.06_20, 0.431_56, 0.004_88, 0.000_17], dtype=np.float32)
CIE_Y = np.array([0.000_0, 0.038_0, 0.32_30, 0.995_0, 0.631_0, 0.107_0, 0.004_10, 0.000_06], dtype=np.float32)
CIE_Z = np.array([0.006_5, 1.772_1, 0.27_20, 0.045_5, 0.000_80, 0.000_01, 0.000_00, 0.000_00], dtype=np.float32)

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
