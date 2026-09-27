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


def vignette(rgb: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Apply a radial vignette falloff.
    strength=0 means no vignette; 1.0 is strong darkening at corners.
    """
    if strength <= 0.0:
        return rgb
    h, w = rgb.shape[:2]
    y = np.linspace(-1.0, 1.0, h)
    x = np.linspace(-1.0, 1.0, w)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv * xv + yv * yv)
    r = np.clip(r, 0.0, 1.0)
    factor = 1.0 - strength * (r ** 2.0)
    return np.clip(rgb * factor[:, :, None], 0.0, 1.0)


def apply_vignette(rgb: np.ndarray, width: int, height: int, strength: float = 0.0) -> np.ndarray:
    """
    Flat-array wrapper for the radial vignette falloff.
    rgb: (H*W, 3) linear RGB.  Returns same shape.
    """
    if strength <= 0.0:
        return rgb
    img = rgb.reshape(height, width, 3)
    img = vignette(img, strength)
    return img.reshape(-1, 3)


def apply_bloom(rgb: np.ndarray, width: int, height: int, strength: float = 0.0, threshold: float = 0.8) -> np.ndarray:
    """Flat-array wrapper for simple box-blur bloom."""
    if strength <= 0.0:
        return rgb
    img = rgb.reshape(height, width, 3)
    bright = np.where(img > threshold, img - threshold, 0.0)
    from scipy.ndimage import uniform_filter
    blurred = uniform_filter(bright, size=5, mode="constant")
    img = np.clip(img + strength * blurred, 0.0, 1.0)
    return img.reshape(-1, 3)


def apply_grain(rgb: np.ndarray, width: int, height: int, strength: float = 0.0, rng=None) -> np.ndarray:
    """Flat-array wrapper for film grain."""
    if strength <= 0.0:
        return rgb
    img = rgb.reshape(height, width, 3)
    noise = rng.normal(0.0, strength / 255.0, img.shape) if rng is not None else np.random.normal(0.0, strength / 255.0, img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return img.reshape(-1, 3)


def apply_chromatic_aberration(rgb: np.ndarray, width: int, height: int, strength: float = 0.0) -> np.ndarray:
    """
    Radial RGB channel separation (red pushed outward, blue inward).
    rgb: (H*W, 3) linear RGB.  Returns same shape.
    """
    if strength == 0.0 or width <= 1 or height <= 1:
        return rgb
    img = rgb.reshape(height, width, 3)
    y = np.linspace(-1.0, 1.0, height)
    x = np.linspace(-1.0, 1.0, width)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv * xv + yv * yv) + 1e-6
    dx = xv / r
    dy = yv / r
    r_norm = np.clip(r, 0.0, 1.0)

    def sample_shifted(channel: np.ndarray, offset: np.ndarray) -> np.ndarray:
        if np.all(offset == 0.0):
            return channel
        yy = yv - dy * offset
        xx = xv - dx * offset
        # Map normalized coords to pixel indices.
        rowf = (1.0 - yy) * 0.5 * (height - 1)
        colf = (xx + 1.0) * 0.5 * (width - 1)
        row0 = np.floor(rowf).astype(int)
        col0 = np.floor(colf).astype(int)
        row1 = np.clip(row0 + 1, 0, height - 1)
        col1 = np.clip(col0 + 1, 0, width - 1)
        row0 = np.clip(row0, 0, height - 1)
        col0 = np.clip(col0, 0, width - 1)
        dr = rowf - row0
        dc = colf - col0
        a = channel[row0, col0] * (1 - dc) + channel[row0, col1] * dc
        b = channel[row1, col0] * (1 - dc) + channel[row1, col1] * dc
        return a * (1 - dr) + b * dr

    out = np.stack([
        sample_shifted(img[:, :, 0], -strength * r_norm),
        sample_shifted(img[:, :, 1], np.zeros_like(r_norm)),
        sample_shifted(img[:, :, 2], strength * r_norm),
    ], axis=-1)
    return np.clip(out, 0.0, 1.0).reshape(-1, 3)


def bloom(rgb: np.ndarray, strength: float = 0.0, threshold: float = 0.8) -> np.ndarray:
    """
    Simple box-blur bloom. Extracts bright pixels, blurs them, and adds back.
    """
    if strength <= 0.0:
        return rgb
    bright = np.where(rgb > threshold, rgb - threshold, 0.0)
    from scipy.ndimage import uniform_filter
    blurred = uniform_filter(bright, size=5, mode="constant")
    return np.clip(rgb + strength * blurred, 0.0, 1.0)


def grain(rgb: np.ndarray, strength: float = 0.0, seed: int = 0) -> np.ndarray:
    """
    Add film-like grain using per-pixel Gaussian noise.
    """
    if strength <= 0.0:
        return rgb
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, strength / 255.0, rgb.shape)
    return np.clip(rgb + noise, 0.0, 1.0)


def post_process(
    rgb: np.ndarray,
    tone_map: str = "linear",
    saturation: float = 1.0,
    vignette_strength: float = 0.0,
    bloom_strength: float = 0.0,
    grain_strength: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Full CPU post-processing pipeline on linear RGB.
    Order: tone map -> saturation -> vignette -> bloom -> grain -> clamp.
    """
    rgb = apply_tone_map(rgb, mode=tone_map)
    if saturation != 1.0:
        grey = np.mean(rgb, axis=-1, keepdims=True)
        rgb = np.clip(grey + (rgb - grey) * saturation, 0.0, 1.0)
    rgb = vignette(rgb, vignette_strength)
    rgb = bloom(rgb, bloom_strength)
    rgb = grain(rgb, grain_strength, seed=seed)
    return np.clip(rgb, 0.0, 1.0)
