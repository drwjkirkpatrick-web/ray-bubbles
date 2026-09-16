"""
Bubble geometry and thin-film interference material.
"""
import numpy as np
from spectrum import WAVELENGTHS


class Bubble:
    """A spherical soap bubble with a thin water film."""

    def __init__(self, centre: np.ndarray, radius: float, base_thickness_nm: float, thickness_variation_nm: float):
        self.centre = np.asarray(centre, dtype=np.float32)
        self.radius = float(radius)
        self.base_thickness = float(base_thickness_nm)
        self.thickness_variation = float(thickness_variation_nm)

    def intersect(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """
        Ray-sphere intersection for many rays.

        Args:
            origins:    (N, 3) ray origins
            directions: (N, 3) normalised ray directions

        Returns:
            (N,) array of hit distances; np.inf where there is no hit.
            Returns the *first* positive intersection along each ray.
        """
        oc = origins - self.centre
        b = np.sum(oc * directions, axis=-1)
        c = np.sum(oc * oc, axis=-1) - self.radius * self.radius
        disc = b * b - c
        hit = disc >= 0
        sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
        # The quadratic is t^2 + 2bt + c = 0, so roots are -b ± sqrt_disc.
        t0 = -b - sqrt_disc
        t1 = -b + sqrt_disc
        t = np.where((t0 > 1e-4) & hit, t0, np.where((t1 > 1e-4) & hit, t1, np.inf))
        return np.where(hit, t, np.inf)

    def normal(self, points: np.ndarray) -> np.ndarray:
        """Outward unit normal at surface points."""
        n = points - self.centre
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def film_thickness_at(self, points: np.ndarray) -> np.ndarray:
        """
        Return film thickness in nm at each point.

        We add a smooth swirl by perturbing the base thickness with a low-frequency
        3D sine pattern, which produces the rainbow-like bands seen on real bubbles.
        """
        p = points * 0.5
        swirl = (
            np.sin(p[:, 0] + 2.0 * p[:, 1]) +
            np.sin(1.7 * p[:, 2] + 0.3 * p[:, 0]) +
            np.sin(0.9 * p[:, 1] - 1.1 * p[:, 2])
        ) / 3.0
        return self.base_thickness + self.thickness_variation * swirl


def thin_film_reflection(cos_theta: np.ndarray, thickness_nm: np.ndarray, n_film: float = 1.33) -> np.ndarray:
    """
    Approximate reflection coefficient for a thin water film in air.

    Uses the standard two-beam interference formula for a thin film,
    evaluated for each wavelength bin in WAVELENGTHS. The result is a
    (N, N_WAVELENGTHS) array of reflection intensities in [0, 1].

    Args:
        cos_theta: cosine of the angle between ray and surface normal (N,)
        thickness_nm: film thickness in nanometres at each hit point (N,)
    """
    # Avoid edge cases: clamp cos_theta slightly away from zero.
    cos_theta = np.clip(np.abs(cos_theta), 0.01, 1.0)
    # Refracted angle inside film via Snell's law.
    sin_theta_t2 = (1.0 - cos_theta * cos_theta) / (n_film * n_film)
    cos_theta_t = np.sqrt(np.clip(1.0 - sin_theta_t2, 0.0, 1.0))

    # Fresnel amplitude reflectivity at air/water boundary, s-polarised average.
    r = (cos_theta - n_film * cos_theta_t) / (cos_theta + n_film * cos_theta_t)
    R = r * r  # intensity reflectivity of one interface (N,)

    # Optical path difference for two passes through the film.
    # wavelengths is (8,); thickness is (N, 1) so broadcasting gives (N, 8).
    phase = (2.0 * np.pi * 2.0 * n_film * thickness_nm[:, None] * cos_theta_t[:, None]) / WAVELENGTHS[None, :]

    # Interference of the two reflected beams.  R has shape (N,), so broadcast to (N, 8).
    R8 = R[:, None]
    reflectance = 2.0 * R8 * (1.0 - np.cos(phase)) / (1.0 + 2.0 * R8 * (1.0 - np.cos(phase)) + 1e-10)
    return np.clip(reflectance, 0.0, 1.0)


def refract(direction: np.ndarray, normal: np.ndarray, eta: float) -> np.ndarray:
    """Snell's-law refraction; returns None when total internal reflection occurs."""
    cos_i = np.clip(np.sum(direction * normal, axis=-1), -1.0, 1.0)
    n = normal * (-1.0 if eta < 1.0 else 1.0)
    cos_i = np.sum(direction * n, axis=-1)
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    tir = sin_t2 > 1.0
    t = np.where(
        tir[:, None],
        np.zeros_like(direction),
        eta * direction - (eta * cos_i[:, None] + np.sqrt(np.clip(1.0 - sin_t2, 0.0, 1.0))[:, None]) * n
    )
    t_norm = np.linalg.norm(t, axis=-1, keepdims=True)
    t_norm = np.where(t_norm == 0, 1, t_norm)
    return t / t_norm
