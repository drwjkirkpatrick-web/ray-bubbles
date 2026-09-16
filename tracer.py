"""
CPU path tracer for ray-bubbles.

The scene is intentionally simple: one or more spherical soap bubbles floating
above a lit ground plane, plus a large sky dome. We trace camera rays, intersect
with the nearest bubble shell, sample the thin-film reflectance, and use a
single-bounce shadow/ground bounce to fake the soft caustic look that soap
bubbles have.
"""
import numpy as np
from spectrum import WAVELENGTHS, spectrum_to_rgb, gamma_encode
from bubbles import Bubble, thin_film_reflection, refract


class Camera:
    """Pinhole camera with a thin-lens depth-of-field option."""

    def __init__(self, origin: np.ndarray, look_at: np.ndarray, fov_deg: float = 45.0, focus_dist: float = None, aperture: float = 0.0):
        self.origin = np.asarray(origin, dtype=np.float32)
        forward = np.asarray(look_at, dtype=np.float32) - self.origin
        self.forward = forward / np.linalg.norm(forward)
        self.right = np.cross(self.forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        if np.linalg.norm(self.right) < 1e-6:
            self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.fov = float(fov_deg)
        self.aperture = float(aperture)
        self.focus_dist = focus_dist if focus_dist is not None else np.linalg.norm(forward)

    def rays(self, width: int, height: int, samples: int = 1, rng: np.random.Generator = None) -> tuple:
        """
        Generate primary rays for the image plane.

        Returns:
            origins: (N, 3) ray origins
            directions: (N, 3) normalised ray directions
            shape: (height, width) to reshape pixel arrays later
        """
        if rng is None:
            rng = np.random.default_rng()
        aspect = width / height
        tan_fov = np.tan(np.radians(self.fov * 0.5))
        # Pixel centres in [-1, 1] with y up.
        xs = np.linspace(-1.0, 1.0, width)
        ys = np.linspace(1.0, -1.0, height)
        px, py = np.meshgrid(xs, ys)
        px = px.astype(np.float32)
        py = py.astype(np.float32)

        if samples > 1:
            # Jitter within each pixel.
            jitter_x = rng.random((height, width), dtype=np.float32) - 0.5
            jitter_y = rng.random((height, width), dtype=np.float32) - 0.5
            dx = (2.0 / width) * jitter_x
            dy = (2.0 / height) * jitter_y
            px = px + dx
            py = py + dy

        px *= aspect * tan_fov
        py *= tan_fov

        # Direction through the pinhole.
        dirs = (self.forward[None, None, :] +
                self.right[None, None, :] * px[:, :, None] +
                self.up[None, None, :] * py[:, :, None])
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        if self.aperture > 0:
            # Thin-lens depth of field: sample a disk on the lens.
            theta = rng.random((height, width), dtype=np.float32) * 2.0 * np.pi
            r = np.sqrt(rng.random((height, width), dtype=np.float32)) * self.aperture
            lens_offset = self.right[None, None, :] * (r * np.cos(theta))[:, :, None] + \
                          self.up[None, None, :] * (r * np.sin(theta))[:, :, None]
            origins = self.origin + lens_offset.reshape(-1, 3)
            focal_points = origins + dirs.reshape(-1, 3) * self.focus_dist
            dirs = focal_points - origins
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        else:
            origins = np.broadcast_to(self.origin, (height * width, 3)).copy()
            dirs = dirs.reshape(-1, 3)

        return origins, dirs, (height, width)


class Scene:
    """Scene holds geometry and lighting."""

    def __init__(self, bubbles, sun_dir: np.ndarray = None, ground_y: float = -2.0):
        self.bubbles = bubbles if isinstance(bubbles, list) else [bubbles]
        if sun_dir is None:
            sun_dir = np.array([0.3, 0.8, -0.5], dtype=np.float32)
        self.sun_dir = sun_dir / np.linalg.norm(sun_dir)
        self.ground_y = float(ground_y)
        # Sky radiance: deep blue zenith, lighter blue/white horizon.
        self.sky = np.array([0.25, 0.40, 0.65, 0.90, 1.05, 1.10, 1.05, 1.00], dtype=np.float32)
        # Sun radiance: bright warm-ish spectrum.
        self.sun = np.array([2.0, 2.1, 2.3, 2.5, 2.6, 2.5, 2.3, 2.2], dtype=np.float32)

    def closest_bubble_intersect(self, origins: np.ndarray, directions: np.ndarray) -> tuple:
        """Intersect all bubbles, return nearest hit distance and bubble index."""
        t_all = np.full((len(self.bubbles), origins.shape[0]), np.inf, dtype=np.float32)
        for i, bubble in enumerate(self.bubbles):
            t_all[i] = bubble.intersect(origins, directions)
        hit_idx = np.argmin(t_all, axis=0)
        t_min = np.min(t_all, axis=0)
        t_min = np.where(np.isfinite(t_min), t_min, np.inf)
        return t_min, hit_idx

    def sky_radiance(self, direction: np.ndarray) -> np.ndarray:
        """Return spectral radiance of the sky in a given direction."""
        cos_sun = np.maximum(np.sum(direction * self.sun_dir, axis=-1), 0.0)
        # Soft sun disk.
        sun_contrib = self.sun * (cos_sun ** 256.0)[:, None]
        # Horizon gradient: low y (downward) gets horizon colour; high y gets zenith.
        t = np.clip(0.5 + 0.5 * direction[:, 1], 0.0, 1.0)
        zenith = np.array([0.05, 0.10, 0.25, 0.40, 0.45, 0.42, 0.38, 0.35], dtype=np.float32)
        horizon = np.array([0.55, 0.65, 0.85, 1.00, 1.05, 1.00, 0.95, 0.90], dtype=np.float32)
        sky_col = zenith + (horizon - zenith) * t[:, None]
        return sky_col + sun_contrib * 0.3

    def ground_intersect(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Distance to the y=ground_y plane, only for downward rays."""
        dy = directions[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_dy = np.where(np.abs(dy) > 1e-6, dy, np.inf)
            t = np.where(
                dy < -1e-6,
                (self.ground_y - origins[:, 1]) / safe_dy,
                np.inf,
            )
        return np.where(t > 0, t, np.inf)

    def bubble_shadow(self, points: np.ndarray, to_sun: np.ndarray) -> np.ndarray:
        """Return True where the point is in sunlight (no bubble occludes it)."""
        in_light = np.ones(points.shape[0], dtype=bool)
        for bubble in self.bubbles:
            t = bubble.intersect(points + 0.001 * to_sun, np.broadcast_to(to_sun, (points.shape[0], 3)))
            in_light = in_light & np.isinf(t)
        return in_light


def trace_once(scene: Scene, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """
    Trace one sample for every pixel and return spectral radiance (N, 8).

    We support two paths:
      1. Camera ray hits the nearest bubble shell -> thin-film reflection colour.
      2. Camera ray misses all bubbles -> sky / ground.
    """
    n = origins.shape[0]
    radiance = np.zeros((n, WAVELENGTHS.size), dtype=np.float32)

    # Bubble intersection: nearest hit per ray.
    t_bubble, hit_idx = scene.closest_bubble_intersect(origins, directions)

    # Split rays by hit/miss.
    hit_mask = np.isfinite(t_bubble)
    miss_mask = ~hit_mask

    # ---- Bubble hits ----
    if np.any(hit_mask):
        hit_o = origins[hit_mask]
        hit_d = directions[hit_mask]
        hit_t = t_bubble[hit_mask]
        points = hit_o + hit_d * hit_t[:, None]
        bubbles_hit = [scene.bubbles[i] for i in hit_idx[hit_mask]]
        # Stack normals/thickness by re-intersecting each bubble (cheap for spheres).
        normals = np.zeros_like(points)
        thickness = np.zeros(points.shape[0], dtype=np.float32)
        for local_i, (world_i, bubble) in enumerate(zip(np.where(hit_mask)[0], bubbles_hit)):
            normals[local_i] = bubble.normal(points[local_i:local_i + 1])[0]
            thickness[local_i] = bubble.film_thickness_at(points[local_i:local_i + 1])[0]

        # View direction is opposite to ray direction.
        view = -hit_d
        cos_theta = np.sum(view * normals, axis=-1)
        reflection = thin_film_reflection(cos_theta, thickness)

        # Reflected sky/sky dome colour.
        reflected_dir = hit_d - 2.0 * np.sum(hit_d * normals, axis=-1)[:, None] * normals
        sky_ref = scene.sky_radiance(reflected_dir)

        # Add a small refracted-through-shell caustic term.
        T = 1.0 - reflection
        refr_dir_in = refract(hit_d, normals, 1.0 / 1.33)
        refr_dir_out = refract(refr_dir_in, -normals, 1.33)
        valid_transmission = ~np.all(refr_dir_out == 0, axis=-1)
        sky_trans = np.zeros_like(sky_ref)
        if np.any(valid_transmission):
            sky_trans[valid_transmission] = scene.sky_radiance(refr_dir_out[valid_transmission])

        radiance[hit_mask] = reflection * sky_ref + 0.15 * T * sky_trans

    # ---- Misses: sky + ground ----
    if np.any(miss_mask):
        miss_d = directions[miss_mask]
        sky = scene.sky_radiance(miss_d)
        # Ground bounce approximation.
        t_g = scene.ground_intersect(origins[miss_mask], miss_d)
        ground_hit = np.isfinite(t_g)
        if np.any(ground_hit):
            g_o = origins[miss_mask][ground_hit]
            g_d = miss_d[ground_hit]
            g_t = t_g[ground_hit]
            g_points = g_o + g_d * g_t[:, None]
            to_sun = scene.sun_dir
            in_light = scene.bubble_shadow(g_points, to_sun)
            sun_dot = np.maximum(np.sum(np.array([0.0, 1.0, 0.0], dtype=np.float32) * to_sun), 0.0)
            ground_colour = 0.04 * sun_dot * scene.sun

            # Specular highlight from bubble caustics: reflection of the nearest bubble.
            # Find distance to nearest bubble centre.
            centres = np.stack([b.centre for b in scene.bubbles])
            diffs = g_points[:, None, :] - centres[None, :, :]
            nearest_i = np.argmin(np.linalg.norm(diffs, axis=-1), axis=-1)
            nearest_centres = centres[nearest_i]
            bubble_vec = nearest_centres - g_points
            bubble_vec = bubble_vec / np.linalg.norm(bubble_vec, axis=-1, keepdims=True)
            caustic = np.maximum(np.sum(bubble_vec * to_sun[None, :], axis=-1), 0.0) ** 8.0
            ground_colour = ground_colour + in_light[:, None] * caustic[:, None] * scene.sun * 0.20

            # Darkening falloff away from caustic for visual interest.
            dist_from_centre = np.linalg.norm(g_points[:, [0, 2]] - nearest_centres[:, [0, 2]], axis=-1)
            ground_colour = ground_colour * (0.6 + 0.4 * np.exp(-dist_from_centre / 2.0))[:, None]

            # Subtle checker pattern.
            checker = ((np.floor(g_points[:, 0] * 2.0).astype(int) + np.floor(g_points[:, 2] * 2.0).astype(int)) % 2).astype(np.float32)
            ground_colour = ground_colour * (0.85 + 0.15 * checker)[:, None]

            miss_radiance = sky
            miss_radiance[ground_hit] = ground_colour
        else:
            miss_radiance = sky
        radiance[miss_mask] = miss_radiance

    return radiance


def render(scene: Scene, camera: Camera, width: int, height: int, samples: int = 64, seed: int = 0) -> np.ndarray:
    """
    Render the scene to an sRGB-encoded float32 image (H, W, 3).
    """
    rng = np.random.default_rng(seed)
    accum = np.zeros((height * width, WAVELENGTHS.size), dtype=np.float32)

    for s in range(samples):
        origins, directions, _ = camera.rays(width, height, samples=1 if s == 0 else 2, rng=rng)
        accum += trace_once(scene, origins, directions)

    # Average samples.
    spectral = accum / samples
    rgb_linear = spectrum_to_rgb(spectral)
    rgb = gamma_encode(rgb_linear)
    return rgb.reshape(height, width, 3)


def save_image(img: np.ndarray, path: str):
    """Save an (H, W, 3) float [0,1] image as a PNG."""
    from PIL import Image
    uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(uint8).save(path)
    print(f"saved {path} ({img.shape[1]}x{img.shape[0]})")
