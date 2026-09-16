"""
Unit tests for ray-bubbles.
"""
import numpy as np
import pytest
from bubbles import Bubble, thin_film_reflection, refract
from spectrum import spectrum_to_rgb, gamma_encode, N_WAVELENGTHS
from tracer import Camera, Scene, render


def test_bubble_intersect_hit():
    b = Bubble(np.array([0.0, 0.0, 0.0]), 1.0, 500.0, 0.0)
    origins = np.array([[0.0, 0.0, -3.0]])
    dirs = np.array([[0.0, 0.0, 1.0]])
    t = b.intersect(origins, dirs)
    assert t[0] > 1.0 and t[0] < 4.0


def test_bubble_intersect_miss():
    b = Bubble(np.array([0.0, 0.0, 0.0]), 1.0, 500.0, 0.0)
    origins = np.array([[0.0, 3.0, 0.0]])
    dirs = np.array([[0.0, 0.0, 1.0]])
    t = b.intersect(origins, dirs)
    assert np.isinf(t[0])


def test_thin_film_shape():
    cos = np.array([0.5, 0.9])
    thick = np.array([400.0, 500.0])
    refl = thin_film_reflection(cos, thick)
    assert refl.shape == (2, N_WAVELENGTHS)
    assert np.all((refl >= 0) & (refl <= 1))


def test_spectrum_to_rgb_white():
    spec = np.ones(N_WAVELENGTHS, dtype=np.float32)
    rgb = spectrum_to_rgb(spec)
    assert rgb.shape == (3,)
    # White-equivalent spectrum should be reasonably balanced.
    assert np.all(rgb > 0.1)


def test_camera_rays_shape():
    cam = Camera(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 0.0]), fov_deg=45.0)
    origins, dirs, shape = cam.rays(10, 10, samples=1)
    assert origins.shape == (100, 3)
    assert dirs.shape == (100, 3)
    assert shape == (10, 10)
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0)


def test_render_small():
    b = Bubble(np.array([0.0, 0.0, 0.0]), 1.0, 450.0, 20.0)
    cam = Camera(np.array([0.0, 0.0, 4.0]), np.array([0.0, 0.0, 0.0]))
    scene = Scene([b])
    img = render(scene, cam, 16, 16, samples=2, seed=1)
    assert img.shape == (16, 16, 3)
    assert np.all((img >= 0) & (img <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
