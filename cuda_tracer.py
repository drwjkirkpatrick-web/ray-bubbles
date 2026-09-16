"""
Placeholder CUDA kernel wrapper for ray-bubbles.

When numba and cupy are installed, --gpu will call this module to render on
the Jetson GPU. Until then the CPU path is fully functional.
"""

# try:
#     import cupy as cp
#     import numba.cuda as cuda
#     _GPU_AVAILABLE = True
# except Exception:
#     _GPU_AVAILABLE = False

_GPU_AVAILABLE = False


def gpu_available() -> bool:
    return _GPU_AVAILABLE


def render_gpu(scene, camera, width: int, height: int, samples: int = 64, seed: int = 42):
    """Raise a clear error when GPU dependencies are missing."""
    raise RuntimeError(
        "GPU path is not installed. Install numba and cupy, then re-run with --gpu."
    )
