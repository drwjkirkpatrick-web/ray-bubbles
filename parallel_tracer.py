"""
CPU-parallel rendering for ray-bubbles.

Renders the same scene as tracer.render() but splits samples across multiple
processes. Each worker produces an independent image with its share of samples;
the host averages them. This uses all CPU cores and avoids the need for a GPU.
"""
import numpy as np
from tracer import render, Camera, Scene


def _worker_render(args):
    """
    Worker that renders one partial image with a fixed sample count and seed.

    Args is a tuple:
        (scene_bubbles, camera_dict, width, height, samples, seed)
    We reconstruct Scene and Camera from plain data so the tuple is picklable.
    """
    from bubbles import Bubble

    bubbles_data, cam_dict, width, height, samples, seed = args
    bubbles = []
    for bd in bubbles_data:
        bubbles.append(Bubble(
            centre=np.array(bd["centre"], dtype=np.float32),
            radius=bd["radius"],
            base_thickness_nm=bd["base"],
            thickness_variation_nm=bd["var"],
        ))
    scene = Scene(bubbles=bubbles)
    camera = Camera(
        origin=np.array(cam_dict["origin"], dtype=np.float32),
        look_at=np.array(cam_dict["look_at"], dtype=np.float32),
        fov_deg=cam_dict["fov"],
        focus_dist=cam_dict.get("focus_dist"),
        aperture=cam_dict.get("aperture", 0.0),
    )
    return render(scene, camera, width, height, samples=samples, seed=seed)


def render_parallel(scene: Scene, camera: Camera, width: int, height: int, samples: int = 64,
                    seed: int = 42, workers: int = None) -> np.ndarray:
    """
    Render using multiple processes, splitting samples across workers.

    Returns a float32 (H, W, 3) sRGB image.
    """
    from multiprocessing import cpu_count
    import concurrent.futures

    if workers is None:
        workers = max(1, cpu_count() - 1)

    # Distribute samples as evenly as possible.
    base = samples // workers
    extra = samples % workers
    per_worker = [base + (1 if i < extra else 0) for i in range(workers)]

    bubbles_data = [
        {
            "centre": b.centre.tolist(),
            "radius": b.radius,
            "base": b.base_thickness,
            "var": b.thickness_variation,
        }
        for b in scene.bubbles
    ]
    cam_dict = {
        "origin": camera.origin.tolist(),
        "look_at": (camera.origin + camera.forward * camera.focus_dist).tolist(),
        "fov": camera.fov,
        "focus_dist": camera.focus_dist,
        "aperture": camera.aperture,
    }

    tasks = []
    for i, s in enumerate(per_worker):
        if s <= 0:
            continue
        tasks.append((bubbles_data, cam_dict, width, height, s, seed + i * 12345))

    if len(tasks) == 1:
        return _worker_render(tasks[0])

    partials = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [ex.submit(_worker_render, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            partials.append(fut.result())

    # Weighted average by sample count.
    total = sum(per_worker[:len(partials)])
    accum = np.zeros((height, width, 3), dtype=np.float64)
    for img, s in zip(partials, per_worker[:len(partials)]):
        accum += img * s
    return (accum / total).astype(np.float32)
