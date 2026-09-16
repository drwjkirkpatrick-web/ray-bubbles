"""
Local GPU render server for the Jetson.

Run with:
    python3 server.py

Then point the Vercel frontend at http://your-jetson-ip:5000/render
"""
import io
import os
import time
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

from tracer import Camera, Scene, render
from bubbles import Bubble
from cuda_tracer import gpu_available, render_gpu
from spectrum import apply_tone_map

app = Flask(__name__)
CORS(app)


def _build_scene_camera(data: dict):
    width = min(int(data.get("width", 960)), 1920)
    height = min(int(data.get("height", 540)), 1080)
    samples = min(int(data.get("samples", 64)), 512)
    film_thickness = float(data.get("film_thickness", 450.0))
    thickness_var = float(data.get("thickness_var", 40.0))
    bubble_radius = float(data.get("bubble_radius", 1.0))
    three = bool(data.get("three", True))
    background = data.get("background", "sky")
    gradient = float(data.get("gradient", 0.5))
    swirl = float(data.get("swirl", 1.0))
    sun_power = float(data.get("sun_power", 1.0))
    rim_power = float(data.get("rim_power", 0.6))
    ground_gloss = float(data.get("ground_gloss", 0.0))
    ground_refl = float(data.get("ground_refl", 0.0))
    exposure = float(data.get("exposure", 1.0))
    tone_map = data.get("tone_map", "linear")
    saturation = float(data.get("saturation", 1.0))
    aperture = float(data.get("aperture", 0.0))
    camera_dist = float(data.get("camera_dist", 6.0))
    fov = float(data.get("fov", 50.0)) if three else float(data.get("fov", 45.0))

    common = dict(
        base_thickness_nm=film_thickness,
        thickness_variation_nm=thickness_var,
        gradient_factor=gradient,
        swirl_strength=swirl,
    )
    if three:
        bubbles = [
            Bubble(centre=np.array([-1.8, 0.9, 0.2], dtype=np.float32), radius=bubble_radius * 0.9, **common),
            Bubble(centre=np.array([0.0, 1.1, -0.3], dtype=np.float32), radius=bubble_radius, **common),
            Bubble(centre=np.array([1.9, 0.75, 0.1], dtype=np.float32), radius=bubble_radius * 1.15, **common),
        ]
        camera = Camera(
            origin=np.array([0.0, 2.0, camera_dist], dtype=np.float32),
            look_at=np.array([0.0, 0.9, 0.0], dtype=np.float32),
            fov_deg=fov,
            aperture=aperture,
            focus_dist=camera_dist,
        )
    else:
        bubbles = [Bubble(centre=np.array([0.0, 0.8, 0.0], dtype=np.float32), radius=bubble_radius, **common)]
        camera = Camera(
            origin=np.array([0.0, 1.5, camera_dist], dtype=np.float32),
            look_at=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            fov_deg=fov,
            aperture=aperture,
            focus_dist=camera_dist,
        )

    scene = Scene(
        bubbles=bubbles,
        background=background,
        sun_power=sun_power,
        rim_power=rim_power,
        ground_gloss=ground_gloss,
        ground_reflectivity=ground_refl,
        exposure=exposure,
    )
    return scene, camera, width, height, samples, tone_map, saturation


@app.route("/health")
def health():
    return jsonify({"gpu": gpu_available(), "status": "ok"})


@app.route("/render", methods=["POST", "OPTIONS"])
def render_route():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(force=True) or {}
    scene, camera, width, height, samples, tone_map, saturation = _build_scene_camera(data)

    t0 = time.time()
    if gpu_available():
        arr = render_gpu(scene, camera, width, height, samples=samples, seed=1, out_path="/tmp/raybubbles_gpu.png")
        pil = Image.fromarray(arr)
    else:
        img = render(scene, camera, width, height, samples=samples, seed=1, tone_map=tone_map, saturation=saturation)
        uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(uint8)
    elapsed = time.time() - t0

    # Apply tone-map / saturation to GPU output on the host so it matches CPU options.
    if gpu_available():
        rgb = np.array(pil).astype(np.float32) / 255.0
        rgb = apply_tone_map(rgb, mode=tone_map)
        if saturation != 1.0:
            grey = np.mean(rgb, axis=-1, keepdims=True)
            rgb = np.clip(grey + (rgb - grey) * saturation, 0.0, 1.0)
        uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(uint8)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    print(f"rendered {width}x{height} in {elapsed:.2f}s")
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
