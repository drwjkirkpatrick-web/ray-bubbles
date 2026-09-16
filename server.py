"""
Local GPU render server for the Jetson.

Run with:
    python3 server.py

Then point the Vercel frontend at http://your-jetson-ip:5000/render
"""
import base64
import io
import json
import os
import time
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

from tracer import Camera, Scene
from bubbles import Bubble
from cuda_tracer import gpu_available, render_gpu

app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"gpu": gpu_available(), "status": "ok"})


@app.route("/render", methods=["POST", "OPTIONS"])
def render_route():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(force=True) or {}
    width = min(int(data.get("width", 960)), 1920)
    height = min(int(data.get("height", 540)), 1080)
    samples = min(int(data.get("samples", 64)), 512)
    film_thickness = float(data.get("film_thickness", 450.0))
    thickness_var = float(data.get("thickness_var", 40.0))
    bubble_radius = float(data.get("bubble_radius", 1.0))
    three = bool(data.get("three", True))
    use_gpu = bool(data.get("gpu", True))

    common = dict(base_thickness_nm=film_thickness, thickness_variation_nm=thickness_var)
    if three:
        bubbles = [
            Bubble(centre=np.array([-1.8, 0.9, 0.2], dtype=np.float32), radius=bubble_radius * 0.9, **common),
            Bubble(centre=np.array([0.0, 1.1, -0.3], dtype=np.float32), radius=bubble_radius, **common),
            Bubble(centre=np.array([1.9, 0.75, 0.1], dtype=np.float32), radius=bubble_radius * 1.15, **common),
        ]
        camera = Camera(
            origin=np.array([0.0, 2.0, 6.0], dtype=np.float32),
            look_at=np.array([0.0, 0.9, 0.0], dtype=np.float32),
            fov_deg=50.0,
        )
    else:
        bubbles = [Bubble(centre=np.array([0.0, 0.8, 0.0], dtype=np.float32), radius=bubble_radius, **common)]
        camera = Camera(
            origin=np.array([0.0, 1.5, 6.0], dtype=np.float32),
            look_at=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            fov_deg=45.0,
        )

    scene = Scene(bubbles=bubbles)

    t0 = time.time()
    if use_gpu and gpu_available():
        arr = render_gpu(scene, camera, width, height, samples=samples, seed=1, out_path="/tmp/raybubbles_gpu.png")
        pil = Image.fromarray(arr)
    else:
        from tracer import render
        img = render(scene, camera, width, height, samples=samples, seed=1)
        uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(uint8)
    elapsed = time.time() - t0

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    print(f"rendered {width}x{height} in {elapsed:.2f}s")
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
