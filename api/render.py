"""
WSGI entrypoint for Vercel.
Vercel expects a top-level app/environ/start_response callable or ASGI.
This module wraps the Vercel handler into a WSGI app.
"""
import base64
import json
from urllib.parse import parse_qs
from tracer import Camera, Scene, render
from bubbles import Bubble
import numpy as np
from PIL import Image
import io

MAX_WIDTH = 320
MAX_HEIGHT = 240
MAX_SAMPLES = 16


app = None  # placeholder to satisfy Vercel discovery if it checks


def _do_render(environ):
    method = environ.get("REQUEST_METHOD", "GET")
    try:
        if method == "GET":
            params = {
                k: v[0] if isinstance(v, list) else v
                for k, v in parse_qs(environ.get("QUERY_STRING", "")).items()
            }
        else:
            length = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(length) if length else b"{}"
            params = json.loads(body.decode()) if body else {}

        width = min(int(params.get("width", 160)), MAX_WIDTH)
        height = min(int(params.get("height", 90)), MAX_HEIGHT)
        samples = min(int(params.get("samples", 8)), MAX_SAMPLES)
        film_thickness = float(params.get("film_thickness", 450.0))
        thickness_var = float(params.get("thickness_var", 40.0))
        bubble_radius = float(params.get("bubble_radius", 1.0))
        three = params.get("three", True)
        if isinstance(three, str):
            three = three.lower() in ("true", "1", "yes", "on")
        else:
            three = bool(three)

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
        img = render(scene, camera, width, height, samples=samples, seed=1)
        uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(uint8)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return {
            "status": 200,
            "headers": [
                ("Content-Type", "image/png"),
                ("Access-Control-Allow-Origin", "*"),
                ("Cache-Control", "public, max-age=60"),
            ],
            "body": png_bytes,
            "isBase64": False,
        }
    except Exception as e:
        err = json.dumps({"error": str(e)})
        return {
            "status": 500,
            "headers": [
                ("Content-Type", "application/json"),
                ("Access-Control-Allow-Origin", "*"),
            ],
            "body": err.encode(),
            "isBase64": False,
        }


def application(environ, start_response):
    result = _do_render(environ)
    start_response(f"{result['status']} OK", result["headers"])
    return [result["body"]]


# Some Vercel versions look for `app`
app = application
