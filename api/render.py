"""
WSGI entrypoint for Vercel.
Vercel expects a top-level app/environ/start_response callable or ASGI.
This module wraps the Vercel handler into a WSGI app.
"""
import json
from urllib.parse import parse_qs
from tracer import Camera, Scene, render
from bubbles import Bubble
from spectrum import apply_tone_map
import numpy as np
from PIL import Image
import io

MAX_WIDTH = 320
MAX_HEIGHT = 240
MAX_SAMPLES = 16


app = None  # placeholder to satisfy Vercel discovery if it checks


def _bool_param(params, key, default):
    v = params.get(key, default)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "on")
    return bool(v)


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
        three = _bool_param(params, "three", True)
        background = params.get("background", "sky")
        gradient = float(params.get("gradient", 0.5))
        swirl = float(params.get("swirl", 1.0))
        sun_power = float(params.get("sun_power", 1.0))
        rim_power = float(params.get("rim_power", 0.6))
        ground_gloss = float(params.get("ground_gloss", 0.0))
        ground_refl = float(params.get("ground_refl", 0.0))
        exposure = float(params.get("exposure", 1.0))
        tone_map = params.get("tone_map", "linear")
        saturation = float(params.get("saturation", 1.0))

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

        scene = Scene(
            bubbles=bubbles,
            background=background,
            sun_power=sun_power,
            rim_power=rim_power,
            ground_gloss=ground_gloss,
            ground_reflectivity=ground_refl,
            exposure=exposure,
        )
        img = render(scene, camera, width, height, samples=samples, seed=1, tone_map=tone_map, saturation=saturation)
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
        import traceback
        err = json.dumps({"error": str(e), "trace": traceback.format_exc()})
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
