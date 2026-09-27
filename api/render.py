"""
WSGI entrypoint for Vercel.
Vercel expects a top-level app/environ/start_response callable or ASGI.
This module wraps the Vercel handler into a WSGI app.
"""
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


def _bool_param(params, key, default):
    v = params.get(key, default)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "on")
    return bool(v)


def _parse_color(params, key):
    v = params.get(key)
    if v is None:
        return None
    if isinstance(v, str):
        if v.startswith("#"):
            return np.array([
                int(v[1:3], 16) / 255.0,
                int(v[3:5], 16) / 255.0,
                int(v[5:7], 16) / 255.0,
            ], dtype=np.float32)
        parts = [float(p) for p in v.split(",")]
        return np.array(parts, dtype=np.float32) / 255.0 if max(parts) > 1.0 else np.array(parts, dtype=np.float32)
    arr = np.array(v, dtype=np.float32)
    return arr / 255.0 if arr.max() > 1.0 else arr


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
        sun_azimuth = float(params["sun_azimuth"]) if "sun_azimuth" in params else None
        sun_elevation = float(params["sun_elevation"]) if "sun_elevation" in params else None
        rim_azimuth = float(params["rim_azimuth"]) if "rim_azimuth" in params else None
        rim_elevation = float(params["rim_elevation"]) if "rim_elevation" in params else None
        sun_color = _parse_color(params, "sun_color")
        rim_color = _parse_color(params, "rim_color")
        sun_disc = float(params.get("sun_disc", 0.0))
        ground_gloss = float(params.get("ground_gloss", 0.0))
        ground_refl = float(params.get("ground_refl", 0.0))
        ground_rough = float(params.get("ground_rough", 0.0))
        exposure = float(params.get("exposure", 1.0))
        tone_map = params.get("tone_map", "linear")
        saturation = float(params.get("saturation", 1.0))
        vignette = float(params.get("vignette", 0.0))
        bloom = float(params.get("bloom", 0.0))
        grain = float(params.get("grain", 0.0))
        chromatic = float(params.get("chromatic", 0.0))

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
            sun_azimuth=sun_azimuth,
            sun_elevation=sun_elevation,
            rim_azimuth=rim_azimuth,
            rim_elevation=rim_elevation,
            sun_color=sun_color,
            rim_color=rim_color,
            sun_disc=sun_disc,
            ground_gloss=ground_gloss,
            ground_reflectivity=ground_refl,
            ground_roughness=ground_rough,
            exposure=exposure,
        )
        img = render(scene, camera, width, height, samples=samples, seed=1,
                     tone_map=tone_map, saturation=saturation, vignette=vignette,
                     bloom=bloom, grain=grain, chromatic=chromatic)
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
