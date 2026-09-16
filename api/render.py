import base64
import json
from urllib.parse import parse_qs
from tracer import Camera, Scene, render
from bubbles import Bubble
import numpy as np

MAX_WIDTH = 320
MAX_HEIGHT = 240
MAX_SAMPLES = 16


app = None  # Vercel entrypoint is the handler() function


def handler(request):
    """Vercel serverless render endpoint (CPU fallback)."""
    if request.method == "OPTIONS":
        return {
            "statusCode": 204,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        }

    try:
        if request.method == "GET":
            params = request.args or {}
        else:
            body = request.body or b"{}"
            params = json.loads(body.decode()) if body else {}

        width = min(int(params.get("width", 160)), MAX_WIDTH)
        height = min(int(params.get("height", 90)), MAX_HEIGHT)
        samples = min(int(params.get("samples", 8)), MAX_SAMPLES)
        film_thickness = float(params.get("film_thickness", 450.0))
        thickness_var = float(params.get("thickness_var", 40.0))
        bubble_radius = float(params.get("bubble_radius", 1.0))
        three = bool(params.get("three", True))

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
        from PIL import Image
        pil = Image.fromarray(uint8)
        import io
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "image/png",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=60",
            },
            "body": base64.b64encode(png_bytes).decode("utf-8"),
            "isBase64Encoded": True,
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
