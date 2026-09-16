import json
import base64
from PIL import Image
import io

app = None  # placeholder to satisfy Vercel discovery


def _health_png():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _do_health(environ):
    try:
        png = _health_png()
        return {
            "status": 200,
            "headers": [
                ("Content-Type", "image/png"),
                ("Access-Control-Allow-Origin", "*"),
                ("Cache-Control", "no-cache"),
            ],
            "body": png,
        }
    except Exception as e:
        err = json.dumps({"error": str(e)})
        return {
            "status": 500,
            "headers": [("Content-Type", "application/json"), ("Access-Control-Allow-Origin", "*")],
            "body": err.encode(),
        }


def application(environ, start_response):
    result = _do_health(environ)
    start_response(f"{result['status']} OK", result["headers"])
    return [result["body"]]


app = application
