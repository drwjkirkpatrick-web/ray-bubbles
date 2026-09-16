import json
import base64
from PIL import Image
import io
import numpy as np

app = None


def handler(request):
    """Minimal health/diagnostic endpoint."""
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
        # Create a tiny red PNG
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "image/png",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache",
            },
            "body": base64.b64encode(png).decode("utf-8"),
            "isBase64Encoded": True,
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e), "trace": __import__("traceback").format_exc()}),
        }
