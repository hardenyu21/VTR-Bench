from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def image_data_url(path: Path, max_png_bytes: int = 500_000) -> str:
    payload = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    if mime == "image/png" and len(payload) > max_png_bytes:
        with Image.open(io.BytesIO(payload)) as source:
            encoded = io.BytesIO()
            source.convert("RGB").save(
                encoded, format="JPEG", quality=92, optimize=True, subsampling=0
            )
        payload = encoded.getvalue()
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size

