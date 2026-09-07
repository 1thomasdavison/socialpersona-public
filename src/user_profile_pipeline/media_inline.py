from __future__ import annotations

import base64
import io
import mimetypes
import os
from functools import lru_cache
from pathlib import Path

import requests

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None
    ImageOps = None


def guess_image_mime(path_or_url: str) -> str:
    mime, _ = mimetypes.guess_type(path_or_url)
    return str(mime or "image/jpeg")


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P":
        return "transparency" in image.info
    return False


def _maybe_resize_image_bytes(data: bytes, mime: str) -> tuple[bytes, str]:
    if not data or Image is None or ImageOps is None:
        return data, mime

    max_dim = _read_int_env("BENCHMARK_INLINE_IMAGE_MAX_DIM", 0)
    jpeg_quality = min(max(_read_int_env("BENCHMARK_INLINE_IMAGE_JPEG_QUALITY", 70), 20), 95)
    png_compress_level = min(max(_read_int_env("BENCHMARK_INLINE_IMAGE_PNG_COMPRESS_LEVEL", 9), 0), 9)
    enable_reencode = bool(_read_int_env("BENCHMARK_INLINE_IMAGE_REENCODE", 0))
    if max_dim <= 0 and not enable_reencode:
        return data, mime

    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            if getattr(img, "is_animated", False):
                try:
                    img.seek(0)
                except Exception:
                    pass
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            resized = img.copy()
            original_size = tuple(resized.size)
            resized_applied = False
            if max_dim > 0 and max(original_size) > max_dim:
                resized.thumbnail((max_dim, max_dim), resample=resampling)
                resized_applied = tuple(resized.size) != original_size

            if not resized_applied and not enable_reencode:
                return data, mime

            if _has_alpha(resized):
                out = io.BytesIO()
                resized.save(out, format="PNG", optimize=True, compress_level=png_compress_level)
                return out.getvalue(), "image/png"

            out = io.BytesIO()
            rgb = resized.convert("RGB")
            rgb.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
            return out.getvalue(), "image/jpeg"
    except Exception:
        return data, mime


@lru_cache(maxsize=2048)
def image_source_to_data_url(source: str) -> str | None:
    raw = (source or "").strip()
    if not raw:
        return None
    if raw.startswith("data:"):
        return raw
    if raw.startswith("gs://"):
        return None

    max_bytes = _read_int_env("BENCHMARK_INLINE_IMAGE_MAX_BYTES", 8 * 1024 * 1024)
    timeout_seconds = _read_int_env("BENCHMARK_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", 20)

    try:
        if raw.startswith(("http://", "https://")):
            response = requests.get(
                raw,
                timeout=timeout_seconds,
                headers={"User-Agent": "socialpersona-inline-image/1.0"},
            )
            response.raise_for_status()
            data = response.content
            mime = str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip() or guess_image_mime(raw)
        else:
            path = Path(raw).expanduser()
            if not path.exists() or not path.is_file():
                return None
            data = path.read_bytes()
            mime = guess_image_mime(str(path))
    except Exception:
        return None

    if not data:
        return None
    data, mime = _maybe_resize_image_bytes(data, mime)
    if not data or len(data) > max_bytes:
        return None
    if not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
