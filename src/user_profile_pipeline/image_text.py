from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tenacity import RetryError

from .llm_client import FIXED_REPAIR_MODEL, OpenAICompatibleChatClient
from .media_inline import image_source_to_data_url
from .multimodal_context import PostsContextBundle, ResolvedPostImage, build_posts_context_bundle, collect_resolved_post_images


VISUAL_MODE_NATIVE = "native"
VISUAL_MODE_TEXT_IMAGE = "text_image"
VISUAL_MODE_WO_TEXT_IMAGE = "wo_text_image"
SUPPORTED_VISUAL_MODES = {
    VISUAL_MODE_NATIVE,
    VISUAL_MODE_TEXT_IMAGE,
    VISUAL_MODE_WO_TEXT_IMAGE,
}
IMAGE_TEXT_PROMPT_VERSION = "image_text_v1"

IMAGE_TEXT_SYSTEM_PROMPT = """You analyze one social-media image for downstream user-interest inference.

Return strict JSON only.

Rules:
- Focus on visible content only.
- Describe the main subject, activity, setting, mood/style, and any clearly readable text.
- Do not infer identity, demographics, private traits, or motivation.
- If the image is low-information, say that briefly.
- Keep summary concise and concrete.
"""

IMAGE_TEXT_JSON_SCHEMA_HINT = """{
  "summary": "string",
  "visible_text": ["string"],
  "tags": ["string"]
}"""


@dataclass(frozen=True)
class ImageTextResult:
    source: str
    summary: str
    visible_text: list[str]
    tags: list[str]
    model_name: str
    prompt_version: str = IMAGE_TEXT_PROMPT_VERSION


def normalize_visual_mode(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return VISUAL_MODE_NATIVE
    if raw not in SUPPORTED_VISUAL_MODES:
        raise ValueError(
            f"Unsupported visual mode: {value}. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_VISUAL_MODES))}"
        )
    return raw


def build_chat_client_for_model_spec(
    spec: Any,
    *,
    timeout_seconds: int,
    max_tokens: int,
    temperature: float = 0.0,
) -> OpenAICompatibleChatClient:
    provider = str(getattr(spec, "provider", "") or "").strip().lower()
    if provider in {"chatanywhere", "bailian", "bailian_batch"}:
        normalized_provider = "openai_compatible"
    elif provider.startswith("vertex"):
        normalized_provider = "vertex"
    else:
        raise ValueError(f"Unsupported image-text provider: {provider}")
    return OpenAICompatibleChatClient(
        provider=normalized_provider,
        base_url=str(getattr(spec, "base_url", "") or ""),
        model=str(getattr(spec, "api_model", "") or getattr(spec, "name", "")),
        api_key_env=str(getattr(spec, "api_key_env", "") or ""),
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        repair_model=(
            FIXED_REPAIR_MODEL
            if normalized_provider == "vertex"
            else str(getattr(spec, "api_model", "") or getattr(spec, "name", ""))
        ),
    )


def prediction_visual_mode(row: dict[str, Any]) -> str:
    return normalize_visual_mode(str(row["visual_mode"]))


def build_posts_context_bundle_for_visual_mode(
    *,
    posts: list[dict[str, Any]],
    visual_mode: str,
    image_text_map: dict[tuple[int, int], ImageTextResult] | None = None,
) -> PostsContextBundle:
    normalized_mode = normalize_visual_mode(visual_mode)
    image_source_preference = str(os.environ.get("BENCHMARK_IMAGE_SOURCE_PREFERENCE", "local_first") or "local_first").strip()
    if normalized_mode == VISUAL_MODE_NATIVE:
        return build_posts_context_bundle(posts, image_source_preference=image_source_preference)
    if normalized_mode == VISUAL_MODE_WO_TEXT_IMAGE:
        return build_posts_context_bundle(posts, drop_raw_images=True)
    if image_text_map is None:
        raise ValueError("image_text_map is required for visual_mode=text_image")
    return build_posts_context_bundle(
        posts,
        image_text_map=image_text_map,
        drop_raw_images=True,
    )


class ImageTextGenerator:
    def __init__(
        self,
        *,
        client: OpenAICompatibleChatClient,
        model_name: str,
        cache_dir: str | Path,
        force: bool = False,
    ) -> None:
        self.client = client
        self.model_name = str(model_name or "").strip() or "model"
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force = bool(force)

    def describe_posts(self, posts: list[dict[str, Any]]) -> dict[tuple[int, int], ImageTextResult]:
        out: dict[tuple[int, int], ImageTextResult] = {}
        images = collect_resolved_post_images(posts)
        total = len(images)
        for idx, image in enumerate(images, start=1):
            cache_path = self.cache_dir / f"{self._cache_key(image)}.json"
            action = "cached" if cache_path.exists() and not self.force else "describing"
            print(
                f"[image_text:{self.model_name}] {action} {idx}/{total} "
                f"post_index={image.post_index} post_image_index={image.post_image_index}",
                flush=True,
            )
            out[(image.post_index, image.post_image_index)] = self.describe_image(image)
        return out

    def describe_image(
        self,
        image: ResolvedPostImage,
        *,
        prepared_source: str | None = None,
        prepared_user_prompt: str | None = None,
    ) -> ImageTextResult:
        cache_path = self.cache_dir / f"{self._cache_key(image)}.json"
        if cache_path.exists() and not self.force:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return self._result_from_payload(payload)
            except Exception:
                pass

        try:
            image_source = prepared_source if prepared_source else self._prepare_image_source(image.source)
            user_prompt = prepared_user_prompt if prepared_user_prompt else self._build_user_prompt(image)
            payload = self.client.chat_json(
                system_prompt=IMAGE_TEXT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_urls=[image_source],
                json_schema_hint=IMAGE_TEXT_JSON_SCHEMA_HINT,
            )
            result = self._normalize_result(image=image, payload=payload)
            self._write_cache(cache_path, result)
            return result
        except Exception as exc:
            if not self._is_data_inspection_failure(exc):
                raise
            result = ImageTextResult(
                source=image.source,
                summary="Image omitted because the provider blocked visual analysis during safety inspection.",
                visible_text=[],
                tags=["image_blocked"],
                model_name=self.model_name,
            )
            self._write_cache(
                cache_path,
                result,
                extra={
                    "blocked": True,
                    "blocked_reason": "data_inspection_failed",
                    "blocked_error": str(exc)[:1200],
                },
            )
            return result

    def _build_user_prompt(self, image: ResolvedPostImage) -> str:
        hint = image.caption_hint or "No existing caption."
        return (
            "Analyze this single image for profile/dialogue personalization.\n\n"
            f"Post index: {image.post_index}\n"
            f"Post image index: {image.post_image_index}\n"
            f"Existing caption hint: {hint}\n\n"
            "Return JSON with:\n"
            '- summary: one concise sentence about the visible content.\n'
            '- visible_text: up to 5 short OCR strings if clearly readable.\n'
            '- tags: up to 6 short topic tags.\n'
        )

    def _cache_key(self, image: ResolvedPostImage) -> str:
        identity = {
            "model_name": self.model_name,
            "prompt_version": IMAGE_TEXT_PROMPT_VERSION,
            "source": image.source,
            "caption_hint": image.caption_hint,
        }
        source = image.source
        if source and not source.startswith(("http://", "https://", "gs://", "data:")):
            try:
                path = Path(source).expanduser().resolve()
                stat = path.stat()
                identity["source"] = str(path)
                identity["size"] = int(stat.st_size)
                identity["mtime_ns"] = int(stat.st_mtime_ns)
            except Exception:
                pass
        raw = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _normalize_result(self, *, image: ResolvedPostImage, payload: dict[str, Any]) -> ImageTextResult:
        summary = " ".join(str(payload.get("summary") or "").split()).strip()
        if not summary:
            summary = "Low-information image with no confident visual summary."
        return ImageTextResult(
            source=image.source,
            summary=summary,
            visible_text=_normalize_short_text_list(payload.get("visible_text"), limit=5),
            tags=_normalize_short_text_list(payload.get("tags"), limit=6),
            model_name=self.model_name,
        )

    def _result_from_payload(self, payload: dict[str, Any]) -> ImageTextResult:
        return ImageTextResult(
            source=str(payload.get("source") or "").strip(),
            summary=" ".join(str(payload.get("summary") or "").split()).strip(),
            visible_text=_normalize_short_text_list(payload.get("visible_text"), limit=5),
            tags=_normalize_short_text_list(payload.get("tags"), limit=6),
            model_name=str(payload.get("model_name") or self.model_name).strip() or self.model_name,
            prompt_version=str(payload.get("prompt_version") or IMAGE_TEXT_PROMPT_VERSION).strip() or IMAGE_TEXT_PROMPT_VERSION,
        )

    def _prepare_image_source(self, source: str) -> str:
        converted = image_source_to_data_url(source)
        if not converted:
            raise ValueError(f"image_text_failed_to_inline_source: {source}")
        return converted

    def _write_cache(
        self,
        cache_path: Path,
        result: ImageTextResult,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "source": result.source,
            "summary": result.summary,
            "visible_text": result.visible_text,
            "tags": result.tags,
            "model_name": result.model_name,
            "prompt_version": result.prompt_version,
            "generated_at": int(time.time()),
        }
        if extra:
            payload.update(extra)
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _is_data_inspection_failure(exc: Exception) -> bool:
        inner: Exception = exc
        if isinstance(exc, RetryError):
            try:
                candidate = exc.last_attempt.exception()
                if isinstance(candidate, Exception):
                    inner = candidate
            except Exception:
                inner = exc
        text = str(inner or "").lower()
        return "datainspectionfailed" in text or "data_inspection_failed" in text


def render_image_text_result(result: ImageTextResult) -> str:
    parts = [result.summary.strip()]
    if result.visible_text:
        parts.append(f"visible_text={'; '.join(result.visible_text)}")
    if result.tags:
        parts.append(f"tags={', '.join(result.tags)}")
    return " | ".join(part for part in parts if part).strip()


def _normalize_short_text_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text[:240])
        if len(out) >= limit:
            break
    return out
