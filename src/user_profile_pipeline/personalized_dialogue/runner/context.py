from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import re
from typing import Any

from ...benchmark.profile_eval.io import _load_posts_with_indices
from ...benchmark.profile_eval.modes import _slugify
from ...image_text import (
    VISUAL_MODE_NATIVE,
    VISUAL_MODE_TEXT_IMAGE,
    ImageTextGenerator,
    build_chat_client_for_model_spec,
    build_posts_context_bundle_for_visual_mode,
    normalize_visual_mode,
)
from ...media_inline import image_source_to_data_url
from ...multimodal_context import (
    build_posts_context_bundle,
    collect_resolved_post_images,
    render_posts_context_payload,
)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'").strip('"')
    return out


def _env_or_dotenv(key: str, dotenv: dict[str, str], default: str = "") -> str:
    return os.environ.get(key, "").strip() or dotenv.get(key, "").strip() or default


def _force_inline_images_for_provider(provider: str) -> bool:
    raw = str(os.environ.get("BENCHMARK_FORCE_INLINE_IMAGES", "") or "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return False
    return str(provider or "").strip().lower() == "vertex"


def _prepare_image_urls_for_provider(image_urls: list[str], *, provider: str) -> list[str]:
    if provider == "vertex" and not _force_inline_images_for_provider(provider):
        return image_urls
    out: list[str] = []
    for value in image_urls:
        url = (value or "").strip()
        if not url:
            continue
        if url.startswith("gs://"):
            continue
        converted = image_source_to_data_url(url)
        if converted:
            out.append(converted)
    return out


def _clear_inline_image_caches() -> None:
    try:
        image_source_to_data_url.cache_clear()
    except Exception:
        pass
    gc.collect()


def _context_limit_lowres_max_dim() -> int:
    raw = str(os.environ.get("BENCHMARK_CONTEXT_LIMIT_LOWRES_MAX_DIM", "512") or "512").strip()
    try:
        return max(32, int(raw))
    except Exception:
        return 512


@contextmanager
def _temporary_inline_image_settings(
    *,
    max_dim: int | None = None,
    reencode: bool | None = None,
) -> Any:
    updates: dict[str, str] = {}
    if max_dim is not None:
        updates["BENCHMARK_INLINE_IMAGE_MAX_DIM"] = str(max_dim)
    if reencode is not None:
        updates["BENCHMARK_INLINE_IMAGE_REENCODE"] = "1" if reencode else "0"

    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _trim_posts_context(context: str, *, max_posts: int, max_chars: int) -> str:
    text = str(context or "").strip()
    if not text:
        return ""
    if max_posts <= 0 and max_chars <= 0:
        return text

    blocks = [block.strip() for block in re.split(r"\n\n(?=\[Post index=\d+\])", text) if block.strip()]
    kept: list[str] = []
    total_chars = 0
    for block in blocks:
        if max_posts > 0 and len(kept) >= max_posts:
            break
        candidate = block
        sep_chars = 2 if kept else 0
        if max_chars > 0 and total_chars + sep_chars + len(candidate) > max_chars:
            remaining = max_chars - total_chars - sep_chars
            if remaining <= 0:
                break
            candidate = candidate[:remaining].rstrip()
            if not candidate:
                break
        kept.append(candidate)
        total_chars += sep_chars + len(candidate)
        if max_chars > 0 and total_chars >= max_chars:
            break
    return "\n\n".join(kept) if kept else text[:max_chars].rstrip()


def _extract_one_image_per_post_urls(context: dict[str, Any]) -> list[str]:
    urls = [str(x).strip() for x in list(context.get("one_image_per_post_urls") or []) if str(x).strip()]
    if urls:
        return urls
    out: list[str] = []
    for row in list(context.get("post_image_url_groups") or []):
        if not isinstance(row, dict):
            continue
        image_urls = [str(x).strip() for x in list(row.get("image_urls") or []) if str(x).strip()]
        if image_urls:
            out.append(image_urls[0])
    return out


def _select_context_and_image_urls(
    *,
    eval_task: dict[str, Any],
    max_posts: int,
    max_chars: int,
    max_images: int,
    one_image_per_post: bool = False,
    context_bundle: Any | None = None,
) -> tuple[str, list[str]]:
    if context_bundle is not None:
        raw_context = str(getattr(context_bundle, "text", "") or "").strip()
        all_image_urls = [str(x).strip() for x in list(getattr(context_bundle, "image_urls", []) or []) if str(x).strip()]
        one_image_per_post_urls = [
            str(x).strip() for x in list(getattr(context_bundle, "one_image_per_post_urls", []) or []) if str(x).strip()
        ]
    else:
        context = (eval_task.get("context") or {}) if isinstance(eval_task.get("context"), dict) else {}
        raw_context = str(context.get("posts_context_text") or "").strip()
        if not raw_context and isinstance(context.get("posts_context_schema"), dict):
            raw_context = render_posts_context_payload(context["posts_context_schema"])
        all_image_urls = [str(x).strip() for x in list(context.get("image_urls") or []) if str(x).strip()]
        one_image_per_post_urls = _extract_one_image_per_post_urls(context)

    trimmed_context = _trim_posts_context(raw_context, max_posts=max_posts, max_chars=max_chars)
    if not trimmed_context:
        trimmed_context = raw_context
    selected_image_urls = list(one_image_per_post_urls if one_image_per_post else all_image_urls)

    if max_images >= 0:
        selected_image_urls = selected_image_urls[:max_images]
    return trimmed_context, selected_image_urls


def _build_visual_context_bundle(
    *,
    gold_ref: dict[str, Any],
    eval_task: dict[str, Any],
    spec: Any,
    visual_mode: str,
    output_dir: Path,
    force: bool,
    image_text_generators: dict[str, ImageTextGenerator],
    bundle_cache: dict[tuple[str, str, str], Any],
    image_text_max_tokens: int,
    image_text_timeout_seconds: int,
    image_text_cache_root: Path | None = None,
    image_text_cache_only: bool = False,
) -> Any | None:
    def restore_remote_media_sources(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        original_posts_cache: dict[str, dict[int, dict[str, Any]]] = {}
        restored_posts: list[dict[str, Any]] = []
        for post in posts:
            original_dir = str(post.get("hard_multimodal_original_posts_dir") or "").strip()
            original_index = int(post.get("hard_multimodal_original_post_index", -1))
            original_row: dict[str, Any] | None = None
            if original_dir and original_index >= 0:
                cached_rows = original_posts_cache.get(original_dir)
                if cached_rows is None:
                    original_posts_path = Path(original_dir).resolve() / "posts.jsonl"
                    if original_posts_path.exists():
                        cached_rows = {
                            int(row.get("_line_index", -1)): row
                            for row in _load_posts_with_indices(original_posts_path)
                            if int(row.get("_line_index", -1)) >= 0
                        }
                    else:
                        cached_rows = {}
                    original_posts_cache[original_dir] = cached_rows
                original_row = cached_rows.get(original_index)

            merged_post = dict(post)
            merged_media: list[dict[str, Any]] = []
            original_media_rows = list((original_row or {}).get("media") or [])
            for media_idx, media in enumerate(list(post.get("media") or [])):
                media_copy = dict(media)
                if media_idx < len(original_media_rows):
                    original_media = original_media_rows[media_idx]
                    original_source_url = str(original_media.get("source_url") or "").strip()
                    if original_source_url.startswith(("http://", "https://", "gs://", "data:")):
                        media_copy["source_url"] = original_source_url
                merged_media.append(media_copy)
            merged_post["media"] = merged_media
            restored_posts.append(merged_post)
        return restored_posts

    normalized_visual_mode = normalize_visual_mode(visual_mode)
    if normalized_visual_mode == VISUAL_MODE_NATIVE:
        user_id = str(eval_task.get("user_id") or gold_ref.get("user_id") or "")
        image_source_preference = (
            "local_first"
            if _force_inline_images_for_provider("vertex")
            else "remote_first"
        )
        cache_key = (str(spec.name), user_id, f"{normalized_visual_mode}:{image_source_preference}")
        existing = bundle_cache.get(cache_key)
        if existing is not None:
            return existing
        posts_path = Path(str(((gold_ref.get("source") or {}).get("posts_path")) or "")).resolve()
        if not posts_path.exists():
            raise FileNotFoundError(f"posts_path missing for visual_mode={normalized_visual_mode}: {posts_path}")
        posts = _load_posts_with_indices(posts_path)
        posts = restore_remote_media_sources(posts)
        bundle = build_posts_context_bundle(posts, image_source_preference=image_source_preference)
        bundle_cache[cache_key] = bundle
        return bundle
    user_id = str(eval_task.get("user_id") or gold_ref.get("user_id") or "")
    cache_key = (str(spec.name), user_id, normalized_visual_mode)
    existing = bundle_cache.get(cache_key)
    if existing is not None:
        return existing

    posts_path = Path(str(((gold_ref.get("source") or {}).get("posts_path")) or "")).resolve()
    if not posts_path.exists():
        raise FileNotFoundError(f"posts_path missing for visual_mode={normalized_visual_mode}: {posts_path}")
    posts = _load_posts_with_indices(posts_path)

    if normalized_visual_mode == VISUAL_MODE_TEXT_IMAGE:
        if not bool(getattr(spec, "multimodal", False)):
            raise ValueError(f"visual_mode=text_image requires a multimodal model, got {spec.name}")
        generator = image_text_generators.get(str(spec.name))
        if generator is None:
            shared_profile_cache_dir = (
                output_dir.parent.parent / "profile_text_image" / "cache" / "_image_text" / _slugify(str(spec.name))
            )
            if image_text_cache_root is not None:
                cache_dir = image_text_cache_root / "_image_text" / _slugify(str(spec.name))
            else:
                cache_dir = (
                    shared_profile_cache_dir
                    if shared_profile_cache_dir.exists()
                    else (output_dir / "_image_text_cache" / _slugify(str(spec.name)))
                )
            generator = ImageTextGenerator(
                client=build_chat_client_for_model_spec(
                    spec,
                    timeout_seconds=int(image_text_timeout_seconds),
                    max_tokens=int(image_text_max_tokens),
                    temperature=0.0,
                ),
                model_name=str(spec.name),
                cache_dir=cache_dir,
                force=force,
            )
            image_text_generators[str(spec.name)] = generator
        if image_text_cache_only:
            image_text_map = _load_cached_image_text_map(generator=generator, posts=posts)
        else:
            image_text_map = generator.describe_posts(posts)
        bundle = build_posts_context_bundle_for_visual_mode(
            posts=posts,
            visual_mode=normalized_visual_mode,
            image_text_map=image_text_map,
        )
    else:
        bundle = build_posts_context_bundle_for_visual_mode(
            posts=posts,
            visual_mode=normalized_visual_mode,
        )
    bundle_cache[cache_key] = bundle
    return bundle


def _load_cached_image_text_map(
    *,
    generator: ImageTextGenerator,
    posts: list[dict[str, Any]],
) -> dict[tuple[int, int], Any]:
    out: dict[tuple[int, int], Any] = {}
    images = collect_resolved_post_images(posts)
    total = len(images)
    missing = 0
    unreadable = 0
    for idx, image in enumerate(images, start=1):
        cache_path = generator.cache_dir / f"{generator._cache_key(image)}.json"
        if not cache_path.exists():
            missing += 1
            if missing <= 8:
                print(
                    f"[image_text:{generator.model_name}] missing_cache_skip {idx}/{total} "
                    f"post_index={image.post_index} post_image_index={image.post_image_index}",
                    flush=True,
                )
            continue
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            out[(image.post_index, image.post_image_index)] = generator._result_from_payload(payload)
        except Exception as exc:
            unreadable += 1
            if unreadable <= 8:
                print(
                    f"[image_text:{generator.model_name}] unreadable_cache_skip {idx}/{total} "
                    f"post_index={image.post_index} post_image_index={image.post_image_index}: {exc}",
                    flush=True,
                )
    if missing or unreadable:
        print(
            f"[image_text:{generator.model_name}] cache_only loaded={len(out)}/{total} "
            f"missing={missing} unreadable={unreadable}",
            flush=True,
        )
    return out
