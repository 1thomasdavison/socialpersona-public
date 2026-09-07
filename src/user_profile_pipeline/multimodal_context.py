from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PostsContextBundle:
    payload: dict[str, Any]
    text: str
    image_urls: list[str]
    one_image_per_post_urls: list[str]
    post_image_url_groups: list[dict[str, Any]]


@dataclass(frozen=True)
class ResolvedPostImage:
    post_index: int
    post_id: str
    post_image_index: int
    source: str
    caption_hint: str


def media_caption(media: dict[str, Any], image_index: int) -> str:
    alt_text = str(media.get("alt_text") or "").strip()
    if alt_text:
        return alt_text
    source_url = str(media.get("source_url") or "").strip()
    if source_url:
        return f"Image available at {source_url}"
    return f"Image {image_index}"


def apply_image_limit(image_urls: list[str], max_images: int) -> list[str]:
    if max_images < 0:
        return list(image_urls)
    return list(image_urls)[:max_images]


def render_posts_context_payload(payload: dict[str, Any]) -> str:
    # Keep prompts compact to reduce upload size and avoid wasting context on whitespace.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_retry_posts_context_text(
    payload: dict[str, Any],
    *,
    redact_text: bool = False,
    text_limit: int = 280,
    image_text_limit: int = 140,
    hashtag_limit: int = 8,
) -> str:
    posts_out: list[dict[str, Any]] = []
    for post in list(payload.get("posts") or []):
        if not isinstance(post, dict):
            continue
        row: dict[str, Any] = {}
        try:
            row["i"] = int(post.get("post_index", -1))
        except Exception:
            row["i"] = -1
        created_at = _normalize_prompt_text(post.get("created_at"), limit=64)
        if created_at:
            row["ts"] = created_at

        text = _normalize_prompt_text(post.get("text"), limit=text_limit)
        if text:
            row["t"] = text
        if not redact_text:
            hashtags = _normalize_short_items(post.get("hashtags"), limit=hashtag_limit, item_limit=40)
            if hashtags:
                row["h"] = hashtags

        image_texts: list[str] = []
        for image_row in list(post.get("image_texts") or []):
            if not isinstance(image_row, dict):
                continue
            image_text = _normalize_prompt_text(image_row.get("image_text"), limit=image_text_limit)
            if image_text:
                image_texts.append(image_text)
        if image_texts:
            row["img"] = image_texts
        posts_out.append(row)

    compact_payload = {
        "schema_version": "multimodal_posts_context_retry_v2",
        "redacted_text": bool(redact_text),
        "posts": posts_out,
    }
    return json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))


def collect_resolved_post_images(posts: list[dict[str, Any]]) -> list[ResolvedPostImage]:
    out: list[ResolvedPostImage] = []
    for post in posts:
        line_idx = int(post.get("_line_index", -1))
        post_id = str(post.get("post_id") or "").strip()
        for media_index, media in enumerate(post.get("media", []) or [], start=1):
            if str(media.get("media_type") or "").strip().lower() != "image":
                continue
            picked = resolve_post_image_url(post=post, media=media)
            if not picked:
                continue
            out.append(
                ResolvedPostImage(
                    post_index=line_idx,
                    post_id=post_id,
                    post_image_index=media_index,
                    source=picked,
                    caption_hint=media_caption(media, media_index),
                )
            )
    return out


def build_posts_context_bundle(
    posts: list[dict[str, Any]],
    *,
    image_text_map: dict[tuple[int, int], Any] | None = None,
    drop_raw_images: bool = False,
    image_source_preference: str = "local_first",
) -> PostsContextBundle:
    posts_payload: list[dict[str, Any]] = []
    image_inputs: list[dict[str, Any]] = []
    image_text_inputs: list[dict[str, Any]] = []
    image_urls: list[str] = []
    one_image_per_post_urls: list[str] = []
    post_image_url_groups: list[dict[str, Any]] = []

    for post in posts:
        line_idx = int(post.get("_line_index", -1))
        post_id = str(post.get("post_id") or "").strip()
        resolved_urls: list[str] = []
        image_input_indices: list[int] = []
        post_image_texts: list[dict[str, Any]] = []

        for media_index, media in enumerate(post.get("media", []) or [], start=1):
            if str(media.get("media_type") or "").strip().lower() != "image":
                continue
            caption = media_caption(media, media_index)
            image_text_row = (image_text_map or {}).get((line_idx, media_index))
            if image_text_row is not None:
                text_value = _image_text_value(image_text_row)
                if text_value:
                    image_text_inputs.append(
                        {
                            "post_index": line_idx,
                            "post_id": post_id,
                            "post_image_index": media_index,
                            "caption": caption,
                            "image_text": text_value,
                        }
                    )
                    post_image_texts.append(
                        {
                            "post_image_index": media_index,
                            "caption": caption,
                            "image_text": text_value,
                        }
                    )
            picked = resolve_post_image_url(
                post=post,
                media=media,
                image_source_preference=image_source_preference,
            )
            if not picked:
                continue
            if drop_raw_images:
                continue
            image_input_index = len(image_inputs) + 1
            image_inputs.append(
                {
                    "image_input_index": image_input_index,
                    "post_index": line_idx,
                    "post_id": post_id,
                    "post_image_index": media_index,
                    "caption": caption,
                }
            )
            image_urls.append(picked)
            resolved_urls.append(picked)
            image_input_indices.append(image_input_index)

        if resolved_urls:
            one_image_per_post_urls.append(resolved_urls[0])
            post_image_url_groups.append(
                {
                    "post_index": line_idx,
                    "post_id": post_id,
                    "image_urls": list(resolved_urls),
                }
            )

        posts_payload.append(
            {
                "post_index": line_idx,
                "post_id": post_id,
                "created_at": str(post.get("created_at") or "").strip(),
                "platform": str(post.get("platform") or "").strip(),
                "post_type": str(post.get("post_type") or "").strip(),
                "language": post.get("language"),
                "text": str(post.get("text") or ""),
                "hashtags": list(post.get("hashtags") or []),
                "mentions": list(post.get("mentions") or []),
                "urls": list(post.get("urls") or []),
                "image_count": len(image_input_indices) if not drop_raw_images else 0,
                "image_input_indices": image_input_indices if not drop_raw_images else [],
                "image_text_count": len(post_image_texts),
                "image_texts": post_image_texts,
            }
        )

    payload = {
        "schema_version": "multimodal_posts_context_v2",
        "posts_count": len(posts_payload),
        "image_count": len(image_inputs) if not drop_raw_images else 0,
        "image_post_count": len(post_image_url_groups) if not drop_raw_images else 0,
        "image_text_count": len(image_text_inputs),
        "image_text_post_count": sum(1 for row in posts_payload if list(row.get("image_texts") or [])),
        "posts": posts_payload,
        "image_inputs": image_inputs if not drop_raw_images else [],
        "image_text_inputs": image_text_inputs,
    }
    return PostsContextBundle(
        payload=payload,
        text=render_posts_context_payload(payload),
        image_urls=image_urls if not drop_raw_images else [],
        one_image_per_post_urls=one_image_per_post_urls if not drop_raw_images else [],
        post_image_url_groups=post_image_url_groups if not drop_raw_images else [],
    )


def _image_text_value(row: Any) -> str:
    if row is None:
        return ""
    if isinstance(row, dict):
        return " ".join(str(row.get("image_text") or row.get("summary") or "").split()).strip()
    summary = getattr(row, "summary", None)
    if summary is not None:
        return " ".join(str(summary).split()).strip()
    return " ".join(str(row).split()).strip()


def _normalize_prompt_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    return text[: max(1, int(limit))]


def _normalize_short_items(value: Any, *, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _normalize_prompt_text(item, limit=item_limit)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def resolve_post_image_url(
    *,
    post: dict[str, Any],
    media: dict[str, Any],
    image_source_preference: str = "local_first",
) -> str:
    posts_dir_raw = str(post.get("_posts_dir") or "").strip()
    posts_dir = Path(posts_dir_raw).resolve() if posts_dir_raw else None
    storage_uri = str(media.get("storage_uri") or "").strip()
    source_url = str(media.get("source_url") or "").strip()
    local_candidate = ""
    remote_candidate = ""

    if posts_dir and storage_uri:
        resolved_local = (posts_dir / storage_uri).resolve()
        if resolved_local.exists() and resolved_local.is_file():
            local_candidate = str(resolved_local)

    if source_url.startswith(("http://", "https://", "gs://", "data:")):
        remote_candidate = source_url

    if str(image_source_preference or "").strip().lower() == "remote_first":
        if remote_candidate:
            return remote_candidate
        if local_candidate:
            return local_candidate
    else:
        if local_candidate:
            return local_candidate
        if remote_candidate:
            return remote_candidate

    if source_url:
        source_local_candidate = Path(source_url).expanduser()
        if source_local_candidate.exists() and source_local_candidate.is_file():
            return str(source_local_candidate.resolve())

    return ""
