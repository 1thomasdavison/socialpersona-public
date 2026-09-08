from __future__ import annotations

import re
from typing import Any

from ...image_text import VISUAL_MODE_TEXT_IMAGE, normalize_visual_mode
from .specs import (
    DIRECT_PROFILE_METHOD,
    PROFILE_INPUT_MODE_IMAGE_CAPTIONS_ONLY,
    PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS,
    PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS,
    PROFILE_INPUT_MODE_TEXT_ONLY,
    SUPPORTED_PROFILE_INPUT_MODES,
    SUPPORTED_PROFILE_METHODS,
)


def _slugify(value: str) -> str:
    lowered = (value or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", lowered)
    return cleaned.strip("_") or "model"


def normalize_profile_input_mode(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS
    if raw not in SUPPORTED_PROFILE_INPUT_MODES:
        raise ValueError(
            f"Unsupported profile input mode: {value}. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_PROFILE_INPUT_MODES))}"
        )
    return raw


def normalize_profile_method(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return DIRECT_PROFILE_METHOD
    if raw not in SUPPORTED_PROFILE_METHODS:
        raise ValueError(
            f"Unsupported profile method: {value}. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_PROFILE_METHODS))}"
        )
    return raw


def _profile_input_mode_uses_image_captions(profile_input_mode: str) -> bool:
    return normalize_profile_input_mode(profile_input_mode) in {
        PROFILE_INPUT_MODE_IMAGE_CAPTIONS_ONLY,
        PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS,
        PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS,
    }


def _profile_input_mode_cache_key(*, visual_mode: str, profile_input_mode: str) -> str:
    visual = normalize_visual_mode(visual_mode)
    input_mode = normalize_profile_input_mode(profile_input_mode)
    if visual == VISUAL_MODE_TEXT_IMAGE and input_mode == PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS:
        return visual
    return f"{visual}__{input_mode}"


def _prediction_profile_input_mode(row: dict[str, Any]) -> str:
    return normalize_profile_input_mode(str(row["profile_input_mode"]))


def _prediction_profile_method(row: dict[str, Any]) -> str:
    return normalize_profile_method(str(row["profile_method"]))


def _posts_for_profile_input_mode(posts: list[dict[str, Any]], *, profile_input_mode: str) -> list[dict[str, Any]]:
    mode = normalize_profile_input_mode(profile_input_mode)
    if mode == PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS:
        return posts

    keep_text = mode in {PROFILE_INPUT_MODE_TEXT_ONLY, PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS}
    out: list[dict[str, Any]] = []
    for post in posts:
        row = dict(post)
        row["created_at"] = ""
        if not keep_text:
            row["text"] = ""
            row["hashtags"] = []
            row["mentions"] = []
            row["urls"] = []
            row["language"] = None
        out.append(row)
    return out
