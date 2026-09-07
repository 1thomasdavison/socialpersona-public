from __future__ import annotations

import base64
import json
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any
import mimetypes

from pydantic import ValidationError

from .gate import NarrowGate
from .llm_client import OpenAICompatibleChatClient
from .prompts import SINGLE_POST_JSON_SCHEMA_HINT, SINGLE_POST_SYSTEM_PROMPT, build_single_post_user_prompt
from .schemas import GateResult, PostObservable, SinglePostProfile


DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;,]+)?;base64,(?P<data>.+)$", flags=re.DOTALL)
EVIDENCE_MODE_FULL = "full"
EVIDENCE_MODE_VISUAL_ONLY = "visual_only"


def build_image_inline_data(
    *,
    storage_uri: str | None,
    source_url: str | None = None,
    media_base_dir: Path | None = None,
) -> tuple[str, str] | None:
    for candidate in (storage_uri, source_url):
        value = (candidate or "").strip()
        if not value:
            continue
        if value.startswith("data:"):
            parsed = _parse_base64_data_url(value)
            if parsed:
                return parsed
            continue
        if value.startswith(("http://", "https://")):
            continue

        local_path = resolve_local_media_path(value, media_base_dir=media_base_dir)
        if not local_path:
            continue
        try:
            raw = local_path.read_bytes()
        except OSError:
            continue

        mime_type, _ = mimetypes.guess_type(local_path.name)
        if not mime_type:
            mime_type = "application/octet-stream"
        b64 = base64.b64encode(raw).decode("ascii")
        return mime_type, b64
    return None


def build_image_data_url(
    *,
    storage_uri: str | None,
    source_url: str | None = None,
    media_base_dir: Path | None = None,
) -> str | None:
    inline_data = build_image_inline_data(
        storage_uri=storage_uri,
        source_url=source_url,
        media_base_dir=media_base_dir,
    )
    if not inline_data:
        return None
    mime_type, b64 = inline_data
    return f"data:{mime_type};base64,{b64}"


def build_post_image_data_urls(post: PostObservable, *, media_base_dir: Path | None = None) -> list[str]:
    urls: list[str] = []
    for media in post.media:
        if media.media_type != "image":
            continue
        data_url = build_image_data_url(
            storage_uri=media.storage_uri,
            source_url=media.source_url,
            media_base_dir=media_base_dir,
        )
        if data_url:
            urls.append(data_url)
    return urls


def build_post_image_inline_parts(
    post: PostObservable,
    *,
    media_base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for media in post.media:
        if media.media_type != "image":
            continue
        inline_data = build_image_inline_data(
            storage_uri=media.storage_uri,
            source_url=media.source_url,
            media_base_dir=media_base_dir,
        )
        if not inline_data:
            continue
        mime_type, b64 = inline_data
        parts.append({"inlineData": {"mimeType": mime_type, "data": b64}})
    return parts


def resolve_local_media_path(uri: str, *, media_base_dir: Path | None = None) -> Path | None:
    path = Path(uri)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        cwd = Path.cwd()
        candidates.append(cwd / path)
        if media_base_dir:
            candidates.append(media_base_dir / path)
            candidates.append(media_base_dir.parent / path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _parse_base64_data_url(value: str) -> tuple[str, str] | None:
    match = DATA_URL_PATTERN.match(value.strip())
    if not match:
        return None
    mime_type = (match.group("mime") or "application/octet-stream").strip() or "application/octet-stream"
    raw_b64 = (match.group("data") or "").strip()
    if not raw_b64:
        return None
    return mime_type, raw_b64


@dataclass
class SinglePostAnalyzer:
    gate: NarrowGate
    client: OpenAICompatibleChatClient
    media_base_dir: Path | None = None
    allow_remote_image_url: bool = False

    def analyze(
        self,
        post: PostObservable,
        media_summary: dict[str, Any] | None = None,
        *,
        evidence_mode: str = EVIDENCE_MODE_FULL,
    ) -> tuple[GateResult, SinglePostProfile]:
        normalized_mode = str(evidence_mode or EVIDENCE_MODE_FULL).strip().lower() or EVIDENCE_MODE_FULL
        if normalized_mode not in {EVIDENCE_MODE_FULL, EVIDENCE_MODE_VISUAL_ONLY}:
            raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")
        gate_result = self.gate.run(post)
        if gate_result.is_noise:
            return gate_result, SinglePostProfile(
                is_noise=True,
                should_use_for_profile=False,
                noise_reason=gate_result.noise_reason,
                post_signal_strength=0.0,
                domains=[],
                uncertainty_note="Skipped by narrow gate.",
            )

        image_urls = self._extract_image_urls(post)
        if normalized_mode == EVIDENCE_MODE_VISUAL_ONLY and not image_urls:
            return gate_result, SinglePostProfile(
                is_noise=False,
                should_use_for_profile=False,
                noise_reason=None,
                post_signal_strength=0.0,
                domains=[],
                uncertainty_note="Visual-only ablation: no usable image evidence was available for this post.",
            )

        if normalized_mode == EVIDENCE_MODE_VISUAL_ONLY:
            post_package = self._build_visual_only_post_package(
                post=post,
                gate_result=gate_result,
                media_summary=media_summary,
            )
        else:
            post_package = self._build_post_package(post=post, gate_result=gate_result, media_summary=media_summary)

        result = self.client.chat_json(
            system_prompt=SINGLE_POST_SYSTEM_PROMPT,
            user_prompt=build_single_post_user_prompt(json.dumps(post_package, ensure_ascii=False, indent=2)),
            image_urls=image_urls or None,
            json_schema_hint=SINGLE_POST_JSON_SCHEMA_HINT,
        )
        try:
            parsed = SinglePostProfile.model_validate(result)
        except ValidationError as exc:
            raise ValueError(f"Model output did not match SinglePostProfile schema: {exc}\nRaw output: {result}") from exc

        # Hard guardrail: the model cannot contradict the gate on noise.
        if gate_result.has_image and not gate_result.is_meme_image:
            parsed.is_noise = False
            if parsed.post_signal_strength <= 0:
                parsed.post_signal_strength = 0.15
            if parsed.noise_reason == "too_little_content":
                parsed.noise_reason = None

        # Deterministic weighting: non-original posts are down-weighted.
        post_type = (post.post_type or "").strip().lower()
        if post_type != "original":
            parsed.post_signal_strength = parsed.post_signal_strength * 0.8
        parsed.post_signal_strength = max(0.0, min(1.0, parsed.post_signal_strength))
        return gate_result, parsed

    def _build_post_package(self, *, post: PostObservable, gate_result: GateResult, media_summary: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "post": json.loads(post.model_dump_json()),
            "gate_result": json.loads(gate_result.model_dump_json()),
            "media_summary": media_summary,
        }

    def _build_visual_only_post_package(
        self,
        *,
        post: PostObservable,
        gate_result: GateResult,
        media_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "post": self._redact_post_for_visual_only(post),
            "gate_result": self._redact_gate_result_for_visual_only(gate_result),
            "media_summary": media_summary,
            "evidence_mode": EVIDENCE_MODE_VISUAL_ONLY,
            "redaction_note": "All textual evidence has been removed. Interests must be supported by image content only.",
        }

    @staticmethod
    def _redact_post_for_visual_only(post: PostObservable) -> dict[str, Any]:
        payload = json.loads(post.model_dump_json())
        payload["text"] = ""
        payload["hashtags"] = []
        payload["mentions"] = []
        payload["urls"] = []
        for media in payload.get("media", []) or []:
            if not isinstance(media, dict):
                continue
            media["storage_uri"] = None
            media["source_url"] = None
            media["alt_text"] = None
        return payload

    @staticmethod
    def _redact_gate_result_for_visual_only(gate_result: GateResult) -> dict[str, Any]:
        payload = json.loads(gate_result.model_dump_json())
        payload["candidate_domains"] = []
        payload["gate_reason"] = [
            reason
            for reason in list(payload.get("gate_reason") or [])
            if str(reason).strip() in {"has_image", "meme_image"}
        ]
        payload["content_units"] = 0
        return payload

    def _extract_image_urls(self, post: PostObservable) -> list[str]:
        urls = build_post_image_data_urls(post, media_base_dir=self.media_base_dir)
        if urls or not self.allow_remote_image_url:
            return urls

        fallback_urls: list[str] = []
        for media in post.media:
            if media.media_type != "image":
                continue
            source_url = (media.source_url or "").strip()
            if source_url.startswith(("http://", "https://", "data:")):
                fallback_urls.append(source_url)
        return fallback_urls

    def _build_image_data_url(self, storage_uri: str | None) -> str | None:
        return build_image_data_url(
            storage_uri=storage_uri,
            source_url=None,
            media_base_dir=self.media_base_dir,
        )

    def _resolve_local_media_path(self, uri: str) -> Path | None:
        return resolve_local_media_path(uri, media_base_dir=self.media_base_dir)
