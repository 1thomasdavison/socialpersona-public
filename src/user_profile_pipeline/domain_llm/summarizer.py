from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm_client import OpenAICompatibleChatClient
from .prompts import (
    DOMAIN_EVIDENCE_FIRST_JSON_SCHEMA_HINT,
    DOMAIN_EVIDENCE_FIRST_SYSTEM_PROMPT,
    DOMAIN_LLM_JSON_SCHEMA_HINT,
    DOMAIN_LLM_SYSTEM_PROMPT,
    build_domain_evidence_first_user_prompt,
    build_domain_llm_user_prompt,
)
from .validator import DomainSummaryValidator


DEFAULT_GENERATION_GOAL: dict[str, Any] = {
    "profile_type": "interest",
    "natural_language_priority": True,
    "exclude_attributes": ["family", "career", "age_range"],
    "downstream_use": "llm_personalization_benchmark",
}


@dataclass
class DomainLLMSummarizer:
    client: OpenAICompatibleChatClient
    validator: DomainSummaryValidator = field(default_factory=DomainSummaryValidator)
    media_base_dir: Path | None = None
    max_images: int = 12

    def summarize_domain_pack(self, domain_pack: dict[str, Any]) -> dict[str, Any]:
        image_urls = self._extract_image_urls(domain_pack)
        generation_goal = domain_pack.get("profiling_target") or DEFAULT_GENERATION_GOAL
        evidence_pack = {
            "domain": domain_pack.get("domain"),
            "domain_definition": domain_pack.get("domain_definition"),
            "profiling_target": generation_goal,
            "observation_window": domain_pack.get("observation_window"),
            "tag_clusters": domain_pack.get("tag_clusters"),
            "representative_posts": domain_pack.get("representative_posts"),
            "chunk_summaries": domain_pack.get("chunk_summaries"),
        }
        pass1 = self.client.chat_json(
            system_prompt=DOMAIN_EVIDENCE_FIRST_SYSTEM_PROMPT,
            user_prompt=build_domain_evidence_first_user_prompt(json.dumps(evidence_pack, ensure_ascii=False, indent=2)),
            image_urls=image_urls or None,
            json_schema_hint=DOMAIN_EVIDENCE_FIRST_JSON_SCHEMA_HINT,
        )
        final_input = {
            "generation_goal": generation_goal,
            "domain_pack": domain_pack,
            "evidence_first_result": pass1,
        }
        result = self.client.chat_json(
            system_prompt=DOMAIN_LLM_SYSTEM_PROMPT,
            user_prompt=build_domain_llm_user_prompt(json.dumps(final_input, ensure_ascii=False, indent=2)),
            image_urls=image_urls or None,
            json_schema_hint=DOMAIN_LLM_JSON_SCHEMA_HINT,
        )
        return self.validator.validate(result, domain_pack, evidence_first_result=pass1)

    def summarize_many(self, domain_packs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pack in domain_packs:
            results.append(self.summarize_domain_pack(pack))
        return results

    def _extract_image_urls(self, domain_pack: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for row in domain_pack.get("representative_posts", []) or []:
            for media in row.get("media", []) or []:
                if str(media.get("media_type", "")).lower() != "image":
                    continue
                encoded = self._build_image_data_url(media.get("storage_uri"))
                if encoded:
                    urls.append(encoded)
                else:
                    source_url = str(media.get("source_url", "")).strip()
                    if source_url.startswith(("http://", "https://", "data:")):
                        urls.append(source_url)
                if len(urls) >= self.max_images:
                    return urls
        return urls

    def _build_image_data_url(self, storage_uri: str | None) -> str | None:
        if not storage_uri:
            return None
        uri = storage_uri.strip()
        if not uri:
            return None
        if uri.startswith("data:"):
            return uri
        if uri.startswith(("http://", "https://")):
            return None
        local_path = self._resolve_local_media_path(uri)
        if not local_path:
            return None
        try:
            raw = local_path.read_bytes()
        except OSError:
            return None
        mime_type, _ = mimetypes.guess_type(local_path.name)
        if not mime_type:
            mime_type = "application/octet-stream"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    def _resolve_local_media_path(self, uri: str) -> Path | None:
        path = Path(uri)
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            cwd = Path.cwd()
            candidates.append(cwd / path)
            if self.media_base_dir:
                candidates.append(self.media_base_dir / path)
                candidates.append(self.media_base_dir.parent / path)
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None
