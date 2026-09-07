from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DOMAIN_NAMES = [
    "sports_outdoor",
    "entertainment",
    "gaming",
    "food_drink",
    "travel_city_exploration",
    "photography_creation",
    "pets",
]

TAG_TYPES = [
    "activity",
    "preference",
    "subject",
    "place",
    "object",
    "routine",
    "relation",
]

NOISE_REASONS = [
    "too_little_content",
    "meme_image_only",
]

EVIDENCE_MODALITIES = [
    "text",
    "visual",
]


class MediaItem(BaseModel):
    media_id: str
    media_type: Literal["image", "video", "gif", "unknown"] = "unknown"
    storage_uri: str | None = None
    source_url: str | None = None
    width: int | None = None
    height: int | None = None
    alt_text: str | None = None
    sha256: str | None = None


class VisibleMeta(BaseModel):
    is_pinned: bool | None = None
    has_link_preview: bool | None = None


class IngestMeta(BaseModel):
    crawl_batch_id: str | None = None
    source_user_hash: str | None = None


class PostObservable(BaseModel):
    schema_version: str = "post_observable_v1"
    user_id: str
    post_id: str
    platform: str
    created_at: datetime
    post_type: str
    language: str | None = None
    text: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    reply_to_post_id: str | None = None
    quote_post_id: str | None = None
    media: list[MediaItem] = Field(default_factory=list)
    visible_meta: VisibleMeta | None = None
    ingest_meta: IngestMeta | None = None


class GateResult(BaseModel):
    schema_version: str = "gate_result_v2"
    post_id: str
    is_noise: bool
    should_send_to_ai: bool
    noise_reason: Literal["too_little_content", "meme_image_only"] | None = None
    has_image: bool
    is_meme_image: bool
    content_units: int
    gate_reason: list[str] = Field(default_factory=list)
    candidate_domains: list[str] = Field(default_factory=list)

    @field_validator("candidate_domains")
    @classmethod
    def validate_domains(cls, values: list[str]) -> list[str]:
        for value in values:
            if value not in DOMAIN_NAMES:
                raise ValueError(f"Invalid domain: {value}")
        return values


class TagResult(BaseModel):
    tag: str
    confidence: float
    tag_type: Literal[
        "activity",
        "preference",
        "subject",
        "place",
        "object",
        "routine",
        "relation",
    ]


class EvidenceItem(BaseModel):
    source: str
    value: str

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return normalize_evidence_source(value)


class DomainResult(BaseModel):
    domain: Literal[
        "sports_outdoor",
        "entertainment",
        "gaming",
        "food_drink",
        "travel_city_exploration",
        "photography_creation",
        "pets",
    ]
    confidence: float
    evidence: list[EvidenceItem]
    evidence_modalities: list[Literal["text", "visual"]] = Field(default_factory=list)
    evidence_source_breakdown: dict[str, int] = Field(default_factory=dict)
    tags: list[TagResult]

    @model_validator(mode="after")
    def populate_evidence_metadata(self) -> "DomainResult":
        breakdown = summarize_evidence_sources(self.evidence)
        modalities = [modality for modality in EVIDENCE_MODALITIES if breakdown.get(modality, 0) > 0]
        if not self.evidence_modalities:
            self.evidence_modalities = modalities
        else:
            self.evidence_modalities = [
                modality for modality in EVIDENCE_MODALITIES if modality in set(self.evidence_modalities)
            ]
        if not self.evidence_source_breakdown:
            self.evidence_source_breakdown = breakdown
        else:
            self.evidence_source_breakdown = {
                modality: int(self.evidence_source_breakdown.get(modality, 0) or 0)
                for modality in EVIDENCE_MODALITIES
                if int(self.evidence_source_breakdown.get(modality, 0) or 0) > 0
            }
        return self


class SinglePostProfile(BaseModel):
    schema_version: str = "single_post_profile_v1"
    is_noise: bool
    should_use_for_profile: bool
    noise_reason: Literal["too_little_content", "meme_image_only"] | None = None
    post_signal_strength: float
    domains: list[DomainResult] = Field(default_factory=list)
    uncertainty_note: str | None = None


class MediaObservation(BaseModel):
    media_id: str
    summary: str
    candidate_domains: list[str] = Field(default_factory=list)
    confidence: float

    @field_validator("candidate_domains")
    @classmethod
    def validate_candidate_domains(cls, values: list[str]) -> list[str]:
        for value in values:
            if value not in DOMAIN_NAMES:
                raise ValueError(f"Invalid domain: {value}")
        return values


class MediaSummary(BaseModel):
    schema_version: str = "media_summary_v1"
    media_observations: list[MediaObservation] = Field(default_factory=list)


def normalize_evidence_source(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "text"
    if text in {"visual", "image", "photo", "picture", "frame", "screenshot", "media", "ocr"}:
        return "visual"
    if text in {"text", "caption", "hashtags", "hashtag", "mention", "mentions", "url", "urls"}:
        return "text"
    if any(token in text for token in ("image", "visual", "photo", "picture", "frame", "screenshot", "media")):
        return "visual"
    if any(token in text for token in ("text", "caption", "hashtag", "mention", "url", "bio")):
        return "text"
    return "text"


def summarize_evidence_sources(evidence: list[EvidenceItem] | None) -> dict[str, int]:
    counts = {modality: 0 for modality in EVIDENCE_MODALITIES}
    for item in evidence or []:
        modality = normalize_evidence_source(getattr(item, "source", None))
        counts[modality] += 1
    return {modality: count for modality, count in counts.items() if count > 0}
