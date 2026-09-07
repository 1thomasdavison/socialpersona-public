from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HIERARCHICAL_PROFILE_METHOD = "hierarchical"
EXTRACTIVE_PROFILE_METHOD = "extractive"
DIRECT_PROFILE_METHOD = "direct"
SUPPORTED_PROFILE_METHODS = {
    DIRECT_PROFILE_METHOD,
    HIERARCHICAL_PROFILE_METHOD,
    EXTRACTIVE_PROFILE_METHOD,
}
USER_LEVEL_PROFILE_METHODS = {
    HIERARCHICAL_PROFILE_METHOD,
    EXTRACTIVE_PROFILE_METHOD,
}

RETRY_CONTEXT_VERSION = "preserve_img_text_timestamp_v1"
PROFILE_INPUT_MODE_TEXT_ONLY = "text_only"
PROFILE_INPUT_MODE_IMAGE_CAPTIONS_ONLY = "image_captions_only"
PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS = "text_image_captions"
PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS = "text_image_captions_timestamps"
SUPPORTED_PROFILE_INPUT_MODES = {
    PROFILE_INPUT_MODE_TEXT_ONLY,
    PROFILE_INPUT_MODE_IMAGE_CAPTIONS_ONLY,
    PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS,
    PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS,
}


@dataclass
class ModelSpec:
    name: str
    provider: str
    multimodal: bool = False
    context_chars: int = 120_000
    temperature: float = 0.1
    max_tokens: int = 1400
    base_url: str = ""
    api_key_env: str = ""
    api_model: str = ""


@dataclass
class InterestAnchor:
    label: str
    evidence_post_indices: list[int] = field(default_factory=list)


@dataclass
class DomainEvalTask:
    task_id: str
    source_tag: str
    user_id: str
    domain: str
    domain_definition: str
    gold_status: str
    gold_summary_natural: str
    gold_long_term_anchors: list[InterestAnchor]
    gold_short_term_anchors: list[InterestAnchor]
    gold_domain_representative_evidence_post_indices: list[int]
    posts: list[dict[str, Any]]
    gold_path: str
    debug_meta: dict[str, Any] = field(default_factory=dict)


DEFAULT_MODEL_SPECS: dict[str, ModelSpec] = {
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        provider="chatanywhere",
        multimodal=True,
        context_chars=220_000,
        base_url="https://api.chatanywhere.tech/v1/chat/completions",
        api_key_env="CHATANYWHERE_API_KEY",
        api_model="gemini-2.5-flash",
    ),
    "gpt-o3": ModelSpec(
        name="gpt-o3",
        provider="chatanywhere",
        multimodal=True,
        context_chars=120_000,
        base_url="https://aifast.site/v1/chat/completions",
        api_key_env="AIFAST_KEY",
        api_model="o4-mini",
    ),
    "gpt-4o-mini": ModelSpec(
        name="gpt-4o-mini",
        provider="chatanywhere",
        multimodal=True,
        context_chars=120_000,
        base_url="https://aifast.site/v1/chat/completions",
        api_key_env="AIFAST_KEY",
        api_model="gpt-4o-mini",
    ),
    "gpt-5.4": ModelSpec(
        name="gpt-5.4",
        provider="chatanywhere",
        multimodal=True,
        context_chars=120_000,
        base_url="https://aifast.site/v1/chat/completions",
        api_key_env="AIFAST_KEY",
        api_model="gpt-5.4",
    ),
    "gpt-5.5": ModelSpec(
        name="gpt-5.5",
        provider="chatanywhere",
        multimodal=True,
        context_chars=120_000,
        base_url="https://aifast.site/v1/chat/completions",
        api_key_env="AIFAST_KEY",
        api_model="gpt-5.5",
    ),
    "qwen3.5-35b-a3b": ModelSpec(
        name="qwen3.5-35b-a3b",
        provider="bailian",
        multimodal=True,
        context_chars=120_000,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        api_key_env="ALI_BAILIAN_1",
        api_model="qwen3.5-35b-a3b",
    ),
    "qwen2.5-vl-7b-instruct": ModelSpec(
        name="qwen2.5-vl-7b-instruct",
        provider="bailian",
        multimodal=True,
        context_chars=120_000,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        api_key_env="ALI_BAILIAN_1",
        api_model="qwen2.5-vl-7b-instruct",
    ),
    "qwen3.7-max": ModelSpec(
        name="qwen3.7-max",
        provider="bailian",
        multimodal=False,
        context_chars=120_000,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        api_key_env="ALI_BAILIAN_1",
        api_model="qwen3.7-max",
    ),
    "qwen3-vl-8b-instruct": ModelSpec(
        name="qwen3-vl-8b-instruct",
        provider="bailian",
        multimodal=True,
        context_chars=120_000,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        api_key_env="ALI_BAILIAN_1",
        api_model="qwen3-vl-8b-instruct",
    ),
    "mock_oracle": ModelSpec(
        name="mock_oracle",
        provider="mock_oracle",
        multimodal=False,
    ),
}
