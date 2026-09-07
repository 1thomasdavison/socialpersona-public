from __future__ import annotations

import re

from ...dashscope_batch import dashscope_batch_model_support_reason
from ...image_text import VISUAL_MODE_NATIVE, normalize_visual_mode
from .specs import ModelSpec


def _parse_models_arg(models_raw: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[,\s]+", models_raw.strip()):
        token = part.strip()
        if token:
            out.append(token)
    return out


def _enable_bailian_batch_if_requested(model_specs: list[ModelSpec], *, enabled: bool) -> list[ModelSpec]:
    if not enabled:
        return model_specs
    updated: list[ModelSpec] = []
    for spec in model_specs:
        if spec.provider != "bailian":
            updated.append(spec)
            continue
        updated.append(
            ModelSpec(
                name=spec.name,
                provider="bailian_batch",
                multimodal=spec.multimodal,
                context_chars=spec.context_chars,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                base_url=spec.base_url,
                api_key_env=spec.api_key_env,
                api_model=spec.api_model,
            )
        )
    return updated


def _validate_bailian_batch_model_specs(
    model_specs: list[ModelSpec],
    *,
    visual_mode: str = VISUAL_MODE_NATIVE,
) -> None:
    errors: list[str] = []
    normalized_visual_mode = normalize_visual_mode(visual_mode)
    for spec in model_specs:
        if spec.provider != "bailian_batch":
            continue
        support_reason = dashscope_batch_model_support_reason(
            model=spec.api_model or spec.name,
            has_images=bool(spec.multimodal and normalized_visual_mode == VISUAL_MODE_NATIVE),
            base_url=spec.base_url,
        )
        if support_reason:
            errors.append(f"{spec.name}: {support_reason}")
    if errors:
        joined = "\n".join(f"- {item}" for item in errors)
        raise ValueError(f"Unsupported DashScope batch model configuration:\n{joined}")
