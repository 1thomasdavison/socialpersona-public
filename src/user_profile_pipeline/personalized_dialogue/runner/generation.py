from __future__ import annotations

from typing import Any

from ...image_text import VISUAL_MODE_NATIVE, normalize_visual_mode
from ...llm_client import OpenAICompatibleChatClient
from ..evaluator import (
    DIALOGUE_SYSTEM_PROMPT,
    DIALOGUE_SYSTEM_PROMPT_PROFILE_ONLY,
    _render_turn1_prompt,
    _render_turn2_prompt,
)
from .context import (
    _clear_inline_image_caches,
    _prepare_image_urls_for_provider,
    _select_context_and_image_urls,
    _temporary_inline_image_settings,
)


def _generate_dialogue_prediction(
    *,
    client: OpenAICompatibleChatClient,
    eval_task: dict[str, Any],
    model_name: str,
    provider: str,
    max_context_posts: int,
    max_context_chars: int,
    max_images: int,
    one_image_per_post: bool = False,
    lowres_max_dim: int = 0,
    retry_mode: str = "",
    visual_mode: str = VISUAL_MODE_NATIVE,
    context_bundle: Any | None = None,
    context_override: str | None = None,
) -> dict[str, Any]:
    context, image_urls = _select_context_and_image_urls(
        eval_task=eval_task,
        max_posts=max_context_posts,
        max_chars=max_context_chars,
        max_images=max_images,
        one_image_per_post=one_image_per_post,
        context_bundle=context_bundle,
    )
    if context_override is not None:
        context = str(context_override)
    ec = (eval_task.get("context") or {}) if isinstance(eval_task.get("context"), dict) else {}
    _profile_only = bool(ec.get("profile_only_mode"))
    system_prompt = DIALOGUE_SYSTEM_PROMPT_PROFILE_ONLY if _profile_only else DIALOGUE_SYSTEM_PROMPT
    _clear_inline_image_caches()
    try:
        with _temporary_inline_image_settings(
            max_dim=(lowres_max_dim if lowres_max_dim > 0 else None),
            reencode=(lowres_max_dim > 0),
        ):
            if _profile_only:
                prepared_image_urls: list[str] = []
            else:
                prepared_image_urls = _prepare_image_urls_for_provider(image_urls, provider=provider)
        turn1_response = client.chat_text(
            system_prompt=system_prompt,
            user_prompt=_render_turn1_prompt(eval_task=eval_task, context=context),
            image_urls=prepared_image_urls or None,
        )
        turn2_response = client.chat_text(
            system_prompt=system_prompt,
            user_prompt=_render_turn2_prompt(
                eval_task=eval_task,
                context=context,
                turn1_response=turn1_response,
            ),
            image_urls=prepared_image_urls or None,
        )
        prediction = {
            "schema_version": "personalized_dialogue_prediction_v1",
            "task_id": eval_task.get("task_id"),
            "user_id": eval_task.get("user_id"),
            "model": model_name,
            "visual_mode": normalize_visual_mode(visual_mode),
            "generation_error": "",
            "dialogue": [
                {
                    "turn_id": 1,
                    "user_prompt": str(((eval_task.get("prompt_template") or {}).get("turn1_user_prompt")) or ""),
                    "assistant_response": turn1_response,
                },
                {
                    "turn_id": 2,
                    "user_prompt": str(((eval_task.get("prompt_template") or {}).get("turn2_user_prompt")) or ""),
                    "assistant_response": turn2_response,
                },
            ],
        }
        if retry_mode:
            prediction["retry_mode"] = retry_mode
        return prediction
    finally:
        _clear_inline_image_caches()
