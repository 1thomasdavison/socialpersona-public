from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

from ...benchmark.profile_eval.io import _load_posts_with_indices
from ...benchmark.profile_eval.specs import DEFAULT_MODEL_SPECS
from ...image_text import VISUAL_MODE_NATIVE, normalize_visual_mode, prediction_visual_mode
from ...llm_client import OpenAICompatibleChatClient
from .. import build_profile_context_from_gold_export


DIALOGUE_JUDGE_SYSTEM_PROMPT = """You are an expert judge for social-media-grounded personalized dialogue.

You will be given two independent recommendation cases, the user's gold profile, and the model responses.
The user requests are intentionally natural; do not require the model to mention post evidence or profile labels explicitly.

Important principles:
1. Reward semantic fit to the correct target interests, not exact wording similarity.
2. For stable_recommendation, reward use of long-term interests.
3. For short_term_exploration, reward use of short-term interests while keeping the suggestion compatible with stable interests.
4. Penalize recommendations involving negative, weak, noisy, or avoid-worthy interests.
5. Reward concrete, actionable, natural recommendations.
6. Penalize generic filler, unsupported assumptions, demographic guesses, and benchmark-like language.
7. The two cases are independent; do not require dialogue continuity across them.
8. Return strict JSON only.

Score each dimension from 0 to 5 using the rubrics below.

RUBRIC: interest_coverage
Whether the response engages the correct target interests (long-term for stable_recommendation, short-term for short_term_exploration) in this scenario.
- 0: No target interest is engaged; response is generic or irrelevant.
- 1: Tangential mention of a target interest but not used as the basis of the recommendation.
- 2: One target interest is partially engaged but the recommendation does not clearly center on it.
- 3: One target interest is clearly and centrally used as the basis of the recommendation.
- 4: Multiple target interests are used naturally, or one interest is used with specific supporting detail.
- 5: Rich, precise engagement with the correct target interests; the recommendation feels tailored to this specific user.

RUBRIC: negative_avoidance
Whether the response avoids recommending things linked to negative, weak, noisy, or avoid-worthy interests.
- 0: The recommendation explicitly endorses a negative/forbidden interest (e.g., suggests something the user dislikes).
- 1: The response is ambiguous about a negative interest; a reasonable reader could infer the recommendation includes it.
- 2: The response mentions a negative interest neutrally without endorsing it.
- 3: No negative interest is mentioned or implied; the recommendation stays in safe territory.
- 4: The response actively steers around a known negative interest while still being helpful.
- 5: The response shows clear awareness of what to avoid and makes a confident, clean recommendation in positive territory.

RUBRIC: concreteness
Whether the recommendation is specific, actionable, and natural rather than generic or vague.
- 0: Entirely generic; could apply to any user (e.g., "try something you enjoy").
- 1: A vague suggestion with no actionable detail.
- 2: Some concrete element is present but the recommendation remains broad.
- 3: A clear, actionable recommendation with at least one specific detail (e.g., a genre, activity, item, or place).
- 4: A specific recommendation with supporting context that makes it easy to act on.
- 5: A vivid, naturally-phrased recommendation with precise detail that feels like a human friend suggested it.

RUBRIC: fluency
Whether the response is well-formed, natural, and free of benchmark artifacts (e.g., scoring language, numbered lists, meta-commentary).
- 0: Incoherent or empty response.
- 1: Contains obvious benchmark artifacts (e.g., "turn 1:", "post index:", "score: 8/10", JSON remnants).
- 2: Awkward phrasing or overly structured language that does not read as natural dialogue.
- 3: Natural and readable but slightly stiff or formulaic.
- 4: Fluid and natural; reads like a real conversational assistant.
- 5: Fully natural, polished, and appropriate to the context; indistinguishable from human-written recommendation.
"""


DIALOGUE_JUDGE_SCHEMA_HINT = """{
  "task_id": "copy the task_id string",
  "interest_coverage": "number from 0 to 5",
  "negative_avoidance": "number from 0 to 5",
  "concreteness": "number from 0 to 5",
  "fluency": "number from 0 to 5",
  "matched_positive_interests": ["interest label strings that are supported by the response"],
  "matched_negative_interests": ["negative/forbidden interest label strings used by the response"],
  "unsupported_interests": ["response claims that are not supported by the gold profile"],
  "missed_core_interest_labels": ["required or important gold labels the response missed"],
  "brief_rationale": "one concise sentence explaining the score",
  "judge_total_score": "optional number from 0 to 20"
}"""
DIALOGUE_EVAL_PROMPT_CONTEXT_VERSION = "split_turn_single_request_no_intent_v1"
DIALOGUE_JUDGE_PROMPT_VERSION = "dialogue_judge_v7_required_scores_score_line_fallback"
REQUIRED_JUDGE_SCORE_KEYS = (
    "interest_coverage",
    "negative_avoidance",
    "concreteness",
    "fluency",
)


class JudgePlaceholderError(ValueError):
    pass

def _build_chat_client_for_spec(spec: Any, *, timeout_seconds: int, max_tokens: int) -> OpenAICompatibleChatClient:
    provider = str(spec.provider or "").strip().lower()
    if provider in {"chatanywhere", "bailian"}:
        normalized_provider = "openai_compatible"
    elif provider.startswith("vertex"):
        normalized_provider = "vertex"
    else:
        normalized_provider = provider or "openai_compatible"
    return OpenAICompatibleChatClient(
        provider=normalized_provider,
        base_url=str(spec.base_url or ""),
        model=str(spec.api_model or spec.name),
        api_key_env=str(spec.api_key_env or ""),
        timeout_seconds=timeout_seconds,
        temperature=float(spec.temperature),
        max_tokens=max_tokens,
        repair_model=str(spec.api_model or spec.name),
    )


def _build_judge_client(
    *,
    judge_model: str,
    judge_provider: str,
    judge_base_url: str,
    judge_api_key_env: str,
    judge_api_model: str,
    judge_timeout_seconds: int,
    judge_max_tokens: int,
) -> OpenAICompatibleChatClient:
    spec = DEFAULT_MODEL_SPECS.get(judge_model)
    if spec is None:
        raise ValueError(f"Unknown judge model: {judge_model}. Add it to DEFAULT_MODEL_SPECS first.")
    provider = (judge_provider or spec.provider).strip().lower()
    if provider in {"chatanywhere", "bailian"}:
        normalized_provider = "openai_compatible"
    elif provider.startswith("vertex"):
        normalized_provider = "vertex"
    else:
        normalized_provider = provider
    base_url = judge_base_url or spec.base_url
    if not base_url and normalized_provider == "openai_compatible":
        base_url = "https://api.chatanywhere.tech/v1/chat/completions"
    if not base_url and normalized_provider == "vertex":
        base_url = "https://aiplatform.googleapis.com/v1"
    api_key_env = judge_api_key_env or spec.api_key_env
    api_model = judge_api_model or spec.api_model or judge_model
    repair_model = (
        "gemini-3-flash-preview"
        if normalized_provider == "vertex"
        else str(os.environ.get("DIALOGUE_JUDGE_REPAIR_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
    )
    return OpenAICompatibleChatClient(
        provider=normalized_provider,
        base_url=base_url,
        model=api_model,
        api_key_env=api_key_env,
        timeout_seconds=judge_timeout_seconds,
        temperature=0.0,
        max_tokens=judge_max_tokens,
        repair_model=repair_model,
    )


def _build_judge_client_from_args(args: argparse.Namespace) -> OpenAICompatibleChatClient:
    return _build_judge_client(
        judge_model=str(args.judge_model or "gpt-o3").strip() or "gpt-o3",
        judge_provider=str(args.judge_provider or "").strip(),
        judge_base_url=str(args.judge_base_url or "").strip(),
        judge_api_key_env=str(args.judge_api_key_env or "").strip(),
        judge_api_model=str(args.judge_api_model or "").strip(),
        judge_timeout_seconds=int(args.judge_timeout_seconds),
        judge_max_tokens=int(args.judge_max_tokens),
    )


def _build_flash_judge_fallback_client_from_args(args: argparse.Namespace) -> OpenAICompatibleChatClient:
    fallback_model = str(os.environ.get("DIALOGUE_JUDGE_FALLBACK_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
    return _build_judge_client(
        judge_model=str(args.judge_model or "gpt-o3").strip() or "gpt-o3",
        judge_provider=str(args.judge_provider or "").strip(),
        judge_base_url=str(args.judge_base_url or "").strip(),
        judge_api_key_env=str(args.judge_api_key_env or "").strip(),
        judge_api_model=fallback_model,
        judge_timeout_seconds=int(args.judge_timeout_seconds),
        judge_max_tokens=int(args.judge_max_tokens),
    )


def _extract_interest_evidence(gold_ref: dict[str, Any]) -> list[dict[str, Any]]:
    posts_path = Path(str(((gold_ref.get("source") or {}).get("posts_path")) or ""))
    evidence_by_index: dict[int, dict[str, Any]] = {}
    if posts_path.exists():
        for row in _load_posts_with_indices(posts_path):
            idx = int(row.get("_line_index", -1))
            if idx >= 0:
                evidence_by_index[idx] = row

    out: list[dict[str, Any]] = []
    for fact in gold_ref.get("selected_interest_facts", []) or []:
        snippets: list[dict[str, Any]] = []
        for idx in fact.get("evidence_post_indices", []) or []:
            row = evidence_by_index.get(int(idx))
            if row is None:
                continue
            snippets.append(
                {
                    "post_index": int(idx),
                    "post_id": str(row.get("post_id") or ""),
                    "text": str(row.get("text") or "").strip(),
                    "hashtags": list(row.get("hashtags") or []),
                }
            )
        out.append(
            {
                "domain": str(fact.get("domain") or ""),
                "label": str(fact.get("label") or ""),
                "horizon": str(fact.get("horizon") or ""),
                "evidence_post_indices": list(fact.get("evidence_post_indices") or []),
                "evidence_snippets": snippets[:2],
            }
        )
    return out


def _extract_profile_context(gold_ref: dict[str, Any]) -> dict[str, Any]:
    existing = gold_ref.get("profile_context")
    if isinstance(existing, dict) and existing:
        return existing

    gold_export_path = Path(str(((gold_ref.get("source") or {}).get("gold_export_path")) or "")).resolve()
    if not gold_export_path.exists():
        return {
            "schema_version": "personalized_dialogue_profile_context_v1",
            "user_id": str(gold_ref.get("user_id") or ""),
            "active_domains": [],
            "top_interest_labels": [],
        }
    try:
        gold_export = json.loads(gold_export_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": "personalized_dialogue_profile_context_v1",
            "user_id": str(gold_ref.get("user_id") or ""),
            "active_domains": [],
            "top_interest_labels": [],
        }
    return build_profile_context_from_gold_export(gold_export=gold_export)


def _build_judge_input(
    *,
    gold_ref: dict[str, Any],
    eval_task: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    prompt_template = eval_task.get("prompt_template", {}) or {}
    return {
        "task_id": str(gold_ref.get("task_id") or ""),
        "user_id": str(gold_ref.get("user_id") or ""),
        "prompt_template": {
            "turn1_user_prompt": str(prompt_template.get("turn1_user_prompt") or ""),
            "turn2_user_prompt": str(prompt_template.get("turn2_user_prompt") or ""),
        },
        "gold_profile_context": _extract_profile_context(gold_ref),
        "gold_interest_facts": _extract_interest_evidence(gold_ref),
        "gold_reference_dialogue": gold_ref.get("reference_dialogue") or [],
        "prediction_dialogue": prediction.get("dialogue") or [],
    }


def _score_0_to_5(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def _weighted_dialogue_score(
    *,
    interest_coverage: float,
    negative_avoidance: float,
    concreteness: float,
    fluency: float,
) -> float:
    return (
        0.40 * interest_coverage
        + 0.25 * negative_avoidance
        + 0.20 * concreteness
        + 0.15 * fluency
    ) / 5.0 * 100.0


def _exception_text(exc: Exception) -> str:
    text = f"{exc.__class__.__name__}: {exc}".strip()
    try:
        from tenacity import RetryError  # type: ignore

        if isinstance(exc, RetryError):
            inner = exc.last_attempt.exception()
            if inner is not None:
                text = f"{text}; last_exception={inner}"
    except Exception:
        pass
    return text


def _prediction_file_matches_visual_mode(path: Path, visual_mode: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if prediction_visual_mode(payload) != normalize_visual_mode(visual_mode):
            return False
    except Exception:
        return False
    if str(payload.get("generation_error") or "").strip():
        return False
    return True


def _judge_file_is_cacheable(path: Path, *, expected_judge_model: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(payload.get("judge_prompt_version") or "").strip() != DIALOGUE_JUDGE_PROMPT_VERSION:
        return False
    if str(expected_judge_model or "").strip():
        if str(payload.get("judge_model") or "").strip() != str(expected_judge_model or "").strip():
            return False
    if str(payload.get("judge_error") or "").strip():
        return False
    return not _judge_payload_looks_like_empty_placeholder(payload)


def _stamp_judge_result(
    payload: dict[str, Any],
    *,
    judge_model: str,
    judge_provider: str,
    judge_api_model: str,
) -> dict[str, Any]:
    payload["judge_model"] = str(judge_model or "").strip()
    payload["judge_provider"] = str(judge_provider or "").strip()
    payload["judge_api_model"] = str(judge_api_model or judge_model or "").strip()
    payload["judge_prompt_version"] = DIALOGUE_JUDGE_PROMPT_VERSION
    return payload


def _judge_payload_looks_like_empty_placeholder(payload: dict[str, Any]) -> bool:
    scores = [
        _score_0_to_5(payload.get("interest_coverage")),
        _score_0_to_5(payload.get("negative_avoidance")),
        _score_0_to_5(payload.get("concreteness")),
        _score_0_to_5(payload.get("fluency")),
    ]
    if any(score != 0.0 for score in scores):
        return False
    if str(payload.get("brief_rationale") or "").strip():
        return False
    for key in [
        "matched_positive_interests",
        "matched_negative_interests",
        "unsupported_interests",
        "missed_core_interest_labels",
    ]:
        if payload.get(key):
            return False
    return True


def _looks_like_context_limit_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    return any(
        needle in text
        for needle in [
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "prompt is too long",
            "input is too large",
            "input length",
            "range of input length",
            "request too large",
            "too many images",
        ]
    )


def _looks_like_transport_timeout_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    return any(
        needle in text
        for needle in [
            "write operation timed out",
            "connection aborted",
            "read timed out",
            "timeouterror",
            "connectionerror",
        ]
    )


def _is_provider_block_message(message: str) -> bool:
    text = str(message or "").lower()
    return any(
        needle in text
        for needle in [
            "data_inspection_failed",
            "inappropriate content",
            "content_policy",
            "safety",
            "moderation",
        ]
    )


def _fallback_prediction(
    *,
    eval_task: dict[str, Any],
    model_name: str,
    error: str,
    visual_mode: str = VISUAL_MODE_NATIVE,
) -> dict[str, Any]:
    prompts = eval_task.get("prompt_template", {}) or {}
    return {
        "schema_version": "personalized_dialogue_prediction_v1",
        "task_id": eval_task.get("task_id"),
        "user_id": eval_task.get("user_id"),
        "model": model_name,
        "visual_mode": normalize_visual_mode(visual_mode),
        "generation_error": error,
        "dialogue": [
            {
                "turn_id": 1,
                "user_prompt": str(prompts.get("turn1_user_prompt") or ""),
                "assistant_response": "",
            },
            {
                "turn_id": 2,
                "user_prompt": str(prompts.get("turn2_user_prompt") or ""),
                "assistant_response": "",
            },
        ],
    }


def _fallback_judge_result(
    *,
    gold_ref: dict[str, Any],
    error: str,
    brief_rationale: str,
) -> dict[str, Any]:
    return {
        "task_id": str(gold_ref.get("task_id") or ""),
        "user_id": str(gold_ref.get("user_id") or ""),
        "interest_coverage": 0.0,
        "negative_avoidance": 0.0,
        "concreteness": 0.0,
        "fluency": 0.0,
        "matched_positive_interests": [],
        "matched_negative_interests": [],
        "unsupported_interests": [],
        "missed_core_interest_labels": [
            str(item.get("label") or "")
            for item in (gold_ref.get("selected_interest_facts") or [])
            if str(item.get("label") or "").strip()
        ],
        "brief_rationale": brief_rationale,
        "judge_total_score": 0.0,
        "final_dialogue_score": 0.0,
        "judge_prompt_version": DIALOGUE_JUDGE_PROMPT_VERSION,
        "judge_error": error,
        "provider_blocked": _is_provider_block_message(error),
    }


def _judge_prediction(
    *,
    judge_client: OpenAICompatibleChatClient,
    judge_input: dict[str, Any],
    retry_note: str = "",
) -> dict[str, Any]:
    retry_block = f"\n{retry_note.strip()}\n" if str(retry_note or "").strip() else ""
    user_prompt = (
        "Evaluate this personalized dialogue prediction.\n\n"
        "Use the scenario rubrics inside gold_reference_dialogue.\n"
        "Score interest_coverage, negative_avoidance, concreteness, and fluency from 0 to 5.\n"
        "Do not require explicit evidence explanation in the model response.\n\n"
        "Do not copy the schema hint or return placeholder zeros. If all four scores are 0, brief_rationale must explain why.\n"
        f"{retry_block}\n"
        f"JUDGE_INPUT:\n{json.dumps(judge_input, ensure_ascii=False, indent=2)}\n\n"
        "Return strict JSON only."
    )
    raw = judge_client.chat_json(
        system_prompt=DIALOGUE_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_schema_hint=DIALOGUE_JUDGE_SCHEMA_HINT,
    )
    return _normalize_judge_raw(raw=raw, judge_input=judge_input)


def _judge_prediction_text_fallback(
    *,
    judge_client: OpenAICompatibleChatClient,
    judge_input: dict[str, Any],
) -> dict[str, Any]:
    compact_input = _build_compact_judge_input(judge_input)
    user_prompt = (
        "Evaluate this personalized dialogue prediction using the compact rubrics.\n"
        "Return exactly one JSON object. Do not use markdown. Do not copy an empty schema.\n"
        "Scores must be numbers from 0 to 5. If a score is 0, explain why in brief_rationale.\n"
        "Required keys: task_id, interest_coverage, negative_avoidance, concreteness, fluency, "
        "matched_positive_interests, matched_negative_interests, unsupported_interests, "
        "missed_core_interest_labels, brief_rationale, judge_total_score.\n\n"
        f"COMPACT_JUDGE_INPUT:\n{json.dumps(compact_input, ensure_ascii=False, separators=(',', ':'))}\n"
    )
    text = judge_client.chat_text(
        system_prompt=DIALOGUE_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    raw = judge_client._parse_json_candidate(text)
    if raw is None:
        repaired_text = judge_client._repair_json_by_model(raw_text=text, json_schema_hint=DIALOGUE_JUDGE_SCHEMA_HINT)
        raw = judge_client._parse_json_candidate(repaired_text)
    if raw is None:
        raise json.JSONDecodeError("Unable to recover a valid JSON object from text fallback.", text, 0)
    result = _normalize_judge_raw(raw=raw, judge_input=judge_input)
    result["judge_retry_mode"] = "flash_text_placeholder_fallback"
    return result


def _judge_prediction_score_line_fallback(
    *,
    judge_client: OpenAICompatibleChatClient,
    judge_input: dict[str, Any],
) -> dict[str, Any]:
    compact_input = _build_compact_judge_input(judge_input)
    user_prompt = (
        "Score this personalized dialogue prediction. Return one plain text line only, not JSON.\n"
        "Format exactly:\n"
        "interest_coverage=<0-5>; negative_avoidance=<0-5>; concreteness=<0-5>; "
        "fluency=<0-5>; rationale=<short reason>\n\n"
        "Use semantic fit to the gold interests. Penalize unsupported or avoid-worthy interests.\n\n"
        f"COMPACT_JUDGE_INPUT:\n{json.dumps(compact_input, ensure_ascii=False, separators=(',', ':'))}\n"
    )
    text = judge_client.chat_text(
        system_prompt=DIALOGUE_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    matched_score_keys: list[str] = []

    def extract_score(key: str) -> float:
        match = re.search(rf"{re.escape(key)}\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if not match:
            return 0.0
        matched_score_keys.append(key)
        return _score_0_to_5(match.group(1))

    rationale_match = re.search(
        r"(?:rationale|brief_rationale)\s*[:=]\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    rationale = " ".join((rationale_match.group(1) if rationale_match else text).split()).strip()
    if not rationale:
        rationale = "Fallback line judge parsed numeric scores but no rationale was provided."
    raw = {
        "task_id": str(judge_input.get("task_id") or ""),
        "interest_coverage": extract_score("interest_coverage"),
        "negative_avoidance": extract_score("negative_avoidance"),
        "concreteness": extract_score("concreteness"),
        "fluency": extract_score("fluency"),
        "matched_positive_interests": [],
        "matched_negative_interests": [],
        "unsupported_interests": [],
        "missed_core_interest_labels": [],
        "brief_rationale": rationale[:500],
        "judge_total_score": 0.0,
    }
    if len(matched_score_keys) < 4:
        raise json.JSONDecodeError("Unable to parse all score_line fallback fields.", text, 0)
    result = _normalize_judge_raw(raw=raw, judge_input=judge_input)
    result["judge_retry_mode"] = "score_line_text_fallback"
    return result


def _build_provider_safe_judge_input(judge_input: dict[str, Any]) -> dict[str, Any]:
    safe_interest_facts: list[dict[str, Any]] = []
    for item in list(judge_input.get("gold_interest_facts") or []):
        if not isinstance(item, dict):
            continue
        safe_interest_facts.append(
            {
                "domain": str(item.get("domain") or ""),
                "label": str(item.get("label") or ""),
                "horizon": str(item.get("horizon") or ""),
                "evidence_post_indices": list(item.get("evidence_post_indices") or []),
            }
        )

    safe_prediction_dialogue: list[dict[str, Any]] = []
    for turn in list(judge_input.get("prediction_dialogue") or []):
        if not isinstance(turn, dict):
            continue
        safe_prediction_dialogue.append(
            {
                "turn_id": int(turn.get("turn_id") or 0),
                "assistant_response": str(turn.get("assistant_response") or "")[:500],
            }
        )

    return {
        "task_id": str(judge_input.get("task_id") or ""),
        "user_id": str(judge_input.get("user_id") or ""),
        "prompt_template": {
            "turn1_user_prompt": "[redacted_for_provider_safety]",
            "turn2_user_prompt": "[redacted_for_provider_safety]",
        },
        "gold_interest_facts": safe_interest_facts,
        "gold_reference_dialogue": [],
        "prediction_dialogue": safe_prediction_dialogue,
        "judge_retry_mode": "provider_safe_input",
    }


def _build_compact_judge_input(judge_input: dict[str, Any]) -> dict[str, Any]:
    compact_refs: list[dict[str, Any]] = []
    for row in list(judge_input.get("gold_reference_dialogue") or []):
        if not isinstance(row, dict):
            continue
        rubric = row.get("scoring_rubric") if isinstance(row.get("scoring_rubric"), dict) else {}
        compact_refs.append(
            {
                "turn_id": row.get("turn_id"),
                "user_prompt": str(row.get("user_prompt") or ""),
                "reference_response": str(row.get("assistant_response") or ""),
                "rubric": {
                    "required_labels": list(rubric.get("required_labels") or []),
                    "anchor_labels": list(rubric.get("anchor_labels") or []),
                    "optional_labels": list(rubric.get("optional_labels") or []),
                    "forbidden_labels": list(rubric.get("forbidden_labels") or [])[:20],
                    "target_horizon": str(rubric.get("target_horizon") or ""),
                },
            }
        )

    compact_facts: list[dict[str, Any]] = []
    for item in list(judge_input.get("gold_interest_facts") or []):
        if not isinstance(item, dict):
            continue
        compact_facts.append(
            {
                "domain": str(item.get("domain") or ""),
                "label": str(item.get("label") or ""),
                "horizon": str(item.get("horizon") or ""),
            }
        )

    compact_prediction: list[dict[str, Any]] = []
    for turn in list(judge_input.get("prediction_dialogue") or []):
        if not isinstance(turn, dict):
            continue
        compact_prediction.append(
            {
                "turn_id": turn.get("turn_id"),
                "assistant_response": str(turn.get("assistant_response") or "")[:900],
            }
        )

    return {
        "task_id": str(judge_input.get("task_id") or ""),
        "user_id": str(judge_input.get("user_id") or ""),
        "gold_interest_facts": compact_facts,
        "gold_reference_dialogue": compact_refs,
        "prediction_dialogue": compact_prediction,
    }


def _normalize_judge_raw(*, raw: dict[str, Any], judge_input: dict[str, Any]) -> dict[str, Any]:
    missing_score_keys = [key for key in REQUIRED_JUDGE_SCORE_KEYS if key not in raw]
    if missing_score_keys:
        raw_preview = json.dumps(raw, ensure_ascii=False)[:1000]
        raise json.JSONDecodeError(
            f"Judge response missing required score fields: {', '.join(missing_score_keys)}",
            raw_preview,
            0,
        )
    if _judge_payload_looks_like_empty_placeholder(raw):
        raise JudgePlaceholderError("judge_returned_empty_placeholder_scores")
    interest_coverage = _score_0_to_5(raw.get("interest_coverage"))
    negative_avoidance = _score_0_to_5(raw.get("negative_avoidance"))
    concreteness = _score_0_to_5(raw.get("concreteness"))
    fluency = _score_0_to_5(raw.get("fluency"))
    weighted_score = _weighted_dialogue_score(
        interest_coverage=interest_coverage,
        negative_avoidance=negative_avoidance,
        concreteness=concreteness,
        fluency=fluency,
    )
    return {
        "task_id": str(judge_input.get("task_id") or ""),
        "user_id": str(judge_input.get("user_id") or ""),
        "interest_coverage": interest_coverage,
        "negative_avoidance": negative_avoidance,
        "concreteness": concreteness,
        "fluency": fluency,
        "matched_positive_interests": [str(x) for x in raw.get("matched_positive_interests", []) if str(x).strip()],
        "matched_negative_interests": [str(x) for x in raw.get("matched_negative_interests", []) if str(x).strip()],
        "unsupported_interests": [str(x) for x in raw.get("unsupported_interests", []) if str(x).strip()],
        "missed_core_interest_labels": [str(x) for x in raw.get("missed_core_interest_labels", []) if str(x).strip()],
        "brief_rationale": str(raw.get("brief_rationale") or "").strip(),
        "judge_total_score": float(raw.get("judge_total_score") or 0.0),
        "final_dialogue_score": weighted_score,
        "judge_prompt_version": DIALOGUE_JUDGE_PROMPT_VERSION,
    }


def _judge_prediction_with_retry(
    *,
    judge_client: OpenAICompatibleChatClient,
    fallback_judge_client: OpenAICompatibleChatClient | None = None,
    judge_input: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _judge_prediction(
            judge_client=judge_client,
            judge_input=judge_input,
        )
    except JudgePlaceholderError:
        if fallback_judge_client is not None:
            try:
                return _judge_prediction_text_fallback(
                    judge_client=fallback_judge_client,
                    judge_input=judge_input,
                )
            except Exception:
                return _judge_prediction_score_line_fallback(
                    judge_client=fallback_judge_client,
                    judge_input=judge_input,
                )
        try:
            return _judge_prediction(
                judge_client=judge_client,
                judge_input=judge_input,
                retry_note=(
                    "The previous judge response looked like an empty copied schema. "
                    "Evaluate the prediction semantically and fill non-placeholder scores and a rationale."
                ),
            )
        except Exception:
            return _judge_prediction_score_line_fallback(
                judge_client=judge_client,
                judge_input=judge_input,
            )
    except Exception as exc:
        if not _is_provider_block_message(_exception_text(exc)):
            try:
                return _judge_prediction_score_line_fallback(
                    judge_client=fallback_judge_client or judge_client,
                    judge_input=judge_input,
                )
            except Exception:
                raise
        retried = _judge_prediction(
            judge_client=fallback_judge_client or judge_client,
            judge_input=_build_provider_safe_judge_input(judge_input),
        )
        retried["judge_retry_mode"] = (
            "provider_safe_input_flash_fallback" if fallback_judge_client is not None else "provider_safe_input"
        )
        return retried
