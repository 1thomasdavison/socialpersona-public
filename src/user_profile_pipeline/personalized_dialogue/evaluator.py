from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from ..benchmark.profile_eval.specs import DEFAULT_MODEL_SPECS
from ..config import load_yaml
from ..llm_client import OpenAICompatibleChatClient
from ..multimodal_context import apply_image_limit, render_posts_context_payload


DIALOGUE_SYSTEM_PROMPT = """You are a concise personalization assistant.

Use only the user's posts, image captions, and raw images that are provided in the context.
Give one natural, practical recommendation that fits the current user request.
Stay close to supported interests, avoid weak/noisy directions, and do not invent demographics or hidden motives.
When images reveal a hobby, object, routine, food, place, or activity that the text alone would not make obvious, it is valid and desirable to use that signal.
Do not mention posts, evidence, pipelines, scoring, benchmarks, profiles, or observation windows unless the user explicitly asks.
"""

DIALOGUE_SYSTEM_PROMPT_PROFILE_ONLY = """You are a concise personalization assistant.

Use only the user's profile information that is provided in the context.
Give one natural, practical recommendation that fits the current user request.
Stay close to supported interests, avoid weak/noisy directions, and do not invent demographics or hidden motives.
Do not mention evidence, pipelines, scoring, benchmarks, profiles, or observation windows unless the user explicitly asks.
"""


@dataclass
class DialogueModelConfig:
    name: str
    provider: str
    base_url: str
    api_key_env: str
    api_model: str
    multimodal: bool = True


class PersonalizedDialogueEvaluator:
    def __init__(
        self,
        *,
        model_config_path: str = "configs/model.yaml",
        model_name: str = "mock_oracle",
        max_images: int = 10,
    ) -> None:
        self.model_cfg = load_yaml(model_config_path) if Path(model_config_path).exists() else {}
        self.model_name = (model_name or "mock_oracle").strip()
        self.max_images = max(0, int(max_images))

    def run(
        self,
        *,
        gold_ref: dict[str, Any],
        eval_task: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prediction = self._predict(gold_ref=gold_ref, eval_task=eval_task)
        metrics = self._score(gold_ref=gold_ref, prediction=prediction)
        return prediction, metrics

    def _predict(self, *, gold_ref: dict[str, Any], eval_task: dict[str, Any]) -> dict[str, Any]:
        scenario_turns = _iter_scenario_turns(eval_task=eval_task)
        if not scenario_turns:
            raise ValueError("eval_task.scenario_turns must contain the current two-scenario dialogue schema")
        if self.model_name == "mock_oracle":
            dialogue = []
            for row in gold_ref.get("reference_dialogue", []) or []:
                dialogue.append(
                    {
                        "turn_id": row.get("turn_id"),
                        "scenario_id": row.get("scenario_id"),
                        "scenario_type": row.get("scenario_type"),
                        "user_prompt": _turn_prompt(eval_task, int(row.get("turn_id", 0))),
                        "assistant_response": row.get("assistant_response", ""),
                    }
                )
            return {
                "schema_version": "personalized_dialogue_prediction_v1",
                "task_id": eval_task.get("task_id"),
                "user_id": eval_task.get("user_id"),
                "model": self.model_name,
                "dialogue": dialogue,
            }

        client, model_config = _build_chat_client(
            model_cfg=self.model_cfg,
            model_name=self.model_name,
        )
        context_obj = (eval_task.get("context") or {}) if isinstance(eval_task.get("context"), dict) else {}
        context = str(context_obj.get("posts_context_text") or "").strip()
        if not context and isinstance(context_obj.get("posts_context_schema"), dict):
            context = render_posts_context_payload(context_obj["posts_context_schema"])
        profile_only = bool(context_obj.get("profile_only_mode"))
        image_urls: list[str] = []
        if not profile_only:
            image_urls = apply_image_limit(list(context_obj.get("image_urls") or []), self.max_images)
            if not bool(model_config.multimodal):
                image_urls = []

        dialogue: list[dict[str, Any]] = []
        for turn in scenario_turns:
            rendered_prompt = _render_scenario_prompt(
                eval_task=eval_task,
                context=context,
                turn=turn,
            )
            system_prompt = DIALOGUE_SYSTEM_PROMPT_PROFILE_ONLY if profile_only else DIALOGUE_SYSTEM_PROMPT
            response = client.chat_text(
                system_prompt=system_prompt,
                user_prompt=rendered_prompt,
                image_urls=image_urls or None,
            )
            dialogue.append(
                {
                    "turn_id": turn.get("turn_id"),
                    "scenario_id": turn.get("scenario_id"),
                    "scenario_type": turn.get("scenario_type"),
                    "user_prompt": str(turn.get("user_prompt") or "").strip(),
                    "assistant_response": response,
                }
            )

        return {
            "schema_version": "personalized_dialogue_prediction_v1",
            "task_id": eval_task.get("task_id"),
            "user_id": eval_task.get("user_id"),
            "model": self.model_name,
            "dialogue": dialogue,
        }

    def _score(self, *, gold_ref: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        pred_by_turn = {
            int(row.get("turn_id", 0)): row
            for row in prediction.get("dialogue", []) or []
        }
        turn_metrics: list[dict[str, Any]] = []
        for gold_turn in gold_ref.get("reference_dialogue", []) or []:
            turn_id = int(gold_turn.get("turn_id", 0))
            pred_row = pred_by_turn.get(turn_id, {})
            pred_text = str(pred_row.get("assistant_response") or "").strip()
            scenario_type = str(gold_turn.get("scenario_type") or "").strip()
            if not scenario_type:
                raise ValueError(f"reference turn {turn_id} is missing scenario_type")
            turn_metrics.append(
                _score_scenario_turn(
                    gold_turn=gold_turn,
                    response_text=pred_text,
                )
            )

        overall_score = (
            sum(float(row["turn_score"]) for row in turn_metrics) / len(turn_metrics)
            if turn_metrics
            else 0.0
        )
        return {
            "schema_version": "personalized_dialogue_metrics_v1",
            "task_id": gold_ref.get("task_id"),
            "user_id": gold_ref.get("user_id"),
            "model": self.model_name,
            "turn_metrics": turn_metrics,
            "overall": {
                "turn_count": len(turn_metrics),
                "claim_recall_mean": round(_mean([row["claim_recall"] for row in turn_metrics]), 4),
                "interest_hit_rate_mean": round(_mean([row["interest_hit_rate"] for row in turn_metrics]), 4),
                "interest_coverage_mean": round(_mean([float(row.get("interest_coverage", 0.0) or 0.0) for row in turn_metrics]), 4),
                "negative_avoidance_mean": round(_mean([float(row.get("negative_avoidance", 0.0) or 0.0) for row in turn_metrics]), 4),
                "concreteness_mean": round(_mean([float(row.get("concreteness", 0.0) or 0.0) for row in turn_metrics]), 4),
                "fluency_mean": round(_mean([float(row.get("fluency", 0.0) or 0.0) for row in turn_metrics]), 4),
                "length_compliance_rate": round(
                    _mean([1.0 if row["length_ok"] else 0.0 for row in turn_metrics]), 4
                ),
                "format_compliance_rate": round(
                    _mean([1.0 if row["format_ok"] else 0.0 for row in turn_metrics]), 4
                ),
                "forbidden_recommendation_rate": round(
                    _mean([1.0 if row.get("forbidden_recommended_labels") else 0.0 for row in turn_metrics]), 4
                ),
                "overall_score": round(overall_score, 4),
            },
            "scenario_breakdown": _scenario_breakdown(turn_metrics),
        }


def _build_chat_client(*, model_cfg: dict[str, Any], model_name: str) -> tuple[OpenAICompatibleChatClient, DialogueModelConfig]:
    normalized = (model_name or "").strip()
    spec = DEFAULT_MODEL_SPECS.get(normalized)
    if spec is None:
        raise ValueError(f"Unknown dialogue model: {normalized}. Add it to DEFAULT_MODEL_SPECS first.")
    provider = spec.provider
    if provider in {"chatanywhere", "bailian"}:
        normalized_provider = "openai_compatible"
    elif provider.startswith("vertex"):
        normalized_provider = "vertex"
    else:
        normalized_provider = provider
    config = DialogueModelConfig(
        name=normalized,
        provider=normalized_provider,
        base_url=str(spec.base_url or model_cfg.get("base_url") or ""),
        api_key_env=str(spec.api_key_env or model_cfg.get("api_key_env") or ""),
        api_model=str(spec.api_model or normalized),
        multimodal=bool(spec.multimodal),
    )

    client = OpenAICompatibleChatClient(
        provider=config.provider,
        base_url=config.base_url,
        model=config.api_model,
        api_key_env=config.api_key_env,
        timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        max_tokens=int(model_cfg.get("max_tokens", 4096)),
    )
    return client, config


def _turn_prompt(eval_task: dict[str, Any], turn_id: int) -> str:
    for row in _iter_scenario_turns(eval_task=eval_task):
        try:
            current_turn_id = int(row.get("turn_id", 0))
        except Exception:
            current_turn_id = 0
        if current_turn_id == int(turn_id):
            return str(row.get("user_prompt") or "").strip()
    raise ValueError(f"scenario turn {turn_id} is missing")


def _iter_scenario_turns(
    *,
    eval_task: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = eval_task.get("scenario_turns")
    if isinstance(rows, list) and rows:
        return sorted((row for row in rows if isinstance(row, dict)), key=lambda row: int(row.get("turn_id", 0) or 0))
    return []


def _render_scenario_prompt(*, eval_task: dict[str, Any], context: str, turn: dict[str, Any]) -> str:
    user_prompt = str(turn.get("user_prompt") or "").strip()
    if not user_prompt:
        raise ValueError(f"scenario turn {turn.get('turn_id')} is missing user_prompt")
    context_obj = (eval_task.get("context") or {}) if isinstance(eval_task.get("context"), dict) else {}
    if context_obj.get("profile_only_mode"):
        return (
            "PERSONALIZATION RULES:\n"
            "- Use only the provided user profile information.\n"
            "- Answer the user's request naturally.\n"
            "- Give one concrete recommendation, not a long list.\n"
            "- Do not mention evidence, post indices, profiles, pipelines, scoring, or observation windows.\n"
            "- Avoid recommendations based on weak, noisy, negative, or contradicted signals.\n\n"
            f"USER PROFILE:\n{context}\n\n"
            f"USER REQUEST:\n{user_prompt}"
        )
    return (
        "PERSONALIZATION RULES:\n"
        "- Use only the provided posts, image captions, and images.\n"
        "- Answer the user's request naturally.\n"
        "- Give one concrete recommendation, not a long list.\n"
        "- Do not mention evidence, post indices, profiles, pipelines, scoring, or observation windows.\n"
        "- Avoid recommendations based on weak, noisy, negative, or contradicted signals.\n\n"
        f"USER POSTS AND IMAGE CONTEXT:\n{context}\n\n"
        f"USER REQUEST:\n{user_prompt}"
    )


def _render_turn1_prompt(*, eval_task: dict[str, Any], context: str) -> str:
    turns = _iter_scenario_turns(eval_task=eval_task)
    if len(turns) < 1:
        raise ValueError("eval_task.scenario_turns is missing turn 1")
    return _render_scenario_prompt(eval_task=eval_task, context=context, turn=turns[0])


def _render_turn2_prompt(*, eval_task: dict[str, Any], context: str, turn1_response: str = "") -> str:
    turns = _iter_scenario_turns(eval_task=eval_task)
    if len(turns) < 2:
        raise ValueError("eval_task.scenario_turns is missing turn 2")
    # The two prompts are independent cases. We intentionally do not include turn1_response.
    return _render_scenario_prompt(eval_task=eval_task, context=context, turn=turns[1])

def _claim_is_matched(*, claim: dict[str, Any], response_text: str) -> bool:
    if not response_text.strip():
        return False
    claim_text = str(claim.get("text") or "").strip()
    if claim_text and fuzz.partial_ratio(_normalize_text(claim_text), _normalize_text(response_text)) >= 70:
        return True
    for label in claim.get("linked_interest_labels", []) or []:
        if _label_is_matched(label=str(label), response_text=response_text):
            return True
    return False


def _label_is_matched(*, label: str, response_text: str) -> bool:
    norm_label = _normalize_text(label)
    norm_response = _normalize_text(response_text)
    if not norm_label or not norm_response:
        return False
    if norm_label in norm_response:
        return True
    return fuzz.partial_ratio(norm_label, norm_response) >= 85


def _normalize_text(value: str) -> str:
    lowered = str(value or "").strip().lower().replace("_", " ")
    return " ".join(re.findall(r"[a-z0-9]+", lowered))


def _length_ok(*, word_count: int) -> bool:
    return 18 <= word_count <= 120


def _format_ok(*, response_text: str) -> bool:
    return len(str(response_text or "").strip()) >= 24


def _score_scenario_turn(*, gold_turn: dict[str, Any], response_text: str) -> dict[str, Any]:
    turn_id = int(gold_turn.get("turn_id", 0) or 0)
    scenario_type = str(gold_turn.get("scenario_type") or "").strip()
    claims = gold_turn.get("claims", []) or []
    rubric = gold_turn.get("scoring_rubric", {}) or {}

    required_labels = _clean_label_list(rubric.get("required_labels") or rubric.get("primary_labels") or [])
    anchor_labels = _clean_label_list(rubric.get("anchor_labels") or [])
    optional_labels = _clean_label_list(rubric.get("optional_labels") or rubric.get("secondary_labels") or [])
    forbidden_labels = _clean_label_list(rubric.get("forbidden_labels") or [])

    matched_claim_ids = [
        str(claim.get("claim_id"))
        for claim in claims
        if _claim_is_matched(claim=claim, response_text=response_text)
    ]
    claim_recall = (len(matched_claim_ids) / len(claims)) if claims else 1.0

    matched_required_labels = [label for label in required_labels if _label_is_matched(label=label, response_text=response_text)]
    matched_anchor_labels = [label for label in anchor_labels if _label_is_matched(label=label, response_text=response_text)]
    matched_optional_labels = [label for label in optional_labels if _label_is_matched(label=label, response_text=response_text)]
    forbidden_recommended_labels = [
        label for label in forbidden_labels if _label_is_recommended(label=label, response_text=response_text)
    ]
    forbidden_safely_negated_labels = [
        label
        for label in forbidden_labels
        if _label_is_matched(label=label, response_text=response_text)
        and label not in forbidden_recommended_labels
    ]

    required_hit_rate = (len(matched_required_labels) / len(required_labels)) if required_labels else 1.0
    anchor_hit_rate = (len(matched_anchor_labels) / len(anchor_labels)) if anchor_labels else 1.0
    optional_hit_rate = (len(matched_optional_labels) / len(optional_labels)) if optional_labels else 0.0

    interest_coverage = _interest_coverage_score(
        scenario_type=scenario_type,
        required_hit_rate=required_hit_rate,
        anchor_hit_rate=anchor_hit_rate,
        optional_hit_rate=optional_hit_rate,
        has_required=bool(required_labels),
        has_anchor=bool(anchor_labels),
    )
    negative_avoidance = _negative_avoidance_score(forbidden_recommended_labels)
    concreteness = _concreteness_score(response_text=response_text)
    fluency = _fluency_score(response_text=response_text)

    turn_score = (
        0.40 * interest_coverage
        + 0.25 * negative_avoidance
        + 0.20 * concreteness
        + 0.15 * fluency
    ) / 5.0 * 100.0

    word_count = len(re.findall(r"\b\w+\b", response_text))
    length_ok = _length_ok(word_count=word_count)
    format_ok = _format_ok(response_text=response_text)
    interest_pool = sorted({*required_labels, *anchor_labels, *optional_labels})
    matched_interest_labels = sorted({*matched_required_labels, *matched_anchor_labels, *matched_optional_labels})
    interest_hit_rate = (len(matched_interest_labels) / len(interest_pool)) if interest_pool else 1.0

    return {
        "turn_id": turn_id,
        "scenario_id": gold_turn.get("scenario_id"),
        "scenario_type": scenario_type,
        "word_count": word_count,
        "length_ok": length_ok,
        "format_ok": format_ok,
        "claim_recall": round(claim_recall, 4),
        "interest_hit_rate": round(interest_hit_rate, 4),
        "required_hit_rate": round(required_hit_rate, 4),
        "anchor_hit_rate": round(anchor_hit_rate, 4),
        "optional_hit_rate": round(optional_hit_rate, 4),
        "interest_coverage": round(interest_coverage, 4),
        "negative_avoidance": round(negative_avoidance, 4),
        "concreteness": round(concreteness, 4),
        "fluency": round(fluency, 4),
        "strategy_score": round(interest_coverage / 5.0, 4),
        "matched_claim_ids": matched_claim_ids,
        "missing_claim_ids": [
            str(claim.get("claim_id"))
            for claim in claims
            if str(claim.get("claim_id")) not in set(matched_claim_ids)
        ],
        "matched_interest_labels": matched_interest_labels,
        "matched_required_labels": matched_required_labels,
        "matched_anchor_labels": matched_anchor_labels,
        "matched_optional_labels": matched_optional_labels,
        # Legacy aliases for downstream scripts that still read primary/secondary labels.
        "matched_primary_labels": matched_required_labels,
        "matched_secondary_labels": sorted({*matched_anchor_labels, *matched_optional_labels}),
        "forbidden_recommended_labels": forbidden_recommended_labels,
        "forbidden_safely_negated_labels": forbidden_safely_negated_labels,
        "turn_score": round(turn_score, 4),
    }


def _clean_label_list(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        label = str(value or "").strip()
        key = _normalize_text(label)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def _interest_coverage_score(
    *,
    scenario_type: str,
    required_hit_rate: float,
    anchor_hit_rate: float,
    optional_hit_rate: float,
    has_required: bool,
    has_anchor: bool,
) -> float:
    if not has_required:
        return 0.0
    if scenario_type == "short_term_exploration":
        anchor_component = anchor_hit_rate if has_anchor else 1.0
        score = 5.0 * (0.70 * required_hit_rate + 0.25 * anchor_component + 0.05 * optional_hit_rate)
    else:
        score = 5.0 * min(1.0, 0.90 * required_hit_rate + 0.10 * optional_hit_rate)
    return max(0.0, min(5.0, score))


def _negative_avoidance_score(forbidden_recommended_labels: list[str]) -> float:
    if not forbidden_recommended_labels:
        return 5.0
    if len(forbidden_recommended_labels) == 1:
        return 2.0
    return 0.0


def _concreteness_score(*, response_text: str) -> float:
    text = str(response_text or "").strip()
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    lowered = text.lower()
    action_markers = [
        "recommend", "try", "go for", "go with", "pick", "choose", "plan", "watch", "read", "listen",
        "visit", "make", "build", "start with", "center", "session", "option", "tonight", "this week",
    ]
    score = 1.0
    if word_count >= 12:
        score += 1.0
    if word_count >= 22:
        score += 1.0
    if any(marker in lowered for marker in action_markers):
        score += 1.0
    if _has_reason_language(text):
        score += 1.0
    return max(0.0, min(5.0, score))


def _fluency_score(*, response_text: str) -> float:
    text = str(response_text or "").strip()
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    if word_count < 5:
        return 1.0
    score = 3.0
    if 12 <= word_count <= 90:
        score += 1.0
    if re.search(r"[.!?]$", text):
        score += 0.5
    if not re.search(r"\{\s*\"|```|\bturn\s*\d\b|post index|benchmark|scoring", text, flags=re.I):
        score += 0.5
    return max(0.0, min(5.0, score))


def _has_reason_language(response_text: str) -> bool:
    lowered = str(response_text or "").lower()
    return any(marker in lowered for marker in ["because", "since", "fits", "while", "still", "so it", "without"])

def _label_is_recommended(*, label: str, response_text: str) -> bool:
    text = str(response_text or "").strip().lower().replace("_", " ")
    phrase = str(label or "").strip().lower().replace("_", " ")
    if not phrase:
        return False
    if phrase not in text and fuzz.partial_ratio(_normalize_text(phrase), _normalize_text(text)) < 85:
        return False

    positive_markers = [
        "recommend",
        "try",
        "go with",
        "lean into",
        "start with",
        "choose",
        "pick",
        "plan around",
        "listen to",
        "play",
        "watch",
        "read",
        "do ",
        "build around",
    ]
    negative_markers = [
        "avoid",
        "skip",
        "not ",
        "don't",
        "do not",
        "wouldn't",
        "would not",
        "rather than",
        "instead of",
        "too noisy",
        "over-read",
        "overread",
        "noisy signal",
    ]
    for match in re.finditer(re.escape(phrase), text):
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 60)
        window = text[start:end]
        if any(marker in window for marker in negative_markers):
            continue
        if any(marker in window for marker in positive_markers):
            return True
        return True
    return False


def _has_blend_language(response_text: str) -> bool:
    lowered = str(response_text or "").strip().lower()
    markers = [
        "but",
        "while",
        "still",
        "keep",
        "mix",
        "blend",
        "without losing",
        "base",
        "fresher",
        "fresh",
        "switch things up",
    ]
    return any(marker in lowered for marker in markers)


def _scenario_breakdown(turn_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in turn_metrics:
        key = str(row.get("scenario_type") or "unknown")
        grouped.setdefault(key, []).append(row)
    out: dict[str, Any] = {}
    for key, rows in grouped.items():
        out[key] = {
            "turn_count": len(rows),
            "mean_turn_score": round(_mean([float(row.get("turn_score", 0.0) or 0.0) for row in rows]), 4),
            "mean_strategy_score": round(_mean([float(row.get("strategy_score", 0.0) or 0.0) for row in rows]), 4),
            "interest_coverage_mean": round(_mean([float(row.get("interest_coverage", 0.0) or 0.0) for row in rows]), 4),
            "negative_avoidance_mean": round(_mean([float(row.get("negative_avoidance", 0.0) or 0.0) for row in rows]), 4),
            "concreteness_mean": round(_mean([float(row.get("concreteness", 0.0) or 0.0) for row in rows]), 4),
            "fluency_mean": round(_mean([float(row.get("fluency", 0.0) or 0.0) for row in rows]), 4),
            "forbidden_recommendation_rate": round(
                _mean([1.0 if row.get("forbidden_recommended_labels") else 0.0 for row in rows]),
                4,
            ),
        }
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)
