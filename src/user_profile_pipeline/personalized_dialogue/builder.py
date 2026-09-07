from __future__ import annotations

from pathlib import Path
from typing import Any

from ..benchmark.profile_eval.io import _load_posts_with_indices
from ..multimodal_context import apply_image_limit, build_posts_context_bundle


DIALOGUE_TEMPLATE_ID = "two_setting_personalized_dialogue_v1"
TURN1_USER_PROMPT = "I want something that fits my usual taste. Could you recommend one option for me?"
TURN2_USER_PROMPT = "I want to try something a bit new, but still something that feels like me. Any suggestion?"

SCENARIO_ORDER = [
    "stable_recommendation",
    "short_term_exploration",
]

SCENARIO_PROMPTS: dict[str, str] = {
    "stable_recommendation": TURN1_USER_PROMPT,
    "short_term_exploration": TURN2_USER_PROMPT,
}


SCORING_WEIGHTS = {
    "interest_coverage": 0.40,
    "negative_avoidance": 0.25,
    "concreteness": 0.20,
    "fluency": 0.15,
}


def build_profile_context_from_gold_export(
    *,
    gold_export: dict[str, Any],
    max_domains: int = 5,
    max_labels_per_domain: int = 4,
) -> dict[str, Any]:
    active_domains: list[dict[str, Any]] = []
    top_interest_labels: list[str] = []
    top_negative_interest_labels: list[str] = []
    seen_labels: set[str] = set()
    seen_negative_labels: set[str] = set()

    for domain_row in gold_export.get("domains", []) or []:
        long_term = [
            str(item.get("label") or "").strip()
            for item in (domain_row.get("long_term_interests") or [])
            if str(item.get("label") or "").strip()
        ][: max_labels_per_domain]
        short_term = [
            str(item.get("label") or "").strip()
            for item in (domain_row.get("short_term_interests") or [])
            if str(item.get("label") or "").strip()
        ][: max_labels_per_domain]
        negative = [
            str(item.get("label") or "").strip()
            for item in (domain_row.get("negative_interests") or [])
            if str(item.get("label") or "").strip()
        ][: max_labels_per_domain]
        summary = str(domain_row.get("summary_natural_gold") or "").strip()
        if str(domain_row.get("status") or "").strip().lower() == "active":
            if not summary and not long_term and not short_term:
                continue
            if len(active_domains) < max_domains:
                active_domains.append(
                    {
                        "domain": str(domain_row.get("domain") or "").strip(),
                        "summary": summary,
                        "long_term_interest_labels": long_term,
                        "short_term_interest_labels": short_term,
                        "negative_interest_labels": negative,
                    }
                )
        elif not negative:
            continue
        for label in [*long_term, *short_term]:
            key = _normalize_label(label)
            if not key or key in seen_labels:
                continue
            seen_labels.add(key)
            top_interest_labels.append(label)
        for label in negative:
            key = _normalize_label(label)
            if not key or key in seen_negative_labels:
                continue
            seen_negative_labels.add(key)
            top_negative_interest_labels.append(label)
    return {
        "schema_version": "personalized_dialogue_profile_context_v2",
        "user_id": gold_export.get("user_id"),
        "active_domains": active_domains,
        "top_interest_labels": top_interest_labels,
        "top_negative_interest_labels": top_negative_interest_labels,
    }


def build_personalized_dialogue_artifacts(
    *,
    gold_export: dict[str, Any],
    posts_jsonl: str | Path,
    max_selected_interests: int = 4,
    max_images: int = -1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    posts_path = Path(posts_jsonl).resolve()
    posts = _load_posts_with_indices(posts_path)
    context_bundle = build_posts_context_bundle(posts)
    context_text = context_bundle.text
    profile_context = build_profile_context_from_gold_export(gold_export=gold_export)
    post_image_groups = context_bundle.post_image_url_groups
    all_image_urls = apply_image_limit(context_bundle.image_urls, max_images)
    one_image_per_post_urls = apply_image_limit(context_bundle.one_image_per_post_urls, max_images)

    selected = _select_interest_facts(
        gold_export=gold_export,
        max_selected_interests=max_selected_interests,
    )
    negative_selected = _select_negative_interest_facts(
        gold_export=gold_export,
        max_selected_interests=max_selected_interests,
    )
    scenario_turns = _build_scenario_turns(gold_export=gold_export)

    task_id = f"persona_dialogue__{gold_export.get('user_id')}"
    prompt_template = {
        "template_id": DIALOGUE_TEMPLATE_ID,
        "scenario_types": [row.get("scenario_type") for row in scenario_turns],
        "turn1_user_prompt": TURN1_USER_PROMPT,
        "turn2_user_prompt": TURN2_USER_PROMPT,
        "case_prompts": [
            {
                "turn_id": row.get("turn_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_type": row.get("scenario_type"),
                "user_prompt": row.get("user_prompt"),
            }
            for row in scenario_turns
        ],
        "scoring_dimensions": SCORING_WEIGHTS,
    }
    gold_ref = {
        "schema_version": "personalized_dialogue_gold_ref_v2",
        "task_id": task_id,
        "user_id": gold_export.get("user_id"),
        "source": {
            "gold_export_path": str(posts_path.parent / "benchmark_gold_export.json"),
            "posts_path": str(posts_path),
        },
        "prompt_template": prompt_template,
        "profile_context": profile_context,
        "selected_interest_facts": selected,
        "negative_interest_facts": negative_selected,
        "reference_dialogue": scenario_turns,
        "quality_checks": {
            "unsupported_claim_count": 0,
            "inactive_domain_as_core_interest": False,
            "turn_count": len(scenario_turns),
            "selected_interest_count": len(selected),
            "negative_interest_count": len(negative_selected),
            "has_stable_recommendation_case": any(
                row.get("scenario_type") == "stable_recommendation" for row in scenario_turns
            ),
            "has_short_term_exploration_case": any(
                row.get("scenario_type") == "short_term_exploration" for row in scenario_turns
            ),
        },
    }

    eval_task = {
        "schema_version": "personalized_dialogue_eval_task_v2",
        "task_id": task_id,
        "user_id": gold_export.get("user_id"),
        "context": {
            "posts_context_schema": context_bundle.payload,
            "posts_context_text": context_text,
            "image_urls": all_image_urls,
            "one_image_per_post_urls": one_image_per_post_urls,
            "post_image_url_groups": post_image_groups,
            "posts_count": len(posts),
            "image_count": len(all_image_urls),
            "image_post_count": len(post_image_groups),
        },
        "prompt_template": prompt_template,
        "profile_context": profile_context,
        "negative_interest_facts": negative_selected,
        "scenario_turns": [
            {
                "turn_id": row.get("turn_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_type": row.get("scenario_type"),
                "domain": row.get("domain"),
                "user_prompt": row.get("user_prompt"),
                "scoring_rubric": row.get("scoring_rubric"),
                "scenario_note": row.get("scenario_note"),
            }
            for row in scenario_turns
        ],
    }
    return gold_ref, eval_task


def _select_interest_facts(*, gold_export: dict[str, Any], max_selected_interests: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for domain_row in gold_export.get("domains", []) or []:
        if str(domain_row.get("status") or "").strip().lower() != "active":
            continue
        domain = str(domain_row.get("domain") or "").strip()
        for bucket_name, base_score in (("long_term_interests", 2.0), ("short_term_interests", 1.4)):
            bucket = domain_row.get(bucket_name, []) or []
            horizon = "long_term" if bucket_name == "long_term_interests" else "short_term"
            for idx, item in enumerate(bucket):
                label = str(item.get("label") or "").strip()
                if not label:
                    continue
                evidence = _normalize_indices(item.get("evidence_post_indices") or [])
                rank_score = base_score + 0.2 * min(_evidence_count(item), 4) + _modality_bonus(item) + (0.15 if idx == 0 else 0.0)
                candidates.append(
                    {
                        "domain": domain,
                        "label": label,
                        "horizon": horizon,
                        "evidence_post_indices": evidence,
                        "rank_score": round(rank_score, 4),
                    }
                )

    deduped: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for row in sorted(candidates, key=lambda x: (-float(x["rank_score"]), x["domain"], x["label"])):
        key = _normalize_label(str(row["label"]))
        if not key or key in seen_labels:
            continue
        seen_labels.add(key)
        deduped.append(row)
        if len(deduped) >= max_selected_interests:
            break
    return deduped


def _select_negative_interest_facts(*, gold_export: dict[str, Any], max_selected_interests: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for domain_row in gold_export.get("domains", []) or []:
        domain = str(domain_row.get("domain") or "").strip()
        for idx, item in enumerate(domain_row.get("negative_interests", []) or []):
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            evidence = _normalize_indices(item.get("evidence_post_indices") or [])
            rank_score = 1.0 + 0.2 * min(_evidence_count(item), 4) + _modality_bonus(item) + (0.1 if idx == 0 else 0.0)
            candidates.append(
                {
                    "domain": domain,
                    "label": label,
                    "horizon": "negative",
                    "evidence_post_indices": evidence,
                    "rank_score": round(rank_score, 4),
                }
            )

    deduped: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for row in sorted(candidates, key=lambda x: (-float(x["rank_score"]), x["domain"], x["label"])):
        key = _normalize_label(str(row["label"]))
        if not key or key in seen_labels:
            continue
        seen_labels.add(key)
        deduped.append(row)
        if len(deduped) >= max_selected_interests:
            break
    return deduped


def _build_scenario_turns(*, gold_export: dict[str, Any]) -> list[dict[str, Any]]:
    builders = [
        _build_stable_recommendation_case,
        _build_short_term_exploration_case,
    ]
    turns: list[dict[str, Any]] = []
    for builder in builders:
        scenario = builder(gold_export=gold_export)
        if scenario is None:
            continue
        turns.append(scenario)

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(turns, start=1):
        out.append({"turn_id": idx, **row})
    return out


def _build_stable_recommendation_case(*, gold_export: dict[str, Any]) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for domain_row in _active_domain_rows(gold_export):
        long_items = [item for item in list(domain_row.get("long_term_interests") or []) if _interest_label(item)]
        if not long_items:
            continue
        domain = str(domain_row.get("domain") or "").strip()
        for rank, item in enumerate(long_items):
            score = _interest_rank_score(item, horizon="long_term", rank=rank)
            score += 0.10 * min(len(long_items), 4)
            scored.append((score, domain_row, item))
    if not scored:
        return None

    _, domain_row, primary_item = sorted(
        scored,
        key=lambda row: (-row[0], str(row[1].get("domain") or ""), _interest_label(row[2])),
    )[0]
    domain = str(domain_row.get("domain") or "").strip()
    long_items = [item for item in list(domain_row.get("long_term_interests") or []) if _interest_label(item)]
    primary = _interest_label(primary_item)
    optional_labels = _dedup_labels(_interest_label(item) for item in long_items if _normalize_label(_interest_label(item)) != _normalize_label(primary))[:2]
    forbidden_labels = _collect_forbidden_labels(gold_export=gold_export, domain_row=domain_row)
    assistant_response = _render_stable_response(domain=domain, primary=primary, optional=optional_labels[:1])
    return {
        "scenario_id": f"stable_recommendation__{domain}",
        "scenario_type": "stable_recommendation",
        "domain": domain,
        "scenario_note": "The user asks for a natural, safe recommendation that fits their usual taste. Evaluate mainly against long-term interests.",
        "user_prompt": SCENARIO_PROMPTS["stable_recommendation"],
        "assistant_response": assistant_response,
        "claims": [
            {
                "claim_id": f"stable__{_normalize_label(primary)}__primary",
                "text": assistant_response,
                "linked_interest_labels": [primary],
                "evidence_post_indices": _normalize_indices(primary_item.get("evidence_post_indices") or []),
            }
        ],
        "scoring_rubric": {
            "target_horizon": "long_term",
            "required_labels": [primary],
            "anchor_labels": [],
            "optional_labels": optional_labels,
            "forbidden_labels": forbidden_labels,
            "weights": SCORING_WEIGHTS,
            # Legacy aliases kept for older metric scripts.
            "primary_labels": [primary],
            "secondary_labels": optional_labels,
        },
    }


def _build_short_term_exploration_case(*, gold_export: dict[str, Any]) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for domain_row in _active_domain_rows(gold_export):
        short_items = [item for item in list(domain_row.get("short_term_interests") or []) if _interest_label(item)]
        if not short_items:
            continue
        long_items = [item for item in list(domain_row.get("long_term_interests") or []) if _interest_label(item)]
        domain = str(domain_row.get("domain") or "").strip()
        for rank, item in enumerate(short_items):
            score = _interest_rank_score(item, horizon="short_term", rank=rank)
            score += 0.35 if long_items else 0.0
            score += 0.10 * min(len(short_items), 4)
            scored.append((score, domain_row, item))
    if not scored:
        return None

    _, domain_row, recent_item = sorted(
        scored,
        key=lambda row: (-row[0], str(row[1].get("domain") or ""), _interest_label(row[2])),
    )[0]
    domain = str(domain_row.get("domain") or "").strip()
    recent = _interest_label(recent_item)
    short_items = [item for item in list(domain_row.get("short_term_interests") or []) if _interest_label(item)]
    long_items = [item for item in list(domain_row.get("long_term_interests") or []) if _interest_label(item)]
    anchor_labels = _dedup_labels(_interest_label(item) for item in sorted(long_items, key=lambda item: -_interest_rank_score(item, horizon="long_term", rank=0)))[:2]
    optional_labels = _dedup_labels(
        _interest_label(item)
        for item in short_items
        if _normalize_label(_interest_label(item)) != _normalize_label(recent)
    )[:2]
    forbidden_labels = _collect_forbidden_labels(gold_export=gold_export, domain_row=domain_row)
    assistant_response = _render_exploration_response(domain=domain, recent=recent, anchors=anchor_labels[:1])
    return {
        "scenario_id": f"short_term_exploration__{domain}",
        "scenario_type": "short_term_exploration",
        "domain": domain,
        "scenario_note": "The user asks for something slightly new that still feels like them. Evaluate mainly against short-term interests, with long-term interests as compatibility anchors.",
        "user_prompt": SCENARIO_PROMPTS["short_term_exploration"],
        "assistant_response": assistant_response,
        "claims": [
            {
                "claim_id": f"explore__{_normalize_label(recent)}__recent",
                "text": assistant_response,
                "linked_interest_labels": [recent, *anchor_labels[:1]],
                "evidence_post_indices": sorted(
                    {
                        *_normalize_indices(recent_item.get("evidence_post_indices") or []),
                        *[
                            idx
                            for anchor in long_items[:1]
                            for idx in _normalize_indices(anchor.get("evidence_post_indices") or [])
                        ],
                    }
                ),
            }
        ],
        "scoring_rubric": {
            "target_horizon": "short_term",
            "required_labels": [recent],
            "anchor_labels": anchor_labels,
            "optional_labels": optional_labels,
            "forbidden_labels": forbidden_labels,
            "requires_anchor_compatibility": bool(anchor_labels),
            "weights": SCORING_WEIGHTS,
            # Legacy aliases kept for older metric scripts.
            "primary_labels": [recent],
            "secondary_labels": anchor_labels,
        },
    }


def _active_domain_rows(gold_export: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in (gold_export.get("domains") or [])
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "active"
    ]


def _interest_label(item: dict[str, Any]) -> str:
    return str((item or {}).get("label") or "").strip()


def _interest_rank_score(item: dict[str, Any], *, horizon: str, rank: int) -> float:
    base = 2.0 if horizon == "long_term" else 1.8
    return base + 0.25 * min(_evidence_count(item), 4) + _modality_bonus(item) + (0.15 if rank == 0 else 0.0)


def _evidence_count(item: dict[str, Any]) -> int:
    ids = list((item or {}).get("evidence_post_ids") or [])
    if ids:
        return len({str(x).strip() for x in ids if str(x).strip()})
    return len(_normalize_indices((item or {}).get("evidence_post_indices") or []))


def _modality_bonus(item: dict[str, Any]) -> float:
    summary = (item or {}).get("evidence_source_summary") or {}
    modalities = {str(x).strip().lower() for x in list(summary.get("modalities") or []) if str(x).strip()}
    if {"text", "visual"}.issubset(modalities):
        return 0.20
    if modalities:
        return 0.08
    return 0.0


def _collect_forbidden_labels(*, gold_export: dict[str, Any], domain_row: dict[str, Any], max_labels: int = 12) -> list[str]:
    labels: list[str] = []
    labels.extend(_interest_label(item) for item in list(domain_row.get("negative_interests") or []))
    for row in gold_export.get("domains", []) or []:
        if row is domain_row:
            continue
        labels.extend(_interest_label(item) for item in list((row or {}).get("negative_interests") or [])[:2])
    return _dedup_labels(labels)[:max_labels]


def _dedup_labels(labels: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label or "").strip()
        key = _normalize_label(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _render_stable_response(*, domain: str, primary: str, optional: list[str]) -> str:
    lead = _natural_recommendation_phrase(domain=domain, label=primary)
    if optional:
        return (
            f"{lead} It is a safe fit for your usual taste, and you can keep it close to "
            f"{optional[0].lower()} if you want a small variation."
        )
    return f"{lead} It is a safe fit for your usual taste without forcing a random detour."


def _render_exploration_response(*, domain: str, recent: str, anchors: list[str]) -> str:
    lead = _natural_recommendation_phrase(domain=domain, label=recent)
    if anchors:
        return (
            f"{lead} It gives you something a bit new while still staying close to your "
            f"{anchors[0].lower()} side, so it should not feel random."
        )
    return f"{lead} It gives you a bit of novelty while still staying close to what your recent taste suggests."


def _natural_recommendation_phrase(*, domain: str, label: str) -> str:
    lowered = label.lower()
    if domain == "entertainment":
        return f"I’d recommend a {lowered}-leaning pick for tonight."
    if domain == "gaming":
        return f"I’d recommend something in the {lowered} lane."
    if domain == "food_drink":
        return f"I’d recommend going for {lowered}."
    if domain == "travel_city_exploration":
        return f"I’d recommend planning a small outing around {lowered}."
    if domain == "photography_creation":
        return f"I’d recommend a small session built around {lowered}."
    if domain == "pets":
        return f"I’d recommend a simple idea centered on {lowered}."
    if domain == "sports_outdoor":
        return f"I’d recommend centering it on {lowered}."
    return f"I’d recommend something centered on {lowered}."


def _normalize_indices(values: list[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for item in values:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _normalize_label(value: str) -> str:
    lowered = value.strip().lower()
    return " ".join(lowered.replace("_", " ").split())
