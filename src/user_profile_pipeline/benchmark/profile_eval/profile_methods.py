from __future__ import annotations

import json
import mimetypes
from typing import Any
from urllib.parse import urlsplit

from ...schemas import DOMAIN_NAMES
from ..evaluator.precheck import MAX_EVIDENCE_PER_ANCHOR
from .prompts import (
    EXTRACTIVE_SELECTION_JSON_SCHEMA_HINT,
    FINAL_PROFILE_JSON_SCHEMA_HINT,
    HIERARCHICAL_CHUNK_JSON_SCHEMA_HINT,
)
from .specs import DomainEvalTask


def _build_domain_prompt(task: DomainEvalTask, posts_context: str) -> str:
    return f"""Evaluate user profile signal for one domain.

DOMAIN: {task.domain}
DOMAIN_DEFINITION: {task.domain_definition}
OUTPUT_STATUS_OPTIONS: active or inactive

POSTS_CONTEXT:
{posts_context}

Instructions:
- Read the posts and decide whether this domain contains a reliable user-interest signal.
- If active, extract only the clearly supported interest tags for this domain.
- Use long_term_interest_tags for recurring or stable interests.
- Use short_term_interest_tags for newer or more time-local interests.
- Return only clearly supported tags. Do not try to fill a quota.
- Most active domains should have only 1-2 reliable tags in total.
- Return more than 2 tags only when the evidence is unusually strong and the tags are clearly distinct.
- It is valid to return zero long_term_interest_tags.
- It is valid to return zero short_term_interest_tags.
- Use short natural-language labels that describe user interest themes rather than raw hashtags, named entities, or one-off events.
- Prefer fewer, broader tags that summarize the user's main tendencies in this domain.
- Do not produce near-duplicate tags across long-term and short-term buckets.
- If two candidate tags largely overlap, keep only the broader or better-supported one.
- Do not split one core interest into both long-term and short-term tags unless the short-term tag adds a clearly distinct recent focus.
- If the evidence is weak, sparse, one-off, or not clearly attributable to user preference, prefer inactive.
- If inactive, return empty tag lists.
- Return JSON only.
"""


def _build_domain_prompt_neutral(task: DomainEvalTask, posts_context: str) -> str:
    return f"""Evaluate user profile signal for one domain.

DOMAIN: {task.domain}
DOMAIN_DEFINITION: {task.domain_definition}
OUTPUT_STATUS_OPTIONS: active or inactive

POSTS_CONTEXT:
{posts_context}

Instructions:
- Read the posts and decide whether this domain contains a reliable user-interest signal.
- If active, extract all clearly supported interest tags for this domain.
- Use long_term_interest_tags for recurring or stable interests.
- Use short_term_interest_tags for newer or more time-local interests.
- Return all clearly supported tags. Do not artificially limit the count.
- It is valid to return zero long_term_interest_tags.
- It is valid to return zero short_term_interest_tags.
- Use short natural-language labels that describe user interest themes rather than raw hashtags, named entities, or one-off events.
- Prefer precise, specific tags that accurately reflect the user's demonstrated interests.
- Do not produce near-duplicate tags across long-term and short-term buckets.
- Do not split one core interest into both long-term and short-term tags unless the short-term tag adds a clearly distinct recent focus.
- If the evidence is weak, sparse, one-off, or not clearly attributable to user preference, prefer inactive.
- If inactive, return empty tag lists.
- Return JSON only.
"""


def _user_group_key(task: DomainEvalTask) -> tuple[str, str, str]:
    return (task.source_tag, task.user_id, task.gold_path)


def _domain_list_text() -> str:
    return "\n".join(f"- {domain}" for domain in DOMAIN_NAMES)


def _timeline_rows_from_payload(
    payload: dict[str, Any],
    *,
    max_posts: int = 200,
    chronological: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order_idx, post in enumerate(list(payload.get("posts") or [])):
        if not isinstance(post, dict):
            continue
        image_captions: list[str] = []
        for image_row in list(post.get("image_texts") or []):
            if not isinstance(image_row, dict):
                continue
            image_text = " ".join(str(image_row.get("image_text") or "").split()).strip()
            if image_text:
                image_captions.append(image_text)
        post_id = str(post.get("post_id") or post.get("post_index") or order_idx).strip()
        rows.append(
            {
                "post_id": post_id,
                "timestamp": str(post.get("created_at") or "").strip(),
                "text": str(post.get("text") or ""),
                "image_captions": image_captions,
                "_order": order_idx,
            }
        )
    if chronological:
        rows.sort(key=lambda row: (str(row.get("timestamp") or ""), int(row.get("_order") or 0)))
    if max_posts and max_posts > 0 and len(rows) > max_posts:
        # Keep the most recent bounded window after chronological ordering.
        rows = rows[-max_posts:]
    return rows


def _public_timeline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "post_id": str(row.get("post_id") or "").strip(),
                "timestamp": str(row.get("timestamp") or "").strip(),
                "text": str(row.get("text") or ""),
                "image_captions": list(row.get("image_captions") or []),
            }
        )
    return out


def _timeline_json_from_rows(rows: list[dict[str, Any]]) -> str:
    return json.dumps(_public_timeline_rows(rows), ensure_ascii=False, separators=(",", ":"))


def _time_span_for_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    timestamps = [str(row.get("timestamp") or "").strip() for row in rows if str(row.get("timestamp") or "").strip()]
    return {
        "start": timestamps[0] if timestamps else "",
        "end": timestamps[-1] if timestamps else "",
    }


def _build_hierarchical_chunk_prompt(*, chunk_id: str, rows: list[dict[str, Any]]) -> str:
    return f"""Interest domains:
{_domain_list_text()}

JSON schema:
{HIERARCHICAL_CHUNK_JSON_SCHEMA_HINT}

Chunk metadata:
{json.dumps({"chunk_id": chunk_id, "time_span": _time_span_for_rows(rows)}, ensure_ascii=False, separators=(",", ":"))}

Posts:
{_timeline_json_from_rows(rows)}
"""


def _build_hierarchical_global_prompt(*, user_id: str, chunk_summaries: list[dict[str, Any]]) -> str:
    return f"""Interest domains:
{_domain_list_text()}

Return the final profile using this JSON schema:
{FINAL_PROFILE_JSON_SCHEMA_HINT}

User id:
{user_id}

Chunk summaries:
{json.dumps(chunk_summaries, ensure_ascii=False, separators=(",", ":"))}
"""


def _build_extractive_selection_prompt(*, user_id: str, timeline_rows: list[dict[str, Any]], k: int) -> str:
    return f"""Interest domains:
{_domain_list_text()}

Selection criteria:
- Relevance: the post directly supports the domain.
- Specificity: the post reveals a concrete interest, not just a vague topic.
- Recurrence: prefer posts consistent with repeated behavior.
- Recency: include recent posts if they suggest recent interests.
- Multimodal grounding: image captions can support interests even when text is short.

Each domain can have at most {int(k)} selected posts.

JSON schema:
{EXTRACTIVE_SELECTION_JSON_SCHEMA_HINT}

User timeline:
{json.dumps({"user_id": user_id, "posts": _public_timeline_rows(timeline_rows)}, ensure_ascii=False, separators=(",", ":"))}
"""


def _build_selected_posts_with_content(
    *,
    user_id: str,
    selection: dict[str, Any],
    timeline_rows: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    rows_by_id = {str(row.get("post_id") or "").strip(): row for row in timeline_rows}
    selected_raw = selection.get("selected_posts_by_domain")
    if not isinstance(selected_raw, dict):
        selected_raw = {}

    selected_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAIN_NAMES:
        selected_by_domain[domain] = []
        seen: set[str] = set()
        raw_items = selected_raw.get(domain)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            post_id = str(item.get("post_id") or "").strip()
            if not post_id or post_id in seen:
                continue
            row = rows_by_id.get(post_id)
            if row is None:
                continue
            merged = {
                **_public_timeline_rows([row])[0],
                "evidence_modalities": list(item.get("evidence_modalities") or []),
                "temporal_role": str(item.get("temporal_role") or "unclear").strip() or "unclear",
                "selection_reason": str(item.get("selection_reason") or "").strip(),
            }
            selected_by_domain[domain].append(merged)
            seen.add(post_id)
            if len(selected_by_domain[domain]) >= max(0, int(k)):
                break
    return {
        "user_id": user_id,
        "selected_posts_by_domain": selected_by_domain,
    }


def _build_extractive_abstractive_prompt(*, selected_posts_with_content: dict[str, Any]) -> str:
    return f"""Interest domains:
{_domain_list_text()}

Return the final profile using this JSON schema:
{FINAL_PROFILE_JSON_SCHEMA_HINT}

Selected representative posts:
{json.dumps(selected_posts_with_content, ensure_ascii=False, separators=(",", ":"))}
"""


def _build_post_id_to_index_map(posts: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for fallback_idx, post in enumerate(posts):
        try:
            post_index = int(post.get("_line_index", fallback_idx))
        except Exception:
            post_index = fallback_idx
        candidates = [
            str(post.get("post_id") or "").strip(),
            str(post_index),
            str(fallback_idx),
        ]
        for candidate in candidates:
            if candidate and candidate not in out:
                out[candidate] = post_index
    return out


def _normalize_evidence_post_ids(value: Any, *, post_id_to_index: dict[str, int], limit: int = 2) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    max_index = max(post_id_to_index.values(), default=-1)
    for raw in value:
        if isinstance(raw, dict):
            raw = raw.get("post_id") or raw.get("post_index") or raw.get("id")
        text = str(raw or "").strip()
        if not text:
            continue
        idx = post_id_to_index.get(text)
        if idx is None:
            try:
                parsed_idx = int(text)
            except (TypeError, ValueError):
                parsed_idx = -1
            if 0 <= parsed_idx <= max_index:
                idx = parsed_idx
        if idx is None or idx in out:
            continue
        out.append(idx)
        if len(out) >= limit:
            break
    return out


def _normalize_evidence_modality(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"text", "image", "both"}:
        return text
    return "both"


def _interest_items_to_anchors(
    items: Any,
    *,
    post_id_to_index: dict[str, int],
    limit_items: int,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("tag") or item.get("label") or item.get("interest") or "").strip()
        if not label:
            continue
        label_key = label.lower()
        if label_key in seen_labels:
            continue
        evidence_ids = item.get("evidence_post_ids") or item.get("evidence_post_indices") or []
        out.append(
            {
                "label": label,
                "evidence_post_ids": [str(x) for x in evidence_ids] if isinstance(evidence_ids, list) else [],
                "evidence_post_indices": _normalize_evidence_post_ids(
                    evidence_ids,
                    post_id_to_index=post_id_to_index,
                    limit=MAX_EVIDENCE_PER_ANCHOR,
                ),
                "evidence_modality": _normalize_evidence_modality(item.get("evidence_modality")),
                "confidence": str(item.get("confidence") or "").strip(),
                "rationale": str(item.get("rationale") or "").strip(),
            }
        )
        seen_labels.add(label_key)
        if len(out) >= limit_items:
            break
    return out


def _extract_final_profile_domain_rows(final_profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    active_domains = {
        str(domain or "").strip()
        for domain in list(final_profile.get("active_domains") or [])
        if str(domain or "").strip()
    }
    profile_obj = final_profile.get("profile")
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(profile_obj, dict):
        return out
    for domain in DOMAIN_NAMES:
        raw_row = profile_obj.get(domain)
        row = dict(raw_row) if isinstance(raw_row, dict) else {}
        row["domain"] = domain
        row["active"] = bool(
            domain in active_domains
            or row.get("active") is True
            or list(row.get("stable_interests") or [])
            or list(row.get("recent_interests") or [])
        )
        row.setdefault("stable_interests", [])
        row.setdefault("recent_interests", [])
        row.setdefault("weak_or_cautionary_interests", [])
        out[domain] = row
    return out


def _profile_is_active(profile: dict[str, Any]) -> bool:
    if isinstance(profile.get("active"), bool):
        return bool(profile.get("active"))
    status = str(profile.get("status") or profile.get("domain_status") or "").strip().lower()
    return status == "active"


def _guess_image_mime(image_url: str) -> str:
    url = (image_url or "").strip()
    if not url:
        return ""
    split = urlsplit(url)
    guess_target = split.path or url
    mime, _ = mimetypes.guess_type(guess_target)
    if not mime:
        mime, _ = mimetypes.guess_type(url)
    return str(mime or "").strip().lower()
