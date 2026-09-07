from __future__ import annotations

import json
from typing import Any


RUBRIC_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for domain-level user-interest profiling from social-media posts.

Your job is to evaluate ACTIVE-domain profiling quality only.

Important principles:
1. Reward correctness of core interest anchors, not wording similarity.
2. Do not reward unsupported claims, even if they sound semantically related.
3. Do not reward weak, fragmented, or one-off signals as core profile interests.
4. Favor fewer, better-supported anchors over broad but weak guesses.
5. Evaluate whether the summary faithfully reflects supported interests.
6. Use the gold core anchors and gold notes as the reference target.
7. Output strict JSON only.

Scoring rubric (0-5 each):
- anchor_correctness
- anchor_coverage
- evidence_grounding
- summary_faithfulness

Definitions:
- anchor_correctness: whether predicted anchors reflect the true core interests in this domain.
- anchor_coverage: whether the prediction covers the main gold anchors without missing important ones.
- evidence_grounding: whether claimed anchors and summary are supported by the cited posts and the post content.
- summary_faithfulness: whether the summary accurately reflects the supported interests and avoids hallucinations.

Return strict JSON with the required schema.
"""


RUBRIC_JUDGE_SCHEMA = {
    "task_id": "string",
    "status_correct": True,
    "status_reason": "string",
    "anchor_correctness": 0,
    "anchor_coverage": 0,
    "evidence_grounding": 0,
    "summary_faithfulness": 0,
    "unsupported_anchor_labels": ["string"],
    "missed_gold_anchor_labels": ["string"],
    "unsupported_summary_claims": ["string"],
    "judge_total_score": 0,
    "brief_rationale": "string",
}

RUBRIC_JUDGE_SCHEMA_HINT = json.dumps(RUBRIC_JUDGE_SCHEMA, ensure_ascii=False, indent=2)


def _format_posts(posts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for fallback_idx, post in enumerate(posts):
        idx = post.get("_line_index", fallback_idx)
        post_id = str(post.get("post_id") or "").strip()
        created_at = str(post.get("created_at") or "").strip()
        text = str(post.get("text") or "").strip()
        hashtags = post.get("hashtags") or []
        mentions = post.get("mentions") or []

        lines.append(f"[Post index={idx}] post_id={post_id} created_at={created_at}")
        lines.append(f"text: {text}" if text else "text:")
        if hashtags:
            lines.append(f"hashtags: {hashtags}")
        if mentions:
            lines.append(f"mentions: {mentions}")
        lines.append("")
    return "\n".join(lines).strip()


def render_rubric_judge_user_prompt(judge_input: dict[str, Any]) -> str:
    task_id = str(judge_input.get("task_id") or "")
    domain = str(judge_input.get("domain") or "")
    domain_definition = str(judge_input.get("domain_definition") or "")
    posts = judge_input.get("posts") or []
    posts_with_indices = _format_posts(posts)

    gold_payload = judge_input.get("gold") or {}
    pred_payload = judge_input.get("prediction") or {}

    gold_json = json.dumps(gold_payload, ensure_ascii=False, indent=2)
    pred_json = json.dumps(pred_payload, ensure_ascii=False, indent=2)

    return f"""Task ID: {task_id}

Domain:
{domain}

Domain definition:
{domain_definition}

Posts:
{posts_with_indices}

Gold reference:
{gold_json}

Model prediction:
{pred_json}

Evaluate the prediction against the gold reference and posts.
Return strict JSON only.
"""
