from __future__ import annotations

import json


ANCHOR_MATCH_SYSTEM_PROMPT = """You are an evaluator for domain-level user-interest anchor labels.

Given one gold anchor label and one predicted anchor label within the same domain,
decide whether they describe the same core user interest.

Judgment rules:
1. Return true only when both labels express the same core interest theme.
2. A wording rewrite, synonym, parent-child phrasing, or broad-vs-specific phrasing can be true when both labels clearly point to the same underlying user-interest cluster in this domain.
3. Examples that should usually be true: "hiking" vs "outdoor recreation", "coffee" vs "coffee culture", "anime" vs "anime fandom".
4. Examples that should usually be false: sibling interests that share only the same domain, such as "basketball" vs "camping", or labels with clearly different focus.
5. Be conservative. Return strict JSON only.
"""

ANCHOR_MATCH_SCHEMA = {
    "is_match": True,
    "reason": "string",
}

ANCHOR_MATCH_SCHEMA_HINT = json.dumps(ANCHOR_MATCH_SCHEMA, ensure_ascii=False, indent=2)


def render_anchor_match_user_prompt(
    *,
    task_id: str,
    domain: str,
    domain_definition: str,
    gold_label: str,
    pred_label: str,
) -> str:
    return f"""Task ID: {task_id}

Domain: {domain}
Domain definition: {domain_definition}

Gold anchor label:
{gold_label}

Predicted anchor label:
{pred_label}

Question:
Do these two labels describe the same core user interest in this domain?
Return strict JSON only.
"""
