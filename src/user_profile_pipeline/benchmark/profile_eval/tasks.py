from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain_llm.pack_builder import DOMAIN_DEFINITIONS
from .io import _load_json, _load_posts_with_indices
from .normalization import _normalize_indices, _normalize_status
from .specs import DomainEvalTask, InterestAnchor


def _parse_gold_anchor_items(items: Any) -> list[InterestAnchor]:
    if not isinstance(items, list):
        return []
    out: list[InterestAnchor] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("interest") or "").strip()
        if not label:
            continue
        indices = _normalize_indices(item.get("evidence_post_indices") or [], limit=2)
        out.append(InterestAnchor(label=label, evidence_post_indices=indices))
    return out
def build_tasks_from_gold_exports(*, gold_export_paths: list[Path], posts_by_user: dict[str, Path], max_posts: int | None) -> list[DomainEvalTask]:
    tasks: list[DomainEvalTask] = []
    seen_task_ids: set[str] = set()
    for gold_path in gold_export_paths:
        payload = _load_json(gold_path)
        user_id = str(payload.get("user_id", "")).strip()
        if not user_id:
            raise ValueError(f"user_id missing in gold export: {gold_path}")
        posts_path = posts_by_user.get(user_id)
        if not posts_path:
            raise ValueError(
                f"Posts JSONL not found for user_id={user_id}. Provide --posts-map or place posts under data/test/*/posts.jsonl"
            )
        posts = _load_posts_with_indices(posts_path, max_posts=max_posts)
        source_tag = gold_path.parent.name
        for domain_row in payload.get("domains", []) or []:
            domain = str(domain_row.get("domain", "")).strip()
            if not domain:
                continue
            task_id = f"{source_tag}__{user_id}__{domain}"
            if task_id in seen_task_ids:
                raise ValueError(
                    f"Duplicate task_id resolved from gold exports: {task_id}. "
                    f"gold_path={gold_path}"
                )
            seen_task_ids.add(task_id)
            tasks.append(
                DomainEvalTask(
                    task_id=task_id,
                    source_tag=source_tag,
                    user_id=user_id,
                    domain=domain,
                    domain_definition=DOMAIN_DEFINITIONS.get(domain, ""),
                    gold_status=_normalize_status(domain_row.get("status")),
                    gold_summary_natural=str(domain_row.get("summary_natural_gold") or domain_row.get("summary_atomic") or "").strip(),
                    gold_long_term_anchors=_parse_gold_anchor_items(domain_row.get("long_term_interests") or []),
                    gold_short_term_anchors=_parse_gold_anchor_items(domain_row.get("short_term_interests") or []),
                    gold_domain_representative_evidence_post_indices=_normalize_indices(
                        domain_row.get("domain_representative_evidence_post_indices")
                        or domain_row.get("domain_evidence_post_indices")
                        or [],
                        limit=4,
                    ),
                    posts=posts,
                    gold_path=str(gold_path),
                    debug_meta=dict(domain_row.get("debug_meta") or {}),
                )
            )
    return tasks
