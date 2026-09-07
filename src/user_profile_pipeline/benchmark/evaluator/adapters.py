from __future__ import annotations

from typing import Any

from .schemas import CanonicalAnchor, CanonicalPrediction, CanonicalTaskView


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "active":
        return "active"
    if text == "inactive":
        return "inactive"
    if "inactive" in text:
        return "inactive"
    if "active" in text:
        return "active"
    return "inactive"


def _normalize_indices(value: Any, *, limit: int | None = None) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx not in out:
            out.append(idx)
        if limit and len(out) >= limit:
            break
    return out


def _normalize_anchor_item(raw: Any, *, source_bucket: str) -> CanonicalAnchor | None:
    if not isinstance(raw, dict):
        label = str(_get(raw, "label", "") or _get(raw, "interest", "")).strip()
        evidence = _normalize_indices(_get(raw, "evidence_post_indices", []) or [], limit=2)
    else:
        label = str(raw.get("label") or raw.get("interest") or "").strip()
        evidence = _normalize_indices(
            raw.get("evidence_post_indices")
            or raw.get("representative_evidence_post_indices")
            or raw.get("evidence_indices")
            or [],
            limit=2,
        )
    if not label:
        return None
    return CanonicalAnchor(
        label=label,
        evidence_post_indices=evidence,
        source_bucket=source_bucket,
    )


def merge_anchor_buckets(long_anchors: list[Any], short_anchors: list[Any]) -> list[CanonicalAnchor]:
    out: list[CanonicalAnchor] = []
    for raw in long_anchors or []:
        anchor = _normalize_anchor_item(raw, source_bucket="long")
        if anchor is not None:
            out.append(anchor)
    for raw in short_anchors or []:
        anchor = _normalize_anchor_item(raw, source_bucket="short")
        if anchor is not None:
            out.append(anchor)
    return out


def _normalize_merged_anchors(items: Any) -> list[CanonicalAnchor]:
    if not isinstance(items, list):
        return []
    out: list[CanonicalAnchor] = []
    for item in items:
        anchor = _normalize_anchor_item(item, source_bucket="merged")
        if anchor is not None:
            out.append(anchor)
    return out


def task_to_canonical(task: Any) -> CanonicalTaskView:
    debug_meta = _get(task, "debug_meta", {})
    notes_parts: list[str] = []
    if isinstance(debug_meta, dict):
        weak = debug_meta.get("dropped_weak_interest_labels") or []
        uncertainty = str(debug_meta.get("uncertainty_note") or "").strip()
        if weak:
            notes_parts.append(f"Weak signals not to reward: {weak}")
        if uncertainty:
            notes_parts.append(f"Uncertainty: {uncertainty}")
    notes = "\n".join(notes_parts)

    return CanonicalTaskView(
        task_id=str(_get(task, "task_id", "") or ""),
        user_id=str(_get(task, "user_id", "") or ""),
        domain=str(_get(task, "domain", "") or ""),
        domain_definition=str(_get(task, "domain_definition", "") or ""),
        posts=list(_get(task, "posts", []) or []),
        gold_status=_normalize_status(_get(task, "gold_status", "inactive")),
        gold_interest_anchors=merge_anchor_buckets(
            list(_get(task, "gold_long_term_anchors", []) or []),
            list(_get(task, "gold_short_term_anchors", []) or []),
        ),
        gold_representative_evidence_post_indices=_normalize_indices(
            _get(task, "gold_domain_representative_evidence_post_indices", [])
            or _get(task, "gold_representative_evidence_post_indices", [])
            or [],
            limit=4,
        ),
        gold_summary_natural=str(_get(task, "gold_summary_natural", "") or "").strip(),
        gold_notes=notes,
    )


def _anchors_from_pred_row(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = row.get(key)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def pred_row_to_canonical(row: dict[str, Any]) -> CanonicalPrediction:
    pred_long = _anchors_from_pred_row(row, "pred_long_term_anchors")
    pred_short = _anchors_from_pred_row(row, "pred_short_term_anchors")
    merged_pred = _normalize_merged_anchors(
        row.get("pred_interest_anchors")
        or row.get("interest_anchors")
        or [],
    )

    anchors = merged_pred if merged_pred else merge_anchor_buckets(pred_long, pred_short)

    return CanonicalPrediction(
        status=_normalize_status(row.get("pred_status") or row.get("status")),
        interest_anchors=anchors,
        summary_natural_pred=str(
            row.get("pred_summary_natural")
            or row.get("summary_natural_pred")
            or row.get("summary")
            or ""
        ).strip(),
        summary_support_post_indices=_normalize_indices(
            row.get("pred_summary_support_post_indices")
            or row.get("summary_support_post_indices")
            or row.get("pred_evidence_post_indices")
            or [],
            limit=4,
        ),
    )
