from __future__ import annotations

from contextlib import contextmanager
import json
import os
from typing import Any

from ...multimodal_context import apply_image_limit, build_posts_context_bundle
from .specs import DomainEvalTask, InterestAnchor, RETRY_CONTEXT_VERSION


def _build_posts_context(posts: list[dict[str, Any]], *, max_images: int) -> tuple[str, list[str]]:
    bundle = build_posts_context_bundle(posts)
    return bundle.text, apply_image_limit(bundle.image_urls, max_images)


def _build_post_block(post: dict[str, Any], *, max_images: int) -> tuple[str, list[str]]:
    return _build_posts_context([post], max_images=max_images)


def _build_posts_context_with_budget(posts: list[dict[str, Any]], *, max_images: int, max_chars: int | None) -> tuple[str, list[str]]:
    # Keep all posts in the prompt and only downgrade image density when needed.
    return _build_posts_context(posts, max_images=max_images)


def _extract_outer_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end >= start:
        return text[start:end + 1]
    return None


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


def _find_duplicate_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for raw in ids:
        item = str(raw or "").strip()
        if not item:
            continue
        if item in seen and item not in duplicates:
            duplicates.append(item)
            continue
        seen.add(item)
    return duplicates


def _raise_on_duplicate_ids(ids: list[str], *, label: str) -> None:
    duplicates = _find_duplicate_ids(ids)
    if not duplicates:
        return
    preview = ", ".join(duplicates[:10])
    suffix = "" if len(duplicates) <= 10 else f" ... (+{len(duplicates) - 10} more)"
    raise ValueError(f"Duplicate {label} detected: {preview}{suffix}")


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


def _looks_like_data_inspection_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    return "datainspectionfailed" in text or "data_inspection_failed" in text or "inappropriate content" in text


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


def _context_limit_lowres_max_dim() -> int:
    raw = str(os.environ.get("BENCHMARK_CONTEXT_LIMIT_LOWRES_MAX_DIM", "512") or "512").strip()
    try:
        return max(32, int(raw))
    except Exception:
        return 512


@contextmanager
def _temporary_inline_image_settings(
    *,
    max_dim: int | None = None,
    reencode: bool | None = None,
) -> Any:
    updates: dict[str, str] = {}
    if max_dim is not None:
        updates["BENCHMARK_INLINE_IMAGE_MAX_DIM"] = str(max_dim)
    if reencode is not None:
        updates["BENCHMARK_INLINE_IMAGE_REENCODE"] = "1" if reencode else "0"

    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


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


def _normalize_anchor_items(value: Any, *, limit_items: int | None, limit_evidence: int) -> list[InterestAnchor]:
    if not isinstance(value, list):
        return []
    out: list[InterestAnchor] = []
    for item in value:
        label = ""
        evidence: list[int] = []
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("interest") or "").strip()
            evidence = _normalize_indices(
                item.get("evidence_post_indices")
                or item.get("representative_evidence_post_indices")
                or item.get("evidence_indices")
                or [],
                limit=limit_evidence,
            )
        else:
            label = str(item or "").strip()
        if not label:
            continue
        out.append(InterestAnchor(label=label, evidence_post_indices=evidence))
        if limit_items is not None and len(out) >= limit_items:
            break
    return out


def _normalize_prediction(
    *,
    task: DomainEvalTask,
    parsed: dict[str, Any] | None,
    raw_text: str,
    model_name: str,
    error: str = "",
) -> dict[str, Any]:
    parsed = parsed or {}
    status = _normalize_status(parsed.get("status"))
    # summary is now optional and ignored by the main benchmark logic.
    summary = str(
        parsed.get("summary_natural_pred")
        or parsed.get("summary")
        or parsed.get("domain_summary")
        or ""
    ).strip()
    long_anchors = _normalize_anchor_items(
        parsed.get("long_term_interest_tags")
        or parsed.get("long_term_interest_anchors"),
        limit_items=None,
        limit_evidence=2,
    )
    short_anchors = _normalize_anchor_items(
        parsed.get("short_term_interest_tags")
        or parsed.get("short_term_interest_anchors"),
        limit_items=None,
        limit_evidence=2,
    )
    summary_support = _normalize_indices(
        parsed.get("summary_support_post_indices")
        or parsed.get("evidence_post_indices")
        or parsed.get("evidence_post_indices_representative")
        or [],
        limit=4,
    )
    # hard canonicalization for inactive predictions
    if status == "inactive":
        long_anchors = []
        short_anchors = []
        summary = ""
        summary_support = []
    return {
        "task_id": task.task_id,
        "source_tag": task.source_tag,
        "user_id": task.user_id,
        "domain": task.domain,
        "model": model_name,
        "pred_status": status,
        "pred_summary_natural": summary,
        "pred_long_term_anchors": [anchor.__dict__ for anchor in long_anchors],
        "pred_short_term_anchors": [anchor.__dict__ for anchor in short_anchors],
        "pred_summary_support_post_indices": summary_support,
        "raw_response_text": raw_text,
        "error": error,
    }


def _fallback_prediction(task: DomainEvalTask, *, model_name: str, error: str) -> dict[str, Any]:
    return _normalize_prediction(
        task=task,
        parsed={
            "status": "inactive",
            "summary_natural_pred": "",
            "long_term_interest_tags": [],
            "short_term_interest_tags": [],
            "summary_support_post_indices": [],
        },
        raw_text="",
        model_name=model_name,
        error=error,
    )


def _prediction_row_is_cacheable(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    if bool(str(row.get("error") or "").strip()):
        return False
    retry_mode = str(row.get("retry_mode") or "").strip()
    if retry_mode in {"compact_context", "compact_context_text_redacted"}:
        if str(row.get("retry_context_version") or "").strip() != RETRY_CONTEXT_VERSION:
            return False
    return True


def _extract_vertex_text(response_obj: dict[str, Any]) -> str:
    candidates = response_obj.get("candidates") or []
    texts: list[str] = []
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
    return "\n".join(texts).strip()


def _parse_prediction_json(raw_text: str) -> dict[str, Any] | None:
    stripped = raw_text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    extracted = _extract_outer_json_object(stripped)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _active_prf(gold: list[str], pred: list[str]) -> tuple[float, float, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == "active" and p == "active")
    fp = sum(1 for g, p in zip(gold, pred) if g != "active" and p == "active")
    fn = sum(1 for g, p in zip(gold, pred) if g == "active" and p != "active")
    return _prf_from_counts(tp, fp, fn)
