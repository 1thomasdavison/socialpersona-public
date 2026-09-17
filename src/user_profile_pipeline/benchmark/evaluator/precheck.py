from __future__ import annotations

from .schemas import CanonicalAnchor, CanonicalPrediction, PrecheckResult

MAX_EVIDENCE_PER_ANCHOR = 2
MAX_SUMMARY_SUPPORT = 4


VALID_STATUS = {"active", "inactive"}


def _dedupe_int_list(items: list[int], *, limit: int) -> list[int]:
    out: list[int] = []
    for item in items:
        value = int(item)
        if value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def _clean_prediction(pred: CanonicalPrediction) -> CanonicalPrediction:
    cleaned_anchors: list[CanonicalAnchor] = []
    for anchor in pred.interest_anchors:
        cleaned_anchors.append(
            CanonicalAnchor(
                label=str(anchor.label or "").strip(),
                evidence_post_indices=_dedupe_int_list(
                    [int(i) for i in anchor.evidence_post_indices],
                    limit=MAX_EVIDENCE_PER_ANCHOR,
                ),
                source_bucket=anchor.source_bucket,
            )
        )
    return CanonicalPrediction(
        status=str(pred.status or "").strip().lower(),
        interest_anchors=cleaned_anchors,
        summary_natural_pred=str(pred.summary_natural_pred or "").strip(),
        summary_support_post_indices=_dedupe_int_list(
            [int(i) for i in pred.summary_support_post_indices],
            limit=MAX_SUMMARY_SUPPORT,
        ),
    )


def validate_prediction(pred: CanonicalPrediction, posts: list[dict]) -> PrecheckResult:
    errors: list[str] = []
    max_idx = len(posts) - 1

    if pred.status not in VALID_STATUS:
        errors.append("invalid_status")

    for i, anchor in enumerate(pred.interest_anchors):
        if not anchor.label.strip():
            errors.append(f"empty_anchor_label[{i}]")

        if len(anchor.evidence_post_indices) > MAX_EVIDENCE_PER_ANCHOR:
            errors.append(f"too_many_anchor_evidence[{i}]")

        for idx in anchor.evidence_post_indices:
            if idx < 0 or idx > max_idx:
                errors.append(f"anchor_evidence_oob[{i}]={idx}")

    if len(pred.summary_support_post_indices) > MAX_SUMMARY_SUPPORT:
        errors.append("too_many_summary_support")

    for idx in pred.summary_support_post_indices:
        if idx < 0 or idx > max_idx:
            errors.append(f"summary_support_oob={idx}")

    hard_fail = (
        "invalid_status" in errors
        or any(e.startswith("anchor_evidence_oob") for e in errors)
        or any(e.startswith("summary_support_oob") for e in errors)
    )

    return PrecheckResult(
        valid=(len(errors) == 0),
        format_errors=errors,
        cleaned_prediction=_clean_prediction(pred),
        hard_fail=hard_fail,
    )
