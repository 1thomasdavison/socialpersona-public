from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from .schemas import CanonicalPrediction, CanonicalTaskView, PrecheckResult, TaskJudgeResult


def build_judge_input(task: CanonicalTaskView, pred: CanonicalPrediction) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "domain_definition": task.domain_definition,
        "posts": task.posts,
        "status_correct": task.gold_status == pred.status,
        "gold": {
            "status": task.gold_status,
            "interest_anchors": [
                {
                    "label": anchor.label,
                    "evidence_post_indices": anchor.evidence_post_indices,
                }
                for anchor in task.gold_interest_anchors
            ],
            "representative_evidence_post_indices": task.gold_representative_evidence_post_indices,
            "summary_natural": task.gold_summary_natural,
            "gold_notes": task.gold_notes,
        },
        "prediction": {
            "status": pred.status,
            "interest_anchors": [
                {
                    "label": anchor.label,
                    "evidence_post_indices": anchor.evidence_post_indices,
                    "source_bucket": anchor.source_bucket,
                }
                for anchor in pred.interest_anchors
            ],
            "summary_natural_pred": pred.summary_natural_pred,
            "summary_support_post_indices": pred.summary_support_post_indices,
        },
    }


def _score_0_to_5(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def _weighted_score(
    anchor_correctness: float,
    anchor_coverage: float,
    evidence_grounding: float,
    summary_faithfulness: float,
) -> float:
    return (
        0.35 * anchor_correctness
        + 0.25 * anchor_coverage
        + 0.25 * evidence_grounding
        + 0.15 * summary_faithfulness
    ) / 5.0 * 100.0


@dataclass
class AnchorMatchOutcome:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    exact: bool = False
    pairs: list[dict[str, str]] = None
    unmatched_pred_labels: list[str] = None
    unmatched_gold_labels: list[str] = None

    def __post_init__(self) -> None:
        if self.pairs is None:
            self.pairs = []
        if self.unmatched_pred_labels is None:
            self.unmatched_pred_labels = []
        if self.unmatched_gold_labels is None:
            self.unmatched_gold_labels = []


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _normalize_label_for_match(text: str) -> str:
    lowered = str(text or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", compact).strip()


def _anchor_match_force_llm(anchor_match_client: Any | None) -> bool:
    if anchor_match_client is None:
        return False
    if hasattr(anchor_match_client, "uses_llm_for_all_pairs"):
        try:
            return bool(anchor_match_client.uses_llm_for_all_pairs())
        except Exception:
            return False
    config = getattr(anchor_match_client, "config", None)
    return bool(getattr(config, "force_llm", False))


def _best_binary_matching(match_matrix: list[list[bool]]) -> list[tuple[int, int]]:
    if not match_matrix:
        return []
    n_gold = len(match_matrix)
    n_pred = len(match_matrix[0]) if match_matrix[0] else 0
    if n_pred == 0:
        return []

    @lru_cache(maxsize=None)
    def dp(i: int, used_mask: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if i >= n_gold:
            return 0, ()

        best_score, best_pairs = dp(i + 1, used_mask)
        for j in range(n_pred):
            if used_mask & (1 << j):
                continue
            if not bool(match_matrix[i][j]):
                continue
            tail_score, tail_pairs = dp(i + 1, used_mask | (1 << j))
            score = 1 + tail_score
            if score > best_score:
                best_score = score
                best_pairs = ((i, j),) + tail_pairs
        return best_score, best_pairs

    _, matched = dp(0, 0)
    return list(matched)


def _extract_anchor_labels(anchors: list[Any]) -> list[str]:
    out: list[str] = []
    for anchor in anchors:
        label = str(getattr(anchor, "label", "") or "").strip()
        if label:
            out.append(label)
    return out


def _labels_are_same_interest(
    *,
    task_id: str,
    domain: str,
    domain_definition: str,
    pred_label: str,
    gold_label: str,
    anchor_match_client: Any | None,
) -> bool:
    pred_norm = _normalize_label_for_match(pred_label)
    gold_norm = _normalize_label_for_match(gold_label)
    if pred_norm and pred_norm == gold_norm and (not _anchor_match_force_llm(anchor_match_client)):
        return True

    if anchor_match_client is None:
        return False

    if not hasattr(anchor_match_client, "is_same_interest"):
        return False

    try:
        return bool(
            anchor_match_client.is_same_interest(
                task_id=task_id,
                domain=domain,
                domain_definition=domain_definition,
                pred_label=pred_label,
                gold_label=gold_label,
            )
        )
    except Exception:
        return False


def score_anchor_matches(
    *,
    task: CanonicalTaskView,
    pred: CanonicalPrediction,
    precheck: PrecheckResult,
    anchor_match_client: Any | None,
) -> AnchorMatchOutcome:
    if task.gold_status != "active":
        return AnchorMatchOutcome()

    gold_labels = _extract_anchor_labels(task.gold_interest_anchors)
    if precheck.hard_fail or pred.status != "active":
        pred_labels: list[str] = []
    else:
        pred_labels = _extract_anchor_labels(pred.interest_anchors)

    if not gold_labels and not pred_labels:
        return AnchorMatchOutcome(precision=1.0, recall=1.0, f1=1.0, exact=True)

    if not gold_labels:
        fp = len(pred_labels)
        return AnchorMatchOutcome(
            tp=0,
            fp=fp,
            fn=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            exact=False,
            unmatched_pred_labels=list(pred_labels),
            unmatched_gold_labels=[],
        )

    if not pred_labels:
        fn = len(gold_labels)
        return AnchorMatchOutcome(
            tp=0,
            fp=0,
            fn=fn,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            exact=False,
            unmatched_pred_labels=[],
            unmatched_gold_labels=list(gold_labels),
        )

    matrix: list[list[bool]] = []
    for gold_label in gold_labels:
        row: list[bool] = []
        for pred_label in pred_labels:
            row.append(
                _labels_are_same_interest(
                    task_id=task.task_id,
                    domain=task.domain,
                    domain_definition=task.domain_definition,
                    pred_label=pred_label,
                    gold_label=gold_label,
                    anchor_match_client=anchor_match_client,
                )
            )
        matrix.append(row)

    matched_pairs_idx = _best_binary_matching(matrix)
    matched_gold_indices = {gi for gi, _ in matched_pairs_idx}
    matched_pred_indices = {pj for _, pj in matched_pairs_idx}
    matched_pairs = [
        {
            "gold_label": gold_labels[gi],
            "pred_label": pred_labels[pj],
        }
        for gi, pj in matched_pairs_idx
    ]

    tp = len(matched_pairs)
    fp = len(pred_labels) - tp
    fn = len(gold_labels) - tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    exact = (fp == 0 and fn == 0)

    unmatched_pred_labels = [label for idx, label in enumerate(pred_labels) if idx not in matched_pred_indices]
    unmatched_gold_labels = [label for idx, label in enumerate(gold_labels) if idx not in matched_gold_indices]

    return AnchorMatchOutcome(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        exact=exact,
        pairs=matched_pairs,
        unmatched_pred_labels=unmatched_pred_labels,
        unmatched_gold_labels=unmatched_gold_labels,
    )


def score_single_task(
    *,
    task: CanonicalTaskView,
    pred: CanonicalPrediction,
    precheck: PrecheckResult,
    judge_client: Any,
    anchor_match_client: Any | None = None,
) -> TaskJudgeResult:
    working_pred = precheck.cleaned_prediction
    anchor_match = score_anchor_matches(
        task=task,
        pred=working_pred,
        precheck=precheck,
        anchor_match_client=anchor_match_client,
    )
    anchor_match_kwargs = {
        "anchor_match_tp": int(anchor_match.tp),
        "anchor_match_fp": int(anchor_match.fp),
        "anchor_match_fn": int(anchor_match.fn),
        "anchor_match_precision": float(anchor_match.precision),
        "anchor_match_recall": float(anchor_match.recall),
        "anchor_match_f1": float(anchor_match.f1),
        "anchor_match_exact": bool(anchor_match.exact),
        "anchor_match_pairs": list(anchor_match.pairs),
        "unmatched_pred_anchor_labels": list(anchor_match.unmatched_pred_labels),
        "unmatched_gold_anchor_labels": list(anchor_match.unmatched_gold_labels),
    }

    if precheck.hard_fail:
        return TaskJudgeResult(
            task_id=task.task_id,
            user_id=task.user_id,
            domain=task.domain,
            gold_status=task.gold_status,
            pred_status=working_pred.status,
            precheck_valid=False,
            format_errors=precheck.format_errors,
            status_correct=False,
            anchor_correctness=0.0,
            anchor_coverage=0.0,
            evidence_grounding=0.0,
            summary_faithfulness=0.0,
            judge_total_score=0.0,
            final_task_score=0.0,
            unsupported_anchor_labels=[],
            missed_gold_anchor_labels=[],
            unsupported_summary_claims=[],
            brief_rationale="hard_fail_precheck",
            **anchor_match_kwargs,
        )

    status_correct = (task.gold_status == working_pred.status)

    if task.gold_status != "active":
        return TaskJudgeResult(
            task_id=task.task_id,
            user_id=task.user_id,
            domain=task.domain,
            gold_status=task.gold_status,
            pred_status=working_pred.status,
            precheck_valid=precheck.valid,
            format_errors=precheck.format_errors,
            status_correct=status_correct,
            anchor_correctness=0.0,
            anchor_coverage=0.0,
            evidence_grounding=0.0,
            summary_faithfulness=0.0,
            judge_total_score=0.0,
            final_task_score=0.0,
            unsupported_anchor_labels=[],
            missed_gold_anchor_labels=[],
            unsupported_summary_claims=[],
            brief_rationale="inactive_task_not_scored",
            **anchor_match_kwargs,
        )

    if working_pred.status != "active":
        return TaskJudgeResult(
            task_id=task.task_id,
            user_id=task.user_id,
            domain=task.domain,
            gold_status=task.gold_status,
            pred_status=working_pred.status,
            precheck_valid=precheck.valid,
            format_errors=precheck.format_errors,
            status_correct=False,
            anchor_correctness=0.0,
            anchor_coverage=0.0,
            evidence_grounding=0.0,
            summary_faithfulness=0.0,
            judge_total_score=0.0,
            final_task_score=0.0,
            unsupported_anchor_labels=[],
            missed_gold_anchor_labels=[anchor.label for anchor in task.gold_interest_anchors],
            unsupported_summary_claims=[],
            brief_rationale="gold_active_but_prediction_not_active",
            **anchor_match_kwargs,
        )

    judge_input = build_judge_input(task, working_pred)
    try:
        judge_json = judge_client.judge_task(judge_input)
    except Exception as exc:
        return TaskJudgeResult(
            task_id=task.task_id,
            user_id=task.user_id,
            domain=task.domain,
            gold_status=task.gold_status,
            pred_status=working_pred.status,
            precheck_valid=precheck.valid,
            format_errors=[*precheck.format_errors, f"judge_error={exc.__class__.__name__}"],
            status_correct=False,
            anchor_correctness=0.0,
            anchor_coverage=0.0,
            evidence_grounding=0.0,
            summary_faithfulness=0.0,
            judge_total_score=0.0,
            final_task_score=0.0,
            unsupported_anchor_labels=[],
            missed_gold_anchor_labels=[anchor.label for anchor in task.gold_interest_anchors],
            unsupported_summary_claims=[],
            brief_rationale=f"judge_error: {exc}",
            **anchor_match_kwargs,
        )

    anchor_correctness = _score_0_to_5(judge_json.get("anchor_correctness"))
    anchor_coverage = _score_0_to_5(judge_json.get("anchor_coverage"))
    evidence_grounding = _score_0_to_5(judge_json.get("evidence_grounding"))
    summary_faithfulness = _score_0_to_5(judge_json.get("summary_faithfulness"))

    final_task_score = _weighted_score(
        anchor_correctness,
        anchor_coverage,
        evidence_grounding,
        summary_faithfulness,
    )

    judge_total_raw = judge_json.get("judge_total_score")
    try:
        judge_total_score = float(judge_total_raw)
    except (TypeError, ValueError):
        judge_total_score = anchor_correctness + anchor_coverage + evidence_grounding + summary_faithfulness

    return TaskJudgeResult(
        task_id=task.task_id,
        user_id=task.user_id,
        domain=task.domain,
        gold_status=task.gold_status,
        pred_status=working_pred.status,
        precheck_valid=precheck.valid,
        format_errors=precheck.format_errors,
        status_correct=bool(judge_json.get("status_correct", True)),
        anchor_correctness=anchor_correctness,
        anchor_coverage=anchor_coverage,
        evidence_grounding=evidence_grounding,
        summary_faithfulness=summary_faithfulness,
        judge_total_score=judge_total_score,
        final_task_score=final_task_score,
        unsupported_anchor_labels=[str(x) for x in judge_json.get("unsupported_anchor_labels", []) if str(x).strip()],
        missed_gold_anchor_labels=[str(x) for x in judge_json.get("missed_gold_anchor_labels", []) if str(x).strip()],
        unsupported_summary_claims=[str(x) for x in judge_json.get("unsupported_summary_claims", []) if str(x).strip()],
        brief_rationale=str(judge_json.get("brief_rationale") or "").strip(),
        **anchor_match_kwargs,
    )
