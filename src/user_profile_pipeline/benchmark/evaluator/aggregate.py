from __future__ import annotations

from typing import Any

from .schemas import TaskJudgeResult


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _active_prf(gold: list[str], pred: list[str]) -> tuple[float, float, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == "active" and p == "active")
    fp = sum(1 for g, p in zip(gold, pred) if g != "active" and p == "active")
    fn = sum(1 for g, p in zip(gold, pred) if g == "active" and p != "active")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate_task_results(
    *,
    task_results: list[TaskJudgeResult],
    status_gold: list[str],
    status_pred: list[str],
    judge_model: str,
    judge_prompt_version: str,
) -> dict[str, Any]:
    active_results = [r for r in task_results if r.gold_status == "active"]

    active_precision, active_recall, active_f1 = _active_prf(status_gold, status_pred)
    anchor_tp = int(sum(r.anchor_match_tp for r in active_results))
    anchor_fp = int(sum(r.anchor_match_fp for r in active_results))
    anchor_fn = int(sum(r.anchor_match_fn for r in active_results))
    anchor_micro_precision, anchor_micro_recall, anchor_micro_f1 = _prf_from_counts(anchor_tp, anchor_fp, anchor_fn)
    anchor_macro_precision = _mean([r.anchor_match_precision for r in active_results])
    anchor_macro_recall = _mean([r.anchor_match_recall for r in active_results])
    anchor_macro_f1 = _mean([r.anchor_match_f1 for r in active_results])
    anchor_exact_accuracy = _mean([1.0 if r.anchor_match_exact else 0.0 for r in active_results])

    benchmark_main_score = _mean([r.final_task_score for r in active_results]) / 100.0 if active_results else 0.0

    per_task_overview: list[dict[str, Any]] = []
    for result in task_results:
        per_task_overview.append(
            {
                "task_id": result.task_id,
                "domain": result.domain,
                "gold_status": result.gold_status,
                "pred_status": result.pred_status,
                "status_correct": result.status_correct,
                "precheck_valid": result.precheck_valid,
                "format_errors": result.format_errors,
                "final_task_score": result.final_task_score,
                "anchor_correctness": result.anchor_correctness,
                "anchor_coverage": result.anchor_coverage,
                "evidence_grounding": result.evidence_grounding,
                "summary_faithfulness": result.summary_faithfulness,
                "unsupported_anchor_labels": result.unsupported_anchor_labels,
                "missed_gold_anchor_labels": result.missed_gold_anchor_labels,
                "unsupported_summary_claims": result.unsupported_summary_claims,
                "brief_rationale": result.brief_rationale,
                "anchor_match_tp": result.anchor_match_tp,
                "anchor_match_fp": result.anchor_match_fp,
                "anchor_match_fn": result.anchor_match_fn,
                "anchor_match_precision": result.anchor_match_precision,
                "anchor_match_recall": result.anchor_match_recall,
                "anchor_match_f1": result.anchor_match_f1,
                "anchor_match_exact": result.anchor_match_exact,
                "anchor_match_pairs": result.anchor_match_pairs,
                "unmatched_pred_anchor_labels": result.unmatched_pred_anchor_labels,
                "unmatched_gold_anchor_labels": result.unmatched_gold_anchor_labels,
            }
        )

    return {
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "n_tasks": len(task_results),
        "n_active_tasks": len(active_results),
        "active_precision": active_precision,
        "active_recall": active_recall,
        "active_f1": active_f1,
        "active_anchor_correctness_mean": _mean([r.anchor_correctness for r in active_results]),
        "active_anchor_coverage_mean": _mean([r.anchor_coverage for r in active_results]),
        "active_evidence_grounding_mean": _mean([r.evidence_grounding for r in active_results]),
        "active_summary_faithfulness_mean": _mean([r.summary_faithfulness for r in active_results]),
        "active_anchor_match_micro_precision": anchor_micro_precision,
        "active_anchor_match_micro_recall": anchor_micro_recall,
        "active_anchor_match_micro_f1": anchor_micro_f1,
        "active_anchor_match_macro_precision": anchor_macro_precision,
        "active_anchor_match_macro_recall": anchor_macro_recall,
        "active_anchor_match_macro_f1": anchor_macro_f1,
        "active_anchor_match_exact_accuracy": anchor_exact_accuracy,
        "active_anchor_match_tp": anchor_tp,
        "active_anchor_match_fp": anchor_fp,
        "active_anchor_match_fn": anchor_fn,
        "benchmark_main_score": benchmark_main_score,
        "per_task_overview": per_task_overview,
    }
