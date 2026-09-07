from __future__ import annotations

from typing import Any

from ..evaluator.adapters import pred_row_to_canonical, task_to_canonical
from ..evaluator.precheck import validate_prediction
from ..evaluator.task_scoring import score_anchor_matches
from .normalization import _active_prf, _prediction_row_is_cacheable, _prf_from_counts
from .specs import DomainEvalTask


class MetricsMixin:
    def _compute_metrics(self, *, tasks: list[DomainEvalTask], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        anchor_match_client = self._build_anchor_match_client()

        status_gold: list[str] = []
        status_pred: list[str] = []
        per_task_overview: list[dict[str, Any]] = []

        anchor_tp = 0
        anchor_fp = 0
        anchor_fn = 0
        n_active_tasks = 0
        n_tasks_total = len(tasks)
        n_tasks_skipped = 0

        for task in tasks:
            task_view = task_to_canonical(task)
            pred_row = predictions.get(task_view.task_id, {})
            skip_reason = ""
            if not _prediction_row_is_cacheable(pred_row):
                error = str((pred_row or {}).get("error") or "").strip()
                if error:
                    skip_reason = error
                else:
                    skip_reason = "stale_prediction_cache"
            if skip_reason:
                n_tasks_skipped += 1
                per_task_overview.append(
                    {
                        "task_id": task_view.task_id,
                        "domain": task_view.domain,
                        "gold_status": task_view.gold_status,
                        "skipped_from_metrics": True,
                        "skip_reason": skip_reason,
                    }
                )
                continue
            pred_view = pred_row_to_canonical(pred_row)
            try:
                precheck = validate_prediction(pred_view, task_view.posts)
                cleaned_pred = precheck.cleaned_prediction
                anchor_match = score_anchor_matches(
                    task=task_view,
                    pred=cleaned_pred,
                    precheck=precheck,
                    anchor_match_client=anchor_match_client,
                )
            except Exception as exc:
                n_tasks_skipped += 1
                per_task_overview.append(
                    {
                        "task_id": task_view.task_id,
                        "domain": task_view.domain,
                        "gold_status": task_view.gold_status,
                        "skipped_from_metrics": True,
                        "skip_reason": f"scoring_exception: {exc}",
                    }
                )
                continue

            status_gold.append(task_view.gold_status)
            status_pred.append(cleaned_pred.status)

            if task_view.gold_status == "active":
                n_active_tasks += 1
                anchor_tp += int(anchor_match.tp)
                anchor_fp += int(anchor_match.fp)
                anchor_fn += int(anchor_match.fn)

            per_task_overview.append(
                {
                    "task_id": task_view.task_id,
                    "domain": task_view.domain,
                    "gold_status": task_view.gold_status,
                    "pred_status": cleaned_pred.status,
                    "status_correct": task_view.gold_status == cleaned_pred.status,
                    "precheck_valid": precheck.valid,
                    "precheck_hard_fail": precheck.hard_fail,
                    "format_errors": list(precheck.format_errors),
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
            )

        active_precision, active_recall, active_f1 = _active_prf(status_gold, status_pred)
        interest_tag_precision, interest_tag_recall, interest_tag_f1 = _prf_from_counts(
            anchor_tp,
            anchor_fp,
            anchor_fn,
        )

        metrics = {
            "scoring_mode": "f1_only",
            "n_tasks": len(status_gold),
            "n_tasks_total": n_tasks_total,
            "n_tasks_skipped": n_tasks_skipped,
            "n_active_tasks": n_active_tasks,
            "active_precision": float(active_precision),
            "active_recall": float(active_recall),
            "active_f1": float(active_f1),
            "interest_tag_precision": float(interest_tag_precision),
            "interest_tag_recall": float(interest_tag_recall),
            "interest_tag_f1": float(interest_tag_f1),
            "active_anchor_match_precision": float(interest_tag_precision),
            "active_anchor_match_recall": float(interest_tag_recall),
            "active_anchor_match_f1": float(interest_tag_f1),
            "anchor_tp": int(anchor_tp),
            "anchor_fp": int(anchor_fp),
            "anchor_fn": int(anchor_fn),
            "active_anchor_match_tp": int(anchor_tp),
            "active_anchor_match_fp": int(anchor_fp),
            "active_anchor_match_fn": int(anchor_fn),
            "per_task_overview": per_task_overview,
        }
        return metrics
