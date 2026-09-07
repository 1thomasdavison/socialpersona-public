from __future__ import annotations

from typing import Any

from .adapters import pred_row_to_canonical, task_to_canonical
from .aggregate import aggregate_task_results
from .precheck import validate_prediction
from .task_scoring import build_judge_input, score_single_task


def score_model_predictions_with_rubric_judge(
    tasks: list[Any],
    predictions: dict[str, dict[str, Any]],
    judge_client: Any,
    anchor_match_client: Any | None = None,
) -> dict[str, Any]:
    task_results = []
    status_gold: list[str] = []
    status_pred: list[str] = []
    prepared_items: list[tuple[Any, Any, Any]] = []

    for task in tasks:
        task_view = task_to_canonical(task)
        pred_row = predictions.get(task_view.task_id, {})
        pred_view = pred_row_to_canonical(pred_row)
        precheck = validate_prediction(pred_view, task_view.posts)
        prepared_items.append((task_view, pred_view, precheck))

        status_gold.append(task_view.gold_status)
        status_pred.append(precheck.cleaned_prediction.status)

    if hasattr(judge_client, "prepare_batch"):
        judge_inputs_for_batch: list[dict[str, Any]] = []
        for task_view, _, precheck in prepared_items:
            cleaned_pred = precheck.cleaned_prediction
            if precheck.hard_fail:
                continue
            if task_view.gold_status != "active":
                continue
            if cleaned_pred.status != "active":
                continue
            judge_inputs_for_batch.append(build_judge_input(task_view, cleaned_pred))
        if judge_inputs_for_batch:
            judge_client.prepare_batch(judge_inputs_for_batch)

    for task_view, pred_view, precheck in prepared_items:
        result = score_single_task(
            task=task_view,
            pred=pred_view,
            precheck=precheck,
            judge_client=judge_client,
            anchor_match_client=anchor_match_client,
        )
        task_results.append(result)

    return aggregate_task_results(
        task_results=task_results,
        status_gold=status_gold,
        status_pred=status_pred,
        judge_model=str(getattr(judge_client.config, "model_name", "")),
        judge_prompt_version=str(getattr(judge_client.config, "prompt_version", "rubric_v1")),
    )
