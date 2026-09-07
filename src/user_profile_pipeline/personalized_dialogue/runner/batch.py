from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from ...benchmark.profile_eval.modes import _slugify
from ...benchmark.profile_eval.specs import DEFAULT_MODEL_SPECS
from ...dashscope_batch import (
    MAX_BATCH_REQUEST_BYTES,
    DashScopeBatchInferenceClient,
    build_openai_batch_chat_request,
    dashscope_batch_model_support_reason,
    extract_openai_chat_message_text,
    iter_jsonl_rows,
    request_json_size_bytes,
)
from ...image_text import VISUAL_MODE_NATIVE, ImageTextGenerator, normalize_visual_mode
from ..evaluator import DIALOGUE_SYSTEM_PROMPT, _render_turn1_prompt, _render_turn2_prompt
from .context import (
    _build_visual_context_bundle,
    _clear_inline_image_caches,
    _context_limit_lowres_max_dim,
    _prepare_image_urls_for_provider,
    _select_context_and_image_urls,
    _temporary_inline_image_settings,
)
from .judging import (
    DIALOGUE_JUDGE_SYSTEM_PROMPT,
    JudgePlaceholderError,
    _exception_text,
    _fallback_judge_result,
    _looks_like_context_limit_error,
    _normalize_judge_raw,
)


def _build_dialogue_batch_request(
    *,
    eval_task: dict[str, Any],
    spec: Any,
    custom_id: str,
    turn_id: int,
    previous_response: str = "",
    max_context_posts: int,
    max_context_chars: int,
    max_images: int,
    one_image_per_post: bool = False,
    lowres_max_dim: int = 0,
    context_bundle: Any | None = None,
) -> dict[str, Any]:
    context, image_urls = _select_context_and_image_urls(
        eval_task=eval_task,
        max_posts=max_context_posts,
        max_chars=max_context_chars,
        max_images=max_images,
        one_image_per_post=one_image_per_post,
        context_bundle=context_bundle,
    )
    user_prompt = (
        _render_turn1_prompt(eval_task=eval_task, context=context)
        if turn_id == 1
        else _render_turn2_prompt(eval_task=eval_task, context=context, turn1_response=previous_response)
    )
    _clear_inline_image_caches()
    try:
        with _temporary_inline_image_settings(
            max_dim=(lowres_max_dim if lowres_max_dim > 0 else None),
            reencode=(lowres_max_dim > 0),
        ):
            prepared_image_urls = _prepare_image_urls_for_provider(image_urls, provider="openai_compatible")
        row = build_openai_batch_chat_request(
            custom_id=custom_id,
            model=str(spec.api_model or spec.name),
            system_prompt=DIALOGUE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_urls=prepared_image_urls or None,
            temperature=float(spec.temperature),
            max_tokens=int(spec.max_tokens),
            force_json=False,
        )
        row_size = request_json_size_bytes(row)
        if row_size > MAX_BATCH_REQUEST_BYTES:
            raise ValueError(f"dialogue batch request exceeds 6MB: custom_id={custom_id} bytes={row_size}")
        return row
    finally:
        _clear_inline_image_caches()


def _run_dialogue_dashscope_batch(
    *,
    spec: Any,
    request_rows: list[dict[str, Any]],
    output_dir: Path,
    batch_name: str,
    poll_seconds: int,
    max_wait_seconds: int,
    completion_window: str,
    callback_url: str,
) -> tuple[dict[str, str], dict[str, str]]:
    if not request_rows:
        return {}, {}
    client = DashScopeBatchInferenceClient(
        base_url=str(spec.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key_env=str(spec.api_key_env or "ALI_BAILIAN"),
        timeout_seconds=int(os.environ.get("BENCHMARK_PROVIDER_TIMEOUT_SECONDS", "120") or 120),
    )
    run_tag = f"{_slugify(spec.name)}_{batch_name}_{int(time.time())}"
    run_dir = output_dir / "_dashscope_batch_runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.jsonl"
    with input_path.open("w", encoding="utf-8") as fh:
        for row in request_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    upload = client.upload_batch_file(local_path=input_path)
    metadata: dict[str, Any] = {}
    if str(callback_url or "").strip():
        metadata["ds_batch_finish_callback"] = str(callback_url).strip()
    batch = client.create_batch(
        input_file_id=str(upload.get("id") or ""),
        endpoint="/v1/chat/completions",
        completion_window=str(completion_window or "24h").strip() or "24h",
        metadata=metadata,
    )
    batch_final = client.wait_batch(
        batch_id=str(batch.get("id") or ""),
        poll_seconds=int(poll_seconds),
        max_wait_seconds=int(max_wait_seconds),
    )
    (run_dir / "upload.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "batch.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "batch_final.json").write_text(json.dumps(batch_final, ensure_ascii=False, indent=2), encoding="utf-8")
    output_file_id = str(batch_final.get("output_file_id") or "").strip()
    error_file_id = str(batch_final.get("error_file_id") or "").strip()
    if output_file_id:
        client.download_file_content(file_id=output_file_id, local_path=run_dir / "result.jsonl")
    if error_file_id:
        client.download_file_content(file_id=error_file_id, local_path=run_dir / "errors.jsonl")

    responses: dict[str, str] = {}
    errors: dict[str, str] = {}
    result_path = run_dir / "result.jsonl"
    if result_path.exists():
        for row in iter_jsonl_rows(result_path):
            custom_id = str(row.get("custom_id") or "").strip()
            if not custom_id:
                continue
            error = row.get("error")
            if error:
                errors[custom_id] = json.dumps(error, ensure_ascii=False)
                continue
            response = row.get("response") or {}
            body = (response.get("body") or {}) if isinstance(response, dict) else {}
            try:
                responses[custom_id] = extract_openai_chat_message_text(body if isinstance(body, dict) else {})
            except Exception as exc:
                errors[custom_id] = f"batch_parse_failed: {exc}"
    error_path = run_dir / "errors.jsonl"
    if error_path.exists():
        for row in iter_jsonl_rows(error_path):
            custom_id = str(row.get("custom_id") or "").strip()
            error = row.get("error")
            if custom_id and error:
                errors[custom_id] = json.dumps(error, ensure_ascii=False)
    return responses, errors


def _generate_dialogue_predictions_batch(
    *,
    spec: Any,
    task_rows: list[dict[str, Any]],
    output_dir: Path,
    max_context_posts: int,
    max_context_chars: int,
    max_images: int,
    poll_seconds: int,
    max_wait_seconds: int,
    completion_window: str,
    callback_url: str,
    visual_mode: str,
    image_text_max_tokens: int,
    image_text_timeout_seconds: int,
    force: bool,
    image_text_cache_root: Path | None,
    image_text_cache_only: bool,
) -> dict[str, dict[str, Any]]:
    normalized_visual_mode = normalize_visual_mode(visual_mode)
    support_reason = dashscope_batch_model_support_reason(
        model=str(spec.api_model or spec.name),
        has_images=bool(spec.multimodal and normalized_visual_mode == VISUAL_MODE_NATIVE),
        base_url=str(spec.base_url or ""),
    )
    if support_reason:
        raise ValueError(support_reason)

    loaded_tasks: dict[str, dict[str, Any]] = {}
    gold_refs: dict[str, dict[str, Any]] = {}
    image_text_generators: dict[str, ImageTextGenerator] = {}
    visual_bundle_cache: dict[tuple[str, str, str], Any] = {}
    context_bundle_by_user: dict[str, Any] = {}
    for task_row in task_rows:
        user_id = str(task_row["user_id"])
        loaded_tasks[user_id] = json.loads(Path(task_row["eval_task_path"]).read_text(encoding="utf-8"))
        gold_refs[user_id] = json.loads(Path(task_row["gold_ref_path"]).read_text(encoding="utf-8"))
        bundle = _build_visual_context_bundle(
            gold_ref=gold_refs[user_id],
            eval_task=loaded_tasks[user_id],
            spec=spec,
            visual_mode=normalized_visual_mode,
            output_dir=output_dir,
            force=force,
            image_text_generators=image_text_generators,
            bundle_cache=visual_bundle_cache,
            image_text_max_tokens=image_text_max_tokens,
            image_text_timeout_seconds=image_text_timeout_seconds,
            image_text_cache_root=image_text_cache_root,
            image_text_cache_only=image_text_cache_only,
        )
        if bundle is not None:
            context_bundle_by_user[user_id] = bundle

    lowres_max_dim = _context_limit_lowres_max_dim()
    stages = [
        ("full_images", False, 0, ""),
        ("one_image_per_post", True, 0, "one_image_per_post"),
        (
            f"one_image_per_post_lowres_{lowres_max_dim}",
            True,
            lowres_max_dim,
            f"one_image_per_post_lowres_{lowres_max_dim}",
        ),
    ]

    turn1: dict[str, str] = {}
    turn2: dict[str, str] = {}
    turn1_retry_mode: dict[str, str] = {}
    turn2_retry_mode: dict[str, str] = {}
    turn1_errors: dict[str, str] = {}
    turn2_errors: dict[str, str] = {}

    pending_turn1 = {str(row["user_id"]): loaded_tasks[str(row["user_id"])] for row in task_rows}
    for stage_name, one_image_per_post, stage_lowres, retry_mode in stages:
        if not pending_turn1:
            break
        request_rows: list[dict[str, Any]] = []
        build_errors: dict[str, str] = {}
        for user_id, eval_task in list(pending_turn1.items()):
            custom_id = f"{user_id}__turn1"
            try:
                request_rows.append(
                    _build_dialogue_batch_request(
                        eval_task=eval_task,
                        spec=spec,
                        custom_id=custom_id,
                        turn_id=1,
                        max_context_posts=max_context_posts,
                        max_context_chars=max_context_chars,
                        max_images=max_images,
                        one_image_per_post=one_image_per_post,
                        lowres_max_dim=stage_lowres,
                        context_bundle=context_bundle_by_user.get(user_id),
                    )
                )
            except Exception as exc:
                build_errors[user_id] = str(exc)
        for user_id, err in build_errors.items():
            pending_turn1.pop(user_id, None)
            turn1_errors[user_id] = f"batch_request_build_failed: {err}"
        responses, errors = _run_dialogue_dashscope_batch(
            spec=spec,
            request_rows=request_rows,
            output_dir=output_dir,
            batch_name=f"turn1_{stage_name}",
            poll_seconds=poll_seconds,
            max_wait_seconds=max_wait_seconds,
            completion_window=completion_window,
            callback_url=callback_url,
        )
        for user_id in list(pending_turn1):
            custom_id = f"{user_id}__turn1"
            if custom_id in responses:
                turn1[user_id] = responses[custom_id]
                if retry_mode:
                    turn1_retry_mode[user_id] = retry_mode
                pending_turn1.pop(user_id, None)
                continue
            err_text = str(errors.get(custom_id) or "dashscope_batch_missing_response")
            if _looks_like_context_limit_error(err_text) and stage_name != stages[-1][0]:
                continue
            pending_turn1.pop(user_id, None)
            turn1_errors[user_id] = f"dashscope_batch_failed: {err_text}"

    for user_id in list(pending_turn1):
        turn1_errors[user_id] = "dashscope_batch_failed: context_limit_after_all_retries"

    pending_turn2 = {user_id: loaded_tasks[user_id] for user_id in turn1}
    for stage_name, one_image_per_post, stage_lowres, retry_mode in stages:
        if not pending_turn2:
            break
        request_rows = []
        build_errors: dict[str, str] = {}
        for user_id, eval_task in list(pending_turn2.items()):
            custom_id = f"{user_id}__turn2"
            try:
                request_rows.append(
                    _build_dialogue_batch_request(
                        eval_task=eval_task,
                        spec=spec,
                        custom_id=custom_id,
                        turn_id=2,
                        previous_response=turn1[user_id],
                        max_context_posts=max_context_posts,
                        max_context_chars=max_context_chars,
                        max_images=max_images,
                        one_image_per_post=one_image_per_post,
                        lowres_max_dim=stage_lowres,
                        context_bundle=context_bundle_by_user.get(user_id),
                    )
                )
            except Exception as exc:
                build_errors[user_id] = str(exc)
        for user_id, err in build_errors.items():
            pending_turn2.pop(user_id, None)
            turn2_errors[user_id] = f"batch_request_build_failed: {err}"
        responses, errors = _run_dialogue_dashscope_batch(
            spec=spec,
            request_rows=request_rows,
            output_dir=output_dir,
            batch_name=f"turn2_{stage_name}",
            poll_seconds=poll_seconds,
            max_wait_seconds=max_wait_seconds,
            completion_window=completion_window,
            callback_url=callback_url,
        )
        for user_id in list(pending_turn2):
            custom_id = f"{user_id}__turn2"
            if custom_id in responses:
                turn2[user_id] = responses[custom_id]
                if retry_mode:
                    turn2_retry_mode[user_id] = retry_mode
                pending_turn2.pop(user_id, None)
                continue
            err_text = str(errors.get(custom_id) or "dashscope_batch_missing_response")
            if _looks_like_context_limit_error(err_text) and stage_name != stages[-1][0]:
                continue
            pending_turn2.pop(user_id, None)
            turn2_errors[user_id] = f"dashscope_batch_failed: {err_text}"

    for user_id in list(pending_turn2):
        turn2_errors[user_id] = "dashscope_batch_failed: context_limit_after_all_retries"

    predictions: dict[str, dict[str, Any]] = {}
    for task_row in task_rows:
        user_id = str(task_row["user_id"])
        eval_task = loaded_tasks[user_id]
        prompts = eval_task.get("prompt_template", {}) or {}
        generation_error_parts = []
        if user_id in turn1_errors:
            generation_error_parts.append(f"turn1: {turn1_errors[user_id]}")
        if user_id in turn2_errors:
            generation_error_parts.append(f"turn2: {turn2_errors[user_id]}")
        prediction = {
            "schema_version": "personalized_dialogue_prediction_v1",
            "task_id": eval_task.get("task_id"),
            "user_id": eval_task.get("user_id"),
            "model": spec.name,
            "visual_mode": normalized_visual_mode,
            "generation_error": "; ".join(generation_error_parts),
            "dialogue": [
                {
                    "turn_id": 1,
                    "user_prompt": str(prompts.get("turn1_user_prompt") or ""),
                    "assistant_response": turn1.get(user_id, ""),
                },
                {
                    "turn_id": 2,
                    "user_prompt": str(prompts.get("turn2_user_prompt") or ""),
                    "assistant_response": turn2.get(user_id, ""),
                },
            ],
        }
        retry_modes = [mode for mode in [turn1_retry_mode.get(user_id), turn2_retry_mode.get(user_id)] if mode]
        if retry_modes:
            prediction["retry_mode"] = ",".join(retry_modes)
        predictions[user_id] = prediction
    return predictions


def _resolve_judge_batch_model_spec() -> Any:
    spec = DEFAULT_MODEL_SPECS.get("qwen3.7-max")
    if spec is None:
        raise ValueError("qwen3.7-max model spec not found in DEFAULT_MODEL_SPECS")
    return spec


def _build_judge_batch_request(
    *,
    custom_id: str,
    judge_model: str,
    judge_input: dict[str, Any],
) -> dict[str, Any] | None:
    user_prompt = (
        "Evaluate this personalized dialogue prediction.\n\n"
        "Use the scenario rubrics inside gold_reference_dialogue.\n"
        "Score interest_coverage, negative_avoidance, concreteness, and fluency from 0 to 5.\n"
        "Do not require explicit evidence explanation in the model response.\n\n"
        "Do not copy the schema hint or return placeholder zeros. "
        "If all four scores are 0, brief_rationale must explain why.\n\n"
        f"JUDGE_INPUT:\n{json.dumps(judge_input, ensure_ascii=False, indent=2)}\n\n"
        "Return strict JSON only."
    )
    row = build_openai_batch_chat_request(
        custom_id=custom_id,
        model=judge_model,
        system_prompt=DIALOGUE_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_urls=None,
        temperature=0.0,
        max_tokens=4096,
        force_json=True,
    )
    if request_json_size_bytes(row) > MAX_BATCH_REQUEST_BYTES:
        return None
    return row


def _run_dialogue_judge_batch(
    *,
    judge_spec: Any,
    judge_inputs: dict[str, dict[str, Any]],
    output_dir: Path,
    poll_seconds: int,
    max_wait_seconds: int,
    completion_window: str,
    callback_url: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not judge_inputs:
        return {}, []

    client = DashScopeBatchInferenceClient(
        base_url=str(judge_spec.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key_env=str(judge_spec.api_key_env or "ALI_BAILIAN_1"),
        api_key_env_fallback="ALI_BAILIAN_2",
        timeout_seconds=int(os.environ.get("BENCHMARK_PROVIDER_TIMEOUT_SECONDS", "120") or 120),
    )

    request_rows: list[dict[str, Any]] = []
    oversized_user_ids: list[str] = []
    for user_id, judge_input in judge_inputs.items():
        row = _build_judge_batch_request(
            custom_id=user_id,
            judge_model=str(judge_spec.api_model or judge_spec.name),
            judge_input=judge_input,
        )
        if row is None:
            oversized_user_ids.append(user_id)
        else:
            request_rows.append(row)

    if not request_rows:
        return {}, oversized_user_ids

    run_tag = f"judge_{_slugify(str(judge_spec.name))}_{int(time.time())}"
    run_dir = output_dir / "_dashscope_batch_runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.jsonl"
    with input_path.open("w", encoding="utf-8") as fh:
        for row in request_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    upload = client.upload_batch_file(local_path=input_path)
    metadata: dict[str, Any] = {}
    if str(callback_url or "").strip():
        metadata["ds_batch_finish_callback"] = str(callback_url).strip()
    batch = client.create_batch(
        input_file_id=str(upload.get("id") or ""),
        endpoint="/v1/chat/completions",
        completion_window=str(completion_window or "24h").strip() or "24h",
        metadata=metadata,
    )
    batch_final = client.wait_batch(
        batch_id=str(batch.get("id") or ""),
        poll_seconds=int(poll_seconds),
        max_wait_seconds=int(max_wait_seconds),
    )
    (run_dir / "upload.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "batch.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "batch_final.json").write_text(json.dumps(batch_final, ensure_ascii=False, indent=2), encoding="utf-8")

    output_file_id = str(batch_final.get("output_file_id") or "").strip()
    error_file_id = str(batch_final.get("error_file_id") or "").strip()
    if output_file_id:
        client.download_file_content(file_id=output_file_id, local_path=run_dir / "result.jsonl")
    if error_file_id:
        client.download_file_content(file_id=error_file_id, local_path=run_dir / "errors.jsonl")

    raw_responses: dict[str, str] = {}
    batch_errors: dict[str, str] = {}

    result_path = run_dir / "result.jsonl"
    if result_path.exists():
        for row in iter_jsonl_rows(result_path):
            custom_id = str(row.get("custom_id") or "").strip()
            if not custom_id:
                continue
            error = row.get("error")
            if error:
                batch_errors[custom_id] = json.dumps(error, ensure_ascii=False)
                continue
            response = row.get("response") or {}
            body = (response.get("body") or {}) if isinstance(response, dict) else {}
            try:
                raw_responses[custom_id] = extract_openai_chat_message_text(body if isinstance(body, dict) else {})
            except Exception as exc:
                batch_errors[custom_id] = f"batch_parse_failed: {exc}"

    error_path = run_dir / "errors.jsonl"
    if error_path.exists():
        for row in iter_jsonl_rows(error_path):
            custom_id = str(row.get("custom_id") or "").strip()
            error = row.get("error")
            if custom_id and error:
                batch_errors[custom_id] = json.dumps(error, ensure_ascii=False)

    judge_results: dict[str, dict[str, Any]] = {}
    for user_id, raw_text in raw_responses.items():
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            judge_results[user_id] = _fallback_judge_result(
                gold_ref={"task_id": "", "user_id": user_id, "selected_interest_facts": []},
                error=f"batch_judge_json_parse_failed",
                brief_rationale="Batch judge returned non-JSON response.",
            )
            continue
        try:
            result = _normalize_judge_raw(raw=raw, judge_input=judge_inputs[user_id])
            result["judge_error"] = ""
            result["provider_blocked"] = False
            judge_results[user_id] = result
        except (json.JSONDecodeError, JudgePlaceholderError) as exc:
            judge_results[user_id] = _fallback_judge_result(
                gold_ref={
                    "task_id": judge_inputs[user_id].get("task_id", ""),
                    "user_id": user_id,
                    "selected_interest_facts": judge_inputs[user_id].get("gold_interest_facts", []),
                },
                error=_exception_text(exc),
                brief_rationale="Batch judge returned invalid score payload.",
            )

    for user_id, error_text in batch_errors.items():
        if user_id not in judge_results:
            judge_results[user_id] = _fallback_judge_result(
                gold_ref={
                    "task_id": judge_inputs.get(user_id, {}).get("task_id", ""),
                    "user_id": user_id,
                    "selected_interest_facts": (judge_inputs.get(user_id) or {}).get("gold_interest_facts", []),
                },
                error=f"dashscope_batch_error: {error_text}",
                brief_rationale="Batch judge API call returned an error.",
            )

    return judge_results, oversized_user_ids
