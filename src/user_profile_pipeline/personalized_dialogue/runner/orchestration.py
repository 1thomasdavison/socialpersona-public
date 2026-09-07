from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from ...benchmark.profile_eval.io import _resolve_model_specs
from ...benchmark.profile_eval.model_selection import (
    _enable_bailian_batch_if_requested,
    _parse_models_arg,
    _validate_bailian_batch_model_specs,
)
from ...benchmark.profile_eval.modes import _slugify
from ...image_text import VISUAL_MODE_NATIVE, ImageTextGenerator, normalize_visual_mode
from ...llm_client import OpenAICompatibleChatClient
from ...multimodal_context import build_retry_posts_context_text
from .. import PersonalizedDialogueEvaluator
from .batch import (
    _generate_dialogue_predictions_batch,
    _resolve_judge_batch_model_spec,
    _run_dialogue_judge_batch,
)
from .context import _build_visual_context_bundle, _context_limit_lowres_max_dim
from .generation import _generate_dialogue_prediction
from .judging import (
    DIALOGUE_EVAL_PROMPT_CONTEXT_VERSION,
    DIALOGUE_JUDGE_PROMPT_VERSION,
    _build_chat_client_for_spec,
    _build_flash_judge_fallback_client_from_args,
    _build_judge_client_from_args,
    _build_judge_input,
    _exception_text,
    _fallback_judge_result,
    _fallback_prediction,
    _is_provider_block_message,
    _judge_file_is_cacheable,
    _judge_prediction_with_retry,
    _looks_like_context_limit_error,
    _looks_like_transport_timeout_error,
    _prediction_file_matches_visual_mode,
    _stamp_judge_result,
)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)


def _discover_dialogue_tasks(dialogue_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for user_dir in sorted(dialogue_root.iterdir()):
        if not user_dir.is_dir():
            continue
        gold_ref_path = user_dir / "personalized_dialogue_gold_ref.json"
        eval_task_path = user_dir / "personalized_dialogue_eval_task.json"
        if not gold_ref_path.exists() or not eval_task_path.exists():
            continue
        tasks.append(
            {
                "user_id": user_dir.name,
                "gold_ref_path": gold_ref_path.resolve(),
                "eval_task_path": eval_task_path.resolve(),
            }
        )
    return tasks


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "intervention_config", ""):
        from .interventions import run_interventions
        return run_interventions(args)
    dialogue_root = Path(args.dialogue_root).resolve()
    if not dialogue_root.exists():
        raise FileNotFoundError(f"dialogue root not found: {dialogue_root}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_rows = _discover_dialogue_tasks(dialogue_root)
    if not task_rows:
        raise ValueError(f"No dialogue tasks discovered under {dialogue_root}")

    models = _parse_models_arg(args.models)
    model_specs = _enable_bailian_batch_if_requested(
        _resolve_model_specs(models),
        enabled=bool(args.bailian_use_batch),
    )
    visual_mode = normalize_visual_mode(str(args.visual_mode or VISUAL_MODE_NATIVE))
    _validate_bailian_batch_model_specs(model_specs, visual_mode=visual_mode)
    judge_client = None
    image_text_cache_root = (
        Path(str(args.image_text_cache_root)).resolve()
        if str(getattr(args, "image_text_cache_root", "") or "").strip()
        else None
    )

    if bool(args.parallel_models) and len(model_specs) > 1:
        workers = min(max(1, int(args.max_parallel_models)), len(model_specs))
        print(f"[parallel] enabled workers={workers} models={len(model_specs)}", flush=True)
        metrics_by_model: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_spec = {
                executor.submit(
                    _run_one_model,
                    spec=spec,
                    args=args,
                    task_rows=task_rows,
                    visual_mode=visual_mode,
                    judge_client=judge_client,
                    output_dir=output_dir,
                    image_text_cache_root=image_text_cache_root,
                ): spec
                for spec in model_specs
            }
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                metrics_by_model[spec.name] = future.result()
        report_models = [metrics_by_model[spec.name] for spec in model_specs]
    else:
        report_models = [
            _run_one_model(
                spec=spec,
                args=args,
                task_rows=task_rows,
                visual_mode=visual_mode,
                judge_client=judge_client,
                output_dir=output_dir,
                image_text_cache_root=image_text_cache_root,
            )
            for spec in model_specs
        ]

    report = {
        "schema_version": "personalized_dialogue_eval_report_v1",
        "prompt_context_version": DIALOGUE_EVAL_PROMPT_CONTEXT_VERSION,
        "judge_prompt_version": DIALOGUE_JUDGE_PROMPT_VERSION,
        "dialogue_root": str(dialogue_root),
        "output_dir": str(output_dir),
        "visual_mode": visual_mode,
        "n_tasks": len(task_rows),
        "judge_model": str(args.judge_model or "gpt-o3"),
        "models": report_models,
    }
    (output_dir / "dialogue_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _run_one_model(
    *,
    spec: Any,
    args: argparse.Namespace,
    task_rows: list[dict[str, Any]],
    visual_mode: str,
    judge_client: OpenAICompatibleChatClient,
    output_dir: Path,
    image_text_cache_root: Path | None,
) -> dict[str, Any]:
    client = None
    if spec.provider != "bailian_batch":
        client = _build_chat_client_for_spec(
            spec,
            timeout_seconds=int(args.timeout_seconds),
            max_tokens=int(args.max_tokens),
        )
    judge_client = _build_judge_client_from_args(args)
    fallback_judge_client = _build_flash_judge_fallback_client_from_args(args)
    model_dir = output_dir / _slugify(spec.name)
    predictions_dir = model_dir / "predictions"
    auto_metrics_dir = model_dir / "auto_metrics"
    judge_dir = model_dir / "judge"
    for path in [model_dir, predictions_dir, auto_metrics_dir, judge_dir]:
        path.mkdir(parents=True, exist_ok=True)

    auto_scorer = PersonalizedDialogueEvaluator(
        model_config_path="configs/model.yaml",
        model_name=spec.name,
    )
    judge_use_bailian_batch = bool(getattr(args, "judge_use_bailian_batch", False))
    if judge_use_bailian_batch:
        judge_batch_spec = _resolve_judge_batch_model_spec()
        expected_judge_model = str(judge_batch_spec.name)
        expected_judge_provider = "bailian"
        expected_judge_api_model = str(judge_batch_spec.api_model or judge_batch_spec.name)
    else:
        judge_batch_spec = None
        expected_judge_model = str(args.judge_model or "gpt-o3").strip() or "gpt-o3"
        expected_judge_provider = str(args.judge_provider or "").strip()
        expected_judge_api_model = str(args.judge_api_model or expected_judge_model).strip() or expected_judge_model
    per_user_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    image_text_generators: dict[str, ImageTextGenerator] = {}
    visual_bundle_cache: dict[tuple[str, str, str], Any] = {}

    provider = "vertex" if str(spec.provider).startswith("vertex") else "openai_compatible"
    cached_user_ids: set[str] = set()
    precomputed_predictions: dict[str, dict[str, Any]] = {}
    if spec.provider == "bailian_batch":
        pending_rows: list[dict[str, Any]] = []
        for task_row in task_rows:
            user_id = str(task_row["user_id"])
            prediction_path = predictions_dir / f"{user_id}.json"
            auto_metrics_path = auto_metrics_dir / f"{user_id}.json"
            judge_path = judge_dir / f"{user_id}.json"
            if user_id in cached_user_ids:
                continue
            if (
                prediction_path.exists()
                and auto_metrics_path.exists()
                and judge_path.exists()
                and not args.force
                and _prediction_file_matches_visual_mode(prediction_path, visual_mode)
                and _judge_file_is_cacheable(judge_path, expected_judge_model=expected_judge_model)
            ):
                cached_user_ids.add(user_id)
                continue
            pending_rows.append(task_row)
        if pending_rows:
            precomputed_predictions = _generate_dialogue_predictions_batch(
                spec=spec,
                task_rows=pending_rows,
                output_dir=model_dir,
                max_context_posts=int(args.max_context_posts),
                max_context_chars=int(args.max_context_chars),
                max_images=int(args.max_images),
                poll_seconds=int(args.bailian_batch_poll_seconds),
                max_wait_seconds=int(args.bailian_batch_max_wait_seconds),
                completion_window=str(args.bailian_batch_completion_window or "24h").strip() or "24h",
                callback_url=str(args.bailian_batch_callback_url or "").strip(),
                visual_mode=visual_mode,
                image_text_max_tokens=int(args.image_text_max_tokens),
                image_text_timeout_seconds=int(args.image_text_timeout_seconds),
                force=bool(args.force),
                image_text_cache_root=image_text_cache_root,
                image_text_cache_only=bool(args.image_text_cache_only),
            )

    if judge_use_bailian_batch and judge_batch_spec is not None:
        judge_inputs_to_batch: dict[str, dict[str, Any]] = {}
        for task_row in task_rows:
            user_id = str(task_row["user_id"])
            judge_path = judge_dir / f"{user_id}.json"
            if judge_path.exists() and not args.force and _judge_file_is_cacheable(judge_path, expected_judge_model=expected_judge_model):
                continue

            gold_ref = json.loads(Path(task_row["gold_ref_path"]).read_text(encoding="utf-8"))
            eval_task = json.loads(Path(task_row["eval_task_path"]).read_text(encoding="utf-8"))

            if spec.provider == "bailian_batch":
                prediction = precomputed_predictions.get(user_id)
                if prediction is None or str(prediction.get("generation_error") or ""):
                    continue
            else:
                prediction_path = predictions_dir / f"{user_id}.json"
                if (prediction_path.exists() and not args.force
                        and _prediction_file_matches_visual_mode(prediction_path, visual_mode)):
                    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
                    if str(prediction.get("generation_error") or ""):
                        continue
                else:
                    try:
                        profile_only = bool(
                            ((eval_task.get("context") or {}) if isinstance(eval_task.get("context"), dict) else {}).get(
                                "profile_only_mode"
                            )
                        )
                        visual_context_bundle = None
                        if not profile_only:
                            visual_context_bundle = _build_visual_context_bundle(
                                gold_ref=gold_ref,
                                eval_task=eval_task,
                                spec=spec,
                                visual_mode=visual_mode,
                                output_dir=model_dir,
                                force=bool(args.force),
                                image_text_generators=image_text_generators,
                                bundle_cache=visual_bundle_cache,
                                image_text_max_tokens=int(args.image_text_max_tokens),
                                image_text_timeout_seconds=int(args.image_text_timeout_seconds),
                                image_text_cache_root=image_text_cache_root,
                                image_text_cache_only=bool(args.image_text_cache_only),
                            )
                        prediction = _generate_dialogue_prediction(
                            client=client,
                            eval_task=eval_task,
                            model_name=spec.name,
                            provider=provider,
                            max_context_posts=int(args.max_context_posts),
                            max_context_chars=int(args.max_context_chars),
                            max_images=0 if profile_only else int(args.max_images),
                            visual_mode=visual_mode,
                            context_bundle=visual_context_bundle,
                        )
                        prediction_path.parent.mkdir(parents=True, exist_ok=True)
                        prediction_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        continue
                    if str(prediction.get("generation_error") or ""):
                        continue

            judge_inputs_to_batch[user_id] = _build_judge_input(
                gold_ref=gold_ref,
                eval_task=eval_task,
                prediction=prediction,
            )

        if judge_inputs_to_batch:
            print(f"[{spec.name}] batch judge: submitting {len(judge_inputs_to_batch)} users to DashScope", flush=True)
            batch_results, oversized = _run_dialogue_judge_batch(
                judge_spec=judge_batch_spec,
                judge_inputs=judge_inputs_to_batch,
                output_dir=model_dir,
                poll_seconds=int(getattr(args, "judge_bailian_batch_poll_seconds", 120)),
                max_wait_seconds=int(getattr(args, "judge_bailian_batch_max_wait_seconds", 86400)),
                completion_window=str(getattr(args, "judge_bailian_batch_completion_window", "24h") or "24h").strip() or "24h",
                callback_url=str(getattr(args, "judge_bailian_batch_callback_url", "") or "").strip(),
            )
            for user_id, result in batch_results.items():
                _stamp_judge_result(
                    result,
                    judge_model=expected_judge_model,
                    judge_provider=expected_judge_provider,
                    judge_api_model=expected_judge_api_model,
                )
                (judge_dir / f"{user_id}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{spec.name}] batch judge: completed {len(batch_results)} users"
                  + (f", oversized={len(oversized)}" if oversized else ""), flush=True)

    total_tasks = len(task_rows)
    for task_index, task_row in enumerate(task_rows, start=1):
        user_id = str(task_row["user_id"])
        print(f"[{spec.name}] processing {task_index}/{total_tasks} user_id={user_id}", flush=True)
        gold_ref = json.loads(Path(task_row["gold_ref_path"]).read_text(encoding="utf-8"))
        eval_task = json.loads(Path(task_row["eval_task_path"]).read_text(encoding="utf-8"))

        prediction_path = predictions_dir / f"{user_id}.json"
        auto_metrics_path = auto_metrics_dir / f"{user_id}.json"
        judge_path = judge_dir / f"{user_id}.json"

        if user_id in cached_user_ids:
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            auto_metrics = json.loads(auto_metrics_path.read_text(encoding="utf-8"))
            judge_result = json.loads(judge_path.read_text(encoding="utf-8"))
            print(f"[{spec.name}] cache hit user_id={user_id}", flush=True)
        elif (
            prediction_path.exists()
            and auto_metrics_path.exists()
            and judge_path.exists()
            and not args.force
            and _prediction_file_matches_visual_mode(prediction_path, visual_mode)
            and _judge_file_is_cacheable(judge_path, expected_judge_model=expected_judge_model)
        ):
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            auto_metrics = json.loads(auto_metrics_path.read_text(encoding="utf-8"))
            judge_result = json.loads(judge_path.read_text(encoding="utf-8"))
            print(f"[{spec.name}] cache hit user_id={user_id}", flush=True)
        elif (
            prediction_path.exists()
            and auto_metrics_path.exists()
            and not args.force
            and _prediction_file_matches_visual_mode(prediction_path, visual_mode)
        ):
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            auto_metrics = json.loads(auto_metrics_path.read_text(encoding="utf-8"))
            generation_error = str(prediction.get("generation_error") or "")
            judge_error = ""
            print(f"[{spec.name}] judge cache miss user_id={user_id}; reuse prediction", flush=True)
            if generation_error:
                judge_result = _fallback_judge_result(
                    gold_ref=gold_ref,
                    error=generation_error,
                    brief_rationale="Prediction unavailable because the provider rejected or failed this sample.",
                )
            else:
                try:
                    judge_result = _judge_prediction_with_retry(
                        judge_client=judge_client,
                        fallback_judge_client=fallback_judge_client,
                        judge_input=_build_judge_input(
                            gold_ref=gold_ref,
                            eval_task=eval_task,
                            prediction=prediction,
                        ),
                    )
                    judge_result["judge_error"] = ""
                    judge_result["provider_blocked"] = False
                except Exception as exc:
                    judge_error = _exception_text(exc)
                    print(f"[{spec.name}] judge failed user_id={user_id}: {judge_error}", flush=True)
                    judge_result = _fallback_judge_result(
                        gold_ref=gold_ref,
                        error=judge_error,
                        brief_rationale="Judge call failed, so this sample was recorded with a zero judge score.",
                    )
            _stamp_judge_result(
                judge_result,
                judge_model=expected_judge_model,
                judge_provider=expected_judge_provider,
                judge_api_model=expected_judge_api_model,
            )
            judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")
        elif spec.provider == "bailian_batch":
            prediction = precomputed_predictions.get(user_id) or _fallback_prediction(
                eval_task=eval_task,
                model_name=spec.name,
                error="dashscope_batch_missing_prediction_row",
                visual_mode=visual_mode,
            )
            generation_error = str(prediction.get("generation_error") or "")
            judge_error = ""
            auto_metrics = auto_scorer._score(gold_ref=gold_ref, prediction=prediction)
            if generation_error:
                judge_result = _fallback_judge_result(
                    gold_ref=gold_ref,
                    error=generation_error,
                    brief_rationale="Prediction unavailable because the provider rejected or failed this sample.",
                )
            else:
                try:
                    judge_result = _judge_prediction_with_retry(
                        judge_client=judge_client,
                        fallback_judge_client=fallback_judge_client,
                        judge_input=_build_judge_input(
                            gold_ref=gold_ref,
                            eval_task=eval_task,
                            prediction=prediction,
                        ),
                    )
                    judge_result["judge_error"] = ""
                    judge_result["provider_blocked"] = False
                except Exception as exc:
                    judge_error = _exception_text(exc)
                    print(f"[{spec.name}] judge failed user_id={user_id}: {judge_error}", flush=True)
                    judge_result = _fallback_judge_result(
                        gold_ref=gold_ref,
                        error=judge_error,
                        brief_rationale="Judge call failed, so this sample was recorded with a zero judge score.",
                    )
            prediction_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
            auto_metrics_path.write_text(json.dumps(auto_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            _stamp_judge_result(
                judge_result,
                judge_model=expected_judge_model,
                judge_provider=expected_judge_provider,
                judge_api_model=expected_judge_api_model,
            )
            judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            generation_error = ""
            judge_error = ""
            profile_only = bool(
                ((eval_task.get("context") or {}) if isinstance(eval_task.get("context"), dict) else {}).get(
                    "profile_only_mode"
                )
            )
            visual_context_bundle = None
            if not profile_only:
                visual_context_bundle = _build_visual_context_bundle(
                    gold_ref=gold_ref,
                    eval_task=eval_task,
                    spec=spec,
                    visual_mode=visual_mode,
                    output_dir=model_dir,
                    force=bool(args.force),
                    image_text_generators=image_text_generators,
                    bundle_cache=visual_bundle_cache,
                    image_text_max_tokens=int(args.image_text_max_tokens),
                    image_text_timeout_seconds=int(args.image_text_timeout_seconds),
                    image_text_cache_root=image_text_cache_root,
                    image_text_cache_only=bool(args.image_text_cache_only),
                )
            try:
                prediction = _generate_dialogue_prediction(
                    client=client,
                    eval_task=eval_task,
                    model_name=spec.name,
                    provider=provider,
                    max_context_posts=int(args.max_context_posts),
                    max_context_chars=int(args.max_context_chars),
                    max_images=0 if profile_only else int(args.max_images),
                    visual_mode=visual_mode,
                    context_bundle=visual_context_bundle,
                )
            except Exception as exc:
                generation_error = _exception_text(exc)
                retry_prediction: dict[str, Any] | None = None
                if _looks_like_context_limit_error(generation_error):
                    try:
                        retry_prediction = _generate_dialogue_prediction(
                            client=client,
                            eval_task=eval_task,
                            model_name=spec.name,
                            provider=provider,
                            max_context_posts=int(args.max_context_posts),
                            max_context_chars=int(args.max_context_chars),
                            max_images=1_000_000,
                            one_image_per_post=True,
                            retry_mode="one_image_per_post",
                            visual_mode=visual_mode,
                            context_bundle=visual_context_bundle,
                        )
                    except Exception as retry_exc:
                        retry_error = _exception_text(retry_exc)
                        generation_error = f"{generation_error}; retry_one_image_per_post_failed: {retry_error}"
                        if _looks_like_context_limit_error(retry_error):
                            lowres_max_dim = _context_limit_lowres_max_dim()
                            try:
                                retry_prediction = _generate_dialogue_prediction(
                                    client=client,
                                    eval_task=eval_task,
                                    model_name=spec.name,
                                    provider=provider,
                                    max_context_posts=int(args.max_context_posts),
                                    max_context_chars=int(args.max_context_chars),
                                    max_images=1_000_000,
                                    one_image_per_post=True,
                                    lowres_max_dim=lowres_max_dim,
                                    retry_mode=f"one_image_per_post_lowres_{lowres_max_dim}",
                                    visual_mode=visual_mode,
                                    context_bundle=visual_context_bundle,
                                )
                            except Exception as lowres_exc:
                                generation_error = f"{generation_error}; retry_lowres_failed: {_exception_text(lowres_exc)}"
                if (
                    retry_prediction is None
                    and visual_context_bundle is not None
                    and _looks_like_transport_timeout_error(generation_error)
                ):
                    compact_context = build_retry_posts_context_text(
                        getattr(visual_context_bundle, "payload", {}) or {},
                        redact_text=False,
                    )
                    try:
                        retry_prediction = _generate_dialogue_prediction(
                            client=client,
                            eval_task=eval_task,
                            model_name=spec.name,
                            provider=provider,
                            max_context_posts=int(args.max_context_posts),
                            max_context_chars=int(args.max_context_chars),
                            max_images=int(args.max_images),
                            retry_mode="compact_context",
                            visual_mode=visual_mode,
                            context_bundle=visual_context_bundle,
                            context_override=compact_context,
                        )
                    except Exception as compact_exc:
                        generation_error = (
                            f"{generation_error}; retry_compact_context_failed: {_exception_text(compact_exc)}"
                        )
                if retry_prediction is None and visual_context_bundle is not None and _is_provider_block_message(generation_error):
                    redacted_context = build_retry_posts_context_text(
                        getattr(visual_context_bundle, "payload", {}) or {},
                        redact_text=True,
                    )
                    try:
                        retry_prediction = _generate_dialogue_prediction(
                            client=client,
                            eval_task=eval_task,
                            model_name=spec.name,
                            provider=provider,
                            max_context_posts=int(args.max_context_posts),
                            max_context_chars=int(args.max_context_chars),
                            max_images=int(args.max_images),
                            retry_mode="compact_context_text_redacted",
                            visual_mode=visual_mode,
                            context_bundle=visual_context_bundle,
                            context_override=redacted_context,
                        )
                    except Exception as redacted_exc:
                        generation_error = (
                            f"{generation_error}; retry_text_redacted_failed: {_exception_text(redacted_exc)}"
                        )
                if retry_prediction is not None:
                    prediction = retry_prediction
                else:
                    print(f"[{spec.name}] generation failed user_id={user_id}: {generation_error}", flush=True)
                    prediction = _fallback_prediction(
                        eval_task=eval_task,
                        model_name=spec.name,
                        error=generation_error,
                        visual_mode=visual_mode,
                    )

            auto_metrics = auto_scorer._score(gold_ref=gold_ref, prediction=prediction)
            if generation_error:
                judge_result = _fallback_judge_result(
                    gold_ref=gold_ref,
                    error=generation_error,
                    brief_rationale="Prediction unavailable because the provider rejected or failed this sample.",
                )
            else:
                try:
                    judge_result = _judge_prediction_with_retry(
                        judge_client=judge_client,
                        fallback_judge_client=fallback_judge_client,
                        judge_input=_build_judge_input(
                            gold_ref=gold_ref,
                            eval_task=eval_task,
                            prediction=prediction,
                        ),
                    )
                    judge_result["judge_error"] = ""
                    judge_result["provider_blocked"] = False
                except Exception as exc:
                    judge_error = _exception_text(exc)
                    print(f"[{spec.name}] judge failed user_id={user_id}: {judge_error}", flush=True)
                    judge_result = _fallback_judge_result(
                        gold_ref=gold_ref,
                        error=judge_error,
                        brief_rationale="Judge call failed, so this sample was recorded with a zero judge score.",
                    )

            prediction_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
            auto_metrics_path.write_text(json.dumps(auto_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            _stamp_judge_result(
                judge_result,
                judge_model=expected_judge_model,
                judge_provider=expected_judge_provider,
                judge_api_model=expected_judge_api_model,
            )
            judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")

        _stamp_judge_result(
            judge_result,
            judge_model=expected_judge_model,
            judge_provider=expected_judge_provider,
            judge_api_model=expected_judge_api_model,
        )
        judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")
        prediction_rows.append(prediction)
        judge_rows.append(judge_result)
        per_user_results.append(
            {
                "user_id": user_id,
                "task_id": str(prediction.get("task_id") or ""),
                "prediction_path": str(prediction_path.resolve()),
                "auto_metrics_path": str(auto_metrics_path.resolve()),
                "judge_path": str(judge_path.resolve()),
                "auto_overall_score": float(((auto_metrics.get("overall") or {}).get("overall_score")) or 0.0),
                "judge_final_dialogue_score": float(judge_result.get("final_dialogue_score") or 0.0),
                "interest_coverage": float(judge_result.get("interest_coverage") or 0.0),
                "negative_avoidance": float(judge_result.get("negative_avoidance") or 0.0),
                "concreteness": float(judge_result.get("concreteness") or 0.0),
                "fluency": float(judge_result.get("fluency") or 0.0),
                "generation_error": str(prediction.get("generation_error") or ""),
                "judge_error": str(judge_result.get("judge_error") or ""),
                "provider_blocked": bool(judge_result.get("provider_blocked")),
            }
        )

    predictions_jsonl = model_dir / "predictions.jsonl"
    judge_jsonl = model_dir / "judge_results.jsonl"
    predictions_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in prediction_rows) + "\n",
        encoding="utf-8",
    )
    judge_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in judge_rows) + "\n",
        encoding="utf-8",
    )

    metrics = {
        "model": spec.name,
        "provider": spec.provider,
        "visual_mode": visual_mode,
        "n_tasks": len(per_user_results),
        "judge_model": str(args.judge_model or "gpt-o3"),
        "dialogue_judge_main_score": _mean([row["judge_final_dialogue_score"] for row in per_user_results]) / 100.0,
        "dialogue_judge_main_score_100": _mean([row["judge_final_dialogue_score"] for row in per_user_results]),
        "interest_coverage_mean": _mean([row["interest_coverage"] for row in per_user_results]),
        "negative_avoidance_mean": _mean([row["negative_avoidance"] for row in per_user_results]),
        "concreteness_mean": _mean([row["concreteness"] for row in per_user_results]),
        "fluency_mean": _mean([row["fluency"] for row in per_user_results]),
        "auto_overall_score_mean": _mean([row["auto_overall_score"] for row in per_user_results]),
        "generation_error_count": sum(1 for row in per_user_results if row["generation_error"]),
        "judge_error_count": sum(1 for row in per_user_results if row["judge_error"]),
        "provider_block_count": sum(1 for row in per_user_results if row["provider_blocked"]),
        "per_user_overview": per_user_results,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
