from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
from pathlib import Path
import time
from typing import Any

from ...image_text import (
    VISUAL_MODE_TEXT_IMAGE,
    ImageTextGenerator,
    normalize_visual_mode,
    prediction_visual_mode,
)
from ...llm_client import OpenAICompatibleChatClient
from ..evaluator import AnchorMatchConfig, AnchorMatchJudgeClient
from ..evaluator.precheck import MAX_LONG_ANCHORS, MAX_SHORT_ANCHORS, MAX_SUMMARY_SUPPORT
from .inference import ProviderInferenceMixin
from .io import _env_or_dotenv, _load_json, _read_dotenv, _resolve_model_specs
from .metrics import MetricsMixin
from .modes import (
    _prediction_profile_input_mode,
    _prediction_profile_method,
    _profile_input_mode_cache_key,
    _slugify,
    normalize_profile_input_mode,
    normalize_profile_method,
)
from .normalization import _fallback_prediction, _normalize_prediction, _prediction_row_is_cacheable
from .profile_methods import _user_group_key
from .profile_prediction import ProfilePredictionMixin
from .specs import (
    DEFAULT_MODEL_SPECS,
    DIRECT_PROFILE_METHOD,
    DomainEvalTask,
    EXTRACTIVE_PROFILE_METHOD,
    HIERARCHICAL_PROFILE_METHOD,
    ModelSpec,
    PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS,
    USER_LEVEL_PROFILE_METHODS,
)


class BenchmarkEvaluator(ProfilePredictionMixin, ProviderInferenceMixin, MetricsMixin):
    def __init__(
        self,
        *,
        output_dir: Path,
        cache_dir: Path | None,
        model_specs: list[ModelSpec],
        force: bool = False,
        force_image_text: bool = False,
        max_images: int = 0,
        max_images_when_trim: int = 0,
        vertex_project_id: str = "",
        vertex_location: str = "",
        vertex_bucket: str = "",
        vertex_access_token_env: str = "VERTEX_ACCESS_TOKEN",
        poll_seconds: int = 20,
        max_wait_seconds: int = 14400,
        parallel_models: bool = False,
        max_parallel_models: int = 1,
        max_parallel_tasks: int = 1,
        request_timeout_seconds: int = 0,
        anchor_match_model: str = "gpt-o3",
        anchor_match_provider: str = "",
        anchor_match_api_key_env: str = "",
        anchor_match_api_model: str = "",
        anchor_match_base_url: str = "",
        anchor_match_prompt_version: str = "anchor_match_v1",
        anchor_match_cache_dir: Path | None = None,
        anchor_match_force_llm: bool = False,
        bailian_use_batch: bool = False,
        bailian_batch_poll_seconds: int = 120,
        bailian_batch_max_wait_seconds: int = 86400,
        bailian_batch_completion_window: str = "24h",
        bailian_batch_callback_url: str = "",
        visual_mode: str = VISUAL_MODE_TEXT_IMAGE,
        profile_input_mode: str = PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS,
        profile_method: str = DIRECT_PROFILE_METHOD,
        profile_method_max_posts: int = 200,
        hierarchical_chunk_size: int = 20,
        extractive_k: int = 12,
        image_text_model: str = "",
        image_text_max_tokens: int = 220,
        image_text_timeout_seconds: int = 120,
        json_repair_provider: str = "",
        json_repair_api_key_env: str = "",
        json_repair_api_model: str = "",
        json_repair_base_url: str = "",
        profile_eval_prompt_variant: str = "conservative",
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir or (self.output_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_specs = model_specs
        self.force = force
        self.force_image_text = force_image_text
        self.max_images = max_images
        self.max_images_when_trim = max_images_when_trim
        self.vertex_project_id = vertex_project_id
        self.vertex_location = vertex_location
        self.vertex_bucket = vertex_bucket
        self.vertex_access_token_env = vertex_access_token_env
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self.parallel_models = bool(parallel_models)
        self.max_parallel_models = max(1, int(max_parallel_models))
        self.max_parallel_tasks = max(1, int(max_parallel_tasks))
        self._dotenv = _read_dotenv(Path(".env"))
        self.request_timeout_seconds = int(request_timeout_seconds)
        self.anchor_match_model = str(anchor_match_model or "gpt-o3").strip() or "gpt-o3"
        self.anchor_match_provider = str(anchor_match_provider or "").strip()
        self.anchor_match_api_key_env = str(anchor_match_api_key_env or "").strip()
        self.anchor_match_api_model = str(anchor_match_api_model or "").strip()
        self.anchor_match_base_url = str(anchor_match_base_url or "").strip()
        prompt_version = str(anchor_match_prompt_version or "anchor_match_v1").strip() or "anchor_match_v1"
        if bool(anchor_match_force_llm) and "force_llm" not in prompt_version:
            prompt_version = f"{prompt_version}_force_llm"
        self.anchor_match_prompt_version = prompt_version
        self.anchor_match_cache_dir = anchor_match_cache_dir
        self.anchor_match_force_llm = bool(anchor_match_force_llm)
        self.bailian_use_batch = bool(bailian_use_batch)
        self.bailian_batch_poll_seconds = int(bailian_batch_poll_seconds)
        self.bailian_batch_max_wait_seconds = int(bailian_batch_max_wait_seconds)
        self.bailian_batch_completion_window = str(bailian_batch_completion_window or "24h").strip() or "24h"
        self.bailian_batch_callback_url = str(bailian_batch_callback_url or "").strip()
        self.visual_mode = normalize_visual_mode(visual_mode)
        self.profile_input_mode = normalize_profile_input_mode(profile_input_mode)
        self.profile_method = normalize_profile_method(profile_method)
        self.profile_method_max_posts = max(0, int(profile_method_max_posts))
        self.hierarchical_chunk_size = max(1, int(hierarchical_chunk_size))
        self.extractive_k = max(0, int(extractive_k))
        self.image_text_model = str(image_text_model or "").strip()
        self.image_text_model_spec = self._resolve_optional_image_text_model_spec(self.image_text_model)
        self.image_text_max_tokens = int(image_text_max_tokens)
        self.image_text_timeout_seconds = int(image_text_timeout_seconds)
        self.json_repair_provider = str(json_repair_provider or "").strip()
        self.json_repair_api_key_env = str(json_repair_api_key_env or "").strip()
        self.json_repair_api_model = str(json_repair_api_model or "").strip()
        self.json_repair_base_url = str(json_repair_base_url or "").strip()
        self.profile_eval_prompt_variant = str(profile_eval_prompt_variant or "conservative").strip() or "conservative"
        self._anchor_match_client: AnchorMatchJudgeClient | None = None
        self._inline_image_part_cache: dict[str, dict[str, Any] | None] = {}
        self._openai_image_data_url_cache: dict[str, str | None] = {}
        self._image_text_generators: dict[str, ImageTextGenerator] = {}
        self._visual_bundle_cache: dict[tuple[str, str, str, str], Any] = {}

    def _release_runtime_caches(self) -> None:
        self._visual_bundle_cache.clear()
        self._image_text_generators.clear()
        self._inline_image_part_cache.clear()
        self._clear_openai_image_caches()
        gc.collect()

    @staticmethod
    def _resolve_optional_image_text_model_spec(model_name: str) -> ModelSpec | None:
        cleaned = str(model_name or "").strip()
        if not cleaned:
            return None
        specs = _resolve_model_specs([cleaned])
        if not specs:
            return None
        return specs[0]

    def _build_isolated_model_evaluator(self, *, spec: ModelSpec) -> "BenchmarkEvaluator":
        return BenchmarkEvaluator(
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            model_specs=[spec],
            force=self.force,
            force_image_text=self.force_image_text,
            max_images=self.max_images,
            max_images_when_trim=self.max_images_when_trim,
            vertex_project_id=self.vertex_project_id,
            vertex_location=self.vertex_location,
            vertex_bucket=self.vertex_bucket,
            vertex_access_token_env=self.vertex_access_token_env,
            poll_seconds=self.poll_seconds,
            max_wait_seconds=self.max_wait_seconds,
            parallel_models=False,
            max_parallel_models=1,
            max_parallel_tasks=self.max_parallel_tasks,
            request_timeout_seconds=self.request_timeout_seconds,
            anchor_match_model=self.anchor_match_model,
            anchor_match_provider=self.anchor_match_provider,
            anchor_match_api_key_env=self.anchor_match_api_key_env,
            anchor_match_api_model=self.anchor_match_api_model,
            anchor_match_base_url=self.anchor_match_base_url,
            anchor_match_prompt_version=self.anchor_match_prompt_version,
            anchor_match_cache_dir=self.anchor_match_cache_dir,
            anchor_match_force_llm=self.anchor_match_force_llm,
            bailian_use_batch=self.bailian_use_batch,
            bailian_batch_poll_seconds=self.bailian_batch_poll_seconds,
            bailian_batch_max_wait_seconds=self.bailian_batch_max_wait_seconds,
            bailian_batch_completion_window=self.bailian_batch_completion_window,
            bailian_batch_callback_url=self.bailian_batch_callback_url,
            visual_mode=self.visual_mode,
            profile_input_mode=self.profile_input_mode,
            profile_method=self.profile_method,
            profile_method_max_posts=self.profile_method_max_posts,
            hierarchical_chunk_size=self.hierarchical_chunk_size,
            extractive_k=self.extractive_k,
            image_text_model=self.image_text_model,
            image_text_max_tokens=self.image_text_max_tokens,
            image_text_timeout_seconds=self.image_text_timeout_seconds,
            json_repair_provider=self.json_repair_provider,
            json_repair_api_key_env=self.json_repair_api_key_env,
            json_repair_api_model=self.json_repair_api_model,
            json_repair_base_url=self.json_repair_base_url,
        )

    def _evaluate_one_model_isolated(
        self,
        *,
        tasks: list[DomainEvalTask],
        spec: ModelSpec,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        isolated = self._build_isolated_model_evaluator(spec=spec)
        return isolated._evaluate_one_model(tasks=tasks, spec=spec)

    def run(self, tasks: list[DomainEvalTask]) -> dict[str, Any]:
        per_model_results: dict[str, tuple[dict[str, dict[str, Any]], dict[str, Any]]] = {}

        if self.parallel_models and len(self.model_specs) > 1:
            workers = min(self.max_parallel_models, len(self.model_specs))
            print(
                f"[parallel] enabled workers={workers} models={len(self.model_specs)}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_spec = {
                    executor.submit(self._evaluate_one_model_isolated, tasks=tasks, spec=spec): spec
                    for spec in self.model_specs
                }
                for future in as_completed(future_to_spec):
                    spec = future_to_spec[future]
                    try:
                        per_model_results[spec.name] = future.result()
                    except Exception as exc:
                        print(f"[parallel] model={spec.name} failed: {exc}", flush=True)
                        predictions = {
                            task.task_id: _fallback_prediction(
                                task,
                                model_name=spec.name,
                                error=f"parallel_model_failed: {exc}",
                            )
                            for task in tasks
                        }
                        metrics = self._compute_metrics(tasks=tasks, predictions=predictions)
                        metrics["model"] = spec.name
                        metrics["provider"] = spec.provider
                        metrics["visual_mode"] = self.visual_mode
                        metrics["profile_input_mode"] = self.profile_input_mode
                        metrics["profile_method"] = self.profile_method
                        metrics["profile_method_max_posts"] = self.profile_method_max_posts
                        metrics["hierarchical_chunk_size"] = self.hierarchical_chunk_size
                        metrics["extractive_k"] = self.extractive_k
                        per_model_results[spec.name] = (predictions, metrics)
        else:
            for spec in self.model_specs:
                per_model_results[spec.name] = self._evaluate_one_model(tasks=tasks, spec=spec)

        report_models: list[dict[str, Any]] = []
        for spec in self.model_specs:
            predictions, metrics = per_model_results[spec.name]
            report_models.append(metrics)

            model_dir = self.output_dir / _slugify(spec.name)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "predictions.jsonl").write_text(
                "\n".join(json.dumps(predictions[t.task_id], ensure_ascii=False) for t in tasks) + "\n",
                encoding="utf-8",
            )
            (model_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        report = {
            "n_tasks": len(tasks),
            "models": report_models,
            "is_official_run": True,
            "scoring_mode": "f1_only",
            "anchor_match_model": self.anchor_match_api_model or self.anchor_match_model,
            "anchor_match_provider": self.anchor_match_provider or None,
            "anchor_match_api_model": self.anchor_match_api_model or None,
            "anchor_match_base_url": self.anchor_match_base_url or None,
            "anchor_match_prompt_version": self.anchor_match_prompt_version,
            "visual_mode": self.visual_mode,
            "profile_input_mode": self.profile_input_mode,
            "profile_method": self.profile_method,
            "profile_method_max_posts": self.profile_method_max_posts,
            "hierarchical_chunk_size": self.hierarchical_chunk_size,
            "extractive_k": self.extractive_k,
            "generated_at": int(time.time()),
            "output_dir": str(self.output_dir),
        }
        (self.output_dir / "benchmark_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def _evaluate_one_model(
        self,
        *,
        tasks: list[DomainEvalTask],
        spec: ModelSpec,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        try:
            predictions = self._run_model(tasks=tasks, spec=spec, force=self.force)
            try:
                metrics = self._compute_metrics(tasks=tasks, predictions=predictions)
            except Exception as exc:
                print(f"[{spec.name}] scoring failed; keep predictions and fallback metrics: {exc}", flush=True)
                fallback_predictions = {
                    task.task_id: _fallback_prediction(
                        task,
                        model_name=spec.name,
                        error=f"scoring_failed: {exc}",
                    )
                    for task in tasks
                }
                metrics = self._compute_metrics(tasks=tasks, predictions=fallback_predictions)
                metrics["scoring_failed"] = True
                metrics["scoring_error"] = str(exc)
            metrics["model"] = spec.name
            metrics["provider"] = spec.provider
            metrics["visual_mode"] = self.visual_mode
            metrics["profile_input_mode"] = self.profile_input_mode
            metrics["profile_method"] = self.profile_method
            metrics["profile_method_max_posts"] = self.profile_method_max_posts
            metrics["hierarchical_chunk_size"] = self.hierarchical_chunk_size
            metrics["extractive_k"] = self.extractive_k
            return predictions, metrics
        finally:
            self._release_runtime_caches()

    def _build_anchor_match_client(self) -> AnchorMatchJudgeClient | None:
        if self.anchor_match_model == "exact_match":
            if self.anchor_match_force_llm or self.anchor_match_api_model or self.anchor_match_provider or self.anchor_match_base_url:
                raise ValueError("exact_match is offline and cannot be combined with LLM overrides")
            return None
        if self._anchor_match_client is not None:
            return self._anchor_match_client

        model_key = self.anchor_match_model
        model_spec = DEFAULT_MODEL_SPECS.get(model_key)
        if model_spec is None:
            raise ValueError(f"Unknown anchor-match model: {model_key}. Add it to DEFAULT_MODEL_SPECS first.")
        provider = (self.anchor_match_provider or model_spec.provider).strip().lower()
        if provider in {"chatanywhere", "bailian"}:
            normalized_provider = "openai_compatible"
        elif provider in {"openai_compatible", "vertex", "vertex_batch"}:
            normalized_provider = "vertex" if provider == "vertex_batch" else provider
        elif provider.startswith("vertex"):
            normalized_provider = "vertex"
        else:
            normalized_provider = "openai_compatible"

        api_model = self.anchor_match_api_model or model_spec.api_model
        if not api_model:
            api_model = "gemini-3-flash-preview" if normalized_provider == "vertex" else model_key

        if self.anchor_match_base_url:
            base_url = self.anchor_match_base_url
        elif normalized_provider == "vertex":
            project_id = self.vertex_project_id or _env_or_dotenv("GOOGLE_CLOUD_PROJECT", self._dotenv)
            location = self.vertex_location or _env_or_dotenv("VERTEX_BATCH_LOCATION", self._dotenv, default="global")
            if project_id:
                base_url = (
                    f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models"
                )
            else:
                base_url = "https://aiplatform.googleapis.com/v1"
        elif str(model_spec.base_url or "").strip():
            base_url = str(model_spec.base_url).strip()
        else:
            base_url = "https://api.chatanywhere.tech/v1/chat/completions"

        api_key_env = self.anchor_match_api_key_env or model_spec.api_key_env
        timeout_seconds = self.request_timeout_seconds if self.request_timeout_seconds > 0 else 120

        llm_client = OpenAICompatibleChatClient(
            provider=normalized_provider,
            base_url=base_url,
            model=api_model,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            temperature=0.0,
            max_tokens=300,
        )
        config = AnchorMatchConfig(
            model_name=api_model,
            prompt_version=self.anchor_match_prompt_version,
            prefer_yes_no=not self.anchor_match_force_llm,
            force_llm=self.anchor_match_force_llm,
        )
        self._anchor_match_client = AnchorMatchJudgeClient(
            llm_client=llm_client,
            config=config,
            cache_root=self.anchor_match_cache_dir or (self.output_dir / "anchor_match_cache"),
        )
        return self._anchor_match_client

    def _run_model(self, *, tasks: list[DomainEvalTask], spec: ModelSpec, force: bool) -> dict[str, dict[str, Any]]:
        prediction_cache_key = _profile_input_mode_cache_key(
            visual_mode=self.visual_mode,
            profile_input_mode=self.profile_input_mode,
        )
        if self.profile_method != DIRECT_PROFILE_METHOD:
            method_cache_key = self.profile_method
            if self.profile_method == HIERARCHICAL_PROFILE_METHOD:
                method_cache_key = (
                    f"{self.profile_method}_c{self.hierarchical_chunk_size}_m{self.profile_method_max_posts}"
                )
            elif self.profile_method == EXTRACTIVE_PROFILE_METHOD:
                method_cache_key = f"{self.profile_method}_k{self.extractive_k}_m{self.profile_method_max_posts}"
            prediction_cache_key = f"{method_cache_key}__{prediction_cache_key}"
        model_cache_dir = (
            self.cache_dir
            / _slugify(prediction_cache_key)
            / _slugify(spec.name)
        )
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        predictions: dict[str, dict[str, Any]] = {}
        pending: list[DomainEvalTask] = []
        for task in tasks:
            cache_path = model_cache_dir / f"{_slugify(task.task_id)}.json"
            if cache_path.exists() and not force:
                try:
                    row = _load_json(cache_path)
                    if prediction_visual_mode(row) != self.visual_mode:
                        raise ValueError("cache visual_mode mismatch")
                    if _prediction_profile_input_mode(row) != self.profile_input_mode:
                        raise ValueError("cache profile_input_mode mismatch")
                    if _prediction_profile_method(row) != self.profile_method:
                        raise ValueError("cache profile_method mismatch")
                    if not _prediction_row_is_cacheable(row):
                        raise ValueError("cache row has provider error")
                    row["from_cache"] = True
                    predictions[task.task_id] = row
                    continue
                except Exception:
                    pass
            pending.append(task)

        if pending:
            if self.profile_method in USER_LEVEL_PROFILE_METHODS:
                pending_group_keys = {_user_group_key(task) for task in pending}
                grouped_tasks = [task for task in tasks if _user_group_key(task) in pending_group_keys]
                try:
                    fresh = self._predict_pending(tasks=grouped_tasks, spec=spec)
                except Exception as exc:
                    fresh = {
                        task.task_id: _fallback_prediction(
                            task,
                            model_name=spec.name,
                            error=f"provider_call_failed: {exc}",
                        )
                        for task in grouped_tasks
                    }
                for task in grouped_tasks:
                    if task.task_id in predictions and not force:
                        continue
                    row = fresh.get(task.task_id) or _fallback_prediction(
                        task,
                        model_name=spec.name,
                        error="missing_prediction_row",
                    )
                    row["from_cache"] = False
                    row["cache_write_mode"] = f"{self.profile_method}_user_group"
                    row["visual_mode"] = self.visual_mode
                    row["profile_input_mode"] = self.profile_input_mode
                    row["profile_method"] = self.profile_method
                    row["profile_method_max_posts"] = self.profile_method_max_posts
                    row["hierarchical_chunk_size"] = self.hierarchical_chunk_size
                    row["extractive_k"] = self.extractive_k
                    predictions[task.task_id] = row
                    cache_path = model_cache_dir / f"{_slugify(task.task_id)}.json"
                    cache_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            elif spec.provider in {"mock_oracle", "chatanywhere", "bailian"}:
                total_pending = len(pending)
                try:
                    fresh = self._predict_pending(tasks=pending, spec=spec)
                except Exception as exc:
                    fresh = {
                        task.task_id: _fallback_prediction(
                            task,
                            model_name=spec.name,
                            error=f"provider_call_failed: {exc}",
                        )
                        for task in pending
                    }
                for task in pending:
                    row = fresh.get(task.task_id) or _fallback_prediction(
                        task,
                        model_name=spec.name,
                        error="missing_prediction_row",
                    )
                    row["from_cache"] = False
                    row["cache_write_mode"] = "per_task_batch"
                    row["pending_total"] = total_pending
                    row["visual_mode"] = self.visual_mode
                    row["profile_input_mode"] = self.profile_input_mode
                    row["profile_method"] = self.profile_method
                    predictions[task.task_id] = row
                    cache_path = model_cache_dir / f"{_slugify(task.task_id)}.json"
                    cache_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                try:
                    fresh = self._predict_pending(tasks=pending, spec=spec)
                except Exception as exc:
                    if spec.provider == "bailian_batch":
                        raise
                    fresh = {
                        task.task_id: _fallback_prediction(
                            task,
                            model_name=spec.name,
                            error=f"provider_call_failed: {exc}",
                        )
                        for task in pending
                    }
                for task in pending:
                    row = fresh.get(task.task_id) or _fallback_prediction(
                        task,
                        model_name=spec.name,
                        error="missing_prediction_row",
                    )
                    row["from_cache"] = False
                    row["visual_mode"] = self.visual_mode
                    row["profile_input_mode"] = self.profile_input_mode
                    row["profile_method"] = self.profile_method
                    predictions[task.task_id] = row
                    cache_path = model_cache_dir / f"{_slugify(task.task_id)}.json"
                    cache_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        return predictions

    def _predict_pending(self, *, tasks: list[DomainEvalTask], spec: ModelSpec) -> dict[str, dict[str, Any]]:
        if spec.provider == "mock_oracle":
            out: dict[str, dict[str, Any]] = {}
            for task in tasks:
                out[task.task_id] = _normalize_prediction(
                    task=task,
                    parsed={
                        "domain": task.domain,
                        "status": task.gold_status,
                        "long_term_interest_anchors": [
                            anchor.__dict__ for anchor in task.gold_long_term_anchors[:MAX_LONG_ANCHORS]
                        ],
                        "short_term_interest_anchors": [
                            anchor.__dict__ for anchor in task.gold_short_term_anchors[:MAX_SHORT_ANCHORS]
                        ],
                        "summary_natural_pred": task.gold_summary_natural,
                        "summary_support_post_indices": task.gold_domain_representative_evidence_post_indices[
                            :MAX_SUMMARY_SUPPORT
                        ],
                    },
                    raw_text="",
                    model_name=spec.name,
                )
            return out
        if self.profile_method == HIERARCHICAL_PROFILE_METHOD:
            return self._predict_hierarchical(tasks=tasks, spec=spec)
        if self.profile_method == EXTRACTIVE_PROFILE_METHOD:
            return self._predict_extractive(tasks=tasks, spec=spec)
        if spec.provider in {"chatanywhere", "bailian"}:
            return self._predict_openai_compatible(tasks=tasks, spec=spec)
        if spec.provider == "bailian_batch":
            return self._predict_bailian_batch(tasks=tasks, spec=spec)
        if spec.provider == "vertex_batch":
            return self._predict_vertex_batch(tasks=tasks, spec=spec)
        raise ValueError(f"Unsupported provider: {spec.provider}")
