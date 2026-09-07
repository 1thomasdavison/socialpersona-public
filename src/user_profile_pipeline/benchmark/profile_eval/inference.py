from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import requests

from ...dashscope_batch import (
    MAX_BATCH_REQUEST_BYTES,
    DashScopeBatchInferenceClient,
    build_openai_batch_chat_request,
    dashscope_batch_model_support_reason,
    extract_openai_chat_message_text,
    iter_jsonl_rows,
    request_json_size_bytes,
)
from ...image_text import (
    VISUAL_MODE_NATIVE,
    VISUAL_MODE_TEXT_IMAGE,
    ImageTextGenerator,
    build_chat_client_for_model_spec,
    build_posts_context_bundle_for_visual_mode,
)
from ...llm_client import OpenAICompatibleChatClient
from ...media_inline import image_source_to_data_url
from ...multimodal_context import (
    apply_image_limit,
    build_posts_context_bundle,
    build_retry_posts_context_text,
)
from ...vertex_batch import VertexBatchInferenceClient
from .io import _env_or_dotenv
from .modes import (
    _posts_for_profile_input_mode,
    _profile_input_mode_uses_image_captions,
    _slugify,
)
from .normalization import (
    _context_limit_lowres_max_dim,
    _extract_vertex_text,
    _fallback_prediction,
    _looks_like_context_limit_error,
    _looks_like_data_inspection_error,
    _looks_like_transport_timeout_error,
    _normalize_prediction,
    _parse_prediction_json,
    _raise_on_duplicate_ids,
    _temporary_inline_image_settings,
)
from .profile_methods import _build_domain_prompt, _build_domain_prompt_neutral, _guess_image_mime
from .prompts import (
    BENCHMARK_EVAL_JSON_SCHEMA_HINT,
    BENCHMARK_EVAL_SYSTEM_PROMPT,
    BENCHMARK_EVAL_SYSTEM_PROMPT_NEUTRAL,
)
from .specs import (
    DomainEvalTask,
    ModelSpec,
    PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS,
    RETRY_CONTEXT_VERSION,
)


class ProviderInferenceMixin:
    def _get_image_text_generator(self, *, spec: ModelSpec) -> ImageTextGenerator:
        caption_spec = self.image_text_model_spec or spec
        if not caption_spec.multimodal:
            raise ValueError(f"visual_mode=text_image requires a multimodal model, got {caption_spec.name}")
        existing = self._image_text_generators.get(caption_spec.name)
        if existing is not None:
            return existing
        client = build_chat_client_for_model_spec(
            caption_spec,
            timeout_seconds=self.image_text_timeout_seconds,
            max_tokens=self.image_text_max_tokens,
            temperature=0.0,
        )
        generator = ImageTextGenerator(
            client=client,
            model_name=caption_spec.name,
            cache_dir=self.cache_dir / "_image_text" / _slugify(caption_spec.name),
            force=self.force_image_text,
        )
        self._image_text_generators[caption_spec.name] = generator
        return generator

    def _build_context_bundle_for_task(self, *, task: DomainEvalTask, spec: ModelSpec) -> Any:
        if _profile_input_mode_uses_image_captions(self.profile_input_mode):
            bundle_owner = self.image_text_model_spec.name if self.image_text_model_spec else spec.name
        else:
            bundle_owner = "__shared__"
        cache_key = (bundle_owner, task.user_id, self.visual_mode, self.profile_input_mode)
        existing = self._visual_bundle_cache.get(cache_key)
        if existing is not None:
            return existing
        if self.profile_input_mode != PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS:
            posts_for_context = _posts_for_profile_input_mode(task.posts, profile_input_mode=self.profile_input_mode)
            if _profile_input_mode_uses_image_captions(self.profile_input_mode):
                image_text_map = self._get_image_text_generator(spec=spec).describe_posts(task.posts)
                bundle = build_posts_context_bundle(
                    posts_for_context,
                    image_text_map=image_text_map,
                    drop_raw_images=True,
                )
            else:
                bundle = build_posts_context_bundle(posts_for_context, drop_raw_images=True)
            self._visual_bundle_cache[cache_key] = bundle
            return bundle
        if self.visual_mode == VISUAL_MODE_NATIVE:
            bundle = build_posts_context_bundle(task.posts)
            self._visual_bundle_cache[cache_key] = bundle
            return bundle
        if self.visual_mode == VISUAL_MODE_TEXT_IMAGE:
            image_text_map = self._get_image_text_generator(spec=spec).describe_posts(task.posts)
            bundle = build_posts_context_bundle_for_visual_mode(
                posts=task.posts,
                visual_mode=self.visual_mode,
                image_text_map=image_text_map,
            )
            self._visual_bundle_cache[cache_key] = bundle
            return bundle
        bundle = build_posts_context_bundle_for_visual_mode(
            posts=task.posts,
            visual_mode=self.visual_mode,
        )
        self._visual_bundle_cache[cache_key] = bundle
        return bundle

    def _build_input_for_task(
        self,
        task: DomainEvalTask,
        spec: ModelSpec,
        *,
        one_image_per_post: bool = False,
        context_text_override: str | None = None,
    ) -> tuple[str, list[str]]:
        bundle = self._build_context_bundle_for_task(task=task, spec=spec)
        context_text = str(context_text_override or bundle.text or "")
        selected_urls = bundle.one_image_per_post_urls if one_image_per_post else bundle.image_urls
        selected_limit = self.max_images_when_trim if one_image_per_post else self.max_images
        image_urls = apply_image_limit(selected_urls, selected_limit)
        user_prompt = _build_domain_prompt(task, context_text) if self.profile_eval_prompt_variant != "neutral" else _build_domain_prompt_neutral(task, context_text)
        if not spec.multimodal:
            return user_prompt, []
        return user_prompt, image_urls

    def _clear_openai_image_caches(self) -> None:
        self._openai_image_data_url_cache.clear()
        try:
            image_source_to_data_url.cache_clear()
        except Exception:
            pass
        gc.collect()

    def _predict_one_openai_task(
        self,
        *,
        client: OpenAICompatibleChatClient,
        task: DomainEvalTask,
        spec: ModelSpec,
        one_image_per_post: bool = False,
        lowres_max_dim: int = 0,
        retry_mode: str = "",
        retry_context_version: str = "",
        context_text_override: str | None = None,
    ) -> dict[str, Any]:
        user_prompt, image_urls = self._build_input_for_task(
            task,
            spec,
            one_image_per_post=one_image_per_post,
            context_text_override=context_text_override,
        )
        self._clear_openai_image_caches()
        try:
            image_urls_for_provider = self._prepare_openai_image_urls(
                image_urls,
                inline_image_max_dim=(lowres_max_dim if lowres_max_dim > 0 else None),
                inline_reencode=(lowres_max_dim > 0),
            )
            system_prompt = BENCHMARK_EVAL_SYSTEM_PROMPT if self.profile_eval_prompt_variant != "neutral" else BENCHMARK_EVAL_SYSTEM_PROMPT_NEUTRAL
            parsed = client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_urls=image_urls_for_provider or None,
                json_schema_hint=BENCHMARK_EVAL_JSON_SCHEMA_HINT,
            )
            prediction = _normalize_prediction(
                task=task,
                parsed=parsed,
                raw_text="",
                model_name=spec.name,
            )
            if retry_mode:
                prediction["retry_mode"] = retry_mode
            if retry_context_version:
                prediction["retry_context_version"] = retry_context_version
            return prediction
        finally:
            self._clear_openai_image_caches()

    def _predict_one_with_retry(
        self,
        *,
        client: OpenAICompatibleChatClient,
        task: DomainEvalTask,
        spec: ModelSpec,
    ) -> dict[str, Any]:
        """Predict one task with the full retry chain.  Thread-safe (reads from caches only)."""
        try:
            return self._predict_one_openai_task(
                client=client,
                task=task,
                spec=spec,
            )
        except Exception as exc:
            err_text = str(exc)
            try:
                from tenacity import RetryError  # type: ignore

                if isinstance(exc, RetryError):
                    inner = exc.last_attempt.exception()
                    if inner is not None:
                        err_text = f"{err_text}; last_exception={inner}"
            except Exception:
                pass
            retry_prediction: dict[str, Any] | None = None
            if spec.multimodal and _looks_like_context_limit_error(err_text):
                try:
                    retry_prediction = self._predict_one_openai_task(
                        client=client,
                        task=task,
                        spec=spec,
                        one_image_per_post=True,
                        retry_mode="one_image_per_post",
                    )
                except Exception as retry_exc:
                    retry_err_text = str(retry_exc)
                    try:
                        from tenacity import RetryError  # type: ignore

                        if isinstance(retry_exc, RetryError):
                            inner = retry_exc.last_attempt.exception()
                            if inner is not None:
                                retry_err_text = f"{retry_err_text}; last_exception={inner}"
                    except Exception:
                        pass
                    err_text = f"{err_text}; retry_one_image_per_post_failed: {retry_err_text}"
                    if _looks_like_context_limit_error(retry_err_text):
                        lowres_max_dim = _context_limit_lowres_max_dim()
                        try:
                            retry_prediction = self._predict_one_openai_task(
                                client=client,
                                task=task,
                                spec=spec,
                                one_image_per_post=True,
                                lowres_max_dim=lowres_max_dim,
                                retry_mode=f"one_image_per_post_lowres_{lowres_max_dim}",
                            )
                        except Exception as lowres_exc:
                            lowres_err_text = str(lowres_exc)
                            try:
                                from tenacity import RetryError  # type: ignore

                                if isinstance(lowres_exc, RetryError):
                                    inner = lowres_exc.last_attempt.exception()
                                    if inner is not None:
                                        lowres_err_text = f"{lowres_err_text}; last_exception={inner}"
                            except Exception:
                                pass
                            err_text = f"{err_text}; retry_lowres_failed: {lowres_err_text}"
            bundle = None
            if retry_prediction is None and (_looks_like_transport_timeout_error(err_text) or _looks_like_data_inspection_error(err_text)):
                try:
                    bundle = self._build_context_bundle_for_task(task=task, spec=spec)
                except Exception:
                    bundle = None
            if retry_prediction is None and bundle is not None and _looks_like_transport_timeout_error(err_text):
                compact_context = build_retry_posts_context_text(getattr(bundle, "payload", {}) or {}, redact_text=False)
                try:
                    retry_prediction = self._predict_one_openai_task(
                        client=client,
                        task=task,
                        spec=spec,
                        retry_mode="compact_context",
                        retry_context_version=RETRY_CONTEXT_VERSION,
                        context_text_override=compact_context,
                    )
                except Exception as compact_exc:
                    compact_err_text = str(compact_exc)
                    try:
                        from tenacity import RetryError  # type: ignore

                        if isinstance(compact_exc, RetryError):
                            inner = compact_exc.last_attempt.exception()
                            if inner is not None:
                                compact_err_text = f"{compact_err_text}; last_exception={inner}"
                    except Exception:
                        pass
                    err_text = f"{err_text}; retry_compact_context_failed: {compact_err_text}"
            if retry_prediction is None and bundle is not None and _looks_like_data_inspection_error(err_text):
                redacted_context = build_retry_posts_context_text(getattr(bundle, "payload", {}) or {}, redact_text=True)
                try:
                    retry_prediction = self._predict_one_openai_task(
                        client=client,
                        task=task,
                        spec=spec,
                        retry_mode="compact_context_text_redacted",
                        retry_context_version=RETRY_CONTEXT_VERSION,
                        context_text_override=redacted_context,
                    )
                except Exception as redacted_exc:
                    redacted_err_text = str(redacted_exc)
                    try:
                        from tenacity import RetryError  # type: ignore

                        if isinstance(redacted_exc, RetryError):
                            inner = redacted_exc.last_attempt.exception()
                            if inner is not None:
                                redacted_err_text = f"{redacted_err_text}; last_exception={inner}"
                    except Exception:
                        pass
                    err_text = f"{err_text}; retry_text_redacted_failed: {redacted_err_text}"
            if retry_prediction is not None:
                return retry_prediction
            raise RuntimeError(f"chat_json_failed: {err_text}")

    def _predict_openai_compatible(self, *, tasks: list[DomainEvalTask], spec: ModelSpec) -> dict[str, dict[str, Any]]:
        base_url = spec.base_url or "https://api.chatanywhere.tech/v1/chat/completions"
        api_key_env = spec.api_key_env or "CHATANYWHERE_API_KEY"
        timeout_seconds = self.request_timeout_seconds or int(os.environ.get("BENCHMARK_PROVIDER_TIMEOUT_SECONDS", "0") or 0)
        client = OpenAICompatibleChatClient(
            provider="openai_compatible",
            base_url=base_url,
            model=spec.api_model or spec.name,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            repair_provider=self.json_repair_provider or "",
            repair_base_url=self.json_repair_base_url or "",
            repair_api_key_env=self.json_repair_api_key_env or "",
            repair_model=self.json_repair_api_model or spec.api_model or spec.name,
        )
        out: dict[str, dict[str, Any]] = {}
        total = len(tasks)
        workers = self.max_parallel_tasks

        if workers > 1:
            # ---- concurrent path ----
            from threading import Lock
            print_lock = Lock()
            completed = [0]

            def _process_one(task: DomainEvalTask, idx: int) -> tuple[str, dict[str, Any]]:
                result = self._predict_one_with_retry(client=client, task=task, spec=spec)
                with print_lock:
                    completed[0] += 1
                    c = completed[0]
                if c % 10 == 0 or c == total:
                    print(f"[{spec.name}] progress {c}/{total}", flush=True)
                return (task.task_id, result)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_one, task, idx): task
                    for idx, task in enumerate(tasks, start=1)
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        tid, prediction = future.result()
                        out[tid] = prediction
                    except Exception as exc:
                        err_text = str(exc)
                        print(f"[{spec.name}] failed task_id={task.task_id}: {err_text}", flush=True)
                        out[task.task_id] = _fallback_prediction(
                            task,
                            model_name=spec.name,
                            error=f"chat_json_failed: {err_text}",
                        )
            self._clear_openai_image_caches()
        else:
            # ---- sequential path (original) ----
            for idx, task in enumerate(tasks, start=1):
                print(f"[{spec.name}] predicting {idx}/{total} task_id={task.task_id}", flush=True)
                try:
                    out[task.task_id] = self._predict_one_with_retry(
                        client=client,
                        task=task,
                        spec=spec,
                    )
                except Exception as exc:
                    err_text = str(exc)
                    print(f"[{spec.name}] failed task_id={task.task_id}: {err_text}", flush=True)
                    out[task.task_id] = _fallback_prediction(
                        task,
                        model_name=spec.name,
                        error=f"chat_json_failed: {err_text}",
                    )
        return out

    def _fetch_inline_part_from_http_url(self, image_url: str) -> dict[str, Any] | None:
        url = (image_url or "").strip()
        if not url:
            return None
        if url in self._inline_image_part_cache:
            return self._inline_image_part_cache[url]

        timeout_seconds = int(os.environ.get("BENCHMARK_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "20") or 20)
        max_bytes = int(os.environ.get("BENCHMARK_INLINE_IMAGE_MAX_BYTES", str(8 * 1024 * 1024)) or (8 * 1024 * 1024))
        headers = {
            "User-Agent": "benchmark-evaluator/1.0",
        }
        part: dict[str, Any] | None = None
        try:
            response = requests.get(url, timeout=timeout_seconds, stream=True, headers=headers)
            response.raise_for_status()
            content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            data = response.content
            if not data or len(data) > max_bytes:
                part = None
            else:
                mime = content_type if content_type.startswith("image/") else _guess_image_mime(url)
                if not mime.startswith("image/"):
                    mime = "image/jpeg"
                part = {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }
        except Exception:
            part = None

        self._inline_image_part_cache[url] = part
        return part

    def _load_inline_part_from_local_file(self, local_path: str) -> dict[str, Any] | None:
        path = Path(local_path).expanduser()
        if not path.exists() or not path.is_file():
            return None
        key = str(path.resolve())
        if key in self._inline_image_part_cache:
            return self._inline_image_part_cache[key]

        max_bytes = int(os.environ.get("BENCHMARK_INLINE_IMAGE_MAX_BYTES", str(8 * 1024 * 1024)) or (8 * 1024 * 1024))
        part: dict[str, Any] | None = None
        try:
            data = path.read_bytes()
            if not data or len(data) > max_bytes:
                part = None
            else:
                mime = _guess_image_mime(str(path))
                if not mime.startswith("image/"):
                    mime = "image/jpeg"
                part = {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }
        except Exception:
            part = None
        self._inline_image_part_cache[key] = part
        return part

    def _to_vertex_image_part(self, image_url: str) -> dict[str, Any] | None:
        url = (image_url or "").strip()
        if not url:
            return None
        if not url.startswith(("http://", "https://", "gs://", "data:")):
            inline_part = self._load_inline_part_from_local_file(url)
            if inline_part is not None:
                return inline_part
            return None
        if url.startswith("data:"):
            header, sep, data = url.partition(",")
            if not sep or not data or ";base64" not in header:
                return None
            mime = header[5:].split(";", 1)[0].strip() or "image/jpeg"
            if not mime.startswith("image/"):
                return None
            return {"inlineData": {"mimeType": mime, "data": data}}
        if url.startswith(("http://", "https://")):
            # Vertex batch has stricter limits for URL links. Prefer inline bytes for HTTP(S) images.
            inline_part = self._fetch_inline_part_from_http_url(url)
            if inline_part is not None:
                return inline_part
        if url.startswith(("http://", "https://", "gs://")):
            mime = _guess_image_mime(url)
            if mime and not mime.startswith("image/"):
                return None
            return {"fileData": {"mimeType": mime or "image/jpeg", "fileUri": url}}
        return None

    def _to_openai_image_url(self, image_url: str) -> str | None:
        url = (image_url or "").strip()
        if not url:
            return None
        if url.startswith("gs://"):
            return None
        if not url.startswith(("http://", "https://", "data:")):
            path = Path(url).expanduser()
            if not path.exists() or not path.is_file():
                return None
        # Allow HTTP(S) URL passthrough (skip base64 download+encode) when env var is set.
        if url.startswith(("http://", "https://")) and int(os.environ.get("BENCHMARK_IMAGE_URL_PASSTHROUGH", "0") or 0):
            return url
        return image_source_to_data_url(url)

    def _prepare_openai_image_urls(
        self,
        image_urls: list[str],
        *,
        inline_image_max_dim: int | None = None,
        inline_reencode: bool | None = None,
    ) -> list[str]:
        out: list[str] = []
        with _temporary_inline_image_settings(
            max_dim=inline_image_max_dim,
            reencode=inline_reencode,
        ):
            for item in image_urls or []:
                converted = self._to_openai_image_url(item)
                if converted:
                    out.append(converted)
        return out

    def _build_openai_batch_request_for_task(
        self,
        *,
        task: DomainEvalTask,
        spec: ModelSpec,
        one_image_per_post: bool = False,
        lowres_max_dim: int = 0,
    ) -> dict[str, Any]:
        user_prompt, image_urls = self._build_input_for_task(
            task,
            spec,
            one_image_per_post=one_image_per_post,
        )
        self._clear_openai_image_caches()
        try:
            image_urls_for_provider = self._prepare_openai_image_urls(
                image_urls,
                inline_image_max_dim=(lowres_max_dim if lowres_max_dim > 0 else None),
                inline_reencode=(lowres_max_dim > 0),
            )
            request_row = build_openai_batch_chat_request(
                custom_id=task.task_id,
                model=spec.api_model or spec.name,
                system_prompt=BENCHMARK_EVAL_SYSTEM_PROMPT if self.profile_eval_prompt_variant != "neutral" else BENCHMARK_EVAL_SYSTEM_PROMPT_NEUTRAL,
                user_prompt=user_prompt,
                image_urls=image_urls_for_provider or None,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                force_json=True,
            )
            row_size = request_json_size_bytes(request_row)
            if row_size > MAX_BATCH_REQUEST_BYTES:
                raise ValueError(
                    f"batch request exceeds 6MB per-request limit: task_id={task.task_id} bytes={row_size}"
                )
            return request_row
        finally:
            self._clear_openai_image_caches()

    def _run_dashscope_batch_requests(
        self,
        *,
        spec: ModelSpec,
        request_rows: list[dict[str, Any]],
        batch_name: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], Path]:
        if not request_rows:
            return {}, {}, self.output_dir / "dashscope_batch_runs"
        _raise_on_duplicate_ids(
            [str(row.get("custom_id") or "").strip() for row in request_rows],
            label=f"DashScope batch request custom_id for model={spec.name}",
        )

        client = DashScopeBatchInferenceClient(
            base_url=spec.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env=spec.api_key_env or "ALI_BAILIAN",
            timeout_seconds=int(os.environ.get("BENCHMARK_PROVIDER_TIMEOUT_SECONDS", "120") or 120),
        )
        run_tag = f"{_slugify(spec.name)}_{batch_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = self.output_dir / "dashscope_batch_runs" / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / "input.jsonl"
        with input_path.open("w", encoding="utf-8") as fh:
            for row in request_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        upload = client.upload_batch_file(local_path=input_path)
        input_file_id = str(upload.get("id") or "").strip()
        if not input_file_id:
            raise RuntimeError(f"DashScope batch upload response missing file id: {upload}")

        metadata: dict[str, Any] = {}
        if self.bailian_batch_callback_url:
            metadata["ds_batch_finish_callback"] = self.bailian_batch_callback_url
        batch = client.create_batch(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=self.bailian_batch_completion_window,
            metadata=metadata,
        )
        batch_id = str(batch.get("id") or "").strip()
        if not batch_id:
            raise RuntimeError(f"DashScope batch create response missing batch id: {batch}")
        batch_final = client.wait_batch(
            batch_id=batch_id,
            poll_seconds=self.bailian_batch_poll_seconds,
            max_wait_seconds=self.bailian_batch_max_wait_seconds,
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

        success_rows: dict[str, dict[str, Any]] = {}
        error_rows: dict[str, str] = {}
        seen_output_custom_ids: set[str] = set()
        duplicate_output_custom_ids: list[str] = []
        result_path = run_dir / "result.jsonl"
        if result_path.exists():
            for row in iter_jsonl_rows(result_path):
                custom_id = str(row.get("custom_id") or "").strip()
                if not custom_id:
                    continue
                if custom_id in seen_output_custom_ids:
                    duplicate_output_custom_ids.append(custom_id)
                    continue
                seen_output_custom_ids.add(custom_id)
                error = row.get("error")
                response = row.get("response")
                if error:
                    error_rows[custom_id] = json.dumps(error, ensure_ascii=False)
                    continue
                success_rows[custom_id] = response if isinstance(response, dict) else {}
        error_path = run_dir / "errors.jsonl"
        if error_path.exists():
            for row in iter_jsonl_rows(error_path):
                custom_id = str(row.get("custom_id") or "").strip()
                error = row.get("error")
                if not custom_id or not error:
                    continue
                if custom_id in seen_output_custom_ids:
                    duplicate_output_custom_ids.append(custom_id)
                    continue
                seen_output_custom_ids.add(custom_id)
                if custom_id and error:
                    error_rows[custom_id] = json.dumps(error, ensure_ascii=False)
        if duplicate_output_custom_ids:
            preview = ", ".join(sorted(set(duplicate_output_custom_ids))[:10])
            raise RuntimeError(
                f"DashScope batch output contains duplicate custom_id rows for model={spec.name}: {preview}"
            )
        return success_rows, error_rows, run_dir

    def _predict_bailian_batch(self, *, tasks: list[DomainEvalTask], spec: ModelSpec) -> dict[str, dict[str, Any]]:
        _raise_on_duplicate_ids(
            [task.task_id for task in tasks],
            label=f"task_id in Bailian batch input for model={spec.name}",
        )
        has_images = bool(spec.multimodal and self.visual_mode == VISUAL_MODE_NATIVE)
        support_reason = dashscope_batch_model_support_reason(
            model=spec.api_model or spec.name,
            has_images=has_images,
            base_url=spec.base_url,
        )
        if support_reason:
            raise ValueError(support_reason)

        predictions: dict[str, dict[str, Any]] = {}
        pending: dict[str, DomainEvalTask] = {task.task_id: task for task in tasks}
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

        for stage_name, one_image_per_post, stage_lowres_max_dim, retry_mode in stages:
            if not pending:
                break
            request_rows: list[dict[str, Any]] = []
            build_failures: dict[str, str] = {}
            for task_id, task in list(pending.items()):
                try:
                    request_rows.append(
                        self._build_openai_batch_request_for_task(
                            task=task,
                            spec=spec,
                            one_image_per_post=one_image_per_post,
                            lowres_max_dim=stage_lowres_max_dim,
                        )
                    )
                except Exception as exc:
                    build_failures[task_id] = str(exc)
            for task_id, err_text in build_failures.items():
                task = pending.pop(task_id)
                predictions[task_id] = _fallback_prediction(
                    task,
                    model_name=spec.name,
                    error=f"batch_request_build_failed: {err_text}",
                )

            if not request_rows:
                continue

            success_rows, error_rows, _ = self._run_dashscope_batch_requests(
                spec=spec,
                request_rows=request_rows,
                batch_name=stage_name,
            )

            for request_row in request_rows:
                task_id = str(request_row.get("custom_id") or "").strip()
                task = pending.get(task_id)
                if task is None:
                    continue
                response = success_rows.get(task_id)
                if response:
                    body = response.get("body") or {}
                    try:
                        raw_text = extract_openai_chat_message_text(body if isinstance(body, dict) else {})
                        parsed = _parse_prediction_json(raw_text)
                        if parsed is None:
                            raise ValueError("batch_parse_failed")
                        prediction = _normalize_prediction(
                            task=task,
                            parsed=parsed,
                            raw_text=raw_text,
                            model_name=spec.name,
                        )
                        if retry_mode:
                            prediction["retry_mode"] = retry_mode
                        predictions[task_id] = prediction
                        pending.pop(task_id, None)
                        continue
                    except Exception as exc:
                        error_rows[task_id] = f"batch_parse_failed: {exc}"

                error_text = str(error_rows.get(task_id) or "dashscope_batch_missing_response")
                if _looks_like_context_limit_error(error_text) and stage_name != stages[-1][0]:
                    continue
                pending.pop(task_id, None)
                predictions[task_id] = _fallback_prediction(
                    task,
                    model_name=spec.name,
                    error=f"dashscope_batch_failed: {error_text}",
                )

        for task_id, task in pending.items():
            predictions[task_id] = _fallback_prediction(
                task,
                model_name=spec.name,
                error="dashscope_batch_failed: context_limit_after_all_retries",
            )
        return predictions

    def _predict_vertex_batch(self, *, tasks: list[DomainEvalTask], spec: ModelSpec) -> dict[str, dict[str, Any]]:
        _raise_on_duplicate_ids(
            [task.task_id for task in tasks],
            label=f"task_id in Vertex batch input for model={spec.name}",
        )
        project_id = self.vertex_project_id or _env_or_dotenv("GOOGLE_CLOUD_PROJECT", self._dotenv)
        location = self.vertex_location or _env_or_dotenv("VERTEX_BATCH_LOCATION", self._dotenv, default="global")
        bucket = self.vertex_bucket or _env_or_dotenv("VERTEX_BATCH_BUCKET", self._dotenv)
        if not project_id:
            raise ValueError("Missing Vertex project id. Set --vertex-project-id or GOOGLE_CLOUD_PROJECT.")
        if not bucket:
            raise ValueError("Missing Vertex batch bucket. Set --vertex-bucket or VERTEX_BATCH_BUCKET.")
        client = VertexBatchInferenceClient(
            project_id=project_id,
            location=location,
            model=spec.api_model or spec.name,
            timeout_seconds=int(os.environ.get("BENCHMARK_PROVIDER_TIMEOUT_SECONDS", "0") or 0),
            access_token_env=self.vertex_access_token_env,
        )
        run_tag = f"{_slugify(spec.name)}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = self.output_dir / "vertex_batch_runs" / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        input_local = run_dir / "input.jsonl"
        with input_local.open("w", encoding="utf-8") as f:
            for task in tasks:
                user_prompt, image_urls = self._build_input_for_task(task, spec)
                parts: list[dict[str, Any]] = [{"text": user_prompt}]
                for url in image_urls:
                    part = self._to_vertex_image_part(url)
                    if part is not None:
                        parts.append(part)
                request_line = {
                    "custom_id": task.task_id,
                    "request": {
                        "systemInstruction": {"role": "system", "parts": [{"text": BENCHMARK_EVAL_SYSTEM_PROMPT}]},
                        "contents": [{"role": "user", "parts": parts}],
                        "generationConfig": {"temperature": spec.temperature, "responseMimeType": "application/json"},
                    },
                }
                f.write(json.dumps(request_line, ensure_ascii=False) + "\n")

        gcs_prefix = f"benchmark_eval/{run_tag}"
        input_uri = f"gs://{bucket}/{gcs_prefix}/input.jsonl"
        output_prefix = f"gs://{bucket}/{gcs_prefix}/output"
        client.upload_local_file(local_path=input_local, gcs_uri=input_uri)
        job = client.submit_gcs_batch_job(
            input_gcs_uri=input_uri,
            output_gcs_uri_prefix=output_prefix,
            display_name=f"benchmark-{run_tag}",
        )
        job_name = str(job.get("name", "")).strip()
        if not job_name:
            raise RuntimeError(f"Vertex batch submit response missing job name: {job}")
        job_final = client.wait_batch_job(
            job_name_or_id=job_name,
            poll_seconds=self.poll_seconds,
            max_wait_seconds=self.max_wait_seconds,
        )
        state = str(job_final.get("state", "")).upper()
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Vertex batch job failed: {job_final}")

        output_uri_prefix = client.get_output_uri_prefix(job_final) or output_prefix
        output_local_dir = run_dir / "outputs"
        client.download_output_prefix(
            output_gcs_uri_prefix=output_uri_prefix,
            local_dir=output_local_dir,
            only_jsonl=True,
        )

        out: dict[str, dict[str, Any]] = {}
        by_task = {task.task_id: task for task in tasks}
        output_files = sorted(output_local_dir.rglob("*.jsonl"))
        unknown_output_rows: list[dict[str, Any]] = []
        duplicate_output_task_ids: list[str] = []
        for path in output_files:
            for raw in path.read_text(encoding="utf-8").splitlines():
                text = raw.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    unknown_output_rows.append(
                        {
                            "file": str(path),
                            "error": "vertex_output_invalid_json_line",
                            "raw_text": text[:1000],
                        }
                    )
                    continue

                task_id = str(row.get("custom_id", "")).strip()
                if not task_id or task_id not in by_task:
                    unknown_output_rows.append(
                        {
                            "file": str(path),
                            "error": "vertex_output_unknown_custom_id",
                            "custom_id": task_id,
                            "row": row,
                        }
                    )
                    continue
                if task_id in out:
                    duplicate_output_task_ids.append(task_id)
                    continue
                task = by_task[task_id]

                status = str(row.get("status", "")).strip()
                response = row.get("response") or {}
                raw_text = _extract_vertex_text(response if isinstance(response, dict) else {})
                parsed = _parse_prediction_json(raw_text)
                if status:
                    out[task_id] = _fallback_prediction(
                        task,
                        model_name=spec.name,
                        error=f"vertex_batch_status_error: {status}",
                    )
                    continue
                if parsed is None:
                    out[task_id] = _fallback_prediction(
                        task,
                        model_name=spec.name,
                        error="vertex_parse_failed",
                    )
                    continue
                out[task_id] = _normalize_prediction(
                    task=task,
                    parsed=parsed,
                    raw_text=raw_text,
                    model_name=spec.name,
                )
        if duplicate_output_task_ids:
            preview = ", ".join(sorted(set(duplicate_output_task_ids))[:10])
            raise RuntimeError(
                f"Vertex batch output contains duplicate custom_id rows for model={spec.name}: {preview}"
            )
        if unknown_output_rows:
            diagnostics_path = run_dir / "unknown_output_rows.json"
            diagnostics_path.write_text(
                json.dumps(unknown_output_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        for task in tasks:
            if task.task_id not in out:
                out[task.task_id] = _fallback_prediction(task, model_name=spec.name, error="vertex_missing_prediction_row")
        return out
