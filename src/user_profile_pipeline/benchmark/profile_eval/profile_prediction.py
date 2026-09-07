from __future__ import annotations

import json
import os
from typing import Any

from ...llm_client import OpenAICompatibleChatClient
from ...schemas import DOMAIN_NAMES
from ..evaluator.precheck import MAX_LONG_ANCHORS, MAX_SHORT_ANCHORS
from .normalization import _normalize_prediction
from .profile_methods import (
    _build_extractive_abstractive_prompt,
    _build_extractive_selection_prompt,
    _build_hierarchical_chunk_prompt,
    _build_hierarchical_global_prompt,
    _build_post_id_to_index_map,
    _build_selected_posts_with_content,
    _extract_final_profile_domain_rows,
    _interest_items_to_anchors,
    _profile_is_active,
    _time_span_for_rows,
    _timeline_rows_from_payload,
    _user_group_key,
)
from .prompts import (
    EXTRACTIVE_ABSTRACTIVE_SYSTEM_PROMPT,
    EXTRACTIVE_SELECTION_JSON_SCHEMA_HINT,
    EXTRACTIVE_SELECTION_SYSTEM_PROMPT,
    FINAL_PROFILE_JSON_SCHEMA_HINT,
    HIERARCHICAL_CHUNK_JSON_SCHEMA_HINT,
    HIERARCHICAL_CHUNK_SYSTEM_PROMPT,
    HIERARCHICAL_GLOBAL_SYSTEM_PROMPT,
)
from .specs import DomainEvalTask, ModelSpec


class ProfilePredictionMixin:
    def _build_profile_client(self, *, spec: ModelSpec) -> OpenAICompatibleChatClient:
        if spec.provider not in {"chatanywhere", "bailian"}:
            raise ValueError(f"{self.profile_method} supports chatanywhere/bailian providers only, got {spec.provider}")
        base_url = spec.base_url or "https://api.chatanywhere.tech/v1/chat/completions"
        api_key_env = spec.api_key_env or "CHATANYWHERE_API_KEY"
        timeout_seconds = self.request_timeout_seconds or int(os.environ.get("BENCHMARK_PROVIDER_TIMEOUT_SECONDS", "0") or 0)
        method_max_tokens = int(os.environ.get("PROFILE_METHOD_MAX_TOKENS", "4000") or 4000)
        return OpenAICompatibleChatClient(
            provider="openai_compatible",
            base_url=base_url,
            model=spec.api_model or spec.name,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            temperature=spec.temperature,
            max_tokens=max(spec.max_tokens, method_max_tokens),
            repair_provider=self.json_repair_provider or "",
            repair_base_url=self.json_repair_base_url or "",
            repair_api_key_env=self.json_repair_api_key_env or "",
            repair_model=self.json_repair_api_model or spec.api_model or spec.name,
        )

    def _method_profile_to_prediction(
        self,
        *,
        task: DomainEvalTask,
        profile: dict[str, Any],
        final_profile: dict[str, Any],
        post_id_to_index: dict[str, int],
        model_name: str,
        artifact_fields: dict[str, Any],
        error: str = "",
    ) -> dict[str, Any]:
        status = "active" if _profile_is_active(profile) else "inactive"
        parsed = {
            "status": status,
            "long_term_interest_anchors": _interest_items_to_anchors(
                profile.get("stable_interests") or profile.get("long_term_interests") or [],
                post_id_to_index=post_id_to_index,
                limit_items=MAX_LONG_ANCHORS,
            ),
            "short_term_interest_anchors": _interest_items_to_anchors(
                profile.get("recent_interests") or profile.get("short_term_interests") or [],
                post_id_to_index=post_id_to_index,
                limit_items=MAX_SHORT_ANCHORS,
            ),
            "summary_natural_pred": "",
            "summary_support_post_indices": [],
        }
        prediction = _normalize_prediction(
            task=task,
            parsed=parsed,
            raw_text=json.dumps(profile, ensure_ascii=False, separators=(",", ":")),
            model_name=model_name,
            error=error,
        )
        prediction[f"{self.profile_method}_profile"] = profile
        prediction[f"{self.profile_method}_final_profile"] = final_profile
        for key, value in artifact_fields.items():
            prediction[key] = value
        return prediction

    def _predictions_from_final_profile(
        self,
        *,
        group_tasks: list[DomainEvalTask],
        final_profile: dict[str, Any],
        post_id_to_index: dict[str, int],
        model_name: str,
        artifact_fields: dict[str, Any],
        final_profile_error: str = "",
    ) -> dict[str, dict[str, Any]]:
        profiles_by_domain = _extract_final_profile_domain_rows(final_profile)
        out: dict[str, dict[str, Any]] = {}
        for task in group_tasks:
            row_error = ""
            profile = profiles_by_domain.get(task.domain)
            if profile is None:
                if final_profile_error:
                    row_error = f"{self.profile_method}_missing_domain: {task.domain}; final_profile_failed: {final_profile_error}"
                profile = {
                    "domain": task.domain,
                    "active": False,
                    "stable_interests": [],
                    "recent_interests": [],
                    "weak_or_cautionary_interests": [],
                }
            out[task.task_id] = self._method_profile_to_prediction(
                task=task,
                profile=profile,
                final_profile=final_profile,
                post_id_to_index=post_id_to_index,
                model_name=model_name,
                artifact_fields=artifact_fields,
                error=row_error,
            )
        return out

    def _predict_hierarchical(self, *, tasks: list[DomainEvalTask], spec: ModelSpec) -> dict[str, dict[str, Any]]:
        if not tasks:
            return {}
        client = self._build_profile_client(spec=spec)
        grouped: dict[tuple[str, str, str], list[DomainEvalTask]] = {}
        for task in tasks:
            grouped.setdefault(_user_group_key(task), []).append(task)

        out: dict[str, dict[str, Any]] = {}
        total_groups = len(grouped)
        for group_idx, (_, group_tasks) in enumerate(grouped.items(), start=1):
            first_task = group_tasks[0]
            print(
                f"[{spec.name}:hierarchical] predicting user_group {group_idx}/{total_groups} "
                f"user_id={first_task.user_id}",
                flush=True,
            )
            bundle = self._build_context_bundle_for_task(task=first_task, spec=spec)
            timeline_rows = _timeline_rows_from_payload(
                getattr(bundle, "payload", {}) or {},
                max_posts=self.profile_method_max_posts,
                chronological=True,
            )
            post_id_to_index = _build_post_id_to_index_map(first_task.posts)
            chunk_summaries: list[dict[str, Any]] = []
            chunk_errors: list[dict[str, str]] = []
            chunk_size = max(1, self.hierarchical_chunk_size)
            for chunk_start in range(0, len(timeline_rows), chunk_size):
                chunk_rows = timeline_rows[chunk_start:chunk_start + chunk_size]
                chunk_id = f"{first_task.user_id}_chunk_{len(chunk_summaries) + 1:03d}"
                try:
                    summary = client.chat_json(
                        system_prompt=HIERARCHICAL_CHUNK_SYSTEM_PROMPT,
                        user_prompt=_build_hierarchical_chunk_prompt(chunk_id=chunk_id, rows=chunk_rows),
                        json_schema_hint=HIERARCHICAL_CHUNK_JSON_SCHEMA_HINT,
                    )
                    if not isinstance(summary, dict):
                        raise ValueError("chunk summary response is not a JSON object")
                    summary.setdefault("chunk_id", chunk_id)
                    summary.setdefault("time_span", _time_span_for_rows(chunk_rows))
                except Exception as exc:
                    err_text = str(exc)
                    chunk_errors.append({"chunk_id": chunk_id, "error": err_text})
                    summary = {
                        "chunk_id": chunk_id,
                        "time_span": _time_span_for_rows(chunk_rows),
                        "domain_summaries": [],
                        "error": err_text,
                    }
                chunk_summaries.append(summary)

            final_profile_error = ""
            try:
                final_profile = client.chat_json(
                    system_prompt=HIERARCHICAL_GLOBAL_SYSTEM_PROMPT,
                    user_prompt=_build_hierarchical_global_prompt(
                        user_id=first_task.user_id,
                        chunk_summaries=chunk_summaries,
                    ),
                    json_schema_hint=FINAL_PROFILE_JSON_SCHEMA_HINT,
                )
                if not isinstance(final_profile, dict):
                    raise ValueError("hierarchical final profile response is not a JSON object")
            except Exception as exc:
                final_profile_error = str(exc)
                final_profile = {
                    "user_id": first_task.user_id,
                    "active_domains": [],
                    "profile": {},
                    "error": final_profile_error,
                }

            artifact_fields = {
                "hierarchical_chunk_size": self.hierarchical_chunk_size,
                "hierarchical_max_posts": self.profile_method_max_posts,
                "hierarchical_chunk_summaries": chunk_summaries,
            }
            if chunk_errors:
                artifact_fields["hierarchical_chunk_errors"] = chunk_errors
            out.update(
                self._predictions_from_final_profile(
                    group_tasks=group_tasks,
                    final_profile=final_profile,
                    post_id_to_index=post_id_to_index,
                    model_name=spec.name,
                    artifact_fields=artifact_fields,
                    final_profile_error=final_profile_error,
                )
            )
        return out

    def _predict_extractive(self, *, tasks: list[DomainEvalTask], spec: ModelSpec) -> dict[str, dict[str, Any]]:
        if not tasks:
            return {}
        client = self._build_profile_client(spec=spec)
        grouped: dict[tuple[str, str, str], list[DomainEvalTask]] = {}
        for task in tasks:
            grouped.setdefault(_user_group_key(task), []).append(task)

        out: dict[str, dict[str, Any]] = {}
        total_groups = len(grouped)
        for group_idx, (_, group_tasks) in enumerate(grouped.items(), start=1):
            first_task = group_tasks[0]
            print(
                f"[{spec.name}:extractive] predicting user_group {group_idx}/{total_groups} "
                f"user_id={first_task.user_id} k={self.extractive_k}",
                flush=True,
            )
            bundle = self._build_context_bundle_for_task(task=first_task, spec=spec)
            timeline_rows = _timeline_rows_from_payload(
                getattr(bundle, "payload", {}) or {},
                max_posts=self.profile_method_max_posts,
                chronological=True,
            )
            post_id_to_index = _build_post_id_to_index_map(first_task.posts)

            selection_error = ""
            try:
                selection = client.chat_json(
                    system_prompt=EXTRACTIVE_SELECTION_SYSTEM_PROMPT,
                    user_prompt=_build_extractive_selection_prompt(
                        user_id=first_task.user_id,
                        timeline_rows=timeline_rows,
                        k=self.extractive_k,
                    ),
                    json_schema_hint=EXTRACTIVE_SELECTION_JSON_SCHEMA_HINT,
                )
                if not isinstance(selection, dict):
                    raise ValueError("extractive selection response is not a JSON object")
            except Exception as exc:
                selection_error = str(exc)
                selection = {
                    "user_id": first_task.user_id,
                    "selected_posts_by_domain": {domain: [] for domain in DOMAIN_NAMES},
                    "excluded_domains": [
                        {"domain": domain, "reason": f"selection_failed: {selection_error}"}
                        for domain in DOMAIN_NAMES
                    ],
                }

            selected_posts_with_content = _build_selected_posts_with_content(
                user_id=first_task.user_id,
                selection=selection,
                timeline_rows=timeline_rows,
                k=self.extractive_k,
            )
            final_profile_error = ""
            try:
                final_profile = client.chat_json(
                    system_prompt=EXTRACTIVE_ABSTRACTIVE_SYSTEM_PROMPT,
                    user_prompt=_build_extractive_abstractive_prompt(
                        selected_posts_with_content=selected_posts_with_content,
                    ),
                    json_schema_hint=FINAL_PROFILE_JSON_SCHEMA_HINT,
                )
                if not isinstance(final_profile, dict):
                    raise ValueError("extractive final profile response is not a JSON object")
            except Exception as exc:
                final_profile_error = str(exc)
                final_profile = {
                    "user_id": first_task.user_id,
                    "active_domains": [],
                    "profile": {},
                    "error": final_profile_error,
                }

            artifact_fields = {
                "extractive_k": self.extractive_k,
                "extractive_max_posts": self.profile_method_max_posts,
                "extractive_selection": selection,
                "extractive_selected_posts_with_content": selected_posts_with_content,
            }
            if selection_error:
                artifact_fields["extractive_selection_error"] = selection_error
            out.update(
                self._predictions_from_final_profile(
                    group_tasks=group_tasks,
                    final_profile=final_profile,
                    post_id_to_index=post_id_to_index,
                    model_name=spec.name,
                    artifact_fields=artifact_fields,
                    final_profile_error=final_profile_error,
                )
            )
        return out
