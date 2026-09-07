from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .judge_prompt import (
    RUBRIC_JUDGE_SCHEMA_HINT,
    RUBRIC_JUDGE_SYSTEM_PROMPT,
    render_rubric_judge_user_prompt,
)


@dataclass
class JudgeConfig:
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_seconds: int = 120
    prompt_version: str = "rubric_v1"

    batch_mode: bool = False
    batch_model: str = ""
    vertex_project_id: str = ""
    vertex_location: str = ""
    vertex_bucket: str = ""
    vertex_access_token_env: str = "VERTEX_ACCESS_TOKEN"
    poll_seconds: int = 20
    max_wait_seconds: int = 14400
    output_dir: str = "outputs"


class RubricJudgeClient:
    def __init__(
        self,
        llm_client: Any | None,
        config: JudgeConfig,
        *,
        cache_root: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config
        self.cache_root = (cache_root or Path("outputs") / "judge_cache").resolve()

    def _slugify(self, value: str) -> str:
        lowered = (value or "").strip().lower()
        cleaned = re.sub(r"[^a-z0-9._-]+", "_", lowered)
        return cleaned.strip("_") or "item"

    def _cache_dir(self) -> Path:
        out = self.cache_root / self.config.model_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _cache_key(self, judge_input: dict[str, Any]) -> str:
        payload = {
            "task_id": judge_input.get("task_id"),
            "judge_model": self.config.model_name,
            "prompt_version": self.config.prompt_version,
            "gold": judge_input.get("gold") or {},
            "pred": judge_input.get("prediction") or {},
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _cache_path(self, judge_input: dict[str, Any]) -> Path:
        return self._cache_dir() / f"{self._cache_key(judge_input)}.json"

    def _normalize_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(5.0, score))

    def _normalize_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out

    def _coerce_judge_json(self, raw: dict[str, Any], *, task_id: str, status_correct: bool) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("judge returned non-object payload")

        anchor_correctness = self._normalize_score(raw.get("anchor_correctness"))
        anchor_coverage = self._normalize_score(raw.get("anchor_coverage"))
        evidence_grounding = self._normalize_score(raw.get("evidence_grounding"))
        summary_faithfulness = self._normalize_score(raw.get("summary_faithfulness"))

        total = raw.get("judge_total_score")
        if total is None:
            total_value = anchor_correctness + anchor_coverage + evidence_grounding + summary_faithfulness
        else:
            try:
                total_num = float(total)
            except (TypeError, ValueError):
                total_num = anchor_correctness + anchor_coverage + evidence_grounding + summary_faithfulness
            total_value = max(0.0, min(20.0, total_num * 4.0 if total_num <= 5.0 else total_num))

        return {
            "task_id": str(raw.get("task_id") or task_id),
            "status_correct": bool(raw.get("status_correct", status_correct)),
            "status_reason": str(raw.get("status_reason") or "").strip(),
            "anchor_correctness": anchor_correctness,
            "anchor_coverage": anchor_coverage,
            "evidence_grounding": evidence_grounding,
            "summary_faithfulness": summary_faithfulness,
            "unsupported_anchor_labels": self._normalize_list(raw.get("unsupported_anchor_labels")),
            "missed_gold_anchor_labels": self._normalize_list(raw.get("missed_gold_anchor_labels")),
            "unsupported_summary_claims": self._normalize_list(raw.get("unsupported_summary_claims")),
            "judge_total_score": float(total_value),
            "brief_rationale": str(raw.get("brief_rationale") or "").strip(),
        }

    def _error_payload(self, *, task_id: str, status_correct: bool, reason: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "status_correct": status_correct,
            "status_reason": reason,
            "anchor_correctness": 0.0,
            "anchor_coverage": 0.0,
            "evidence_grounding": 0.0,
            "summary_faithfulness": 0.0,
            "unsupported_anchor_labels": [],
            "missed_gold_anchor_labels": [],
            "unsupported_summary_claims": [],
            "judge_total_score": 0.0,
            "brief_rationale": reason,
        }

    def _extract_outer_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end >= start:
            return text[start:end + 1]
        return None

    def _parse_prediction_json(self, raw_text: str) -> dict[str, Any] | None:
        stripped = raw_text.strip()
        if not stripped:
            return None
        candidates = [stripped]
        extracted = self._extract_outer_json_object(stripped)
        if extracted and extracted not in candidates:
            candidates.append(extracted)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _extract_vertex_text(self, response_obj: dict[str, Any]) -> str:
        candidates = response_obj.get("candidates") or []
        texts: list[str] = []
        if not isinstance(candidates, list):
            return ""
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    texts.append(str(part["text"]))
        return "\n".join(texts).strip()

    def _write_cache(self, *, cache_key: str, payload: dict[str, Any]) -> None:
        cache_path = self._cache_dir() / f"{cache_key}.json"
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def prepare_batch(self, judge_inputs: list[dict[str, Any]]) -> None:
        if not self.config.batch_mode:
            return
        if not judge_inputs:
            return

        missing: list[tuple[str, dict[str, Any]]] = []
        for judge_input in judge_inputs:
            key = self._cache_key(judge_input)
            cache_path = self._cache_dir() / f"{key}.json"
            if cache_path.exists():
                continue
            missing.append((key, judge_input))

        if not missing:
            return

        project_id = str(self.config.vertex_project_id or "").strip()
        location = str(self.config.vertex_location or "global").strip() or "global"
        bucket = str(self.config.vertex_bucket or "").strip()
        if not project_id:
            raise ValueError("judge batch requires vertex_project_id")
        if not bucket:
            raise ValueError("judge batch requires vertex_bucket")

        from ...vertex_batch import VertexBatchInferenceClient

        model_name = str(self.config.batch_model or self.config.model_name).strip()
        client = VertexBatchInferenceClient(
            project_id=project_id,
            location=location,
            model=model_name,
            timeout_seconds=self.config.timeout_seconds,
            access_token_env=self.config.vertex_access_token_env,
        )

        run_tag = f"{self._slugify(self.config.model_name)}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = Path(self.config.output_dir).resolve() / "judge_vertex_batch_runs" / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        input_local = run_dir / "input.jsonl"
        ordered_keys: list[str] = []
        input_by_key: dict[str, dict[str, Any]] = {key: payload for key, payload in missing}

        with input_local.open("w", encoding="utf-8") as f:
            for key, judge_input in missing:
                user_prompt = render_rubric_judge_user_prompt(judge_input)
                request_line = {
                    "custom_id": key,
                    "request": {
                        "systemInstruction": {"role": "system", "parts": [{"text": RUBRIC_JUDGE_SYSTEM_PROMPT}]},
                        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                        "generationConfig": {
                            "temperature": self.config.temperature,
                            "responseMimeType": "application/json",
                            "maxOutputTokens": self.config.max_tokens,
                        },
                    },
                }
                f.write(json.dumps(request_line, ensure_ascii=False) + "\n")
                ordered_keys.append(key)

        gcs_prefix = f"judge_eval/{run_tag}"
        input_uri = f"gs://{bucket}/{gcs_prefix}/input.jsonl"
        output_prefix = f"gs://{bucket}/{gcs_prefix}/output"
        client.upload_local_file(local_path=input_local, gcs_uri=input_uri)
        job = client.submit_gcs_batch_job(
            input_gcs_uri=input_uri,
            output_gcs_uri_prefix=output_prefix,
            display_name=f"judge-{run_tag}",
        )
        job_name = str(job.get("name", "")).strip()
        if not job_name:
            raise RuntimeError(f"judge vertex submit response missing job name: {job}")

        job_final = client.wait_batch_job(
            job_name_or_id=job_name,
            poll_seconds=int(self.config.poll_seconds),
            max_wait_seconds=int(self.config.max_wait_seconds),
        )
        state = str(job_final.get("state", "")).upper()
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"judge vertex batch failed: {job_final}")

        output_uri_prefix = client.get_output_uri_prefix(job_final) or output_prefix
        output_local_dir = run_dir / "outputs"
        client.download_output_prefix(
            output_gcs_uri_prefix=output_uri_prefix,
            local_dir=output_local_dir,
            only_jsonl=True,
        )

        seen: set[str] = set()
        cursor = 0
        output_files = sorted(output_local_dir.rglob("*.jsonl"))
        for path in output_files:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    if cursor >= len(ordered_keys):
                        continue
                    key = ordered_keys[cursor]
                    cursor += 1
                    judge_input = input_by_key[key]
                    payload = self._error_payload(
                        task_id=str(judge_input.get("task_id") or ""),
                        status_correct=bool(judge_input.get("status_correct", False)),
                        reason="judge_batch_invalid_json_line",
                    )
                    self._write_cache(cache_key=key, payload=payload)
                    seen.add(key)
                    continue

                key = str(row.get("custom_id", "")).strip()
                if not key or key not in input_by_key:
                    if cursor >= len(ordered_keys):
                        continue
                    key = ordered_keys[cursor]
                    cursor += 1

                judge_input = input_by_key[key]
                status = str(row.get("status", "")).strip()
                response = row.get("response") or {}
                raw_text = self._extract_vertex_text(response if isinstance(response, dict) else {})
                parsed_raw = self._parse_prediction_json(raw_text)

                if status or parsed_raw is None:
                    reason = f"judge_batch_status_error: {status}" if status else "judge_batch_parse_failed"
                    payload = self._error_payload(
                        task_id=str(judge_input.get("task_id") or ""),
                        status_correct=bool(judge_input.get("status_correct", False)),
                        reason=reason,
                    )
                else:
                    payload = self._coerce_judge_json(
                        parsed_raw,
                        task_id=str(judge_input.get("task_id") or ""),
                        status_correct=bool(judge_input.get("status_correct", False)),
                    )

                self._write_cache(cache_key=key, payload=payload)
                seen.add(key)

        for key, judge_input in missing:
            if key in seen:
                continue
            payload = self._error_payload(
                task_id=str(judge_input.get("task_id") or ""),
                status_correct=bool(judge_input.get("status_correct", False)),
                reason="judge_batch_missing_prediction_row",
            )
            self._write_cache(cache_key=key, payload=payload)

    def judge_task(self, judge_input: dict[str, Any]) -> dict[str, Any]:
        cache_path = self._cache_path(judge_input)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                raise ValueError(f"judge cache invalid object: {cache_path}")
            return cached

        if self.config.batch_mode:
            raise ValueError("judge cache miss in batch_mode; call prepare_batch(...) first")

        if self.llm_client is None:
            raise ValueError("judge llm_client is not configured")

        task_id = str(judge_input.get("task_id") or "")
        status_correct = bool(judge_input.get("status_correct", False))
        raw = self.llm_client.chat_json(
            system_prompt=RUBRIC_JUDGE_SYSTEM_PROMPT,
            user_prompt=render_rubric_judge_user_prompt(judge_input),
            json_schema_hint=RUBRIC_JUDGE_SCHEMA_HINT,
        )
        parsed = self._coerce_judge_json(raw, task_id=task_id, status_correct=status_correct)
        cache_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        return parsed
