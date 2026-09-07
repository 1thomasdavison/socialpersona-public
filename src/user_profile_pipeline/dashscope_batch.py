from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import requests


DASHSCOPE_BATCH_DOC_URL = "https://www.alibabacloud.com/help/en/model-studio/batch-inference"
OPENAI_BATCH_DOC_URL = "https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai"

TERMINAL_BATCH_STATES = {
    "completed",
    "failed",
    "expired",
    "cancelling",
    "cancelled",
}

CN_TEXT_BATCH_MODELS = {
    "qwen3-max",
    "qwen3.7-max",
    "qwen-max",
    "qwen-max-latest",
    "qwen3.5-plus",
    "qwen-plus",
    "qwen-plus-latest",
    "qwen3.5-flash",
    "qwen-flash",
    "qwen-long-latest",
    "qwq-plus",
    "deepseek-r1",
    "deepseek-v3",
}

CN_MULTIMODAL_BATCH_MODELS = {
    "qwen3.5-plus",
    "qwen3.5-flash",
    "qwen3-vl-plus",
    "qwen3-vl-flash",
    "qwen-vl-max",
    "qwen-vl-max-latest",
    "qwen-vl-plus",
    "qwen-vl-plus-latest",
    "qwen-vl-ocr",
}

INTL_TEXT_BATCH_MODELS = {
    "qwen-max",
    "qwen-plus",
    "qwen-flash",
    "qwen-turbo",
}

MAX_BATCH_REQUEST_BYTES = 6 * 1024 * 1024


def normalize_dashscope_base_url(base_url: str) -> str:
    raw = (base_url or "").strip()
    if not raw:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    normalized = raw.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    return normalized


def infer_dashscope_region(base_url: str) -> str:
    normalized = normalize_dashscope_base_url(base_url).lower()
    if "dashscope.aliyuncs.com" in normalized and "intl" not in normalized:
        return "cn"
    return "intl"


def dashscope_batch_model_support_reason(
    *,
    model: str,
    has_images: bool,
    base_url: str,
) -> str | None:
    model_name = (model or "").strip()
    if not model_name:
        return "batch model name is empty"
    region = infer_dashscope_region(base_url)
    if region == "cn":
        supported = CN_MULTIMODAL_BATCH_MODELS if has_images else CN_TEXT_BATCH_MODELS
    else:
        supported = INTL_TEXT_BATCH_MODELS
    if model_name in supported:
        return None
    mode = "multimodal" if has_images else "text"
    region_label = "Chinese mainland" if region == "cn" else "international"
    return (
        f"DashScope batch does not officially list model={model_name!r} for {mode} in the {region_label} region. "
        f"See {DASHSCOPE_BATCH_DOC_URL}"
    )


def build_openai_batch_chat_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_urls: list[str] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1200,
    force_json: bool = False,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_content: str | list[dict[str, Any]] = user_prompt
    if image_urls:
        user_content = [{"type": "text", "text": user_prompt}]
        for url in image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": url}})

    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if force_json:
        body["response_format"] = {"type": "json_object"}
    if extra_body:
        body.update(extra_body)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def request_json_size_bytes(request_row: dict[str, Any]) -> int:
    return len(json.dumps(request_row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def extract_openai_chat_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"batch response missing choices: {payload}")
    message = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError(f"batch response missing message: {payload}")
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _looks_like_balance_or_quota_error(status_code: int, body_text: str) -> bool:
    if status_code == 402:
        return True
    text = (body_text or "").lower()
    tokens = [
        "insufficient balance",
        "insufficient_balance",
        "quota exceeded",
        "quota_exceeded",
        "account arrears",
        "account_arrears",
        "resource throttled",
        "resourcethrottled.insufficientbalance",
        "balance not enough",
    ]
    return any(token in text for token in tokens)


class DashScopeBatchInferenceClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        api_key_env_fallback: str = "",
        timeout_seconds: int = 120,
        request_retries: int = 5,
        retry_backoff_seconds: float = 2.0,
        max_retry_backoff_seconds: float = 30.0,
    ) -> None:
        self.base_url = normalize_dashscope_base_url(base_url)
        self.api_key_env = (api_key_env or "").strip()
        if not self.api_key_env:
            raise ValueError("api_key_env is required")
        self.api_key_env_fallback = (api_key_env_fallback or "").strip()
        self.timeout_seconds: float | None = float(timeout_seconds) if float(timeout_seconds) > 0 else None
        self.request_retries = max(1, int(request_retries))
        self.retry_backoff_seconds = max(0.2, float(retry_backoff_seconds))
        self.max_retry_backoff_seconds = max(1.0, float(max_retry_backoff_seconds))
        self._session = requests.Session()
        self._session.trust_env = False

    def upload_batch_file(self, *, local_path: str | Path) -> dict[str, Any]:
        src = Path(local_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"batch input file not found: {src}")
        with src.open("rb") as fh:
            return self._request_json(
                method="POST",
                url=f"{self.base_url}/files",
                action="upload_batch_file",
                data={"purpose": "batch"},
                files={"file": (src.name, fh, "application/octet-stream")},
            )

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_file_id": str(input_file_id).strip(),
            "endpoint": str(endpoint).strip() or "/v1/chat/completions",
            "completion_window": str(completion_window).strip() or "24h",
        }
        clean_metadata = {str(k): str(v) for k, v in (metadata or {}).items() if str(v).strip()}
        if clean_metadata:
            payload["metadata"] = clean_metadata
        return self._request_json(
            method="POST",
            url=f"{self.base_url}/batches",
            action="create_batch",
            json=payload,
        )

    def retrieve_batch(self, *, batch_id: str) -> dict[str, Any]:
        cleaned = str(batch_id or "").strip()
        if not cleaned:
            raise ValueError("batch_id is required")
        return self._request_json(
            method="GET",
            url=f"{self.base_url}/batches/{cleaned}",
            action="retrieve_batch",
        )

    def cancel_batch(self, *, batch_id: str) -> dict[str, Any]:
        cleaned = str(batch_id or "").strip()
        if not cleaned:
            raise ValueError("batch_id is required")
        return self._request_json(
            method="POST",
            url=f"{self.base_url}/batches/{cleaned}/cancel",
            action="cancel_batch",
        )

    def wait_batch(
        self,
        *,
        batch_id: str,
        poll_seconds: int = 120,
        max_wait_seconds: int = 86400,
    ) -> dict[str, Any]:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be > 0")
        deadline = time.time() + max_wait_seconds if max_wait_seconds > 0 else None
        last_status = ""
        while True:
            batch = self.retrieve_batch(batch_id=batch_id)
            status = str(batch.get("status") or "").strip().lower()
            if status != last_status:
                print(f"[dashscope_batch] batch={batch_id} status={status}", flush=True)
                last_status = status
            if status in TERMINAL_BATCH_STATES:
                return batch
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    f"DashScope batch did not finish within {max_wait_seconds}s: batch_id={batch_id} status={status}"
                )
            time.sleep(poll_seconds)

    def download_file_content(
        self,
        *,
        file_id: str,
        local_path: str | Path | None = None,
    ) -> str:
        cleaned = str(file_id or "").strip()
        if not cleaned:
            raise ValueError("file_id is required")
        response = self._request(
            method="GET",
            url=f"{self.base_url}/files/{cleaned}/content",
            action="download_file_content",
        )
        text = response.text
        if local_path is not None:
            path = Path(local_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def _headers(self, *, json_body: bool = True) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._resolve_api_key()}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _resolve_key_from_env(self, env_var: str) -> str:
        key = os.environ.get(env_var, "").strip()
        if key:
            return key
        dotenv_candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for path in dotenv_candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                for raw in path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key_name, value = line.split("=", 1)
                    if key_name.strip() != env_var:
                        continue
                    cleaned = value.strip().strip("'").strip('"')
                    if cleaned:
                        return cleaned
            except OSError:
                continue
        return ""

    def _resolve_api_key(self) -> str:
        key = self._resolve_key_from_env(self.api_key_env)
        if key:
            return key
        raise ValueError(f"Missing API key in environment variable: {self.api_key_env}")

    def _resolve_api_key_with_fallback(self, *, prefer_fallback: bool = False) -> str:
        primary = self.api_key_env
        secondary = self.api_key_env_fallback
        if prefer_fallback and secondary:
            key = self._resolve_key_from_env(secondary)
            if key:
                return key
        key = self._resolve_key_from_env(primary)
        if key:
            return key
        if secondary:
            key = self._resolve_key_from_env(secondary)
            if key:
                return key
        raise ValueError(f"Missing API key for {primary}" + (f" and fallback {secondary}" if secondary else ""))

    def _request_json(self, *, method: str, url: str, action: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method=method, url=url, action=action, **kwargs)
        try:
            return response.json()
        except Exception as exc:
            raise RuntimeError(
                f"{action} returned non-JSON response: status={response.status_code} body={response.text[:1200]}"
            ) from exc

    def _request(self, *, method: str, url: str, action: str, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        wants_json = ("json" in kwargs) and ("files" not in kwargs)
        if not headers:
            headers = self._headers(json_body=wants_json)
        elif "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self._resolve_api_key()}"
        if wants_json and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        switched_to_fallback = False
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                if not switched_to_fallback and self.api_key_env_fallback:
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        try:
                            body = (resp.text or "")[:2000]
                        except Exception:
                            body = ""
                        if _looks_like_balance_or_quota_error(resp.status_code, body):
                            fallback_key = self._resolve_key_from_env(self.api_key_env_fallback)
                            if fallback_key:
                                switched_to_fallback = True
                                headers["Authorization"] = f"Bearer {fallback_key}"
                                attempt = 0
                                continue
                if attempt >= self.request_retries:
                    body = ""
                    if getattr(exc, "response", None) is not None:
                        try:
                            body = str(exc.response.text or "")[:1200]
                        except Exception:
                            body = ""
                    raise RuntimeError(
                        f"{action} failed after {attempt} attempts: {exc} body={body}"
                    ) from exc
                sleep_for = min(
                    self.max_retry_backoff_seconds,
                    self.retry_backoff_seconds * math.pow(2.0, attempt - 1),
                )
                time.sleep(sleep_for)


def iter_jsonl_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    src = Path(path)
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        yield json.loads(line)
