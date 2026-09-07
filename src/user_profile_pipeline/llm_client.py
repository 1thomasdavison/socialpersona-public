from __future__ import annotations

import json
import mimetypes
import os
import re
from typing import Any

from google.auth.transport.requests import Request as GoogleAuthRequest
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .google_auth_env import build_google_credentials, read_dotenv_var


FIXED_REPAIR_MODEL = "gemini-3-flash-preview"


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        provider: str = "openai_compatible",
        base_url: str,
        model: str,
        api_key_env: str,
        timeout_seconds: int = 60,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        repair_model: str = "gemini-3-flash-preview",
        repair_provider: str = "",
        repair_base_url: str = "",
        repair_api_key_env: str = "",
        api_key_override: str = "",
        repair_api_key_override: str = "",
    ) -> None:
        requested_provider = (provider or "openai_compatible").strip().lower()
        if requested_provider not in {"openai_compatible", "vertex"}:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = requested_provider
        if self.provider == "vertex":
            normalized_base_url = (base_url or "").strip()
            if (not normalized_base_url) or ("chatanywhere" in normalized_base_url.lower()):
                normalized_base_url = "https://aiplatform.googleapis.com/v1"
            self.base_url = normalized_base_url
        else:
            self.base_url = base_url
        self.model = model
        if self.provider == "vertex":
            normalized_repair = self._normalize_vertex_model_name((repair_model or "").strip())
            if normalized_repair and normalized_repair != FIXED_REPAIR_MODEL:
                raise ValueError(
                    f"Unsupported repair model: {repair_model}. "
                    f"repair_model must be fixed to {FIXED_REPAIR_MODEL}."
                )
            self.repair_model = FIXED_REPAIR_MODEL
        else:
            # For OpenAI-compatible providers (for example DashScope/Qwen), repair with
            # the same provider/model instead of forcing a Vertex model id.
            self.repair_model = (repair_model or model or "").strip() or model
        requested_repair_provider = (repair_provider or self.provider).strip().lower()
        if requested_repair_provider in {"chatanywhere", "bailian"}:
            requested_repair_provider = "openai_compatible"
        if requested_repair_provider == "vertex_batch":
            requested_repair_provider = "vertex"
        if requested_repair_provider not in {"openai_compatible", "vertex"}:
            raise ValueError(f"Unsupported repair provider: {repair_provider}")
        self.repair_provider = requested_repair_provider
        if self.repair_provider == "vertex":
            normalized_repair_model = self._normalize_vertex_model_name((self.repair_model or "").strip())
            if normalized_repair_model and normalized_repair_model != FIXED_REPAIR_MODEL:
                raise ValueError(
                    f"Unsupported repair model: {self.repair_model}. "
                    f"repair_model must be fixed to {FIXED_REPAIR_MODEL} for vertex."
                )
            self.repair_model = FIXED_REPAIR_MODEL
            normalized_repair_base_url = (repair_base_url or "").strip()
            if (not normalized_repair_base_url) or ("chatanywhere" in normalized_repair_base_url.lower()):
                normalized_repair_base_url = "https://aiplatform.googleapis.com/v1"
            self.repair_base_url = normalized_repair_base_url
            self.repair_api_key = (repair_api_key_override or "").strip() or self._resolve_api_key_optional(repair_api_key_env or api_key_env)
        else:
            self.repair_base_url = (repair_base_url or self.base_url or "").strip()
            self.repair_api_key = (repair_api_key_override or "").strip() or self._resolve_api_key(repair_api_key_env or api_key_env)
        self.vertex_access_token_env = "VERTEX_ACCESS_TOKEN"
        if self.provider == "vertex":
            self.api_key = (api_key_override or "").strip() or self._resolve_api_key_optional(api_key_env)
        else:
            self.api_key = (api_key_override or "").strip() or self._resolve_api_key(api_key_env)
        self.timeout_seconds: float | None = float(timeout_seconds) if float(timeout_seconds) > 0 else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = requests.Session()
        # Do not inherit shell-level network env overrides for model traffic.
        self._session.trust_env = False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
        json_schema_hint: str | None = None,
    ) -> dict[str, Any]:
        user_content: str | list[dict[str, Any]] = user_prompt
        if image_urls:
            user_content = [{"type": "text", "text": user_prompt}]
            for url in image_urls:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
        content = self._generate_text(
            model=self.model,
            system_prompt=system_prompt,
            user_content=user_content,
            force_json=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        parsed = self._parse_json_candidate(content)
        if parsed is not None and not self._parsed_json_is_suspicious(parsed):
            return parsed

        # Some OpenAI-compatible Gemini endpoints respond poorly to `response_format=json_object`.
        # Fall back to plain text generation plus local JSON repair when the parsed object looks truncated.
        fallback_content = self._generate_text(
            model=self.model,
            system_prompt=system_prompt,
            user_content=user_content,
            force_json=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        fallback = self._parse_json_candidate(fallback_content)
        if fallback is not None and not self._parsed_json_is_suspicious(fallback):
            return fallback

        repaired_text = self._repair_json_by_model(raw_text=content, json_schema_hint=json_schema_hint)
        repaired = self._parse_json_candidate(repaired_text)
        if repaired is not None and not self._parsed_json_is_suspicious(repaired):
            return repaired

        repaired_fallback_text = self._repair_json_by_model(
            raw_text=fallback_content,
            json_schema_hint=json_schema_hint,
        )
        repaired_fallback = self._parse_json_candidate(repaired_fallback_text)
        if repaired_fallback is not None:
            return repaired_fallback

        if parsed is not None:
            return parsed
        raise json.JSONDecodeError("Unable to recover a valid JSON object.", content, 0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str] | None = None,
    ) -> str:
        user_content: str | list[dict[str, Any]] = user_prompt
        if image_urls:
            user_content = [{"type": "text", "text": user_prompt}]
            for url in image_urls:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )
        return self._generate_text(
            model=self.model,
            system_prompt=system_prompt,
            user_content=user_content,
            force_json=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).strip()

    def _resolve_api_key(self, api_key_env: str) -> str:
        api_key = os.environ.get(api_key_env)
        if api_key:
            return api_key

        api_key = self._read_dotenv_var(api_key_env)
        if api_key:
            return api_key

        raise ValueError(f"Missing API key in environment variable: {api_key_env}")

    def _resolve_api_key_optional(self, api_key_env: str) -> str:
        api_key = os.environ.get(api_key_env)
        if api_key:
            return api_key
        api_key = self._read_dotenv_var(api_key_env)
        if api_key:
            return api_key
        return ""

    def _read_dotenv_var(self, key: str) -> str | None:
        return read_dotenv_var(key)

    def _generate_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        force_json: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if self.provider == "vertex":
            data = self._post_vertex_generate_content(
                model=model,
                system_prompt=system_prompt,
                user_content=user_content,
                force_json=force_json,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._extract_vertex_message_text(data)

        normalized_model = (model or "").strip().lower()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if normalized_model.startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["temperature"] = temperature
            payload["max_tokens"] = max_tokens
        # DashScope Qwen reasoning models consume ~100:1 reasoning:output tokens,
        # occasionally producing empty visible responses. Disable thinking mode.
        if "dashscope" in (self.base_url or "").lower():
            payload["enable_thinking"] = False
        if force_json and self._should_use_openai_response_format(model):
            payload["response_format"] = {"type": "json_object"}
        data = self._post_chat_completion(payload)
        return self._extract_message_text(data["choices"][0]["message"]["content"])

    def _should_use_openai_response_format(self, model: str) -> bool:
        if self.provider != "openai_compatible":
            return False
        base = (self.base_url or "").strip().lower()
        normalized_model = (model or "").strip().lower()
        if "chatanywhere" in base and normalized_model == "gemini-2.5-flash":
            return False
        return True

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._session.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self._request_timeout(),
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = ""
            try:
                body = str(response.text or "")[:1200]
            except Exception:
                body = ""
            raise requests.HTTPError(
                f"chat_completion_http_error status={response.status_code} body={body}",
                response=response,
            ) from exc
        return response.json()

    def _post_vertex_generate_content(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        force_json: bool,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        endpoint = self._build_vertex_endpoint(model)
        parts = self._to_vertex_parts(user_content)
        payload: dict[str, Any] = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                },
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if force_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self._resolve_vertex_access_token()}"
        response = self._session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self._request_timeout(),
        )
        response.raise_for_status()
        return response.json()

    def _request_timeout(self) -> float | None:
        try:
            value = float(self.timeout_seconds)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    def _build_vertex_endpoint(self, model: str) -> str:
        model_name = self._normalize_vertex_model_name(model)
        if "{model}" in self.base_url:
            return self.base_url.format(model=model_name)

        base = self.base_url.rstrip("/")
        if base.endswith(":generateContent"):
            return base
        if "/projects/" in base and "/locations/" in base:
            if base.endswith("/publishers/google/models"):
                return f"{base}/{model_name}:generateContent"
            return f"{base}/publishers/google/models/{model_name}:generateContent"
        if base.endswith("/publishers/google/models"):
            return f"{base}/{model_name}:generateContent"
        if base.endswith("/v1"):
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or str(
                self._read_dotenv_var("GOOGLE_CLOUD_PROJECT") or ""
            ).strip()
            location = os.environ.get("VERTEX_BATCH_LOCATION", "").strip() or str(
                self._read_dotenv_var("VERTEX_BATCH_LOCATION") or "global"
            ).strip()
            location = location or "global"
            if project_id:
                return (
                    f"{base}/projects/{project_id}/locations/{location}"
                    f"/publishers/google/models/{model_name}:generateContent"
                )
            return f"{base}/publishers/google/models/{model_name}:generateContent"
        return f"{base}/publishers/google/models/{model_name}:generateContent"

    def _normalize_vertex_model_name(self, model: str) -> str:
        cleaned = (model or "").strip()
        if "/models/" in cleaned:
            return cleaned.split("/models/", 1)[1].strip("/")
        if cleaned.startswith("models/"):
            return cleaned[len("models/"):].strip("/")
        if cleaned.startswith("google/"):
            return cleaned.split("/", 1)[1].strip("/")
        return cleaned

    def _resolve_vertex_access_token(self) -> str:
        token = os.environ.get(self.vertex_access_token_env, "").strip()
        if token:
            return token
        token = self._read_dotenv_var(self.vertex_access_token_env) or ""
        token = token.strip()
        if token:
            return token
        try:
            credentials, _ = build_google_credentials(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if not credentials.valid:
                credentials.refresh(GoogleAuthRequest())
            token = str(getattr(credentials, "token", "") or "").strip()
            if token:
                return token
        except Exception:
            pass
        try:
            token = (
                requests.get(  # nosec B113
                    "http://metadata/computeMetadata/v1/instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"},
                    timeout=2,
                )
                .json()
                .get("access_token", "")
                .strip()
            )
            if token:
                return token
        except Exception:
            pass
        raise ValueError(
            "Missing Vertex access token. Set VERTEX_ACCESS_TOKEN or configure "
            "GOOGLE_APPLICATION_CREDENTIALS for ADC."
        )

    def _to_vertex_parts(self, user_content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if isinstance(user_content, str):
            parts.append({"text": user_content})
            return parts

        for item in user_content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append({"text": str(item.get("text", ""))})
                continue
            if item.get("type") != "image_url":
                continue
            image_obj = item.get("image_url")
            if not isinstance(image_obj, dict):
                continue
            image_part = self._to_vertex_image_part(str(image_obj.get("url", "")).strip())
            if image_part is not None:
                parts.append(image_part)

        if not parts:
            parts.append({"text": ""})
        return parts

    def _to_vertex_image_part(self, image_url: str) -> dict[str, Any] | None:
        if not image_url:
            return None

        if image_url.startswith("data:"):
            header, sep, data = image_url.partition(",")
            if not sep or not data:
                return None
            if ";base64" not in header:
                return None
            mime = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
            return {
                "inlineData": {
                    "mimeType": mime,
                    "data": data,
                },
            }

        if image_url.startswith(("http://", "https://", "gs://")):
            mime, _ = mimetypes.guess_type(image_url)
            if not mime or not mime.startswith("image/"):
                mime = "image/jpeg"
            return {
                "fileData": {
                    "mimeType": mime,
                    "fileUri": image_url,
                },
            }

        return None

    def _extract_vertex_message_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not isinstance(candidates, list):
            candidates = []

        text_parts: list[str] = []
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
                    text_parts.append(str(part["text"]))

        if text_parts:
            return "\n".join(text_parts)
        raise ValueError(f"Vertex response has no text candidate: {data}")

    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>...</think> blocks that reasoning models leak into output."""
        stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return stripped.strip()

    def _extract_message_text(self, content: Any) -> str:
        raw: str
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            raw = "\n".join(text_parts)
        else:
            raw = str(content)
        # Strip <think>...</think> blocks that reasoning models (e.g. GPT-5.4)
        # may include in the visible output, tanking fluency scores.
        return self._strip_think_blocks(raw)

    def _parse_json_with_repair(self, raw_text: str) -> dict[str, Any] | None:
        candidates: list[str] = []
        stripped = raw_text.strip()
        if stripped:
            candidates.append(stripped)

        no_fence = self._strip_code_fences(stripped)
        if no_fence and no_fence not in candidates:
            candidates.append(no_fence)

        for text in list(candidates):
            extracted = self._extract_outer_json_object(text)
            if extracted and extracted not in candidates:
                candidates.append(extracted)
            balanced = self._close_truncated_json_object(text)
            if balanced and balanced not in candidates:
                candidates.append(balanced)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _parse_json_candidate(self, raw_text: str) -> dict[str, Any] | None:
        stripped = raw_text.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return self._parse_json_with_repair(raw_text)

    def _parsed_json_is_suspicious(self, parsed: dict[str, Any]) -> bool:
        if not isinstance(parsed, dict):
            return True
        if not parsed:
            return True
        if "schema_version" not in parsed:
            return False
        keys = {str(k) for k in parsed.keys()}
        if len(keys) <= 3:
            return True
        if keys.issubset(
            {
                "schema_version",
                "domain",
                "user_id",
                "stable_interests",
                "short_term_interests",
                "weak_or_uncertain_interests",
                "candidate_interests",
            }
        ) and len(keys) <= 4:
            return True
        return False

    def _strip_code_fences(self, text: str) -> str:
        if not text:
            return text
        trimmed = text.strip()
        if not trimmed.startswith("```"):
            return trimmed
        lines = trimmed.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return trimmed

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
        if end < 0:
            last = text.rfind("}")
            if last > start:
                return text[start:last + 1].strip()
            return None
        return text[start:end + 1].strip()

    def _close_truncated_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None

        candidate = text[start:].strip()
        if not candidate:
            return None

        stack: list[str] = []
        in_string = False
        escape = False
        for ch in candidate:
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
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in {"}", "]"} and stack and stack[-1] == ch:
                stack.pop()

        if escape:
            candidate = candidate[:-1]
        if in_string:
            candidate = candidate + '"'
        candidate = re.sub(r",\s*$", "", candidate.strip())
        while stack:
            closer = stack.pop()
            candidate = re.sub(r",\s*$", "", candidate.rstrip())
            candidate = f"{candidate}{closer}"
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        normalized = candidate.strip()
        if normalized == text.strip():
            return None
        return normalized

    def _repair_json_by_model(self, *, raw_text: str, json_schema_hint: str | None = None) -> str:
        repair_system_prompt = (
            "You repair malformed JSON. "
            "Return exactly one valid JSON object and no markdown."
        )
        schema_block = json_schema_hint.strip() if json_schema_hint else "{ \"type\": \"object\" }"
        repair_user_prompt = f"""The following model output should be a JSON object but is malformed.

Fix only JSON syntax issues, keep original fields/content as much as possible.
Do not add extra fields.
Do not remove fields unless absolutely necessary.
Ensure types match the schema hint.

Output ONLY valid JSON object text.

Schema hint:
{schema_block}

Malformed output:
{raw_text}
"""
        if (
            self.repair_provider == self.provider
            and self.repair_base_url == self.base_url
            and self.repair_api_key == self.api_key
        ):
            return self._generate_text(
                model=self.repair_model or self.model,
                system_prompt=repair_system_prompt,
                user_content=repair_user_prompt,
                force_json=True,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        repair_client = OpenAICompatibleChatClient(
            provider=self.repair_provider,
            base_url=self.repair_base_url,
            model=self.repair_model or self.model,
            api_key_env="__unused__",
            timeout_seconds=int(self.timeout_seconds or 60),
            temperature=0.0,
            max_tokens=self.max_tokens,
            repair_model=self.repair_model or self.model,
            api_key_override=self.repair_api_key,
            repair_api_key_override=self.repair_api_key,
        )
        return repair_client._generate_text(
            model=self.repair_model or self.model,
            system_prompt=repair_system_prompt,
            user_content=repair_user_prompt,
            force_json=True,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
