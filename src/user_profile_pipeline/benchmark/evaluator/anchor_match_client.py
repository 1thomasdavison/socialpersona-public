from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .anchor_match_prompt import (
    ANCHOR_MATCH_SCHEMA_HINT,
    ANCHOR_MATCH_SYSTEM_PROMPT,
    render_anchor_match_user_prompt,
)

ANCHOR_MATCH_CACHE_FORMAT_VERSION = "anchor_match_cache_v2"


@dataclass
class AnchorMatchConfig:
    model_name: str = "gemini-3-flash"
    prompt_version: str = "anchor_match_v1"
    prefer_yes_no: bool = True
    force_llm: bool = False


class AnchorMatchJudgeClient:
    def __init__(
        self,
        llm_client: Any | None,
        config: AnchorMatchConfig,
        *,
        cache_root: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config
        self.cache_root = (cache_root or Path("outputs") / "anchor_match_cache").resolve()

    def uses_llm_for_all_pairs(self) -> bool:
        return bool(self.config.force_llm)

    def _slugify(self, value: str) -> str:
        lowered = (value or "").strip().lower()
        cleaned = re.sub(r"[^a-z0-9._-]+", "_", lowered)
        return cleaned.strip("_") or "item"

    def _cache_dir(self) -> Path:
        out = self.cache_root / self._slugify(self.config.model_name)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _normalize_label(self, text: str) -> str:
        lowered = str(text or "").strip().lower()
        compact = re.sub(r"[^a-z0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", compact).strip()

    def _cache_key(
        self,
        *,
        domain: str,
        pred_label: str,
        gold_label: str,
    ) -> str:
        left = self._normalize_label(pred_label)
        right = self._normalize_label(gold_label)
        ordered = sorted([left, right])
        payload = {
            "cache_format_version": ANCHOR_MATCH_CACHE_FORMAT_VERSION,
            "judge_model": self.config.model_name,
            "prompt_version": self.config.prompt_version,
            "domain": str(domain or "").strip().lower(),
            "label_a": ordered[0],
            "label_b": ordered[1],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _coerce_match_json(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("anchor matcher returned non-object payload")
        if "is_match" in raw:
            is_match = self._coerce_bool(raw.get("is_match"))
        elif "is_same_interest" in raw:
            is_match = self._coerce_bool(raw.get("is_same_interest"))
        elif "match" in raw:
            is_match = self._coerce_bool(raw.get("match"))
        elif "result" in raw:
            is_match = self._coerce_bool(raw.get("result"))
        elif "same" in raw:
            is_match = self._coerce_bool(raw.get("same"))
        else:
            is_match = False
        reason = str(
            raw.get("reason")
            or raw.get("rationale")
            or raw.get("explanation")
            or raw.get("analysis")
            or ""
        ).strip()
        return {"is_match": is_match, "reason": reason}

    def _error_payload(self, *, reason: str) -> dict[str, Any]:
        return {"is_match": False, "reason": reason, "transient_error": True}

    def _is_error_payload(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return True
        if bool(payload.get("transient_error")):
            return True
        reason = str(payload.get("reason") or "").strip().lower()
        return any(
            needle in reason
            for needle in [
                "anchor_match_error",
                "anchor_match_llm_not_configured",
                "403 client error",
                "forbidden",
            ]
        )

    def _parse_yes_no(self, text: str) -> bool | None:
        lowered = str(text or "").strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        bool_match = re.search(r"\bis[_ ]?match\b\s*[:=]\s*(true|false)", lowered)
        if bool_match:
            return bool_match.group(1) == "true"
        json_bool_match = re.search(r"\"is_match\"\s*:\s*(true|false)", lowered)
        if json_bool_match:
            return json_bool_match.group(1) == "true"
        if lowered.startswith("yes"):
            return True
        if lowered.startswith("no"):
            return False
        if " yes" in f" {lowered} " and " no" not in f" {lowered} ":
            return True
        if " no" in f" {lowered} " and " yes" not in f" {lowered} ":
            return False
        return None

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        parsed = self._parse_yes_no(str(value or ""))
        if parsed is not None:
            return parsed
        return False

    def _fallback_yes_no_call(
        self,
        *,
        task_id: str,
        domain: str,
        domain_definition: str,
        pred_label: str,
        gold_label: str,
    ) -> dict[str, Any] | None:
        if self.llm_client is None or not hasattr(self.llm_client, "_generate_text"):
            return None
        try:
            user_prompt = (
                f"Task ID: {task_id}\n\n"
                f"Domain: {domain}\n"
                f"Domain definition: {domain_definition}\n\n"
                f"Gold anchor label: {gold_label}\n"
                f"Predicted anchor label: {pred_label}\n\n"
                "Question: Do these two labels describe the same core user interest in this domain?\n"
                "Answer exactly YES or NO."
            )
            text = self.llm_client._generate_text(
                model=self.llm_client.model,
                system_prompt=(
                    "Decide if the two anchor labels are the same core interest. "
                    "Answer exactly YES or NO. Do not output markdown or code fences."
                ),
                user_content=user_prompt,
                force_json=False,
                temperature=0.0,
                max_tokens=256,
            )
            parsed = self._parse_yes_no(text)
            if parsed is None:
                return None
            return {"is_match": bool(parsed), "reason": f"fallback_yes_no:{text.strip()[:80]}"}
        except Exception:
            return None

    def _quick_exact_match(self, *, pred_label: str, gold_label: str) -> bool:
        pred_norm = self._normalize_label(pred_label)
        gold_norm = self._normalize_label(gold_label)
        if not pred_norm or not gold_norm:
            return False
        return pred_norm == gold_norm

    def judge_pair(
        self,
        *,
        task_id: str,
        domain: str,
        domain_definition: str,
        pred_label: str,
        gold_label: str,
    ) -> dict[str, Any]:
        if (not self.uses_llm_for_all_pairs()) and self._quick_exact_match(
            pred_label=pred_label,
            gold_label=gold_label,
        ):
            return {"is_match": True, "reason": "normalized_exact_match"}

        cache_key = self._cache_key(
            domain=domain,
            pred_label=pred_label,
            gold_label=gold_label,
        )
        cache_path = self._cache_dir() / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                raise ValueError(f"anchor match cache invalid object: {cache_path}")
            if not self._is_error_payload(cached):
                return self._coerce_match_json(cached)

        if self.llm_client is None:
            return self._error_payload(reason="anchor_match_llm_not_configured")

        if bool(self.config.prefer_yes_no):
            yes_no_payload = self._fallback_yes_no_call(
                task_id=task_id,
                domain=domain,
                domain_definition=domain_definition,
                pred_label=pred_label,
                gold_label=gold_label,
            )
            if yes_no_payload is not None:
                cache_path.write_text(json.dumps(yes_no_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return yes_no_payload

        try:
            raw = self.llm_client.chat_json(
                system_prompt=ANCHOR_MATCH_SYSTEM_PROMPT,
                user_prompt=render_anchor_match_user_prompt(
                    task_id=task_id,
                    domain=domain,
                    domain_definition=domain_definition,
                    gold_label=gold_label,
                    pred_label=pred_label,
                ),
                json_schema_hint=ANCHOR_MATCH_SCHEMA_HINT,
            )
            payload = self._coerce_match_json(raw)
        except Exception as exc:
            if bool(self.config.prefer_yes_no):
                fallback_payload = self._fallback_yes_no_call(
                    task_id=task_id,
                    domain=domain,
                    domain_definition=domain_definition,
                    pred_label=pred_label,
                    gold_label=gold_label,
                )
                if fallback_payload is not None:
                    cache_path.write_text(
                        json.dumps(fallback_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    return fallback_payload

            message = str(exc)
            last_attempt = getattr(exc, "last_attempt", None)
            if last_attempt is not None and hasattr(last_attempt, "exception"):
                inner = last_attempt.exception()
                if inner is not None:
                    message = f"{message}; last_exception={inner}"
            payload = self._error_payload(reason=f"anchor_match_error: {message}")
        if not self._is_error_payload(payload):
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def is_same_interest(
        self,
        *,
        task_id: str,
        domain: str,
        domain_definition: str,
        pred_label: str,
        gold_label: str,
    ) -> bool:
        judged = self.judge_pair(
            task_id=task_id,
            domain=domain,
            domain_definition=domain_definition,
            pred_label=pred_label,
            gold_label=gold_label,
        )
        return bool(judged.get("is_match", False))
