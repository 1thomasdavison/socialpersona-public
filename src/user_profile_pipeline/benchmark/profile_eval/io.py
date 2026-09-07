from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .specs import DEFAULT_MODEL_SPECS, ModelSpec


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def _env_or_dotenv(key: str, dotenv: dict[str, str], default: str = "") -> str:
    return os.environ.get(key, "").strip() or dotenv.get(key, "").strip() or default


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_posts_with_indices(path: Path, *, max_posts: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    posts_dir = path.resolve().parent
    for line_idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        row["_line_index"] = line_idx
        row["_posts_dir"] = str(posts_dir)
        rows.append(row)
    if max_posts and max_posts > 0:
        return rows[:max_posts]
    return rows


def _discover_posts_by_user(data_test_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not data_test_root.exists():
        return out
    for path in sorted(data_test_root.rglob("posts.jsonl")):
        dir_user_id = path.parent.name.strip()
        try:
            first = next((ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
            row = json.loads(first) if first else {}
        except (StopIteration, OSError, json.JSONDecodeError):
            row = {}
        user_id = str(row.get("user_id", "")).strip()
        candidates = []
        if user_id:
            candidates.append(user_id)
        if dir_user_id and dir_user_id not in candidates:
            candidates.append(dir_user_id)
        for candidate in candidates:
            if candidate and candidate not in out:
                out[candidate] = path
    return out


def _parse_posts_map(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in items:
        text = (raw or "").strip()
        if not text:
            continue
        if "=" in text:
            user_id, path = text.split("=", 1)
            uid = user_id.strip()
            p = Path(path.strip())
            if uid and p.exists():
                out[uid] = p
            continue
        p = Path(text)
        if not p.exists() or not p.is_file():
            continue
        try:
            first = next((ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
            if not first:
                continue
            uid = str(json.loads(first).get("user_id", "")).strip()
        except (StopIteration, OSError, json.JSONDecodeError):
            continue
        if uid:
            out[uid] = p
    return out


def _resolve_model_specs(model_names: list[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name in model_names:
        key = name.strip()
        if not key:
            continue
        if key not in DEFAULT_MODEL_SPECS:
            raise ValueError(f"Unknown model: {key}. Add it to DEFAULT_MODEL_SPECS before running the benchmark.")
        specs.append(DEFAULT_MODEL_SPECS[key])
    return specs
