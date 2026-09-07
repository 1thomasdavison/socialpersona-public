from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .gate import DomainConfig
from .schemas import DOMAIN_NAMES


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        example_path = config_path.with_name(f"{config_path.name}.example")
        hint = f" Copy {example_path} first." if example_path.is_file() else ""
        raise FileNotFoundError(f"YAML config not found: {config_path}.{hint}")

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a top-level mapping: {config_path}")
    return payload


def load_domain_configs(path: str | Path) -> list[DomainConfig]:
    """Load and validate narrow-gate domain cue configuration."""

    raw = load_yaml(path)
    items = raw.get("domains")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Domain config must contain a non-empty 'domains' list: {path}")

    configs: list[DomainConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Domain entry {index} must be a mapping: {path}")
        name = str(item.get("name") or "").strip()
        if name not in DOMAIN_NAMES:
            raise ValueError(f"Unknown domain {name!r} in {path}; expected one of {DOMAIN_NAMES}")
        if name in seen:
            raise ValueError(f"Duplicate domain {name!r} in {path}")
        seen.add(name)
        configs.append(
            DomainConfig(
                name=name,
                seed_keywords={
                    str(value).strip().lower()
                    for value in item.get("seed_keywords", [])
                    if str(value).strip()
                },
                seed_hashtags={
                    str(value).strip().lower().lstrip("#")
                    for value in item.get("seed_hashtags", [])
                    if str(value).strip().lstrip("#")
                },
            )
        )
    return configs
