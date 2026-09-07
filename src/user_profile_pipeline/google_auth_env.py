from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from google.auth import default as google_auth_default
from google.oauth2 import service_account


def _dotenv_candidates() -> list[Path]:
    return [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]


def read_dotenv_var(key: str) -> str | None:
    cleaned_key = str(key or "").strip()
    if not cleaned_key:
        return None
    checked: set[Path] = set()
    for path in _dotenv_candidates():
        resolved = path.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if not path.exists() or not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() != cleaned_key:
                    continue
                text = value.strip()
                if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
                    text = text[1:-1]
                return text or None
        except OSError:
            continue
    return None


def ensure_google_auth_env_from_dotenv() -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"):
        if os.environ.get(key, "").strip():
            continue
        value = (read_dotenv_var(key) or "").strip()
        if not value:
            continue
        os.environ[key] = value
        resolved[key] = value
    return resolved


def build_google_credentials(*, scopes: Iterable[str]) -> tuple[Any, str]:
    ensure_google_auth_env_from_dotenv()
    scope_list = [str(scope).strip() for scope in scopes if str(scope).strip()]
    key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if key_file:
        key_path = Path(key_file).expanduser()
        if key_path.exists() and key_path.is_file():
            credentials = service_account.Credentials.from_service_account_file(
                str(key_path),
                scopes=scope_list or None,
            )
            project_id = (
                os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
                or str(getattr(credentials, "project_id", "") or "").strip()
            )
            if project_id and not os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip():
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            return credentials, project_id

    credentials, project_id = google_auth_default(scopes=scope_list or None)
    cleaned_project_id = str(project_id or "").strip()
    if cleaned_project_id and not os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip():
        os.environ["GOOGLE_CLOUD_PROJECT"] = cleaned_project_id
    return credentials, cleaned_project_id
