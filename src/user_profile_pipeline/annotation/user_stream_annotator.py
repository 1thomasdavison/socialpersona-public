from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import re
import threading
import webbrowser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


URL_RE = re.compile(r"(https?://[^\s<]+)")
DEFAULT_MAX_POSTS = 20


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_default_output_path(dataset_root: Path) -> Path:
    dataset_name = dataset_root.name.strip() or "dataset"
    return Path("output") / "annotation_sessions" / f"{dataset_name}_review.json"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _query_int(params: dict[str, list[str]], key: str) -> int | None:
    values = params.get(key)
    if not values:
        return None
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return None


def _build_user_path(user_index: int, limit_value: str, media_mode: str) -> str:
    return (
        f"/user/{user_index}"
        f"?limit={quote(limit_value, safe='')}"
        f"&media={quote(media_mode, safe='')}"
    )


def _external_location(handler: BaseHTTPRequestHandler, location: str) -> str:
    if location.startswith(("http://", "https://")):
        return location
    host = handler.headers.get("X-Forwarded-Host") or handler.headers.get("Host")
    proto = handler.headers.get("X-Forwarded-Proto") or "http"
    if not host:
        return location
    return f"{proto}://{host}{location}"


def _display_datetime(value: str | None) -> str:
    if not value:
        return "unknown"
    return value.replace("T", " ").replace(".000Z", "Z")


def _derive_username(user_id: str, metadata: dict[str, Any]) -> str:
    username = str(metadata.get("username") or "").strip()
    if username:
        return username
    return user_id[2:] if user_id.startswith("x_") else user_id


def _linkify_text(text: str) -> str:
    escaped = html.escape(text)
    linked = URL_RE.sub(
        lambda match: (
            f'<a href="{html.escape(match.group(1), quote=True)}" target="_blank" rel="noreferrer">'
            f"{html.escape(match.group(1))}</a>"
        ),
        escaped,
    )
    return linked.replace("\n", "<br>")


@dataclass
class UserTimeline:
    user_id: str
    username: str
    user_dir: Path
    metadata: dict[str, Any]
    posts: list[dict[str, Any]]

    @property
    def post_count(self) -> int:
        return len(self.posts)

    @property
    def source_url(self) -> str | None:
        value = str(self.metadata.get("user_url") or "").strip()
        return value or None

    def media_summary(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for post in self.posts:
            for media in post.get("media", []) or []:
                media_type = str(media.get("media_type") or "unknown").strip() or "unknown"
                counts[media_type] += 1
        return counts

    def language_summary(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for post in self.posts:
            language = str(post.get("language") or "unknown").strip() or "unknown"
            counts[language] += 1
        return counts


def load_user_timelines(dataset_root: Path) -> list[UserTimeline]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    users: list[UserTimeline] = []
    for user_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        posts_path = user_dir / "posts.jsonl"
        if not posts_path.exists():
            continue

        metadata_path = user_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        posts: list[dict[str, Any]] = []
        with posts_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {posts_path} at line {line_number}: {exc}"
                    ) from exc
                posts.append(record)

        user_id = str(metadata.get("username") or "").strip()
        if user_id:
            normalized_user_id = f"x_{user_id}" if not user_dir.name.startswith("x_") else user_dir.name
        else:
            normalized_user_id = user_dir.name

        users.append(
            UserTimeline(
                user_id=normalized_user_id,
                username=_derive_username(normalized_user_id, metadata),
                user_dir=user_dir,
                metadata=metadata,
                posts=posts,
            )
        )

    if not users:
        raise ValueError(f"No users with posts.jsonl found under {dataset_root}")
    return users


@dataclass
class AnnotationSession:
    dataset_root: str
    created_at: str
    updated_at: str
    current_index: int
    annotations: dict[str, dict[str, Any]]

    @classmethod
    def create(cls, dataset_root: Path) -> "AnnotationSession":
        timestamp = utc_now_iso()
        return cls(
            dataset_root=str(dataset_root),
            created_at=timestamp,
            updated_at=timestamp,
            current_index=0,
            annotations={},
        )

    @classmethod
    def load(cls, path: Path, dataset_root: Path) -> "AnnotationSession":
        if not path.exists():
            return cls.create(dataset_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            dataset_root=str(payload.get("dataset_root") or dataset_root),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            current_index=_safe_int(payload.get("current_index"), 0),
            annotations=dict(payload.get("annotations") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_index": self.current_index,
            "annotations": self.annotations,
        }


class AnnotationStore:
    def __init__(self, path: Path, dataset_root: Path) -> None:
        self.path = path.expanduser().resolve()
        self.csv_path = self.path.with_suffix(".csv")
        self.dataset_root = dataset_root.expanduser().resolve()
        self.lock = threading.Lock()
        self.session = AnnotationSession.load(self.path, self.dataset_root)

    def save(self) -> None:
        ensure_parent_dir(self.path)
        self.session.updated_at = utc_now_iso()
        self.path.write_text(
            json.dumps(self.session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def export_csv(self, users: list[UserTimeline]) -> None:
        ensure_parent_dir(self.csv_path)
        user_map = {user.user_id: user for user in users}
        rows: list[dict[str, Any]] = []
        for user_id, annotation in sorted(
            self.session.annotations.items(),
            key=lambda item: _safe_int(item[1].get("user_index"), 10**9),
        ):
            user = user_map.get(user_id)
            rows.append(
                {
                    "user_id": user_id,
                    "username": user.username if user else annotation.get("username", ""),
                    "user_index": annotation.get("user_index", ""),
                    "label": annotation.get("label", ""),
                    "notes": annotation.get("notes", ""),
                    "annotated_at": annotation.get("annotated_at", ""),
                    "post_count": user.post_count if user else annotation.get("post_count", ""),
                }
            )

        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "user_id",
                    "username",
                    "user_index",
                    "label",
                    "notes",
                    "annotated_at",
                    "post_count",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def annotate(
        self,
        *,
        user: UserTimeline,
        user_index: int,
        label: str,
        notes: str,
        current_index: int,
        users: list[UserTimeline],
    ) -> None:
        cleaned_notes = notes.strip()
        with self.lock:
            self.session.current_index = max(0, current_index)
            self.session.annotations[user.user_id] = {
                "user_id": user.user_id,
                "username": user.username,
                "user_index": user_index,
                "label": label,
                "notes": cleaned_notes,
                "post_count": user.post_count,
                "annotated_at": utc_now_iso(),
            }
            self.save()
            self.export_csv(users)


def _find_next_unreviewed(users: list[UserTimeline], annotations: dict[str, dict[str, Any]], start: int) -> int:
    if not users:
        return 0
    for index in range(max(0, start), len(users)):
        if users[index].user_id not in annotations:
            return index
    for index in range(0, max(0, start)):
        if users[index].user_id not in annotations:
            return index
    return min(max(0, start), len(users) - 1)


def _asset_url(user: UserTimeline, storage_uri: str) -> str:
    return f"/asset/{quote(user.user_id)}/{quote(storage_uri, safe='/')}"


def _render_media_items(user: UserTimeline, post: dict[str, Any]) -> str:
    fragments: list[str] = []
    for media in post.get("media", []) or []:
        media_type = str(media.get("media_type") or "unknown").strip()
        storage_uri = str(media.get("storage_uri") or "").strip()
        source_url = str(media.get("source_url") or "").strip()
        alt_text = str(media.get("alt_text") or "").strip()
        src = _asset_url(user, storage_uri) if storage_uri else source_url
        availability = "local file" if storage_uri else "remote only"
        if not src:
            continue
        alt_attr = html.escape(alt_text or media_type or "media", quote=True)
        if media_type == "video":
            fragments.append(
                f"""
                <div class="media-card">
                  <video controls preload="none" src="{html.escape(src, quote=True)}"></video>
                  <div class="media-meta">
                    <span>{html.escape(media_type)}</span>
                    <span>{html.escape(availability)}</span>
                    <a href="{html.escape(src, quote=True)}" target="_blank" rel="noreferrer">open source</a>
                  </div>
                </div>
                """
            )
        else:
            fragments.append(
                f"""
                <div class="media-card">
                  <a href="{html.escape(src, quote=True)}" target="_blank" rel="noreferrer">
                    <img src="{html.escape(src, quote=True)}" alt="{alt_attr}" loading="lazy" decoding="async" fetchpriority="low">
                  </a>
                  <div class="media-meta">
                    <span>{html.escape(media_type or "image")}</span>
                    <span>{html.escape(availability)}</span>
                    <a href="{html.escape(src, quote=True)}" target="_blank" rel="noreferrer">open source</a>
                  </div>
                </div>
                """
            )
    return "".join(fragments)


def _render_deferred_media(user: UserTimeline, post: dict[str, Any], post_key: str) -> str:
    media_items = post.get("media", []) or []
    if not media_items:
        return ""
    media_html = _render_media_items(user, post)
    template_id = f"media-template-{post_key}"
    target_id = f"media-target-{post_key}"
    return f"""
      <div class="media-deferred">
        <button type="button" class="secondary media-toggle" data-template-id="{template_id}" data-target-id="{target_id}">
          Load media ({len(media_items)})
        </button>
        <div id="{target_id}"></div>
        <template id="{template_id}">
          <div class="media-grid">{media_html}</div>
        </template>
      </div>
    """


def _render_post_card(user: UserTimeline, post: dict[str, Any], index: int, media_mode: str) -> str:
    text = str(post.get("text") or "").strip()
    created_at = _display_datetime(post.get("created_at"))
    post_type = str(post.get("post_type") or "unknown").strip()
    language = str(post.get("language") or "unknown").strip()
    post_id = str(post.get("post_id") or "").strip()
    urls = post.get("urls", []) or []
    url_links = " ".join(
        f'<a href="{html.escape(str(url), quote=True)}" target="_blank" rel="noreferrer">{html.escape(str(url))}</a>'
        for url in urls
        if str(url).strip()
    )
    post_key = re.sub(r"[^a-zA-Z0-9_-]", "_", post_id or f"post-{index}")
    inline_media_html = _render_media_items(user, post)
    media_section = ""
    if inline_media_html:
        if media_mode == "on":
            media_section = "<div class='media-grid'>" + inline_media_html + "</div>"
        else:
            media_section = _render_deferred_media(user, post, post_key)
    return f"""
    <article class="post-card">
      <div class="post-head">
        <div class="post-index">#{index + 1}</div>
        <div class="post-meta">
          <span>{html.escape(created_at)}</span>
          <span>{html.escape(post_type)}</span>
          <span>{html.escape(language)}</span>
          <span>{html.escape(post_id)}</span>
        </div>
      </div>
      <div class="post-text">{_linkify_text(text)}</div>
      {media_section}
      {f"<div class='post-links'>{url_links}</div>" if url_links else ""}
    </article>
    """


def _render_user_page(
    *,
    users: list[UserTimeline],
    store: AnnotationStore,
    user_index: int,
    max_posts: int | None,
    current_limit: int | None,
    show_all_posts: bool,
    media_mode: str,
) -> str:
    user_index = min(max(user_index, 0), len(users) - 1)
    user = users[user_index]
    saved = store.session.annotations.get(user.user_id, {})
    total_users = len(users)
    annotations = store.session.annotations
    labeled_count = len([1 for row in annotations.values() if row.get("label") in {"qualified", "rejected", "skip"}])
    qualified_count = len([1 for row in annotations.values() if row.get("label") == "qualified"])
    rejected_count = len([1 for row in annotations.values() if row.get("label") == "rejected"])
    skip_count = len([1 for row in annotations.values() if row.get("label") == "skip"])
    prev_index = max(0, user_index - 1)
    next_index = min(total_users - 1, user_index + 1)
    next_unreviewed_index = _find_next_unreviewed(users, annotations, user_index + 1)
    media_counts = user.media_summary()
    language_counts = user.language_summary()
    local_media_count = sum(
        1 for post in user.posts for media in (post.get("media", []) or []) if media.get("storage_uri")
    )
    remote_only_media_count = sum(
        1
        for post in user.posts
        for media in (post.get("media", []) or [])
        if (not media.get("storage_uri")) and media.get("source_url")
    )
    note_value = html.escape(str(saved.get("notes") or ""))
    current_label = str(saved.get("label") or "")
    effective_limit = None if show_all_posts else (current_limit if current_limit is not None else max_posts)
    posts = user.posts[:effective_limit] if effective_limit and effective_limit > 0 else user.posts
    truncated = len(posts) < len(user.posts)

    user_url_html = (
        f'<a href="{html.escape(user.source_url, quote=True)}" target="_blank" rel="noreferrer">{html.escape(user.source_url)}</a>'
        if user.source_url
        else "<span class='muted'>not available</span>"
    )
    metadata_block = json.dumps(user.metadata or {}, ensure_ascii=False, indent=2)
    normalized_media_mode = "on" if media_mode == "on" else "deferred"
    post_cards = "".join(
        _render_post_card(user, post, idx, normalized_media_mode) for idx, post in enumerate(posts)
    )
    progress = (labeled_count / total_users * 100.0) if total_users else 0.0
    limit_value = "all" if effective_limit is None else str(effective_limit)
    prev_url = _build_user_path(prev_index, limit_value, normalized_media_mode)
    next_url = _build_user_path(next_index, limit_value, normalized_media_mode)
    next_unreviewed_url = _build_user_path(next_unreviewed_index, limit_value, normalized_media_mode)
    view_options = [
        ("20", 20),
        ("40", 40),
        ("100", 100),
    ]
    view_links = " ".join(
        f'<a class="view-link{" active" if effective_limit == option_limit else ""}" href="{_build_user_path(user_index, str(option_limit), normalized_media_mode)}">{label}</a>'
        for label, option_limit in view_options
    )
    view_links += (
        f' <a class="view-link{" active" if effective_limit is None else ""}" '
        f'href="{_build_user_path(user_index, "all", normalized_media_mode)}">all</a>'
    )
    media_links = " ".join(
        [
            f'<a class="view-link{" active" if normalized_media_mode == "deferred" else ""}" href="{_build_user_path(user_index, limit_value, "deferred")}">on demand</a>',
            f'<a class="view-link{" active" if normalized_media_mode == "on" else ""}" href="{_build_user_path(user_index, limit_value, "on")}">auto load</a>',
        ]
    )

    status_label = {
        "qualified": "\u5408\u683c",
        "rejected": "\u4e0d\u5408\u683c",
        "skip": "\u8df3\u8fc7",
    }.get(current_label, "\u672a\u6807\u6ce8")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>User Annotation</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --panel: #fffaf3;
      --ink: #201a16;
      --muted: #75685f;
      --accent: #c76a2a;
      --accent-strong: #9f4f18;
      --line: #e4d7c9;
      --good: #2f7d4b;
      --bad: #ab3b3b;
      --skip: #8a6a18;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff5de 0, transparent 28%),
        linear-gradient(180deg, #f8f2ea 0%, var(--bg) 100%);
    }}
    a {{ color: var(--accent-strong); }}
    .layout {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 20px;
    }}
    .panel {{
      background: rgba(255, 250, 243, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(74, 49, 31, 0.08);
    }}
    .sidebar {{
      position: sticky;
      top: 20px;
      align-self: start;
      padding: 18px;
    }}
    .content {{
      padding: 18px;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .topline {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 14px;
      background: #f0e3d7;
    }}
    .muted {{ color: var(--muted); }}
    .progress {{
      height: 12px;
      border-radius: 999px;
      background: #ecdfd1;
      overflow: hidden;
      margin-top: 8px;
    }}
    .progress > div {{
      height: 100%;
      width: {progress:.2f}%;
      background: linear-gradient(90deg, #d28a45, #b55921);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .stat {{
      padding: 12px;
      border-radius: 14px;
      background: #fff;
      border: 1px solid var(--line);
    }}
    .stat strong {{
      display: block;
      font-size: 24px;
      margin-bottom: 4px;
    }}
    .actions {{
      display: grid;
      gap: 10px;
      margin-top: 18px;
    }}
    textarea {{
      width: 100%;
      min-height: 110px;
      resize: vertical;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    button, .nav-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }}
    button:hover, .nav-link:hover {{
      transform: translateY(-1px);
      opacity: 0.96;
    }}
    .qualified {{ background: var(--good); color: #fff; }}
    .rejected {{ background: var(--bad); color: #fff; }}
    .skip {{ background: var(--skip); color: #fff; }}
    .secondary {{
      background: #fff;
      color: var(--ink);
      border: 1px solid var(--line);
    }}
    .nav-group {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .jump-form {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      margin-top: 14px;
    }}
    .jump-form input {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    .summary-list {{
      display: grid;
      gap: 8px;
      margin: 12px 0 0;
      font-size: 14px;
    }}
    .summary-item {{
      padding: 10px 12px;
      border-radius: 12px;
      background: #fff;
      border: 1px solid var(--line);
    }}
    .hero {{
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .hero-card {{
      padding: 14px;
      border-radius: 14px;
      background: #fff;
      border: 1px solid var(--line);
    }}
    .stream {{
      display: grid;
      gap: 16px;
    }}
    .post-card {{
      padding: 16px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid var(--line);
    }}
    .post-head {{
      display: flex;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 10px;
    }}
    .post-index {{
      min-width: 48px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f6eadf;
      font-weight: 700;
      text-align: center;
    }}
    .post-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .post-text {{
      font-size: 16px;
      line-height: 1.6;
      white-space: normal;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .media-card {{
      overflow: hidden;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fbf8f3;
    }}
    .media-card img, .media-card video {{
      display: block;
      width: 100%;
      max-height: 360px;
      object-fit: cover;
      background: #eaded1;
    }}
    .media-meta {{
      padding: 8px 10px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .media-meta span:first-child {{
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-weight: 700;
    }}
    .post-links {{
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 13px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      padding: 12px;
      border-radius: 12px;
      background: #fff;
      border: 1px solid var(--line);
      max-height: 280px;
      overflow: auto;
    }}
    .status-chip {{
      display: inline-flex;
      padding: 6px 10px;
      border-radius: 999px;
      background: #fff;
      border: 1px solid var(--line);
      margin-top: 10px;
      font-size: 14px;
    }}
    details {{
      margin-top: 16px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    .view-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .view-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 10px;
      border-radius: 999px;
      background: #fff;
      border: 1px solid var(--line);
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
    }}
    .view-link.active {{
      background: #f0e3d7;
    }}
    .media-deferred {{
      margin-top: 14px;
      display: grid;
      gap: 10px;
    }}
    .media-toggle {{
      width: auto;
      justify-self: start;
    }}
    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
      }}
      .hero-grid, .stats, .nav-group {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel sidebar">
      <div class="topline">
        <h2>Annotation</h2>
        <div class="badge">{user_index + 1} / {total_users}</div>
      </div>
      <div class="muted">Current user</div>
      <h1>@{html.escape(user.username)}</h1>
      <div class="status-chip">status: {status_label}</div>
      <div class="summary-list">
        <div class="summary-item"><strong>User ID</strong><br>{html.escape(user.user_id)}</div>
        <div class="summary-item"><strong>X profile</strong><br>{user_url_html}</div>
        <div class="summary-item"><strong>Posts</strong><br>{user.post_count}</div>
      </div>

      <div style="margin-top: 18px;">
        <div><strong>Progress</strong></div>
        <div class="muted">{labeled_count} reviewed, {total_users - labeled_count} remaining</div>
        <div class="progress"><div></div></div>
      </div>

      <div class="stats">
        <div class="stat"><strong>{qualified_count}</strong><span class="muted">qualified</span></div>
        <div class="stat"><strong>{rejected_count}</strong><span class="muted">rejected</span></div>
        <div class="stat"><strong>{skip_count}</strong><span class="muted">skipped</span></div>
        <div class="stat"><strong>{progress:.1f}%</strong><span class="muted">complete</span></div>
      </div>

      <form class="actions" method="post" action="/annotate">
        <input type="hidden" name="user_index" value="{user_index}">
        <input type="hidden" name="limit" value="{limit_value}">
        <input type="hidden" name="media_mode" value="{normalized_media_mode}">
        <textarea name="notes" placeholder="review notes...">{note_value}</textarea>
        <button type="submit" class="qualified" name="label" value="qualified">Qualified [Q]</button>
        <button type="submit" class="rejected" name="label" value="rejected">Rejected [W]</button>
        <button type="submit" class="skip" name="label" value="skip">Skip [S]</button>
      </form>

      <div class="nav-group">
        <a class="nav-link secondary" href="{prev_url}">Prev</a>
        <a class="nav-link secondary" href="{next_url}">Next</a>
        <a class="nav-link secondary" href="{next_unreviewed_url}">Next Unreviewed</a>
      </div>
      <form class="jump-form" id="jump-form">
        <input type="hidden" name="limit" value="{limit_value}">
        <input type="hidden" name="media_mode" value="{normalized_media_mode}">
        <input type="number" name="index" min="1" max="{total_users}" value="{user_index + 1}" aria-label="User index">
        <button type="submit" class="secondary">Go</button>
      </form>

      <div style="margin-top: 18px;">
        <div><strong>Tips</strong></div>
        <div class="muted">Q = qualified, W = rejected, S = skip, Left/Right = prev/next. Keyboard shortcuts work when the note box is not focused.</div>
      </div>
    </aside>

    <main class="panel content">
      <section class="hero">
        <div class="topline">
          <div>
            <div class="muted">Reviewing timeline</div>
            <h2>@{html.escape(user.username)}</h2>
          </div>
          <div class="badge">saved to {html.escape(str(store.path))}</div>
        </div>
        <div class="hero-grid">
          <div class="hero-card">
            <strong>Language mix</strong>
            <div class="muted">{html.escape(", ".join(f"{lang}: {count}" for lang, count in language_counts.most_common(5)) or "n/a")}</div>
          </div>
          <div class="hero-card">
            <strong>Media mix</strong>
            <div class="muted">{html.escape(", ".join(f"{media}: {count}" for media, count in media_counts.most_common()) or "no media")}</div>
          </div>
          <div class="hero-card">
            <strong>Media availability</strong>
            <div class="muted">local: {local_media_count}, remote only: {remote_only_media_count}</div>
          </div>
          <div class="hero-card">
            <strong>Metadata</strong>
            <div class="muted">collected: {html.escape(_display_datetime(user.metadata.get("collected_at")))}</div>
          </div>
        </div>
        <details>
          <summary>Raw metadata</summary>
          <pre>{html.escape(metadata_block)}</pre>
        </details>
      </section>

      <section>
        <div class="topline">
          <h2>Tweet Stream</h2>
          <div class="badge">{len(posts)} shown{" / " + str(user.post_count) if truncated else ""}</div>
        </div>
        <div class="view-controls">
          <span class="muted">Show posts:</span>
          {view_links}
        </div>
        <div class="view-controls">
          <span class="muted">Media:</span>
          {media_links}
        </div>
        {"<div class='muted' style='margin-bottom: 12px;'>Output truncated by max-posts setting.</div>" if truncated else ""}
        <div class="stream">
          {post_cards}
        </div>
      </section>
    </main>
  </div>
  <script>
    document.addEventListener("keydown", function(event) {{
      if (event.target && ["TEXTAREA", "INPUT"].includes(event.target.tagName)) {{
        return;
      }}
      if (event.key === "q" || event.key === "Q") {{
        document.querySelector("button[value='qualified']").click();
      }} else if (event.key === "w" || event.key === "W") {{
        document.querySelector("button[value='rejected']").click();
      }} else if (event.key === "s" || event.key === "S") {{
        document.querySelector("button[value='skip']").click();
      }} else if (event.key === "ArrowLeft") {{
        window.location.href = "{prev_url}";
      }} else if (event.key === "ArrowRight") {{
        window.location.href = "{next_url}";
      }}
    }});
    const jumpForm = document.getElementById("jump-form");
    if (jumpForm) {{
      jumpForm.addEventListener("submit", function(event) {{
        event.preventDefault();
        const indexInput = jumpForm.querySelector("input[name='index']");
        const limitInput = jumpForm.querySelector("input[name='limit']");
        const mediaModeInput = jumpForm.querySelector("input[name='media_mode']");
        const rawIndex = parseInt(indexInput.value || "1", 10);
        const clampedIndex = Math.min(Math.max(rawIndex, 1), {total_users});
        const currentLimitValue = (limitInput && limitInput.value) ? limitInput.value : "all";
        const currentMediaMode = (mediaModeInput && mediaModeInput.value) ? mediaModeInput.value : "deferred";
        window.location.href =
          "/user/" + (clampedIndex - 1) +
          "?limit=" + encodeURIComponent(currentLimitValue) +
          "&media=" + encodeURIComponent(currentMediaMode);
      }});
    }}
    document.querySelectorAll(".media-toggle").forEach(function(button) {{
      button.addEventListener("click", function() {{
        const templateId = button.getAttribute("data-template-id");
        const targetId = button.getAttribute("data-target-id");
        const template = document.getElementById(templateId);
        const target = document.getElementById(targetId);
        if (!template || !target || target.childNodes.length > 0) {{
          return;
        }}
        target.appendChild(template.content.cloneNode(true));
        button.disabled = true;
        button.textContent = "Media loaded";
      }});
    }});
  </script>
</body>
</html>
"""


def _redirect(handler: BaseHTTPRequestHandler, location: str) -> None:
    handler.send_response(HTTPStatus.SEE_OTHER)
    handler.send_header("Location", _external_location(handler, location))
    handler.end_headers()


def _safe_write(handler: BaseHTTPRequestHandler, data: bytes) -> bool:
    try:
        handler.wfile.write(data)
        return True
    except (BrokenPipeError, ConnectionResetError):
        return False


def _send_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    content_type, _ = mimetypes.guess_type(path.name)
    file_size = path.stat().st_size
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(file_size))
    handler.end_headers()

    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                if not _safe_write(handler, chunk):
                    return
    except (BrokenPipeError, ConnectionResetError):
        return


def _serve_asset(handler: BaseHTTPRequestHandler, users_by_id: dict[str, UserTimeline], user_id: str, relative_path: str) -> None:
    user = users_by_id.get(user_id)
    if user is None:
        handler.send_error(HTTPStatus.NOT_FOUND, "Unknown user")
        return

    candidate = (user.user_dir / relative_path).resolve()
    user_root = user.user_dir.resolve()
    try:
        candidate.relative_to(user_root)
    except ValueError:
        handler.send_error(HTTPStatus.FORBIDDEN, "Invalid asset path")
        return
    if not candidate.exists() or not candidate.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
        return

    _send_file(handler, candidate)


def make_handler(
    *,
    users: list[UserTimeline],
    store: AnnotationStore,
    max_posts: int | None,
) -> type[BaseHTTPRequestHandler]:
    users_by_id = {user.user_id: user for user in users}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                start_index = _find_next_unreviewed(users, store.session.annotations, store.session.current_index)
                _redirect(self, f"/user/{start_index}")
                return

            if parsed.path.startswith("/user/"):
                try:
                    user_index = int(parsed.path.split("/", 2)[2])
                except (IndexError, ValueError):
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid user index")
                    return
                params = parse_qs(parsed.query)
                requested_limit = _query_int(params, "limit")
                show_all_posts = params.get("limit", [""])[0] == "all"
                media_mode = params.get("media", ["deferred"])[0].strip() or "deferred"
                page = _render_user_page(
                    users=users,
                    store=store,
                    user_index=user_index,
                    max_posts=max_posts,
                    current_limit=requested_limit,
                    show_all_posts=show_all_posts,
                    media_mode=media_mode,
                )
                body = page.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                _safe_write(self, body)
                return

            if parsed.path == "/goto":
                params = parse_qs(parsed.query)
                requested_index = _query_int(params, "index")
                requested_limit = params.get("limit", [""])[0]
                requested_media_mode = params.get("media", ["deferred"])[0] or "deferred"
                if requested_index is None:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing user index")
                    return
                user_index = min(max(requested_index - 1, 0), len(users) - 1)
                _redirect(self, _build_user_path(user_index, requested_limit or "all", requested_media_mode))
                return

            if parsed.path.startswith("/asset/"):
                trimmed = parsed.path[len("/asset/") :]
                parts = trimmed.split("/", 1)
                if len(parts) != 2:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid asset path")
                    return
                user_id = unquote(parts[0])
                relative_path = unquote(parts[1])
                _serve_asset(self, users_by_id, user_id, relative_path)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            if self.path != "/annotate":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            content_length = _safe_int(self.headers.get("Content-Length"), 0)
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = parse_qs(raw_body, keep_blank_values=True)
            user_index = _safe_int(payload.get("user_index", ["0"])[0], 0)
            user_index = min(max(user_index, 0), len(users) - 1)
            label = str(payload.get("label", ["skip"])[0]).strip() or "skip"
            notes = str(payload.get("notes", [""])[0])
            limit_value = str(payload.get("limit", [""])[0]).strip()
            media_mode = str(payload.get("media_mode", ["deferred"])[0]).strip() or "deferred"
            if label not in {"qualified", "rejected", "skip"}:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid label")
                return

            user = users[user_index]
            store.annotate(
                user=user,
                user_index=user_index,
                label=label,
                notes=notes,
                current_index=user_index,
                users=users,
            )
            next_index = _find_next_unreviewed(users, store.session.annotations, user_index + 1)
            next_path = _build_user_path(next_index, limit_value or "all", media_mode)
            _redirect(self, next_path)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def run_annotation_server(
    *,
    dataset_root: Path,
    output_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_posts: int | None = DEFAULT_MAX_POSTS,
    open_browser: bool = True,
) -> None:
    users = load_user_timelines(dataset_root)
    resolved_output = (output_path or build_default_output_path(dataset_root)).expanduser().resolve()
    store = AnnotationStore(resolved_output, dataset_root)
    store.save()
    store.export_csv(users)

    handler_cls = make_handler(users=users, store=store, max_posts=max_posts)
    server = ThreadingHTTPServer((host, port), handler_cls)
    url = f"http://{host}:{port}/"

    print(f"Loaded {len(users)} users from {Path(dataset_root).expanduser().resolve()}")
    print(f"Annotation JSON: {store.path}")
    print(f"Annotation CSV: {store.csv_path}")
    print(f"Open {url} in your browser")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")
    finally:
        server.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review user tweet streams and label each user as qualified or rejected."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/user_211_safe"),
        help="Directory containing per-user folders with posts.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the annotation JSON file. CSV will be written next to it.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local server to.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local server.")
    parser.add_argument(
        "--max-posts",
        type=int,
        default=DEFAULT_MAX_POSTS,
        help="Optional limit on posts shown per user page.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser tab.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_annotation_server(
        dataset_root=args.dataset_root,
        output_path=args.output,
        host=args.host,
        port=args.port,
        max_posts=args.max_posts,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
