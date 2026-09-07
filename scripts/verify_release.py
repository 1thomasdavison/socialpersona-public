#!/usr/bin/env python3
"""Fail closed on unsafe public data, package artifacts, and tracked history."""
import argparse
from pathlib import Path
import re
import subprocess

from user_profile_pipeline.release_privacy import verify_data

FORBIDDEN_PARTS = {"private", "outputs", "results", "artifacts", "archive", "archives", "logs", "tmp", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".zip", ".gz", ".tar", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".pdf", ".mp4"}
SECRET_RE = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
PRIVATE_PATH_RE = re.compile(r"(?:[A-Z]:[/\\](?:Users|socialpersona)|/home/[\w.-]+/)", re.I)


def forbidden(path):
    p = Path(path)
    return (any(x in FORBIDDEN_PARTS for x in p.parts)
            or p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name == ".env"
            or "mapping" in p.name.lower() or p.name == ".DS_Store")


def verify_package(root, require_git=False):
    issues = verify_data(root)
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if ".git" in rel.parts:
            continue
        if path.is_symlink():
            issues.append(f"Symlink not permitted: {rel}")
        if forbidden(rel):
            issues.append(f"Forbidden artifact: {rel}")
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                issues.append(f"Unreviewed binary: {rel}")
                continue
            if SECRET_RE.search(content):
                issues.append(f"Credential pattern: {rel}")
            if PRIVATE_PATH_RE.search(content):
                issues.append(f"Private filesystem path: {rel}")
    if (root / ".git").is_dir():
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], text=True, encoding="utf-8", capture_output=True, check=True).stdout
        revisions = git("rev-list", "--all").splitlines()
        if require_git and not revisions:
            issues.append("No committed release to inspect")
        seen = set()
        for revision in revisions:
            for line in git("ls-tree", "-r", revision).splitlines():
                meta, name = line.split("\t", 1)
                mode, kind, oid = meta.split()
                if forbidden(name) or mode in ("120000", "160000"):
                    issues.append(f"Forbidden history artifact: {revision[:8]}:{name}")
                if oid not in seen and kind == "blob":
                    seen.add(oid)
                    content = git("cat-file", "blob", oid)
                    if SECRET_RE.search(content) or PRIVATE_PATH_RE.search(content):
                        issues.append(f"Sensitive history blob: {revision[:8]}:{name}")
    elif require_git:
        issues.append("Release Git history is required")
    return issues


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=root if (root / "data/users").is_dir() else root / "socialpersona_public")
    parser.add_argument("--require-git", action="store_true")
    args = parser.parse_args()
    issues = verify_package(args.release_dir.resolve(), args.require_git)
    if issues:
        print("Release checks FAILED:")
        for issue in issues[:30]:
            print(" - " + issue)
        raise SystemExit(1)
    print("PASS: all 100 users, strict text/caption/profile schemas, all references, data checksums, package scan and available Git history")


if __name__ == "__main__":
    main()
