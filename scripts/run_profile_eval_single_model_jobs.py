#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


MODELS = [
    "gemini-2.5-flash",
    "qwen2.5-vl-7b-instruct",
    "qwen3-vl-8b-instruct",
    "qwen3.5-35b-a3b",
    "gpt-4o-mini",
    "gpt-5.4",
]

METHODS = {
    "direct": {
        "profile_method": "direct",
        "out_base": "outputs/profile_eval_no_safety_filter_20260425",
    },
    "hierarchical": {
        "profile_method": "hierarchical",
        "out_base": "outputs/profile_eval_hierarchical_no_safety_filter_20260425",
    },
    "extractive": {
        "profile_method": "extractive",
        "out_base": "outputs/profile_eval_extractive_no_safety_filter_20260425",
    },
}

ANCHOR_MATCH_MODEL = "gpt-o3"
ANCHOR_MATCH_PROVIDER = "chatanywhere"
ANCHOR_MATCH_API_MODEL = "o3"
ANCHOR_MATCH_BASE_URL = "https://api.chatanywhere.tech/v1/chat/completions"
ANCHOR_MATCH_PROMPT_VERSION = "anchor_match_v2_core_interest_force_llm"

JSON_REPAIR_PROVIDER = "openai_compatible"
JSON_REPAIR_API_KEY_ENV = "ALI_BAILIAN_1"
JSON_REPAIR_API_MODEL = "qwen2.5-vl-7b-instruct"
JSON_REPAIR_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

DEFAULT_GOLD_LIST = Path("outputs/profile_eval_no_safety_filter_20260425/state/up_to_100_gold_files.txt")


def slugify(value: str) -> str:
    lowered = (value or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", lowered)
    return cleaned.strip("_") or "model"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fixed_gold_files(gold_list_file: Path, gold_root: Path, max_gold_users: int) -> list[Path]:
    if not gold_list_file.is_file():
        raise FileNotFoundError(f"fixed gold-user list is missing: {gold_list_file}")
    names = [
        Path(line.strip()).name
        for line in gold_list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate gold users in {gold_list_file}: {duplicate_names}")
    if max_gold_users < 0:
        raise ValueError("max_gold_users must be zero (all listed users) or a positive integer")
    requested_users = max_gold_users or len(names)
    if max_gold_users > 0:
        names = names[:max_gold_users]
    if len(names) != requested_users:
        raise RuntimeError(
            f"requested {requested_users} gold users, but {gold_list_file} contains only {len(names)}"
        )
    out = []
    for name in names:
        path = gold_root / name
        if not path.exists():
            raise FileNotFoundError(f"fixed gold file is missing under filtered gold root: {path}")
        out.append(path)
    if not out:
        raise RuntimeError(f"no gold users were resolved from {gold_list_file}")
    return out


def batch_endpoints(total_users: int, batch_size: int) -> list[int]:
    if total_users <= 0:
        raise ValueError("total_users must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    endpoints = list(range(batch_size, total_users + 1, batch_size))
    if not endpoints or endpoints[-1] != total_users:
        endpoints.append(total_users)
    return endpoints


def expected_tasks(gold_files: list[Path], end: int) -> int:
    total = 0
    for path in gold_files[:end]:
        payload = load_json(path)
        total += len(payload.get("domains") or [])
    return total


def metric_is_valid(path: Path, *, model: str, method: str, expected_n_tasks: int) -> bool:
    if not path.exists():
        return False
    try:
        metrics = load_json(path)
    except Exception:
        return False
    if str(metrics.get("model") or "") != model:
        return False
    if str(metrics.get("profile_method") or "").lower() != method.lower():
        return False
    if int(metrics.get("n_tasks") or 0) != expected_n_tasks:
        return False
    if int(metrics.get("n_tasks_skipped") or 0) != 0:
        return False
    return True


def copy_model_result(src_batch_dir: Path, dst_batch_dir: Path, model: str) -> None:
    model_slug = slugify(model)
    src_model_dir = src_batch_dir / model_slug
    dst_model_dir = dst_batch_dir / model_slug
    if not (src_model_dir / "metrics.json").exists():
        raise FileNotFoundError(f"missing model metrics: {src_model_dir / 'metrics.json'}")
    dst_model_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_model_dir.exists():
        shutil.rmtree(dst_model_dir)
    shutil.copytree(src_model_dir, dst_model_dir)


def merge_batch_report(
    *,
    out_base: Path,
    batch_label: str,
    method: str,
    expected_n_tasks: int,
    total_gold_users: int,
    batch_size: int,
    sync_latest: bool,
    project_root: Path,
    python_exe: str,
) -> bool:
    batch_dir = out_base / "incremental" / batch_label
    metrics_by_model: list[dict[str, Any]] = []
    for model in MODELS:
        metrics_path = batch_dir / slugify(model) / "metrics.json"
        if not metric_is_valid(metrics_path, model=model, method=method, expected_n_tasks=expected_n_tasks):
            return False
        metrics_by_model.append(load_json(metrics_path))

    report = {
        "n_tasks": expected_n_tasks,
        "models": metrics_by_model,
        "is_official_run": True,
        "scoring_mode": "f1_only",
        "anchor_match_model": ANCHOR_MATCH_API_MODEL,
        "anchor_match_provider": ANCHOR_MATCH_PROVIDER,
        "anchor_match_api_model": ANCHOR_MATCH_API_MODEL,
        "anchor_match_base_url": ANCHOR_MATCH_BASE_URL,
        "anchor_match_prompt_version": ANCHOR_MATCH_PROMPT_VERSION,
        "visual_mode": "text_image",
        "profile_input_mode": "text_image_captions_timestamps",
        "profile_method": method,
        "profile_method_max_posts": 200,
        "hierarchical_chunk_size": 20,
        "extractive_k": 12,
        "generated_at": int(time.time()),
        "output_dir": str(batch_dir.resolve()),
        "merged_from_single_model_jobs": True,
    }
    write_json(batch_dir / "benchmark_report.json", report)

    if sync_latest:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        subprocess.run(
            [
                python_exe,
                "scripts/sync_profile_latest_scores.py",
                "--out-base",
                str(out_base),
                "--total-gold-users",
                str(total_gold_users),
                "--batch-size",
                str(batch_size),
                "--anchor-match-model",
                ANCHOR_MATCH_API_MODEL,
                "--anchor-match-provider",
                ANCHOR_MATCH_PROVIDER,
                "--anchor-match-base-url",
                ANCHOR_MATCH_BASE_URL,
                "--anchor-match-prompt-version",
                ANCHOR_MATCH_PROMPT_VERSION,
            ],
            cwd=project_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    return True


def run_one_method_model(
    *,
    project_root: Path,
    python_exe: str,
    cache_root: Path,
    anchor_root: Path,
    data_root: Path,
    gold_files: list[Path],
    method_key: str,
    model: str,
    batch_size: int,
    max_gold_users: int,
    work_root: Path,
    merge_lock: threading.Lock,
    retry_attempts: int,
    retry_sleep_seconds: float,
) -> None:
    method_cfg = METHODS[method_key]
    method = method_cfg["profile_method"]
    out_base = project_root / method_cfg["out_base"]
    log_dir = out_base / "logs" / "single_model_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{slugify(model)}.log"
    temp_method_root = work_root / method_key / slugify(model)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    with log_path.open("a", encoding="utf-8") as log:
        print(
            f"[start] method={method_key} model={model} time={time.strftime('%Y-%m-%d %H:%M:%S')}",
            file=log,
            flush=True,
        )
        for end in batch_endpoints(max_gold_users, batch_size):
            batch_label = f"up_to_{end:03d}"
            expected_n = expected_tasks(gold_files, end)
            canonical_metric = out_base / "incremental" / batch_label / slugify(model) / "metrics.json"
            if metric_is_valid(canonical_metric, model=model, method=method, expected_n_tasks=expected_n):
                with merge_lock:
                    merge_batch_report(
                        out_base=out_base,
                        batch_label=batch_label,
                        method=method,
                        expected_n_tasks=expected_n,
                        total_gold_users=max_gold_users,
                        batch_size=batch_size,
                        sync_latest=True,
                        project_root=project_root,
                        python_exe=python_exe,
                    )
                print(f"[skip-valid] {batch_label}", file=log, flush=True)
                continue

            temp_batch_dir = temp_method_root / "incremental" / batch_label
            temp_metric = temp_batch_dir / slugify(model) / "metrics.json"
            attempts_used = 0
            while not metric_is_valid(temp_metric, model=model, method=method, expected_n_tasks=expected_n):
                attempts_used += 1
                temp_batch_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    python_exe,
                    "-m",
                    "user_profile_pipeline.benchmark.profile_eval",
                    "--gold-exports",
                    *[str(path) for path in gold_files[:end]],
                    "--data-test-root",
                    str(data_root),
                    "--output-dir",
                    str(temp_batch_dir),
                    "--cache-dir",
                    str(cache_root),
                    "--anchor-match-cache-dir",
                    str(anchor_root),
                    "--models",
                    model,
                    "--visual-mode",
                    "text_image",
                    "--profile-method",
                    method,
                    "--max-posts",
                    "0",
                    "--json-repair-provider",
                    JSON_REPAIR_PROVIDER,
                    "--json-repair-api-key-env",
                    JSON_REPAIR_API_KEY_ENV,
                    "--json-repair-api-model",
                    JSON_REPAIR_API_MODEL,
                    "--json-repair-base-url",
                    JSON_REPAIR_BASE_URL,
                    "--anchor-match-model",
                    ANCHOR_MATCH_MODEL,
                    "--anchor-match-provider",
                    ANCHOR_MATCH_PROVIDER,
                    "--anchor-match-api-key-env",
                    "CHATANYWHERE_API_KEY",
                    "--anchor-match-api-model",
                    ANCHOR_MATCH_API_MODEL,
                    "--anchor-match-base-url",
                    ANCHOR_MATCH_BASE_URL,
                    "--anchor-match-prompt-version",
                    ANCHOR_MATCH_PROMPT_VERSION,
                    "--anchor-match-force-llm",
                ]
                print(
                    f"[eval] {batch_label} expected_tasks={expected_n} attempt={attempts_used}/{retry_attempts}",
                    file=log,
                    flush=True,
                )
                proc = subprocess.run(cmd, cwd=project_root, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"[exit] {batch_label} code={proc.returncode}", file=log, flush=True)
                if metric_is_valid(temp_metric, model=model, method=method, expected_n_tasks=expected_n):
                    break
                if attempts_used >= retry_attempts:
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"{method_key}/{model}/{batch_label} failed with exit code {proc.returncode}"
                        )
                    raise RuntimeError(f"{method_key}/{model}/{batch_label} finished but metrics are still invalid")
                print(
                    f"[retry] {batch_label} attempt={attempts_used} sleep_seconds={retry_sleep_seconds}",
                    file=log,
                    flush=True,
                )
                time.sleep(max(0.0, retry_sleep_seconds))

            with merge_lock:
                canonical_batch = out_base / "incremental" / batch_label
                copy_model_result(temp_batch_dir, canonical_batch, model)
                full = merge_batch_report(
                    out_base=out_base,
                    batch_label=batch_label,
                    method=method,
                    expected_n_tasks=expected_n,
                    total_gold_users=max_gold_users,
                    batch_size=batch_size,
                    sync_latest=True,
                    project_root=project_root,
                    python_exe=python_exe,
                )
            print(f"[merged] {batch_label} full_report={full}", file=log, flush=True)
        print(
            f"[done] method={method_key} model={model} time={time.strftime('%Y-%m-%d %H:%M:%S')}",
            file=log,
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run profile eval as independent single-model Windows jobs.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["direct", "hierarchical", "extractive"],
        choices=sorted(METHODS),
        help="Profile construction methods.",
    )
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--max-concurrent-jobs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-gold-users", type=int, default=100)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/profile_eval_cache_20260425"))
    parser.add_argument("--anchor-root", type=Path, default=Path("outputs/profile_eval_anchor_cache_20260425"))
    parser.add_argument("--data-root", type=Path, default=Path("data/user_211_safe"))
    parser.add_argument("--gold-root", type=Path, default=Path("outputs/profile_eval_inputs_20260425_filtered/gold"))
    parser.add_argument(
        "--gold-list-file",
        type=Path,
        default=DEFAULT_GOLD_LIST,
        help="Stable ordered user list; relative paths are resolved from --project-root.",
    )
    parser.add_argument("--work-root", type=Path, default=Path("outputs/profile_eval_single_model_jobs_20260425"))
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    python_exe = args.python_exe or str(project_root / ".venv/Scripts/python.exe")
    if not Path(python_exe).exists():
        python_exe = sys.executable

    cache_root = (project_root / args.cache_root).resolve() if not args.cache_root.is_absolute() else args.cache_root
    anchor_root = (project_root / args.anchor_root).resolve() if not args.anchor_root.is_absolute() else args.anchor_root
    data_root = (project_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    gold_root = (project_root / args.gold_root).resolve() if not args.gold_root.is_absolute() else args.gold_root
    gold_list_file = (
        (project_root / args.gold_list_file).resolve()
        if not args.gold_list_file.is_absolute()
        else args.gold_list_file
    )
    work_root = (project_root / args.work_root).resolve() if not args.work_root.is_absolute() else args.work_root
    gold_files = fixed_gold_files(gold_list_file, gold_root, int(args.max_gold_users))
    total_gold_users = len(gold_files)

    merge_lock = threading.Lock()
    selected_models = list(dict.fromkeys(args.models))
    jobs = [
        (method_key, model)
        for method_key in args.methods
        for model in selected_models
    ]
    max_workers = max(1, min(int(args.max_concurrent_jobs), len(jobs)))
    print(
        json.dumps(
            {
                "event": "single_model_jobs_start",
                "methods": args.methods,
                "models": selected_models,
                "jobs": len(jobs),
                "max_workers": max_workers,
                "max_gold_users": total_gold_users,
                "batch_size": args.batch_size,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                run_one_method_model,
                project_root=project_root,
                python_exe=python_exe,
                cache_root=cache_root,
                anchor_root=anchor_root,
                data_root=data_root,
                gold_files=gold_files,
                method_key=method_key,
                model=model,
                batch_size=int(args.batch_size),
                max_gold_users=total_gold_users,
                work_root=work_root,
                merge_lock=merge_lock,
                retry_attempts=max(1, int(args.retry_attempts)),
                retry_sleep_seconds=max(0.0, float(args.retry_sleep_seconds)),
            ): (method_key, model)
            for method_key, model in jobs
        }
        for future in as_completed(future_to_job):
            method_key, model = future_to_job[future]
            try:
                future.result()
                print(json.dumps({"event": "job_done", "method": method_key, "model": model}, ensure_ascii=False), flush=True)
            except Exception as exc:
                message = f"{method_key}/{model}: {exc}"
                failures.append(message)
                print(json.dumps({"event": "job_failed", "method": method_key, "model": model, "error": str(exc)}, ensure_ascii=False), flush=True)

    if failures:
        print(json.dumps({"event": "single_model_jobs_failed", "failures": failures}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps({"event": "single_model_jobs_done"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
