from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...image_text import VISUAL_MODE_TEXT_IMAGE, normalize_visual_mode
from .evaluator import BenchmarkEvaluator
from .io import _discover_posts_by_user, _parse_posts_map, _resolve_model_specs
from .model_selection import (
    _enable_bailian_batch_if_requested,
    _parse_models_arg,
    _validate_bailian_batch_model_specs,
)
from .modes import normalize_profile_input_mode, normalize_profile_method
from .specs import DIRECT_PROFILE_METHOD, PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS
from .tasks import build_tasks_from_gold_exports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark evaluator for domain-level user profiling.")
    parser.add_argument("--gold-exports", nargs="*", default=[], help="One or more benchmark gold export JSON files.")
    parser.add_argument(
        "--gold-export-dir",
        default="",
        help="Optional directory containing benchmark gold export JSON files.",
    )
    parser.add_argument(
        "--models",
        default="mock_oracle",
        help="Comma/space separated model names registered in specs.py.",
    )
    parser.add_argument("--output-dir", default="outputs/benchmark_eval")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-run model predictions.")
    parser.add_argument("--force-image-text", action="store_true", help="Ignore image-text cache and re-run captions.")
    parser.add_argument("--max-posts", type=int, default=20, help="Default is 20 for low-memory hosts; set 0 to use all posts.")
    parser.add_argument("--max-images", type=int, default=0, help="Default is 0 for low-memory hosts; set -1 to include all images.")
    parser.add_argument("--max-images-when-trim", type=int, default=0, help="Default is 0 for low-memory hosts; set -1 to keep one image per post when downgraded.")
    parser.add_argument("--posts-map", nargs="*", default=[], help="Optional mapping list: user_id=/abs/path/posts.jsonl or direct posts.jsonl paths.")
    parser.add_argument("--data-test-root", default="data/test")
    parser.add_argument("--vertex-project-id", default="")
    parser.add_argument("--vertex-location", default="")
    parser.add_argument("--vertex-bucket", default="")
    parser.add_argument("--vertex-access-token-env", default="VERTEX_ACCESS_TOKEN")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-wait-seconds", type=int, default=14400)
    parser.add_argument("--parallel-models", action="store_true", help="Run per-model evaluation in parallel.")
    parser.add_argument("--max-parallel-models", type=int, default=1, help="Maximum parallel model workers.")
    parser.add_argument("--max-parallel-tasks", type=int, default=1, help="Maximum parallel task workers within a single model (1 = sequential).")
    parser.add_argument("--bailian-use-batch", action="store_true", help="Use DashScope OpenAI-compatible Batch File API for Bailian models that officially support batch.")
    parser.add_argument("--bailian-batch-poll-seconds", type=int, default=120)
    parser.add_argument("--bailian-batch-max-wait-seconds", type=int, default=86400)
    parser.add_argument("--bailian-batch-completion-window", default="24h")
    parser.add_argument("--bailian-batch-callback-url", default="", help="Optional Beijing-region callback URL passed as ds_batch_finish_callback metadata.")
    parser.add_argument(
        "--visual-mode",
        default=VISUAL_MODE_TEXT_IMAGE,
        choices=["native", "text_image", "wo_text_image"],
    )
    parser.add_argument(
        "--profile-input-mode",
        default=PROFILE_INPUT_MODE_TEXT_IMAGE_CAPTIONS_TIMESTAMPS,
        choices=["text_only", "image_captions_only", "text_image_captions", "text_image_captions_timestamps"],
        help=(
            "Profile construction input ablation mode: text_only | image_captions_only | "
            "text_image_captions | text_image_captions_timestamps"
        ),
    )
    parser.add_argument(
        "--profile-method",
        default=DIRECT_PROFILE_METHOD,
        choices=["direct", "hierarchical", "extractive"],
        help=(
            "Profile generation method: direct | hierarchical | extractive. "
            "hierarchical uses 20-post chunk summaries by default; extractive uses K=12 by default."
        ),
    )
    parser.add_argument("--profile-method-max-posts", type=int, default=200)
    parser.add_argument("--hierarchical-chunk-size", type=int, default=20)
    parser.add_argument("--extractive-k", type=int, default=12)
    parser.add_argument(
        "--profile-eval-prompt-variant",
        default="conservative",
        choices=["conservative", "neutral"],
        help="Prompt variant for profile evaluation: conservative (default, prefer fewer/broader tags) or neutral (return all supported tags).",
    )
    parser.add_argument(
        "--image-text-model",
        default="",
        help="Optional shared caption model for text_image mode. Defaults to the current eval model.",
    )
    parser.add_argument("--image-text-max-tokens", type=int, default=220)
    parser.add_argument("--image-text-timeout-seconds", type=int, default=120)
    parser.add_argument("--json-repair-provider", default="", help="Optional override provider for JSON repair.")
    parser.add_argument("--json-repair-api-key-env", default="", help="Optional override API key env var for JSON repair.")
    parser.add_argument("--json-repair-api-model", default="", help="Optional override model id for JSON repair.")
    parser.add_argument("--json-repair-base-url", default="", help="Optional override base URL for JSON repair.")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--anchor-match-model", default="gpt-o3", help="Registered LLM judge, or exact_match for offline normalized exact-label matching.")
    parser.add_argument("--anchor-match-provider", default="", help="openai_compatible|vertex|chatanywhere|bailian")
    parser.add_argument("--anchor-match-api-key-env", default="", help="Optional override for anchor-match api key env var.")
    parser.add_argument("--anchor-match-api-model", default="", help="Optional override for the anchor-match LLM model id.")
    parser.add_argument("--anchor-match-base-url", default="", help="Optional override for the anchor-match base URL.")
    parser.add_argument("--anchor-match-prompt-version", default="anchor_match_v1")
    parser.add_argument("--anchor-match-cache-dir", default="", help="Optional anchor-match cache directory")
    parser.add_argument(
        "--anchor-match-force-llm",
        action="store_true",
        help="Disable exact-match and YES/NO fallback shortcuts; force anchor matching through the anchor-match LLM.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    gold_paths = [Path(x).resolve() for x in list(args.gold_exports or [])]
    gold_export_dir = Path(str(args.gold_export_dir or "").strip()).resolve() if str(args.gold_export_dir or "").strip() else None
    if gold_export_dir is not None:
        if not gold_export_dir.exists() or not gold_export_dir.is_dir():
            raise FileNotFoundError(f"gold export dir not found: {gold_export_dir}")
        gold_paths.extend(sorted(path.resolve() for path in gold_export_dir.glob("*.json")))
    if not gold_paths:
        raise ValueError("No gold exports resolved. Provide --gold-exports and/or --gold-export-dir.")
    for path in gold_paths:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"gold export not found: {path}")

    posts_by_user = _discover_posts_by_user(Path(args.data_test_root).resolve())
    posts_by_user.update(_parse_posts_map(args.posts_map))

    tasks = build_tasks_from_gold_exports(
        gold_export_paths=gold_paths,
        posts_by_user=posts_by_user,
        max_posts=int(args.max_posts) if int(args.max_posts) > 0 else None,
    )
    models = _parse_models_arg(args.models)
    model_specs = _enable_bailian_batch_if_requested(
        _resolve_model_specs(models),
        enabled=bool(args.bailian_use_batch),
    )
    visual_mode = normalize_visual_mode(str(args.visual_mode or VISUAL_MODE_TEXT_IMAGE))
    profile_input_mode = normalize_profile_input_mode(str(args.profile_input_mode or ""))
    profile_method = normalize_profile_method(str(args.profile_method or DIRECT_PROFILE_METHOD))
    _validate_bailian_batch_model_specs(model_specs, visual_mode=visual_mode)

    cache_dir = Path(args.cache_dir).resolve() if str(args.cache_dir).strip() else None
    anchor_match_cache_dir = Path(args.anchor_match_cache_dir).resolve() if str(args.anchor_match_cache_dir).strip() else None

    evaluator = BenchmarkEvaluator(
        output_dir=Path(args.output_dir).resolve(),
        cache_dir=cache_dir,
        model_specs=model_specs,
        force=bool(args.force),
        force_image_text=bool(args.force_image_text),
        max_images=int(args.max_images),
        max_images_when_trim=int(args.max_images_when_trim),
        vertex_project_id=str(args.vertex_project_id or "").strip(),
        vertex_location=str(args.vertex_location or "").strip(),
        vertex_bucket=str(args.vertex_bucket or "").strip(),
        vertex_access_token_env=str(args.vertex_access_token_env or "VERTEX_ACCESS_TOKEN").strip(),
        poll_seconds=int(args.poll_seconds),
        max_wait_seconds=int(args.max_wait_seconds),
        parallel_models=bool(args.parallel_models),
        max_parallel_models=int(args.max_parallel_models),
        max_parallel_tasks=int(args.max_parallel_tasks),
        bailian_use_batch=bool(args.bailian_use_batch),
        bailian_batch_poll_seconds=int(args.bailian_batch_poll_seconds),
        bailian_batch_max_wait_seconds=int(args.bailian_batch_max_wait_seconds),
        bailian_batch_completion_window=str(args.bailian_batch_completion_window or "24h").strip() or "24h",
        bailian_batch_callback_url=str(args.bailian_batch_callback_url or "").strip(),
        visual_mode=visual_mode,
        profile_input_mode=profile_input_mode,
        profile_method=profile_method,
        profile_method_max_posts=int(args.profile_method_max_posts),
        hierarchical_chunk_size=int(args.hierarchical_chunk_size),
        extractive_k=int(args.extractive_k),
        profile_eval_prompt_variant=str(args.profile_eval_prompt_variant or "conservative").strip() or "conservative",
        image_text_model=str(args.image_text_model or "").strip(),
        image_text_max_tokens=int(args.image_text_max_tokens),
        image_text_timeout_seconds=int(args.image_text_timeout_seconds),
        json_repair_provider=str(args.json_repair_provider or "").strip(),
        json_repair_api_key_env=str(args.json_repair_api_key_env or "").strip(),
        json_repair_api_model=str(args.json_repair_api_model or "").strip(),
        json_repair_base_url=str(args.json_repair_base_url or "").strip(),
        request_timeout_seconds=int(args.timeout_seconds),
        anchor_match_model=str(args.anchor_match_model or "gpt-o3").strip() or "gpt-o3",
        anchor_match_provider=str(args.anchor_match_provider or "").strip(),
        anchor_match_api_key_env=str(args.anchor_match_api_key_env or "").strip(),
        anchor_match_api_model=str(args.anchor_match_api_model or "").strip(),
        anchor_match_base_url=str(args.anchor_match_base_url or "").strip(),
        anchor_match_prompt_version=str(args.anchor_match_prompt_version or "anchor_match_v1").strip() or "anchor_match_v1",
        anchor_match_cache_dir=anchor_match_cache_dir,
        anchor_match_force_llm=bool(args.anchor_match_force_llm),
    )
    return evaluator.run(tasks)
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    report = run_from_args(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
