from __future__ import annotations

import argparse

from ...image_text import VISUAL_MODE_NATIVE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run personalized dialogue evaluation with model predictions and LLM judge scoring.")
    parser.add_argument("--dialogue-root", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--intervention-config", default="", help="Frozen config for budgeted profile-input interventions.")
    parser.add_argument("--intervention-action", choices=["prepare", "pilot", "run", "summarize"], default="prepare")
    parser.add_argument("--intervention-stage", choices=["all", "dialogue", "temporal"], default="all")
    parser.add_argument("--intervention-workers", type=int, default=0, help="Intervention concurrency, capped at 16; zero uses config.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--max-context-posts", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--visual-mode", default=VISUAL_MODE_NATIVE)
    parser.add_argument(
        "--image-text-cache-root",
        default="",
        help="Optional cache root containing _image_text/<model>/ captions; useful for reusing profile eval captions.",
    )
    parser.add_argument("--image-text-max-tokens", type=int, default=220)
    parser.add_argument("--image-text-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--image-text-cache-only",
        action="store_true",
        help="For visual_mode=text_image, read existing image-text cache only and skip missing captions instead of generating them.",
    )
    parser.add_argument("--bailian-use-batch", action="store_true")
    parser.add_argument("--bailian-batch-poll-seconds", type=int, default=120)
    parser.add_argument("--bailian-batch-max-wait-seconds", type=int, default=86400)
    parser.add_argument("--bailian-batch-completion-window", default="24h")
    parser.add_argument("--bailian-batch-callback-url", default="")
    # Judge batch mode (DashScope / qwen3.7-max)
    parser.add_argument("--judge-use-bailian-batch", action="store_true",
                        help="Use DashScope batch inference with qwen3.7-max as judge instead of per-user API calls.")
    parser.add_argument("--judge-bailian-batch-poll-seconds", type=int, default=120,
                        help="Seconds between polls when waiting for judge batch to complete.")
    parser.add_argument("--judge-bailian-batch-max-wait-seconds", type=int, default=86400,
                        help="Maximum seconds to wait for judge batch to finish.")
    parser.add_argument("--judge-bailian-batch-completion-window", default="24h",
                        help="DashScope batch completion window for judge batch (e.g., 24h).")
    parser.add_argument("--judge-bailian-batch-callback-url", default="",
                        help="Optional DashScope callback URL for judge batch completion.")
    parser.add_argument("--judge-model", default="gpt-o3")
    parser.add_argument("--judge-provider", default="")
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--judge-api-key-env", default="")
    parser.add_argument("--judge-api-model", default="")
    parser.add_argument("--judge-timeout-seconds", type=int, default=120)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--parallel-models", action="store_true")
    parser.add_argument("--max-parallel-models", type=int, default=1)
    return parser
