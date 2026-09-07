from __future__ import annotations

import argparse
import json
from pathlib import Path

import orjson
from .config import load_domain_configs, load_yaml
from .gate import NarrowGate
from .llm_client import FIXED_REPAIR_MODEL, OpenAICompatibleChatClient
from .schemas import PostObservable
from .single_post import SinglePostAnalyzer
from .vertex_batch import VertexBatchInferenceClient
from .workflow import run_user_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="User profiling pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("profile-post", help="Run narrow gate + single-post AI analysis")
    p1.add_argument("--post-json", required=True, help="Path to one post JSON file")
    p1.add_argument("--domains-config", default="configs/domains.yaml")
    p1.add_argument("--model-config", default="configs/model.yaml")
    p1.add_argument("--out-dir", default="outputs")

    p2 = subparsers.add_parser("aggregate-posts", help="Run multi-post aggregation from raw posts JSONL and single-post profile JSONL")
    p2.add_argument("--posts-jsonl", required=True, help="Path to raw posts JSONL")
    p2.add_argument("--profiles-jsonl", required=True, help="Path to single-post profiles JSONL; each row must contain post_id and profile or be profile payload with _post_id")
    p2.add_argument("--tag-aliases", default="configs/tag_aliases.yaml")
    p2.add_argument("--out-json", default="outputs/multi_post_aggregate.json")

    p3 = subparsers.add_parser("summarize-domains", help="Build evidence-constrained domain packs and run domain-level LLM summarization")
    p3.add_argument("--aggregate-json", required=True, help="Path to aggregate JSON produced by aggregate-posts")
    p3.add_argument("--posts-jsonl", required=True, help="Path to raw posts JSONL")
    p3.add_argument("--model-config", default="configs/model.domain.yaml")
    p3.add_argument("--domain-model", default="gpt-5.4", help="Model for cross-post domain-level aggregation LLM")
    p3.add_argument(
        "--repair-model",
        default="gpt-5.4",
        help="Model used only for malformed-JSON rewrite.",
    )
    p3.add_argument("--out-json", default="outputs/domain_llm_summary.json")
    p3.add_argument("--out-packs-jsonl", default="outputs/domain_packs.jsonl")
    p3.add_argument("--include-absent", dest="include_absent", action="store_true", default=True, help="Include domains with status=absent (default)")
    p3.add_argument("--exclude-absent", dest="include_absent", action="store_false", help="Exclude domains with status=absent")
    p3.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one domain summarization fails")

    p4 = subparsers.add_parser("vertex-batch-submit", help="Submit a Vertex Gemini batch prediction job from GCS JSONL input")
    p4.add_argument("--project-id", default="61866670-6b45-46ec-9d9", help="Google Cloud project ID")
    p4.add_argument("--location", default="us-central1", help="Vertex job location, e.g. us-central1 or global")
    p4.add_argument("--model-config", default="configs/model.yaml")
    p4.add_argument("--model", default="", help="Override model path or model id; default comes from model-config")
    p4.add_argument("--input-gcs-uri", required=True, help="GCS JSONL input URI, e.g. gs://bucket/path/input.jsonl")
    p4.add_argument("--output-gcs-uri-prefix", required=True, help="GCS output prefix, e.g. gs://bucket/path/output")
    p4.add_argument("--display-name", default="", help="Optional batch job display name")
    p4.add_argument("--access-token-env", default="VERTEX_ACCESS_TOKEN", help="Env var name for Vertex OAuth access token")
    p4.add_argument("--out-job-json", default="outputs/vertex_batch_submit_job.json")

    p5 = subparsers.add_parser("vertex-batch-status", help="Get Vertex batch job status by job id or full job name")
    p5.add_argument("--project-id", required=True, help="Google Cloud project ID")
    p5.add_argument("--location", required=True, help="Vertex job location, e.g. us-central1 or global")
    p5.add_argument("--job-name-or-id", required=True, help="Batch job id or full name projects/.../batchPredictionJobs/...")
    p5.add_argument("--model-config", default="configs/model.yaml")
    p5.add_argument("--access-token-env", default="VERTEX_ACCESS_TOKEN", help="Env var name for Vertex OAuth access token")
    p5.add_argument("--out-job-json", default="outputs/vertex_batch_status_job.json")

    p6 = subparsers.add_parser("vertex-batch-download", help="Download Vertex batch output JSONL files from GCS to local")
    p6.add_argument("--project-id", required=True, help="Google Cloud project ID")
    p6.add_argument("--location", required=True, help="Vertex job location, e.g. us-central1 or global")
    p6.add_argument("--job-name-or-id", required=True, help="Batch job id or full name projects/.../batchPredictionJobs/...")
    p6.add_argument("--model-config", default="configs/model.yaml")
    p6.add_argument("--output-gcs-uri-prefix", default="", help="Optional override for output GCS prefix")
    p6.add_argument("--local-dir", default="outputs/vertex_batch_outputs")
    p6.add_argument("--wait", action="store_true", help="Poll until batch job reaches a terminal state before downloading")
    p6.add_argument("--poll-seconds", type=int, default=30)
    p6.add_argument("--max-wait-seconds", type=int, default=7200)
    p6.add_argument("--include-non-jsonl", action="store_true", help="Also download non-JSONL files under output prefix")
    p6.add_argument("--access-token-env", default="VERTEX_ACCESS_TOKEN", help="Env var name for Vertex OAuth access token")

    p7 = subparsers.add_parser("run-user-pipeline", help="Run one user end-to-end from posts to personalized-dialogue metrics")
    p7.add_argument("--user-id", required=True, help="Dataset user id, e.g. x_example")
    p7.add_argument("--dataset-root", default="data/user_250", help="Root directory containing per-user subdirectories")
    p7.add_argument("--out-root", default="outputs/user_pipeline", help="Root directory for pipeline outputs")
    p7.add_argument("--run-tag", default="", help="Optional run tag; default is UTC timestamp")
    p7.add_argument("--domains-config", default="configs/domains.yaml")
    p7.add_argument("--model-config", default="configs/model.yaml")
    p7.add_argument("--domain-model-config", default="configs/model.domain.yaml")
    p7.add_argument("--tag-aliases", default="configs/tag_aliases.yaml")
    p7.add_argument("--domain-model", default="gpt-5.4")
    p7.add_argument("--benchmark-rewrite-model", default="gpt-5.4")
    p7.add_argument(
        "--repair-model",
        default="gpt-5.4",
        help="Model used only for malformed-JSON rewrite.",
    )
    p7.add_argument("--dialogue-model", default="mock_oracle", help="Dialogue experiment model. Default mock_oracle enables an offline smoke test.")
    p7.add_argument("--max-posts", type=int, default=None, help="Optional cap on loaded posts for one user.")
    p7.add_argument("--force", action="store_true", help="Re-run stages even if outputs already exist.")

    return parser


def run_profile_post(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.post_json, "rb") as f:
        post = PostObservable.model_validate(orjson.loads(f.read()))

    domain_configs = load_domain_configs(args.domains_config)
    gate = NarrowGate(domain_configs=domain_configs)

    model_cfg = load_yaml(args.model_config)
    client = OpenAICompatibleChatClient(
        provider=model_cfg["provider"],
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        api_key_env=model_cfg["api_key_env"],
        timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        max_tokens=int(model_cfg.get("max_tokens", 1200)),
    )
    analyzer = SinglePostAnalyzer(
        gate=gate,
        client=client,
        media_base_dir=Path(args.post_json).resolve().parent,
    )

    gate_result, profile = analyzer.analyze(post)

    gate_path = out_dir / f"{post.post_id}.gate.json"
    single_path = out_dir / f"{post.post_id}.single_post.json"

    gate_path.write_text(json.dumps(json.loads(gate_result.model_dump_json()), ensure_ascii=False, indent=2), encoding="utf-8")
    single_path.write_text(json.dumps(json.loads(profile.model_dump_json()), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "gate_result_path": str(gate_path),
        "single_post_result_path": str(single_path),
    }, ensure_ascii=False, indent=2))


def run_aggregate_posts(args: argparse.Namespace) -> None:
    from .aggregation import MultiPostAggregator

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregator = MultiPostAggregator.from_alias_config(args.tag_aliases)
    result = aggregator.aggregate_from_paths(args.posts_jsonl, args.profiles_jsonl)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate_result_path": str(out_path)}, ensure_ascii=False, indent=2))


def run_summarize_domains(args: argparse.Namespace) -> None:
    from .domain_llm import DomainLLMSummarizer, DomainPackBuilder

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packs_path = Path(args.out_packs_jsonl)
    packs_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate_payload = json.loads(Path(args.aggregate_json).read_text(encoding="utf-8"))
    posts = []
    with Path(args.posts_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            posts.append(PostObservable.model_validate(orjson.loads(line)))

    pack_builder = DomainPackBuilder(include_absent=bool(args.include_absent))
    domain_packs = pack_builder.build_packs(aggregate_result=aggregate_payload, posts=posts)
    with packs_path.open("w", encoding="utf-8") as f:
        for row in domain_packs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    model_cfg = load_yaml(args.model_config)
    client = OpenAICompatibleChatClient(
        provider=model_cfg["provider"],
        base_url=model_cfg["base_url"],
        model=args.domain_model or model_cfg["model"],
        api_key_env=model_cfg["api_key_env"],
        timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        max_tokens=int(model_cfg.get("max_tokens", 1200)),
        repair_model=FIXED_REPAIR_MODEL,
    )
    summarizer = DomainLLMSummarizer(
        client=client,
        media_base_dir=Path(args.posts_jsonl).resolve().parent,
    )

    summaries: list[dict] = []
    errors: list[dict] = []
    for pack in domain_packs:
        domain = pack.get("domain")
        try:
            summaries.append(summarizer.summarize_domain_pack(pack))
        except Exception as exc:
            errors.append({"domain": domain, "error": str(exc), "used_fallback": not bool(args.stop_on_error)})
            if args.stop_on_error:
                raise
            fallback_summary = summarizer.validator.validate({}, pack)
            fallback_note = "LLM summarization failed; returned conservative fallback from algorithmic evidence only."
            if fallback_summary.get("uncertainty_note"):
                fallback_summary["uncertainty_note"] = f"{fallback_summary['uncertainty_note']} {fallback_note}"
            else:
                fallback_summary["uncertainty_note"] = fallback_note
            summaries.append(fallback_summary)

    batch = {
        "schema_version": "domain_llm_batch_v1",
        "user_id": domain_packs[0].get("user_id") if domain_packs else None,
        "observation_window": aggregate_payload.get("observation_window"),
        "n_domain_packs": len(domain_packs),
        "n_domain_summaries": len(summaries),
        "n_errors": len(errors),
        "domain_summaries": summaries,
        "errors": errors,
    }
    out_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "domain_packs_path": str(packs_path),
        "domain_summary_path": str(out_path),
        "n_domain_packs": len(domain_packs),
        "n_domain_summaries": len(summaries),
        "n_errors": len(errors),
    }, ensure_ascii=False, indent=2))


def _load_model_cfg_if_exists(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    loaded = load_yaml(path)
    return loaded if isinstance(loaded, dict) else {}


def _build_vertex_batch_client_for_submit(args: argparse.Namespace, model_cfg: dict) -> VertexBatchInferenceClient:
    model = (args.model or model_cfg.get("model", "")).strip()
    if not model:
        raise ValueError("Model is required. Provide --model or set `model` in --model-config.")
    return VertexBatchInferenceClient(
        project_id=args.project_id,
        location=args.location,
        model=model,
        timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
        access_token_env=args.access_token_env,
    )


def _build_vertex_batch_client(args: argparse.Namespace, model_cfg: dict) -> VertexBatchInferenceClient:
    return VertexBatchInferenceClient(
        project_id=args.project_id,
        location=args.location,
        timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
        access_token_env=args.access_token_env,
    )


def run_vertex_batch_submit(args: argparse.Namespace) -> None:
    model_cfg = _load_model_cfg_if_exists(args.model_config)
    client = _build_vertex_batch_client_for_submit(args, model_cfg)
    job = client.submit_gcs_batch_job(
        input_gcs_uri=args.input_gcs_uri,
        output_gcs_uri_prefix=args.output_gcs_uri_prefix,
        display_name=(args.display_name or "").strip() or None,
    )

    out_path = Path(args.out_job_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    job_name = str(job.get("name", "")).strip()
    print(json.dumps({
        "job_name": job_name,
        "job_id": client.to_job_id(job_name) if job_name else None,
        "state": job.get("state"),
        "input_gcs_uri": args.input_gcs_uri,
        "output_gcs_uri_prefix": args.output_gcs_uri_prefix,
        "job_json_path": str(out_path),
    }, ensure_ascii=False, indent=2))


def run_vertex_batch_status(args: argparse.Namespace) -> None:
    model_cfg = _load_model_cfg_if_exists(args.model_config)
    client = _build_vertex_batch_client(args, model_cfg)
    job = client.get_batch_job(job_name_or_id=args.job_name_or_id)

    out_path = Path(args.out_job_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    output_prefix = client.get_output_uri_prefix(job)
    print(json.dumps({
        "job_name": job.get("name"),
        "job_id": client.to_job_id(str(job.get("name", "")).strip()) if job.get("name") else None,
        "state": job.get("state"),
        "completion_stats": job.get("completionStats", {}),
        "output_gcs_uri_prefix": output_prefix,
        "job_json_path": str(out_path),
    }, ensure_ascii=False, indent=2))


def run_vertex_batch_download(args: argparse.Namespace) -> None:
    model_cfg = _load_model_cfg_if_exists(args.model_config)
    client = _build_vertex_batch_client(args, model_cfg)

    if args.wait:
        job = client.wait_batch_job(
            job_name_or_id=args.job_name_or_id,
            poll_seconds=int(args.poll_seconds),
            max_wait_seconds=int(args.max_wait_seconds),
        )
    else:
        job = client.get_batch_job(job_name_or_id=args.job_name_or_id)

    output_prefix = (args.output_gcs_uri_prefix or "").strip() or client.get_output_uri_prefix(job)
    if not output_prefix:
        raise ValueError(
            "Cannot determine output GCS prefix. Provide --output-gcs-uri-prefix explicitly."
        )

    local_dir = Path(args.local_dir)
    downloaded_uris = client.download_output_prefix(
        output_gcs_uri_prefix=output_prefix,
        local_dir=local_dir,
        only_jsonl=not bool(args.include_non_jsonl),
    )
    summary = client.summarize_downloaded_jsonl(local_dir)

    print(json.dumps({
        "job_name": job.get("name"),
        "job_id": client.to_job_id(str(job.get("name", "")).strip()) if job.get("name") else None,
        "state": job.get("state"),
        "output_gcs_uri_prefix": output_prefix,
        "downloaded_local_dir": str(local_dir),
        "downloaded_uris": downloaded_uris,
        "download_summary": summary,
    }, ensure_ascii=False, indent=2))


def run_user_pipeline_command(args: argparse.Namespace) -> None:
    if (args.repair_model or "").strip() != FIXED_REPAIR_MODEL:
        raise ValueError(f"--repair-model must be {FIXED_REPAIR_MODEL}")
    manifest = run_user_pipeline(
        dataset_root=args.dataset_root,
        out_root=args.out_root,
        user_id=args.user_id,
        run_tag=(args.run_tag or "").strip() or None,
        domains_config=args.domains_config,
        model_config=args.model_config,
        domain_model_config=args.domain_model_config,
        tag_aliases=args.tag_aliases,
        domain_model=args.domain_model,
        benchmark_rewrite_model=args.benchmark_rewrite_model,
        repair_model=args.repair_model,
        dialogue_model=args.dialogue_model,
        max_posts=args.max_posts,
        force=bool(args.force),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "aggregate-posts":
        run_aggregate_posts(args)
        return

    if args.command == "profile-post":
        run_profile_post(args)
        return

    if args.command == "summarize-domains":
        run_summarize_domains(args)
        return

    if args.command == "vertex-batch-submit":
        run_vertex_batch_submit(args)
        return

    if args.command == "vertex-batch-status":
        run_vertex_batch_status(args)
        return

    if args.command == "vertex-batch-download":
        run_vertex_batch_download(args)
        return

    if args.command == "run-user-pipeline":
        run_user_pipeline_command(args)


if __name__ == "__main__":
    main()
