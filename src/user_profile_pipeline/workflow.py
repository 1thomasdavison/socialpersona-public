from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson
from .aggregation import MultiPostAggregator
from .config import load_domain_configs, load_yaml
from .domain_llm import BenchmarkGoldExporter, DomainLLMSummarizer, DomainPackBuilder
from .gate import NarrowGate
from .llm_client import OpenAICompatibleChatClient
from .personalized_dialogue import PersonalizedDialogueEvaluator, build_personalized_dialogue_artifacts
from .schemas import PostObservable, SinglePostProfile
from .single_post import SinglePostAnalyzer


@dataclass
class UserPipelinePaths:
    user_id: str
    user_dir: Path
    posts_jsonl: Path
    metadata_json: Path | None
    run_dir: Path
    single_post_dir: Path
    single_post_profiles_jsonl: Path
    single_post_profiles_visual_only_jsonl: Path
    gate_results_jsonl: Path
    single_post_summary_json: Path
    aggregate_json: Path
    aggregate_visual_only_json: Path
    domain_dir: Path
    domain_packs_jsonl: Path
    domain_packs_visual_only_jsonl: Path
    domain_summary_json: Path
    domain_summary_visual_only_json: Path
    benchmark_gold_json: Path
    benchmark_gold_visual_only_json: Path
    dialogue_dir: Path
    dialogue_gold_ref_json: Path
    dialogue_eval_task_json: Path
    dialogue_prediction_json: Path
    dialogue_metrics_json: Path
    manifest_json: Path


def resolve_user_pipeline_paths(
    *,
    dataset_root: str | Path,
    out_root: str | Path,
    user_id: str,
    run_tag: str | None = None,
) -> UserPipelinePaths:
    dataset_root_path = Path(dataset_root).resolve()
    user_dir = dataset_root_path / user_id
    posts_jsonl = user_dir / "posts.jsonl"
    metadata_json = user_dir / "metadata.json"
    effective_run_tag = (run_tag or datetime.utcnow().strftime("%Y%m%d_%H%M%S")).strip()
    run_dir = Path(out_root).resolve() / user_id / effective_run_tag
    single_post_dir = run_dir / "single_post_analysis"
    domain_dir = run_dir / "domain_llm_batch"
    dialogue_dir = run_dir / "personalized_dialogue"
    return UserPipelinePaths(
        user_id=user_id,
        user_dir=user_dir,
        posts_jsonl=posts_jsonl,
        metadata_json=metadata_json if metadata_json.exists() else None,
        run_dir=run_dir,
        single_post_dir=single_post_dir,
        single_post_profiles_jsonl=single_post_dir / "profiles_for_aggregate.jsonl",
        single_post_profiles_visual_only_jsonl=single_post_dir / "profiles_for_aggregate_visual_only.jsonl",
        gate_results_jsonl=single_post_dir / "gate_results.jsonl",
        single_post_summary_json=single_post_dir / "single_post_batch_summary.json",
        aggregate_json=run_dir / "multi_post_aggregate.json",
        aggregate_visual_only_json=run_dir / "multi_post_aggregate_visual_only.json",
        domain_dir=domain_dir,
        domain_packs_jsonl=domain_dir / "domain_packs.jsonl",
        domain_packs_visual_only_jsonl=domain_dir / "domain_packs_visual_only.jsonl",
        domain_summary_json=domain_dir / "domain_llm_summary_batch.json",
        domain_summary_visual_only_json=domain_dir / "domain_llm_summary_batch_visual_only.json",
        benchmark_gold_json=domain_dir / "benchmark_gold_export.json",
        benchmark_gold_visual_only_json=domain_dir / "benchmark_gold_export_visual_only.json",
        dialogue_dir=dialogue_dir,
        dialogue_gold_ref_json=dialogue_dir / "personalized_dialogue_gold_ref.json",
        dialogue_eval_task_json=dialogue_dir / "personalized_dialogue_eval_task.json",
        dialogue_prediction_json=dialogue_dir / "prediction.json",
        dialogue_metrics_json=dialogue_dir / "metrics.json",
        manifest_json=run_dir / "run_manifest.json",
    )


def run_user_pipeline(
    *,
    dataset_root: str | Path,
    out_root: str | Path,
    user_id: str,
    run_tag: str | None = None,
    domains_config: str = "configs/domains.yaml",
    model_config: str = "configs/model.yaml",
    domain_model_config: str = "configs/model.domain.yaml",
    tag_aliases: str = "configs/tag_aliases.yaml",
    domain_model: str = "gpt-5.4",
    benchmark_rewrite_model: str = "gpt-5.4",
    repair_model: str = "gpt-5.4",
    dialogue_model: str = "mock_oracle",
    max_posts: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    paths = resolve_user_pipeline_paths(
        dataset_root=dataset_root,
        out_root=out_root,
        user_id=user_id,
        run_tag=run_tag,
    )
    if not paths.posts_jsonl.exists():
        raise FileNotFoundError(f"posts.jsonl not found for user: {paths.posts_jsonl}")

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    posts = _load_posts(paths.posts_jsonl, max_posts=max_posts)
    if not posts:
        raise ValueError(f"No posts loaded from {paths.posts_jsonl}")
    single_post_summary = _run_single_post_stage(
        posts=posts,
        paths=paths,
        domains_config=domains_config,
        model_config=model_config,
        force=force,
    )
    aggregate_result = _run_aggregate_stage(
        paths=paths,
        profiles_path=paths.single_post_profiles_jsonl,
        out_path=paths.aggregate_json,
        tag_aliases=tag_aliases,
        force=force,
    )
    aggregate_result_visual_only = _run_aggregate_stage(
        paths=paths,
        profiles_path=paths.single_post_profiles_visual_only_jsonl,
        out_path=paths.aggregate_visual_only_json,
        tag_aliases=tag_aliases,
        force=force,
    )
    domain_summary = _run_domain_stage(
        posts=posts,
        paths=paths,
        aggregate_result=aggregate_result,
        domain_packs_path=paths.domain_packs_jsonl,
        domain_summary_path=paths.domain_summary_json,
        benchmark_gold_path=paths.benchmark_gold_json,
        batch_manifest_path=paths.domain_dir / "batch_manifest.json",
        variant_label="full",
        force=force,
        domain_model_config=domain_model_config,
        domain_model=domain_model,
        benchmark_rewrite_model=benchmark_rewrite_model,
        repair_model=repair_model,
    )
    domain_summary_visual_only = _run_domain_stage(
        posts=posts,
        paths=paths,
        aggregate_result=aggregate_result_visual_only,
        domain_packs_path=paths.domain_packs_visual_only_jsonl,
        domain_summary_path=paths.domain_summary_visual_only_json,
        benchmark_gold_path=paths.benchmark_gold_visual_only_json,
        batch_manifest_path=paths.domain_dir / "batch_manifest_visual_only.json",
        variant_label="visual_only",
        domain_model_config=domain_model_config,
        domain_model=domain_model,
        benchmark_rewrite_model=benchmark_rewrite_model,
        repair_model=repair_model,
        force=force,
    )
    dialogue_outputs = _run_dialogue_stage(
        paths=paths,
        model_config=model_config,
        dialogue_model=dialogue_model,
        force=force,
    )

    manifest = {
        "schema_version": "user_pipeline_run_v1",
        "user_id": user_id,
        "dataset_root": str(Path(dataset_root).resolve()),
        "run_dir": str(paths.run_dir),
        "posts_jsonl": str(paths.posts_jsonl),
        "metadata_json": str(paths.metadata_json) if paths.metadata_json else None,
        "single_post_summary": single_post_summary,
        "aggregate_summary": {
            "aggregate_json": str(paths.aggregate_json),
            "aggregate_visual_only_json": str(paths.aggregate_visual_only_json),
            "n_input_posts": aggregate_result.get("n_input_posts"),
            "n_kept_posts": aggregate_result.get("n_kept_posts"),
            "n_kept_posts_visual_only": aggregate_result_visual_only.get("n_kept_posts"),
            "n_domains": len(aggregate_result.get("domains", []) or []),
        },
        "domain_summary": {
            "domain_summary_json": str(paths.domain_summary_json),
            "benchmark_gold_json": str(paths.benchmark_gold_json),
            "domain_summary_visual_only_json": str(paths.domain_summary_visual_only_json),
            "benchmark_gold_visual_only_json": str(paths.benchmark_gold_visual_only_json),
            "n_domain_summaries": domain_summary.get("n_domain_summaries"),
            "n_errors": domain_summary.get("n_errors"),
            "n_domain_summaries_visual_only": domain_summary_visual_only.get("n_domain_summaries"),
            "n_errors_visual_only": domain_summary_visual_only.get("n_errors"),
        },
        "dialogue_summary": dialogue_outputs,
    }
    paths.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _run_single_post_stage(
    *,
    posts: list[PostObservable],
    paths: UserPipelinePaths,
    domains_config: str,
    model_config: str,
    force: bool,
) -> dict[str, Any]:
    paths.single_post_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = paths.single_post_profiles_jsonl
    profiles_visual_only_path = paths.single_post_profiles_visual_only_jsonl
    gates_path = paths.gate_results_jsonl
    summary_path = paths.single_post_summary_json
    if (
        summary_path.exists()
        and profiles_path.exists()
        and profiles_visual_only_path.exists()
        and gates_path.exists()
        and not force
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    model_cfg = load_yaml(model_config)
    client = OpenAICompatibleChatClient(
        provider=model_cfg["provider"],
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        api_key_env=model_cfg["api_key_env"],
        timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        max_tokens=int(model_cfg.get("max_tokens", 1200)),
    )
    gate = NarrowGate(domain_configs=load_domain_configs(domains_config))
    analyzer = SinglePostAnalyzer(
        gate=gate,
        client=client,
        media_base_dir=paths.posts_jsonl.parent,
    )

    profile_rows: list[dict[str, Any]] = []
    profile_rows_visual_only: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for post in posts:
        gate_result = None
        try:
            gate_result, profile = analyzer.analyze(post)
        except Exception as exc:
            gate_result = gate.run(post)
            profile = _build_fallback_profile(
                gate_result=gate_result,
                reason=f"single_post_analysis_failed: {exc}",
            )
            errors.append({"post_id": post.post_id, "error": str(exc), "used_fallback": True})

        try:
            _, profile_visual_only = analyzer.analyze(post, evidence_mode="visual_only")
        except Exception as exc:
            if gate_result is None:
                gate_result = gate.run(post)
            profile_visual_only = _build_fallback_profile(
                gate_result=gate_result,
                reason=f"single_post_analysis_visual_only_failed: {exc}",
            )
            errors.append({"post_id": post.post_id, "error": str(exc), "stage": "visual_only", "used_fallback": True})

        profile_rows.append({"post_id": post.post_id, "profile": json.loads(profile.model_dump_json())})
        profile_rows_visual_only.append({"post_id": post.post_id, "profile": json.loads(profile_visual_only.model_dump_json())})
        gate_rows.append({"post_id": post.post_id, "gate_result": json.loads(gate_result.model_dump_json())})

    _write_jsonl(profiles_path, profile_rows)
    _write_jsonl(profiles_visual_only_path, profile_rows_visual_only)
    _write_jsonl(gates_path, gate_rows)

    summary = {
        "schema_version": "single_post_local_batch_v1",
        "user_id": paths.user_id,
        "profiles_for_aggregate_path": str(profiles_path),
        "profiles_for_aggregate_visual_only_path": str(profiles_visual_only_path),
        "gate_results_path": str(gates_path),
        "n_posts_total": len(posts),
        "n_profiles": len(profile_rows),
        "n_profiles_visual_only": len(profile_rows_visual_only),
        "n_errors": len(errors),
        "errors": errors,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _run_aggregate_stage(
    *,
    paths: UserPipelinePaths,
    profiles_path: Path,
    out_path: Path,
    tag_aliases: str,
    force: bool,
) -> dict[str, Any]:
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    aggregator = MultiPostAggregator.from_alias_config(tag_aliases)
    result = aggregator.aggregate_from_paths(
        paths.posts_jsonl,
        profiles_path,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _run_domain_stage(
    *,
    posts: list[PostObservable],
    paths: UserPipelinePaths,
    aggregate_result: dict[str, Any],
    domain_packs_path: Path,
    domain_summary_path: Path,
    benchmark_gold_path: Path,
    batch_manifest_path: Path,
    variant_label: str,
    domain_model_config: str,
    domain_model: str,
    benchmark_rewrite_model: str,
    repair_model: str,
    force: bool,
) -> dict[str, Any]:
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    if domain_summary_path.exists() and benchmark_gold_path.exists() and not force:
        return json.loads(domain_summary_path.read_text(encoding="utf-8"))

    post_id_to_index = {post.post_id: idx for idx, post in enumerate(posts) if post.post_id}
    pack_builder = DomainPackBuilder(include_absent=True)
    domain_packs = pack_builder.build_packs(aggregate_result=aggregate_result, posts=posts)
    _write_jsonl(domain_packs_path, domain_packs)

    model_cfg = load_yaml(domain_model_config)
    summarizer = DomainLLMSummarizer(
        client=OpenAICompatibleChatClient(
            provider=model_cfg["provider"],
            base_url=model_cfg["base_url"],
            model=domain_model or model_cfg["model"],
            api_key_env=model_cfg["api_key_env"],
            timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
            temperature=float(model_cfg.get("temperature", 0.1)),
            max_tokens=int(model_cfg.get("max_tokens", 1200)),
            repair_model=repair_model,
        ),
        media_base_dir=paths.posts_jsonl.parent,
    )

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for pack in domain_packs:
        domain = str(pack.get("domain", "")).strip()
        try:
            summaries.append(summarizer.summarize_domain_pack(pack))
        except Exception as exc:
            errors.append({"domain": domain, "error": str(exc), "used_fallback": True})
            summaries.append(summarizer.validator.validate({}, pack))

    final_batch = {
        "schema_version": "domain_llm_batch_v1",
        "user_id": paths.user_id,
        "observation_window": aggregate_result.get("observation_window"),
        "n_domain_packs": len(domain_packs),
        "n_domain_summaries": len(summaries),
        "n_errors": len(errors),
        "domain_summaries": summaries,
        "errors": errors,
        "batch_meta": {
            "execution_mode": "local_direct",
            "domain_model": domain_model,
            "benchmark_rewrite_model": benchmark_rewrite_model,
            "repair_model": repair_model,
        },
    }
    domain_summary_path.write_text(json.dumps(final_batch, ensure_ascii=False, indent=2), encoding="utf-8")

    benchmark_rewrite_error = None
    try:
        rewrite_client = OpenAICompatibleChatClient(
            provider=model_cfg["provider"],
            base_url=model_cfg["base_url"],
            model=benchmark_rewrite_model or model_cfg["model"],
            api_key_env=model_cfg["api_key_env"],
            timeout_seconds=int(model_cfg.get("timeout_seconds", 60)),
            temperature=float(model_cfg.get("temperature", 0.1)),
            max_tokens=int(model_cfg.get("max_tokens", 1200)),
            repair_model=repair_model,
        )
        benchmark_export = BenchmarkGoldExporter(client=rewrite_client).export_user_batch(
            batch_result=final_batch,
            domain_packs=domain_packs,
            post_id_to_index=post_id_to_index,
            rewrite_enabled=True,
        )
    except Exception as exc:
        benchmark_rewrite_error = str(exc)
        benchmark_export = BenchmarkGoldExporter(client=None).export_user_batch(
            batch_result=final_batch,
            domain_packs=domain_packs,
            post_id_to_index=post_id_to_index,
            rewrite_enabled=False,
        )

    benchmark_gold_path.write_text(json.dumps(benchmark_export, ensure_ascii=False, indent=2), encoding="utf-8")
    batch_manifest_path.write_text(
        json.dumps(
            {
                "execution_mode": "local_direct",
                "variant": variant_label,
                "domain_packs_path": str(domain_packs_path),
                "final_summary_path": str(domain_summary_path),
                "benchmark_export_path": str(benchmark_gold_path),
                "benchmark_rewrite_error": benchmark_rewrite_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return final_batch


def _run_dialogue_stage(
    *,
    paths: UserPipelinePaths,
    model_config: str,
    dialogue_model: str,
    force: bool,
) -> dict[str, Any]:
    paths.dialogue_dir.mkdir(parents=True, exist_ok=True)
    if (
        paths.dialogue_gold_ref_json.exists()
        and paths.dialogue_eval_task_json.exists()
        and paths.dialogue_prediction_json.exists()
        and paths.dialogue_metrics_json.exists()
        and not force
    ):
        metrics = json.loads(paths.dialogue_metrics_json.read_text(encoding="utf-8"))
        return {
            "dialogue_gold_ref_json": str(paths.dialogue_gold_ref_json),
            "dialogue_eval_task_json": str(paths.dialogue_eval_task_json),
            "dialogue_prediction_json": str(paths.dialogue_prediction_json),
            "dialogue_metrics_json": str(paths.dialogue_metrics_json),
            "dialogue_model": dialogue_model,
            "overall_score": ((metrics.get("overall") or {}).get("overall_score")),
        }

    benchmark_gold = json.loads(paths.benchmark_gold_json.read_text(encoding="utf-8"))
    gold_ref, eval_task = build_personalized_dialogue_artifacts(
        gold_export=benchmark_gold,
        posts_jsonl=paths.posts_jsonl,
    )
    gold_ref["source"]["gold_export_path"] = str(paths.benchmark_gold_json)
    paths.dialogue_gold_ref_json.write_text(json.dumps(gold_ref, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.dialogue_eval_task_json.write_text(json.dumps(eval_task, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluator = PersonalizedDialogueEvaluator(
        model_config_path=model_config,
        model_name=dialogue_model,
    )
    prediction, metrics = evaluator.run(gold_ref=gold_ref, eval_task=eval_task)
    paths.dialogue_prediction_json.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.dialogue_metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "dialogue_gold_ref_json": str(paths.dialogue_gold_ref_json),
        "dialogue_eval_task_json": str(paths.dialogue_eval_task_json),
        "dialogue_prediction_json": str(paths.dialogue_prediction_json),
        "dialogue_metrics_json": str(paths.dialogue_metrics_json),
        "dialogue_model": dialogue_model,
        "overall_score": ((metrics.get("overall") or {}).get("overall_score")),
    }


def _load_posts(posts_jsonl: Path, *, max_posts: int | None) -> list[PostObservable]:
    posts: list[PostObservable] = []
    with posts_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            posts.append(PostObservable.model_validate(orjson.loads(line)))
            if max_posts is not None and len(posts) >= max_posts:
                break
    return posts


def _build_fallback_profile(*, gate_result: Any, reason: str) -> SinglePostProfile:
    if gate_result.is_noise:
        return SinglePostProfile(
            is_noise=True,
            should_use_for_profile=False,
            noise_reason=gate_result.noise_reason,
            post_signal_strength=0.0,
            domains=[],
            uncertainty_note="Skipped by narrow gate.",
        )
    return SinglePostProfile(
        is_noise=False,
        should_use_for_profile=False,
        noise_reason=None,
        post_signal_strength=0.0,
        domains=[],
        uncertainty_note=reason,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
