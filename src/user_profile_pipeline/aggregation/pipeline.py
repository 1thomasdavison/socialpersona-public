from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..schemas import DOMAIN_NAMES, PostObservable, SinglePostProfile
from .dedup import NearDuplicateClusterer
from .representative import chunk_records, select_representative_posts
from .scoring import TemporalScorer
from .tag_cluster import TagClusterBuilder


@dataclass
class MultiPostAggregator:
    tag_aliases: dict[str, dict[str, str]] | None = None
    duplicate_clusterer: NearDuplicateClusterer | None = None
    scorer: TemporalScorer | None = None
    tag_cluster_builder: TagClusterBuilder | None = None

    def __post_init__(self) -> None:
        self.tag_aliases = self.tag_aliases or {}
        self.duplicate_clusterer = self.duplicate_clusterer or NearDuplicateClusterer()
        self.scorer = self.scorer or TemporalScorer()
        self.tag_cluster_builder = self.tag_cluster_builder or TagClusterBuilder(aliases=self.tag_aliases)

    @classmethod
    def from_alias_config(cls, path: str | Path | None) -> "MultiPostAggregator":
        aliases = load_yaml(path) if path else {}
        return cls(tag_aliases=aliases or {})

    def aggregate(self, posts: list[PostObservable], single_post_profiles: list[dict[str, Any] | SinglePostProfile]) -> dict[str, Any]:
        post_by_id = {p.post_id: p for p in posts}
        profile_rows = [self._coerce_profile_row(row) for row in single_post_profiles]
        filtered_profiles = [
            row for row in profile_rows
            if row["post_id"] in post_by_id and not row["profile"].is_noise and row["profile"].should_use_for_profile
        ]
        kept_posts = [post_by_id[row["post_id"]] for row in filtered_profiles]
        dup_clusters = self.duplicate_clusterer.cluster(kept_posts)
        dup_weight_by_post = {}
        for cluster in dup_clusters:
            size = max(1, len(cluster.member_post_ids))
            for post_id in cluster.member_post_ids:
                dup_weight_by_post[post_id] = 1.0 / size
        if not filtered_profiles:
            observed = self._observation_window(posts)
            return {
                "schema_version": "multi_post_aggregate_v1",
                "observation_window": observed,
                "n_input_posts": len(posts),
                "n_kept_posts": 0,
                "duplicate_clusters": [],
                "tag_clusters": [],
                "domains": [],
            }

        observed = self._observation_window([post_by_id[row["post_id"]] for row in filtered_profiles])
        observed_end = datetime.fromisoformat(observed["observed_end"].replace("Z", "+00:00"))
        window_days = observed["window_days"]

        domain_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        tag_obs_records: list[tuple[str, str, float, Any]] = []

        for row in filtered_profiles:
            profile = row["profile"]
            post = post_by_id[row["post_id"]]
            dup_weight = dup_weight_by_post.get(post.post_id, 1.0)
            created_at = post.created_at.isoformat().replace("+00:00", "Z") if post.created_at.tzinfo else post.created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            for domain_result in profile.domains:
                evidence_modalities = sorted(set(domain_result.evidence_modalities or []))
                evidence_modality = self._classify_evidence_modality(evidence_modalities)
                domain_records[domain_result.domain].append(
                    {
                        "post_id": post.post_id,
                        "created_at": post.created_at if post.created_at.tzinfo else post.created_at.replace(tzinfo=timezone.utc),
                        "confidence": domain_result.confidence,
                        "dup_weight": dup_weight,
                        "post_signal_strength": profile.post_signal_strength,
                        "text": post.text,
                        "raw_tags": [t.tag for t in domain_result.tags],
                        "evidence_modalities": evidence_modalities,
                        "evidence_modality": evidence_modality,
                        "evidence_source_breakdown": dict(domain_result.evidence_source_breakdown or {}),
                    }
                )
                tag_obs_records.append((post.post_id, created_at, dup_weight, domain_result))

        tag_observations = self.tag_cluster_builder.collect_observations(tag_obs_records)
        tag_clusters = self.tag_cluster_builder.build(tag_observations)
        canonical_tags_by_post_domain: dict[tuple[str, str], list[str]] = defaultdict(list)
        for cluster in tag_clusters:
            for post_id in cluster.evidence_post_ids:
                canonical_tags_by_post_domain[(post_id, cluster.domain)].append(cluster.canonical_tag)

        domain_outputs = []
        cluster_source_summaries: dict[tuple[str, str], dict[str, Any]] = {}
        for domain in DOMAIN_NAMES:
            records = domain_records.get(domain, [])
            stats = self.scorer.compute_domain_stats(domain, records, window_days, observed_end)
            for record in records:
                record["canonical_tags"] = sorted(set(canonical_tags_by_post_domain.get((record["post_id"], domain), [])))

            tag_stats = []
            domain_tag_clusters = [c for c in tag_clusters if c.domain == domain]
            for cluster in domain_tag_clusters:
                cluster_records = [
                    {
                        "post_id": record["post_id"],
                        "created_at": record["created_at"],
                        "confidence": next((obs.confidence for obs in tag_observations if obs.post_id == record["post_id"] and obs.domain == domain and obs.normalized_tag in {m for m in cluster.members}), 0.0),
                        "dup_weight": record["dup_weight"],
                        "evidence_modalities": list(record.get("evidence_modalities", []) or []),
                        "evidence_source_breakdown": dict(record.get("evidence_source_breakdown", {}) or {}),
                    }
                    for record in records
                    if record["post_id"] in set(cluster.evidence_post_ids)
                ]
                if not cluster_records:
                    continue
                tag_score = self.scorer.compute_interest_stats(cluster.canonical_tag, cluster_records, window_days, observed_end)
                source_summary = self._build_source_summary(cluster_records)
                cluster_source_summaries[(domain, cluster.canonical_tag)] = source_summary
                tag_stats.append(
                    {
                        "interest": cluster.canonical_tag,
                        "members": cluster.members,
                        "n_posts": cluster.n_posts,
                        "weighted_support": cluster.weighted_support,
                        "mean_confidence": cluster.mean_confidence,
                        "evidence_post_ids": cluster.evidence_post_ids,
                        "stable_score": tag_score.stable_score,
                        "recent_score": tag_score.recent_score,
                        "status": tag_score.status,
                        "distinct_time_bins": tag_score.distinct_time_bins,
                        "evidence_modalities": source_summary["modalities"],
                        "evidence_source_breakdown": source_summary["breakdown"],
                    }
                )

            representatives = select_representative_posts(records, limit=15)
            domain_source_summary = self._build_source_summary(records)
            chunks = chunk_records(records, chunk_size=30)
            chunk_summaries = [
                {
                    "chunk_index": idx,
                    "start_post_id": chunk[0]["post_id"],
                    "end_post_id": chunk[-1]["post_id"],
                    "n_posts": len(chunk),
                    "top_tags": self._top_tags(chunk),
                }
                for idx, chunk in enumerate(chunks)
            ]
            domain_outputs.append(
                {
                    "domain": domain,
                    "support_posts": stats.support_posts,
                    "weak_posts": stats.weak_posts,
                    "effective_support_posts": stats.effective_support_posts,
                    "mean_confidence": stats.mean_confidence,
                    "distinct_time_bins": stats.distinct_time_bins,
                    "recent_90_support": stats.recent_90_support,
                    "recent_30_support": stats.recent_30_support,
                    "recent_90_mean_confidence": stats.recent_90_mean_confidence,
                    "stable_score": stats.stable_score,
                    "recent_score": stats.recent_score,
                    "status": stats.status,
                    "evidence_modalities": domain_source_summary["modalities"],
                    "evidence_source_breakdown": domain_source_summary["breakdown"],
                    "post_signal_sources": self._build_post_signal_sources(records),
                    "stable_interests": [x for x in tag_stats if x["status"] == "stable"],
                    "short_term_interests": [x for x in tag_stats if x["status"] == "recent"],
                    "weak_or_uncertain_interests": [x for x in tag_stats if x["status"] == "weak_signal"],
                    "representative_posts": [self._serialize_record(r) for r in representatives],
                    "chunk_summaries": chunk_summaries,
                }
            )

        return {
            "schema_version": "multi_post_aggregate_v1",
            "observation_window": observed,
            "n_input_posts": len(posts),
            "n_kept_posts": len(filtered_profiles),
            "duplicate_clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "member_post_ids": cluster.member_post_ids,
                    "duplicate_cluster_size": len(cluster.member_post_ids),
                    "dup_weight": round(1.0 / max(1, len(cluster.member_post_ids)), 6),
                }
                for cluster in dup_clusters
            ],
            "tag_clusters": [
                {
                    "domain": cluster.domain,
                    "canonical_tag": cluster.canonical_tag,
                    "members": cluster.members,
                    "n_posts": cluster.n_posts,
                    "weighted_support": cluster.weighted_support,
                    "mean_confidence": cluster.mean_confidence,
                    "evidence_post_ids": cluster.evidence_post_ids,
                    "evidence_modalities": cluster_source_summaries.get((cluster.domain, cluster.canonical_tag), {}).get("modalities", []),
                    "evidence_source_breakdown": cluster_source_summaries.get((cluster.domain, cluster.canonical_tag), {}).get("breakdown", {}),
                }
                for cluster in tag_clusters
            ],
            "domains": domain_outputs,
        }

    def aggregate_from_paths(self, posts_path: str | Path, profiles_path: str | Path) -> dict[str, Any]:
        posts = self._load_jsonl_posts(posts_path)
        profiles = self._load_jsonl(profiles_path)
        return self.aggregate(posts, profiles)

    def _coerce_profile_row(self, row: dict[str, Any] | SinglePostProfile) -> dict[str, Any]:
        if isinstance(row, SinglePostProfile):
            raise ValueError("SinglePostProfile instances require a wrapping dict with post_id for aggregation.")
        post_id = row.get("post_id") or row.get("_post_id")
        profile_payload = row.get("profile", row)
        profile = SinglePostProfile.model_validate(profile_payload)
        if not post_id:
            raise ValueError("Each profile row must include post_id or _post_id for aggregation.")
        return {"post_id": post_id, "profile": profile}

    @staticmethod
    def _observation_window(posts: list[PostObservable]) -> dict[str, Any]:
        if not posts:
            return {
                "post_count_total": 0,
                "observed_start": None,
                "observed_end": None,
                "window_days": 0,
            }
        ordered = sorted(posts, key=lambda p: p.created_at)
        start = ordered[0].created_at if ordered[0].created_at.tzinfo else ordered[0].created_at.replace(tzinfo=timezone.utc)
        end = ordered[-1].created_at if ordered[-1].created_at.tzinfo else ordered[-1].created_at.replace(tzinfo=timezone.utc)
        return {
            "post_count_total": len(posts),
            "observed_start": start.isoformat().replace("+00:00", "Z"),
            "observed_end": end.isoformat().replace("+00:00", "Z"),
            "window_days": max(1, (end - start).days),
        }

    @staticmethod
    def _load_jsonl_posts(path: str | Path) -> list[PostObservable]:
        rows = MultiPostAggregator._load_jsonl(path)
        return [PostObservable.model_validate(row) for row in rows]

    @staticmethod
    def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _top_tags(chunk: list[dict[str, Any]]) -> list[str]:
        counts: dict[str, int] = defaultdict(int)
        for row in chunk:
            for tag in row.get("canonical_tags", []):
                counts[tag] += 1
        return [tag for tag, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:5]]

    @staticmethod
    def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            **record,
            "created_at": record["created_at"].isoformat().replace("+00:00", "Z"),
        }

    @staticmethod
    def _classify_evidence_modality(evidence_modalities: list[str]) -> str:
        modalities = sorted({str(x).strip() for x in evidence_modalities if str(x).strip()})
        if modalities == ["text"]:
            return "text_only"
        if modalities == ["visual"]:
            return "visual_only"
        if modalities:
            return "mixed"
        return "unknown"

    def _build_source_summary(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        modalities: list[str] = []
        breakdown = {
            "text_only_post_ids": [],
            "visual_only_post_ids": [],
            "mixed_post_ids": [],
            "unknown_post_ids": [],
        }
        for record in records:
            for modality in record.get("evidence_modalities", []) or []:
                token = str(modality).strip()
                if token and token not in modalities:
                    modalities.append(token)

            post_id = str(record.get("post_id") or "").strip()
            if not post_id:
                continue
            bucket = f"{self._classify_evidence_modality(record.get('evidence_modalities', []))}_post_ids"
            if bucket not in breakdown:
                bucket = "unknown_post_ids"
            if post_id not in breakdown[bucket]:
                breakdown[bucket].append(post_id)

        return {
            "modalities": modalities,
            "breakdown": {key: value for key, value in breakdown.items() if value},
        }

    def _build_post_signal_sources(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            post_id = str(record.get("post_id") or "").strip()
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            rows.append(
                {
                    "post_id": post_id,
                    "evidence_modalities": list(record.get("evidence_modalities", []) or []),
                    "evidence_modality": self._classify_evidence_modality(record.get("evidence_modalities", [])),
                    "evidence_source_breakdown": dict(record.get("evidence_source_breakdown", {}) or {}),
                    "confidence": float(record.get("confidence", 0.0) or 0.0),
                }
            )
        return rows
