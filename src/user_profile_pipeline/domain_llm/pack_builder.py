from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..schemas import DOMAIN_NAMES, PostObservable


DOMAIN_DEFINITIONS: dict[str, str] = {
    "sports_outdoor": "sports participation, exercise, fitness routines, hiking, running, cycling, camping, outdoor recreation, and active-use sports gear.",
    "entertainment": "movies, TV, music, concerts, books/comics/anime, celebrities, and media consumption.",
    "gaming": "video games, gaming hardware/platforms, esports, game fandom, game streaming, and playing/watching games.",
    "food_drink": "cooking, meals, restaurants, cafes, recipes, coffee, tea, cocktails, and other food/drink consumption or creation.",
    "travel_city_exploration": "trips, flights, hotels, cities, neighborhoods, sightseeing, landmarks, museums, and city walks/exploration.",
    "photography_creation": "taking photos, cameras, lenses, editing, visual creation, making images/videos/artworks.",
    "pets": "pets, pet ownership, pet care, dogs, cats, training, grooming, adoption, veterinary care, pet products, and spending time with companion animals.",
}

TAG_DISPLAY_OVERRIDES: dict[str, str] = {
    "retro_gam": "retro gaming",
    "pc_gam": "PC gaming",
    "gam_hardware": "gaming hardware",
    "science_book": "science books",
    "movie_watch": "movies",
    "homebrew": "homebrewing",
    "craft_beer": "craft beer",
    "physical_media": "physical game media",
    "din_out": "dining out",
    "caffeinat_drink": "caffeinated drinks",
    "healthy_eat": "healthy eating",
    "restaurant_meal": "restaurant meals",
}


@dataclass
class DomainPackBuilder:
    include_absent: bool = True

    def build_packs(
        self,
        *,
        aggregate_result: dict[str, Any],
        posts: list[PostObservable],
    ) -> list[dict[str, Any]]:
        post_by_id = {p.post_id: p for p in posts}
        user_id = posts[0].user_id if posts else None
        observation_window = aggregate_result.get("observation_window", {})
        observed_end = self._parse_utc(observation_window.get("observed_end"))
        domain_rows = aggregate_result.get("domains", [])
        tag_clusters = aggregate_result.get("tag_clusters", [])
        packs: list[dict[str, Any]] = []

        for domain_name in DOMAIN_NAMES:
            domain_row = next((d for d in domain_rows if d.get("domain") == domain_name), None)
            if not domain_row:
                domain_row = self._default_domain_row(domain_name)

            status = str(domain_row.get("status", "absent"))
            domain_present = status != "absent"
            if not self.include_absent and not domain_present:
                continue

            support_posts = [str(x) for x in domain_row.get("support_posts", [])]
            weak_posts = [str(x) for x in domain_row.get("weak_posts", [])]
            domain_post_count = len(set(support_posts + weak_posts))

            interest_stats_by_name = self._collect_interest_stats(domain_row)
            domain_tag_clusters = [
                self._build_tag_cluster_entry(cluster, interest_stats_by_name, post_by_id)
                for cluster in tag_clusters
                if cluster.get("domain") == domain_name
            ]

            representative_posts = self._build_representative_posts(
                domain_row.get("representative_posts", []),
                post_by_id,
                observed_end,
            )
            chunk_summaries = self._build_chunk_summaries(
                domain_name=domain_name,
                chunk_rows=domain_row.get("chunk_summaries", []),
                post_by_id=post_by_id,
            )

            stable_score = float(domain_row.get("stable_score", 0.0))
            recent_score = float(domain_row.get("recent_score", 0.0))
            pack = {
                "schema_version": "domain_pack_v1",
                "user_id": user_id,
                "domain": domain_name,
                "domain_definition": DOMAIN_DEFINITIONS.get(domain_name, ""),
                "profiling_target": {
                    "profile_type": "interest",
                    "natural_language_priority": True,
                    "exclude_attributes": ["family", "career", "age_range"],
                    "downstream_use": "llm_personalization_benchmark",
                },
                "observation_window": {
                    "observed_start": observation_window.get("observed_start"),
                    "observed_end": observation_window.get("observed_end"),
                    "window_days": int(observation_window.get("window_days", 0) or 0),
                    "post_count_total": int(observation_window.get("post_count_total", 0) or 0),
                    "post_count_in_domain_after_filtering": domain_post_count,
                },
                "algorithm_summary": {
                    "domain_present": domain_present,
                    "stable_score": stable_score,
                    "recent_score": recent_score,
                    "status": status,
                    "support_posts": len(support_posts),
                    "weak_posts": len(weak_posts),
                    "effective_support_posts": float(domain_row.get("effective_support_posts", 0.0)),
                    "distinct_time_bins": int(domain_row.get("distinct_time_bins", 0) or 0),
                    "recent_90_support": float(domain_row.get("recent_90_support", 0.0)),
                    "recent_30_support": float(domain_row.get("recent_30_support", 0.0)),
                    "mean_confidence": float(domain_row.get("mean_confidence", 0.0)),
                    "evidence_modalities": list(domain_row.get("evidence_modalities", []) or []),
                    "evidence_source_breakdown": dict(domain_row.get("evidence_source_breakdown", {}) or {}),
                },
                "post_signal_sources": list(domain_row.get("post_signal_sources", []) or []),
                "tag_clusters": domain_tag_clusters,
                "representative_posts": representative_posts,
                "chunk_summaries": chunk_summaries,
            }
            packs.append(pack)

        return packs

    @staticmethod
    def _default_domain_row(domain_name: str) -> dict[str, Any]:
        return {
            "domain": domain_name,
            "support_posts": [],
            "weak_posts": [],
            "effective_support_posts": 0.0,
            "mean_confidence": 0.0,
            "distinct_time_bins": 0,
            "recent_90_support": 0.0,
            "recent_30_support": 0.0,
            "stable_score": 0.0,
            "recent_score": 0.0,
            "status": "absent",
            "evidence_modalities": [],
            "evidence_source_breakdown": {},
            "post_signal_sources": [],
            "stable_interests": [],
            "short_term_interests": [],
            "weak_or_uncertain_interests": [],
            "representative_posts": [],
            "chunk_summaries": [],
        }

    @staticmethod
    def _collect_interest_stats(domain_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for field in ("stable_interests", "short_term_interests", "weak_or_uncertain_interests"):
            for item in domain_row.get(field, []) or []:
                interest = str(item.get("interest", "")).strip()
                if not interest:
                    continue
                rows[interest] = {
                    "stable_score": float(item.get("stable_score", 0.0)),
                    "recent_score": float(item.get("recent_score", 0.0)),
                    "status": str(item.get("status", "weak_signal")),
                    "evidence_post_ids": [str(x) for x in item.get("evidence_post_ids", [])],
                    "evidence_modalities": list(item.get("evidence_modalities", []) or []),
                    "evidence_source_breakdown": dict(item.get("evidence_source_breakdown", {}) or {}),
                }
        return rows

    def _build_tag_cluster_entry(
        self,
        cluster: dict[str, Any],
        interest_stats_by_name: dict[str, dict[str, Any]],
        post_by_id: dict[str, PostObservable],
    ) -> dict[str, Any]:
        canonical_tag = str(cluster.get("canonical_tag", ""))
        stats = interest_stats_by_name.get(canonical_tag, {})
        evidence_ids = [str(x) for x in cluster.get("evidence_post_ids", [])]
        first_seen, last_seen = self._first_last_seen(evidence_ids, post_by_id)
        evidence_examples = self._build_evidence_examples(evidence_ids, post_by_id, limit=3)
        return {
            "canonical_tag": canonical_tag,
            "display_label": self._humanize_tag(canonical_tag, cluster.get("members", [])),
            "members": [str(x) for x in cluster.get("members", [])],
            "n_posts": int(cluster.get("n_posts", 0) or 0),
            "weighted_support": float(cluster.get("weighted_support", 0.0)),
            "mean_confidence": float(cluster.get("mean_confidence", 0.0)),
            "stable_score": float(stats.get("stable_score", cluster.get("stable_score", 0.0))),
            "recent_score": float(stats.get("recent_score", cluster.get("recent_score", 0.0))),
            "status": str(stats.get("status", cluster.get("status", "weak_signal"))),
            "evidence_modalities": list(stats.get("evidence_modalities", cluster.get("evidence_modalities", [])) or []),
            "evidence_source_breakdown": dict(
                stats.get("evidence_source_breakdown", cluster.get("evidence_source_breakdown", {})) or {}
            ),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "evidence_post_ids": evidence_ids,
            "evidence_examples": evidence_examples,
        }

    @classmethod
    def _humanize_tag(cls, canonical_tag: str, members: list[str]) -> str:
        canonical_tag = str(canonical_tag or "").strip()
        if canonical_tag in TAG_DISPLAY_OVERRIDES:
            return TAG_DISPLAY_OVERRIDES[canonical_tag]

        member_candidates = [str(x).strip() for x in members or [] if str(x).strip()]
        member_candidates = [x for x in member_candidates if "_" in x or len(x.split()) > 1]
        if member_candidates:
            return member_candidates[0].replace("_", " ")

        return canonical_tag.replace("_", " ")

    def _build_evidence_examples(
        self,
        evidence_ids: list[str],
        post_by_id: dict[str, PostObservable],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        rows = []
        for pid in evidence_ids[:limit]:
            post = post_by_id.get(pid)
            if not post:
                continue
            created_at = post.created_at if post.created_at.tzinfo else post.created_at.replace(tzinfo=timezone.utc)
            rows.append(
                {
                    "post_id": pid,
                    "created_at": created_at.isoformat().replace("+00:00", "Z"),
                    "text": post.text,
                    "hashtags": post.hashtags,
                }
            )
        return rows

    def _build_representative_posts(
        self,
        representative_rows: list[dict[str, Any]],
        post_by_id: dict[str, PostObservable],
        observed_end: datetime | None,
    ) -> list[dict[str, Any]]:
        rows = []
        for row in representative_rows:
            post_id = str(row.get("post_id", ""))
            if not post_id:
                continue
            post = post_by_id.get(post_id)
            hashtags = post.hashtags if post else []
            media = []
            if post:
                for item in post.media:
                    media.append(
                        {
                            "media_type": item.media_type,
                            "storage_uri": item.storage_uri,
                            "source_url": item.source_url,
                            "alt_text": item.alt_text,
                        }
                    )
            rows.append(
                {
                    "post_id": post_id,
                    "created_at": row.get("created_at"),
                    "post_type": post.post_type if post else None,
                    "text": row.get("text"),
                    "hashtags": hashtags,
                    "media": media,
                    "domain_confidence": float(row.get("confidence", 0.0)),
                    "post_signal_strength": float(row.get("post_signal_strength", 0.0)),
                    "tags": [str(x) for x in row.get("canonical_tags", [])],
                    "evidence_modalities": list(row.get("evidence_modalities", []) or []),
                    "evidence_modality": str(row.get("evidence_modality", "") or "").strip() or None,
                    "evidence_source_breakdown": dict(row.get("evidence_source_breakdown", {}) or {}),
                    "why_selected": self._why_selected(row=row, observed_end=observed_end),
                    "relative_recency": self._relative_recency_label(row.get("created_at"), observed_end),
                }
            )
        return rows

    def _build_chunk_summaries(
        self,
        *,
        domain_name: str,
        chunk_rows: list[dict[str, Any]],
        post_by_id: dict[str, PostObservable],
    ) -> list[dict[str, Any]]:
        rows = []
        for chunk in chunk_rows:
            idx = int(chunk.get("chunk_index", 0))
            start_id = str(chunk.get("start_post_id", ""))
            end_id = str(chunk.get("end_post_id", ""))
            start_dt = self._post_created_at(start_id, post_by_id)
            end_dt = self._post_created_at(end_id, post_by_id)
            top_tags = [str(x) for x in chunk.get("top_tags", [])]
            rows.append(
                {
                    "chunk_id": f"{domain_name}_chunk_{idx + 1:02d}",
                    "time_range": self._format_month_range(start_dt, end_dt),
                    "summary": self._chunk_summary_text(top_tags=top_tags, n_posts=int(chunk.get("n_posts", 0) or 0)),
                }
            )
        return rows

    @staticmethod
    def _why_selected(*, row: dict[str, Any], observed_end: datetime | None) -> str:
        confidence = float(row.get("confidence", 0.0))
        created_at_raw = row.get("created_at")
        created_at = DomainPackBuilder._parse_utc(created_at_raw)
        if observed_end and created_at:
            delta_days = (observed_end - created_at).days
            if confidence >= 0.8 and delta_days <= 90:
                return "high_confidence_recent_post"
            if delta_days > 90:
                return "older_supporting_post"
        if confidence >= 0.8:
            return "high_confidence_supporting_post"
        return "representative_supporting_post"

    @staticmethod
    def _relative_recency_label(created_at_raw: Any, observed_end: datetime | None) -> str | None:
        created_at = DomainPackBuilder._parse_utc(created_at_raw)
        if not created_at or not observed_end:
            return None
        delta_days = (observed_end - created_at).days
        if delta_days <= 30:
            return "last_30_days"
        if delta_days <= 90:
            return "last_90_days"
        return "older"

    @staticmethod
    def _chunk_summary_text(*, top_tags: list[str], n_posts: int) -> str:
        if top_tags:
            tag_text = ", ".join(top_tags[:3])
            return f"Chunk contains {n_posts} posts, with top recurring tags: {tag_text}."
        return f"Chunk contains {n_posts} posts with limited recurring tags."

    @staticmethod
    def _post_created_at(post_id: str, post_by_id: dict[str, PostObservable]) -> datetime | None:
        post = post_by_id.get(post_id)
        if not post:
            return None
        if post.created_at.tzinfo:
            return post.created_at
        return post.created_at.replace(tzinfo=timezone.utc)

    @staticmethod
    def _first_last_seen(evidence_ids: list[str], post_by_id: dict[str, PostObservable]) -> tuple[str | None, str | None]:
        dates = [DomainPackBuilder._post_created_at(pid, post_by_id) for pid in evidence_ids]
        dates = [d for d in dates if d is not None]
        if not dates:
            return None, None
        dates_sorted = sorted(dates)
        return dates_sorted[0].strftime("%Y-%m"), dates_sorted[-1].strftime("%Y-%m")

    @staticmethod
    def _format_month_range(start: datetime | None, end: datetime | None) -> str | None:
        if not start and not end:
            return None
        if start and not end:
            return start.strftime("%Y-%m")
        if end and not start:
            return end.strftime("%Y-%m")
        assert start is not None and end is not None
        return f"{start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}"

    @staticmethod
    def _parse_utc(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
