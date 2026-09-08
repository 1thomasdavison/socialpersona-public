from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schemas import DomainResult
from .text_utils import normalize_tag


@dataclass
class TagObservation:
    post_id: str
    created_at: str
    domain: str
    raw_tag: str
    normalized_tag: str
    confidence: float
    dup_weight: float


@dataclass
class CanonicalTagCluster:
    domain: str
    canonical_tag: str
    members: list[str]
    n_posts: int
    weighted_support: float
    mean_confidence: float
    evidence_post_ids: list[str]


class TagClusterBuilder:
    def __init__(self, aliases: dict[str, dict[str, str]] | None = None, fuzzy_threshold: int = 90, embedding_threshold: float = 0.72) -> None:
        self.aliases = aliases or {}
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold

    def collect_observations(
        self,
        records: Iterable[tuple[str, str, float, DomainResult]],
    ) -> list[TagObservation]:
        observations: list[TagObservation] = []
        for post_id, created_at, dup_weight, domain_result in records:
            alias_map = self.aliases.get(domain_result.domain, {})
            for tag in domain_result.tags:
                normalized = normalize_tag(tag.tag)
                normalized = alias_map.get(normalized, normalized)
                observations.append(
                    TagObservation(
                        post_id=post_id,
                        created_at=created_at,
                        domain=domain_result.domain,
                        raw_tag=tag.tag,
                        normalized_tag=normalized,
                        confidence=tag.confidence,
                        dup_weight=dup_weight,
                    )
                )
        return observations

    def build(self, observations: list[TagObservation]) -> list[CanonicalTagCluster]:
        by_domain: dict[str, list[TagObservation]] = defaultdict(list)
        for obs in observations:
            by_domain[obs.domain].append(obs)

        all_clusters: list[CanonicalTagCluster] = []
        for domain, domain_obs in by_domain.items():
            normalized_tags = sorted({o.normalized_tag for o in domain_obs})
            merged_groups = self._merge_domain_tags(domain, normalized_tags)
            obs_by_tag: dict[str, list[TagObservation]] = defaultdict(list)
            for obs in domain_obs:
                canonical = merged_groups.get(obs.normalized_tag, obs.normalized_tag)
                obs_by_tag[canonical].append(obs)

            for canonical, items in sorted(obs_by_tag.items()):
                member_tags = sorted({o.normalized_tag for o in items} | {o.raw_tag for o in items})
                all_clusters.append(
                    CanonicalTagCluster(
                        domain=domain,
                        canonical_tag=canonical,
                        members=member_tags,
                        n_posts=len({o.post_id for o in items}),
                        weighted_support=round(sum(o.dup_weight for o in items), 6),
                        mean_confidence=round(mean(o.confidence for o in items), 6),
                        evidence_post_ids=sorted({o.post_id for o in items}),
                    )
                )
        return all_clusters

    def _merge_domain_tags(self, domain: str, tags: list[str]) -> dict[str, str]:
        if not tags:
            return {}

        parent = {tag: tag for tag in tags}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            chosen = min([ra, rb], key=lambda t: (len(t.split("_")), len(t), t))
            other = rb if chosen == ra else ra
            parent[other] = chosen

        alias_map = self.aliases.get(domain, {})
        for tag in tags:
            mapped = alias_map.get(tag)
            if mapped and mapped in parent:
                union(tag, mapped)

        for i, ta in enumerate(tags):
            for tb in tags[i + 1 :]:
                if fuzz.ratio(ta, tb) >= self.fuzzy_threshold:
                    union(ta, tb)

        if len(tags) >= 2:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            matrix = vectorizer.fit_transform(tags)
            sim = cosine_similarity(matrix)
            dist = 1 - sim
            clustering = AgglomerativeClustering(
                metric="precomputed",
                linkage="average",
                distance_threshold=1 - self.embedding_threshold,
                n_clusters=None,
            )
            labels = clustering.fit_predict(dist)
            groups: dict[int, list[str]] = defaultdict(list)
            for tag, label in zip(tags, labels):
                groups[int(label)].append(tag)
            for members in groups.values():
                if len(members) > 1:
                    canonical = min(members, key=lambda t: (len(t.split("_")), len(t), t))
                    for member in members:
                        union(member, canonical)

        final_map: dict[str, str] = {}
        grouped: dict[str, list[str]] = defaultdict(list)
        for tag in tags:
            grouped[find(tag)].append(tag)
        for members in grouped.values():
            counts = Counter(members)
            canonical = min(counts.keys(), key=lambda t: (len(t.split("_")), len(t), t))
            for member in members:
                final_map[member] = canonical
        return final_map
