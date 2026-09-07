from __future__ import annotations

from dataclasses import dataclass

from ..schemas import PostObservable
from .text_utils import jaccard_similarity, token_counter, tokenize_for_dup


@dataclass
class DuplicateCluster:
    cluster_id: str
    member_post_ids: list[str]


class NearDuplicateClusterer:
    def __init__(self, jaccard_threshold: float = 0.82, cosine_threshold: float = 0.90) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.cosine_threshold = cosine_threshold

    def cluster(self, posts: list[PostObservable]) -> list[DuplicateCluster]:
        ordered = sorted(posts, key=lambda p: p.created_at)
        reps: list[tuple[str, list[str], dict[str, int], list[str]]] = []
        clusters: list[DuplicateCluster] = []

        for post in ordered:
            tokens = tokenize_for_dup(post.text)
            counter = token_counter(post.text)
            assigned = False
            for i, (_, member_ids, rep_counter, rep_tokens) in enumerate(reps):
                jac = jaccard_similarity(tokens, rep_tokens)
                if jac >= self.jaccard_threshold:
                    member_ids.append(post.post_id)
                    clusters[i].member_post_ids.append(post.post_id)
                    assigned = True
                    break
                # lightweight bag overlap, avoids extra dependency
                dot = sum(counter[k] * rep_counter.get(k, 0) for k in counter)
                na = sum(v * v for v in counter.values()) ** 0.5
                nb = sum(v * v for v in rep_counter.values()) ** 0.5
                cosine = dot / (na * nb) if na and nb else 0.0
                if cosine >= self.cosine_threshold:
                    member_ids.append(post.post_id)
                    clusters[i].member_post_ids.append(post.post_id)
                    assigned = True
                    break
            if not assigned:
                cluster_id = f"dup_{len(clusters):04d}"
                reps.append((cluster_id, [post.post_id], dict(counter), tokens))
                clusters.append(DuplicateCluster(cluster_id=cluster_id, member_post_ids=[post.post_id]))
        return clusters
