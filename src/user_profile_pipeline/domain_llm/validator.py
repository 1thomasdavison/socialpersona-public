from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class DomainSummaryValidator:
    stable_floor: float = 0.55
    stable_min_evidence_posts: int = 2
    nontrivial_cluster_min_posts: int = 2
    nontrivial_cluster_min_weighted_support: float = 1.2
    max_representative_evidence_posts: int = 3

    def validate(
        self,
        summary: dict[str, Any],
        domain_pack: dict[str, Any],
        evidence_first_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        domain = str(domain_pack.get("domain", ""))
        canonical_to_evidence, member_to_canonical, allowed_ids = self._build_indices(domain_pack)
        algorithm = domain_pack.get("algorithm_summary", {})
        stable_score = float(algorithm.get("stable_score", 0.0))
        status = str(algorithm.get("status", "absent"))
        actions: list[dict[str, Any]] = []

        raw_stable = summary.get("stable_interests", [])
        raw_short = summary.get("short_term_interests", [])
        raw_weak = summary.get("weak_or_uncertain_interests", [])
        pre_counts = {
            "stable_interests": len(raw_stable) if isinstance(raw_stable, list) else 0,
            "short_term_interests": len(raw_short) if isinstance(raw_short, list) else 0,
            "weak_or_uncertain_interests": len(raw_weak) if isinstance(raw_weak, list) else 0,
        }

        cleaned_stable = self._clean_bucket(
            items=raw_stable,
            domain_pack=domain_pack,
            canonical_to_evidence=canonical_to_evidence,
            member_to_canonical=member_to_canonical,
            allowed_ids=allowed_ids,
            bucket_name="stable_interests",
            actions=actions,
        )
        cleaned_short = self._clean_bucket(
            items=raw_short,
            domain_pack=domain_pack,
            canonical_to_evidence=canonical_to_evidence,
            member_to_canonical=member_to_canonical,
            allowed_ids=allowed_ids,
            bucket_name="short_term_interests",
            actions=actions,
        )
        cleaned_weak = self._clean_bucket(
            items=raw_weak,
            domain_pack=domain_pack,
            canonical_to_evidence=canonical_to_evidence,
            member_to_canonical=member_to_canonical,
            allowed_ids=allowed_ids,
            bucket_name="weak_or_uncertain_interests",
            actions=actions,
        )

        moved_to_weak: list[dict[str, Any]] = []
        for item in cleaned_stable:
            evidence_ids = item.get("all_evidence_post_ids", [])
            if stable_score < self.stable_floor or len(set(evidence_ids)) < self.stable_min_evidence_posts:
                moved_to_weak.append(item)
        if moved_to_weak:
            moved_names = {
                self._normalize_interest_text(str(x.get("label") or ""))
                for x in moved_to_weak
            }
            cleaned_stable = [
                x
                for x in cleaned_stable
                if self._normalize_interest_text(str(x.get("label") or "")) not in moved_names
            ]
            cleaned_weak = self._merge_by_interest(cleaned_weak + moved_to_weak)
            for item in moved_to_weak:
                actions.append(
                    {
                        "type": "downgrade_interest",
                        "interest": item.get("label"),
                        "from": "stable_interests",
                        "to": "weak_or_uncertain_interests",
                        "reason": "insufficient_stability_or_evidence_spread",
                    }
                )

        cleaned_stable = self._merge_by_interest(cleaned_stable)
        cleaned_short = self._merge_by_interest(cleaned_short)
        cleaned_weak = self._merge_by_interest(cleaned_weak)

        if evidence_first_result:
            pass1_candidates = self._normalize_pass1_candidates(
                evidence_first_result,
                domain_pack=domain_pack,
                canonical_to_evidence=canonical_to_evidence,
                member_to_canonical=member_to_canonical,
                allowed_ids=allowed_ids,
                actions=actions,
            )
            if not cleaned_stable and not cleaned_short and pass1_candidates:
                cleaned_stable, cleaned_short, cleaned_weak = self._backfill_from_pass1(
                    pass1_candidates=pass1_candidates,
                    cleaned_stable=cleaned_stable,
                    cleaned_short=cleaned_short,
                    cleaned_weak=cleaned_weak,
                    algorithm=algorithm,
                    actions=actions,
                )

        if status in {"stable", "stable_with_recent_surge"} and not cleaned_stable:
            seeded = self._seed_interest_from_clusters(domain_pack, required_status="stable")
            if seeded:
                cleaned_stable = self._merge_by_interest(cleaned_stable + [seeded])
                actions.append(
                    {
                        "type": "seed_interest_from_cluster",
                        "interest": seeded["label"],
                        "to": "stable_interests",
                        "reason": "status_requires_stable_interest_with_nontrivial_cluster",
                    }
                )

        if status == "recent" and not cleaned_short:
            seeded = self._seed_interest_from_clusters(domain_pack, required_status="recent")
            if seeded:
                cleaned_short = self._merge_by_interest(cleaned_short + [seeded])
                actions.append(
                    {
                        "type": "seed_interest_from_cluster",
                        "interest": seeded["label"],
                        "to": "short_term_interests",
                        "reason": "status_requires_recent_interest_with_nontrivial_cluster",
                    }
                )

        domain_summary = str(summary.get("domain_summary", "")).strip()
        if not domain_summary or self._is_placeholder_summary(domain_summary):
            domain_summary = self._build_domain_summary(
                domain_pack=domain_pack,
                stable_interests=cleaned_stable,
                short_term_interests=cleaned_short,
                weak_interests=cleaned_weak,
            )
            actions.append(
                {
                    "type": "rewrite_domain_summary",
                    "reason": "missing_or_placeholder_summary",
                }
            )

        uncertainty_note = summary.get("uncertainty_note")
        if uncertainty_note is not None:
            uncertainty_note = str(uncertainty_note).strip() or None

        structured_interests = self._structured_interest_set(cleaned_stable, cleaned_short, cleaned_weak)
        if uncertainty_note is None and not structured_interests:
            uncertainty_note = "Evidence in this domain is currently too sparse or fragmented for a reliable interest profile."
            actions.append(
                {
                    "type": "set_uncertainty_note",
                    "reason": "no_structured_interests",
                }
            )
        elif uncertainty_note is None and stable_score < self.stable_floor:
            uncertainty_note = "Evidence is limited or uneven; treat these interests as provisional."
            actions.append(
                {
                    "type": "set_uncertainty_note",
                    "reason": "low_stable_score",
                }
            )

        post_counts = {
            "stable_interests": len(cleaned_stable),
            "short_term_interests": len(cleaned_short),
            "weak_or_uncertain_interests": len(cleaned_weak),
        }
        if pre_counts != post_counts:
            actions.append(
                {
                    "type": "interest_bucket_count_change",
                    "from": pre_counts,
                    "to": post_counts,
                }
            )

        domain_all_evidence_post_ids, domain_representative_evidence_post_ids = self._collect_domain_evidence_post_ids(
            stable_interests=cleaned_stable,
            short_term_interests=cleaned_short,
            weak_interests=cleaned_weak,
            domain_pack=domain_pack,
        )

        return {
            "schema_version": "domain_llm_summary_v3",
            "domain": domain,
            "stable_interests": cleaned_stable,
            "short_term_interests": cleaned_short,
            "weak_or_uncertain_interests": cleaned_weak,
            "domain_all_evidence_post_ids": domain_all_evidence_post_ids,
            "domain_representative_evidence_post_ids": domain_representative_evidence_post_ids,
            "domain_summary": domain_summary,
            "uncertainty_note": uncertainty_note,
            "validator_actions": actions,
            "validator_debug": {
                "raw_model_output": summary,
                "pre_validation_counts": pre_counts,
                "post_validation_counts": post_counts,
            },
        }

    @classmethod
    def _build_indices(cls, domain_pack: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, str], set[str]]:
        canonical_to_evidence: dict[str, list[str]] = {}
        member_to_canonical: dict[str, str] = {}
        allowed_ids: set[str] = set()

        for cluster in domain_pack.get("tag_clusters", []) or []:
            canonical = str(cluster.get("canonical_tag", "")).strip()
            if not canonical:
                continue

            evidence = [str(x).strip() for x in cluster.get("evidence_post_ids", []) if str(x).strip()]
            canonical_to_evidence[canonical] = evidence
            for pid in evidence:
                allowed_ids.add(pid)

            cls._add_mapping(member_to_canonical, canonical, canonical)
            cls._add_mapping(member_to_canonical, canonical.replace("_", " "), canonical)

            display_label = str(cluster.get("display_label", "")).strip()
            if display_label:
                cls._add_mapping(member_to_canonical, display_label, canonical)

            for member in cluster.get("members", []) or []:
                token = str(member).strip()
                if not token:
                    continue
                cls._add_mapping(member_to_canonical, token, canonical)
                cls._add_mapping(member_to_canonical, token.replace("_", " "), canonical)

        for row in domain_pack.get("representative_posts", []) or []:
            pid = str(row.get("post_id", "")).strip()
            if pid:
                allowed_ids.add(pid)

        return canonical_to_evidence, member_to_canonical, allowed_ids

    @staticmethod
    def _add_mapping(mapping: dict[str, str], token: str, canonical: str) -> None:
        key = DomainSummaryValidator._normalize_interest_text(token)
        if key and key not in mapping:
            mapping[key] = canonical

    def _clean_bucket(
        self,
        *,
        items: Any,
        domain_pack: dict[str, Any],
        canonical_to_evidence: dict[str, list[str]],
        member_to_canonical: dict[str, str],
        allowed_ids: set[str],
        bucket_name: str,
        actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not isinstance(items, list):
            actions.append({"type": "drop_bucket", "bucket": bucket_name, "reason": "bucket_is_not_list"})
            return rows

        for item in items:
            if not isinstance(item, dict):
                actions.append({"type": "drop_interest", "bucket": bucket_name, "reason": "item_is_not_object"})
                continue

            raw_label = str(item.get("label") or "").strip()
            description = str(item.get("description", "")).strip()

            all_ids = self._normalize_post_ids(item.get("all_evidence_post_ids", []), allowed_ids=allowed_ids)
            representative_ids = self._normalize_post_ids(
                item.get("representative_evidence_post_ids", []),
                allowed_ids=allowed_ids,
            )

            canonical_tags = self._normalize_canonical_tags(
                item.get("canonical_tags", []),
                member_to_canonical,
            )

            if not canonical_tags and all_ids:
                canonical_tags = self._infer_canonical_tags_from_ids(all_ids, domain_pack)

            if not canonical_tags and raw_label:
                guessed = self._lookup_canonical(raw_label, member_to_canonical)
                if guessed:
                    canonical_tags = [guessed]

            if not raw_label and canonical_tags:
                raw_label = self._humanize_canonical_tags(canonical_tags, domain_pack)

            if self._is_excluded_attribute_label(raw_label):
                actions.append(
                    {
                        "type": "drop_interest",
                        "bucket": bucket_name,
                        "reason": "excluded_non_interest_attribute",
                        "label": raw_label,
                    }
                )
                continue

            if not raw_label:
                actions.append(
                    {
                        "type": "drop_interest",
                        "bucket": bucket_name,
                        "reason": "missing_label_after_normalization",
                    }
                )
                continue

            if not all_ids and canonical_tags:
                backfill_ids: list[str] = []
                for tag in canonical_tags:
                    backfill_ids.extend(canonical_to_evidence.get(tag, []))
                all_ids = self._normalize_post_ids(backfill_ids)

            representative_ids = self._pick_representative_ids(
                all_ids=all_ids,
                preferred_ids=representative_ids,
            )

            if not description:
                description = self._default_description(raw_label, canonical_tags, domain_pack)

            reasoning = str(item.get("reasoning", "")).strip() or "Evidence-grounded interest retained after validation."

            rows.append(
                {
                    "label": raw_label,
                    "canonical_tags": canonical_tags,
                    "description": description,
                    "confidence": self._clip01(item.get("confidence", 0.5)),
                    "all_evidence_post_ids": all_ids,
                    "representative_evidence_post_ids": representative_ids,
                    "reasoning": reasoning,
                }
            )

        return rows

    def _normalize_canonical_tags(
        self,
        raw_tags: Any,
        member_to_canonical: dict[str, str],
    ) -> list[str]:
        if not isinstance(raw_tags, list):
            return []
        out: list[str] = []
        for tag in raw_tags:
            norm = self._lookup_canonical(str(tag).strip(), member_to_canonical)
            if norm and norm not in out:
                out.append(norm)
        return out

    @staticmethod
    def _lookup_canonical(token: str, member_to_canonical: dict[str, str]) -> str | None:
        key = DomainSummaryValidator._normalize_interest_text(token)
        if not key:
            return None
        return member_to_canonical.get(key)

    @staticmethod
    def _infer_canonical_tags_from_ids(
        ids: list[str],
        domain_pack: dict[str, Any],
    ) -> list[str]:
        id_set = set(ids)
        matched: list[str] = []
        for cluster in domain_pack.get("tag_clusters", []) or []:
            cluster_ids = set(str(x) for x in cluster.get("evidence_post_ids", []) if str(x).strip())
            if not (id_set & cluster_ids):
                continue
            canonical = str(cluster.get("canonical_tag", "")).strip()
            if canonical and canonical not in matched:
                matched.append(canonical)
        return matched[:4]

    @staticmethod
    def _humanize_canonical_tags(canonical_tags: list[str], domain_pack: dict[str, Any]) -> str:
        display_by_tag = {
            str(c.get("canonical_tag", "")).strip(): str(c.get("display_label", "")).strip()
            for c in domain_pack.get("tag_clusters", []) or []
        }
        labels = [display_by_tag.get(tag) or tag.replace("_", " ") for tag in canonical_tags if tag]
        labels = [x for x in labels if x]
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        return labels[0]

    @staticmethod
    def _default_description(label: str, canonical_tags: list[str], domain_pack: dict[str, Any]) -> str:
        _ = domain_pack
        if canonical_tags:
            return f"Evidence in this domain consistently points to {label} as a meaningful interest signal."
        return f"Evidence in this domain points to {label} as a potential interest signal."

    @staticmethod
    def _is_excluded_attribute_label(label: str) -> bool:
        if not label:
            return False
        text = DomainSummaryValidator._normalize_interest_text(label)
        tokens = set(text.split())
        blocked_tokens = {"family", "career", "job", "profession", "age"}
        return bool(tokens & blocked_tokens) or ("age range" in text)

    def _merge_by_interest(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []

        best: dict[str, dict[str, Any]] = {}
        for item in items:
            key = str(item.get("label") or "").strip().lower()
            if not key:
                continue

            prev = best.get(key)
            if prev is None:
                best[key] = dict(item)
                continue

            prev_conf = float(prev.get("confidence", 0.0))
            item_conf = float(item.get("confidence", 0.0))
            base = dict(item if item_conf > prev_conf else prev)

            prev_all = DomainSummaryValidator._normalize_post_ids(prev.get("all_evidence_post_ids", []))
            item_all = DomainSummaryValidator._normalize_post_ids(item.get("all_evidence_post_ids", []))
            merged_all = DomainSummaryValidator._normalize_post_ids(prev_all + item_all)

            prev_repr = DomainSummaryValidator._normalize_post_ids(prev.get("representative_evidence_post_ids", []))
            item_repr = DomainSummaryValidator._normalize_post_ids(item.get("representative_evidence_post_ids", []))
            preferred_repr = prev_repr + item_repr + prev_all + item_all
            merged_repr = self._pick_representative_ids(
                all_ids=merged_all,
                preferred_ids=preferred_repr,
            )

            base["all_evidence_post_ids"] = merged_all
            base["representative_evidence_post_ids"] = merged_repr
            base["canonical_tags"] = sorted(
                set(prev.get("canonical_tags", [])) | set(item.get("canonical_tags", []))
            )
            if not base.get("description"):
                base["description"] = prev.get("description") or item.get("description")
            if not base.get("reasoning"):
                base["reasoning"] = prev.get("reasoning") or item.get("reasoning")
            best[key] = base

        merged = list(best.values())
        for item in merged:
            label = str(item.get("label") or "").strip()
            item["label"] = label
            item["canonical_tags"] = sorted(set(item.get("canonical_tags", [])))
            all_ids = DomainSummaryValidator._normalize_post_ids(item.get("all_evidence_post_ids", []))
            representative_ids = DomainSummaryValidator._normalize_post_ids(
                item.get("representative_evidence_post_ids", [])
            )
            representative_ids = self._pick_representative_ids(
                all_ids=all_ids,
                preferred_ids=representative_ids,
            )

            item["all_evidence_post_ids"] = all_ids
            item["representative_evidence_post_ids"] = representative_ids

        return sorted(merged, key=lambda x: (-float(x.get("confidence", 0.0)), str(x.get("label", ""))))

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            n = float(value)
        except (TypeError, ValueError):
            n = 0.5
        return max(0.0, min(1.0, n))

    @staticmethod
    def _normalize_post_ids(raw_ids: Any, *, allowed_ids: set[str] | None = None) -> list[str]:
        if not isinstance(raw_ids, list):
            return []
        out: list[str] = []
        for value in raw_ids:
            pid = str(value).strip()
            if not pid:
                continue
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            if pid not in out:
                out.append(pid)
        return out

    def _pick_representative_ids(
        self,
        *,
        all_ids: list[str],
        preferred_ids: list[str] | None = None,
    ) -> list[str]:
        if not all_ids:
            return []
        picks: list[str] = []
        for pid in (preferred_ids or []):
            if pid not in all_ids or pid in picks:
                continue
            picks.append(pid)
            if len(picks) >= self.max_representative_evidence_posts:
                return picks
        for pid in all_ids:
            if pid in picks:
                continue
            picks.append(pid)
            if len(picks) >= self.max_representative_evidence_posts:
                break
        return picks

    def _seed_interest_from_clusters(self, domain_pack: dict[str, Any], *, required_status: str) -> dict[str, Any] | None:
        candidates: list[tuple[float, dict[str, Any]]] = []

        for cluster in domain_pack.get("tag_clusters", []) or []:
            if str(cluster.get("status", "")) != required_status:
                continue
            if not self._is_nontrivial_cluster(cluster):
                continue
            if self._should_skip_seed_cluster(cluster, domain_pack=domain_pack):
                continue

            canonical = str(cluster.get("canonical_tag", "")).strip()
            if not canonical:
                continue

            label = str(cluster.get("display_label", "")).strip() or canonical.replace("_", " ")
            if self._is_excluded_attribute_label(label):
                continue

            all_evidence_ids = self._normalize_post_ids(cluster.get("evidence_post_ids", []))
            representative_evidence_ids = self._pick_representative_ids(all_ids=all_evidence_ids)
            mean_confidence = self._clip01(cluster.get("mean_confidence", 0.6))
            weighted_support = float(cluster.get("weighted_support", 0.0) or 0.0)
            n_posts = int(cluster.get("n_posts", 0) or 0)
            label_bonus = 0.08 * max(0, len(label.split()) - 1)
            rank = weighted_support * max(0.1, mean_confidence) + (0.05 * n_posts) + label_bonus

            candidates.append(
                (
                    rank,
                    {
                        "label": label,
                        "canonical_tags": [canonical],
                        "description": f"Posts in this domain repeatedly point to {label}.",
                        "confidence": mean_confidence,
                        "all_evidence_post_ids": all_evidence_ids,
                        "representative_evidence_post_ids": representative_evidence_ids,
                        "reasoning": "Backfilled from a validated cluster when LLM output was missing.",
                    },
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _should_skip_seed_cluster(self, cluster: dict[str, Any], *, domain_pack: dict[str, Any]) -> bool:
        canonical = str(cluster.get("canonical_tag", "")).strip()
        if not canonical:
            return True

        modalities = sorted(
            {
                str(x).strip()
                for x in cluster.get("evidence_modalities", []) or []
                if str(x).strip()
            }
        )

        if canonical.endswith("_ownership") and modalities == ["visual"]:
            return True

        if domain_pack.get("domain") == "pets" and canonical in {"cat", "dog"} and modalities == ["visual"]:
            return True

        if domain_pack.get("domain") == "food_drink" and canonical in {"cook", "home_cook", "food_preparation"}:
            if self._cluster_food_seed_is_ambiguous(cluster):
                return True

        return False

    @staticmethod
    def _cluster_food_seed_is_ambiguous(cluster: dict[str, Any]) -> bool:
        texts = [
            " ".join(str(row.get("text") or "").split()).strip().lower()
            for row in cluster.get("evidence_examples", []) or []
            if isinstance(row, dict)
        ]
        texts = [text for text in texts if text]
        if not texts:
            return False

        literal_food_markers = (
            "recipe",
            "ingredients",
            "dish",
            "meal",
            "kitchen",
            "restaurant",
            "cafe",
            "coffee",
            "tea",
            "burger",
            "shawarma",
            "ramen",
            "cheesecake",
            "dessert",
            "eat ",
            "eating",
            "ate ",
            "drank ",
            "drink ",
            "drinking",
            "made ",
            "making ",
            "i cooked",
            "i made",
            "we cooked",
            "we made",
        )
        slang_or_ambiguous_markers = (
            "cooked or nah",
            "been cooking",
            "can cook",
            "chef",
        )

        has_literal = any(any(marker in text for marker in literal_food_markers) for text in texts)
        has_slang = any(any(marker in text for marker in slang_or_ambiguous_markers) for text in texts)
        return has_slang and not has_literal

    def _build_domain_summary(
        self,
        *,
        domain_pack: dict[str, Any],
        stable_interests: list[dict[str, Any]],
        short_term_interests: list[dict[str, Any]],
        weak_interests: list[dict[str, Any]],
    ) -> str:
        _ = domain_pack
        if not stable_interests and not short_term_interests and not weak_interests:
            return self._build_no_interest_summary(domain_pack)

        parts: list[str] = []

        if stable_interests:
            labels = ", ".join(item["label"] for item in stable_interests[:2])
            parts.append(f"This domain most clearly reflects a sustained interest in {labels}.")

        if short_term_interests:
            labels = ", ".join(item["label"] for item in short_term_interests[:2])
            parts.append(f"There is also a more recent or emerging focus on {labels}.")

        if weak_interests:
            labels = ", ".join(item["label"] for item in weak_interests[:2])
            parts.append(
                "Weaker signals also point to "
                f"{labels}, but the evidence is not yet strong enough to treat them as central interests."
            )

        return " ".join(parts)

    @staticmethod
    def _build_no_interest_summary(domain_pack: dict[str, Any]) -> str:
        _ = domain_pack
        return (
            "The current evidence in this domain is too sparse, too one-off, or too noisy "
            "to support a reliable interest description."
        )

    def _is_nontrivial_cluster(self, cluster: dict[str, Any]) -> bool:
        n_posts = int(cluster.get("n_posts", 0) or 0)
        weighted_support = float(cluster.get("weighted_support", 0.0) or 0.0)
        return (
            n_posts >= self.nontrivial_cluster_min_posts
            or weighted_support >= self.nontrivial_cluster_min_weighted_support
        )

    @staticmethod
    def _structured_interest_set(
        stable_interests: list[dict[str, Any]],
        short_term_interests: list[dict[str, Any]],
        weak_interests: list[dict[str, Any]],
    ) -> set[str]:
        names: set[str] = set()
        for row in stable_interests + short_term_interests + weak_interests:
            interest = str(row.get("label") or "").strip()
            if interest:
                names.add(interest)
        return names

    @staticmethod
    def _normalize_interest_text(text: str) -> str:
        lowered = text.lower().replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _is_placeholder_summary(domain_summary: str) -> bool:
        text = domain_summary.lower().strip()
        if not text:
            return True
        placeholders = (
            "domain evidence status is",
            "summary is kept conservative",
            "the domain is stable",
            "support_posts=",
            "weak_posts=",
            "stable_score=",
        )
        return any(p in text for p in placeholders)

    def _normalize_pass1_candidates(
        self,
        evidence_first_result: dict[str, Any],
        *,
        domain_pack: dict[str, Any],
        canonical_to_evidence: dict[str, list[str]],
        member_to_canonical: dict[str, str],
        allowed_ids: set[str],
        actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = evidence_first_result.get("candidate_interests", []) if isinstance(evidence_first_result, dict) else []
        if not isinstance(candidates, list):
            return []

        normalized_input: list[dict[str, Any]] = []
        hints_by_label: dict[str, str] = {}
        for item in candidates:
            if not isinstance(item, dict):
                continue

            label = str(item.get("label") or "").strip()
            if label:
                hint = str(item.get("time_hint", "")).strip()
                hints_by_label[self._normalize_interest_text(label)] = hint

            normalized_input.append(
                {
                    "label": label,
                    "canonical_tags": item.get("canonical_tags", []),
                    "description": item.get("description", ""),
                    "confidence": item.get("confidence", 0.5),
                    "all_evidence_post_ids": item.get("evidence_post_ids", []),
                    "representative_evidence_post_ids": item.get("evidence_post_ids", []),
                    "reasoning": item.get("reasoning", ""),
                }
            )

        rows = self._clean_bucket(
            items=normalized_input,
            domain_pack=domain_pack,
            canonical_to_evidence=canonical_to_evidence,
            member_to_canonical=member_to_canonical,
            allowed_ids=allowed_ids,
            bucket_name="pass1_candidates",
            actions=actions,
        )

        for row in rows:
            key = self._normalize_interest_text(str(row.get("label") or ""))
            hint = hints_by_label.get(key, "weak_candidate")
            if hint not in {"stable_candidate", "recent_candidate", "weak_candidate"}:
                hint = "weak_candidate"
            row["time_hint"] = hint

        return rows

    def _backfill_from_pass1(
        self,
        *,
        pass1_candidates: list[dict[str, Any]],
        cleaned_stable: list[dict[str, Any]],
        cleaned_short: list[dict[str, Any]],
        cleaned_weak: list[dict[str, Any]],
        algorithm: dict[str, Any],
        actions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        status = str(algorithm.get("status", "absent"))
        for item in pass1_candidates:
            payload = dict(item)
            hint = str(payload.pop("time_hint", "weak_candidate") or "weak_candidate")
            target_bucket = self._bucket_from_hint(hint, status)

            if target_bucket == "stable_interests":
                cleaned_stable.append(payload)
            elif target_bucket == "short_term_interests":
                cleaned_short.append(payload)
            else:
                cleaned_weak.append(payload)

            actions.append(
                {
                    "type": "backfill_interest_from_pass1",
                    "interest": payload.get("label"),
                    "to": target_bucket,
                    "reason": f"pass1_{hint}",
                }
            )

        return (
            self._merge_by_interest(cleaned_stable),
            self._merge_by_interest(cleaned_short),
            self._merge_by_interest(cleaned_weak),
        )

    @staticmethod
    def _bucket_from_hint(hint: str, status: str) -> str:
        if hint == "stable_candidate":
            return "stable_interests"
        if hint == "recent_candidate":
            return "short_term_interests"
        if hint == "weak_candidate":
            return "weak_or_uncertain_interests"

        if status in {"stable", "stable_with_recent_surge"}:
            return "stable_interests"
        if status == "recent":
            return "short_term_interests"
        return "weak_or_uncertain_interests"

    def _collect_domain_evidence_post_ids(
        self,
        *,
        stable_interests: list[dict[str, Any]],
        short_term_interests: list[dict[str, Any]],
        weak_interests: list[dict[str, Any]],
        domain_pack: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        all_ids: list[str] = []
        preferred_ids: list[str] = []

        for item in stable_interests + short_term_interests + weak_interests:
            item_all = self._normalize_post_ids(item.get("all_evidence_post_ids", []))
            item_representative = self._normalize_post_ids(item.get("representative_evidence_post_ids", []))
            all_ids.extend(item_all)
            preferred_ids.extend(item_representative)

        for cluster in domain_pack.get("tag_clusters", []) or []:
            all_ids.extend(self._normalize_post_ids(cluster.get("evidence_post_ids", [])))

        for row in domain_pack.get("representative_posts", []) or []:
            pid = str(row.get("post_id", "")).strip()
            if not pid:
                continue
            all_ids.append(pid)
            preferred_ids.append(pid)

        dedup_all_ids = self._normalize_post_ids(all_ids)
        representative_ids = self._pick_representative_ids(
            all_ids=dedup_all_ids,
            preferred_ids=preferred_ids,
        )
        return dedup_all_ids, representative_ids
