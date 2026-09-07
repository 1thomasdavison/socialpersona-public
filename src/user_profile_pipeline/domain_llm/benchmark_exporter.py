from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..llm_client import OpenAICompatibleChatClient
from .prompts import (
    BENCHMARK_GOLD_REWRITE_JSON_SCHEMA_HINT,
    BENCHMARK_GOLD_REWRITE_SYSTEM_PROMPT,
    build_benchmark_gold_rewrite_user_prompt,
)

PROFILE_BENCHMARK_EXPORT_SCHEMA: dict[str, Any] = {
    "schema_version": "profile_benchmark_export_v3",
    "user_id": "string",
    "observation_window": {
        "post_count_total": "int",
        "observed_start": "string",
        "observed_end": "string",
        "window_days": "int",
    },
    "domains": [
        {
            "domain": "string",
            "status": "active|inactive",
            "long_term_interests": [
                {
                    "label": "string",
                    "evidence_post_ids": ["string"],
                    "evidence_post_indices": ["int"],
                    "evidence_source_summary": {
                        "modalities": ["text|visual"],
                        "text_only_post_ids": ["string"],
                        "text_only_post_indices": ["int"],
                        "visual_only_post_ids": ["string"],
                        "visual_only_post_indices": ["int"],
                        "mixed_post_ids": ["string"],
                        "mixed_post_indices": ["int"],
                    },
                }
            ],
            "short_term_interests": [
                {
                    "label": "string",
                    "evidence_post_ids": ["string"],
                    "evidence_post_indices": ["int"],
                    "evidence_source_summary": {
                        "modalities": ["text|visual"],
                        "text_only_post_ids": ["string"],
                        "text_only_post_indices": ["int"],
                        "visual_only_post_ids": ["string"],
                        "visual_only_post_indices": ["int"],
                        "mixed_post_ids": ["string"],
                        "mixed_post_indices": ["int"],
                    },
                }
            ],
            "negative_interests": [
                {
                    "label": "string",
                    "evidence_post_ids": ["string"],
                    "evidence_post_indices": ["int"],
                    "evidence_source_summary": {
                        "modalities": ["text|visual"],
                        "text_only_post_ids": ["string"],
                        "text_only_post_indices": ["int"],
                        "visual_only_post_ids": ["string"],
                        "visual_only_post_indices": ["int"],
                        "mixed_post_ids": ["string"],
                        "mixed_post_indices": ["int"],
                    },
                }
            ],
            "domain_all_evidence_post_indices": ["int"],
            "domain_representative_evidence_post_indices": ["int"],
            "domain_evidence_post_indices": ["int"],
            "domain_evidence_source_summary": {
                "modalities": ["text|visual"],
                "text_only_post_ids": ["string"],
                "text_only_post_indices": ["int"],
                "visual_only_post_ids": ["string"],
                "visual_only_post_indices": ["int"],
                "mixed_post_ids": ["string"],
                "mixed_post_indices": ["int"],
            },
            "summary_atomic": "string",
            "summary_natural_gold": "string",
            "debug_meta": {
                "source_domain_summary": "string",
                "uncertainty_note": "string|null",
                "dropped_weak_interest_labels": ["string"],
            },
        }
    ],
}


@dataclass
class BenchmarkGoldExporter:
    client: OpenAICompatibleChatClient | None = None
    max_long_term: int = 3
    max_short_term: int = 3
    max_negative: int = 3
    max_domain_evidence: int = 4
    max_interest_evidence_indices: int = 3
    max_evidence_examples_per_interest: int = 2

    def export_user_batch(
        self,
        *,
        batch_result: dict[str, Any],
        domain_packs: list[dict[str, Any]] | None = None,
        post_id_to_index: dict[str, int] | None = None,
        summary_natural_gold_by_domain: dict[str, str] | None = None,
        rewrite_enabled: bool = True,
    ) -> dict[str, Any]:
        post_id_to_index = post_id_to_index or {}
        domain_packs = domain_packs or []
        pack_by_domain = self._build_pack_map(domain_packs)
        summary_natural_gold_by_domain = summary_natural_gold_by_domain or {}

        exported_domains: list[dict[str, Any]] = []
        for domain_row in batch_result.get("domain_summaries", []) or []:
            domain = str(domain_row.get("domain", "")).strip()
            if not domain:
                continue

            domain_pack = pack_by_domain.get(domain, {})
            exported_domains.append(
                self._export_domain(
                    domain_row=domain_row,
                    domain_pack=domain_pack,
                    post_id_to_index=post_id_to_index,
                    summary_natural_gold=summary_natural_gold_by_domain.get(domain),
                    rewrite_enabled=rewrite_enabled,
                )
            )

        return {
            "schema_version": "profile_benchmark_export_v3",
            "user_id": batch_result.get("user_id"),
            "observation_window": batch_result.get("observation_window"),
            "domains": exported_domains,
        }

    def build_rewrite_batch_requests(
        self,
        *,
        batch_result: dict[str, Any],
        domain_packs: list[dict[str, Any]] | None = None,
        post_id_to_index: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        post_id_to_index = post_id_to_index or {}
        domain_packs = domain_packs or []
        pack_by_domain = self._build_pack_map(domain_packs)
        requests_out: list[dict[str, Any]] = []

        for domain_row in batch_result.get("domain_summaries", []) or []:
            domain = str(domain_row.get("domain", "")).strip()
            if not domain:
                continue
            domain_pack = pack_by_domain.get(domain, {})
            status = self._derive_status(domain_row)
            long_term = self._select_interest_rows(
                items=domain_row.get("stable_interests", []),
                post_id_to_index=post_id_to_index,
                max_items=self.max_long_term,
                domain_pack=domain_pack,
            )
            short_term = self._select_interest_rows(
                items=domain_row.get("short_term_interests", []),
                post_id_to_index=post_id_to_index,
                max_items=self.max_short_term,
                domain_pack=domain_pack,
            )
            weak_labels = self._collect_interest_labels(
                domain_row.get("weak_or_uncertain_interests", [])
            )
            rewrite_input = self._build_rewrite_input(
                domain=domain,
                domain_row=domain_row,
                domain_pack=domain_pack,
                status=status,
                long_term=long_term,
                short_term=short_term,
                weak_labels=weak_labels,
                post_id_to_index=post_id_to_index,
            )
            user_prompt = build_benchmark_gold_rewrite_user_prompt(
                json.dumps(rewrite_input, ensure_ascii=False, indent=2)
            )
            requests_out.append(
                {
                    "domain": domain,
                    "request": {
                        "systemInstruction": {
                            "role": "system",
                            "parts": [{"text": BENCHMARK_GOLD_REWRITE_SYSTEM_PROMPT}],
                        },
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": user_prompt}],
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.1,
                            "responseMimeType": "application/json",
                        },
                    },
                }
            )
        return requests_out

    @staticmethod
    def _build_pack_map(domain_packs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(pack.get("domain", "")).strip(): pack
            for pack in domain_packs
            if str(pack.get("domain", "")).strip()
        }

    def _export_domain(
        self,
        *,
        domain_row: dict[str, Any],
        domain_pack: dict[str, Any],
        post_id_to_index: dict[str, int],
        summary_natural_gold: str | None,
        rewrite_enabled: bool,
    ) -> dict[str, Any]:
        domain = str(domain_row.get("domain", "")).strip()
        status = self._derive_status(domain_row)

        long_term = self._select_interest_rows(
            items=domain_row.get("stable_interests", []),
            post_id_to_index=post_id_to_index,
            max_items=self.max_long_term,
            domain_pack=domain_pack,
        )
        short_term = self._select_interest_rows(
            items=domain_row.get("short_term_interests", []),
            post_id_to_index=post_id_to_index,
            max_items=self.max_short_term,
            domain_pack=domain_pack,
        )
        negative_interests = self._select_interest_rows(
            items=domain_row.get("weak_or_uncertain_interests", []),
            post_id_to_index=post_id_to_index,
            max_items=self.max_negative,
            domain_pack=domain_pack,
        )
        weak_labels = self._collect_interest_labels(
            domain_row.get("weak_or_uncertain_interests", [])
        )

        domain_evidence_ids = self._derive_domain_evidence_ids(
            status=status,
            domain_row=domain_row,
            long_term=long_term,
            short_term=short_term,
            domain_pack=domain_pack,
        )
        domain_evidence_indices = self._map_post_ids_to_indices(
            domain_evidence_ids,
            post_id_to_index,
            limit=self.max_domain_evidence,
        )
        domain_all_evidence_ids = [
            str(x).strip()
            for x in domain_row.get("domain_all_evidence_post_ids", []) or []
            if str(x).strip()
        ] or domain_evidence_ids
        domain_representative_evidence_ids = [
            str(x).strip()
            for x in domain_row.get("domain_representative_evidence_post_ids", []) or []
            if str(x).strip()
        ] or domain_evidence_ids
        domain_all_evidence_indices = self._map_post_ids_to_indices(
            domain_all_evidence_ids,
            post_id_to_index,
            limit=None,
        )
        domain_representative_evidence_indices = self._map_post_ids_to_indices(
            domain_representative_evidence_ids,
            post_id_to_index,
            limit=None,
        )
        domain_evidence_source_summary = self._build_evidence_source_summary(
            evidence_post_ids=domain_all_evidence_ids,
            domain_pack=domain_pack,
            post_id_to_index=post_id_to_index,
        )

        summary_atomic = self._build_summary_atomic(
            domain=domain,
            status=status,
            long_term=long_term,
            short_term=short_term,
        )

        resolved_summary = str(summary_natural_gold or "").strip()
        if not resolved_summary and rewrite_enabled:
            resolved_summary = self._rewrite_summary_natural_gold(
                domain=domain,
                domain_row=domain_row,
                domain_pack=domain_pack,
                status=status,
                long_term=long_term,
                short_term=short_term,
                weak_labels=weak_labels,
                post_id_to_index=post_id_to_index,
            )
        if not resolved_summary:
            resolved_summary = self._soft_fallback_summary(
                domain=domain,
                domain_row=domain_row,
                status=status,
                summary_atomic=summary_atomic,
            )

        return {
            "domain": domain,
            "status": status,
            "long_term_interests": long_term if status == "active" else [],
            "short_term_interests": short_term if status == "active" else [],
            "negative_interests": negative_interests,
            "domain_all_evidence_post_indices": domain_all_evidence_indices,
            "domain_representative_evidence_post_indices": domain_representative_evidence_indices,
            "domain_evidence_post_indices": domain_evidence_indices,
            "domain_evidence_source_summary": domain_evidence_source_summary,
            "summary_atomic": summary_atomic,
            "summary_natural_gold": resolved_summary,
            "debug_meta": {
                "source_domain_summary": domain_row.get("domain_summary"),
                "uncertainty_note": domain_row.get("uncertainty_note"),
                "dropped_weak_interest_labels": weak_labels,
            },
        }

    @staticmethod
    def _derive_status(domain_row: dict[str, Any]) -> str:
        stable = domain_row.get("stable_interests", []) or []
        short_term = domain_row.get("short_term_interests", []) or []

        if stable or short_term:
            return "active"
        return "inactive"

    @staticmethod
    def _collect_interest_labels(items: list[dict[str, Any]]) -> list[str]:
        labels: list[str] = []
        for item in items or []:
            label = str(item.get("label") or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _select_interest_rows(
        self,
        *,
        items: list[dict[str, Any]],
        post_id_to_index: dict[str, int],
        max_items: int,
        domain_pack: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in (items or [])[:max_items]:
            label = str(item.get("label") or "").strip()
            if not label:
                continue

            evidence_post_ids = [
                str(x).strip()
                for x in item.get("all_evidence_post_ids", []) or []
                if str(x).strip()
            ]
            evidence_post_indices = self._map_post_ids_to_indices(
                evidence_post_ids,
                post_id_to_index,
                limit=self.max_interest_evidence_indices,
            )
            rows.append(
                {
                    "label": label,
                    "evidence_post_ids": evidence_post_ids,
                    "evidence_post_indices": evidence_post_indices,
                    "evidence_source_summary": self._build_evidence_source_summary(
                        evidence_post_ids=evidence_post_ids,
                        domain_pack=domain_pack,
                        post_id_to_index=post_id_to_index,
                    ),
                }
            )
        return rows

    def _derive_domain_evidence_ids(
        self,
        *,
        status: str,
        domain_row: dict[str, Any],
        long_term: list[dict[str, Any]],
        short_term: list[dict[str, Any]],
        domain_pack: dict[str, Any],
    ) -> list[str]:
        ids: list[str] = []

        if status == "active":
            for item in long_term + short_term:
                ids.extend(item.get("evidence_post_ids", []))
        else:
            for item in domain_row.get("weak_or_uncertain_interests", []) or []:
                ids.extend(
                    str(x).strip()
                    for x in item.get("all_evidence_post_ids", []) or []
                    if str(x).strip()
                )

        if not ids:
            for row in domain_pack.get("representative_posts", []) or []:
                pid = str(row.get("post_id", "")).strip()
                if pid:
                    ids.append(pid)

        deduped: list[str] = []
        for pid in ids:
            if pid and pid not in deduped:
                deduped.append(pid)
        return deduped[: self.max_domain_evidence]

    @staticmethod
    def _map_post_ids_to_indices(
        post_ids: list[str],
        post_id_to_index: dict[str, int],
        limit: int | None,
    ) -> list[int]:
        out: list[int] = []
        for pid in post_ids:
            idx = post_id_to_index.get(pid)
            if idx is None:
                continue
            if idx not in out:
                out.append(idx)
            if limit is not None and len(out) >= limit:
                break
        return out

    @staticmethod
    def _build_summary_atomic(
        *,
        domain: str,
        status: str,
        long_term: list[dict[str, Any]],
        short_term: list[dict[str, Any]],
    ) -> str:
        if status == "inactive":
            return f"In {domain}, no reliable interest can be established from the observed posts."

        long_labels = [str(x.get("label") or "").strip() for x in long_term if str(x.get("label") or "").strip()]
        short_labels = [str(x.get("label") or "").strip() for x in short_term if str(x.get("label") or "").strip()]

        if long_labels and short_labels:
            return (
                f"In {domain}, the user shows a long-term interest in "
                f"{', '.join(long_labels)}. "
                f"Recent or emerging interests include {', '.join(short_labels)}."
            )

        if long_labels:
            return (
                f"In {domain}, the user shows a long-term interest in "
                f"{', '.join(long_labels)}."
            )

        if short_labels:
            return (
                f"In {domain}, the strongest identifiable signal is a recent or emerging interest in "
                f"{', '.join(short_labels)}."
            )

        return f"In {domain}, no reliable interest can be established from the observed posts."

    def _rewrite_summary_natural_gold(
        self,
        *,
        domain: str,
        domain_row: dict[str, Any],
        domain_pack: dict[str, Any],
        status: str,
        long_term: list[dict[str, Any]],
        short_term: list[dict[str, Any]],
        weak_labels: list[str],
        post_id_to_index: dict[str, int],
    ) -> str:
        if self.client is None:
            return self._soft_fallback_summary(
                domain=domain,
                domain_row=domain_row,
                status=status,
                summary_atomic=self._build_summary_atomic(
                    domain=domain,
                    status=status,
                    long_term=long_term,
                    short_term=short_term,
                ),
            )

        rewrite_input = self._build_rewrite_input(
            domain=domain,
            domain_row=domain_row,
            domain_pack=domain_pack,
            status=status,
            long_term=long_term,
            short_term=short_term,
            weak_labels=weak_labels,
            post_id_to_index=post_id_to_index,
        )

        result = self.client.chat_json(
            system_prompt=BENCHMARK_GOLD_REWRITE_SYSTEM_PROMPT,
            user_prompt=build_benchmark_gold_rewrite_user_prompt(
                json.dumps(rewrite_input, ensure_ascii=False, indent=2)
            ),
            json_schema_hint=BENCHMARK_GOLD_REWRITE_JSON_SCHEMA_HINT,
        )

        summary = str(result.get("summary_natural_gold", "")).strip()
        if not summary:
            return self._soft_fallback_summary(
                domain=domain,
                domain_row=domain_row,
                status=status,
                summary_atomic=self._build_summary_atomic(
                    domain=domain,
                    status=status,
                    long_term=long_term,
                    short_term=short_term,
                ),
            )
        return summary

    def _build_rewrite_input(
        self,
        *,
        domain: str,
        domain_row: dict[str, Any],
        domain_pack: dict[str, Any],
        status: str,
        long_term: list[dict[str, Any]],
        short_term: list[dict[str, Any]],
        weak_labels: list[str],
        post_id_to_index: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "domain": domain,
            "domain_definition": domain_pack.get("domain_definition"),
            "status": status,
            "long_term_interests": self._attach_evidence_examples(
                long_term,
                domain_pack=domain_pack,
                post_id_to_index=post_id_to_index,
            ),
            "short_term_interests": self._attach_evidence_examples(
                short_term,
                domain_pack=domain_pack,
                post_id_to_index=post_id_to_index,
            ),
            # Internal hints for weak-domain rewriting only; not part of formal export schema.
            "weak_signal_labels_internal": weak_labels[:2],
            "weak_signal_examples": self._collect_weak_signal_examples(
                domain_row=domain_row,
                domain_pack=domain_pack,
                post_id_to_index=post_id_to_index,
            ),
            "source_domain_summary": domain_row.get("domain_summary"),
            "uncertainty_note": domain_row.get("uncertainty_note"),
        }

    def _attach_evidence_examples(
        self,
        items: list[dict[str, Any]],
        *,
        domain_pack: dict[str, Any],
        post_id_to_index: dict[str, int],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            out.append(
                {
                    **item,
                    "evidence_examples": self._lookup_evidence_examples(
                        evidence_post_ids=item.get("evidence_post_ids", []),
                        domain_pack=domain_pack,
                        post_id_to_index=post_id_to_index,
                        limit=self.max_evidence_examples_per_interest,
                    ),
                }
            )
        return out

    def _collect_weak_signal_examples(
        self,
        *,
        domain_row: dict[str, Any],
        domain_pack: dict[str, Any],
        post_id_to_index: dict[str, int],
    ) -> list[dict[str, Any]]:
        weak_items = domain_row.get("weak_or_uncertain_interests", []) or []
        weak_ids: list[str] = []
        for item in weak_items[:2]:
            weak_ids.extend(
                str(x).strip()
                for x in item.get("all_evidence_post_ids", []) or []
                if str(x).strip()
            )
        return self._lookup_evidence_examples(
            evidence_post_ids=weak_ids,
            domain_pack=domain_pack,
            post_id_to_index=post_id_to_index,
            limit=2,
        )

    def _lookup_evidence_examples(
        self,
        *,
        evidence_post_ids: list[str],
        domain_pack: dict[str, Any],
        post_id_to_index: dict[str, int],
        limit: int,
    ) -> list[dict[str, Any]]:
        representative_by_id = {
            str(row.get("post_id", "")).strip(): row
            for row in domain_pack.get("representative_posts", []) or []
            if str(row.get("post_id", "")).strip()
        }

        cluster_examples_by_id: dict[str, dict[str, Any]] = {}
        for cluster in domain_pack.get("tag_clusters", []) or []:
            for ex in cluster.get("evidence_examples", []) or []:
                pid = str(ex.get("post_id", "")).strip()
                if pid and pid not in cluster_examples_by_id:
                    cluster_examples_by_id[pid] = ex

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pid in evidence_post_ids:
            if not pid or pid in seen:
                continue
            seen.add(pid)

            rep = representative_by_id.get(pid)
            if rep:
                rows.append(
                    {
                        "post_id": pid,
                        "post_index": post_id_to_index.get(pid),
                        "created_at": rep.get("created_at"),
                        "text": rep.get("text"),
                        "relative_recency": rep.get("relative_recency"),
                        "tags": rep.get("tags"),
                    }
                )
            else:
                ex = cluster_examples_by_id.get(pid)
                if ex:
                    rows.append(
                        {
                            "post_id": pid,
                            "post_index": post_id_to_index.get(pid),
                            "created_at": ex.get("created_at"),
                            "text": ex.get("text"),
                            "relative_recency": None,
                            "tags": [],
                        }
                    )

            if len(rows) >= limit:
                break
        return rows

    def _build_evidence_source_summary(
        self,
        *,
        evidence_post_ids: list[str],
        domain_pack: dict[str, Any],
        post_id_to_index: dict[str, int],
    ) -> dict[str, Any]:
        post_sources = {
            str(row.get("post_id", "")).strip(): row
            for row in domain_pack.get("post_signal_sources", []) or []
            if str(row.get("post_id", "")).strip()
        }
        modalities: list[str] = []
        breakdown: dict[str, list[str]] = {
            "text_only_post_ids": [],
            "visual_only_post_ids": [],
            "mixed_post_ids": [],
        }
        for pid in evidence_post_ids:
            if not pid:
                continue
            source_row = post_sources.get(pid, {})
            for modality in source_row.get("evidence_modalities", []) or []:
                token = str(modality).strip()
                if token and token not in modalities:
                    modalities.append(token)
            bucket = str(source_row.get("evidence_modality") or "").strip().lower()
            key = f"{bucket}_post_ids"
            if key not in breakdown:
                continue
            if pid not in breakdown[key]:
                breakdown[key].append(pid)

        out: dict[str, Any] = {"modalities": modalities}
        for key, post_ids in breakdown.items():
            if not post_ids:
                continue
            out[key] = post_ids
            indices = self._map_post_ids_to_indices(post_ids, post_id_to_index, limit=None)
            out[key.replace("_post_ids", "_post_indices")] = indices
        return out

    @staticmethod
    def _soft_fallback_summary(
        *,
        domain: str,
        domain_row: dict[str, Any],
        status: str,
        summary_atomic: str,
    ) -> str:
        source = str(domain_row.get("domain_summary") or "").strip()
        uncertainty = str(domain_row.get("uncertainty_note") or "").strip()

        if status == "active":
            if source:
                return source
            return summary_atomic

        # inactive
        if source:
            return source
        if uncertainty:
            return f"This domain does not currently provide a reliable profile signal. {uncertainty}"
        return "This domain does not currently provide a reliable profile signal."

    @staticmethod
    def _is_bad_rewrite(summary: str) -> bool:
        return not bool(str(summary or "").strip())
