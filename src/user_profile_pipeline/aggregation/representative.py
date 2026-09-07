from __future__ import annotations

from collections import defaultdict


def select_representative_posts(records: list[dict], limit: int = 15) -> list[dict]:
    if not records:
        return []
    sorted_records = sorted(
        records,
        key=lambda r: (
            -(r.get("confidence", 0.0) * r.get("post_signal_strength", 0.0)),
            r.get("created_at"),
        ),
    )
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for record in sorted_records:
        tags = record.get("canonical_tags") or ["__untagged__"]
        by_tag[tags[0]].append(record)

    picks: list[dict] = []
    seen_ids: set[str] = set()
    # coverage round
    for _, group in sorted(by_tag.items()):
        for item in group:
            if item["post_id"] not in seen_ids:
                picks.append(item)
                seen_ids.add(item["post_id"])
                break
            if len(picks) >= limit:
                return picks[:limit]

    # fill by strength and time diversity
    for item in sorted_records:
        if item["post_id"] in seen_ids:
            continue
        picks.append(item)
        seen_ids.add(item["post_id"])
        if len(picks) >= limit:
            break

    # if still too many, keep earliest/middle/latest diversity
    if len(picks) > limit:
        picks = sorted(picks, key=lambda x: x["created_at"])
        if limit >= 3:
            head = picks[: max(1, limit // 3)]
            mid_start = max(0, len(picks) // 2 - max(1, limit // 6))
            mid = picks[mid_start : mid_start + max(1, limit // 3)]
            tail = picks[-max(1, limit - len(head) - len(mid)) :]
            merged = []
            seen = set()
            for bucket in [head, mid, tail]:
                for x in bucket:
                    if x["post_id"] not in seen:
                        merged.append(x)
                        seen.add(x["post_id"])
            picks = merged[:limit]
    return picks[:limit]


def chunk_records(records: list[dict], chunk_size: int = 30) -> list[list[dict]]:
    ordered = sorted(records, key=lambda r: r["created_at"])
    return [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]
