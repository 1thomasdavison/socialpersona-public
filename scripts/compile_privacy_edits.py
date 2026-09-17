#!/usr/bin/env python3
"""Compile reviewed literal edits into private, reproducible rewrite rules."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

from user_profile_pipeline.release_privacy import read_json, write_json


def compile_rules(review_dirs, overrides_path, output, supplemental=None):
    overrides = read_json(overrides_path)
    edits = defaultdict(dict)
    record_texts = {}
    counts = Counter()
    decisions_log = []
    seen = set()
    for root in review_dirs:
        proposals = {x["key"]: x for x in read_json(root / "proposals.json")}
        decisions = []
        for path in sorted(root.glob("batch_*.json")):
            batch = read_json(path)
            if not batch.get("validated"):
                raise ValueError("Unvalidated editorial response")
            decisions.extend(batch["decisions"])
        if len(decisions) != len(proposals) or {d["key"] for d in decisions} != set(proposals):
            raise ValueError("Editorial coverage is incomplete")
        for decision in decisions:
            key = decision["key"]
            if key in seen:
                raise ValueError("Duplicate proposal across review directories")
            seen.add(key)
            proposal = proposals[key]
            override = overrides.get(key, {})
            action = override.get("action", decision["action"])
            counts[action] += 1
            counts["editorial_overrides"] += bool(override)
            decisions_log.append({"key": key, "action": action, "reason": override.get("reason", decision["reason"])})
            if action != "accept":
                continue
            original = proposal["text"]
            rid = proposal["record_id"]
            record_texts[rid] = original
            span = proposal["proposed"]
            term, replacement = span["find"], override.get("replace", span["replace"])
            if not term or term not in original or not isinstance(replacement, str):
                raise ValueError("Invalid approved literal edit")
            # Identical prose may occur in multiple fields; one consistent edit.
            prior = edits[original].get(term)
            if prior is not None and prior != replacement:
                raise ValueError("Conflicting replacement for identical text")
            edits[original][term] = replacement
    unknown_overrides = set(overrides) - seen
    if unknown_overrides:
        raise ValueError("Editorial overrides refer to missing proposals")
    rewrites = {}
    for original, replacements in edits.items():
        # One pass avoids processing newly inserted placeholders a second time.
        pattern = re.compile("|".join(re.escape(s) for s in sorted(replacements, key=len, reverse=True)))
        rewritten = pattern.sub(lambda m: replacements[m.group()], original)
        if rewritten != original:
            rewrites[original] = rewritten
    counts["rewritten_unique_strings"] = len(rewrites)
    extra = read_json(supplemental) if supplemental else {"replace": {}, "rewrite": {}}
    for key in ("replace", "rewrite"):
        if not isinstance(extra.get(key), dict) or not all(isinstance(k, str) and k and isinstance(v, str) for k, v in extra[key].items()):
            raise ValueError("Invalid supplemental editorial rules")
    rewrites.update(extra["rewrite"])
    write_json(output, {"replace": extra["replace"], "rewrite": rewrites, "review_counts": dict(counts)})
    write_json(output.with_name(output.stem + "_decisions.json"), decisions_log)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path)
    args = parser.parse_args()
    print(compile_rules(args.review_dirs, args.overrides, args.output, args.supplemental))


if __name__ == "__main__":
    main()
