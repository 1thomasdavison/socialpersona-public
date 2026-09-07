"""Conservative, local-only public derivatives; never copy source free text.

This is data minimization, not a proof of anonymity or a faithful paraphraser.
The fixed vocabulary and schema are the disclosure boundary. Source strings may
select a vocabulary entry, but may never become public string values.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets

VERSION = "privacy-derived-v1"
PHASES = 4
MIN_USERS = 5
MAX_TOPICS = 5
BUCKETS = ("long_term_interests", "short_term_interests", "negative_interests")
MODELS = (
    "gemini-2.5-flash", "gpt-4o-mini", "gpt-5.4",
    "qwen2.5-vl-7b-instruct", "qwen3-vl-8b-instruct", "qwen3.5-35b-a3b",
)
# Intentionally no people, places, organizations, titles, dates, demographics,
# health, religion, politics, intimate information, or identifying OCR strings.
VOCAB = {
    "sports_outdoor": {
        "outdoor activities": r"outdoor|recreation|camping|campground",
        "walking and hiking": r"walk(?:ing|s)?|hik(?:e|es|ing)|trail|trekking",
        "running": r"running|jogging|marathon",
        "cycling": r"cycling|bicycl\w*|biking",
        "water sports": r"swimming|surfing|kayaking|paddling|snorkeling",
        "fitness": r"fitness|workout|gym|weightlifting|yoga",
        "spectator sports": r"sports?|football|soccer|basketball|baseball|tennis|wrestling|racing",
        "nature observation": r"wildlife|birdwatching|stargazing|aurora|nature|garden(?:ing)?",
    },
    "entertainment": {
        "music": r"music|songs?|singing|singer|album|guitar|piano",
        "live performances": r"concert|performance|theater|theatre|musical|festival",
        "film and television": r"movie|film|television|cinema|sitcom|tv|anime",
        "reading": r"reading|books?|novels?|literature|comic|manga",
        "comedy": r"comedy|humou?r|comedian|jokes?",
    },
    "gaming": {
        "video games": r"video games?|gaming|videogames?|gameplay|console",
        "strategy games": r"strategy games?|turn.based|tactical games?",
        "role-playing games": r"role.playing|rpg|mmorpg",
        "board and card games": r"board games?|card games?|tabletop|puzzles?",
        "competitive gaming": r"esports?|competitive gaming|speedrun",
    },
    "food_drink": {
        "cooking": r"cooking|recipes?|homemade|home.made|meal preparation",
        "dining": r"dining|restaurant|cafe|caf\u00e9|breakfast|lunch|dinner|meal|food",
        "baking and desserts": r"baking|dessert|cakes?|pastry|pastries|cookies?|ice cream|chocolate",
        "coffee and tea": r"coffee|tea|espresso|latte",
        "savory food": r"savory|pizza|pasta|noodles?|bread|rice|soup|sandwich|barbecue",
    },
    "travel_city_exploration": {
        "travel": r"travel|tourism|tourist|vacation|holiday|sightseeing|trips?",
        "city exploration": r"city|urban|architecture|skyline|street|neighborhood",
        "cultural visits": r"museum|exhibition|gallery|historic|heritage",
        "nature visits": r"landscape|mountain|beach|forest|lake|ocean|waterfall|desert",
        "cruise travel": r"cruise|cruising",
    },
    "photography_creation": {
        "photography": r"photograph\w*|camera|photo|photos",
        "visual art": r"art|painting|drawing|illustration|sketch|sculpture",
        "video creation": r"videography|filmmaking|video editing|animation",
        "crafts and design": r"crafts?|knitting|sewing|design|ceramics|pottery",
    },
    "pets": {
        "dogs": r"dogs?|pupp(?:y|ies)|canine",
        "cats": r"cats?|kittens?|feline",
        "pet care": r"pets?|pet care|animal care|grooming",
    },
}
FALLBACK = {domain: domain.replace("_", " ") + " interests" for domain in VOCAB}
PATTERNS = {
    label: re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.I)
    for terms in VOCAB.values() for label, pattern in terms.items()
}
NOISE = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|[@#][\w]+|\[[^\]]*\]", re.I)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_posts(path: Path):
    # JSON strings can contain U+2028; str.splitlines() corrupts such JSONL.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def extract_topics(text: str, domain: str | None = None) -> set[str]:
    cleaned = NOISE.sub(" ", str(text))
    labels = VOCAB[domain] if domain else PATTERNS
    return {label for label in labels if PATTERNS[label].search(cleaned)}


def topic_summary(topics: list[str], kind: str) -> str:
    return f"Generalized {kind} topics: " + ("; ".join(topics) if topics else "content withheld") + "."


def source_phase_map(posts: list[dict]) -> tuple[dict[str, int], dict[int, int]]:
    if len(posts) < PHASES:
        raise ValueError("At least four source records are required")
    dates = []
    for post in posts:
        date = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
        dates.append((date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp())
    order = sorted(range(len(posts)), key=lambda i: (dates[i], i))
    by_index = {index: min(PHASES - 1, rank * PHASES // len(posts)) for rank, index in enumerate(order)}
    ids = [p["post_id"] for p in posts]
    if len(set(ids)) != len(ids) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("Missing or duplicate source post IDs")
    return {p["post_id"]: by_index[i] for i, p in enumerate(posts)}, by_index


def resolved_phases(interest: dict, by_id: dict, by_index: dict, audit: Counter) -> list[int]:
    ids = interest.get("evidence_post_ids", [])
    indices = interest.get("evidence_post_indices", [])
    id_phases = {by_id[x] for x in ids if x in by_id}
    index_phases = {by_index[x] for x in indices if type(x) is int and x in by_index}
    audit["unresolved_source_evidence_ids"] += sum(x not in by_id for x in ids)
    audit["invalid_source_evidence_indices"] += sum(type(x) is not int or x not in by_index for x in indices)
    if ids and id_phases != index_phases:
        audit["source_evidence_phase_disagreements"] += 1
    # Explicit IDs are authoritative. Do not silently attach missing IDs to
    # unrelated index positions. Indices are used only if IDs are absent.
    return sorted(id_phases if ids else index_phases)


def build_release(source: Path, destination: Path, mapping_path: Path, expected_users: int = 100) -> dict:
    source, destination, mapping_path = source.resolve(), destination.resolve(), mapping_path.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and release must be separate directory trees")
    if mapping_path == destination or destination in mapping_path.parents:
        raise ValueError("Identity mapping must stay outside the release")
    if (destination / "data").exists() or mapping_path.exists():
        raise ValueError("Refusing to overwrite an existing release or private mapping")
    dirs = sorted(p for p in source.iterdir() if p.is_dir())
    if len(dirs) != expected_users:
        raise ValueError(f"Expected {expected_users} users, got {len(dirs)}")
    prepared = []
    support = {label: set() for label in PATTERNS}
    audit = Counter()
    for d in dirs:
        posts = read_posts(d / "posts.jsonl")
        profile = read_json(d / "gold_profile.json")
        captions = read_json(d / "image_captions.json")
        by_id, by_index = source_phase_map(posts)
        phase_topics = [Counter() for _ in range(PHASES)]
        model_topics = {model: Counter() for model in MODELS}
        all_topics = set()
        audit["source_users"] += 1
        audit["source_posts"] += len(posts)
        for i, post in enumerate(posts):
            text = post.get("text", "")
            topics = extract_topics(text)
            phase_topics[by_index[i]].update(topics)
            all_topics.update(topics)
            audit["posts_with_mentions"] += bool(re.search(r"@[\w]+", text))
            audit["posts_with_hashtags"] += bool(re.search(r"#[\w]+", text))
            audit["exact_timestamps_removed"] += bool(post.get("created_at"))
            audit["media_hashes_removed"] += sum(bool(m.get("sha256")) for m in post.get("media", []))
        audit["source_images"] += len(captions["captions"])
        for entry in captions["captions"].values():
            for model, value in entry["model_captions"].items():
                if model not in MODELS:
                    raise ValueError("Unexpected caption model")
                topics = extract_topics(value.get("summary", ""))
                model_topics[model].update(topics)
                all_topics.update(topics)
                audit["source_captions"] += 1
        if {dom["domain"] for dom in profile["domains"]} != set(VOCAB) or len(profile["domains"]) != len(VOCAB):
            raise ValueError("Unexpected domain schema")
        for dom in profile["domains"]:
            if dom["status"] not in ("active", "inactive"):
                raise ValueError("Unexpected domain status")
            for bucket in BUCKETS:
                for interest in dom[bucket]:
                    all_topics.update(extract_topics(interest["label"], dom["domain"]))
                    audit["source_interest_labels"] += 1
        for topic in all_topics:
            support[topic].add(d.name)
        prepared.append((d.name, posts, profile, phase_topics, model_topics, by_id, by_index))
    allowed = {label for label, users in support.items() if len(users) >= MIN_USERS}
    # A new random assignment, never the old handle-sorted sequence or a hash.
    secrets.SystemRandom().shuffle(prepared)
    mapping = []
    def select(counter):
        ranked = sorted((label for label in counter if label in allowed), key=lambda label: (-counter[label], label))
        return sorted(ranked[:MAX_TOPICS])
    for serial, (old_id, _, profile, phase_topics, model_topics, by_id, by_index) in enumerate(prepared, 1):
        uid = f"participant_{serial:03d}"
        mapping.append({"source_id": old_id, "public_id": uid})
        out = destination / "data" / "users" / uid
        out.mkdir(parents=True)
        rows = []
        for phase in range(PHASES):
            topics = select(phase_topics[phase])
            rows.append({"user_id": uid, "post_id": f"{uid}_phase_{phase + 1}",
                         "relative_phase": phase + 1, "record_type": "aggregated_topic_summary",
                         "topics": topics, "text": topic_summary(topics, "text"), "media": []})
        (out / "posts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
        result = {"user_id": uid, "release_variant": VERSION, "domains": []}
        for dom in profile["domains"]:
            name = dom["domain"]
            item = {"domain": name, "status": dom["status"]}
            for bucket in BUCKETS:
                merged = {}
                for interest in dom[bucket]:
                    labels = sorted(extract_topics(interest["label"], name) & allowed)
                    if not labels:
                        labels = [FALLBACK[name]]
                        audit["labels_generalized_to_domain"] += 1
                    phases = resolved_phases(interest, by_id, by_index, audit)
                    for label in labels:
                        merged.setdefault(label, set()).update(phases)
                item[bucket] = [
                    {"label": label, "evidence_post_indices": sorted(phases),
                     "evidence_post_ids": [rows[i]["post_id"] for i in sorted(phases)]}
                    for label, phases in sorted(merged.items())
                ]
                audit["released_interest_labels"] += len(item[bucket])
            result["domains"].append(item)
        write_json(out / "gold_profile.json", result)
        summaries = {}
        for model in MODELS:
            topics = select(model_topics[model])
            summaries[model] = {"topics": topics, "summary": topic_summary(topics, "visual")}
        write_json(out / "image_captions.json", {"user_id": uid, "release_variant": VERSION,
                   "aggregation": "user_level", "model_summaries": summaries})
    write_json(mapping_path, {"release_variant": VERSION, "mapping": mapping})
    write_json(destination / "data" / "vocabulary.json", {
        "release_variant": VERSION, "minimum_source_users_per_topic": MIN_USERS,
        "domains": {domain: sorted(set(terms) & allowed) + [FALLBACK[domain]] for domain, terms in VOCAB.items()},
    })
    report = {"release_variant": VERSION, "source_audit": dict(sorted(audit.items())),
              "public_counts": {"users": len(prepared), "timeline_summaries": len(prepared) * PHASES,
                                "caption_model_summaries": len(prepared) * len(MODELS)},
              "minimum_source_users_per_topic": MIN_USERS,
              "allowed_topic_count": len(allowed), "not_an_anonymity_guarantee": True,
              "paper_scores_reproducible_from_derivative": False}
    write_json(destination / "data" / "privacy_audit.json", report)
    write_manifest(destination)
    return report


def write_manifest(root: Path):
    files = sorted(p for p in (root / "data").rglob("*") if p.is_file() and p.name != "manifest.json")
    write_json(root / "data" / "manifest.json", {"release_variant": VERSION, "files": {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})


def verify_data(root: Path, expected_users: int = 100) -> list[str]:
    """Strict allowlist validation: additional keys and arbitrary text fail closed."""
    issues = []
    def check(condition, message):
        if not condition:
            issues.append(message)
    def keys(value, expected, where):
        check(isinstance(value, dict) and set(value) == set(expected), where + ": unexpected fields")
    def topics(value, where):
        check(isinstance(value, list) and len(value) <= MAX_TOPICS and value == sorted(set(value))
              and all(x in PATTERNS for x in value), where + ": invalid vocabulary")
    try:
        data = root / "data"
        check({p.name for p in data.iterdir()} <= {"users", "vocabulary.json", "privacy_audit.json", "manifest.json", "README.md"}, "Unexpected data artifacts")
        vocabulary = read_json(data / "vocabulary.json")
        keys(vocabulary, ("release_variant", "minimum_source_users_per_topic", "domains"), "vocabulary")
        check(vocabulary["release_variant"] == VERSION and vocabulary["minimum_source_users_per_topic"] == MIN_USERS, "Invalid vocabulary metadata")
        keys(vocabulary["domains"], VOCAB, "vocabulary domains")
        allowed = set()
        for domain, labels in vocabulary["domains"].items():
            permitted = set(VOCAB[domain]) | {FALLBACK[domain]}
            check(isinstance(labels, list) and len(labels) == len(set(labels)) and set(labels) <= permitted and FALLBACK[domain] in labels, "Invalid domain vocabulary")
            allowed.update(labels)
        report = read_json(data / "privacy_audit.json")
        keys(report, ("release_variant", "source_audit", "public_counts", "minimum_source_users_per_topic", "allowed_topic_count", "not_an_anonymity_guarantee", "paper_scores_reproducible_from_derivative"), "privacy audit")
        check(report["release_variant"] == VERSION and report["minimum_source_users_per_topic"] == MIN_USERS
              and report["allowed_topic_count"] == len(allowed - set(FALLBACK.values()))
              and report["not_an_anonymity_guarantee"] is True and report["paper_scores_reproducible_from_derivative"] is False, "Invalid audit declarations")
        check(report["public_counts"] == {"users": expected_users, "timeline_summaries": expected_users * PHASES, "caption_model_summaries": expected_users * len(MODELS)}, "Invalid public counts")
        audit_keys = {"source_users", "source_posts", "source_images", "source_captions", "source_interest_labels", "posts_with_mentions", "posts_with_hashtags", "exact_timestamps_removed", "media_hashes_removed", "labels_generalized_to_domain", "unresolved_source_evidence_ids", "invalid_source_evidence_indices", "source_evidence_phase_disagreements", "released_interest_labels"}
        check(set(report["source_audit"]) <= audit_keys and all(type(n) is int and n >= 0 for n in report["source_audit"].values()), "Invalid source audit fields")
        users = root / "data" / "users"
        actual = {p.name for p in users.iterdir()}
        check(actual == {f"participant_{i:03d}" for i in range(1, expected_users + 1)}, "Unexpected user set")
        for d in sorted(users.iterdir()):
            if not d.is_dir():
                continue
            uid = d.name
            check({p.name for p in d.iterdir()} == {"posts.jsonl", "gold_profile.json", "image_captions.json"}, uid + ": unexpected files")
            rows = read_posts(d / "posts.jsonl")
            check(len(rows) == PHASES, uid + ": expected four summaries")
            for i, row in enumerate(rows):
                where = uid + f"/phase-{i + 1}"
                keys(row, ("user_id", "post_id", "relative_phase", "record_type", "topics", "text", "media"), where)
                topics(row["topics"], where)
                check(set(row["topics"]) <= allowed, where + ": suppressed topic")
                check(row["user_id"] == uid and row["post_id"] == f"{uid}_phase_{i + 1}", where + ": invalid ID")
                check(row["relative_phase"] == i + 1 and row["record_type"] == "aggregated_topic_summary" and row["media"] == [], where + ": invalid metadata")
                check(row["text"] == topic_summary(row["topics"], "text"), where + ": free text detected")
            profile = read_json(d / "gold_profile.json")
            keys(profile, ("user_id", "release_variant", "domains"), uid + "/profile")
            check(profile["user_id"] == uid and profile["release_variant"] == VERSION, uid + ": invalid profile metadata")
            check(len(profile["domains"]) == len(VOCAB) and {x["domain"] for x in profile["domains"]} == set(VOCAB), uid + ": invalid domains")
            for dom in profile["domains"]:
                keys(dom, ("domain", "status", *BUCKETS), uid + "/domain")
                check(dom["status"] in ("active", "inactive"), uid + ": invalid status")
                for bucket in BUCKETS:
                    seen = set()
                    for interest in dom[bucket]:
                        keys(interest, ("label", "evidence_post_indices", "evidence_post_ids"), uid + "/interest")
                        label = interest["label"]
                        check(label in allowed, uid + ": suppressed label")
                        check(label in VOCAB[dom["domain"]] or label == FALLBACK[dom["domain"]], uid + ": unapproved label")
                        check(label not in seen, uid + ": duplicate label")
                        seen.add(label)
                        idx = interest["evidence_post_indices"]
                        valid = idx == sorted(set(idx)) and all(type(i) is int and 0 <= i < PHASES for i in idx)
                        check(valid, uid + ": invalid evidence indices")
                        if valid:
                            check(interest["evidence_post_ids"] == [rows[i]["post_id"] for i in idx], uid + ": evidence ID/index disagreement")
            caps = read_json(d / "image_captions.json")
            keys(caps, ("user_id", "release_variant", "aggregation", "model_summaries"), uid + "/captions")
            check(caps["user_id"] == uid and caps["release_variant"] == VERSION and caps["aggregation"] == "user_level", uid + ": invalid captions metadata")
            keys(caps["model_summaries"], MODELS, uid + "/models")
            for value in caps["model_summaries"].values():
                keys(value, ("topics", "summary"), uid + "/summary")
                topics(value["topics"], uid + "/summary")
                check(set(value["topics"]) <= allowed, uid + ": suppressed caption topic")
                check(value["summary"] == topic_summary(value["topics"], "visual"), uid + ": free caption text detected")
        manifest = read_json(root / "data" / "manifest.json")
        expected = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root / "data").rglob("*") if p.is_file() and p.name != "manifest.json"}
        check(manifest == {"release_variant": VERSION, "files": expected}, "Data manifest mismatch")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        # Exception values may contain a source string: publish type only.
        issues.append("Invalid release structure: " + type(error).__name__)
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-users", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_release(args.source_users, args.release_dir, args.mapping_output)
    problems = verify_data(args.release_dir)
    if problems:
        raise SystemExit("Release validation failed: " + "; ".join(problems[:10]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
