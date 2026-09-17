"""Local, span-level de-identification of previously scrubbed records.

Rules and identity mappings are private inputs. Public entities are not private
merely because they are names. NER suggestions require contextual review; they
are deliberately not a blanket person/organization removal rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .release_privacy import BUCKETS, MODELS, VOCAB, read_json, read_posts, write_json

VERSION = "targeted-deidentified-v2"
TOKEN = re.compile(r"\[(?:PERSON|PET|PATH|LOCATION|ORGANIZATION|URL|ACCOUNT|EMAIL|PHONE|ADDRESS|DATE|COORDINATES|IDENTIFIER|PRIVATE_TEXT)\]")
PATTERNS = (
    ("email", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.I), "[EMAIL]"),
    ("url", re.compile(r"(?:https?://|www\.)[^\s<>\[\]]+", re.I), "[URL]"),
    ("account", re.compile(r"(?<!\w)@[\w.]+", re.UNICODE), "[ACCOUNT]"),
    ("coordinates", re.compile(r"(?<![\w.])-?\d{1,2}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}(?!\d)"), "[COORDINATES]"),
    ("date", re.compile(r"(?<!\d)(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?!\d)"), "[DATE]"),
    ("phone", re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\w)"), "[PHONE]"),
    ("identifier", re.compile(r"(?<!\w)(?:\+?\d{10,}|[a-fA-F0-9]{32,64})(?!\w)"), "[IDENTIFIER]"),
    ("address", re.compile(r"\b\d{1,6}\s+(?:[A-Za-z]+\s+){1,5}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr)\b(?:\.?\s*(?:Apt|Suite|Unit)\s*[\w-]+)?", re.I), "[ADDRESS]"),
)


CALENDAR_PATTERNS = (
    ("private_path", re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users)/)[^\s<>&]+"), "[PATH]"),
    ("date", re.compile(r"\[(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}\]"), "[DATE]"),
    ("date", re.compile(r"(?<!\w)(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:19|20)\d{2}(?!\w)"), "[DATE]"),
    ("date", re.compile(r"(?<!\d)\d{1,2}/\d{1,2}/\d{2}(?!\d)"), "[DATE]"),
    ("date", re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*[-–]\s*\d{1,2})?(?:,?\s+\d{4})?\b", re.I), "[DATE]"),
    ("date", re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?\b", re.I), "[DATE]"),
)

def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c) != "Cf")


def write_manifest(root):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((root / "data").rglob("*")) if p.is_file() and p != root / "data/manifest.json"}
    write_json(root / "data/manifest.json", {"release_variant": VERSION, "files": files})


class Scrubber:
    def __init__(self, rules=None, handles=(), *, calendar_dates=False):
        self.calendar_dates = calendar_dates
        rules = rules or {}
        self.counts = Counter()
        self.phrases = []
        # Longest first, and exact lexical boundaries: Ann must not match annual.
        replacements = dict(rules.get("replace", {}))
        replacements.update({h: "[ACCOUNT]" for h in handles if h})
        for term, replacement in sorted(replacements.items(), key=lambda x: -len(x[0])):
            if not term or not isinstance(replacement, str):
                raise ValueError("Invalid replacement rule")
            pattern = re.compile(r"(?<!\w)" + re.escape(normalize(term)) + r"(?!\w)", re.I)
            self.phrases.append((pattern, replacement))
        self.rewrites = rules.get("rewrite", {})
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.rewrites.items()):
            raise ValueError("Invalid exact rewrite rule")

    def __call__(self, value):
        if not isinstance(value, str):
            raise ValueError("Text must be a string")
        text = normalize(value)
        # Never treat existing placeholders as entities or match their contents.
        def clean(part):
            for name, pattern, replacement in PATTERNS:
                part, n = pattern.subn(replacement, part)
                self.counts[name] += n
            return part
        pieces = re.split("(" + TOKEN.pattern + ")", text)
        text = "".join(p if TOKEN.fullmatch(p) else clean(p) for p in pieces)
        # Keep semantic hashtags; remove only a marker attached to a replacement.
        text = re.sub(r"#(?=\[)", "", text)
        if text in self.rewrites:
            text = self.rewrites[text]
            self.counts["reviewed_rewrite"] += 1
            # A proposed rewrite cannot reintroduce direct identifiers.
            for _, pattern, replacement in PATTERNS:
                text = pattern.sub(replacement, text)
        for pattern, replacement in self.phrases:
            if not pattern.search(text):
                continue
            pieces = re.split("(" + TOKEN.pattern + ")", text)
            for i, part in enumerate(pieces):
                if not TOKEN.fullmatch(part):
                    pieces[i], n = pattern.subn(lambda _: replacement, part)
                    self.counts["keyword"] += n
            text = "".join(pieces)
        text = re.sub(r"#(?=\[)", "", text)
        if self.calendar_dates:
            for name, pattern, replacement in CALENDAR_PATTERNS:
                text, n = pattern.subn(replacement, text)
                self.counts[name] += n
        if text != value:
            self.counts["changed_strings"] += 1
        self.counts["strings"] += 1
        return text


def build_candidate(source, destination, mapping_path, rules_path, handles_path, expected_users=100):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    inputs = [Path(p).resolve() for p in (mapping_path, rules_path, handles_path)]
    if destination.exists():
        raise ValueError("Destination must not exist; never overwrite a release")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate trees")
    if any(destination == p or destination in p.parents for p in inputs):
        raise ValueError("Private inputs must remain outside the candidate")
    mapping = read_json(inputs[0])["mapping"]
    assignments = {x["source_id"]: x["public_id"] for x in mapping}
    folders = sorted(p for p in source.iterdir() if p.is_dir())
    if len(folders) != expected_users or set(assignments) != {d.name for d in folders}:
        raise ValueError("User set differs from the reviewed mapping")
    if len(set(assignments.values())) != expected_users or any(not re.fullmatch(r"participant_\d{3}", x) for x in assignments.values()):
        raise ValueError("Invalid public IDs")
    rules = read_json(inputs[1])
    handles = []
    for row in read_json(inputs[2])["mapping"]:
        handle = row["original_id"]
        handles.extend([handle, handle.removeprefix("x_")])
    total = Counter()
    source_hashes = {}
    destination.mkdir(parents=True)
    for folder in folders:
        uid = assignments[folder.name]
        scoped = rules.get("users", {}).get(folder.name, {})
        effective = {k: {**rules.get(k, {}), **scoped.get(k, {})} for k in ("replace", "rewrite")}
        scrub = Scrubber(effective, handles, calendar_dates=True)
        posts = read_posts(folder / "posts.jsonl")
        profile = read_json(folder / "gold_profile.json")
        captions = read_json(folder / "image_captions.json")
        for name in ("posts.jsonl", "gold_profile.json", "image_captions.json"):
            source_hashes[f"{folder.name}/{name}"] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        old_ids = [p["post_id"] for p in posts]
        if len(set(old_ids)) != len(old_ids) or any(not x for x in old_ids):
            raise ValueError("Missing or duplicate post ID")
        index = {p: i for i, p in enumerate(old_ids)}
        ids = {p: f"{uid}_post_{i+1:04d}" for p, i in index.items()}
        dates = [date.fromisoformat(p["created_at"][:10]) for p in posts]
        origin = min(dates)
        out = destination / "data" / "users" / uid
        out.mkdir(parents=True)
        rows = [{"user_id": uid, "post_id": ids[p["post_id"]],
                 "relative_day": (dates[i] - origin).days,
                 "post_type": p.get("post_type"), "language": p.get("language"),
                 "text": scrub(p.get("text", "")), "media": []}
                for i, p in enumerate(posts)]
        (out / "posts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rows), encoding="utf-8", newline="\n")
        gold = {"user_id": uid, "release_variant": VERSION, "domains": []}
        location_labels = {}
        if len(profile["domains"]) != len(VOCAB) or {d["domain"] for d in profile["domains"]} != set(VOCAB):
            raise ValueError("Invalid source domains")
        for domain in profile["domains"]:
            if domain["status"] not in ("active", "inactive"):
                raise ValueError("Invalid domain status")
            dom = {"domain": domain["domain"], "status": domain["status"]}
            for bucket in BUCKETS:
                dom[bucket] = []
                for interest in domain[bucket]:
                    evidence = interest.get("evidence_post_ids", [])
                    if evidence:
                        if any(e not in index for e in evidence):
                            raise ValueError("Unresolved evidence IDs")
                        positions = sorted({index[e] for e in evidence})
                        total["repaired_evidence_indices"] += positions != sorted(set(interest.get("evidence_post_indices", [])))
                    else:
                        positions = sorted(set(interest.get("evidence_post_indices", [])))
                        if any(type(i) is not int or not 0 <= i < len(posts) for i in positions):
                            raise ValueError("Invalid evidence index")
                    label = scrub(interest["label"])
                    if label.strip() == "[LOCATION]" and domain["domain"] == "travel_city_exploration":
                        area = location_labels.setdefault(interest["label"], len(location_labels) + 1)
                        label = f"exploring local area {area}"
                    total["changed_labels"] += label != interest["label"]
                    dom[bucket].append({"label": label, "evidence_post_indices": positions,
                                        "evidence_post_ids": [rows[i]["post_id"] for i in positions]})
                    total["interest_entries"] += 1
            gold["domains"].append(dom)
        write_json(out / "gold_profile.json", gold)
        images = {}
        for serial, item in enumerate(captions["captions"].values(), 1):
            model_captions = {}
            for model, entry in item["model_captions"].items():
                if model not in MODELS:
                    raise ValueError("Unexpected model")
                model_captions[model] = {"summary": scrub(entry["summary"])}
                total["captions"] += 1
            images[f"image_{serial:05d}"] = {"model_captions": model_captions}
        write_json(out / "image_captions.json", {"user_id": uid, "release_variant": VERSION,
                   "post_association": "unavailable", "captions": images})
        total["users"] += 1
        total["posts"] += len(rows)
        total["images"] += len(images)
        total.update(scrub.counts)
    report = {"release_variant": VERSION, "counts": dict(total),
              "privacy_review": "local_candidate", "paper_scores_reproduced": False}
    write_json(destination / "data" / "privacy_audit.json", report)
    write_manifest(destination)
    # Source hashes and rule contents stay beside private rules, never in output.
    write_json(inputs[1].parent / (destination.name + "_provenance.json"), {
        "source_files": source_hashes, "rules_sha256": hashlib.sha256(inputs[1].read_bytes()).hexdigest()})
    errors = verify_candidate(destination, expected_users)
    if errors:
        raise ValueError("Candidate validation failed: " + "; ".join(errors[:3]))
    return report


def verify_candidate(root, expected_users=100):
    """Check format, direct identifiers, checksums and evidence; not anonymity."""
    root = Path(root)
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    def prose(value):
        check(isinstance(value, str), "Non-string prose")
        if isinstance(value, str):
            for name, pattern, _ in (*PATTERNS, *CALENDAR_PATTERNS):
                check(not pattern.search(value), "Residual " + name)
    try:
        folders = sorted((root / "data/users").iterdir())
        check(len(folders) == expected_users, "Wrong user count")
        actual = Counter()
        for folder in folders:
            uid = folder.name
            check(bool(re.fullmatch(r"participant_\d{3}", uid)), "Invalid participant")
            check({p.name for p in folder.iterdir()} == {"posts.jsonl", "gold_profile.json", "image_captions.json"}, "Unexpected user file")
            rows = read_posts(folder / "posts.jsonl")
            check(bool(rows), "Empty timeline")
            for i, row in enumerate(rows):
                check(set(row) == {"user_id", "post_id", "relative_day", "post_type", "language", "text", "media"}, "Invalid post schema")
                check(row["user_id"] == uid and row["post_id"] == f"{uid}_post_{i+1:04d}", "Invalid post ID")
                check(type(row["relative_day"]) is int and row["relative_day"] >= 0 and row["media"] == [], "Invalid post metadata")
                check(row["post_type"] in (None, "original", "reply", "quote", "repost", "retweet"), "Invalid post type")
                check(row["language"] is None or bool(re.fullmatch(r"[a-z]{2,3}", row["language"])), "Invalid language code")
                prose(row["text"])
            gold = read_json(folder / "gold_profile.json")
            check(set(gold) == {"user_id", "release_variant", "domains"} and gold["user_id"] == uid and gold["release_variant"] == VERSION, "Invalid profile")
            check(len(gold["domains"]) == len(VOCAB) and {d["domain"] for d in gold["domains"]} == set(VOCAB), "Invalid domains")
            for dom in gold["domains"]:
                check(set(dom) == {"domain", "status", *BUCKETS} and dom["status"] in ("active", "inactive"), "Invalid domain schema")
                for bucket in BUCKETS:
                    for entry in dom[bucket]:
                        check(set(entry) == {"label", "evidence_post_ids", "evidence_post_indices"}, "Invalid interest schema")
                        prose(entry["label"])
                        positions = entry["evidence_post_indices"]
                        check(all(type(i) is int and 0 <= i < len(rows) for i in positions), "Evidence out of range")
                        check(positions == sorted(set(positions)), "Unsorted evidence")
                        check(entry["evidence_post_ids"] == [rows[i]["post_id"] for i in positions], "Evidence mismatch")
                        actual["interest_entries"] += 1
            caps = read_json(folder / "image_captions.json")
            check(set(caps) == {"user_id", "release_variant", "post_association", "captions"} and caps["user_id"] == uid and caps["release_variant"] == VERSION and caps["post_association"] == "unavailable", "Invalid captions")
            for i, (key, entry) in enumerate(caps["captions"].items(), 1):
                check(key == f"image_{i:05d}" and set(entry) == {"model_captions"}, "Invalid image")
                check(set(entry["model_captions"]) <= set(MODELS), "Unexpected caption model")
                for value in entry["model_captions"].values():
                    check(set(value) == {"summary"}, "Invalid caption fields")
                    prose(value["summary"])
                    actual["captions"] += 1
            actual.update(users=1, posts=len(rows), images=len(caps["captions"]))
        audit = read_json(root / "data/privacy_audit.json")
        check(set(audit) == {"release_variant", "counts", "privacy_review", "paper_scores_reproduced"}, "Invalid audit fields")
        check(audit["release_variant"] == VERSION and audit["privacy_review"] == "local_candidate" and audit["paper_scores_reproduced"] is False, "Invalid audit declarations")
        check(all(type(v) is int and v >= 0 for v in audit["counts"].values()), "Invalid counts")
        check(set(audit["counts"]) <= {"users", "posts", "images", "captions", "interest_entries", "changed_labels", "repaired_evidence_indices", "strings", "changed_strings", "keyword", "reviewed_rewrite", "email", "url", "account", "coordinates", "date", "phone", "identifier", "address", "private_path"}, "Unexpected audit counters")
        for key, n in actual.items():
            check(audit["counts"].get(key) == n, "Count mismatch: " + key)
        manifest = read_json(root / "data/manifest.json")
        check(set(manifest) == {"release_variant", "files"} and manifest["release_variant"] == VERSION, "Invalid manifest")
        files = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*") if p.is_file() and p != root / "data/manifest.json"}
        check(not any(p.is_symlink() for p in (root / "data").rglob("*")), "Symlink in data")
        check(files == set(manifest["files"]), "Manifest file set mismatch")
        for name in files:
            check(manifest["files"].get(name) == hashlib.sha256((root / name).read_bytes()).hexdigest(), "Checksum mismatch: " + name)
        check(files - {"data/README.md"} == {f"data/users/{d.name}/{name}" for d in folders for name in ("posts.jsonl", "gold_profile.json", "image_captions.json")} | {"data/privacy_audit.json"}, "Unexpected data file")
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        errors.append("Malformed candidate: " + type(exc).__name__)
    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-users", "destination", "mapping", "rules", "handles"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.source_users, args.destination, args.mapping, args.rules, args.handles), indent=2))


if __name__ == "__main__":
    main()
