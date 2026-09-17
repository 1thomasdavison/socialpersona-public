import json
from pathlib import Path
import tempfile
import unittest

from user_profile_pipeline.release_privacy import BUCKETS, MODELS, VOCAB, write_json
from user_profile_pipeline.targeted_privacy import Scrubber, build_candidate, verify_candidate


class TargetedPrivacyTests(unittest.TestCase):
    def test_relative_time_survives_prompt_rendering_but_not_ablation(self):
        from user_profile_pipeline.multimodal_context import post_timestamp
        from user_profile_pipeline.benchmark.profile_eval.modes import _posts_for_profile_input_mode
        post = {"relative_day": 33, "text": "Running"}
        self.assertEqual(post_timestamp(post), "relative day 0000033")
        self.assertLess(post_timestamp({"relative_day": 9}), post_timestamp({"relative_day": 100}))
        self.assertEqual(post_timestamp({"created_at": "2025-01-01", "relative_day": 33}), "2025-01-01")
        ablated = _posts_for_profile_input_mode([post], profile_input_mode="text_only")
        self.assertEqual(post_timestamp(ablated[0]), "")
        self.assertEqual(post["relative_day"], 33)

    def test_private_spans_and_public_interests(self):
        scrub = Scrubber({"replace": {"Private Lane School": "[ORGANIZATION]"}}, ["privatehandle"])
        text = "I love Taylor Swift, Arsenal FC and Genshin Impact; not Fortnite. #gaming #coffee"
        self.assertEqual(scrub(text), text)
        result = scrub("Ｍail me at name@example.com or @privatehandle; www.example.com "
                       "Private Lane School, 12 Maple Street; +1 (212) 555-1234 on 2026-09-16")
        for value in ("name@example", "privatehandle", "www.example", "Private Lane", "Maple", "555", "2026"):
            self.assertNotIn(value, result)
        self.assertEqual(scrub(result), result)

    def test_keywords_have_boundaries_and_do_not_delete_hashtags(self):
        scrub = Scrubber({"replace": {"Ann": "[PERSON]", "PrivateQuarter": "[LOCATION]"}})
        self.assertEqual(scrub("Ann's annual concert #PrivateQuarter #kpop"), "[PERSON]'s annual concert [LOCATION] #kpop")
        self.assertEqual(scrub("[PERSON] enjoys Pokémon."), "[PERSON] enjoys Pokémon.")

    def test_exact_rewrite_is_applied_after_contacts_removed(self):
        scrub = Scrubber({"rewrite": {"I live by SecretHill. [ACCOUNT]": "I enjoy local walks. [ACCOUNT]"}})
        self.assertEqual(scrub("I live by SecretHill. @somebody"), "I enjoy local walks. [ACCOUNT]")

    def test_residual_calendar_dates_and_paths(self):
        scrub = Scrubber(calendar_dates=True)
        result = scrub("[010825] 03242025 8/8/88 November 15-17, 2022 C:\\Users\\Alias&gt;command")
        self.assertEqual(result, "[DATE] [DATE] [DATE] [DATE] [PATH]&gt;command")
        self.assertEqual(scrub("27 dresses and 911 tv series"), "27 dresses and 911 tv series")

    def fixture(self, root):
        source = root / "source"
        d = source / "old_1"
        d.mkdir(parents=True)
        posts = [{"post_id": "old_b", "created_at": "2025-02-03", "text": "I like Pokémon. @hiddenhandle", "media": [{"sha256": "abcdef"}]},
                 {"post_id": "old_a", "created_at": "2025-01-01", "text": "Coffee\u2028and tea."}]
        (d / "posts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in posts), encoding="utf-8")
        domains = [{"domain": name, "status": "active", **{b: [] for b in BUCKETS}} for name in VOCAB]
        domains[0][BUCKETS[0]] = [{"label": "PrivateHill trail running", "evidence_post_ids": ["old_a"], "evidence_post_indices": [0]}]
        domains[1][BUCKETS[1]] = [{"label": "Taylor Swift", "evidence_post_ids": ["old_b"], "evidence_post_indices": [1]}]
        write_json(d / "gold_profile.json", {"domains": domains})
        write_json(d / "image_captions.json", {"captions": {"private_image": {"model_captions": {MODELS[0]: {"summary": "PrivateHill trail and trees"}}}}})
        write_json(root / "mapping.json", {"mapping": [{"source_id": "old_1", "public_id": "participant_001"}]})
        write_json(root / "rules.json", {"replace": {"PrivateHill": "[LOCATION]"}})
        write_json(root / "handles.json", {"mapping": [{"original_id": "x_hiddenhandle"}]})
        return source

    def test_complete_candidate_preserves_records_and_repairs_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            out = root / "candidate"
            result = build_candidate(source, out, root / "mapping.json", root / "rules.json", root / "handles.json", expected_users=1)
            self.assertEqual(verify_candidate(out, expected_users=1), [])
            self.assertEqual(result["counts"]["posts"], 2)
            self.assertEqual(result["counts"]["interest_entries"], 2)
            folder = out / "data/users/participant_001"
            gold = json.loads((folder / "gold_profile.json").read_text(encoding="utf-8"))
            entry = gold["domains"][0][BUCKETS[0]][0]
            self.assertEqual(entry["label"], "[LOCATION] trail running")
            self.assertEqual(entry["evidence_post_indices"], [1])
            self.assertEqual(entry["evidence_post_ids"], ["participant_001_post_0002"])
            self.assertEqual(gold["domains"][1][BUCKETS[1]][0]["label"], "Taylor Swift")
            with (folder / "posts.jsonl").open(encoding="utf-8") as stream:
                posts = [json.loads(line) for line in stream]
            self.assertEqual([p["relative_day"] for p in posts], [33, 0])
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            public = "\n".join(p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file())
            for token in ("old_a", "old_b", "hiddenhandle", "abcdef", "PrivateHill", "2025-01-01"):
                self.assertNotIn(token, public)
            with self.assertRaises(ValueError):
                build_candidate(source, out, root / "mapping.json", root / "rules.json", root / "handles.json", expected_users=1)
            posts[0]["leak"] = "private metadata"
            (folder / "posts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in posts), encoding="utf-8")
            self.assertIn("Invalid post schema", verify_candidate(out, expected_users=1))

    def test_unresolved_evidence_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            p = source / "old_1/gold_profile.json"
            gold = json.loads(p.read_text(encoding="utf-8"))
            gold["domains"][0][BUCKETS[0]][0]["evidence_post_ids"] = ["missing"]
            write_json(p, gold)
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                build_candidate(source, root / "out", root / "mapping.json", root / "rules.json", root / "handles.json", expected_users=1)


if __name__ == "__main__":
    unittest.main()
